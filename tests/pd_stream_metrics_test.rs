//! Regression coverage for vLLM P/D stream ownership.
//!
//! Decode load and the vLLM-compatible logical-request gauge must stay live
//! until a streaming response body completes or its client drops it.

mod common;

use axum::{
    body::Body,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router as AxumRouter,
};
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::{
    convert::Infallible,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::{sync::Notify, task::JoinHandle};
use vllm_router_rs::{
    config::{KvConnector, PolicyConfig, RouterConfig, RoutingMode},
    core::Worker,
    protocols::spec::ChatCompletionRequest,
    routers::{RouterFactory, RouterTrait},
};

struct TestServer {
    url: String,
    task: JoinHandle<()>,
}

impl TestServer {
    async fn start(app: AxumRouter) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test worker");
        let url = format!("http://{}", listener.local_addr().expect("worker address"));
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serve test worker");
        });
        Self { url, task }
    }

    async fn stop(mut self) {
        self.task.abort();
        let _ = (&mut self.task).await;
    }
}

#[derive(Clone)]
struct DecodeState {
    body_started: Arc<Notify>,
    release_body: Arc<Notify>,
    requests: Arc<Mutex<Vec<Value>>>,
}

#[derive(Clone, Default)]
struct PrefillState {
    requests: Arc<Mutex<Vec<Value>>>,
}

async fn healthy() -> StatusCode {
    StatusCode::OK
}

async fn prefill(State(state): State<PrefillState>, Json(request): Json<Value>) -> Response {
    state.requests.lock().unwrap().push(request);
    Json(json!({
        "kv_transfer_params": {
            "remote_engine_id": "prefill-engine",
            "remote_host": "127.0.0.1",
            "remote_port": 9000,
            "tp_size": 1,
            "transfer_mode": "push",
        }
    }))
    .into_response()
}

async fn decode(State(state): State<DecodeState>, Json(request): Json<Value>) -> Response {
    state.requests.lock().unwrap().push(request);
    let first_state = state.clone();
    let stream = futures_util::stream::once(async move {
        first_state.body_started.notify_one();
        Ok::<Bytes, Infallible>(Bytes::from_static(b"data: first\n\n"))
    })
    .chain(futures_util::stream::once(async move {
        state.release_body.notified().await;
        Ok::<Bytes, Infallible>(Bytes::from_static(b"data: [DONE]\n\n"))
    }));
    Body::from_stream(stream).into_response()
}

async fn wait_for_load(worker: &Arc<dyn Worker>, expected: usize) {
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if worker.load() == expected {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("worker load reached expected value");
}

#[tokio::test]
async fn pd_stream_holds_load_and_logical_activity_through_concurrent_cancellation() {
    let prefill_state = PrefillState::default();
    let prefill = TestServer::start(
        AxumRouter::new()
            .route("/health", get(healthy))
            .route("/v1/chat/completions", post(prefill))
            .with_state(prefill_state.clone()),
    )
    .await;
    let decode_state = DecodeState {
        body_started: Arc::new(Notify::new()),
        release_body: Arc::new(Notify::new()),
        requests: Arc::new(Mutex::new(Vec::new())),
    };
    let decode = TestServer::start(
        AxumRouter::new()
            .route("/health", get(healthy))
            .route("/v1/chat/completions", post(decode))
            .with_state(decode_state.clone()),
    )
    .await;

    let config = RouterConfig {
        mode: RoutingMode::VllmPrefillDecode {
            prefill_urls: vec![(prefill.url.clone(), None)],
            decode_urls: vec![decode.url.clone()],
            prefill_policy: None,
            decode_policy: None,
            discovery_address: None,
        },
        policy: PolicyConfig::CacheAware {
            cache_threshold: 0.3,
            balance_abs_threshold: 999,
            balance_rel_threshold: 9.9,
            eviction_interval_secs: 0,
            max_tree_size: 100_000,
        },
        kv_connector: KvConnector::Nixl,
        disable_retries: true,
        ..Default::default()
    };
    let context = common::create_test_context(config);
    let router: Arc<dyn RouterTrait> = Arc::from(
        RouterFactory::create_router(&context)
            .await
            .expect("create P/D router"),
    );
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": true,
    }))
    .expect("valid chat request");

    let response = router.route_chat(None, &request, None).await;
    assert_eq!(response.status(), StatusCode::OK);
    let prefill_worker = context
        .worker_registry
        .get_prefill_workers()
        .into_iter()
        .next()
        .expect("registered prefill worker");
    let decode_worker = context
        .worker_registry
        .get_decode_workers()
        .into_iter()
        .next()
        .expect("registered decode worker");
    assert_eq!(decode_worker.load(), 1, "decode owns the open response");
    assert_eq!(context.request_metrics.running_count(), 1);

    let mut body = response.into_body().into_data_stream();
    let first = body.next().await.expect("first chunk").expect("chunk data");
    assert_eq!(first, Bytes::from_static(b"data: first\n\n"));
    decode_state.body_started.notified().await;
    assert_eq!(decode_worker.load(), 1, "decode stays busy between chunks");
    assert_eq!(context.request_metrics.running_count(), 1);

    decode_state.release_body.notify_one();
    let done = body.next().await.expect("done chunk").expect("chunk data");
    assert_eq!(done, Bytes::from_static(b"data: [DONE]\n\n"));
    assert!(body.next().await.is_none());
    wait_for_load(&decode_worker, 0).await;
    assert_eq!(context.request_metrics.running_count(), 0);

    // The first sequential response caches the NIXL push identity. The next
    // request must use the concurrent NIXL path and its cached identity.
    let response = router.route_chat(None, &request, None).await;
    assert_eq!(response.status(), StatusCode::OK);
    {
        let prefill_requests = prefill_state.requests.lock().unwrap();
        let decode_requests = decode_state.requests.lock().unwrap();
        assert_eq!(prefill_requests.len(), 2, "both requests reached prefill");
        assert_eq!(decode_requests.len(), 2, "both requests reached decode");
        assert_eq!(
            decode_requests[1]["kv_transfer_params"]["do_remote_prefill"],
            json!(true),
            "second decode dispatch uses NIXL push"
        );
        assert_eq!(
            decode_requests[1]["kv_transfer_params"]["remote_engine_id"],
            json!("prefill-engine"),
            "second decode dispatch uses the cached prefill identity"
        );
    }

    assert_eq!(
        prefill_worker.load(),
        0,
        "prefill completed before streaming"
    );
    assert_eq!(decode_worker.load(), 1);
    assert_eq!(context.request_metrics.running_count(), 1);
    drop(response);
    wait_for_load(&prefill_worker, 0).await;
    wait_for_load(&decode_worker, 0).await;
    assert_eq!(context.request_metrics.running_count(), 0);

    prefill.stop().await;
    decode.stop().await;
}
