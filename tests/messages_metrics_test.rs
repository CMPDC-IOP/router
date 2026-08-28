//! Regression coverage for request activity on the native Anthropic Messages
//! endpoint, which is served through the transparent-proxy fallback.

mod common;

use axum::{
    body::{to_bytes, Body},
    extract::{Json, Request, State},
    http::{Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{any, get, post},
    Router as AxumRouter,
};
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::json;
use std::{convert::Infallible, future::Future, sync::Arc, time::Duration};
use tokio::{
    sync::{oneshot, Notify},
    task::JoinHandle,
};
use tower::ServiceExt;
use vllm_router_rs::{
    config::{RouterConfig, RoutingMode},
    routers::{RouterFactory, RouterTrait},
    server::{build_app_with_request_tracing, AppState},
};

#[derive(Clone)]
struct Gates {
    received: Arc<Notify>,
    release_headers: Arc<Notify>,
    body_started: Arc<Notify>,
    release_body: Arc<Notify>,
}

impl Default for Gates {
    fn default() -> Self {
        Self {
            received: Arc::new(Notify::new()),
            release_headers: Arc::new(Notify::new()),
            body_started: Arc::new(Notify::new()),
            release_body: Arc::new(Notify::new()),
        }
    }
}

#[derive(Clone, Copy)]
enum WorkerReply {
    Json,
    Stream,
}

#[derive(Clone)]
struct WorkerState {
    gates: Gates,
    reply: WorkerReply,
}

struct GatedWorker {
    url: String,
    gates: Gates,
    task: JoinHandle<()>,
}

impl GatedWorker {
    async fn start(reply: WorkerReply) -> Self {
        let gates = Gates::default();
        let state = WorkerState {
            gates: gates.clone(),
            reply,
        };
        let app = AxumRouter::new()
            .route("/health", get(|| async { StatusCode::OK }))
            .route("/v1/messages", post(worker_handler))
            .fallback(any(worker_handler))
            .with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock worker");
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("mock worker serves");
        });
        Self { url, gates, task }
    }

    async fn stop(&mut self) {
        self.task.abort();
        let _ = (&mut self.task).await;
    }
}

impl Drop for GatedWorker {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn worker_handler(State(state): State<WorkerState>, _request: Request) -> Response {
    state.gates.received.notify_one();
    state.gates.release_headers.notified().await;

    match state.reply {
        WorkerReply::Json => Json(json!({"ok": true})).into_response(),
        WorkerReply::Stream => {
            let gates = state.gates;
            let first_gates = gates.clone();
            let stream = futures_util::stream::once(async move {
                first_gates.body_started.notify_one();
                Ok::<Bytes, Infallible>(Bytes::from_static(b"first"))
            })
            .chain(futures_util::stream::once(async move {
                gates.release_body.notified().await;
                Ok::<Bytes, Infallible>(Bytes::from_static(b"second"))
            }));
            Body::from_stream(stream).into_response()
        }
    }
}

async fn build_app(worker_url: String) -> (axum::Router, Arc<vllm_router_rs::server::AppContext>) {
    let config = RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: vec![worker_url],
        },
        worker_startup_timeout_secs: 1,
        worker_startup_check_interval_secs: 1,
        disable_retries: true,
        ..Default::default()
    };
    let context = common::create_test_context(config.clone());
    let router: Arc<dyn RouterTrait> = Arc::from(
        RouterFactory::create_router(&context)
            .await
            .expect("create regular router"),
    );
    let state = Arc::new(AppState {
        router,
        context: Arc::clone(&context),
        concurrency_queue_tx: None,
        router_manager: None,
        prometheus_handle: None,
    });
    let app = build_app_with_request_tracing(
        state,
        config.max_payload_size,
        vec!["x-request-id".to_string()],
        vec![],
        true,
        false,
    );
    (app, context)
}

async fn bounded<F: Future>(future: F) -> F::Output {
    tokio::time::timeout(Duration::from_secs(5), future)
        .await
        .expect("test gate timed out")
}

fn request(path: &str) -> Request {
    Request::builder()
        .method(Method::POST)
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"test","messages":[]}"#))
        .unwrap()
}

#[tokio::test]
async fn messages_fallback_counts_after_assignment_through_non_streaming_body() {
    let worker = GatedWorker::start(WorkerReply::Json).await;
    let (app, context) = build_app(worker.url.clone()).await;
    let aborted_app = app.clone();
    let normal_app = app.clone();
    let transparent_app = app.clone();

    let aborted_request =
        tokio::spawn(async move { aborted_app.oneshot(request("/v1/messages")).await.unwrap() });
    bounded(worker.gates.received.notified()).await;
    assert_eq!(context.request_metrics.running_count(), 1);
    aborted_request.abort();
    let _ = bounded(aborted_request).await;
    assert_eq!(
        context.request_metrics.running_count(),
        0,
        "cancelled request releases activity"
    );
    worker.gates.release_headers.notify_one();

    let (response_tx, response_rx) = oneshot::channel();
    tokio::spawn(async move {
        let _ = response_tx.send(normal_app.oneshot(request("/v1/messages")).await.unwrap());
    });

    bounded(worker.gates.received.notified()).await;
    assert_eq!(
        context.request_metrics.running_count(),
        1,
        "worker was selected before headers"
    );
    worker.gates.release_headers.notify_one();
    let response = bounded(response_rx).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        context.request_metrics.running_count(),
        1,
        "body still owns activity"
    );
    let body = bounded(to_bytes(response.into_body(), usize::MAX))
        .await
        .unwrap();
    assert_eq!(body, Bytes::from_static(br#"{"ok":true}"#));
    assert_eq!(context.request_metrics.running_count(), 0);

    let (response_tx, response_rx) = oneshot::channel();
    tokio::spawn(async move {
        let _ = response_tx.send(
            transparent_app
                .oneshot(request("/transparent-admin"))
                .await
                .unwrap(),
        );
    });
    bounded(worker.gates.received.notified()).await;
    assert_eq!(
        context.request_metrics.running_count(),
        0,
        "only exact POST /v1/messages is inference"
    );
    worker.gates.release_headers.notify_one();
    let response = bounded(response_rx).await.unwrap();
    let _ = bounded(to_bytes(response.into_body(), usize::MAX))
        .await
        .unwrap();
}

#[tokio::test]
async fn messages_stream_keeps_activity_until_completed_or_cancelled() {
    let worker = GatedWorker::start(WorkerReply::Stream).await;
    let (app, context) = build_app(worker.url.clone()).await;
    let (response_tx, response_rx) = oneshot::channel();
    tokio::spawn(async move {
        let _ = response_tx.send(app.oneshot(request("/v1/messages")).await.unwrap());
    });
    bounded(worker.gates.received.notified()).await;
    worker.gates.release_headers.notify_one();
    let response = bounded(response_rx).await.unwrap();
    assert_eq!(context.request_metrics.running_count(), 1);

    let body_task =
        tokio::spawn(async move { to_bytes(response.into_body(), usize::MAX).await.unwrap() });
    bounded(worker.gates.body_started.notified()).await;
    assert_eq!(
        context.request_metrics.running_count(),
        1,
        "stream remains active between chunks"
    );
    worker.gates.release_body.notify_one();
    assert_eq!(
        bounded(body_task).await.unwrap(),
        Bytes::from_static(b"firstsecond")
    );
    assert_eq!(
        context.request_metrics.running_count(),
        0,
        "completed stream releases activity"
    );

    // A client that abandons a response before reading it must release the
    // guard just as a normally completed stream does.
    let worker = GatedWorker::start(WorkerReply::Stream).await;
    let (app, context) = build_app(worker.url.clone()).await;
    let (response_tx, response_rx) = oneshot::channel();
    tokio::spawn(async move {
        let _ = response_tx.send(app.oneshot(request("/v1/messages")).await.unwrap());
    });
    bounded(worker.gates.received.notified()).await;
    worker.gates.release_headers.notify_one();
    let response = bounded(response_rx).await.unwrap();
    assert_eq!(context.request_metrics.running_count(), 1);
    drop(response);
    assert_eq!(
        context.request_metrics.running_count(),
        0,
        "dropped stream releases activity"
    );
}

#[tokio::test]
async fn messages_upstream_failure_does_not_leak_activity() {
    let mut worker = GatedWorker::start(WorkerReply::Json).await;
    let (app, context) = build_app(worker.url.clone()).await;
    bounded(worker.stop()).await;

    let response = bounded(app.oneshot(request("/v1/messages"))).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(context.request_metrics.running_count(), 0);
}
