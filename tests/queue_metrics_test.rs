//! Integration coverage for the router admission waiting gauge.

mod common;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use common::mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tower::ServiceExt;
use vllm_router_rs::config::{RouterConfig, RoutingMode};
use vllm_router_rs::middleware::ConcurrencyLimiter;
use vllm_router_rs::routers::{RouterFactory, RouterTrait};
use vllm_router_rs::server::{build_app_with_request_tracing, AppState};

fn chat_request() -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"mock-model","messages":[{"role":"user","content":"hello"}],"stream":false}"#,
        ))
        .unwrap()
}

async fn wait_for_count<F>(mut current: F, expected: u64) -> u64
where
    F: FnMut() -> u64,
{
    let deadline = Instant::now() + Duration::from_secs(3);
    let mut last = current();
    while Instant::now() < deadline {
        last = current();
        if last == expected {
            return last;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    last
}

#[tokio::test]
async fn standard_waiting_tracks_real_pending_request() {
    let mut worker = MockWorker::new(MockWorkerConfig {
        port: 0,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 800,
        fail_rate: 0.0,
        stream_chunk_delay_ms: 0,
    });
    let worker_url = worker.start().await.unwrap();

    let config = RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: vec![worker_url],
        },
        max_concurrent_requests: 1,
        rate_limit_tokens_per_second: Some(1),
        queue_size: 8,
        queue_timeout_secs: 5,
        worker_startup_timeout_secs: 2,
        worker_startup_check_interval_secs: 1,
        ..Default::default()
    };
    let context = common::create_test_context(config.clone());
    context.request_metrics.enable_waiting();
    let router: Arc<dyn RouterTrait> = Arc::from(
        RouterFactory::create_router(&context)
            .await
            .expect("router creation"),
    );

    let (limiter, processor) = ConcurrencyLimiter::new(
        context.rate_limiter.clone(),
        config.queue_size,
        Duration::from_secs(config.queue_timeout_secs),
    );
    let processor = tokio::spawn(processor.expect("queue enabled").run());

    let state = Arc::new(AppState {
        router,
        context: context.clone(),
        concurrency_queue_tx: limiter.queue_tx,
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

    let first = tokio::spawn(app.clone().oneshot(chat_request()));
    assert_eq!(
        wait_for_count(|| context.request_metrics.running_count(), 1).await,
        1
    );

    let second = tokio::spawn(app.oneshot(chat_request()));
    assert_eq!(
        wait_for_count(|| context.request_metrics.waiting_count(), 1).await,
        1,
        "the admitted second request must be visible while waiting for a token"
    );

    assert_eq!(first.await.unwrap().unwrap().status(), StatusCode::OK);
    assert_eq!(second.await.unwrap().unwrap().status(), StatusCode::OK);
    assert_eq!(
        wait_for_count(|| context.request_metrics.waiting_count(), 0).await,
        0
    );
    assert_eq!(context.request_metrics.running_count(), 0);

    processor.abort();
    worker.stop().await;
}
