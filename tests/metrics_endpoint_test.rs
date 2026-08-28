//! The main Router port exposes the same Prometheus recorder as the legacy listener.

mod common;

use axum::body::{to_bytes, Body};
use axum::http::{header::CONTENT_TYPE, Request, StatusCode};
use std::sync::Arc;
use tower::ServiceExt;
use vllm_router_rs::config::{RouterConfig, RoutingMode};
use vllm_router_rs::routers::{RouterFactory, RouterTrait};
use vllm_router_rs::server::{build_app_with_request_tracing, AppState};

#[tokio::test]
async fn main_port_metrics_route_renders_prometheus_handle() {
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let config = RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: vec![],
        },
        ..Default::default()
    };
    let context =
        metrics::with_local_recorder(&recorder, || common::create_test_context(config.clone()));
    let router: Arc<dyn RouterTrait> = Arc::from(
        RouterFactory::create_router(&context)
            .await
            .expect("router creation"),
    );
    let state = Arc::new(AppState {
        router,
        context,
        concurrency_queue_tx: None,
        router_manager: None,
        prometheus_handle: Some(handle),
    });
    let app = build_app_with_request_tracing(
        state,
        config.max_payload_size,
        vec!["x-request-id".to_string()],
        vec![],
        true,
        false,
    );

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(CONTENT_TYPE).unwrap(),
        "text/plain; version=0.0.4; charset=utf-8"
    );
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("# TYPE vllm:num_requests_running gauge"));
    assert!(text.contains("vllm:num_requests_running 0"));

    let families: Result<Vec<_>, _> = prometheus_scraper::parse_payload(
        text.as_bytes(),
        prometheus_scraper::Format::Text(prometheus_scraper::TextFormat::Prometheus),
    )
    .collect();
    let families = families.expect("standard Prometheus parser must accept the scrape");
    assert!(
        families
            .iter()
            .any(|family| family.name == "vllm:num_requests_running"),
        "the parser must preserve the vLLM-compatible metric name"
    );
}
