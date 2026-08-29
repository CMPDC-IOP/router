//! Regression tests for Anthropic Messages error mapping through the
//! transparent proxy.
//!
//! vLLM workers answer client-side context-length violations on their native
//! `/v1/messages` endpoint with HTTP 500 `internal_error`. The router must
//! surface those as HTTP 400 `invalid_request_error` so callers such as
//! LiteLLM stop treating client mistakes as server faults, while genuine
//! server errors and upstream 4xx pass through unchanged.

mod common;

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use common::mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType};
use reqwest::Client;
use serde_json::{json, Value};
use std::sync::Arc;
use tower::ServiceExt;
use vllm_router_rs::config::{
    CircuitBreakerConfig, ConnectionMode, PolicyConfig, RetryConfig, RouterConfig, RoutingMode,
};
use vllm_router_rs::routers::RouterFactory;

/// The context-length error text observed from production vLLM workers.
fn overflow_message() -> String {
    "This model's maximum context length is 262144 tokens. However, you requested 128000 output tokens and your prompt contains at least 134145 input tokens, for a total of at least 262145 tokens. Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_tokens, value=134145)".to_string()
}

async fn start_worker() -> (MockWorker, String) {
    let mut worker = MockWorker::new(MockWorkerConfig {
        port: 0,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
        stream_chunk_delay_ms: 0,
    });
    let url = worker.start().await.unwrap();
    (worker, url)
}

/// Boot the router with transparent proxy enabled, as in production.
async fn build_app(worker_url: String) -> axum::Router {
    let config = RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: vec![worker_url],
        },
        policy: PolicyConfig::Random,
        host: "127.0.0.1".to_string(),
        port: 3002,
        max_payload_size: 256 * 1024 * 1024,
        request_timeout_secs: 600,
        worker_startup_timeout_secs: 1,
        worker_startup_check_interval_secs: 1,
        discovery: None,
        intra_node_data_parallel_size: 1,
        api_key: None,
        api_key_validation_urls: vec![],
        metrics: None,
        log_dir: None,
        log_level: None,
        request_id_headers: None,
        max_concurrent_requests: 64,
        queue_size: 0,
        queue_timeout_secs: 60,
        rate_limit_tokens_per_second: None,
        cors_allowed_origins: vec![],
        retry: RetryConfig::default(),
        circuit_breaker: CircuitBreakerConfig::default(),
        disable_retries: false,
        disable_circuit_breaker: false,
        health_check: vllm_router_rs::config::HealthCheckConfig::default(),
        enable_igw: false,
        connection_mode: ConnectionMode::Http,
        history_backend: vllm_router_rs::config::HistoryBackend::Memory,
        enable_profiling: false,
        profile_timeout_secs: 30,
        kv_connector: vllm_router_rs::config::KvConnector::Nixl,
    };

    let app_context = common::create_test_context(config.clone());
    let router = RouterFactory::create_router(&app_context).await.unwrap();
    let router = Arc::from(router);

    // Let the router discover and health-check the worker.
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    common::test_app::create_test_app(Arc::clone(&router), Client::new(), &config)
}

async fn post_json(
    app: axum::Router,
    path: &str,
    body: Value,
) -> (StatusCode, Value, Option<String>) {
    let req = Request::builder()
        .method("POST")
        .uri(path)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();

    let response = app.oneshot(req).await.unwrap();
    let status = response.status();
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let value = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes)
            .unwrap_or(Value::String(String::from_utf8_lossy(&bytes).into_owned()))
    };
    (status, value, content_type)
}

fn messages_request(scenario: Option<&str>, stream: bool) -> Value {
    let mut body = json!({
        "model": "glm-5.3-flash",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hello"}]
    });
    if stream {
        body["stream"] = json!(true);
    }
    if let Some(scenario) = scenario {
        body["mock_messages_error"] = json!(scenario);
    }
    body
}

#[tokio::test]
async fn test_messages_context_overflow_returns_400_invalid_request_error() {
    let (mut worker, worker_url) = start_worker().await;
    let app = build_app(worker_url).await;

    let (status, value, content_type) = post_json(
        app,
        "/v1/messages",
        messages_request(Some("context_overflow"), false),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {value}");
    assert_eq!(content_type.as_deref(), Some("application/json"));
    assert_eq!(value["type"], "error");
    assert_eq!(value["error"]["type"], "invalid_request_error");
    assert_eq!(value["error"]["message"], json!(overflow_message()));

    worker.stop().await;
}

#[tokio::test]
async fn test_messages_streaming_context_overflow_returns_400() {
    // The worker rejects before the stream starts, so the reply is a JSON
    // error even for stream=true; the mapping must apply to it too.
    let (mut worker, worker_url) = start_worker().await;
    let app = build_app(worker_url).await;

    let (status, value, _) = post_json(
        app,
        "/v1/messages",
        messages_request(Some("context_overflow"), true),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {value}");
    assert_eq!(value["type"], "error");
    assert_eq!(value["error"]["type"], "invalid_request_error");
    assert_eq!(value["error"]["message"], json!(overflow_message()));

    worker.stop().await;
}

#[tokio::test]
async fn test_messages_max_completion_tokens_overflow_returns_400() {
    let (mut worker, worker_url) = start_worker().await;
    let app = build_app(worker_url).await;

    let (status, value, _) = post_json(
        app,
        "/v1/messages",
        messages_request(Some("max_completion_tokens"), false),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {value}");
    assert_eq!(value["error"]["type"], "invalid_request_error");
    assert!(value["error"]["message"]
        .as_str()
        .unwrap()
        .contains("max_completion_tokens=128000 cannot be greater than max_model_len"));

    worker.stop().await;
}

#[tokio::test]
async fn test_messages_input_length_overflow_returns_400() {
    let (mut worker, worker_url) = start_worker().await;
    let app = build_app(worker_url).await;

    let (status, value, _) = post_json(
        app,
        "/v1/messages",
        messages_request(Some("input_length"), false),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {value}");
    assert_eq!(value["error"]["type"], "invalid_request_error");
    assert!(value["error"]["message"]
        .as_str()
        .unwrap()
        .contains("exceeds model's maximum context length"));

    worker.stop().await;
}

#[tokio::test]
async fn test_messages_worker_crashed_stays_500() {
    let (mut worker, worker_url) = start_worker().await;
    let app = build_app(worker_url).await;

    let (status, value, _) = post_json(
        app,
        "/v1/messages",
        messages_request(Some("worker_crashed"), false),
    )
    .await;

    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "body: {value}");
    assert_eq!(value["type"], "error");
    assert_eq!(value["error"]["type"], "internal_error");
    assert_eq!(
        value["error"]["message"],
        json!("worker crashed while processing request")
    );

    worker.stop().await;
}

#[tokio::test]
async fn test_messages_upstream_400_passes_through_unchanged() {
    let (mut worker, worker_url) = start_worker().await;
    let app = build_app(worker_url).await;

    let (status, value, _) = post_json(
        app,
        "/v1/messages",
        messages_request(Some("upstream_400"), false),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {value}");
    assert_eq!(value["type"], "error");
    assert_eq!(value["error"]["type"], "invalid_request_error");
    assert_eq!(
        value["error"]["message"],
        json!("This model's maximum context length is 262144 tokens. However, you requested 128000 output tokens.")
    );

    worker.stop().await;
}

#[tokio::test]
async fn test_messages_success_passes_through() {
    let (mut worker, worker_url) = start_worker().await;
    let app = build_app(worker_url).await;

    let (status, value, _) = post_json(app, "/v1/messages", messages_request(None, false)).await;

    assert_eq!(status, StatusCode::OK, "body: {value}");
    assert_eq!(value["type"], "message");
    assert_eq!(value["role"], "assistant");

    worker.stop().await;
}

#[tokio::test]
async fn test_chat_completions_context_overflow_keeps_400() {
    // The registered OpenAI route forwards worker 4xx verbatim: a large
    // max_tokens triggers the worker's OpenAI-layer validation, which already
    // answers with 400 BadRequestError.
    let (mut worker, worker_url) = start_worker().await;
    let app = build_app(worker_url).await;

    let body = json!({
        "model": "glm-5.3-flash",
        "max_tokens": 128000,
        "messages": [{"role": "user", "content": "hello"}]
    });
    let (status, value, _) = post_json(app, "/v1/chat/completions", body).await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {value}");
    assert_eq!(value["error"]["type"], "BadRequestError");
    assert_eq!(value["error"]["code"], 400);
    assert_eq!(value["error"]["message"], json!(overflow_message()));

    worker.stop().await;
}
