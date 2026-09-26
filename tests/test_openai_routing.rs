//! Comprehensive integration tests for OpenAI backend functionality

use axum::{
    body::Body,
    extract::{Request, State},
    http::{header::CONTENT_TYPE, Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
};
use tokio::task::JoinHandle;
use tower::ServiceExt;
use vllm_router_rs::{
    config::{RouterConfig, RoutingMode},
    protocols::spec::{
        ChatCompletionRequest, ChatMessage, CompletionRequest, GenerateRequest, PromptInput,
        UserMessageContent,
    },
    routers::{openai_router::OpenAIRouter, RouterFactory, RouterTrait},
};

mod common;
use common::mock_openai_server::MockOpenAIServer;

/// Helper function to create a minimal chat completion request for testing
fn create_minimal_chat_request() -> ChatCompletionRequest {
    let val = json!({
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "max_tokens": 100
    });
    serde_json::from_value(val).unwrap()
}

/// Helper function to create a minimal completion request for testing
fn create_minimal_completion_request() -> CompletionRequest {
    CompletionRequest {
        model: Some("gpt-3.5-turbo".to_string()),
        prompt: PromptInput::String("Hello".to_string()),
        suffix: None,
        max_tokens: Some(100),
        temperature: None,
        top_p: None,
        n: None,
        stream: false,
        stream_options: None,
        logprobs: None,
        echo: false,
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        best_of: None,
        logit_bias: None,
        user: None,
        seed: None,
        top_k: None,
        min_p: None,
        min_tokens: None,
        repetition_penalty: None,
        regex: None,
        ebnf: None,
        json_schema: None,
        stop_token_ids: None,
        no_stop_trim: false,
        ignore_eos: false,
        skip_special_tokens: true,
        lora_path: None,
        session_params: None,
        return_hidden_states: false,
        other: serde_json::Map::new(),
    }
}

// ============= Basic Unit Tests =============

/// Test basic OpenAI router creation and configuration
#[tokio::test]
async fn test_openai_router_creation() {
    let router = OpenAIRouter::new("https://api.openai.com".to_string(), None).await;

    assert!(router.is_ok(), "Router creation should succeed");

    let router = router.unwrap();
    assert_eq!(router.router_type(), "openai");
    assert!(!router.is_pd_mode());
}

/// Test health endpoints
#[tokio::test]
async fn test_openai_router_health() {
    // Health probes intentionally omit auth. An auth-required upstream should
    // return 401, which still proves the endpoint is reachable and healthy.
    let mock_server =
        MockOpenAIServer::new_with_auth(Some("Bearer health-test-token".to_string())).await;
    let router = OpenAIRouter::new(mock_server.base_url(), None)
        .await
        .unwrap();

    let req = Request::builder()
        .method(Method::GET)
        .uri("/health")
        .body(Body::empty())
        .unwrap();

    let response = router.health(req).await;
    assert_eq!(response.status(), StatusCode::OK);
}

/// Test server info endpoint
#[tokio::test]
async fn test_openai_router_server_info() {
    let router = OpenAIRouter::new("https://api.openai.com".to_string(), None)
        .await
        .unwrap();

    let req = Request::builder()
        .method(Method::GET)
        .uri("/info")
        .body(Body::empty())
        .unwrap();

    let response = router.get_server_info(req).await;
    assert_eq!(response.status(), StatusCode::OK);

    let (_, body) = response.into_parts();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
    let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();

    assert!(body_str.contains("openai"));
}

/// Test models endpoint
#[tokio::test]
async fn test_openai_router_models() {
    // Use mock server for deterministic models response
    let mock_server = MockOpenAIServer::new().await;
    let router = OpenAIRouter::new(mock_server.base_url(), None)
        .await
        .unwrap();

    let req = Request::builder()
        .method(Method::GET)
        .uri("/models")
        .body(Body::empty())
        .unwrap();

    let response = router.get_models(req).await;
    assert_eq!(response.status(), StatusCode::OK);

    let (_, body) = response.into_parts();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
    let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
    let models: serde_json::Value = serde_json::from_str(&body_str).unwrap();

    assert_eq!(models["object"], "list");
    assert!(models["data"].is_array());
}

/// Test router factory with OpenAI routing mode
#[tokio::test]
async fn test_router_factory_openai_mode() {
    let routing_mode = RoutingMode::OpenAI {
        worker_urls: vec!["https://api.openai.com".to_string()],
    };

    let router_config =
        RouterConfig::new(routing_mode, vllm_router_rs::config::PolicyConfig::Random);

    let app_context = common::create_test_context(router_config);

    let router = vllm_router_rs::routers::RouterFactory::create_router(&app_context).await;
    assert!(
        router.is_ok(),
        "Router factory should create OpenAI router successfully"
    );

    let router = router.unwrap();
    assert_eq!(router.router_type(), "openai");
}

/// Test that unsupported endpoints return proper error codes
#[tokio::test]
async fn test_unsupported_endpoints() {
    let router = OpenAIRouter::new("https://api.openai.com".to_string(), None)
        .await
        .unwrap();

    // Test generate endpoint (VLLM-specific, should not be supported)
    let generate_request = GenerateRequest {
        prompt: None,
        text: Some("Hello world".to_string()),
        input_ids: None,
        parameters: None,
        sampling_params: None,
        stream: false,
        return_logprob: false,
        lora_path: None,
        session_params: None,
        return_hidden_states: false,
        rid: None,
    };

    let response = router.route_generate(None, &generate_request, None).await;
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);

    // Test completion endpoint (should also not be supported)
    let completion_request = create_minimal_completion_request();
    let response = router
        .route_completion(None, &completion_request, None)
        .await;
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
}

// ============= Mock Server E2E Tests =============

/// Test chat completion with mock OpenAI server
#[tokio::test]
async fn test_openai_router_chat_completion_with_mock() {
    // Start a mock OpenAI server
    let mock_server = MockOpenAIServer::new().await;
    let base_url = mock_server.base_url();

    // Create router pointing to mock server
    let router = OpenAIRouter::new(base_url, None).await.unwrap();

    // Create a minimal chat completion request
    let mut chat_request = create_minimal_chat_request();
    chat_request.messages = vec![ChatMessage::User {
        role: "user".to_string(),
        content: UserMessageContent::Text("Hello, how are you?".to_string()),
        name: None,
    }];
    chat_request.temperature = Some(0.7);

    // Route the request
    let response = router.route_chat(None, &chat_request, None).await;

    // Should get a successful response from mock server
    assert_eq!(response.status(), StatusCode::OK);

    let (_, body) = response.into_parts();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
    let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
    let chat_response: serde_json::Value = serde_json::from_str(&body_str).unwrap();

    // Verify it's a valid chat completion response
    assert_eq!(chat_response["object"], "chat.completion");
    assert_eq!(chat_response["model"], "gpt-3.5-turbo");
    assert!(!chat_response["choices"].as_array().unwrap().is_empty());
}

/// Test full E2E flow with Axum server
#[tokio::test]
async fn test_openai_e2e_with_server() {
    // Start mock OpenAI server
    let mock_server = MockOpenAIServer::new().await;
    let base_url = mock_server.base_url();

    // Create router
    let router = OpenAIRouter::new(base_url, None).await.unwrap();

    // Create Axum app with chat completions endpoint
    let app = Router::new().route(
        "/v1/chat/completions",
        post({
            let router = Arc::new(router);
            move |req: Request<Body>| {
                let router = router.clone();
                async move {
                    let (parts, body) = req.into_parts();
                    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
                    let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();

                    let chat_request: ChatCompletionRequest =
                        serde_json::from_str(&body_str).unwrap();

                    router
                        .route_chat(Some(&parts.headers), &chat_request, None)
                        .await
                }
            }
        }),
    );

    // Make a request to the server
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, world!"
                    }
                ],
                "max_tokens": 100
            })
            .to_string(),
        ))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let response_json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    // Verify the response structure
    assert_eq!(response_json["object"], "chat.completion");
    assert_eq!(response_json["model"], "gpt-3.5-turbo");
    assert!(!response_json["choices"].as_array().unwrap().is_empty());
}

/// Test streaming chat completions pass-through with mock server
#[tokio::test]
async fn test_openai_router_chat_streaming_with_mock() {
    let mock_server = MockOpenAIServer::new().await;
    let base_url = mock_server.base_url();
    let router = OpenAIRouter::new(base_url, None).await.unwrap();

    // Build a streaming chat request
    let val = json!({
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "max_tokens": 10,
        "stream": true
    });
    let chat_request: ChatCompletionRequest = serde_json::from_value(val).unwrap();

    let response = router.route_chat(None, &chat_request, None).await;
    assert_eq!(response.status(), StatusCode::OK);

    // Should be SSE
    let headers = response.headers();
    let ct = headers
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .to_ascii_lowercase();
    assert!(ct.contains("text/event-stream"));

    // Read entire stream body and assert chunks + DONE
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("chat.completion.chunk"));
    assert!(text.contains("[DONE]"));
}

/// Test circuit breaker functionality
#[tokio::test]
async fn test_openai_router_circuit_breaker() {
    // Create router with circuit breaker config
    let cb_config = vllm_router_rs::config::CircuitBreakerConfig {
        failure_threshold: 2,
        success_threshold: 1,
        timeout_duration_secs: 1,
        window_duration_secs: 10,
    };

    let router = OpenAIRouter::new(
        "http://invalid-url-that-will-fail".to_string(),
        Some(cb_config),
    )
    .await
    .unwrap();

    let chat_request = create_minimal_chat_request();

    // First few requests should fail and record failures
    for _ in 0..3 {
        let response = router.route_chat(None, &chat_request, None).await;
        // Should get either an error or circuit breaker response
        assert!(
            response.status() == StatusCode::INTERNAL_SERVER_ERROR
                || response.status() == StatusCode::SERVICE_UNAVAILABLE
        );
    }
}

/// Test that Authorization header is forwarded in /v1/models
#[tokio::test]
async fn test_openai_router_models_auth_forwarding() {
    // Start a mock server that requires Authorization
    let expected_auth = "Bearer test-token".to_string();
    let mock_server = MockOpenAIServer::new_with_auth(Some(expected_auth.clone())).await;
    let router = OpenAIRouter::new(mock_server.base_url(), None)
        .await
        .unwrap();

    // 1) Without auth header -> expect 401
    let req = Request::builder()
        .method(Method::GET)
        .uri("/models")
        .body(Body::empty())
        .unwrap();

    let response = router.get_models(req).await;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    // 2) With auth header -> expect 200
    let req = Request::builder()
        .method(Method::GET)
        .uri("/models")
        .header("Authorization", expected_auth)
        .body(Body::empty())
        .unwrap();

    let response = router.get_models(req).await;
    assert_eq!(response.status(), StatusCode::OK);

    let (_, body) = response.into_parts();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
    let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
    let models: serde_json::Value = serde_json::from_str(&body_str).unwrap();
    assert_eq!(models["object"], "list");
}

// ============= Reasoning Effort Integration Tests =============

/// Test chat completion with the new reasoning fields (echo, reasoning_effort, include_reasoning)
#[tokio::test]
async fn test_openai_router_chat_with_reasoning_fields() {
    let mock_server = MockOpenAIServer::new().await;
    let base_url = mock_server.base_url();
    let router = OpenAIRouter::new(base_url, None).await.unwrap();

    // Create a chat request with all three new reasoning fields
    let val = json!({
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ],
        "max_tokens": 100,
        "echo": true,
        "reasoning_effort": "low",
        "include_reasoning": false
    });
    let chat_request: ChatCompletionRequest = serde_json::from_value(val).unwrap();

    // Verify the fields are correctly deserialized
    assert_eq!(chat_request.echo, Some(true));
    assert!(!chat_request.include_reasoning);
    assert!(matches!(
        chat_request.reasoning_effort,
        Some(vllm_router_rs::protocols::spec::ReasoningEffort::Low)
    ));

    // Route the request to mock server
    let response = router.route_chat(None, &chat_request, None).await;

    // Should get a successful response
    assert_eq!(response.status(), StatusCode::OK);

    let (_, body) = response.into_parts();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
    let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
    let chat_response: serde_json::Value = serde_json::from_str(&body_str).unwrap();

    // Verify it's a valid chat completion response
    assert_eq!(chat_response["object"], "chat.completion");
}
type CapturedRequests = Arc<Mutex<Vec<Value>>>;

struct RouterFixture {
    router: Box<dyn RouterTrait>,
    requests: CapturedRequests,
    upstream: JoinHandle<()>,
}

impl RouterFixture {
    async fn start() -> Self {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new()
            .route("/health", get(|| async { StatusCode::OK }))
            .route("/v1/chat/completions", post(capture_chat_request))
            .with_state(requests.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr: SocketAddr = listener.local_addr().unwrap();
        let upstream = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let config = RouterConfig {
            mode: RoutingMode::Regular {
                worker_urls: vec![format!("http://{addr}")],
            },
            worker_startup_timeout_secs: 2,
            worker_startup_check_interval_secs: 1,
            ..Default::default()
        };
        let context = common::create_test_context(config);
        let router = RouterFactory::create_router(&context).await.unwrap();

        Self {
            router,
            requests,
            upstream,
        }
    }

    fn requests(&self) -> Vec<Value> {
        self.requests.lock().unwrap().clone()
    }
}

impl Drop for RouterFixture {
    fn drop(&mut self) {
        self.upstream.abort();
    }
}

async fn capture_chat_request(
    State(requests): State<CapturedRequests>,
    Json(request): Json<Value>,
) -> Response {
    requests.lock().unwrap().push(request.clone());

    if request["stream"] == true {
        return (
            [(CONTENT_TYPE, "text/event-stream")],
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n",
        )
            .into_response();
    }

    let has_tool_result = request["messages"]
        .as_array()
        .unwrap()
        .iter()
        .any(|message| message["role"] == "tool");
    let response = if has_tool_result {
        json!({"choices":[{"message":{"role":"assistant","content":"42"},"finish_reason":"stop"}]})
    } else {
        json!({"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"exec","arguments":"{\"operation\":\"add\",\"left\":20,\"right\":22}"}}]},"finish_reason":"tool_calls"}]})
    };
    Json(response).into_response()
}

fn function_request(stream: bool) -> Value {
    json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Add 20 and 22."}],
        "stream": stream,
        "tool_choice": "auto",
        "parallel_tool_calls": false,
        "tools": [
            {"type":"function","function":{"name":"exec","parameters":{"type":"object"},"strict":true}},
            {"type":"function","function":{"name":"optional","parameters":{"type":"object"},"strict":false}},
            {"type":"function","function":{"name":"legacy","parameters":{"type":"object"}}}
        ]
    })
}

fn assert_tools_forwarded(request: &Value) {
    let expected = function_request(false);
    assert_eq!(request["tools"], expected["tools"]);
    assert_eq!(request["tool_choice"], "auto");
    assert_eq!(request["parallel_tool_calls"], false);
}

#[tokio::test]
async fn function_strict_is_forwarded_and_tool_result_can_continue() {
    let fixture = RouterFixture::start().await;
    let initial_value = function_request(false);
    let initial: ChatCompletionRequest = serde_json::from_value(initial_value.clone()).unwrap();

    let first = fixture.router.route_chat(None, &initial, None).await;
    assert_eq!(first.status(), StatusCode::OK);
    let first_body = axum::body::to_bytes(first.into_body(), usize::MAX)
        .await
        .unwrap();
    let first: Value = serde_json::from_slice(&first_body).unwrap();
    let message = &first["choices"][0]["message"];
    let tool_call = &message["tool_calls"][0];
    assert_eq!(tool_call["function"]["name"], "exec");

    // Execute only this fixed, controlled fixture operation.
    let args: Value =
        serde_json::from_str(tool_call["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["operation"], "add");
    let result = args["left"].as_i64().unwrap() + args["right"].as_i64().unwrap();
    assert_eq!(result, 42);

    let mut continuation_value = initial_value;
    let messages = continuation_value["messages"].as_array_mut().unwrap();
    messages.push(message.clone());
    messages
        .push(json!({"role":"tool","tool_call_id":tool_call["id"],"content":result.to_string()}));
    let continuation: ChatCompletionRequest = serde_json::from_value(continuation_value).unwrap();
    let final_response = fixture.router.route_chat(None, &continuation, None).await;
    let final_body = axum::body::to_bytes(final_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let final_response: Value = serde_json::from_slice(&final_body).unwrap();
    assert_eq!(final_response["choices"][0]["message"]["content"], "42");

    let mut streaming_value = function_request(true);
    streaming_value["tools"] = function_request(false)["tools"].clone();
    let streaming: ChatCompletionRequest = serde_json::from_value(streaming_value).unwrap();
    let stream_response = fixture.router.route_chat(None, &streaming, None).await;
    let stream_body = axum::body::to_bytes(stream_response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert!(String::from_utf8(stream_body.to_vec())
        .unwrap()
        .contains("[DONE]"));

    let forwarded = fixture.requests();
    assert_eq!(forwarded.len(), 3);
    for request in &forwarded {
        assert_tools_forwarded(request);
    }
    assert_eq!(forwarded[0]["stream"], false);
    assert_eq!(forwarded[2]["stream"], true);
    let continued_messages = forwarded[1]["messages"].as_array().unwrap();
    assert_eq!(continued_messages.last().unwrap()["role"], "tool");
    assert_eq!(continued_messages.last().unwrap()["content"], "42");
}
