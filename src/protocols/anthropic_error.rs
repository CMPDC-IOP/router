//! Status-code mapping for the Anthropic Messages endpoint (`/v1/messages`).
//!
//! vLLM workers expose a native `/v1/messages` endpoint, and the router
//! forwards such requests to them through the transparent proxy. When request
//! validation fails inside the worker — for example when the prompt plus the
//! requested output tokens exceed the model context window — the worker's
//! OpenAI layer raises a 400-class `BadRequestError`, but the worker's
//! Anthropic error mapping does not recognize that exception type and answers
//! with HTTP 500 `internal_error`:
//!
//! ```json
//! {"type": "error", "error": {"type": "internal_error", "message": "This model's maximum context length is ..."}}
//! ```
//!
//! Clients such as LiteLLM then classify a client mistake as a server fault,
//! pollute 5xx alerting, and retry requests that can never succeed.
//!
//! [`rewrite_context_overflow`] repairs exactly this case on the router side:
//! a 5xx from `/v1/messages` whose JSON body reports a context-length
//! violation is answered as HTTP 400 `invalid_request_error` with the
//! original message preserved. Every other response — genuine server
//! failures, upstream 4xx, and successful replies — passes through unchanged.

use axum::{
    body::{to_bytes, Body},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
};

/// Upstream error bodies larger than this are not inspected; they are passed
/// through with their original status. Real validation errors are a few
/// hundred bytes, so the cap never engages in practice.
const MAX_INSPECTED_BODY_BYTES: usize = 64 * 1024;

/// Fragments that identify a client-side context-length violation in vLLM
/// error text. The match is deliberately narrow so that unrelated server
/// failures (e.g. a crashed worker) never qualify, and it is only consulted
/// when the worker already returned an unstructured 5xx body.
const CONTEXT_OVERFLOW_MARKERS: [&str; 3] = [
    "This model's maximum context length is",
    "cannot be greater than max_model_len",
    "exceeds model's maximum context length",
];

/// Rewrite worker-side context-overflow `internal_error` responses from
/// `/v1/messages` into Anthropic-style 400 `invalid_request_error` replies.
pub async fn rewrite_context_overflow(path: &str, response: Response) -> Response {
    // Only the Anthropic Messages endpoint is affected by the worker-side
    // mapping; every other transparently proxied path passes through as-is.
    if path != "/v1/messages" || !response.status().is_server_error() {
        return response;
    }

    if !is_json_response(&response) {
        return response;
    }

    let (parts, body) = response.into_parts();
    let bytes = match to_bytes(body, MAX_INSPECTED_BODY_BYTES).await {
        Ok(bytes) => bytes,
        // The body exceeded the inspection cap and cannot be recovered. Keep
        // the upstream status and headers instead of guessing a class.
        Err(_) => return Response::from_parts(parts, Body::empty()),
    };

    let text = String::from_utf8_lossy(&bytes);
    if !is_context_overflow_message(&text) {
        return Response::from_parts(parts, Body::from(bytes));
    }

    let message = extract_error_message(&bytes).unwrap_or_else(|| text.trim().to_string());
    let body = AnthropicErrorResponse {
        kind: "error",
        error: AnthropicErrorBody {
            kind: "invalid_request_error",
            message: &message,
        },
    };
    (StatusCode::BAD_REQUEST, axum::Json(body)).into_response()
}

/// Anthropic-style error envelope. Declared as a struct (instead of a
/// `serde_json::Value`) so the wire format keeps the canonical Anthropic field
/// order, `"type"` before `"error"`.
#[derive(serde::Serialize)]
struct AnthropicErrorResponse<'a> {
    #[serde(rename = "type")]
    kind: &'a str,
    error: AnthropicErrorBody<'a>,
}

#[derive(serde::Serialize)]
struct AnthropicErrorBody<'a> {
    #[serde(rename = "type")]
    kind: &'a str,
    message: &'a str,
}

/// Whether the response carries a JSON body. Error replies that stream SSE are
/// never inspected, so a stalled upstream cannot hold the rewrite open.
fn is_json_response(response: &Response) -> bool {
    response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .is_some_and(|mime| mime.trim().eq_ignore_ascii_case("application/json"))
}

fn is_context_overflow_message(text: &str) -> bool {
    CONTEXT_OVERFLOW_MARKERS
        .iter()
        .any(|marker| text.contains(marker))
}

/// Pull the human-readable message out of a worker error body. Both the
/// Anthropic envelope (`{"type":"error","error":{...}}`) and the OpenAI
/// envelope (`{"error":{...}}`) nest the message at `error.message`.
fn extract_error_message(bytes: &[u8]) -> Option<String> {
    let value: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    value
        .get("error")?
        .get("message")?
        .as_str()
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    /// The exact body observed from a vLLM worker when a request asked for
    /// 128000 output tokens on top of a 134145-token prompt.
    fn vllm_context_overflow_body() -> String {
        serde_json::json!({
            "type": "error",
            "error": {
                "type": "internal_error",
                "message": "This model's maximum context length is 262144 tokens. However, you requested 128000 output tokens and your prompt contains at least 134145 input tokens, for a total of at least 262145 tokens. Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_tokens, value=134145)"
            }
        })
        .to_string()
    }

    fn json_response(status: StatusCode, content_type: &str, body: String) -> Response {
        Response::builder()
            .status(status)
            .header(
                header::CONTENT_TYPE,
                HeaderValue::from_str(content_type).unwrap(),
            )
            .body(Body::from(body))
            .unwrap()
    }

    async fn body_string(response: Response) -> String {
        String::from_utf8(
            to_bytes(response.into_body(), MAX_INSPECTED_BODY_BYTES)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn rewrites_worker_context_overflow_500_to_invalid_request_400() {
        let response = json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "application/json",
            vllm_context_overflow_body(),
        );

        let response = rewrite_context_overflow("/v1/messages", response).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let value: serde_json::Value = serde_json::from_str(&body_string(response).await).unwrap();
        assert_eq!(value["type"], "error");
        assert_eq!(value["error"]["type"], "invalid_request_error");
        assert!(value["error"]["message"]
            .as_str()
            .unwrap()
            .starts_with("This model's maximum context length is 262144 tokens"));
        assert!(value["error"]["message"]
            .as_str()
            .unwrap()
            .ends_with("(parameter=input_tokens, value=134145)"));
    }

    #[tokio::test]
    async fn rewrites_max_completion_tokens_marker() {
        let body = serde_json::json!({
            "type": "error",
            "error": {
                "type": "internal_error",
                "message": "max_completion_tokens=128000 cannot be greater than max_model_len=max_total_tokens=262144. Please reduce the number of requested output tokens."
            }
        })
        .to_string();
        let response = json_response(StatusCode::INTERNAL_SERVER_ERROR, "application/json", body);

        let response = rewrite_context_overflow("/v1/messages", response).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let value: serde_json::Value = serde_json::from_str(&body_string(response).await).unwrap();
        assert_eq!(value["error"]["type"], "invalid_request_error");
        assert!(value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("max_completion_tokens=128000"));
    }

    #[tokio::test]
    async fn rewrites_input_length_marker() {
        let body = serde_json::json!({
            "type": "error",
            "error": {
                "type": "internal_error",
                "message": "Input length (270000 tokens) exceeds model's maximum context length (262144 tokens)."
            }
        })
        .to_string();
        let response = json_response(StatusCode::INTERNAL_SERVER_ERROR, "application/json", body);

        let response = rewrite_context_overflow("/v1/messages", response).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let value: serde_json::Value = serde_json::from_str(&body_string(response).await).unwrap();
        assert_eq!(value["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn emits_canonical_anthropic_field_order() {
        let response = json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "application/json",
            vllm_context_overflow_body(),
        );

        let response = rewrite_context_overflow("/v1/messages", response).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let raw = body_string(response).await;
        assert!(
            raw.starts_with(
                r#"{"type":"error","error":{"type":"invalid_request_error","message":"#
            ),
            "unexpected body: {raw}"
        );
    }

    #[tokio::test]
    async fn keeps_worker_crashed_500_untouched() {
        let body = serde_json::json!({
            "type": "error",
            "error": {
                "type": "internal_error",
                "message": "worker crashed while processing request"
            }
        })
        .to_string();
        let response = json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "application/json",
            body.clone(),
        );

        let response = rewrite_context_overflow("/v1/messages", response).await;
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body_string(response).await, body);
    }

    #[tokio::test]
    async fn keeps_other_5xx_untouched() {
        let body = serde_json::json!({
            "type": "error",
            "error": {
                "type": "internal_error",
                "message": "CUDA out of memory. Tried to allocate 2.00 GiB"
            }
        })
        .to_string();
        let response = json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "application/json",
            body.clone(),
        );

        let response = rewrite_context_overflow("/v1/messages", response).await;
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body_string(response).await, body);
    }

    #[tokio::test]
    async fn keeps_upstream_400_untouched() {
        // A fixed worker already mapping the violation to Anthropic semantics
        // must not be re-wrapped by the router.
        let body = serde_json::json!({
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "This model's maximum context length is 262144 tokens. However, you requested 128000 output tokens."
            }
        })
        .to_string();
        let response = json_response(StatusCode::BAD_REQUEST, "application/json", body.clone());

        let response = rewrite_context_overflow("/v1/messages", response).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(body_string(response).await, body);
    }

    #[tokio::test]
    async fn keeps_success_response_quoting_the_error() {
        // A model answer may quote the error text; successful replies must
        // never be rewritten.
        let body = serde_json::json!({
            "id": "chatcmpl-1",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "This model's maximum context length is 262144 tokens."}
            ]
        })
        .to_string();
        let response = json_response(StatusCode::OK, "application/json", body.clone());

        let response = rewrite_context_overflow("/v1/messages", response).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(body_string(response).await, body);
    }

    #[tokio::test]
    async fn ignores_other_paths() {
        for path in [
            "/generate",
            "/v1/chat/completions",
            "/v1/messages/count_tokens",
        ] {
            let response = rewrite_context_overflow(
                path,
                json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "application/json",
                    vllm_context_overflow_body(),
                ),
            )
            .await;
            assert_eq!(
                response.status(),
                StatusCode::INTERNAL_SERVER_ERROR,
                "{path}"
            );
        }
    }

    #[tokio::test]
    async fn requires_json_content_type() {
        // SSE replies are streamed; inspecting them could stall on a hung
        // upstream, so they are never rewritten.
        let response = json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "text/event-stream",
            vllm_context_overflow_body(),
        );

        let response = rewrite_context_overflow("/v1/messages", response).await;
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn accepts_content_type_with_charset_parameter() {
        let response = json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "application/json; charset=utf-8",
            vllm_context_overflow_body(),
        );

        let response = rewrite_context_overflow("/v1/messages", response).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn falls_back_to_raw_text_when_body_is_not_json() {
        let body = "This model's maximum context length is 262144 tokens".to_string();
        let response = json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "application/json",
            body.clone(),
        );

        let response = rewrite_context_overflow("/v1/messages", response).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let value: serde_json::Value = serde_json::from_str(&body_string(response).await).unwrap();
        assert_eq!(value["error"]["type"], "invalid_request_error");
        assert_eq!(value["error"]["message"], body);
    }

    #[tokio::test]
    async fn preserves_upstream_status_when_body_exceeds_cap() {
        let large_body = format!(
            "This model's maximum context length is 262144 tokens. {}",
            "x".repeat(MAX_INSPECTED_BODY_BYTES)
        );
        let response = json_response(StatusCode::BAD_GATEWAY, "application/json", large_body);

        let response = rewrite_context_overflow("/v1/messages", response).await;
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(body_string(response).await, "");
    }
}
