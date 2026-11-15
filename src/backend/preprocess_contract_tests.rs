//! HTTP preparation contract with render and tokenize endpoints.

#![cfg(all(test, feature = "cpu", feature = "ready"))]

use super::{prepare_http_token_inputs, warm_http_render_support, HttpTokenInput};
use crate::core::{BasicWorker, Worker, WorkerRegistry, WorkerType};
use axum::{
    extract::{Path, State},
    http::{header::AUTHORIZATION, HeaderMap, StatusCode, Uri},
    routing::post,
    Json, Router,
};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, RwLock},
    time::Duration,
};
use tokio::task::JoinHandle;

const DISABLED_MESSAGE: &str =
    "Scale-out endpoints are disabled. Restart vllm serve with --enable-scale-out to enable them.";

#[derive(Clone)]
struct Reply {
    status: StatusCode,
    body: String,
}

impl Reply {
    fn json(status: StatusCode, body: Value) -> Self {
        Self {
            status,
            body: body.to_string(),
        }
    }

    fn disabled() -> Self {
        Self::json(
            StatusCode::FORBIDDEN,
            json!({"error": {
                "message": DISABLED_MESSAGE,
                "type": "PermissionDeniedError",
                "param": null,
                "code": 403
            }}),
        )
    }

    fn tokenized() -> Self {
        Self::json(
            StatusCode::OK,
            json!({"tokens": [101, 102], "count": 2, "max_model_len": 100}),
        )
    }
}

#[derive(Clone)]
struct Replies {
    render: Reply,
    tokenize: Reply,
    probe: Reply,
    required_auth: Option<String>,
    probe_delay: Duration,
}

impl Replies {
    fn new(render: Reply) -> Self {
        Self {
            render,
            tokenize: Reply::tokenized(),
            required_auth: None,
            probe_delay: Duration::ZERO,
            probe: Reply::json(
                StatusCode::UNPROCESSABLE_ENTITY,
                json!({"detail":"Input should be an object"}),
            ),
        }
    }
}

#[derive(Clone, Debug)]
struct SeenRequest {
    worker: String,
    path: String,
    authorization: Option<String>,
    body: Value,
}

struct FixtureState {
    workers: RwLock<HashMap<String, Replies>>,
    seen: Mutex<Vec<SeenRequest>>,
}

struct Fixture {
    base: String,
    state: Arc<FixtureState>,
    server: JoinHandle<()>,
    client: reqwest::Client,
    instances: HashMap<String, Arc<dyn Worker>>,
}

impl Fixture {
    async fn start(workers: Vec<(&str, Replies)>) -> Self {
        let state = Arc::new(FixtureState {
            workers: RwLock::new(
                workers
                    .into_iter()
                    .map(|(name, replies)| (name.to_owned(), replies))
                    .collect(),
            ),
            seen: Mutex::new(Vec::new()),
        });
        let app = Router::new()
            .route("/{worker}/v1/completions/render", post(respond))
            .route("/{worker}/v1/chat/completions/render", post(respond))
            .route("/{worker}/v1/responses/render", post(respond))
            .route("/{worker}/tokenize", post(respond))
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let instances = state
            .workers
            .read()
            .unwrap()
            .keys()
            .map(|name| {
                let url = format!("{base}/{name}");
                (
                    url.clone(),
                    Arc::new(BasicWorker::new(url, WorkerType::Regular)) as Arc<dyn Worker>,
                )
            })
            .collect();
        Self {
            base,
            state,
            server,
            client: reqwest::Client::builder().no_proxy().build().unwrap(),
            instances,
        }
    }

    fn worker(&self, name: &str) -> String {
        format!("{}/{}", self.base, name)
    }

    fn targets(&self, urls: Vec<String>) -> Vec<Arc<dyn Worker>> {
        urls.iter().map(|url| self.instances[url].clone()).collect()
    }

    fn set_render(&self, worker: &str, reply: Reply) {
        self.state
            .workers
            .write()
            .unwrap()
            .get_mut(worker)
            .unwrap()
            .render = reply;
    }

    async fn prepare(
        &self,
        worker: &str,
        body: &Value,
        route: &str,
        headers: Option<&HeaderMap>,
        api_key: Option<&str>,
    ) -> Result<HttpTokenInput, &'static str> {
        let url = self.worker(worker);
        prepare_http_token_inputs(
            &self.client,
            self.targets(vec![url.clone()]),
            body,
            route,
            headers,
            api_key,
            Duration::from_secs(3),
        )
        .await
        .remove(&url)
        .unwrap()
    }

    fn seen(&self, worker: &str) -> Vec<SeenRequest> {
        self.state
            .seen
            .lock()
            .unwrap()
            .iter()
            .filter(|request| request.worker == worker)
            .cloned()
            .collect()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}

async fn respond(
    State(state): State<Arc<FixtureState>>,
    Path(worker): Path<String>,
    uri: Uri,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> (StatusCode, [(&'static str, &'static str); 1], String) {
    let probe = uri.path().ends_with("/render") && body.is_array();
    let replies = state.workers.read().unwrap().get(&worker).unwrap().clone();
    let reply = &replies;
    let reply = if probe && reply.render.status.is_success() {
        &reply.probe
    } else if uri.path().ends_with("/render") {
        &reply.render
    } else {
        &reply.tokenize
    };
    let authorized = replies.required_auth.as_ref().is_none_or(|expected| {
        headers
            .get(AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            == Some(expected.as_str())
    });
    state.seen.lock().unwrap().push(SeenRequest {
        worker,
        path: uri.path().to_owned(),
        authorization: headers
            .get(AUTHORIZATION)
            .map(|value| value.to_str().unwrap().to_owned()),
        body,
    });
    if probe && !replies.probe_delay.is_zero() {
        tokio::time::sleep(replies.probe_delay).await;
    }
    if !authorized {
        return (
            StatusCode::UNAUTHORIZED,
            [("content-type", "application/json")],
            json!({"error":"Invalid API key"}).to_string(),
        );
    }
    (
        reply.status,
        [("content-type", "application/json")],
        reply.body.clone(),
    )
}

#[tokio::test]
async fn only_confirmed_disabled_render_or_missing_endpoint_uses_tokenize() {
    let exact_error = json!({"error": {
        "message": DISABLED_MESSAGE,
        "type": "PermissionDeniedError",
        "code": 403
    }});
    let fixture = Fixture::start(vec![
        ("disabled", Replies::new(Reply::disabled())),
        (
            "unauthorized",
            Replies::new(Reply::json(StatusCode::UNAUTHORIZED, exact_error.clone())),
        ),
        (
            "forbidden",
            Replies::new(Reply::json(
                StatusCode::FORBIDDEN,
                json!({"error":{"type":"PermissionDeniedError","message":"Access denied","code":403}}),
            )),
        ),
        (
            "wrong_type",
            Replies::new(Reply::json(
                StatusCode::FORBIDDEN,
                json!({"error":{"type":"AuthenticationError","message":DISABLED_MESSAGE,"code":403}}),
            )),
        ),
        (
            "wrong_code",
            Replies::new(Reply::json(
                StatusCode::FORBIDDEN,
                json!({"error":{"type":"PermissionDeniedError","message":DISABLED_MESSAGE,"code":401}}),
            )),
        ),
        (
            "malformed",
            Replies::new(Reply {
                status: StatusCode::FORBIDDEN,
                body: "not-json".to_owned(),
            }),
        ),
        (
            "missing",
            Replies::new(Reply::json(StatusCode::NOT_FOUND, json!({}))),
        ),
        (
            "method",
            Replies::new(Reply::json(StatusCode::METHOD_NOT_ALLOWED, json!({}))),
        ),
        (
            "unimplemented",
            Replies::new(Reply::json(StatusCode::NOT_IMPLEMENTED, json!({}))),
        ),
        (
            "tokenize_forbidden",
            Replies {
                tokenize: Reply::disabled(),
                ..Replies::new(Reply::json(StatusCode::NOT_FOUND, json!({})))
            },
        ),
    ])
    .await;
    let names = [
        "disabled",
        "unauthorized",
        "forbidden",
        "wrong_type",
        "wrong_code",
        "malformed",
        "missing",
        "method",
        "unimplemented",
        "tokenize_forbidden",
    ];
    let body = json!({"model":"m", "prompt":"Hello"});
    let results = prepare_http_token_inputs(
        &fixture.client,
        fixture.targets(names.iter().map(|name| fixture.worker(name)).collect()),
        &body,
        "/v1/completions",
        None,
        None,
        Duration::from_secs(3),
    )
    .await;

    for name in ["disabled", "missing", "method", "unimplemented"] {
        let input = results[&fixture.worker(name)].as_ref().unwrap();
        assert_eq!(input.source, "tokenize", "{name}");
        assert_eq!(input.prompts.as_ref(), &[vec![101, 102]], "{name}");
        let seen = fixture.seen(name);
        assert_eq!(seen.len(), 2, "{name}");
        assert!(seen[0].path.ends_with("/v1/completions/render"), "{name}");
        assert_eq!(seen[0].body, json!([]), "{name}");
        assert!(seen[1].path.ends_with("/tokenize"), "{name}");
        assert_eq!(seen[1].body, body, "{name}");
    }
    for name in [
        "unauthorized",
        "forbidden",
        "wrong_type",
        "wrong_code",
        "malformed",
    ] {
        assert_eq!(
            results[&fixture.worker(name)].as_ref().err(),
            Some(&"preparation_rejected"),
            "{name}"
        );
        let seen = fixture.seen(name);
        assert_eq!(seen.len(), 1, "{name}");
        assert!(seen[0].path.ends_with("/render"), "{name}");
    }
    assert_eq!(
        results[&fixture.worker("tokenize_forbidden")]
            .as_ref()
            .err(),
        Some(&"preparation_rejected")
    );
    let seen = fixture.seen("tokenize_forbidden");
    assert_eq!(seen.len(), 2);
    assert!(seen[1].path.ends_with("/tokenize"));
}

#[tokio::test]
async fn fallback_forwards_authorization_and_prefers_configured_api_key() {
    let fixture = Fixture::start(vec![
        ("forwarded", Replies::new(Reply::disabled())),
        ("configured", Replies::new(Reply::disabled())),
    ])
    .await;
    let body = json!({"model":"m", "prompt":"Hello"});
    let mut headers = HeaderMap::new();
    headers.insert(AUTHORIZATION, "Bearer incoming".parse().unwrap());

    for (name, api_key, expected) in [
        ("forwarded", None, "Bearer incoming"),
        ("configured", Some("backend-key"), "Bearer backend-key"),
    ] {
        let url = fixture.worker(name);
        let results = prepare_http_token_inputs(
            &fixture.client,
            fixture.targets(vec![url.clone()]),
            &body,
            "/v1/completions",
            Some(&headers),
            api_key,
            Duration::from_secs(3),
        )
        .await;
        assert_eq!(results[&url].as_ref().unwrap().source, "tokenize");
        let seen = fixture.seen(name);
        assert_eq!(seen.len(), 2, "{name}");
        for request in seen {
            assert_eq!(request.authorization.as_deref(), Some(expected), "{name}");
        }
    }
}

#[tokio::test]
async fn disabled_render_never_receives_complex_chat_or_falls_back_inexactly() {
    let fixture = Fixture::start(vec![
        ("tools", Replies::new(Reply::disabled())),
        ("prefill_ids", Replies::new(Reply::disabled())),
    ])
    .await;
    for (name, body) in [
        (
            "tools",
            json!({"model":"m","messages":[{"role":"user","content":"Hello"}],
                "tools":[{"type":"function","function":{"name":"lookup"}}]}),
        ),
        (
            "prefill_ids",
            json!({"model":"m","messages":[{"role":"user","content":"Hello"}],
                "kv_transfer_params":{"prompt_token_ids":[7,8]}}),
        ),
    ] {
        let url = fixture.worker(name);
        let results = prepare_http_token_inputs(
            &fixture.client,
            fixture.targets(vec![url.clone()]),
            &body,
            "/v1/chat/completions",
            None,
            None,
            Duration::from_secs(3),
        )
        .await;
        assert_eq!(results[&url].as_ref().err(), Some(&"render_unavailable"));
        let seen = fixture.seen(name);
        assert_eq!(seen.len(), 1, "{name}");
        assert!(seen[0].path.ends_with("/v1/chat/completions/render"));
        assert_eq!(seen[0].body, json!([]), "{name}");
    }
}

#[tokio::test]
async fn successful_render_survives_another_workers_rejection() {
    let fixture = Fixture::start(vec![
        (
            "good",
            Replies::new(Reply::json(StatusCode::OK, json!({"token_ids":[7,8,9]}))),
        ),
        (
            "bad",
            Replies::new(Reply::json(StatusCode::INTERNAL_SERVER_ERROR, json!({}))),
        ),
    ])
    .await;
    let good = fixture.worker("good");
    let bad = fixture.worker("bad");
    let body = json!({"model":"m", "prompt":"Hello"});
    let results = prepare_http_token_inputs(
        &fixture.client,
        fixture.targets(vec![good.clone(), bad.clone()]),
        &body,
        "/v1/completions",
        None,
        None,
        Duration::from_secs(3),
    )
    .await;
    let input = results[&good].as_ref().unwrap();
    assert_eq!(input.source, "render");
    assert_eq!(input.prompts.as_ref(), &[vec![7, 8, 9]]);
    assert_eq!(results[&bad].as_ref().err(), Some(&"preparation_rejected"));
    for name in ["good", "bad"] {
        let seen = fixture.seen(name);
        assert_eq!(seen[0].body, json!([]), "{name}");
        if name == "good" {
            assert_eq!(seen.len(), 2, "{name}");
            assert_eq!(seen[1].body, body, "{name}");
        } else {
            assert_eq!(seen.len(), 1, "{name}");
        }
    }
}

#[tokio::test]
async fn enabled_render_receives_the_original_request_after_validation_probe() {
    let rendered = Replies::new(Reply::json(StatusCode::OK, json!({"token_ids":[7,8,9]})));
    let mut bad_request = rendered.clone();
    bad_request.probe = Reply::json(StatusCode::BAD_REQUEST, json!({"error":"Invalid request"}));
    let mut successful_probe = rendered.clone();
    successful_probe.probe = Reply::json(StatusCode::OK, json!({"token_ids":[9999]}));
    let fixture = Fixture::start(vec![
        ("validation", rendered),
        ("bad_request", bad_request),
        ("success", successful_probe),
    ])
    .await;
    for (worker, route, body) in [
        (
            "validation",
            "/v1/chat/completions",
            json!({"model":"m","messages":[{"role":"user","content":"Hello"}],
                "tools":[{"type":"function","function":{"name":"lookup"}}],
                "chat_template_kwargs":{"enable_thinking":false}}),
        ),
        (
            "bad_request",
            "/v1/completions",
            json!({"model":"m","prompt":"Hello","truncate_prompt_tokens":3}),
        ),
        (
            "success",
            "/v1/responses",
            json!({"model":"m","input":"Hello","instructions":"Be concise"}),
        ),
    ] {
        let url = fixture.worker(worker);
        let inputs = prepare_http_token_inputs(
            &fixture.client,
            fixture.targets(vec![url.clone()]),
            &body,
            route,
            None,
            Some("worker-key"),
            Duration::from_secs(3),
        )
        .await;
        let input = inputs[&url].as_ref().unwrap();
        assert_eq!(input.source, "render");
        assert_eq!(input.prompts.as_ref(), &[vec![7, 8, 9]]);
        let requests = fixture.seen(worker);
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].body, json!([]));
        assert_eq!(requests[1].body, body);
        assert!(requests.iter().all(|request| {
            request.path == format!("/{worker}{route}/render")
                && request.authorization.as_deref() == Some("Bearer worker-key")
        }));
    }
}

#[tokio::test]
async fn direct_token_ids_do_not_probe_render() {
    let fixture = Fixture::start(vec![("direct", Replies::new(Reply::disabled()))]).await;
    let url = fixture.worker("direct");
    let inputs = prepare_http_token_inputs(
        &fixture.client,
        fixture.targets(vec![url.clone()]),
        &json!({"model":"m","prompt":[7,8,9]}),
        "/v1/completions",
        None,
        None,
        Duration::from_secs(3),
    )
    .await;
    assert_eq!(inputs[&url].as_ref().unwrap().source, "token_ids");
    assert_eq!(
        inputs[&url].as_ref().unwrap().prompts.as_ref(),
        &[vec![7, 8, 9]]
    );
    assert!(fixture.seen("direct").is_empty());
}

#[tokio::test]
async fn registration_and_concurrent_first_use_each_probe_only_once() {
    let rendered = Replies::new(Reply::json(StatusCode::OK, json!({"token_ids":[7,8,9]})));
    let fixture = Fixture::start(vec![
        ("enabled", rendered.clone()),
        ("disabled", Replies::new(Reply::disabled())),
        ("cold", rendered),
    ])
    .await;
    warm_http_render_support(
        &fixture.client,
        &fixture.targets(vec![fixture.worker("enabled"), fixture.worker("disabled")]),
        None,
    )
    .await;
    let body = json!({"model":"m","messages":[{"role":"user","content":"Hello"}]});
    for (worker, source, ids, probes) in [
        ("enabled", "render", vec![7, 8, 9], 3),
        ("disabled", "tokenize", vec![101, 102], 3),
        ("cold", "render", vec![7, 8, 9], 1),
    ] {
        let results = futures_util::future::join_all(
            (0..8).map(|_| fixture.prepare(worker, &body, "/v1/chat/completions", None, None)),
        )
        .await;
        for result in results {
            let input = result.unwrap();
            assert_eq!(input.source, source);
            assert_eq!(input.prompts.as_ref(), std::slice::from_ref(&ids));
        }
        let seen = fixture.seen(worker);
        assert_eq!(
            seen.iter()
                .filter(|request| request.body == json!([]))
                .count(),
            probes
        );
        assert_eq!(
            seen.iter().filter(|request| request.body == body).count(),
            8
        );
    }
}

#[tokio::test]
async fn failed_unauthenticated_warm_retries_and_never_caches_credentials() {
    let mut replies = Replies::new(Reply::json(StatusCode::OK, json!({"token_ids":[7,8,9]})));
    replies.required_auth = Some("Bearer first-key".to_owned());
    let fixture = Fixture::start(vec![("auth", replies)]).await;
    let targets = fixture.targets(vec![fixture.worker("auth")]);
    warm_http_render_support(&fixture.client, &targets, None).await;
    assert!(targets[0]
        .http_render_support(super::HTTP_RENDER_PATHS[0])
        .unwrap()
        .get()
        .is_none());

    let body = json!({"model":"m","messages":[{"role":"user","content":"Hello"}]});
    assert_eq!(
        fixture
            .prepare(
                "auth",
                &body,
                "/v1/chat/completions",
                None,
                Some("first-key")
            )
            .await
            .unwrap()
            .source,
        "render"
    );
    fixture
        .state
        .workers
        .write()
        .unwrap()
        .get_mut("auth")
        .unwrap()
        .required_auth = Some("Bearer second-key".to_owned());
    let mut headers = HeaderMap::new();
    headers.insert(AUTHORIZATION, "Bearer second-key".parse().unwrap());
    assert_eq!(
        fixture
            .prepare("auth", &body, "/v1/chat/completions", Some(&headers), None)
            .await
            .unwrap()
            .source,
        "render"
    );
    headers.insert(AUTHORIZATION, "Bearer wrong-key".parse().unwrap());
    assert_eq!(
        fixture
            .prepare("auth", &body, "/v1/chat/completions", Some(&headers), None)
            .await
            .err(),
        Some("preparation_rejected")
    );
    headers.insert(AUTHORIZATION, "Bearer second-key".parse().unwrap());
    assert_eq!(
        fixture
            .prepare("auth", &body, "/v1/chat/completions", Some(&headers), None)
            .await
            .unwrap()
            .source,
        "render"
    );

    let seen = fixture.seen("auth");
    assert_eq!(
        seen.iter()
            .filter(|request| request.body == json!([]))
            .count(),
        4
    );
    let actual: Vec<_> = seen
        .iter()
        .filter(|request| request.body == body)
        .map(|request| request.authorization.as_deref().unwrap())
        .collect();
    assert_eq!(
        actual,
        [
            "Bearer first-key",
            "Bearer second-key",
            "Bearer wrong-key",
            "Bearer second-key"
        ]
    );
    assert!(seen.iter().all(|request| request.path.ends_with("/render")));
}

#[tokio::test]
async fn cancelled_registration_probe_can_be_initialized_on_first_use() {
    let mut replies = Replies::new(Reply::json(StatusCode::OK, json!({"token_ids":[7,8,9]})));
    replies.probe_delay = Duration::from_secs(5);
    let fixture = Fixture::start(vec![("slow", replies)]).await;
    let targets = fixture.targets(vec![fixture.worker("slow")]);
    tokio::time::timeout(
        Duration::from_secs(2),
        warm_http_render_support(&fixture.client, &targets, None),
    )
    .await
    .expect("optional warmup must be bounded");
    assert!(targets[0]
        .http_render_support(super::HTTP_RENDER_PATHS[0])
        .unwrap()
        .get()
        .is_none());
    fixture
        .state
        .workers
        .write()
        .unwrap()
        .get_mut("slow")
        .unwrap()
        .probe_delay = Duration::ZERO;
    let body = json!({"model":"m","messages":[{"role":"user","content":"Hello"}]});
    assert_eq!(
        fixture
            .prepare("slow", &body, "/v1/chat/completions", None, None)
            .await
            .unwrap()
            .source,
        "render"
    );
}

#[tokio::test]
async fn confirmed_unavailable_downgrades_support_but_real_forbidden_does_not() {
    let fixture = Fixture::start(vec![(
        "changed",
        Replies::new(Reply::json(StatusCode::OK, json!({"token_ids":[7,8,9]}))),
    )])
    .await;
    let targets = fixture.targets(vec![fixture.worker("changed")]);
    warm_http_render_support(&fixture.client, &targets, None).await;
    let body = json!({"model":"m","messages":[{"role":"user","content":"Hello"}]});
    fixture.set_render(
        "changed",
        Reply::json(
            StatusCode::FORBIDDEN,
            json!({"error":{"type":"PermissionDeniedError","code":403,"message":"Access denied"}}),
        ),
    );
    assert_eq!(
        fixture
            .prepare("changed", &body, "/v1/chat/completions", None, None)
            .await
            .err(),
        Some("preparation_rejected")
    );
    fixture.set_render("changed", Reply::disabled());
    for _ in 0..2 {
        assert_eq!(
            fixture
                .prepare("changed", &body, "/v1/chat/completions", None, None)
                .await
                .unwrap()
                .source,
            "tokenize"
        );
    }
    let seen = fixture.seen("changed");
    assert_eq!(
        seen.iter()
            .filter(|request| request.body == json!([]))
            .count(),
        3
    );
    assert_eq!(
        seen.iter()
            .filter(|request| request.path.ends_with("/render") && request.body == body)
            .count(),
        2
    );
    assert_eq!(
        seen.iter()
            .filter(|request| request.path.ends_with("/tokenize"))
            .count(),
        2
    );
}

#[tokio::test]
async fn route_capabilities_are_independent_and_reregistering_worker_resets_them() {
    let fixture = Fixture::start(vec![("replaced", Replies::new(Reply::disabled()))]).await;
    let url = fixture.worker("replaced");
    let old = fixture.targets(vec![url.clone()]).remove(0);
    let registry = WorkerRegistry::new();
    let id = registry.register(old.clone());
    let chat = json!({"model":"m","messages":[{"role":"user","content":"Hello"}]});
    assert_eq!(
        fixture
            .prepare("replaced", &chat, "/v1/chat/completions", None, None)
            .await
            .unwrap()
            .source,
        "tokenize"
    );
    fixture.set_render(
        "replaced",
        Reply::json(StatusCode::OK, json!({"token_ids":[7,8,9]})),
    );
    assert_eq!(
        fixture
            .prepare(
                "replaced",
                &json!({"model":"m","prompt":"Hello"}),
                "/v1/completions",
                None,
                None
            )
            .await
            .unwrap()
            .source,
        "render"
    );

    let replacement: Arc<dyn Worker> = Arc::new(BasicWorker::new(url.clone(), WorkerType::Regular));
    registry.remove(&id).unwrap();
    assert_ne!(registry.register(replacement.clone()), id);
    let current = registry.get_by_model_fast("unknown");
    assert_eq!(current.len(), 1);
    assert!(Arc::ptr_eq(&current[0], &replacement));
    assert!(replacement
        .http_render_support(super::HTTP_RENDER_PATHS[0])
        .unwrap()
        .get()
        .is_none());
    warm_http_render_support(&fixture.client, &current, None).await;
    let result = prepare_http_token_inputs(
        &fixture.client,
        current,
        &chat,
        "/v1/chat/completions",
        None,
        None,
        Duration::from_secs(3),
    )
    .await;
    assert_eq!(result[&url].as_ref().unwrap().source, "render");
    assert!(!old
        .http_render_support(super::HTTP_RENDER_PATHS[0])
        .unwrap()
        .get()
        .unwrap()
        .load(std::sync::atomic::Ordering::Acquire));
}
