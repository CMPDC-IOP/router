//! Regression tests for static worker startup race handling.
//!
//! The router must be able to serve traffic as soon as one worker is healthy,
//! while workers that failed the startup probe stay in the registry as
//! unhealthy (never routable) until the background health checker recovers
//! them. A startup that cannot reach any worker must time out with a clear
//! error.

mod common;

use common::mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use vllm_router_rs::config::{PolicyConfig, RouterConfig, RoutingMode};
use vllm_router_rs::protocols::spec::ChatCompletionRequest;
use vllm_router_rs::routers::{RouterFactory, RouterTrait};
use vllm_router_rs::server::AppContext;

fn worker_config(port: u16) -> MockWorkerConfig {
    MockWorkerConfig {
        port,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
        stream_chunk_delay_ms: 0,
    }
}

/// Reserve an OS port that nothing is listening on yet.
async fn reserve_port() -> u16 {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .unwrap();
    let port = listener.local_addr().unwrap().port();
    // Close the listener: the mock worker will bind this exact port later.
    port
}

struct TestContext {
    workers: Vec<MockWorker>,
    router: Arc<dyn RouterTrait>,
    app_context: Arc<AppContext>,
    _health_checker: vllm_router_rs::core::HealthChecker,
}

impl TestContext {
    /// Build a router for the given URLs, with a fast background health
    /// checker so late-starting workers are recovered quickly.
    async fn new(worker_urls: Vec<String>, startup_timeout_secs: u64) -> Self {
        let config = RouterConfig {
            mode: RoutingMode::Regular { worker_urls },
            policy: PolicyConfig::CacheAware {
                cache_threshold: 0.3,
                balance_abs_threshold: 999,
                balance_rel_threshold: 9.9,
                eviction_interval_secs: 0,
                max_tree_size: 1000,
            },
            port: 3005,
            worker_startup_timeout_secs: startup_timeout_secs,
            worker_startup_check_interval_secs: 1,
            health_check: vllm_router_rs::config::HealthCheckConfig {
                check_interval_secs: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        let app_context = common::create_test_context(config);
        let router = RouterFactory::create_router(&app_context).await.unwrap();
        let router = Arc::from(router);

        // Production starts this in server::startup; tests need it for the
        // late-recovery scenarios.
        let health_checker = app_context.worker_registry.start_health_checker(1);

        // Give the background checker a moment before tests assert.
        tokio::time::sleep(Duration::from_millis(200)).await;

        Self {
            workers: Vec::new(),
            router,
            app_context,
            _health_checker: health_checker,
        }
    }

    fn worker(&self, url: &str) -> Arc<dyn vllm_router_rs::core::Worker> {
        self.app_context
            .worker_registry
            .get_by_url(url)
            .unwrap_or_else(|| panic!("worker {url} should be registered"))
    }

    async fn chat_request(&self) -> axum::response::Response {
        let body: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false,
        }))
        .expect("valid chat request");

        self.router.route_chat(None, &body, None).await
    }

    async fn shutdown(mut self) {
        tokio::time::sleep(Duration::from_millis(100)).await;
        self._health_checker.shutdown().await;
        for worker in &mut self.workers {
            worker.stop().await;
        }
    }
}

/// A healthy worker A plus a not-yet-started worker B: the router must serve
/// traffic from A only, and B must never enter the routable pool while down.
#[tokio::test]
async fn test_only_healthy_worker_is_routable() {
    let mut worker_a = MockWorker::new(worker_config(0));
    let url_a = worker_a.start().await.unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Port that will fail to connect until B actually starts.
    let b_port = reserve_port().await;
    let url_b = format!("http://127.0.0.1:{b_port}");

    let mut ctx = TestContext::new(vec![url_a.clone(), url_b.clone()], 3).await;

    // Both workers are registered, but only A is routable.
    let worker_a_handle = ctx.worker(&url_a);
    let worker_b_handle = ctx.worker(&url_b);
    assert!(
        worker_a_handle.is_healthy(),
        "healthy worker must be routable"
    );
    assert!(
        !worker_b_handle.is_healthy(),
        "worker that failed the startup probe must be unhealthy"
    );

    // Requests must succeed and only ever reach A.
    for _ in 0..5 {
        let response = ctx.chat_request().await;
        assert!(
            response.status().is_success(),
            "routing must use the healthy worker, got {}",
            response.status()
        );
    }
    assert_eq!(
        worker_a_handle.load(),
        0,
        "load must return to zero after successful requests"
    );
    assert_eq!(
        worker_b_handle.load(),
        0,
        "unreachable worker must not receive traffic"
    );

    ctx.workers.push(worker_a);
    ctx.shutdown().await;
}

/// A worker that starts after the router must be recovered by the background
/// health checker and brought into the routable pool automatically.
#[tokio::test]
async fn test_late_worker_recovers_and_becomes_routable() {
    let mut worker_a = MockWorker::new(worker_config(0));
    let url_a = worker_a.start().await.unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;

    let b_port = reserve_port().await;
    let url_b = format!("http://127.0.0.1:{b_port}");

    let mut ctx = TestContext::new(vec![url_a.clone(), url_b.clone()], 3).await;
    assert!(!ctx.worker(&url_b).is_healthy(), "B starts unhealthy");

    // Start B after the router is already serving.
    let mut worker_b = MockWorker::new(worker_config(b_port));
    let url_b_started = worker_b.start().await.unwrap();
    assert_eq!(url_b_started, url_b, "B must bind the reserved port");

    // Wait for the background health checker (1s interval, success threshold).
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        if ctx.worker(&url_b).is_healthy() {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "late worker must be recovered by the health checker"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // A recovered worker must not carry stale load from before it was up.
    let worker_b_handle = ctx.worker(&url_b);
    assert_eq!(
        worker_b_handle.load(),
        0,
        "recovered worker must start with a clean load counter"
    );

    // And it must now be routable: send requests until B actually serves one.
    let b_port = url_b.rsplit(':').next().unwrap().parse::<u16>().unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    let mut saw_b = false;
    while std::time::Instant::now() < deadline {
        let response = ctx.chat_request().await;
        assert!(response.status().is_success());
        if common::mock_worker::get_captured_requests(b_port)
            .iter()
            .any(|r| r.path == "/v1/chat/completions")
        {
            saw_b = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(saw_b, "recovered worker must be routable");

    ctx.workers.push(worker_a);
    ctx.workers.push(worker_b);
    ctx.shutdown().await;
}

/// If no worker at all is reachable, startup must fail with a clear timeout
/// error instead of serving from an empty pool.
#[tokio::test]
async fn test_startup_times_out_when_no_worker_is_healthy() {
    let port_a = reserve_port().await;
    let port_b = reserve_port().await;
    let urls = vec![
        format!("http://127.0.0.1:{port_a}"),
        format!("http://127.0.0.1:{port_b}"),
    ];

    let config = RouterConfig {
        mode: RoutingMode::Regular { worker_urls: urls },
        port: 3005,
        worker_startup_timeout_secs: 2,
        worker_startup_check_interval_secs: 1,
        ..Default::default()
    };
    let app_context = common::create_test_context(config);

    let result = RouterFactory::create_router(&app_context).await;
    let err = result.expect_err("startup must fail when no worker is healthy");
    assert!(
        err.contains("Timeout"),
        "error should be an explicit timeout, got: {err}"
    );
}
