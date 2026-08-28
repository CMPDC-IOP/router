//! Regression tests for exactly-once worker load tracking.
//!
//! Covers the paths that previously leaked or double-decremented the load
//! counter with the cache-aware policy: retryable failures, streaming
//! completion, streams that embed `data: [DONE]` in message content, client
//! disconnects and concurrent in-flight requests. Health checks must never
//! mutate load accounting.

mod common;

use common::mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType};
use futures_util::StreamExt;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use vllm_router_rs::config::{PolicyConfig, RouterConfig, RoutingMode};
use vllm_router_rs::protocols::spec::ChatCompletionRequest;
use vllm_router_rs::routers::{RouterFactory, RouterTrait};
use vllm_router_rs::server::AppContext;

struct TestContext {
    workers: Vec<MockWorker>,
    router: Arc<dyn RouterTrait>,
    app_context: Arc<AppContext>,
    worker_url: String,
}

impl TestContext {
    async fn new(config: MockWorkerConfig, retry_max_retries: u32) -> Self {
        Self::new_with_options(config, retry_max_retries, 1, 10).await
    }

    async fn new_with_options(
        config: MockWorkerConfig,
        retry_max_retries: u32,
        dp_size: usize,
        retry_backoff_ms: u64,
    ) -> Self {
        let router_config = RouterConfig {
            mode: RoutingMode::Regular {
                worker_urls: vec![],
            },
            policy: PolicyConfig::CacheAware {
                cache_threshold: 0.3,
                balance_abs_threshold: 999,
                balance_rel_threshold: 9.9,
                eviction_interval_secs: 0,
                max_tree_size: 1000,
            },
            port: 3004,
            worker_startup_timeout_secs: 1,
            worker_startup_check_interval_secs: 1,
            retry: vllm_router_rs::config::RetryConfig {
                max_retries: retry_max_retries,
                initial_backoff_ms: retry_backoff_ms,
                max_backoff_ms: retry_backoff_ms,
                backoff_multiplier: 1.0,
                jitter_factor: 0.0,
            },
            intra_node_data_parallel_size: dp_size,
            ..Default::default()
        };

        let mut worker = MockWorker::new(config);
        let worker_url = worker.start().await.unwrap();

        // Give the mock server a moment to accept connections.
        tokio::time::sleep(Duration::from_millis(200)).await;

        let router_config = RouterConfig {
            mode: RoutingMode::Regular {
                worker_urls: vec![worker_url.clone()],
            },
            ..router_config
        };

        let app_context = common::create_test_context(router_config);
        let router = RouterFactory::create_router(&app_context).await.unwrap();
        let router = Arc::from(router);

        tokio::time::sleep(Duration::from_millis(300)).await;

        Self {
            workers: vec![worker],
            router,
            app_context,
            worker_url,
        }
    }

    /// Baseline load, i.e. the number of simulated in-flight requests that
    /// are not part of the request under test.
    fn baseline_load(&self) -> usize {
        self.worker_load()
    }

    fn worker_load(&self) -> usize {
        self.app_context
            .worker_registry
            .get_by_url(&self.worker_url)
            .expect("worker registered")
            .load()
    }

    fn running_requests(&self) -> u64 {
        self.app_context.request_metrics.running_count()
    }

    fn add_in_flight(&self, n: usize) {
        let worker = self
            .app_context
            .worker_registry
            .get_by_url(&self.worker_url)
            .unwrap();
        for _ in 0..n {
            worker.increment_load();
        }
    }

    fn chat_body(stream: bool) -> ChatCompletionRequest {
        serde_json::from_value(json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": stream,
        }))
        .expect("valid chat request")
    }

    async fn chat_request(&self, stream: bool) -> axum::response::Response {
        self.router
            .route_chat(None, &Self::chat_body(stream), None)
            .await
    }

    fn spawn_chat(&self, stream: bool) -> tokio::task::JoinHandle<axum::response::Response> {
        let router = self.router.clone();
        let body = Self::chat_body(stream);
        tokio::spawn(async move { router.route_chat(None, &body, None).await })
    }

    async fn shutdown(mut self) {
        tokio::time::sleep(Duration::from_millis(100)).await;
        for worker in &mut self.workers {
            worker.stop().await;
        }
    }
}

#[tokio::test]
async fn test_dp_ranks_count_one_logical_running_request() {
    let ctx = TestContext::new_with_options(
        MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 500,
            fail_rate: 0.0,
            stream_chunk_delay_ms: 0,
        },
        0,
        4,
        10,
    )
    .await;

    assert_eq!(ctx.app_context.worker_registry.get_all().len(), 4);
    let request = ctx.spawn_chat(false);

    assert_eq!(
        wait_for_running(&ctx, 1, Duration::from_secs(2)).await,
        1,
        "four DP rank representations must still count one logical request"
    );
    assert!(request.await.unwrap().status().is_success());
    assert_eq!(ctx.running_requests(), 0);

    ctx.shutdown().await;
}

#[tokio::test]
async fn test_standard_running_stays_set_across_retry_backoff() {
    let ctx = TestContext::new_with_options(
        MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 1.0,
            stream_chunk_delay_ms: 0,
        },
        2,
        1,
        300,
    )
    .await;

    let request = ctx.spawn_chat(false);

    assert_eq!(wait_for_running(&ctx, 1, Duration::from_secs(1)).await, 1);
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert_eq!(
        ctx.running_requests(),
        1,
        "logical request must remain running between retry attempts"
    );

    assert_eq!(
        request.await.unwrap().status(),
        axum::http::StatusCode::INTERNAL_SERVER_ERROR
    );
    assert_eq!(ctx.running_requests(), 0);

    ctx.shutdown().await;
}

async fn wait_for_load(ctx: &TestContext, expected: usize, timeout: Duration) -> usize {
    let start = std::time::Instant::now();
    let mut last = ctx.worker_load();
    while start.elapsed() < timeout {
        last = ctx.worker_load();
        if last == expected {
            return last;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    last
}

async fn wait_for_running(ctx: &TestContext, expected: u64, timeout: Duration) -> u64 {
    let start = std::time::Instant::now();
    let mut last = ctx.running_requests();
    while start.elapsed() < timeout {
        last = ctx.running_requests();
        if last == expected {
            return last;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    last
}

#[tokio::test]
async fn test_retryable_failure_preserves_baseline_load() {
    // fail_rate = 1.0: every attempt returns 500 (retryable). A baseline of
    // two simulated in-flight requests must survive the failed request: the
    // guard releases exactly its own increment once. The previous manual
    // cleanup double-decremented, silently erasing the baseline.
    let ctx = TestContext::new(
        MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 1.0,
            stream_chunk_delay_ms: 0,
        },
        2,
    )
    .await;

    ctx.add_in_flight(2);
    assert_eq!(ctx.baseline_load(), 2);

    let response = ctx.chat_request(false).await;
    assert_eq!(
        response.status(),
        axum::http::StatusCode::INTERNAL_SERVER_ERROR
    );

    // Exactly one decrement per attempt: baseline is intact.
    assert_eq!(ctx.worker_load(), 2);
    assert_eq!(ctx.running_requests(), 0);

    ctx.shutdown().await;
}

#[tokio::test]
async fn test_retryable_streaming_headers_release_load_before_stalled_body() {
    // A cache-aware streaming attempt that receives 500 headers must be
    // cancelled before retrying, rather than retaining its load guard in a
    // task blocked on the never-ending upstream body.
    let ctx = TestContext::new(
        MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 1.0,
            stream_chunk_delay_ms: 1,
        },
        2,
    )
    .await;

    let baseline = ctx.baseline_load();
    let response = ctx.chat_request(true).await;
    assert_eq!(
        response.status(),
        axum::http::StatusCode::INTERNAL_SERVER_ERROR
    );
    assert!(
        response
            .headers()
            .get(axum::http::header::CONTENT_LENGTH)
            .is_none(),
        "an aborted upstream body must not retain its content-length"
    );

    let load = wait_for_load(&ctx, baseline, Duration::from_millis(250)).await;
    assert_eq!(
        load, baseline,
        "retryable streaming attempts must release load without waiting for the body"
    );

    ctx.shutdown().await;
}

#[tokio::test]
async fn test_streaming_completion_releases_load_exactly_once() {
    // Slow stream with the literal `data: [DONE]` marker embedded in chunk
    // content. The load must stay held for the whole stream and be released
    // exactly once when forwarding ends.
    let ctx = TestContext::new(
        MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
            stream_chunk_delay_ms: 60,
        },
        0,
    )
    .await;

    ctx.add_in_flight(1);
    let baseline = ctx.baseline_load();
    assert_eq!(baseline, 1);

    let response = ctx.chat_request(true).await;
    assert!(response.status().is_success());
    assert_eq!(ctx.running_requests(), 1);

    let mut stream = response.into_body().into_data_stream();
    let mut marker_seen = false;

    while let Some(chunk) = stream.next().await {
        if let Ok(bytes) = chunk {
            if String::from_utf8_lossy(&bytes).contains("data: [DONE]") {
                marker_seen = true;
            }
        }
    }
    assert!(
        marker_seen,
        "stream should have contained the [DONE] marker"
    );

    let load = wait_for_load(&ctx, baseline, Duration::from_secs(3)).await;
    assert_eq!(
        load, baseline,
        "load must return to baseline after stream end"
    );
    assert_eq!(ctx.running_requests(), 0);

    ctx.shutdown().await;
}

#[tokio::test]
async fn test_embedded_done_marker_does_not_release_load_early() {
    // While the stream is still open after a chunk whose content embeds
    // `data: [DONE]`, the request must still be counted as in flight.
    let ctx = TestContext::new(
        MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
            stream_chunk_delay_ms: 100,
        },
        0,
    )
    .await;

    let baseline = ctx.baseline_load();

    let response = ctx.chat_request(true).await;
    assert!(response.status().is_success());

    let mut stream = response.into_body().into_data_stream();

    // Chunk 1 (plain content).
    let _ = stream.next().await.expect("first chunk");
    assert_eq!(ctx.worker_load(), baseline + 1, "load held during stream");
    assert_eq!(
        ctx.running_requests(),
        1,
        "logical request held during stream"
    );

    // Chunk 2 embeds `data: [DONE]` inside message content; the load must
    // still be held afterwards.
    let chunk = stream.next().await.expect("second chunk").unwrap();
    assert!(String::from_utf8_lossy(&chunk).contains("data: [DONE]"));
    assert_eq!(
        ctx.worker_load(),
        baseline + 1,
        "embedded marker must not release the load early"
    );

    // Finish the stream: load returns to baseline exactly once.
    while stream.next().await.is_some() {}
    let load = wait_for_load(&ctx, baseline, Duration::from_secs(3)).await;
    assert_eq!(load, baseline);
    assert_eq!(ctx.running_requests(), 0);

    ctx.shutdown().await;
}

#[tokio::test]
async fn test_client_disconnect_releases_load() {
    // Dropping the response body mid-stream (client disconnect) must release
    // the load guard via the forwarding task teardown.
    let ctx = TestContext::new(
        MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
            stream_chunk_delay_ms: 150,
        },
        0,
    )
    .await;

    let baseline = ctx.baseline_load();

    let response = ctx.chat_request(true).await;
    assert!(response.status().is_success());

    let mut stream = response.into_body().into_data_stream();
    let _ = stream.next().await.expect("first chunk");
    assert_eq!(ctx.worker_load(), baseline + 1);
    assert_eq!(ctx.running_requests(), 1);

    // Simulate the client disconnecting by dropping the body.
    drop(stream);

    let load = wait_for_load(&ctx, baseline, Duration::from_secs(5)).await;
    assert_eq!(
        load, baseline,
        "client disconnect must release the in-flight load"
    );
    assert_eq!(
        wait_for_running(&ctx, 0, Duration::from_secs(1)).await,
        0,
        "client disconnect must release the logical running count"
    );

    ctx.shutdown().await;
}

#[tokio::test]
async fn test_concurrent_requests_track_load_accurately() {
    let ctx = TestContext::new(
        MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 500,
            fail_rate: 0.0,
            stream_chunk_delay_ms: 0,
        },
        0,
    )
    .await;

    let baseline = ctx.baseline_load();
    let concurrency = 8;

    let tasks: Vec<_> = (0..concurrency).map(|_| ctx.spawn_chat(false)).collect();

    // While all requests are in flight, the load must reflect all of them.
    let mut observed_max = baseline;
    let mut observed_running_max = 0;
    let deadline = std::time::Instant::now() + Duration::from_millis(400);
    while std::time::Instant::now() < deadline {
        observed_max = observed_max.max(ctx.worker_load());
        observed_running_max = observed_running_max.max(ctx.running_requests());
        if observed_max >= baseline + concurrency && observed_running_max >= concurrency as u64 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        observed_max,
        baseline + concurrency,
        "all in-flight requests must be counted"
    );
    assert_eq!(
        observed_running_max, concurrency as u64,
        "standard running gauge must count each logical request once"
    );

    for task in tasks {
        assert!(task.await.unwrap().status().is_success());
    }

    // No drift after completion: exactly-once increment and decrement.
    let load = wait_for_load(&ctx, baseline, Duration::from_secs(3)).await;
    assert_eq!(load, baseline, "load must return to baseline without drift");
    assert_eq!(ctx.running_requests(), 0);

    ctx.shutdown().await;
}

#[tokio::test]
async fn test_cancelled_request_releases_standard_running_count() {
    let ctx = TestContext::new(
        MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 5_000,
            fail_rate: 0.0,
            stream_chunk_delay_ms: 0,
        },
        0,
    )
    .await;

    let request = ctx.spawn_chat(false);

    assert_eq!(
        wait_for_running(&ctx, 1, Duration::from_secs(2)).await,
        1,
        "request must be counted after backend assignment"
    );

    request.abort();
    let _ = request.await;
    assert_eq!(
        wait_for_running(&ctx, 0, Duration::from_secs(1)).await,
        0,
        "aborting the request future must release the logical count"
    );

    ctx.shutdown().await;
}
