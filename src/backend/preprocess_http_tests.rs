//! Regression for early disabled-render responses during large HTTP uploads.

#![cfg(all(test, feature = "cpu", feature = "ready", target_os = "linux"))]

use super::prepare_http_token_inputs;
use crate::core::{BasicWorker, Worker as BackendWorker, WorkerType};
use serde_json::{json, Value};
use std::{
    os::fd::AsRawFd,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    task::{JoinHandle, JoinSet},
};

const DISABLED: &str = r#"{"error":{"message":"Scale-out endpoints are disabled. Restart vllm serve with --enable-scale-out to enable them.","type":"PermissionDeniedError","code":403}}"#;

struct Worker {
    url: String,
    port: u16,
    render_lengths: Arc<Mutex<Vec<usize>>>,
    tokenize_inputs: Arc<Mutex<Vec<Value>>>,
    server: JoinHandle<()>,
    instance: Arc<dyn BackendWorker>,
}

impl Worker {
    async fn start() -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        // A large upload must encounter backpressure before the early response.
        let receive_buffer: libc::c_int = 65536;
        assert_eq!(
            unsafe {
                libc::setsockopt(
                    listener.as_raw_fd(),
                    libc::SOL_SOCKET,
                    libc::SO_RCVBUF,
                    &receive_buffer as *const _ as *const _,
                    std::mem::size_of_val(&receive_buffer) as libc::socklen_t,
                )
            },
            0
        );
        let render_lengths = Arc::new(Mutex::new(Vec::new()));
        let tokenize_inputs = Arc::new(Mutex::new(Vec::new()));
        let lengths = render_lengths.clone();
        let inputs = tokenize_inputs.clone();
        let server = tokio::spawn(async move {
            let mut connections = JoinSet::new();
            loop {
                tokio::select! {
                    accepted = listener.accept() => {
                        let (stream, _) = accepted.unwrap();
                        let lengths = lengths.clone();
                        let inputs = inputs.clone();
                        connections.spawn(async move {
                            let mut stream = BufReader::new(stream);
                            loop {
                                let mut line = String::new();
                                if stream.read_line(&mut line).await.unwrap_or(0) == 0 {
                                    return;
                                }
                                let path = line.split_whitespace().nth(1).unwrap().to_owned();
                                let mut length = 0;
                                loop {
                                    line.clear();
                                    if stream.read_line(&mut line).await.unwrap_or(0) == 0 {
                                        return;
                                    }
                                    if line == "\r\n" {
                                        break;
                                    }
                                    if let Some((name, value)) = line.split_once(':') {
                                        if name.eq_ignore_ascii_case("content-length") {
                                            length = value.trim().parse::<usize>().unwrap();
                                        }
                                    }
                                }
                                let is_render = path.ends_with("/render");
                                if is_render {
                                    lengths.lock().unwrap().push(length);
                                    let reply = format!(
                                        "HTTP/1.1 403 Forbidden\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                                        DISABLED.len(), DISABLED
                                    );
                                    if stream.get_mut().write_all(reply.as_bytes()).await.is_err() {
                                        return;
                                    }
                                    // Match the original reproducer: respond before reading,
                                    // then drain the body after the client has seen the 403.
                                    tokio::time::sleep(Duration::from_millis(50)).await;
                                }
                                let mut body = vec![0; length];
                                if stream.read_exact(&mut body).await.is_err() {
                                    return;
                                }
                                if !is_render {
                                    assert_eq!(path, "/tokenize");
                                    inputs.lock().unwrap().push(serde_json::from_slice(&body).unwrap());
                                    let body = r#"{"tokens":[101,102],"count":2,"max_model_len":100}"#;
                                    let reply = format!(
                                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                                        body.len(), body
                                    );
                                    if stream.get_mut().write_all(reply.as_bytes()).await.is_err() {
                                        return;
                                    }
                                }
                            }
                        });
                    }
                    Some(result) = connections.join_next(), if !connections.is_empty() => {
                        result.unwrap();
                    }
                }
            }
        });
        Self {
            url: format!("http://{addr}"),
            port: addr.port(),
            render_lengths,
            tokenize_inputs,
            server,
            instance: Arc::new(BasicWorker::new(
                format!("http://{addr}"),
                WorkerType::Regular,
            )),
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.server.abort();
    }
}

fn client_fds(port: u16) -> usize {
    let tcp = std::fs::read_to_string("/proc/self/net/tcp").unwrap();
    let sockets: std::collections::HashSet<_> = tcp
        .lines()
        .skip(1)
        .filter_map(|line| {
            let fields: Vec<_> = line.split_whitespace().collect();
            let remote_port = fields[2].rsplit(':').next()?;
            (u16::from_str_radix(remote_port, 16).ok() == Some(port))
                .then(|| format!("socket:[{}]", fields[9]))
        })
        .collect();
    std::fs::read_dir("/proc/self/fd")
        .unwrap()
        .filter_map(|entry| std::fs::read_link(entry.ok()?.path()).ok())
        .filter(|target| sockets.contains(target.to_str().unwrap_or_default()))
        .count()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn disabled_render_does_not_upload_large_prompts_or_accumulate_connections() {
    let worker = Worker::start().await;
    let client = reqwest::Client::builder()
        .no_proxy()
        .pool_idle_timeout(Duration::from_millis(100))
        .build()
        .unwrap();
    let body =
        json!({"model":"m","messages":[{"role":"user","content":"x".repeat(4 * 1024 * 1024)}]});
    for _ in 0..6 {
        let inputs = prepare_http_token_inputs(
            &client,
            vec![worker.instance.clone()],
            &body,
            "/v1/chat/completions",
            None,
            None,
            Duration::from_secs(3),
        )
        .await;
        let input = inputs[&worker.url].as_ref().unwrap();
        assert_eq!(input.source, "tokenize");
        assert_eq!(input.prompts.as_ref(), &[vec![101, 102]]);
    }
    let released = tokio::time::timeout(Duration::from_secs(1), async {
        while client_fds(worker.port) != 0 {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await;
    assert!(
        released.is_ok(),
        "disabled render retained {} client socket FDs",
        client_fds(worker.port)
    );
    assert_eq!(*worker.render_lengths.lock().unwrap(), vec![2; 1]);
    let inputs = worker.tokenize_inputs.lock().unwrap();
    assert_eq!(inputs.len(), 6);
    assert!(inputs.iter().all(|input| *input == body));
}
