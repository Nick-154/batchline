//! End-to-end tests: a real server on a loopback port, the bundled
//! ResNet-14, and results checked against running the model directly.

use batchline::batcher::BatchConfig;
use batchline::client::Connection;
use onyx::{Session, SessionOptions, Tensor};
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

const MODEL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/resnet14.onnx");
const INPUT_LEN: usize = 3 * 32 * 32;

struct TestServer {
    addr: String,
    stop: Option<oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<Result<(), String>>,
}

async fn start(config: BatchConfig) -> TestServer {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let (stop, stopped) = oneshot::channel::<()>();
    let task = tokio::spawn(async move {
        batchline::serve(listener, MODEL, 0, config, async {
            let _ = stopped.await;
        })
        .await
    });
    TestServer {
        addr,
        stop: Some(stop),
        task,
    }
}

fn image(seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(1);
    (0..INPUT_LEN)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn direct(input: &[f32]) -> Vec<f32> {
    let session = Session::from_file(MODEL, SessionOptions::default()).unwrap();
    let x = Tensor::new(vec![1, 3, 32, 32], input.to_vec()).unwrap();
    session.run_one(x).unwrap().into_data()
}

fn close(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_requests_are_batched_and_each_gets_its_own_answer() {
    let server = start(BatchConfig {
        max_batch: 16,
        max_delay: Duration::from_millis(50),
        queue_capacity: 64,
    })
    .await;

    let calls: Vec<_> = (0..12u64)
        .map(|i| {
            let addr = server.addr.clone();
            tokio::spawn(async move {
                let mut conn = Connection::connect(&addr).await.unwrap();
                let response = conn.post_f32("/v1/infer", &image(i)).await.unwrap();
                (i, response)
            })
        })
        .collect();

    let mut largest_batch = 0;
    for call in calls {
        let (i, response) = call.await.unwrap();
        assert_eq!(response.status, 200);
        let batch: usize = response.header("x-batch-size").unwrap().parse().unwrap();
        largest_batch = largest_batch.max(batch);
        // Different inputs give different logits, so a mix-up between
        // requests in the same batch would fail this comparison.
        assert!(close(&response.f32_body(), &direct(&image(i))), "request {i}");
    }
    assert!(largest_batch > 1, "requests sent together should share a batch");
}

#[tokio::test]
async fn json_round_trip_and_keep_alive() {
    let server = start(BatchConfig::default()).await;
    let mut conn = Connection::connect(&server.addr).await.unwrap();
    let input = image(7);
    let json = format!(
        "{{\"input\":[{}]}}",
        input.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",")
    );
    // Two requests on the same connection.
    for _ in 0..2 {
        let response = conn.post_json("/v1/infer", &json).await.unwrap();
        assert_eq!(response.status, 200);
        let body: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        let output: Vec<f32> = body["output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        assert!(close(&output, &direct(&input)));
        assert_eq!(body["batch_size"], 1);
    }
}

#[tokio::test]
async fn bad_requests_get_400_with_a_reason() {
    let server = start(BatchConfig::default()).await;
    let mut conn = Connection::connect(&server.addr).await.unwrap();

    let wrong_size = conn.post_f32("/v1/infer", &[1.0, 2.0, 3.0]).await.unwrap();
    assert_eq!(wrong_size.status, 400);
    assert!(String::from_utf8_lossy(&wrong_size.body).contains("expected 3072 values, got 3"));

    let not_json = conn.post_json("/v1/infer", "{not json").await.unwrap();
    assert_eq!(not_json.status, 400);

    let ragged = conn
        .request("POST", "/v1/infer", Some("application/octet-stream"), &[0u8; 5])
        .await
        .unwrap();
    assert_eq!(ragged.status, 400);
}

#[tokio::test]
async fn model_info_health_and_metrics() {
    let server = start(BatchConfig::default()).await;
    let mut conn = Connection::connect(&server.addr).await.unwrap();

    assert_eq!(conn.get("/healthz").await.unwrap().status, 200);

    let info: serde_json::Value = serde_json::from_slice(&conn.get("/v1/model").await.unwrap().body).unwrap();
    assert_eq!(info["item_shape"], serde_json::json!([3, 32, 32]));
    assert_eq!(info["input_len"], 3072);

    conn.post_f32("/v1/infer", &image(1)).await.unwrap();
    conn.post_f32("/v1/infer", &[0.0]).await.unwrap();
    let metrics = String::from_utf8(conn.get("/metrics").await.unwrap().body).unwrap();
    assert!(
        metrics.contains("batchline_requests_total{outcome=\"ok\"} 1"),
        "{metrics}"
    );
    assert!(
        metrics.contains("batchline_requests_total{outcome=\"invalid\"} 1"),
        "{metrics}"
    );
    assert!(metrics.contains("batchline_batches_total 1"), "{metrics}");
    assert!(
        metrics.contains("batchline_inference_seconds_count 1"),
        "{metrics}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn overload_returns_503_with_retry_after() {
    // One request at a time and room for one more: a burst must be shed.
    let server = start(BatchConfig {
        max_batch: 1,
        max_delay: Duration::ZERO,
        queue_capacity: 1,
    })
    .await;
    let calls: Vec<_> = (0..40u64)
        .map(|i| {
            let addr = server.addr.clone();
            tokio::spawn(async move {
                let mut conn = Connection::connect(&addr).await.unwrap();
                conn.post_f32("/v1/infer", &image(i)).await.unwrap()
            })
        })
        .collect();
    let (mut ok, mut shed) = (0, 0);
    for call in calls {
        let response = call.await.unwrap();
        match response.status {
            200 => ok += 1,
            503 => {
                assert_eq!(response.header("retry-after"), Some("1"));
                shed += 1;
            }
            other => panic!("unexpected status {other}"),
        }
    }
    assert!(ok >= 1 && shed >= 1, "ok={ok} shed={shed}");
    assert_eq!(ok + shed, 40);
}

#[tokio::test]
async fn graceful_shutdown_stops_the_server() {
    let mut server = start(BatchConfig::default()).await;
    let mut conn = Connection::connect(&server.addr).await.unwrap();
    assert_eq!(conn.post_f32("/v1/infer", &image(3)).await.unwrap().status, 200);
    drop(conn);
    server.stop.take().unwrap().send(()).unwrap();
    let result = tokio::time::timeout(Duration::from_secs(5), server.task).await;
    assert!(matches!(result, Ok(Ok(Ok(())))), "server did not stop cleanly");
}
