//! Closed-loop load generator.
//!
//! Each of N clients keeps one connection open and sends requests back to
//! back for a fixed time, so N is the number of requests in flight. For
//! every concurrency level it reports throughput, latency percentiles, the
//! mean batch size the server reported, and how many requests were shed.

use batchline::client::Connection;
use clap::Parser;
use std::time::{Duration, Instant};

/// Load generator for batchline.
#[derive(Parser, Debug)]
struct Args {
    /// Server address.
    #[arg(long, default_value = "127.0.0.1:8080")]
    addr: String,
    /// Concurrency levels to run, comma separated.
    #[arg(long, default_value = "1,4,16,64", value_delimiter = ',')]
    concurrency: Vec<usize>,
    /// Measured seconds per level.
    #[arg(long, default_value_t = 5.0)]
    seconds: f64,
    /// Unmeasured warm-up seconds per level.
    #[arg(long, default_value_t = 1.0)]
    warmup: f64,
}

#[derive(Default)]
struct ClientStats {
    latencies: Vec<Duration>,
    batch_sum: u64,
    shed: u64,
    errors: u64,
}

fn input(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(1);
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

async fn client(addr: String, values: Vec<f32>, measure_from: Instant, stop_at: Instant) -> ClientStats {
    let mut stats = ClientStats::default();
    let Ok(mut conn) = Connection::connect(&addr).await else {
        stats.errors += 1;
        return stats;
    };
    while Instant::now() < stop_at {
        let sent = Instant::now();
        let response = conn.post_f32("/v1/infer", &values).await;
        let measured = sent >= measure_from;
        match response {
            Ok(r) if r.status == 200 => {
                if measured {
                    stats.latencies.push(sent.elapsed());
                    stats.batch_sum += r.header("x-batch-size").and_then(|v| v.parse().ok()).unwrap_or(1);
                }
            }
            Ok(r) if r.status == 503 => {
                if measured {
                    stats.shed += 1;
                }
                // Back off briefly, as the Retry-After header asks.
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
            _ => {
                stats.errors += 1;
                return stats;
            }
        }
    }
    stats
}

fn percentile(sorted: &[Duration], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = ((sorted.len() as f64 * p).ceil() as usize).clamp(1, sorted.len());
    sorted[rank - 1].as_secs_f64() * 1e3
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let mut probe = match Connection::connect(&args.addr).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: cannot connect to {}: {e}", args.addr);
            std::process::exit(1);
        }
    };
    let info = probe.get("/v1/model").await.expect("GET /v1/model");
    let info: serde_json::Value = serde_json::from_slice(&info.body).expect("model info is JSON");
    let input_len = info["input_len"].as_u64().expect("input_len") as usize;
    println!(
        "server {}: max_batch {}, max_delay {} ms, queue {}",
        args.addr, info["max_batch"], info["max_delay_ms"], info["queue_capacity"]
    );
    println!(
        "{:>8}{:>10}{:>10}{:>10}{:>10}{:>12}{:>8}",
        "clients", "req/s", "p50 ms", "p95 ms", "p99 ms", "mean batch", "shed"
    );

    for &clients in &args.concurrency {
        let start = Instant::now();
        let measure_from = start + Duration::from_secs_f64(args.warmup);
        let stop_at = measure_from + Duration::from_secs_f64(args.seconds);
        let tasks: Vec<_> = (0..clients)
            .map(|i| {
                tokio::spawn(client(
                    args.addr.clone(),
                    input(input_len, i as u64),
                    measure_from,
                    stop_at,
                ))
            })
            .collect();

        let mut all = ClientStats::default();
        for task in tasks {
            let stats = task.await.expect("client task");
            all.latencies.extend(stats.latencies);
            all.batch_sum += stats.batch_sum;
            all.shed += stats.shed;
            all.errors += stats.errors;
        }
        all.latencies.sort_unstable();
        let done = all.latencies.len();
        println!(
            "{clients:>8}{:>10.0}{:>10.2}{:>10.2}{:>10.2}{:>12.1}{:>8}",
            done as f64 / args.seconds,
            percentile(&all.latencies, 0.50),
            percentile(&all.latencies, 0.95),
            percentile(&all.latencies, 0.99),
            all.batch_sum as f64 / done.max(1) as f64,
            all.shed,
        );
        if all.errors > 0 {
            eprintln!(
                "  {} client(s) stopped on a connection or protocol error",
                all.errors
            );
        }
    }
}
