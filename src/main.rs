use batchline::batcher::BatchConfig;
use clap::Parser;
use std::time::Duration;
use tokio::net::TcpListener;

/// Dynamic-batching inference server for ONNX models.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Path to the .onnx model.
    #[arg(long)]
    model: String,
    /// Address to listen on.
    #[arg(long, default_value = "127.0.0.1:8080")]
    addr: String,
    /// Largest batch handed to the model. 1 disables batching.
    #[arg(long, default_value_t = 32)]
    max_batch: usize,
    /// Longest a request waits for others to join its batch, in milliseconds.
    #[arg(long, default_value_t = 2.0)]
    max_delay_ms: f64,
    /// Requests allowed to wait before new ones get 503.
    #[arg(long, default_value_t = 256)]
    queue_capacity: usize,
    /// Model threads. 0 uses every core.
    #[arg(long, default_value_t = 0)]
    threads: usize,
}

async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut term = signal(SignalKind::terminate()).expect("install SIGTERM handler");
        tokio::select! {
            _ = ctrl_c => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    let _ = ctrl_c.await;
    eprintln!("shutting down: finishing in-flight requests");
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    if args.max_batch == 0 || args.queue_capacity == 0 || args.max_delay_ms < 0.0 {
        eprintln!("error: max-batch and queue-capacity must be at least 1, max-delay-ms at least 0");
        std::process::exit(2);
    }
    let config = BatchConfig {
        max_batch: args.max_batch,
        max_delay: Duration::from_secs_f64(args.max_delay_ms / 1e3),
        queue_capacity: args.queue_capacity,
    };
    let listener = match TcpListener::bind(&args.addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("error: cannot listen on {}: {e}", args.addr);
            std::process::exit(1);
        }
    };
    eprintln!(
        "batchline listening on {} (model {}, max batch {}, max delay {} ms, queue {})",
        args.addr, args.model, config.max_batch, args.max_delay_ms, config.queue_capacity
    );
    if let Err(e) = batchline::serve(listener, &args.model, args.threads, config, shutdown_signal()).await {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
