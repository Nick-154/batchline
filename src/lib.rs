//! batchline: a dynamic-batching inference server.

pub mod batcher;
pub mod client;
pub mod metrics;
pub mod model;
pub mod server;

use batcher::{BatchConfig, Batcher};
use metrics::Metrics;
use model::OnnxModel;
use std::future::Future;
use std::sync::Arc;
use tokio::net::TcpListener;

/// Load `model_path`, start the batcher, and serve on `listener` until
/// `shutdown` resolves. In-flight requests are allowed to finish.
pub async fn serve(
    listener: TcpListener,
    model_path: &str,
    threads: usize,
    config: BatchConfig,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), String> {
    let model = Arc::new(OnnxModel::load(model_path, threads)?);
    let item_shape = Arc::new(model.item_shape().to_vec());
    let metrics = Arc::new(Metrics::default());
    let batcher = Batcher::start(model, config, metrics.clone());
    let app = server::router(server::AppState {
        batcher,
        metrics,
        item_shape,
        config,
    });
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
        .map_err(|e| e.to_string())
}
