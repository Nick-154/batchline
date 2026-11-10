//! HTTP front end.
//!
//! | Route | |
//! |---|---|
//! | `POST /v1/infer` | Run one input. JSON `{"input": [..]}` or raw little-endian `f32` bytes. |
//! | `GET /v1/model` | Input shape and batching settings. |
//! | `GET /metrics` | Prometheus metrics. |
//! | `GET /healthz` | Liveness check. |

use crate::batcher::{BatchConfig, Batcher, InferError};
use crate::metrics::Metrics;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::header::{ACCEPT, CONTENT_TYPE, RETRY_AFTER};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use serde::Deserialize;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;

const BINARY: &str = "application/octet-stream";
const JSON: &str = "application/json";

#[derive(Clone)]
pub struct AppState {
    pub batcher: Batcher,
    pub metrics: Arc<Metrics>,
    pub item_shape: Arc<Vec<usize>>,
    pub config: BatchConfig,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/infer", post(infer))
        .route("/v1/model", get(model_info))
        .route("/metrics", get(metrics))
        .route("/healthz", get(|| async { "ok\n" }))
        .with_state(state)
}

#[derive(Deserialize)]
struct InferRequest {
    input: Vec<f32>,
}

fn has(headers: &HeaderMap, name: axum::http::HeaderName, value: &str) -> bool {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with(value))
}

fn error(status: StatusCode, message: impl std::fmt::Display) -> Response {
    let body = serde_json::json!({ "error": message.to_string() }).to_string();
    (status, [(CONTENT_TYPE, JSON)], body).into_response()
}

async fn infer(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    let binary_in = has(&headers, CONTENT_TYPE, BINARY);
    let input = if binary_in {
        if !body.len().is_multiple_of(4) {
            state.metrics.requests_invalid.fetch_add(1, Relaxed);
            return error(
                StatusCode::BAD_REQUEST,
                "binary body must be a whole number of f32 values",
            );
        }
        body.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    } else {
        match serde_json::from_slice::<InferRequest>(&body) {
            Ok(request) => request.input,
            Err(e) => {
                state.metrics.requests_invalid.fetch_add(1, Relaxed);
                return error(StatusCode::BAD_REQUEST, format!("invalid JSON body: {e}"));
            }
        }
    };

    let reply = match state.batcher.infer(input).await {
        Ok(reply) => reply,
        Err(e @ InferError::BadInput(_)) => return error(StatusCode::BAD_REQUEST, e),
        Err(e @ (InferError::Overloaded | InferError::ShuttingDown)) => {
            // 503 with Retry-After tells well-behaved clients to back off.
            let mut response = error(StatusCode::SERVICE_UNAVAILABLE, e);
            response
                .headers_mut()
                .insert(RETRY_AFTER, HeaderValue::from_static("1"));
            return response;
        }
        Err(e @ InferError::Model(_)) => return error(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let queue_ms = reply.queue_wait.as_secs_f64() * 1e3;
    // Answer in the format the request used unless the client asks otherwise.
    let binary_out = has(&headers, ACCEPT, BINARY) || (binary_in && !has(&headers, ACCEPT, JSON));
    let mut response = if binary_out {
        let bytes: Vec<u8> = reply.output.iter().flat_map(|v| v.to_le_bytes()).collect();
        ([(CONTENT_TYPE, BINARY)], bytes).into_response()
    } else {
        let body = serde_json::json!({
            "output": reply.output,
            "batch_size": reply.batch_size,
            "queue_ms": queue_ms,
        });
        ([(CONTENT_TYPE, JSON)], body.to_string()).into_response()
    };
    response
        .headers_mut()
        .insert("x-batch-size", HeaderValue::from(reply.batch_size as u64));
    response
}

async fn model_info(State(state): State<AppState>) -> Response {
    let body = serde_json::json!({
        "item_shape": *state.item_shape,
        "input_len": state.batcher.input_len(),
        "max_batch": state.config.max_batch,
        "max_delay_ms": state.config.max_delay.as_secs_f64() * 1e3,
        "queue_capacity": state.config.queue_capacity,
    });
    ([(CONTENT_TYPE, JSON)], body.to_string()).into_response()
}

async fn metrics(State(state): State<AppState>) -> Response {
    (
        [(CONTENT_TYPE, "text/plain; version=0.0.4")],
        state.metrics.render(),
    )
        .into_response()
}
