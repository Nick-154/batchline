# batchline

A dynamic-batching inference server in Rust. Clients send one input per
HTTP request; the server groups requests that arrive close together into
a single batch, runs the model once, and returns each client its own
result. It serves ONNX models through [onyx](https://github.com/Nick-154/onyx),
a small inference runtime, and is built on tokio and axum.

```sh
cargo run --release -- --model models/resnet14.onnx
curl -s localhost:8080/v1/model
# {"input_len":3072,"item_shape":[3,32,32],"max_batch":32,"max_delay_ms":0.0,"queue_capacity":256}
```

## Why batch

A model run has a fixed cost per call and can use several cores (or a
whole GPU) at once. Handling requests one at a time leaves that capacity
idle. Batching trades a little coordination for throughput, and this is
the core idea of production servers such as NVIDIA Triton.

## How it works

```
 HTTP handlers ──try_send──▶ bounded queue ──▶ batch loop ──spawn_blocking──▶ model
      ▲                          │ full?                │
      └────── oneshot reply ─────┘ 503 + Retry-After    └─ split output per request
```

- **One batching task** (`src/batcher.rs`) owns the model. It takes the
  oldest queued request, then keeps adding requests until the batch has
  `max_batch` items or the oldest one has waited `max_delay`.
- **Batch size follows load.** While a batch is running, new requests
  queue up. When the model is free, everything that queued is taken at
  once, with no extra wait. With the default `max_delay` of 0 this is the
  only way batches form: an idle server answers a lone request
  immediately, and a busy one batches automatically.
- **Backpressure.** The queue is bounded. When it is full the server
  answers `503` with `Retry-After` straight away instead of letting
  latency grow without limit.
- **Cancelled work is skipped.** If a client disconnects while its
  request is queued, the request is dropped before the model runs.
- **The async runtime never blocks.** The model is CPU-bound, so each
  batch runs on tokio's blocking pool and handlers stay responsive.
- **Graceful shutdown.** On SIGINT or SIGTERM the server stops accepting
  connections and finishes in-flight requests.

## API

| Route | |
|---|---|
| `POST /v1/infer` | One input, as JSON `{"input": [...]}` or as raw little-endian `f32` bytes (`Content-Type: application/octet-stream`). The reply uses the same format and carries an `x-batch-size` header. |
| `GET /v1/model` | Input shape and batching settings. |
| `GET /metrics` | Prometheus text format: requests by outcome, batches, queue depth, and histograms of batch size, queue wait and inference time. |
| `GET /healthz` | Liveness. |

Errors are JSON `{"error": "..."}`: `400` for malformed or wrongly sized
input, `503` when the queue is full, `500` if the model fails.

## Options

| Flag | Default | |
|---|---|---|
| `--model` | (required) | Path to the `.onnx` file. One input with a dynamic batch dimension. |
| `--addr` | `127.0.0.1:8080` | Listen address. |
| `--max-batch` | `32` | Largest batch. `1` disables batching. |
| `--max-delay-ms` | `0` | Longest a request waits for others to join its batch. |
| `--queue-capacity` | `256` | Waiting requests allowed before new ones get `503`. |
| `--threads` | `0` | Model threads; `0` uses every core. |

## Results

`loadgen` (in this repository) runs N clients that each hold one
connection and send requests back to back. Model: the bundled ResNet-14
(32x32 input, raw `f32` bodies). Server and load generator ran on the
same 2-vCPU Intel Xeon @ 2.10 GHz cloud VM, 6 seconds per row after a
1-second warm-up.

**Throughput, requests/s**

| Clients | No batching (`--max-batch 1`) | Batching, delay 0 | Batching, delay 2 ms |
|---:|---:|---:|---:|
| 1 | 454 | 498 | 198 |
| 4 | 482 | 700 | 499 |
| 16 | 524 | 961 | 776 |
| 64 | 573 | 992 | 965 |

**p99 latency, ms**

| Clients | No batching | Batching, delay 0 | Batching, delay 2 ms |
|---:|---:|---:|---:|
| 1 | 3.8 | 3.4 | 6.9 |
| 4 | 13.0 | 11.1 | 11.3 |
| 16 | 45.1 | 27.0 | 28.0 |
| 64 | 149.9 | 86.7 | 87.1 |

At 64 clients batching gives 1.7x the throughput (573 to 992 requests/s)
and 1.7x lower p99 latency (150 ms to 87 ms). Mean batch size rises with
load on its own: 1.0, 2.5, 10.3 and 32.0 for the four rows.

Things to know when reading these numbers:

- **Where the gain comes from.** onyx runs a single image on one thread
  and spreads a batch across cores. So on this 2-core machine most of
  the gain is that a batch lets the one model executor use both cores.
  Running several unbatched model instances side by side would recover
  part of it; batching gets there with one copy of the model. On a GPU,
  where one call costs about the same for 1 or 32 inputs, the gain from
  batching is much larger.
- **A fixed delay costs latency at low load** (the 2 ms column at 1 and 4
  clients) and buys nothing here, which is why the default is 0.
- **The load generator shares the CPU with the server**, so absolute
  numbers are lower than the server alone could reach.
- **Closed-loop load** cannot show unbounded queue growth, because each
  client waits for its answer before sending again. Load shedding is
  covered by tests rather than by a benchmark.

Reproduce:

```sh
cargo build --release
./target/release/batchline --model models/resnet14.onnx --max-batch 32 &
./target/release/loadgen --concurrency 1,4,16,64 --seconds 6
```

## Tests

`cargo test` runs 17 tests.

- Batcher (`src/batcher.rs`), against a mock model that records batch
  sizes: results reach the right caller, a partial batch is flushed after
  `max_delay`, `max_batch` is never exceeded, batches grow while the
  model is busy, a full queue refuses immediately, a model error reaches
  every caller in the batch, cancelled requests never reach the model,
  and the task stops when its handles are dropped.
- End to end (`tests/http.rs`): a real server on a loopback port with
  ResNet-14. Twelve concurrent requests with different inputs are each
  checked against a direct model run, so a mix-up inside a batch would
  be caught. Also JSON and keep-alive, `400` and `503` paths, metrics,
  and graceful shutdown.

## Docker

```sh
docker build -t batchline .
docker run --rm -p 8080:8080 batchline
```

## Limits and next steps

- One model, one executor. Several model instances behind the same queue
  would be the next step for many-core machines.
- An open-loop mode for the load generator (fixed arrival rate) to
  measure latency under overload with and without shedding.
- Per-request deadlines, so a request that has waited too long is
  dropped rather than run.
- HTTP/1.1 only.
