//! Dynamic batching.
//!
//! Callers submit one input at a time. A single background task groups
//! whatever is waiting into a batch, runs the model once for the whole
//! batch, and hands each caller its own slice of the result.
//!
//! A batch is closed when it reaches `max_batch` items or when its oldest
//! item has waited `max_delay`, whichever comes first. Two properties fall
//! out of that rule:
//!
//! * Under light load a request waits at most `max_delay` for company.
//! * Under heavy load there is no added wait at all. Requests pile up
//!   while the previous batch is running, their deadline has usually
//!   passed by the time the model is free, and the next batch is simply
//!   everything that queued in the meantime. Batch size grows with load
//!   on its own.
//!
//! The queue is bounded. When it is full, new requests are refused
//! immediately instead of being left to wait: a fast "try again" is more
//! useful to a client than an answer that arrives after its timeout.

use crate::metrics::Metrics;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio::time::{timeout_at, Instant};

/// Something that can run a batch of fixed-size inputs.
pub trait Model: Send + Sync + 'static {
    /// Number of `f32` values in one input item.
    fn input_len(&self) -> usize;

    /// Run `batch` items whose values are concatenated in `inputs`.
    /// Returns the outputs concatenated in the same order; every item must
    /// produce the same number of values.
    fn run_batch(&self, inputs: Vec<f32>, batch: usize) -> Result<Vec<f32>, String>;
}

#[derive(Clone, Copy, Debug)]
pub struct BatchConfig {
    /// Largest batch handed to the model.
    pub max_batch: usize,
    /// Longest a request waits for more requests to join its batch.
    pub max_delay: Duration,
    /// Requests allowed to wait before new ones are refused.
    pub queue_capacity: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        BatchConfig {
            max_batch: 32,
            max_delay: Duration::from_millis(2),
            queue_capacity: 256,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum InferError {
    /// Input has the wrong number of values.
    BadInput(String),
    /// The queue is full.
    Overloaded,
    /// The model failed; every request in that batch gets this error.
    Model(String),
    /// The batching task has stopped.
    ShuttingDown,
}

impl std::fmt::Display for InferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferError::BadInput(m) => write!(f, "bad input: {m}"),
            InferError::Overloaded => write!(f, "server overloaded, retry shortly"),
            InferError::Model(m) => write!(f, "model error: {m}"),
            InferError::ShuttingDown => write!(f, "server is shutting down"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Reply {
    pub output: Vec<f32>,
    /// Size of the batch this request ran in.
    pub batch_size: usize,
    /// Time between submission and the start of the batch.
    pub queue_wait: Duration,
}

struct Job {
    input: Vec<f32>,
    enqueued: Instant,
    reply: oneshot::Sender<Result<Reply, InferError>>,
}

/// Handle for submitting requests. Cheap to clone; the batching task stops
/// once every handle has been dropped and the queue has drained.
#[derive(Clone)]
pub struct Batcher {
    queue: mpsc::Sender<Job>,
    input_len: usize,
    metrics: Arc<Metrics>,
}

impl Batcher {
    /// Spawn the batching task on the current tokio runtime.
    pub fn start(model: Arc<dyn Model>, config: BatchConfig, metrics: Arc<Metrics>) -> Batcher {
        assert!(config.max_batch >= 1, "max_batch must be at least 1");
        assert!(config.queue_capacity >= 1, "queue_capacity must be at least 1");
        let (queue, jobs) = mpsc::channel(config.queue_capacity);
        let input_len = model.input_len();
        tokio::spawn(batch_loop(model, config, jobs, metrics.clone()));
        Batcher {
            queue,
            input_len,
            metrics,
        }
    }

    pub fn input_len(&self) -> usize {
        self.input_len
    }

    /// Submit one input and wait for its result.
    pub async fn infer(&self, input: Vec<f32>) -> Result<Reply, InferError> {
        if input.len() != self.input_len {
            self.metrics.requests_invalid.fetch_add(1, Relaxed);
            return Err(InferError::BadInput(format!(
                "expected {} values, got {}",
                self.input_len,
                input.len()
            )));
        }
        let (reply, result) = oneshot::channel();
        let job = Job {
            input,
            enqueued: Instant::now(),
            reply,
        };
        // try_send, not send: a full queue is an immediate refusal.
        match self.queue.try_send(job) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.metrics.requests_rejected.fetch_add(1, Relaxed);
                return Err(InferError::Overloaded);
            }
            Err(mpsc::error::TrySendError::Closed(_)) => return Err(InferError::ShuttingDown),
        }
        self.metrics.queue_depth.fetch_add(1, Relaxed);
        let outcome = result.await.unwrap_or(Err(InferError::ShuttingDown));
        match &outcome {
            Ok(_) => self.metrics.requests_ok.fetch_add(1, Relaxed),
            Err(_) => self.metrics.requests_failed.fetch_add(1, Relaxed),
        };
        outcome
    }
}

async fn batch_loop(
    model: Arc<dyn Model>,
    config: BatchConfig,
    mut jobs: mpsc::Receiver<Job>,
    metrics: Arc<Metrics>,
) {
    // Wait (indefinitely) for the request that opens the next batch.
    while let Some(first) = jobs.recv().await {
        let deadline = first.enqueued + config.max_delay;
        let mut batch = vec![first];
        while batch.len() < config.max_batch {
            // `timeout_at` polls `recv` before it checks the clock, so jobs
            // that are already queued are taken even if the deadline has
            // passed. Only an empty queue makes the deadline matter.
            match timeout_at(deadline, jobs.recv()).await {
                Ok(Some(job)) => batch.push(job),
                Ok(None) | Err(_) => break, // all senders gone, or time is up
            }
        }
        metrics.queue_depth.fetch_sub(batch.len() as u64, Relaxed);

        // A caller that gave up (client disconnected, request timed out)
        // has dropped its receiver. Do not spend model time on it.
        let before = batch.len();
        batch.retain(|job| !job.reply.is_closed());
        metrics.requests_cancelled.fetch_add((before - batch.len()) as u64, Relaxed);
        if batch.is_empty() {
            continue;
        }

        let size = batch.len();
        let started = Instant::now();
        let mut inputs = Vec::with_capacity(size * model.input_len());
        let mut waiting = Vec::with_capacity(size);
        for job in batch {
            inputs.extend_from_slice(&job.input);
            metrics.queue_wait.observe((started - job.enqueued).as_secs_f64());
            waiting.push((job.reply, started - job.enqueued));
        }

        // The model is CPU-bound. Run it on the blocking pool so the async
        // worker threads stay free to accept and answer requests.
        let runner = model.clone();
        let result = tokio::task::spawn_blocking(move || runner.run_batch(inputs, size))
            .await
            .unwrap_or_else(|e| Err(format!("model task panicked: {e}")));
        metrics.observe_batch(size, started.elapsed());

        let result = result.and_then(|outputs| {
            if outputs.is_empty() || outputs.len() % size != 0 {
                Err(format!(
                    "model returned {} values for a batch of {size}",
                    outputs.len()
                ))
            } else {
                Ok(outputs)
            }
        });
        match result {
            Ok(outputs) => {
                let per_item = outputs.len() / size;
                for ((reply, queue_wait), output) in waiting.into_iter().zip(outputs.chunks(per_item)) {
                    // The caller may have left while the model ran; ignore that.
                    let _ = reply.send(Ok(Reply {
                        output: output.to_vec(),
                        batch_size: size,
                        queue_wait,
                    }));
                }
            }
            Err(message) => {
                for (reply, _) in waiting {
                    let _ = reply.send(Err(InferError::Model(message.clone())));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Doubles every value and records the size of each batch it sees.
    /// Optionally sleeps to simulate a slow model, or fails.
    #[derive(Default)]
    struct Recorder {
        batches: Mutex<Vec<usize>>,
        work: Duration,
        fail: bool,
    }

    impl Model for Recorder {
        fn input_len(&self) -> usize {
            2
        }

        fn run_batch(&self, inputs: Vec<f32>, batch: usize) -> Result<Vec<f32>, String> {
            self.batches.lock().unwrap().push(batch);
            std::thread::sleep(self.work);
            if self.fail {
                return Err("boom".into());
            }
            Ok(inputs.iter().map(|v| v * 2.0).collect())
        }
    }

    fn start(model: Arc<Recorder>, config: BatchConfig) -> (Batcher, Arc<Metrics>) {
        let metrics = Arc::new(Metrics::default());
        (Batcher::start(model, config, metrics.clone()), metrics)
    }

    fn config(max_batch: usize, delay_ms: u64, queue_capacity: usize) -> BatchConfig {
        BatchConfig {
            max_batch,
            max_delay: Duration::from_millis(delay_ms),
            queue_capacity,
        }
    }

    #[tokio::test]
    async fn each_caller_gets_its_own_result() {
        let model = Arc::new(Recorder::default());
        let (batcher, _) = start(model.clone(), config(8, 50, 64));
        let calls = (0..8).map(|i| {
            let batcher = batcher.clone();
            tokio::spawn(async move { (i, batcher.infer(vec![i as f32, 100.0 + i as f32]).await) })
        });
        for call in calls.collect::<Vec<_>>() {
            let (i, reply) = call.await.unwrap();
            let reply = reply.unwrap();
            assert_eq!(reply.output, vec![2.0 * i as f32, 200.0 + 2.0 * i as f32]);
            assert_eq!(reply.batch_size, 8);
        }
        // All eight arrived within the delay window: one batch, closed by size.
        assert_eq!(*model.batches.lock().unwrap(), vec![8]);
    }

    #[tokio::test]
    async fn partial_batch_is_flushed_after_max_delay() {
        let model = Arc::new(Recorder::default());
        let (batcher, _) = start(model.clone(), config(8, 20, 64));
        let started = std::time::Instant::now();
        let (a, b, c) = tokio::join!(
            batcher.infer(vec![1.0, 1.0]),
            batcher.infer(vec![2.0, 2.0]),
            batcher.infer(vec![3.0, 3.0]),
        );
        let waited = started.elapsed();
        for reply in [a, b, c] {
            assert_eq!(reply.unwrap().batch_size, 3);
        }
        assert_eq!(*model.batches.lock().unwrap(), vec![3]);
        // It waited for the window to close, and not much longer.
        assert!(waited >= Duration::from_millis(20), "{waited:?}");
        assert!(waited < Duration::from_millis(500), "{waited:?}");
    }

    #[tokio::test]
    async fn never_exceeds_max_batch() {
        let model = Arc::new(Recorder::default());
        let (batcher, metrics) = start(model.clone(), config(4, 30, 64));
        let calls: Vec<_> = (0..10)
            .map(|_| {
                let batcher = batcher.clone();
                tokio::spawn(async move { batcher.infer(vec![0.0, 0.0]).await })
            })
            .collect();
        for call in calls {
            call.await.unwrap().unwrap();
        }
        let batches = model.batches.lock().unwrap().clone();
        assert_eq!(batches.iter().sum::<usize>(), 10);
        assert!(batches.iter().all(|&b| b <= 4), "{batches:?}");
        assert_eq!(metrics.requests_ok.load(Relaxed), 10);
        assert_eq!(metrics.queue_depth.load(Relaxed), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn batches_grow_while_the_model_is_busy() {
        // The model takes 30 ms. With no artificial delay (max_delay = 0),
        // requests that arrive during a run must still be grouped together.
        let model = Arc::new(Recorder {
            work: Duration::from_millis(30),
            ..Default::default()
        });
        let (batcher, _) = start(model.clone(), config(16, 0, 64));
        let first = tokio::spawn({
            let batcher = batcher.clone();
            async move { batcher.infer(vec![0.0, 0.0]).await }
        });
        tokio::time::sleep(Duration::from_millis(10)).await; // first batch is now running
        let rest: Vec<_> = (0..6)
            .map(|_| {
                let batcher = batcher.clone();
                tokio::spawn(async move { batcher.infer(vec![0.0, 0.0]).await })
            })
            .collect();
        assert_eq!(first.await.unwrap().unwrap().batch_size, 1);
        for call in rest {
            assert_eq!(call.await.unwrap().unwrap().batch_size, 6);
        }
        assert_eq!(*model.batches.lock().unwrap(), vec![1, 6]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn full_queue_refuses_immediately() {
        let model = Arc::new(Recorder {
            work: Duration::from_millis(100),
            ..Default::default()
        });
        let (batcher, metrics) = start(model, config(1, 0, 2));
        // One request occupies the model; two more fill the queue.
        let mut accepted = Vec::new();
        for _ in 0..3 {
            let batcher = batcher.clone();
            accepted.push(tokio::spawn(async move { batcher.infer(vec![0.0, 0.0]).await }));
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let started = std::time::Instant::now();
        let refused = batcher.infer(vec![0.0, 0.0]).await;
        assert_eq!(refused, Err(InferError::Overloaded));
        assert!(started.elapsed() < Duration::from_millis(20), "refusal must not wait");
        assert_eq!(metrics.requests_rejected.load(Relaxed), 1);
        for call in accepted {
            call.await.unwrap().unwrap(); // the accepted ones still complete
        }
    }

    #[tokio::test]
    async fn wrong_input_size_is_rejected_before_queueing() {
        let model = Arc::new(Recorder::default());
        let (batcher, metrics) = start(model.clone(), config(4, 1, 4));
        let err = batcher.infer(vec![1.0, 2.0, 3.0]).await.unwrap_err();
        assert!(matches!(err, InferError::BadInput(_)));
        assert!(model.batches.lock().unwrap().is_empty());
        assert_eq!(metrics.requests_invalid.load(Relaxed), 1);
    }

    #[tokio::test]
    async fn model_error_reaches_every_caller_in_the_batch() {
        let model = Arc::new(Recorder {
            fail: true,
            ..Default::default()
        });
        let (batcher, metrics) = start(model, config(4, 20, 16));
        let (a, b) = tokio::join!(batcher.infer(vec![0.0, 0.0]), batcher.infer(vec![0.0, 0.0]));
        assert_eq!(a, Err(InferError::Model("boom".into())));
        assert_eq!(b, Err(InferError::Model("boom".into())));
        assert_eq!(metrics.requests_failed.load(Relaxed), 2);
        // The batcher keeps serving after a failure.
        assert!(batcher.infer(vec![0.0, 0.0]).await.is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_requests_are_not_sent_to_the_model() {
        let model = Arc::new(Recorder {
            work: Duration::from_millis(60),
            ..Default::default()
        });
        let (batcher, metrics) = start(model.clone(), config(8, 0, 16));
        let running = tokio::spawn({
            let batcher = batcher.clone();
            async move { batcher.infer(vec![0.0, 0.0]).await }
        });
        tokio::time::sleep(Duration::from_millis(10)).await;
        // Queue three requests behind the running batch, then abandon two.
        let queued: Vec<_> = (0..3)
            .map(|_| {
                let batcher = batcher.clone();
                tokio::spawn(async move { batcher.infer(vec![0.0, 0.0]).await })
            })
            .collect();
        tokio::time::sleep(Duration::from_millis(10)).await;
        queued[0].abort();
        queued[1].abort();
        running.await.unwrap().unwrap();
        assert_eq!(queued.into_iter().last().unwrap().await.unwrap().unwrap().batch_size, 1);
        assert_eq!(*model.batches.lock().unwrap(), vec![1, 1]);
        assert_eq!(metrics.requests_cancelled.load(Relaxed), 2);
    }

    #[tokio::test]
    async fn loop_stops_when_all_handles_are_dropped() {
        let model = Arc::new(Recorder::default());
        let (batcher, _) = start(model.clone(), config(4, 1, 4));
        batcher.infer(vec![0.0, 0.0]).await.unwrap();
        drop(batcher);
        // The task holds the only other reference to the model; once it
        // exits, ours is the last one.
        for _ in 0..100 {
            if Arc::strong_count(&model) == 1 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("batching task did not stop");
    }
}
