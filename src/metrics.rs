//! Lock-free counters and histograms, rendered in the Prometheus text format.
//!
//! Everything is an atomic integer, so recording a sample on the request
//! path never takes a lock or allocates.

use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Duration;

/// Cumulative histogram with fixed bucket bounds.
pub struct Histogram {
    bounds: &'static [f64],
    /// One count per bound, plus a final one for "larger than every bound".
    counts: Vec<AtomicU64>,
    /// Sum of all samples in millionths, so it fits an integer.
    sum_micros: AtomicU64,
}

impl Histogram {
    pub fn new(bounds: &'static [f64]) -> Self {
        Histogram {
            bounds,
            counts: (0..=bounds.len()).map(|_| AtomicU64::new(0)).collect(),
            sum_micros: AtomicU64::new(0),
        }
    }

    pub fn observe(&self, value: f64) {
        let bucket = self.bounds.partition_point(|&b| b < value);
        self.counts[bucket].fetch_add(1, Relaxed);
        self.sum_micros.fetch_add((value * 1e6).round() as u64, Relaxed);
    }

    pub fn count(&self) -> u64 {
        self.counts.iter().map(|c| c.load(Relaxed)).sum()
    }

    pub fn sum(&self) -> f64 {
        self.sum_micros.load(Relaxed) as f64 / 1e6
    }

    fn render(&self, name: &str, help: &str, out: &mut String) {
        let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} histogram");
        let mut cumulative = 0;
        for (bound, count) in self.bounds.iter().zip(&self.counts) {
            cumulative += count.load(Relaxed);
            let _ = writeln!(out, "{name}_bucket{{le=\"{bound}\"}} {cumulative}");
        }
        cumulative += self.counts[self.bounds.len()].load(Relaxed);
        let _ = writeln!(out, "{name}_bucket{{le=\"+Inf\"}} {cumulative}");
        let _ = writeln!(out, "{name}_sum {}\n{name}_count {cumulative}", self.sum());
    }
}

const SECONDS: &[f64] = &[
    0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
];
const BATCH_SIZES: &[f64] = &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0];

pub struct Metrics {
    pub requests_ok: AtomicU64,
    /// Turned away because the queue was full.
    pub requests_rejected: AtomicU64,
    /// Malformed or wrongly sized input.
    pub requests_invalid: AtomicU64,
    /// The model returned an error.
    pub requests_failed: AtomicU64,
    /// Caller disconnected before its batch ran.
    pub requests_cancelled: AtomicU64,
    pub batches: AtomicU64,
    pub queue_depth: AtomicU64,
    pub batch_size: Histogram,
    pub queue_wait: Histogram,
    pub inference: Histogram,
}

impl Default for Metrics {
    fn default() -> Self {
        Metrics {
            requests_ok: AtomicU64::new(0),
            requests_rejected: AtomicU64::new(0),
            requests_invalid: AtomicU64::new(0),
            requests_failed: AtomicU64::new(0),
            requests_cancelled: AtomicU64::new(0),
            batches: AtomicU64::new(0),
            queue_depth: AtomicU64::new(0),
            batch_size: Histogram::new(BATCH_SIZES),
            queue_wait: Histogram::new(SECONDS),
            inference: Histogram::new(SECONDS),
        }
    }
}

impl Metrics {
    pub fn observe_batch(&self, size: usize, inference: Duration) {
        self.batches.fetch_add(1, Relaxed);
        self.batch_size.observe(size as f64);
        self.inference.observe(inference.as_secs_f64());
    }

    /// Mean number of requests per executed batch so far.
    pub fn mean_batch_size(&self) -> f64 {
        match self.batch_size.count() {
            0 => 0.0,
            n => self.batch_size.sum() / n as f64,
        }
    }

    pub fn render(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "# HELP batchline_requests_total Requests by outcome.\n# TYPE batchline_requests_total counter"
        );
        for (outcome, counter) in [
            ("ok", &self.requests_ok),
            ("rejected", &self.requests_rejected),
            ("invalid", &self.requests_invalid),
            ("failed", &self.requests_failed),
            ("cancelled", &self.requests_cancelled),
        ] {
            let _ = writeln!(
                out,
                "batchline_requests_total{{outcome=\"{outcome}\"}} {}",
                counter.load(Relaxed)
            );
        }
        let _ = writeln!(
            out,
            "# HELP batchline_batches_total Batches executed.\n# TYPE batchline_batches_total counter\n\
             batchline_batches_total {}",
            self.batches.load(Relaxed)
        );
        let _ = writeln!(
            out,
            "# HELP batchline_queue_depth Requests waiting for a batch.\n# TYPE batchline_queue_depth gauge\n\
             batchline_queue_depth {}",
            self.queue_depth.load(Relaxed)
        );
        self.batch_size
            .render("batchline_batch_size", "Requests per executed batch.", &mut out);
        self.queue_wait.render(
            "batchline_queue_wait_seconds",
            "Time a request waited before its batch started.",
            &mut out,
        );
        self.inference.render(
            "batchline_inference_seconds",
            "Model execution time per batch.",
            &mut out,
        );
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_buckets_are_cumulative_and_inclusive() {
        static BOUNDS: &[f64] = &[1.0, 2.0, 4.0];
        let h = Histogram::new(BOUNDS);
        for v in [0.5, 1.0, 1.5, 4.0, 9.0] {
            h.observe(v);
        }
        let mut text = String::new();
        h.render("x", "help", &mut text);
        // le="1" includes the sample equal to 1.
        assert!(text.contains("x_bucket{le=\"1\"} 2"), "{text}");
        assert!(text.contains("x_bucket{le=\"2\"} 3"), "{text}");
        assert!(text.contains("x_bucket{le=\"4\"} 4"), "{text}");
        assert!(text.contains("x_bucket{le=\"+Inf\"} 5"), "{text}");
        assert!(text.contains("x_count 5"), "{text}");
        assert_eq!(h.sum(), 16.0);
    }

    #[test]
    fn render_lists_every_outcome() {
        let m = Metrics::default();
        m.requests_ok.fetch_add(3, Relaxed);
        m.observe_batch(3, Duration::from_millis(2));
        let text = m.render();
        assert!(text.contains("batchline_requests_total{outcome=\"ok\"} 3"));
        assert!(text.contains("batchline_requests_total{outcome=\"rejected\"} 0"));
        assert!(text.contains("batchline_batches_total 1"));
        assert_eq!(m.mean_batch_size(), 3.0);
    }
}
