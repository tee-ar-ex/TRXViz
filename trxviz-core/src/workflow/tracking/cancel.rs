//! Cooperative job control — cancel flag + progress callback — threaded
//! from the GUI's spinner overlay through to the tracker loops.
//!
//! Wraps an `Arc<Inner>` holding (a) an `AtomicBool` for cancellation and
//! (b) an optional progress callback. The GUI keeps one `CancelFlag` per
//! in-flight job keyed by node UUID; clicking Cancel calls
//! `request_cancel()`, which the worker thread observes on its next
//! `is_cancelled()` poll and returns `WorkflowError::Cancelled`. The GUI
//! also installs a progress callback at dispatch time so the worker can
//! emit `WorkflowJobMessage::Progress` every ~1024 attempts without
//! knowing about the GUI's message channel directly.
//!
//! Why bundle the two concerns: they always travel together (same
//! lifetime, same node, same thread), and keeping them in one type
//! means the tracker's signature stays `&CancelFlag` everywhere.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

type ProgressCallback = Box<dyn Fn(u64, u64) + Send + Sync + 'static>;

struct Inner {
    cancelled: AtomicBool,
    /// Optional progress callback. `None` for headless / CLI callers
    /// that don't need progress reporting. The GUI wraps its job-
    /// message channel send in this callback so progress flows
    /// through the same mpsc that carries Started / Finished.
    on_progress: Option<ProgressCallback>,
}

/// Cooperative cancellation + progress handle shared between a GUI
/// thread and a worker thread. Cheap to clone (`Arc` bump); `Default`
/// constructs a flag in the "not cancelled, no progress reporting"
/// state.
#[derive(Clone)]
pub struct CancelFlag {
    inner: Arc<Inner>,
}

impl CancelFlag {
    /// Construct a fresh flag with no progress callback. Appropriate
    /// for headless / CLI / test callers where the tracker's progress
    /// output has nowhere to go.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                cancelled: AtomicBool::new(false),
                on_progress: None,
            }),
        }
    }

    /// Construct a flag that forwards progress reports through the
    /// given callback. The GUI plugs its channel-send here so the
    /// worker doesn't need to know about `WorkflowJobMessage` types.
    pub fn with_progress_callback<F>(callback: F) -> Self
    where
        F: Fn(u64, u64) + Send + Sync + 'static,
    {
        Self {
            inner: Arc::new(Inner {
                cancelled: AtomicBool::new(false),
                on_progress: Some(Box::new(callback)),
            }),
        }
    }

    /// True once someone has called `request_cancel()` on any clone
    /// of this flag. Cheap (single relaxed atomic load) — safe to call
    /// in a hot loop.
    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::Relaxed)
    }

    /// Signal cancellation. Any worker polling `is_cancelled()` on
    /// this flag (or any of its clones) will see `true` on its next
    /// poll and should wind down gracefully.
    pub fn request_cancel(&self) {
        self.inner.cancelled.store(true, Ordering::Relaxed);
    }

    /// Fire the progress callback (if one is installed). No-op when
    /// constructed via `CancelFlag::new()`. `done` and `total` are
    /// in whatever unit the tracker chooses (attempts, seeds, batches);
    /// the GUI renders `done / total` as a progress bar.
    pub fn report_progress(&self, done: u64, total: u64) {
        if let Some(callback) = &self.inner.on_progress {
            callback(done, total);
        }
    }
}

impl Default for CancelFlag {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for CancelFlag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancelFlag")
            .field("cancelled", &self.is_cancelled())
            .field("has_progress_callback", &self.inner.on_progress.is_some())
            .finish()
    }
}
