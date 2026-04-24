//! Per-thread accumulator used by the rayon fold+reduce in every CPU
//! tracker. Owns the concatenated streamline points, the per-streamline
//! offsets, two reusable branch scratch buffers (forward / backward), a
//! `DirectionGetter::Scratch` hook for algorithm-specific per-thread state,
//! and rejection counters.
//!
//! Both Yeh and Dipy-prob used near-identical versions of this struct
//! before the tracking/ extraction. Keeping the underlying allocations
//! alive across attempts is the big allocation win over the
//! one-Vec-per-streamline pattern — a kept streamline is appended
//! directly into `positions`, the boundary pushed to `offsets`.

/// Outcome of a single seed attempt.
#[derive(Clone, Copy, Debug)]
pub enum AttemptOutcome {
    /// Kept. `acc.positions` already contains the streamline; a new
    /// boundary was pushed to `acc.offsets`.
    Kept,
    /// Seed voxel had no viable initial direction (no peak above threshold,
    /// or outside the mask).
    NoInitial,
    /// Streamline entered a `roa_mask` mid-track — whole streamline
    /// discarded. One branch rejecting this way rejects the whole attempt.
    RejectRoa,
    /// Assembled streamline shorter than the plan's `min_len_mm`.
    RejectMinLen,
    /// Streamline did not hit every wired ROI (AND-semantics).
    RejectRoi,
    /// Endpoint-region rule failed.
    RejectEnd,
    /// An endpoint landed in a `no_end_mask`.
    RejectNoEnd,
    /// Post-hoc Hausdorff filter rejected the streamline.
    RejectHausdorff,
    /// Yeh-specific: attempt picked a voxel outside the seed mask or with
    /// no fixels. Kept as a separate counter so the "what rejected my
    /// streamlines" log line is informative.
    SkipEmpty,
}

/// Rejection / kept counters, accumulated per thread and summed during
/// the rayon `reduce`. Each counter maps to one `AttemptOutcome` variant.
#[derive(Default, Clone, Copy, Debug)]
pub struct RejectionCounts {
    pub kept: usize,
    pub no_initial: usize,
    pub skip_empty: usize,
    pub roa: usize,
    pub min_len: usize,
    pub roi: usize,
    pub end: usize,
    pub no_end: usize,
    pub hausdorff: usize,
}

impl RejectionCounts {
    pub fn total_attempts(&self) -> usize {
        self.kept
            + self.no_initial
            + self.skip_empty
            + self.roa
            + self.min_len
            + self.roi
            + self.end
            + self.no_end
            + self.hausdorff
    }

    pub fn merge(&mut self, other: &RejectionCounts) {
        self.kept += other.kept;
        self.no_initial += other.no_initial;
        self.skip_empty += other.skip_empty;
        self.roa += other.roa;
        self.min_len += other.min_len;
        self.roi += other.roi;
        self.end += other.end;
        self.no_end += other.no_end;
        self.hausdorff += other.hausdorff;
    }

    pub fn bump(&mut self, outcome: AttemptOutcome) {
        match outcome {
            AttemptOutcome::Kept => self.kept += 1,
            AttemptOutcome::NoInitial => self.no_initial += 1,
            AttemptOutcome::SkipEmpty => self.skip_empty += 1,
            AttemptOutcome::RejectRoa => self.roa += 1,
            AttemptOutcome::RejectMinLen => self.min_len += 1,
            AttemptOutcome::RejectRoi => self.roi += 1,
            AttemptOutcome::RejectEnd => self.end += 1,
            AttemptOutcome::RejectNoEnd => self.no_end += 1,
            AttemptOutcome::RejectHausdorff => self.hausdorff += 1,
        }
    }
}

/// Per-thread scratch + output buffers. `S` is the `DirectionGetter`'s
/// `Scratch` type (peak candidate list for Yeh, PMF-on-sphere for Dipy).
///
/// Rayon workers keep one of these alive across the chunk of attempts they
/// own. Reusing the underlying allocations (via `Vec::clear`, which preserves
/// capacity) is why a run of 1M attempts doesn't spend its time in the
/// allocator.
///
/// Merge semantics: at the end of the parallel section, `reduce` pairs up
/// partial accumulators. `other`'s positions are appended to `self`, and
/// `other`'s offsets are rebased by the current `positions.len()` — O(#
/// streamlines), not O(# points), so merges stay cheap.
pub struct ThreadAccum<S> {
    /// Concatenated points for every streamline this thread has kept.
    pub positions: Vec<[f32; 3]>,
    /// Offsets into `positions`. TRX/streamline-set convention: N
    /// streamlines → N+1 offsets, with a leading 0.
    pub offsets: Vec<u32>,
    /// Reusable scratch for the forward branch of `track_one_streamline`.
    pub fwd_scratch: Vec<[f32; 3]>,
    /// Reusable scratch for the backward branch of `track_one_streamline`.
    pub bwd_scratch: Vec<[f32; 3]>,
    /// Algorithm-specific scratch (e.g. Yeh's candidate-peak Vec, Dipy's
    /// PMF-on-sphere Vec). Kept inside the accumulator so it's reused
    /// across attempts on the same thread.
    pub dg_scratch: S,
    /// Per-thread rejection + kept counters.
    pub counts: RejectionCounts,
}

impl<S: Default> ThreadAccum<S> {
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            offsets: vec![0u32],
            fwd_scratch: Vec::new(),
            bwd_scratch: Vec::new(),
            dg_scratch: S::default(),
            counts: RejectionCounts::default(),
        }
    }
}

impl<S: Default> Default for ThreadAccum<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> ThreadAccum<S> {
    /// Merge `other` into `self`. Used as the rayon `reduce` step after the
    /// parallel `fold`. Appends `other.positions` then translates its
    /// offsets by the element count we had before the append.
    pub fn merge(mut self, other: ThreadAccum<S>) -> ThreadAccum<S> {
        let base = self.positions.len() as u32;
        self.positions.extend(other.positions);
        // Skip other.offsets[0] (always 0) — our last offset already
        // marks the boundary between the two thread-locals' streamlines.
        for off in other.offsets.into_iter().skip(1) {
            self.offsets.push(base + off);
        }
        self.counts.merge(&other.counts);
        self
    }
}
