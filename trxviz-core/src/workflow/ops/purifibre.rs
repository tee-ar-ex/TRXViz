//! Purifibre op — FOD-coherence streamline cleanup.
//!
//! Consumes streamlines on input 0 and a prebuilt boundary field (the
//! sTODI) on input 1, scores each selected streamline, and emits two
//! outputs:
//!
//! - Output 0: **scored passthrough** — the full input selection with
//!   a `"fico"` DPS field attached. Wire a `ColorByDps("fico")` node
//!   to visualize the score distribution without any filtering.
//! - Output 1: **filtered survivors** — only streamlines whose FICO
//!   score was at or above the `puri_fraction` percentile.
//!
//! Both outputs share the same underlying `StreamlineDataset` (with
//! the FICO DPS baked in); they differ only in `selected_streamlines`.
//! See `crate::workflow::purifibre` for the scoring algorithm.

use std::borrow::Cow;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use super::super::{
    EvalCtx, EvaluatedValue, PortKind, WorkflowNodeKind, WorkflowOp, WorkflowValue,
    expect_boundary_field_input, expect_streamline_input, mark_expensive_success,
    prime_expensive_record, sync_node_state_from_run_record,
};
use crate::workflow::methods::OpCategory;
use crate::data::trx_data::TrxGpuData;
use crate::workflow::purifibre::{PurifibreParams, apply_puri_threshold, purifibre_score};
use crate::workflow::types::{CachedPurifibre, StreamlineDataset, StreamlineFlow};

/// DPS field name written onto the output streamlines. Wire
/// `ColorByDps("fico")` downstream to color by the score.
const FICO_DPS_NAME: &str = "fico";

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PurifibreOp {
    #[serde(default = "default_trim_fraction")]
    pub trim_fraction: f32,
    #[serde(default = "default_puri_fraction")]
    pub puri_fraction: f32,
    #[serde(default = "default_spherical_smoothing_deg")]
    pub spherical_smoothing_deg: f32,
}

// Defaults mirror nibrary's `purifibre.h` line-for-line (10% trim,
// 10% discard, 15° spherical Gaussian).
fn default_trim_fraction() -> f32 {
    0.10
}
fn default_puri_fraction() -> f32 {
    0.10
}
fn default_spherical_smoothing_deg() -> f32 {
    15.0
}

impl Default for PurifibreOp {
    fn default() -> Self {
        Self {
            trim_fraction: default_trim_fraction(),
            puri_fraction: default_puri_fraction(),
            spherical_smoothing_deg: default_spherical_smoothing_deg(),
        }
    }
}

impl WorkflowOp for PurifibreOp {
    fn tag(&self) -> &'static str {
        "purifibre"
    }

    fn title(&self) -> &'static str {
        "Purifibre"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline, PortKind::BoundaryField]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        // Output 0: scored passthrough (all input streamlines + FICO DPS).
        // Output 1: filtered survivors only.
        &[PortKind::Streamline, PortKind::Streamline]
    }

    fn category(&self) -> OpCategory {
        OpCategory::StreamlineFilter
    }

    fn citation_keys(&self) -> &'static [&'static str] {
        &["purifibre", "nibrary"]
    }

    fn boilerplate(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Owned(format!(
            "Streamlines were cleaned with Purifibre [@purifibre;@nibrary] \
             (FOD-coherence-based filtering) using a {:.0}% trim fraction, \
             a {:.0}% discard fraction, and a {:.1}° spherical Gaussian \
             smoother.",
            self.trim_fraction * 100.0,
            self.puri_fraction * 100.0,
            self.spherical_smoothing_deg,
        )))
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<EvaluatedValue>> {
        let flow = expect_streamline_input(ctx.inputs, self.title())?;
        let (bf_plan, bf_stale) =
            expect_boundary_field_input(ctx.inputs.get(1).and_then(|o| o.as_ref()), self.title())?;

        let before = flow.selected_streamlines.len();

        // Look up the actual boundary-field data in the execution cache.
        // The upstream `StreamlineDirectionFieldOp` emits a plan synchronously
        // but builds the field as an expensive job, so the cache may
        // be empty on the first evaluation — in that case we emit the
        // input as a stale passthrough (no filtering, no FICO DPS)
        // until the field is ready.
        let bf_cached = ctx
            .execution_cache
            .boundary_field_cache
            .get(&bf_plan.build_node_uuid)
            .cloned();

        // Register our interest in this BoundaryField so the GUI's
        // post-render cache-pruning sweep (in `jobs.rs`) doesn't drop
        // the field out from under us. Purifibre is a *consumer* of
        // the field (not a renderer of it), so without this the retain
        // filter — which only keeps fields referenced by bundle or
        // glyph DRAWS — would evict our upstream and leave us in the
        // `bf_cached=None` stale path every frame.
        ctx.scene_plan
            .boundary_fields_in_use
            .insert(bf_plan.build_node_uuid);

        // ── fingerprints ────────────────────────────────────────────
        //
        // Two fingerprints split the work:
        //
        // - `score_fingerprint` covers everything that affects the
        //   per-streamline FICO computation (the expensive part).
        // - `filter_fingerprint` is `score_fingerprint` plus
        //   `puri_fraction` (the cheap percentile cut).
        //
        // Re-thresholding without re-scoring keeps the puri_fraction
        // slider responsive — dragging it only re-runs the sort, not
        // the per-segment field lookups.
        //
        // Notably absent from `score_fingerprint`: the boundary
        // field's internal `CachedBoundaryField::fingerprint` value.
        // That number changes whenever the field rebuilds (different
        // voxel size, different upstream streamlines, etc.), and
        // including it in our fingerprint here would re-score every
        // time the field's hash drifted — even when our cached scores
        // are still semantically correct. Instead we treat the field
        // as identified by its build-node UUID, and rely on the
        // workflow's `bf_stale` flag to tell us when the upstream
        // field is in flux. The `bf_plan.build_node_uuid` term in the
        // hash means swapping to a different field source (different
        // upstream node) does invalidate, which is the only case we
        // care about.
        let score_fingerprint = {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            self.trim_fraction.to_bits().hash(&mut h);
            self.spherical_smoothing_deg.to_bits().hash(&mut h);
            flow.dataset.gpu_data.nb_streamlines.hash(&mut h);
            flow.dataset.gpu_data.nb_vertices.hash(&mut h);
            before.hash(&mut h);
            let stride = (before / 64).max(1);
            for i in (0..before).step_by(stride) {
                flow.selected_streamlines[i].hash(&mut h);
            }
            bf_plan.build_node_uuid.hash(&mut h);
            h.finish()
        };
        let filter_fingerprint = {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            score_fingerprint.hash(&mut h);
            self.puri_fraction.to_bits().hash(&mut h);
            h.finish()
        };
        let upstream_stale = ctx.upstream_stale() || bf_stale;

        let record = ctx
            .execution_cache
            .node_runs
            .entry(ctx.node.uuid)
            .or_default();
        prime_expensive_record(record, filter_fingerprint);
        sync_node_state_from_run_record(ctx.node_state, record);

        // Decide which level of work we need:
        //   - filter cache hit → reuse everything
        //   - score cache hit  → re-threshold only (cheap)
        //   - score cache miss → re-score from scratch (expensive)
        //
        // No `eval_mode == Settled` gate — the fingerprint match is the
        // only gate we need. Scoring is rayon-parallel and fast enough
        // for typical bundle sizes; if it ever isn't, this op should
        // graduate to the expensive-job system (mirrors how
        // `StreamlineDirectionField` does it).
        let cache_state = match ctx.execution_cache.purifibre_cache.get(&ctx.node.uuid) {
            Some(c) if c.filter_fingerprint == filter_fingerprint => CacheState::FilterHit,
            Some(c) if c.score_fingerprint == score_fingerprint => CacheState::ScoreHit,
            _ => CacheState::Miss,
        };

        // Field not built yet: prefer to keep serving the most recently
        // cached scored output (with FICO DPS) marked stale, so that
        // downstream consumers (ColorByDps etc.) keep seeing the FICO
        // field even while the upstream field rebuilds. Only fall back
        // to a true input passthrough when there is genuinely nothing
        // cached to serve.
        if bf_cached.is_none() {
            if let Some(c) = ctx.execution_cache.purifibre_cache.get(&ctx.node.uuid) {
                let scored_flow = StreamlineFlow {
                    dataset: c.scored_dataset.clone(),
                    selected_streamlines: c.scored_selection.clone(),
                    color_mode: flow.color_mode.clone(),
                    scalar_auto_range: flow.scalar_auto_range,
                    scalar_range_min: flow.scalar_range_min,
                    scalar_range_max: flow.scalar_range_max,
                    scalar_colormap: flow.scalar_colormap,
                };
                let filtered_flow = StreamlineFlow {
                    dataset: c.scored_dataset.clone(),
                    selected_streamlines: c.filtered_selection.clone(),
                    color_mode: flow.color_mode.clone(),
                    scalar_auto_range: flow.scalar_auto_range,
                    scalar_range_min: flow.scalar_range_min,
                    scalar_range_max: flow.scalar_range_max,
                    scalar_colormap: flow.scalar_colormap,
                };
                ctx.node_state.summary = format!("{} (boundary field rebuilding)", c.summary);
                return Ok(vec![
                    EvaluatedValue {
                        value: WorkflowValue::Streamline(scored_flow),
                        stale: true,
                    },
                    EvaluatedValue {
                        value: WorkflowValue::Streamline(filtered_flow),
                        stale: true,
                    },
                ]);
            }
            ctx.node_state.summary = "Purifibre: waiting for boundary field.".into();
            return Ok(vec![
                EvaluatedValue {
                    value: WorkflowValue::Streamline(flow.clone()),
                    stale: true,
                },
                EvaluatedValue {
                    value: WorkflowValue::Streamline(flow),
                    stale: true,
                },
            ]);
        }

        // ── do work as needed ───────────────────────────────────────
        match cache_state {
            CacheState::Miss => {
                // Field is ready (checked above). Score from scratch.
                let bf = bf_cached.as_ref().expect("checked above");
                let params = PurifibreParams {
                    trim_fraction: self.trim_fraction,
                    puri_fraction: self.puri_fraction,
                    spherical_smoothing_deg: self.spherical_smoothing_deg,
                };
                let report = purifibre_score(
                    &flow.dataset.gpu_data,
                    &flow.selected_streamlines,
                    &bf.field,
                    &params,
                );
                let scored_gpu = attach_fico_dps(&flow.dataset.gpu_data, report.fico.clone());
                let scored_dataset = Arc::new(StreamlineDataset {
                    name: flow.dataset.name.clone(),
                    gpu_data: Arc::new(scored_gpu),
                    backing: flow.dataset.backing.clone(),
                });
                let summary = report.summary.clone();
                ctx.execution_cache.purifibre_cache.insert(
                    ctx.node.uuid,
                    CachedPurifibre {
                        score_fingerprint,
                        filter_fingerprint,
                        scored_dataset,
                        scored_selection: flow.selected_streamlines.clone(),
                        fico_scores: report.fico,
                        filtered_selection: report.survivors,
                        summary: summary.clone(),
                    },
                );
                let record = ctx
                    .execution_cache
                    .node_runs
                    .entry(ctx.node.uuid)
                    .or_default();
                mark_expensive_success(record, filter_fingerprint, summary.clone());
                sync_node_state_from_run_record(ctx.node_state, record);
                ctx.node_state.summary = summary;
            }
            CacheState::ScoreHit => {
                // FICO scores match; only the threshold (puri_fraction)
                // changed. Re-derive the survivor list and update.
                let cached = ctx
                    .execution_cache
                    .purifibre_cache
                    .get(&ctx.node.uuid)
                    .expect("checked above");
                let (survivors, threshold) = apply_puri_threshold(
                    &cached.fico_scores,
                    &cached.scored_selection,
                    self.puri_fraction,
                );
                let summary = format!(
                    "{} → {} (FICO ≥ {:.3})",
                    cached.scored_selection.len(),
                    survivors.len(),
                    threshold,
                );
                let entry = ctx
                    .execution_cache
                    .purifibre_cache
                    .get_mut(&ctx.node.uuid)
                    .expect("checked above");
                entry.filter_fingerprint = filter_fingerprint;
                entry.filtered_selection = survivors;
                entry.summary = summary.clone();
                let record = ctx
                    .execution_cache
                    .node_runs
                    .entry(ctx.node.uuid)
                    .or_default();
                mark_expensive_success(record, filter_fingerprint, summary.clone());
                sync_node_state_from_run_record(ctx.node_state, record);
                ctx.node_state.summary = summary;
            }
            CacheState::FilterHit => {
                // Everything matches; nothing to recompute. Still
                // call mark_expensive_success so the run record's
                // last_success_fingerprint is asserted to equal the
                // current filter_fingerprint — clears any spurious
                // "Stale" set by `prime_expensive_record` if the
                // fingerprint shifted between runs (e.g., via
                // upstream hash-formula changes).
                let cached_summary = ctx
                    .execution_cache
                    .purifibre_cache
                    .get(&ctx.node.uuid)
                    .map(|c| c.summary.clone())
                    .unwrap_or_default();
                let record = ctx
                    .execution_cache
                    .node_runs
                    .entry(ctx.node.uuid)
                    .or_default();
                mark_expensive_success(record, filter_fingerprint, cached_summary.clone());
                sync_node_state_from_run_record(ctx.node_state, record);
                ctx.node_state.summary = cached_summary;
            }
        }

        // ── emit outputs ────────────────────────────────────────────
        let cache = ctx
            .execution_cache
            .purifibre_cache
            .get(&ctx.node.uuid)
            .expect("populated by the match arm above");
        // Stale only when our score depended on a stale upstream input.
        // (Pure puri_fraction changes don't make us stale — we just
        // re-thresholded.)
        let stale = upstream_stale;
        let scored_flow = StreamlineFlow {
            dataset: cache.scored_dataset.clone(),
            selected_streamlines: cache.scored_selection.clone(),
            color_mode: flow.color_mode.clone(),
            scalar_auto_range: flow.scalar_auto_range,
            scalar_range_min: flow.scalar_range_min,
            scalar_range_max: flow.scalar_range_max,
            scalar_colormap: flow.scalar_colormap,
        };
        let filtered_flow = StreamlineFlow {
            dataset: cache.scored_dataset.clone(),
            selected_streamlines: cache.filtered_selection.clone(),
            color_mode: flow.color_mode.clone(),
            scalar_auto_range: flow.scalar_auto_range,
            scalar_range_min: flow.scalar_range_min,
            scalar_range_max: flow.scalar_range_max,
            scalar_colormap: flow.scalar_colormap,
        };
        ctx.node_state.summary = if stale {
            format!("{} (stale)", cache.summary)
        } else {
            cache.summary.clone()
        };
        Ok(vec![
            EvaluatedValue {
                value: WorkflowValue::Streamline(scored_flow),
                stale,
            },
            EvaluatedValue {
                value: WorkflowValue::Streamline(filtered_flow),
                stale,
            },
        ])
    }
}

/// Three-way cache classification used by the evaluate body to decide
/// how much work to do.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CacheState {
    /// Filter fingerprint matches — outputs are entirely cached.
    FilterHit,
    /// Score fingerprint matches but the puri_fraction changed — we
    /// can reuse FICO scores and just re-threshold.
    ScoreHit,
    /// Neither matches — full re-score required.
    Miss,
}

/// Clone the input `gpu_data` and push a new `"fico"` DPS field onto
/// it (or overwrite any existing `"fico"` slot). `fico` must have
/// length `nb_streamlines`; extras or shorts are treated as a caller
/// bug and the result is clamped rather than panicking.
fn attach_fico_dps(src: &TrxGpuData, fico: Vec<f32>) -> TrxGpuData {
    let mut out = src.clone();

    // If a "fico" slot already exists (e.g. from a previous purifibre
    // pass upstream, or loaded from the file), replace it rather than
    // double-registering.
    let mut values = fico;
    if values.len() != out.nb_streamlines {
        values.resize(out.nb_streamlines, f32::NAN);
    }
    if let Some(pos) = out
        .dps_data
        .iter_mut()
        .position(|(n, _)| n == FICO_DPS_NAME)
    {
        out.dps_data[pos].1 = values;
    } else {
        out.dps_data.push((FICO_DPS_NAME.to_string(), values));
        if !out.dps_names.iter().any(|n| n == FICO_DPS_NAME) {
            out.dps_names.push(FICO_DPS_NAME.to_string());
        }
    }
    out
}

impl From<PurifibreOp> for WorkflowNodeKind {
    fn from(op: PurifibreOp) -> Self {
        Self::Purifibre {
            trim_fraction: op.trim_fraction,
            puri_fraction: op.puri_fraction,
            spherical_smoothing_deg: op.spherical_smoothing_deg,
        }
    }
}
