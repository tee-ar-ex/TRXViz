//! Shared helpers for the two tractography ops (`YehTractographyOp` and
//! `DipyTractographyOp`). Pre-refactor both ops open-coded the same
//! "merge optional TrackingPlan overrides into my slider values + record
//! which fields the plan drove" logic, producing identical
//! hardcoded-field-name push loops. This module collects the pattern
//! once so adding a new overrideable scalar means editing exactly one
//! place.

use std::collections::BTreeMap;

use super::super::types::{NodeEvalState, TrackingPlan};

/// Which numeric fields a given tractography op exposes as overrideable.
/// Every tractography op shares the first five (min/max len, max angle,
/// step size, fixel threshold); `smooth_fraction` is Yeh-specific.
/// Adding a new field here requires updating the six helper methods
/// below in lockstep — the compiler catches any drift because every
/// branch is explicit.
#[derive(Clone, Copy)]
pub(super) struct TrackingFieldSet {
    pub smooth_fraction: bool,
}

impl TrackingFieldSet {
    pub const YEH: Self = Self {
        smooth_fraction: true,
    };
    pub const DIPY: Self = Self {
        smooth_fraction: false,
    };
}

/// Op-local default values (the op's own sliders). Consumed once per
/// evaluate() to produce `EffectiveTrackingParams` alongside the plan.
pub(super) struct OpTrackingDefaults {
    pub min_len_mm: f32,
    pub max_len_mm: f32,
    pub max_angle_deg: f32,
    pub step_size_mm: f32,
    pub fixel_threshold: f32,
    /// `Some(local)` for Yeh (which has the slider); `None` for Dipy
    /// (ignored — we never construct nor hash this field for Dipy).
    pub smooth_fraction: Option<f32>,
}

/// Scalar tracking params merged with a `TrackingPlan`'s optional
/// overrides. Each field falls back to the op's own slider value when
/// the plan left the corresponding slot `None`.
pub(super) struct EffectiveTrackingParams {
    pub min_len_mm: f32,
    pub max_len_mm: f32,
    pub max_angle_deg: f32,
    pub step_size_mm: f32,
    pub fixel_threshold: f32,
    pub smooth_fraction: Option<f32>,
}

impl EffectiveTrackingParams {
    pub fn merge(defaults: OpTrackingDefaults, plan: Option<&TrackingPlan>) -> Self {
        Self {
            min_len_mm: plan
                .and_then(|p| p.min_len_mm)
                .unwrap_or(defaults.min_len_mm),
            max_len_mm: plan
                .and_then(|p| p.max_len_mm)
                .unwrap_or(defaults.max_len_mm),
            max_angle_deg: plan
                .and_then(|p| p.max_angle_deg)
                .unwrap_or(defaults.max_angle_deg),
            step_size_mm: plan
                .and_then(|p| p.step_size_mm)
                .unwrap_or(defaults.step_size_mm),
            fixel_threshold: plan
                .and_then(|p| p.fixel_threshold)
                .unwrap_or(defaults.fixel_threshold),
            smooth_fraction: defaults
                .smooth_fraction
                .map(|local| plan.and_then(|p| p.smooth_fraction).unwrap_or(local)),
        }
    }

    /// Fold the effective scalars into a hasher. Called from the op's
    /// fingerprint computation so a plan override that changes a value
    /// invalidates cached results.
    pub fn hash_into<H: std::hash::Hasher>(&self, h: &mut H) {
        use std::hash::Hash;
        self.min_len_mm.to_bits().hash(h);
        self.max_len_mm.to_bits().hash(h);
        self.max_angle_deg.to_bits().hash(h);
        self.step_size_mm.to_bits().hash(h);
        self.fixel_threshold.to_bits().hash(h);
        if let Some(v) = self.smooth_fraction {
            v.to_bits().hash(h);
        }
    }
}

/// Record which plan fields ended up driving this op's behavior. The
/// GUI reads `node_state.overridden_fields` to grey out the
/// corresponding sliders and renders `node_state.overridden_values` so
/// the user sees the plan's numbers in place of their own.
///
/// `fixel_otsu` is a plan-informational value rather than a true
/// override (it feeds the fixel-threshold sentinel randomization), so
/// it only lands in `overridden_values` — never in `overridden_fields`.
pub(super) fn record_plan_overrides(
    node_state: &mut NodeEvalState,
    plan: &TrackingPlan,
    fields: TrackingFieldSet,
) {
    let mut names: Vec<String> = Vec::new();
    let mut values: BTreeMap<String, f32> = BTreeMap::new();

    if plan.seed_mask.is_some() {
        names.push("seed_mask".into());
    }
    if let Some(v) = plan.min_len_mm {
        names.push("min_len_mm".into());
        values.insert("min_len_mm".into(), v);
    }
    if let Some(v) = plan.max_len_mm {
        names.push("max_len_mm".into());
        values.insert("max_len_mm".into(), v);
    }
    if let Some(v) = plan.max_angle_deg {
        names.push("max_angle_deg".into());
        values.insert("max_angle_deg".into(), v);
    }
    if let Some(v) = plan.step_size_mm {
        names.push("step_size_mm".into());
        values.insert("step_size_mm".into(), v);
    }
    if let Some(v) = plan.fixel_threshold {
        names.push("fixel_threshold".into());
        values.insert("fixel_threshold".into(), v);
    }
    if fields.smooth_fraction
        && let Some(v) = plan.smooth_fraction
    {
        names.push("smooth_fraction".into());
        values.insert("smooth_fraction".into(), v);
    }
    if let Some(v) = plan.fixel_otsu {
        values.insert("fixel_otsu".into(), v);
    }

    node_state.overridden_fields = names;
    node_state.overridden_values = values;
}
