//! Runtime methods-boilerplate generation primitives.
//!
//! Ops opt in to the methods-boilerplate system through three
//! default-impl methods on [`WorkflowOp`](super::WorkflowOp):
//!
//! - [`WorkflowOp::citation_keys`](super::WorkflowOp::citation_keys) —
//!   BibTeX keys into `trxviz-core/data/citations.bib` that this op
//!   contributes when it appears in a workflow.
//! - [`WorkflowOp::boilerplate`](super::WorkflowOp::boilerplate) — a
//!   single sentence, with the op's parameter values interpolated,
//!   written in past tense and suitable for inclusion in a paper's
//!   Methods section. Use Pandoc `[@key]` citation syntax.
//! - [`WorkflowOp::describe`](super::WorkflowOp::describe) — a
//!   parameter-free short description used by the per-op reference
//!   documentation.
//!
//! The assembly that walks a [`WorkflowDocument`](super::types::WorkflowDocument)
//! and emits a final `methods.md` + `references.bib` pair lives in
//! [`generate_methods_report`]. Both the CLI `methods` subcommand and the
//! GUI modal call through this one function — everything downstream is
//! just file-writing or textbox-filling.

/// The canonical non-endorsement notice, shown verbatim in every
/// surface where TRXViz cites upstream methods — the GUI methods
/// modal, the CLI `methods` subcommand's stderr banner, the exported
/// `methods.md` header, and every auto-generated per-op doc page.
///
/// It must not be paraphrased at any surface; doing so risks drift
/// that subtly shifts the legal and ethical message ("we credit the
/// authors — we do not claim their approval"). A CI check asserts
/// every generated surface contains this exact string.
pub const NON_ENDORSEMENT_NOTICE: &str = "\
TRXViz provides re-implementations or ports of the methods referenced below. \
These implementations are NOT the authoritative versions — for the canonical \
implementations, please use the original software packages. Any differences \
in behavior or bugs are the responsibility of TRXViz, not the original \
authors. The presence of a citation here indicates only that TRXViz uses a \
method derived from that work and that users should credit the original \
authors; it does not imply that the original authors have reviewed, \
contributed to, or endorsed TRXViz.";

/// Coarse grouping used by per-op documentation to organize the
/// reference section in the docs and in the GUI op palette. Kept
/// deliberately shallow — most ops will map to one of these without
/// ambiguity, and finer buckets can come later if the taxonomy grows
/// too flat.
///
/// The default for an op is [`OpCategory::Other`]; ops override
/// [`WorkflowOp::category`](super::WorkflowOp::category) to pick
/// a more specific bucket.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpCategory {
    /// Loads data from disk or an asset slot: streamlines, volumes,
    /// surfaces, parcellations, CIFTI, ODX.
    Source,
    /// Transforms streamlines: subset, spatial query, dedup, tip
    /// prune, Purifibre, merge, etc.
    StreamlineFilter,
    /// Runs tractography or prepares a tractography plan
    /// (DIPY/Yeh/PTT direction getters, plan builders, region
    /// attachments).
    Tractography,
    /// Constructs ROIs/ROA masks (`RoiFromParcel`, `RoiFromVolume`,
    /// `RoiFromShape`, the `Parcel*` reactive ops).
    Roi,
    /// Builds derived surfaces or projects data onto them
    /// (bundle/parcel surface build, surface projection density/mean
    /// DPS, streamline direction field).
    Surface,
    /// Assigns color to streamlines, surfaces, or fixels without
    /// otherwise altering the data (`ColorBy*`, `UniformColor`).
    Coloring,
    /// Terminal display nodes consumed by the 3D/slice renderer
    /// (`StreamlineDisplay`, `SurfaceDisplay`, `VolumeDisplay`,
    /// `Fixel*Display`, `OdfGlyphRenderer`, etc.).
    Display,
    /// Writes data back out (`SaveStreamlines`). Distinct from
    /// `Source` so the docs can emphasize the one-way nature.
    Io,
    /// Catch-all for ops that don't fit cleanly into the buckets above
    /// yet. New ops start here; move them to a specific category once
    /// their role stabilizes.
    Other,
}

impl OpCategory {
    /// Human-readable label used in docs section headings and the GUI
    /// op-palette. Singular because headings read like "Tractography
    /// ops" (with the "ops" suffix supplied by the caller).
    pub fn label(self) -> &'static str {
        match self {
            OpCategory::Source => "Source",
            OpCategory::StreamlineFilter => "Streamline filter",
            OpCategory::Tractography => "Tractography",
            OpCategory::Roi => "ROI",
            OpCategory::Surface => "Surface",
            OpCategory::Coloring => "Coloring",
            OpCategory::Display => "Display",
            OpCategory::Io => "I/O",
            OpCategory::Other => "Other",
        }
    }
}

/// Small prose helper for ops whose methods sentence changes shape
/// based on a boolean parameter. Returns `if_true` when `cond` is true
/// and `if_false` otherwise — just a readability aid so conditional
/// wording in `boilerplate()` impls doesn't devolve into nested
/// `if`/`format!`. Mirrors the spirit of qsirecon's `ConditionalDoc`.
pub fn conditional<'a>(cond: bool, if_true: &'a str, if_false: &'a str) -> &'a str {
    if cond { if_true } else { if_false }
}

/// Shared BibTeX database, baked into the binary so exported reports
/// can ship a matching `.bib` without needing the source tree at
/// runtime. The same file is pointed at by `mkdocs-bibtex` for the
/// published docs — there is only one source of truth.
pub const CITATIONS_BIB: &str = include_str!("../../data/citations.bib");

/// Output of [`generate_methods_report`]. Every field is ready for
/// direct use: `body_markdown` can be written as `methods.md`,
/// `bibtex` as `references.bib`, and the two are self-consistent
/// (every `[@key]` in the body has a matching entry in the bibtex).
#[derive(Clone, Debug)]
pub struct MethodsReport {
    /// Markdown body, in Pandoc-citeproc-compatible form. Starts with
    /// the non-endorsement notice (as a blockquote), followed by a
    /// `## Methods` heading, the TRXViz preamble sentence, and one
    /// sentence per opted-in op in topological order.
    pub body_markdown: String,
    /// BibTeX keys actually referenced by `body_markdown`, in
    /// order-of-first-appearance (TRXViz itself is always first).
    /// De-duplicated.
    pub citation_keys: Vec<String>,
    /// Filtered [`CITATIONS_BIB`] containing only entries whose keys
    /// appear in `citation_keys`. Everything else — comments,
    /// `@string` macros, unused entries — is stripped so the emitted
    /// `.bib` has nothing the user didn't actually use.
    pub bibtex: String,
}

/// Per-op information captured for documentation generation. Populated
/// once by [`all_op_doc_info`] against a default instance of every
/// [`WorkflowOp`](super::WorkflowOp) implementation; the
/// `trxviz-docgen` crate walks the returned vec to emit the `docs/
/// reference/ops/` tree.
///
/// The fields mirror the trait's docgen-relevant methods so the docgen
/// crate doesn't need to depend on `WorkflowOp` directly.
#[derive(Clone, Debug)]
pub struct OpDocInfo {
    /// Stable machine-readable tag (from `WorkflowOp::tag`). Used as
    /// the per-op page filename.
    pub tag: &'static str,
    /// Human-readable title for page headers and the palette.
    pub title: &'static str,
    /// Parameter-free prose description for the page body.
    pub describe: std::borrow::Cow<'static, str>,
    /// Grouping for the index / sidebar.
    pub category: OpCategory,
    /// BibTeX keys this op contributes in its default configuration.
    /// Ops whose citation set depends on parameters (e.g. DIPY PTT vs
    /// probabilistic) surface the default-configuration set here;
    /// per-variant detail belongs in the boilerplate sentence.
    pub citation_keys: &'static [&'static str],
    /// Input ports. `None` for ops with dynamic port counts
    /// (`SurfaceOverlayStack`); the doc page explains the shape in prose.
    pub input_ports: Option<&'static [super::PortKind]>,
    /// Output ports (always static).
    pub output_ports: &'static [super::PortKind],
    /// A representative `WorkflowNodeKind` value for this op, used by
    /// the docs generator to resolve per-op port label overrides via
    /// [`super::port_labels`]. For ops with dynamic ports
    /// (`SurfaceOverlayStack`) this carries the default-instance shape.
    pub node_kind: super::ops::WorkflowNodeKind,
    /// Top-level parameter fields of the op struct and their
    /// default-instance values, reflected via `serde_json`. Empty for
    /// unit-struct ops (e.g. `Merge`, `AddRoi`). Values are emitted as
    /// JSON literals — scalars unadorned, compound values in their
    /// JSON form — so the docs show the shape a user would see in a
    /// saved project file.
    pub parameters: Vec<OpParameter>,
}

/// A single parameter entry on an op's docs page.
#[derive(Clone, Debug)]
pub struct OpParameter {
    /// Field name as it appears in the serialized form.
    pub name: String,
    /// Default value rendered as a JSON literal. For scalars this is
    /// e.g. `0.1`, `true`, `"fico"`; for compound values it's the
    /// compact JSON (`[0.0,0.0,0.0]`, `{"structure":"cortex_left"}`).
    pub default_json: String,
}

/// Reflect a `WorkflowNodeKind` variant's payload into a list of
/// parameter entries for docs rendering. `WorkflowNodeKind` is
/// externally tagged so struct-variants serialize to
/// `{"VariantName": {fields}}`; unit-variants serialize to the bare
/// string `"VariantName"`. Returns an empty vec for unit-variants and
/// for anything whose shape deviates from the normal struct-variant
/// form, which is the natural "no parameters" story.
fn reflect_parameters(kind: &super::ops::WorkflowNodeKind) -> Vec<OpParameter> {
    let Ok(value) = serde_json::to_value(kind) else {
        return Vec::new();
    };
    // Externally-tagged: unwrap the single-key outer object to reach
    // the payload. Unit variants arrive as bare strings and yield no
    // parameters.
    let serde_json::Value::Object(outer) = value else {
        return Vec::new();
    };
    let Some((_variant, inner)) = outer.into_iter().next() else {
        return Vec::new();
    };
    let serde_json::Value::Object(map) = inner else {
        return Vec::new();
    };
    map.into_iter()
        .map(|(name, value)| OpParameter {
            name,
            default_json: value.to_string(),
        })
        .collect()
}

/// Snapshot every `WorkflowOp` implementation's doc-relevant metadata
/// into a `Vec<OpDocInfo>`. The order is deterministic (matches the
/// in-source registry order) so the emitted docs are stable across
/// builds.
///
/// New ops must be added both here and in
/// `super::ops::validate_registry`; a test in this module diffs the
/// two to catch drift.
pub fn all_op_doc_info() -> Vec<OpDocInfo> {
    use super::op::WorkflowOp;
    use super::ops::*;
    use super::types::DipyDirectionGetter;
    use super::{
        BundleSurfaceBuildMode, BundleSurfaceColorMode, DpsFieldName, DpvFieldName, GroupFilter,
        ParcelIdSet, SurfaceDisplaySpace,
    };
    use crate::data::cifti::CiftiStructure;
    use crate::data::loaded_files::VolumeColormap;
    use crate::data::orientation_field::{BoundaryGlyphColorMode, DirectionFieldBinningMode};
    use crate::renderer::mesh_renderer::SurfaceColormap;
    use crate::units::Millimeters;
    use trx_rs::DuplicateRemovalParams;

    use super::ops::WorkflowNodeKind as K;

    fn describe<O: WorkflowOp>(op: &O, kind: K) -> OpDocInfo {
        OpDocInfo {
            tag: op.tag(),
            title: op.title(),
            describe: op.describe(),
            category: op.category(),
            citation_keys: op.citation_keys(),
            input_ports: Some(op.input_ports()),
            output_ports: op.output_ports(),
            parameters: reflect_parameters(&kind),
            node_kind: kind,
        }
    }

    let mut v = Vec::new();
    v.push(describe(
        &StreamlineSourceOp { source_id: 0 },
        K::StreamlineSource { source_id: 0 },
    ));
    v.push(describe(
        &ParcellationSourceOp { source_id: 0 },
        K::ParcellationSource { source_id: 0 },
    ));
    v.push(describe(
        &VolumeSourceOp { source_id: 0 },
        K::VolumeSource { source_id: 0 },
    ));
    v.push(describe(
        &CiftiSourceOp { source_id: 0 },
        K::CiftiSource { source_id: 0 },
    ));
    v.push(describe(
        &SurfaceSourceOp { source_id: 0 },
        K::SurfaceSource { source_id: 0 },
    ));
    v.push(describe(
        &OdxSourceOp { source_id: 0 },
        K::OdxSource { source_id: 0 },
    ));
    let limit = LimitStreamlinesOp::default();
    v.push(describe(
        &limit,
        K::LimitStreamlines {
            limit: limit.limit,
            randomize: limit.randomize,
            seed: limit.seed,
        },
    ));
    v.push(describe(
        &GroupSelectOp {
            groups: GroupFilter::All,
        },
        K::GroupSelect {
            groups: GroupFilter::All,
        },
    ));
    let rand = RandomSubsetOp::default();
    v.push(describe(
        &rand,
        K::RandomSubset {
            limit: rand.limit,
            seed: rand.seed,
        },
    ));
    v.push(describe(
        &SphereQueryOp {
            center: [0.0; 3],
            radius_mm: Millimeters(0.0),
        },
        K::SphereQuery {
            center: [0.0; 3],
            radius_mm: Millimeters(0.0),
        },
    ));
    v.push(describe(
        &RemoveDuplicatesOp {
            params: DuplicateRemovalParams::default(),
        },
        K::RemoveDuplicates {
            params: DuplicateRemovalParams::default(),
        },
    ));
    let tip = TipPruneOp::default();
    v.push(describe(
        &tip,
        K::TipPrune {
            voxel_size_mm: tip.voxel_size_mm,
            iterations: tip.iterations,
            min_support: tip.min_support,
            max_unsupported_fraction: tip.max_unsupported_fraction,
        },
    ));
    let puri = PurifibreOp::default();
    v.push(describe(
        &puri,
        K::Purifibre {
            trim_fraction: puri.trim_fraction,
            puri_fraction: puri.puri_fraction,
            spherical_smoothing_deg: puri.spherical_smoothing_deg,
        },
    ));
    v.push(describe(&MergeOp, K::Merge));
    v.push(describe(
        &AddGroupsFromParcellationOp,
        K::AddGroupsFromParcellation,
    ));
    v.push(describe(
        &ParcelSelectOp::default(),
        K::ParcelSelect {
            labels: ParcelIdSet::default(),
        },
    ));
    v.push(describe(&ParcelRoiOp, K::ParcelROI));
    v.push(describe(&ParcelRoaOp, K::ParcelROA));
    v.push(describe(
        &ParcelEndOp { endpoint_count: 1 },
        K::ParcelEnd { endpoint_count: 1 },
    ));
    v.push(describe(
        &ParcelCropOp { keep_inside: true },
        K::ParcelLimiting,
    ));
    v.push(describe(
        &ParcelCropOp { keep_inside: false },
        K::ParcelTerminative,
    ));
    v.push(describe(&ColorByDirectionOp, K::ColorByDirection));
    v.push(describe(&ColorByGroupOp, K::ColorByGroup));
    v.push(describe(
        &ColorByDpvOp {
            field: DpvFieldName::default(),
            colormap: SurfaceColormap::default(),
        },
        K::ColorByDPV {
            field: DpvFieldName::default(),
            colormap: SurfaceColormap::default(),
        },
    ));
    v.push(describe(
        &ColorByDpsOp {
            field: DpsFieldName::default(),
            colormap: SurfaceColormap::default(),
        },
        K::ColorByDPS {
            field: DpsFieldName::default(),
            colormap: SurfaceColormap::default(),
        },
    ));
    v.push(describe(
        &UniformColorOp { color: [0.0; 4] },
        K::UniformColor { color: [0.0; 4] },
    ));
    v.push(describe(
        &SurfaceDepthQueryOp {
            depth_mm: Millimeters(0.0),
        },
        K::SurfaceDepthQuery {
            depth_mm: Millimeters(0.0),
        },
    ));
    v.push(describe(
        &CiftiStructureOp {
            structure: CiftiStructure::CortexLeft,
            map_index: 0,
        },
        K::CiftiStructure {
            structure: CiftiStructure::CortexLeft,
            map_index: 0,
        },
    ));
    v.push(describe(
        &SurfaceProjectionDensityOp {
            depth_mm: Millimeters(0.0),
        },
        K::SurfaceProjectionDensity {
            depth_mm: Millimeters(0.0),
        },
    ));
    v.push(describe(
        &SurfaceProjectionMeanDpsOp {
            depth_mm: Millimeters(0.0),
            field: DpsFieldName::default(),
        },
        K::SurfaceProjectionMeanDps {
            depth_mm: Millimeters(0.0),
            field: DpsFieldName::default(),
        },
    ));
    let sd = StreamlineDisplayOp::default();
    v.push(describe(
        &sd,
        K::StreamlineDisplay {
            enabled: sd.enabled,
            render_style: sd.render_style,
            tube_radius_mm: sd.tube_radius_mm,
            tube_sides: sd.tube_sides,
            slab_half_width_mm: sd.slab_half_width_mm,
            opacity: sd.opacity,
        },
    ));
    v.push(describe(
        &SaveStreamlinesOp::default(),
        K::SaveStreamlines {
            output_path: String::new(),
        },
    ));
    v.push(describe(
        &OdxFixelScalarSelectOp::default(),
        K::OdxFixelScalarSelect {
            dpf_name: String::new(),
        },
    ));
    v.push(describe(
        &OdxVolumeSelectOp::default(),
        K::OdxVolumeSelect {
            dpv_name: String::new(),
        },
    ));
    v.push(describe(
        &ColorByFixelScalarsOp {
            colormap: SurfaceColormap::Inferno,
            range: None,
            length_scale_by_scalar: false,
        },
        K::ColorByFixelScalars {
            colormap: SurfaceColormap::Inferno,
            range: None,
            length_scale_by_scalar: false,
        },
    ));
    let f3 = Fixel3DDisplayOp::default();
    v.push(describe(
        &f3,
        K::Fixel3DDisplay {
            line_width: f3.line_width,
            length_scale: f3.length_scale,
            opacity: f3.opacity,
            offset_from_slice: f3.offset_from_slice,
            visible: f3.visible,
            auto_gate_from_otsu: f3.auto_gate_from_otsu,
            opacity_gate: f3.opacity_gate,
        },
    ));
    let f2 = Fixel2DDisplayOp::default();
    v.push(describe(
        &f2,
        K::Fixel2DDisplay {
            line_width: f2.line_width,
            opacity: f2.opacity,
            slab_thickness_mm: f2.slab_thickness_mm,
            length_scale: f2.length_scale,
            visible: f2.visible,
            auto_gate_from_otsu: f2.auto_gate_from_otsu,
            opacity_gate: f2.opacity_gate,
        },
    ));
    let odf = OdfGlyphRendererOp::default();
    v.push(describe(
        &odf,
        K::OdfGlyphRenderer {
            scale: odf.scale,
            subtract_iso: odf.subtract_iso,
            norm_within_voxel: odf.norm_within_voxel,
            opacity: odf.opacity,
            offset_from_slice: odf.offset_from_slice,
            gloss: odf.gloss,
            vertex_colormap: odf.vertex_colormap,
            slice_axis: odf.slice_axis,
            opacity_gate: odf.opacity_gate,
            size_gate: odf.size_gate,
            detail: odf.detail,
            visible: odf.visible,
        },
    ));
    let pd = ParcellationDisplayOp::default();
    v.push(describe(
        &pd,
        K::ParcellationDisplay {
            labels: pd.labels.clone(),
            opacity: pd.opacity,
        },
    ));
    let bsb = BundleSurfaceBuildOp::default();
    v.push(describe(
        &bsb,
        K::BundleSurfaceBuild {
            per_group: bsb.per_group,
            build_mode: bsb.build_mode,
            voxel_size_mm: bsb.voxel_size_mm,
            threshold: bsb.threshold,
            smooth_sigma: bsb.smooth_sigma,
            min_component_volume_mm3: bsb.min_component_volume_mm3,
            tube_radius_mm: bsb.tube_radius_mm,
            tube_sides: bsb.tube_sides,
            opacity: bsb.opacity,
        },
    ));
    let vd = VolumeDisplayOp::default();
    v.push(describe(
        &vd,
        K::VolumeDisplay {
            colormap: vd.colormap,
            opacity: vd.opacity,
            window_center: vd.window_center,
            window_width: vd.window_width,
        },
    ));
    // SurfaceOverlayStack has dynamic ports — surface `None` so the
    // docgen can annotate the page instead of trying to draw them.
    let overlay = SurfaceOverlayStackOp::default();
    let overlay_kind = K::SurfaceOverlayStack {
        layers: overlay.layers.clone(),
    };
    v.push(OpDocInfo {
        tag: overlay.tag(),
        title: overlay.title(),
        describe: overlay.describe(),
        category: overlay.category(),
        citation_keys: overlay.citation_keys(),
        input_ports: None,
        output_ports: overlay.output_ports(),
        parameters: reflect_parameters(&overlay_kind),
        node_kind: overlay_kind,
    });
    let surf = SurfaceDisplayOp::default();
    v.push(describe(
        &surf,
        K::SurfaceDisplay {
            color: surf.color,
            opacity: surf.opacity,
            outline_color: surf.outline_color,
            outline_thickness: surf.outline_thickness,
            show_projection_map: surf.show_projection_map,
            map_opacity: surf.map_opacity,
            map_threshold: surf.map_threshold,
            gloss: surf.gloss,
            projection_colormap: surf.projection_colormap,
            range_min: surf.range_min,
            range_max: surf.range_max,
            space: surf.space,
        },
    ));
    let sdf = StreamlineDirectionFieldOp::default();
    v.push(describe(
        &sdf,
        K::StreamlineDirectionField {
            voxel_size_mm: sdf.voxel_size_mm,
            sphere_lod: sdf.sphere_lod,
            normalization: sdf.normalization,
            binning_mode: sdf.binning_mode,
        },
    ));
    v.push(describe(
        &BundleSurfaceDisplayOp {
            color_mode: BundleSurfaceColorMode::Solid,
            outline_thickness: 0.0,
        },
        K::BundleSurfaceDisplay {
            color_mode: BundleSurfaceColorMode::Solid,
            outline_thickness: 0.0,
        },
    ));
    v.push(describe(
        &BoundaryGlyphDisplayOp {
            enabled: true,
            scale: 1.0,
            density_3d_step: 1,
            slice_density_step: 1,
            color_mode: BoundaryGlyphColorMode::DirectionRgb,
            min_contacts: 1,
        },
        K::BoundaryGlyphDisplay {
            enabled: true,
            scale: 1.0,
            density_3d_step: 1,
            slice_density_step: 1,
            color_mode: BoundaryGlyphColorMode::DirectionRgb,
            min_contacts: 1,
        },
    ));
    v.push(describe(&ParcelSurfaceBuildOp, K::ParcelSurfaceBuild));
    let rfp = RoiFromParcelOp::default();
    v.push(describe(
        &rfp,
        K::RoiFromParcel {
            labels: rfp.labels.clone(),
        },
    ));
    let rfv = RoiFromVolumeOp::default();
    v.push(describe(
        &rfv,
        K::RoiFromVolume {
            threshold: rfv.threshold,
        },
    ));
    let rfs = RoiFromShapeOp::default();
    v.push(describe(
        &rfs,
        K::RoiFromShape {
            center_ras: rfs.center_ras,
            radius_or_half_extent_mm: rfs.radius_or_half_extent_mm,
            shape: rfs.shape,
        },
    ));
    let vmd = VoxelMaskDisplayOp::default();
    v.push(describe(
        &vmd,
        K::VoxelMaskDisplay {
            color: vmd.color,
            opacity: vmd.opacity,
            smooth_sigma: vmd.smooth_sigma,
            min_component_volume_mm3: vmd.min_component_volume_mm3,
            style: vmd.style,
            slice_mode: vmd.slice_mode,
        },
    ));
    let ph = PrepareHausdorffPlanOp::default();
    v.push(describe(
        &ph,
        K::PrepareHausdorffPlan {
            tolerance_mm: ph.tolerance_mm,
            seed_tolerance_mm: ph.seed_tolerance_mm,
            tracking_metric: ph.tracking_metric.clone(),
            otsu_scope: ph.otsu_scope,
            seed_fixel_otsu_factor: ph.seed_fixel_otsu_factor,
            not_end_fixel_otsu_factor: ph.not_end_fixel_otsu_factor,
            max_reference_points: ph.max_reference_points,
        },
    ));
    let ps = PrepareSimplePlanOp::default();
    v.push(describe(
        &ps,
        K::PrepareSimplePlan {
            override_step: ps.override_step,
            step_size_mm: ps.step_size_mm,
            override_angle: ps.override_angle,
            max_angle_deg: ps.max_angle_deg,
            override_min_len: ps.override_min_len,
            min_len_mm: ps.min_len_mm,
            override_max_len: ps.override_max_len,
            max_len_mm: ps.max_len_mm,
            override_fixel_threshold: ps.override_fixel_threshold,
            fixel_threshold: ps.fixel_threshold,
            override_smooth: ps.override_smooth,
            smooth_fraction: ps.smooth_fraction,
            override_fixel_otsu: ps.override_fixel_otsu,
            fixel_otsu: ps.fixel_otsu,
        },
    ));
    v.push(describe(&AddRoiOp, K::AddRoi));
    v.push(describe(&AddRoaOp, K::AddRoa));
    v.push(describe(&AddEndRegionOp, K::AddEndRegion));
    v.push(describe(&AddNoEndOp, K::AddNoEnd));
    v.push(describe(&AddLimitingOp, K::AddLimiting));
    v.push(describe(&AddTermOp, K::AddTerm));
    let dipy = DipyTractographyOp::default();
    v.push(describe(
        &dipy,
        K::DipyTractography {
            step_size_mm: dipy.step_size_mm,
            max_angle_deg: dipy.max_angle_deg,
            min_len_mm: dipy.min_len_mm,
            max_len_mm: dipy.max_len_mm,
            fixel_threshold: dipy.fixel_threshold,
            relative_peak_threshold: dipy.relative_peak_threshold,
            seeds_per_voxel: dipy.seeds_per_voxel,
            max_points: dipy.max_points,
            rng_seed: dipy.rng_seed,
            direction_getter: dipy.direction_getter,
        },
    ));
    let yeh = YehTractographyOp::default();
    v.push(describe(
        &yeh,
        K::YehTractography {
            step_size_mm: yeh.step_size_mm,
            max_angle_deg: yeh.max_angle_deg,
            min_len_mm: yeh.min_len_mm,
            max_len_mm: yeh.max_len_mm,
            fixel_threshold: yeh.fixel_threshold,
            smooth_fraction: yeh.smooth_fraction,
            max_points: yeh.max_points,
            target_streamlines: yeh.target_streamlines,
            max_seed_attempts: yeh.max_seed_attempts,
            rng_seed: yeh.rng_seed,
        },
    ));
    // Silence unused-import warnings for the enums only used to spell
    // defaults above.
    let _ = BundleSurfaceBuildMode::MarchingCubes;
    let _ = VolumeColormap::Grayscale;
    let _ = DirectionFieldBinningMode::default();
    let _ = SurfaceDisplaySpace::Anatomical;
    let _ = DipyDirectionGetter::Probabilistic;
    v
}

/// Walk `doc.graph` in topological order, collect every opted-in op's
/// `boilerplate()` sentence and its `citation_keys()`, and emit a
/// self-consistent (markdown, bibtex) pair. The returned report
/// always credits TRXViz itself via the `trxviz` key.
///
/// Isolated ops (no wires) are included. Nodes that form cycles get
/// appended at the end in UUID order so they still appear somewhere
/// in the report; TRXViz's evaluator rejects cyclic graphs earlier,
/// so this only matters for corrupted files.
pub fn generate_methods_report(doc: &super::types::WorkflowDocument) -> MethodsReport {
    let order = topo_sort(&doc.graph);

    let mut sentences: Vec<String> = Vec::new();
    let mut keys_in_order: Vec<String> = Vec::new();
    let mut keys_seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    // TRXViz itself always gets cited via the preamble.
    keys_seen.insert("trxviz".to_string());
    keys_in_order.push("trxviz".to_string());

    for uuid in &order {
        let Some(node) = doc.graph.get(*uuid) else {
            continue;
        };
        if let Some(text) = super::ops::boilerplate(&node.op) {
            sentences.push(text);
        }
        for k in super::ops::citation_keys(&node.op) {
            if keys_seen.insert((*k).to_string()) {
                keys_in_order.push((*k).to_string());
            }
        }
    }

    let mut body = String::new();
    body.push_str(
        "<!-- TRXViz methods section. Generated automatically — do not edit by hand. -->\n\n",
    );
    // Non-endorsement notice. Blockquote so Pandoc renders it as a
    // callout rather than blending it into the methods prose.
    body.push_str("> **Not the authoritative implementation.** ");
    body.push_str(NON_ENDORSEMENT_NOTICE);
    body.push_str("\n\n");
    body.push_str("## Methods\n\n");
    body.push_str("The following analysis was performed using TRXViz [@trxviz].");
    if !sentences.is_empty() {
        body.push(' ');
        body.push_str(&sentences.join(" "));
    }
    body.push('\n');

    let used: std::collections::HashSet<&str> = keys_in_order.iter().map(|s| s.as_str()).collect();
    let bibtex = filter_bibtex(CITATIONS_BIB, &used);

    MethodsReport {
        body_markdown: body,
        citation_keys: keys_in_order,
        bibtex,
    }
}

/// Kahn's-algorithm topological sort over the workflow graph, with
/// UUID as the deterministic tie-break so the output of
/// `generate_methods_report` is stable across runs. Nodes whose
/// in-degree never drops to zero (i.e. members of a cycle) are
/// appended at the end in UUID order.
fn topo_sort(graph: &super::graph::WorkflowGraph) -> Vec<super::types::WorkflowNodeUuid> {
    use super::types::WorkflowNodeUuid;
    use std::collections::{BTreeMap, BTreeSet};

    let mut in_deg: BTreeMap<WorkflowNodeUuid, usize> =
        graph.nodes().map(|(u, _)| (u, 0usize)).collect();
    for wire in graph.wires() {
        if in_deg.contains_key(&wire.from.node) && in_deg.contains_key(&wire.to.node) {
            *in_deg.entry(wire.to.node).or_insert(0) += 1;
        }
    }

    let mut ready: BTreeSet<WorkflowNodeUuid> = in_deg
        .iter()
        .filter_map(|(u, d)| (*d == 0).then_some(*u))
        .collect();

    let mut order: Vec<WorkflowNodeUuid> = Vec::with_capacity(graph.len());
    while let Some(&u) = ready.iter().next() {
        ready.remove(&u);
        order.push(u);
        in_deg.remove(&u);
        for wire in graph.wires().filter(|w| w.from.node == u) {
            if let Some(d) = in_deg.get_mut(&wire.to.node) {
                *d = d.saturating_sub(1);
                if *d == 0 {
                    ready.insert(wire.to.node);
                }
            }
        }
    }

    // Cycle remnants — append in UUID order so they still appear.
    for (u, _) in graph.nodes() {
        if !order.contains(&u) {
            order.push(u);
        }
    }
    order
}

/// Minimal BibTeX filter: keeps only entries whose key is in `used`,
/// dropping everything else (header comments, `@string` macros,
/// irrelevant entries). Not a full BibTeX parser — we rely on two
/// shape invariants of [`CITATIONS_BIB`]:
///
///   1. Every entry is `@type{key, ... }` (or `@type(key, ... )`), with
///      balanced braces inside the body.
///   2. Keys are ASCII identifiers (no commas, no `}`), so the comma
///      after the key is unambiguous.
///
/// Both hold for the hand-curated `citations.bib`; if we ever adopt a
/// third-party bib file we'd want a real parser. For now the scanner
/// is ~30 lines and zero deps.
fn filter_bibtex(src: &str, used: &std::collections::HashSet<&str>) -> String {
    let bytes = src.as_bytes();
    let mut out = String::new();
    let mut i = 0usize;

    while i < bytes.len() {
        if bytes[i] != b'@' {
            i += 1;
            continue;
        }
        let entry_start = i;
        i += 1; // past '@'

        // Entry type: consecutive ASCII letters.
        while i < bytes.len() && bytes[i].is_ascii_alphabetic() {
            i += 1;
        }
        if i >= bytes.len() || (bytes[i] != b'{' && bytes[i] != b'(') {
            continue;
        }
        let open = bytes[i];
        let close = if open == b'{' { b'}' } else { b')' };
        i += 1; // past opening delimiter; brace depth is now 1

        // Skip whitespace, then read the key up to ',' (or to the
        // closing delimiter for keyless entries like @comment).
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let key_start = i;
        while i < bytes.len() && bytes[i] != b',' && bytes[i] != close {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let key = std::str::from_utf8(&bytes[key_start..i])
            .unwrap_or("")
            .trim();

        // Balance braces to find the end of the entry. We're at either
        // ',' (depth still 1) or the closing delimiter (which will
        // bring depth to 0 on the first step). Field *values* always
        // use braces for quoting even when the entry itself is
        // paren-delimited, so we track the outer depth via `open`/`close`
        // and nested depth via `{`/`}` — both contribute to the same
        // counter, which zeroes only at the outermost close.
        let mut depth = 1i32;
        while i < bytes.len() && depth > 0 {
            let b = bytes[i];
            if b == open {
                depth += 1;
            } else if b == close {
                depth -= 1;
            } else if open != b'{' && b == b'{' {
                depth += 1;
            } else if open != b'{' && b == b'}' {
                depth -= 1;
            }
            i += 1;
        }

        if used.contains(key) {
            out.push_str(&src[entry_start..i]);
            out.push('\n');
            out.push('\n');
        }
    }

    out
}

#[cfg(test)]
mod tests {
    //! Round-trip tests exercising the registry-level dispatch for the
    //! new methods-boilerplate trait methods. These verify that a
    //! `WorkflowNodeKind` built from a serialized workflow — the shape
    //! callers will actually have — returns sensible citations, a
    //! parameter-interpolated sentence, and a usable category. As more
    //! ops opt in to the boilerplate system, add cases here rather
    //! than proliferating per-op test modules.
    use super::super::WorkflowNodeKind;
    use super::super::ops;
    use super::super::types::DipyDirectionGetter;
    use super::*;

    #[test]
    fn non_endorsement_notice_names_trxviz_and_is_explicit() {
        // Cheap guard against accidental paraphrase. The exact string
        // is asserted by the CI docs-surface grep test in a later
        // chunk; here we just catch careless edits to the const.
        assert!(NON_ENDORSEMENT_NOTICE.contains("TRXViz"));
        assert!(NON_ENDORSEMENT_NOTICE.contains("NOT the authoritative"));
        assert!(NON_ENDORSEMENT_NOTICE.contains("endorsed"));
    }

    #[test]
    fn purifibre_boilerplate_interpolates_parameters_and_cites_correctly() {
        let kind = WorkflowNodeKind::Purifibre {
            trim_fraction: 0.10,
            puri_fraction: 0.15,
            spherical_smoothing_deg: 15.0,
        };
        assert_eq!(ops::category(&kind), OpCategory::StreamlineFilter);
        assert_eq!(ops::citation_keys(&kind), &["purifibre", "nibrary"]);
        let text = ops::boilerplate(&kind).expect("purifibre contributes methods prose");
        assert!(text.contains("Purifibre"), "missing method name: {text}");
        assert!(
            text.contains("[@purifibre;@nibrary]"),
            "missing citation: {text}"
        );
        assert!(text.contains("10% trim"), "trim% not interpolated: {text}");
        assert!(
            text.contains("15% discard"),
            "puri% not interpolated: {text}"
        );
        assert!(
            text.contains("15.0°"),
            "smoothing deg not interpolated: {text}"
        );
    }

    #[test]
    fn dipy_tractography_cites_ptt_only_when_ptt_direction_getter_selected() {
        let probabilistic = WorkflowNodeKind::DipyTractography {
            step_size_mm: 0.5,
            max_angle_deg: 60.0,
            min_len_mm: 10.0,
            max_len_mm: 300.0,
            fixel_threshold: 0.1,
            relative_peak_threshold: 0.25,
            seeds_per_voxel: 1,
            max_points: 501,
            rng_seed: 42,
            direction_getter: DipyDirectionGetter::Probabilistic,
        };
        assert_eq!(
            ops::citation_keys(&probabilistic),
            &["gpustreamlines", "dipy"]
        );
        let prob_text = ops::boilerplate(&probabilistic).expect("has prose");
        assert!(
            prob_text.contains("probabilistic"),
            "method not named: {prob_text}"
        );
        assert!(
            prob_text.contains("[@gpustreamlines]"),
            "missing GPUStreamlines citation: {prob_text}"
        );
        assert!(
            !prob_text.contains("[@ptt]"),
            "spurious PTT citation: {prob_text}"
        );

        let ptt = WorkflowNodeKind::DipyTractography {
            step_size_mm: 0.5,
            max_angle_deg: 60.0,
            min_len_mm: 10.0,
            max_len_mm: 300.0,
            fixel_threshold: 0.1,
            relative_peak_threshold: 0.25,
            seeds_per_voxel: 1,
            max_points: 501,
            rng_seed: 42,
            direction_getter: DipyDirectionGetter::Ptt {
                probe_length_mm: 1.5,
                probe_quality: 4,
                probe_radius_mm: 0.0,
                probe_count: 1,
                max_curvature_per_mm: 1.0,
                data_support_exponent: 1.0,
                min_data_support: 0.05,
                rejection_sampling_max_try: 100,
            },
        };
        assert_eq!(
            ops::citation_keys(&ptt),
            &["ptt", "gpustreamlines_ptt_ismrm", "gpustreamlines", "dipy"]
        );
        let ptt_text = ops::boilerplate(&ptt).expect("has prose");
        assert!(
            ptt_text.contains("[@ptt]"),
            "missing PTT citation: {ptt_text}"
        );
        assert!(
            ptt_text.contains("gpustreamlines_ptt_ismrm"),
            "missing GPU PTT citation: {ptt_text}"
        );
        assert!(
            ptt_text.contains("Parallel Transport"),
            "method not named: {ptt_text}"
        );
    }

    #[test]
    fn yeh_tractography_cites_dsi_studio_and_gqi() {
        let kind = WorkflowNodeKind::YehTractography {
            step_size_mm: 1.0,
            max_angle_deg: 60.0,
            min_len_mm: 10.0,
            max_len_mm: 300.0,
            fixel_threshold: 0.05,
            smooth_fraction: 0.25,
            max_points: 501,
            target_streamlines: 30_000,
            max_seed_attempts: 10_000_000,
            rng_seed: 42,
        };
        assert_eq!(ops::category(&kind), OpCategory::Tractography);
        assert_eq!(
            ops::citation_keys(&kind),
            &["yeh2025dsistudio", "yeh2013gqi", "yeh2020shape"]
        );
        let text = ops::boilerplate(&kind).expect("has prose");
        assert!(text.contains("DSI Studio"), "method not named: {text}");
        assert!(
            text.contains("[@yeh2025dsistudio;@yeh2013gqi]"),
            "missing DSI Studio / GQI citation: {text}"
        );
        assert!(
            text.contains("[@yeh2020shape]"),
            "missing augmented-tracking citation: {text}"
        );
        assert!(
            text.contains("30000"),
            "target_streamlines not interpolated: {text}"
        );
    }

    #[test]
    fn hausdorff_plan_cites_yeh_2020_shape_and_interpolates_tolerance() {
        let kind = WorkflowNodeKind::PrepareHausdorffPlan {
            tolerance_mm: 12.0,
            seed_tolerance_mm: 2.0,
            tracking_metric: None,
            otsu_scope: odx_rs::qc::OtsuScope::AllFixels,
            seed_fixel_otsu_factor: 0.5,
            not_end_fixel_otsu_factor: 0.9,
            max_reference_points: 20_000,
        };
        assert_eq!(ops::category(&kind), OpCategory::Tractography);
        assert_eq!(ops::citation_keys(&kind), &["yeh2020shape"]);
        let text = ops::boilerplate(&kind).expect("has prose");
        assert!(
            text.contains("[@yeh2020shape]"),
            "missing shape-analysis citation: {text}"
        );
        assert!(text.contains("12.0"), "tolerance not interpolated: {text}");
        assert!(text.contains("20000"), "max_ref not interpolated: {text}");
    }

    #[test]
    fn filter_bibtex_keeps_only_used_entries_and_preserves_their_bodies() {
        let src = r#"
% Header comment.

@article{keep_me,
  title = {Keep},
  author = {Someone}
}

@article{drop_me,
  title = {Drop}
}

@article{nested_braces,
  title = {Has {nested} braces},
  note = {and a , comma}
}
"#;
        let mut used = std::collections::HashSet::new();
        used.insert("keep_me");
        used.insert("nested_braces");
        let out = filter_bibtex(src, &used);
        assert!(out.contains("keep_me"));
        assert!(out.contains("Keep"));
        assert!(!out.contains("drop_me"));
        assert!(!out.contains("Drop"));
        // Nested-brace entry survives intact, comma-in-value included.
        assert!(out.contains("nested_braces"));
        assert!(out.contains("Has {nested} braces"));
        assert!(out.contains("and a , comma"));
    }

    #[test]
    fn generate_methods_report_emits_topo_ordered_prose_and_matching_bibtex() {
        use super::super::WorkflowNode;
        use super::super::graph::{GraphPos, InPort, OutPort, WorkflowGraph};
        use super::super::types::{WorkflowDocument, WorkflowNodeUuid, default_document};

        // Build tracking → purifibre. Tractography (uuid 2) is
        // upstream; Purifibre (uuid 1) consumes its streamlines via
        // its first input port.
        let mut graph = WorkflowGraph::new();
        let track_uuid = WorkflowNodeUuid(2);
        let puri_uuid = WorkflowNodeUuid(1);
        graph.insert_node(
            WorkflowNode {
                uuid: track_uuid,
                op: WorkflowNodeKind::DipyTractography {
                    step_size_mm: 0.5,
                    max_angle_deg: 60.0,
                    min_len_mm: 10.0,
                    max_len_mm: 300.0,
                    fixel_threshold: 0.1,
                    relative_peak_threshold: 0.25,
                    seeds_per_voxel: 1,
                    max_points: 501,
                    rng_seed: 42,
                    direction_getter: DipyDirectionGetter::Probabilistic,
                },
                label: "track".into(),
            },
            GraphPos::ZERO,
        );
        graph.insert_node(
            WorkflowNode {
                uuid: puri_uuid,
                op: WorkflowNodeKind::Purifibre {
                    trim_fraction: 0.10,
                    puri_fraction: 0.15,
                    spherical_smoothing_deg: 15.0,
                },
                label: "puri".into(),
            },
            GraphPos::ZERO,
        );
        graph.connect(
            OutPort {
                node: track_uuid,
                output: 0,
            },
            InPort {
                node: puri_uuid,
                input: 0,
            },
        );

        let mut doc: WorkflowDocument = default_document();
        doc.graph = graph;

        let report = generate_methods_report(&doc);

        // Topo order: tracking first, purifibre second — even though
        // UUIDs were assigned in reverse order, the wire dictates
        // dependency. A tractography substring must appear before
        // "Purifibre" in the body.
        let track_idx = report
            .body_markdown
            .find("streamline tractography")
            .expect("tractography sentence present");
        let puri_idx = report
            .body_markdown
            .find("Purifibre")
            .expect("purifibre sentence present");
        assert!(
            track_idx < puri_idx,
            "tractography should precede purifibre in topological order: \n{}",
            report.body_markdown
        );

        // Non-endorsement notice is rendered as a blockquote and must
        // appear before the methods heading.
        assert!(
            report
                .body_markdown
                .contains("Not the authoritative implementation"),
            "missing non-endorsement notice"
        );
        let notice_idx = report
            .body_markdown
            .find("Not the authoritative implementation")
            .unwrap();
        let methods_idx = report.body_markdown.find("## Methods").unwrap();
        assert!(notice_idx < methods_idx);

        // Citation keys: trxviz first, then order-of-appearance.
        assert_eq!(
            report.citation_keys.first().map(String::as_str),
            Some("trxviz")
        );
        assert!(report.citation_keys.contains(&"gpustreamlines".to_string()));
        assert!(report.citation_keys.contains(&"dipy".to_string()));
        assert!(report.citation_keys.contains(&"purifibre".to_string()));
        assert!(report.citation_keys.contains(&"nibrary".to_string()));
        // No PTT in this probabilistic-only workflow.
        assert!(!report.citation_keys.contains(&"ptt".to_string()));

        // Dedup: each key appears exactly once.
        let mut sorted = report.citation_keys.clone();
        sorted.sort();
        let mut deduped = sorted.clone();
        deduped.dedup();
        assert_eq!(sorted, deduped, "citation_keys must be deduped");

        // BibTeX: contains every used key and nothing else.
        for k in &report.citation_keys {
            assert!(
                report.bibtex.contains(&format!("{{{k},")),
                "bibtex missing entry for {k}:\n{}",
                report.bibtex
            );
        }
        // Unused keys from citations.bib are filtered out.
        assert!(
            !report.bibtex.contains("yeh2013gqi"),
            "unused entry leaked into bibtex: {}",
            report.bibtex
        );
        assert!(!report.bibtex.contains("ptt,"));
    }

    #[test]
    fn generate_methods_report_on_empty_document_still_credits_trxviz() {
        use super::super::types::default_document;

        let doc = default_document();
        let report = generate_methods_report(&doc);

        assert_eq!(report.citation_keys, vec!["trxviz".to_string()]);
        assert!(report.body_markdown.contains("[@trxviz]"));
        assert!(report.body_markdown.contains("Not the authoritative"));
        assert!(report.bibtex.contains("trxviz"));
    }

    #[test]
    fn generate_methods_report_end_to_end_for_purifibre_plus_dipy_plus_display() {
        // Builds a realistic little workflow — a source, Purifibre, DIPY
        // tractography, and a pure-display node — and checks every output
        // of `generate_methods_report` is self-consistent:
        //
        // - body prose interpolates parameter values and includes every
        //   expected citation key,
        // - citation_keys lists keys in order-of-first-appearance with
        //   `trxviz` always first and no duplicates,
        // - the filtered bibtex contains exactly the used entries and
        //   excludes unused ones,
        // - the display node contributes no prose and no citations.
        use super::super::graph::{InPort, OutPort};
        use super::super::types::DipyDirectionGetter;
        use super::super::types::{WorkflowNode, WorkflowNodeUuid, default_document};
        use super::super::{GraphPos, WorkflowNodeKind};

        let mut doc = default_document();
        let src = WorkflowNodeUuid(1);
        let puri = WorkflowNodeUuid(2);
        let track = WorkflowNodeUuid(3);
        let display = WorkflowNodeUuid(4);
        doc.graph.insert_node(
            WorkflowNode {
                uuid: src,
                op: WorkflowNodeKind::StreamlineSource { source_id: 0 },
                label: "src".into(),
            },
            GraphPos::new(0.0, 0.0),
        );
        doc.graph.insert_node(
            WorkflowNode {
                uuid: puri,
                op: WorkflowNodeKind::Purifibre {
                    trim_fraction: 0.10,
                    puri_fraction: 0.10,
                    spherical_smoothing_deg: 15.0,
                },
                label: "puri".into(),
            },
            GraphPos::new(0.0, 0.0),
        );
        doc.graph.insert_node(
            WorkflowNode {
                uuid: track,
                op: WorkflowNodeKind::DipyTractography {
                    step_size_mm: 0.5,
                    max_angle_deg: 60.0,
                    min_len_mm: 10.0,
                    max_len_mm: 300.0,
                    fixel_threshold: 0.1,
                    relative_peak_threshold: 0.25,
                    seeds_per_voxel: 1,
                    max_points: 501,
                    rng_seed: 42,
                    direction_getter: DipyDirectionGetter::Probabilistic,
                },
                label: "track".into(),
            },
            GraphPos::new(0.0, 0.0),
        );
        doc.graph.insert_node(
            WorkflowNode {
                uuid: display,
                op: WorkflowNodeKind::ColorByDirection,
                label: "display".into(),
            },
            GraphPos::new(0.0, 0.0),
        );
        doc.graph.connect(
            OutPort {
                node: src,
                output: 0,
            },
            InPort {
                node: puri,
                input: 0,
            },
        );
        doc.graph.connect(
            OutPort {
                node: puri,
                output: 0,
            },
            InPort {
                node: track,
                input: 0,
            },
        );
        doc.graph.connect(
            OutPort {
                node: track,
                output: 0,
            },
            InPort {
                node: display,
                input: 0,
            },
        );
        doc.next_node_uuid = 5;

        let report = generate_methods_report(&doc);

        // Non-endorsement blockquote + TRXViz preamble + parameter-interpolated
        // sentences from both citable ops.
        assert!(report.body_markdown.contains("Not the authoritative"));
        assert!(report.body_markdown.contains("[@trxviz]"));
        assert!(report.body_markdown.contains("Purifibre"));
        assert!(report.body_markdown.contains("[@purifibre;@nibrary]"));
        assert!(report.body_markdown.contains("10% trim"));
        assert!(report.body_markdown.contains("15.0°"));
        // DIPY sentence — exact citation token depends on the direction
        // getter; Probabilistic doesn't include [@ptt].
        assert!(report.body_markdown.contains("[@dipy"));
        assert!(!report.body_markdown.contains("[@ptt"));
        // Parameter values from DIPY.
        assert!(report.body_markdown.contains("0.5"));
        assert!(report.body_markdown.contains("60"));

        // citation_keys: trxviz first, then in order of first appearance.
        assert_eq!(
            report.citation_keys.first().map(String::as_str),
            Some("trxviz")
        );
        let keys: std::collections::HashSet<&str> =
            report.citation_keys.iter().map(String::as_str).collect();
        for expected in ["trxviz", "purifibre", "nibrary", "dipy"] {
            assert!(keys.contains(expected), "missing citation key: {expected}");
        }
        assert!(
            !keys.contains("ptt"),
            "ptt should not be cited for Probabilistic"
        );
        // No duplicates.
        let mut deduped: Vec<&str> = report.citation_keys.iter().map(String::as_str).collect();
        deduped.sort();
        deduped.dedup();
        assert_eq!(deduped.len(), report.citation_keys.len());

        // Filtered bibtex: every used key present, unused keys absent.
        for expected in &report.citation_keys {
            assert!(
                report.bibtex.contains(&format!("@article{{{expected}"))
                    || report.bibtex.contains(&format!("@software{{{expected}"))
                    || report
                        .bibtex
                        .contains(&format!("@inproceedings{{{expected}")),
                "bibtex missing entry for used key: {expected}\n---\n{}",
                report.bibtex,
            );
        }
        // `ptt` is a real entry in citations.bib but shouldn't appear here
        // because the workflow doesn't use PTT.
        assert!(
            !report.bibtex.contains("@article{ptt,"),
            "filtered bibtex leaked the unused `ptt` entry",
        );
    }

    #[test]
    fn display_op_has_no_boilerplate_and_empty_citations() {
        // Pure display nodes don't represent citable methods:
        // boilerplate() returns None and citation_keys() returns an
        // empty slice. Category is set per-op (ColorByDirection groups
        // under Coloring in the reference index).
        let kind = WorkflowNodeKind::ColorByDirection;
        assert!(ops::boilerplate(&kind).is_none());
        assert!(ops::citation_keys(&kind).is_empty());
        assert_eq!(ops::category(&kind), OpCategory::Coloring);
    }
}
