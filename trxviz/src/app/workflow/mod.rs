mod graph_viewer;
mod jobs;
mod project_io;

pub use graph_viewer::{
    GraphEditSummary, WorkflowGraphViewer, snarl_from_graph, sync_graph_from_snarl,
};
pub(crate) use jobs::workflow_job_kind_title;
pub use project_io::{
    WorkspacePane, apply_gui_slice_view_state, capture_gui_slice_view_state,
    default_workspace_tree, gui_load_project, gui_save_project,
};
pub use trxviz_core::workflow::*;

/// Returns `true` when the only difference between `before` and `after` is in
/// parameters that are render-only (GPU uniforms / visibility flags). These
/// changes can be applied immediately via `mark_render_only_edit` without
/// touching `document_revision` or triggering the expensive-job machinery.
///
/// Returns `false` for any change that may affect a fingerprinted computation
/// (mesh geometry, colormap baking, scalar projection, etc.), which must go
/// through `mark_workflow_semantic_edit`.
pub fn is_render_only_change(before: &WorkflowNodeKind, after: &WorkflowNodeKind) -> bool {
    use WorkflowNodeKind as K;
    match (before, after) {
        (K::VolumeDisplay { .. }, K::VolumeDisplay { .. }) => true,
        (K::BoundaryGlyphDisplay { .. }, K::BoundaryGlyphDisplay { .. }) => true,
        (K::SurfaceDisplay { .. }, K::SurfaceDisplay { .. }) => true,
        (K::SurfaceOverlayStack { .. }, K::SurfaceOverlayStack { .. }) => true,
        (
            K::StreamlineDisplay {
                enabled: _,
                render_style: rs1,
                tube_radius_mm: tr1,
                tube_sides: ts1,
                slab_half_width_mm: _,
            },
            K::StreamlineDisplay {
                enabled: _,
                render_style: rs2,
                tube_radius_mm: tr2,
                tube_sides: ts2,
                slab_half_width_mm: _,
            },
        ) => rs1 == rs2 && tr1 == tr2 && ts1 == ts2,
        (
            K::BundleSurfaceBuild {
                per_group: pg1,
                build_mode: bm1,
                voxel_size_mm: vs1,
                threshold: t1,
                smooth_sigma: ss1,
                min_component_volume_mm3: mc1,
                tube_radius_mm: tr1,
                tube_sides: ts1,
                opacity: _,
            },
            K::BundleSurfaceBuild {
                per_group: pg2,
                build_mode: bm2,
                voxel_size_mm: vs2,
                threshold: t2,
                smooth_sigma: ss2,
                min_component_volume_mm3: mc2,
                tube_radius_mm: tr2,
                tube_sides: ts2,
                opacity: _,
            },
        ) => {
            pg1 == pg2
                && bm1 == bm2
                && vs1 == vs2
                && t1 == t2
                && ss1 == ss2
                && mc1 == mc2
                && tr1 == tr2
                && ts1 == ts2
        }
        (K::ParcellationDisplay { .. }, K::ParcellationDisplay { .. }) => true,
        (K::OdfGlyphRenderer { .. }, K::OdfGlyphRenderer { .. }) => true,
        (K::Fixel3DDisplay { .. }, K::Fixel3DDisplay { .. }) => true,
        (K::Fixel2DDisplay { .. }, K::Fixel2DDisplay { .. }) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::is_render_only_change;
    use trxviz_core::data::loaded_files::VolumeColormap;
    use trxviz_core::data::trx_data::RenderStyle;
    use trxviz_core::workflow::WorkflowNodeKind;

    #[test]
    fn streamline_render_style_change_is_not_render_only() {
        let before = WorkflowNodeKind::StreamlineDisplay {
            enabled: true,
            render_style: RenderStyle::Flat,
            tube_radius_mm: 0.4,
            tube_sides: 8,
            slab_half_width_mm: 5.0,
        };
        let after = WorkflowNodeKind::StreamlineDisplay {
            enabled: true,
            render_style: RenderStyle::Tubes,
            tube_radius_mm: 0.4,
            tube_sides: 8,
            slab_half_width_mm: 5.0,
        };
        assert!(!is_render_only_change(&before, &after));
    }

    #[test]
    fn volume_display_change_is_render_only() {
        let before = WorkflowNodeKind::VolumeDisplay {
            colormap: VolumeColormap::Grayscale,
            opacity: 1.0,
            window_center: 0.5,
            window_width: 1.0,
        };
        let after = WorkflowNodeKind::VolumeDisplay {
            colormap: VolumeColormap::Grayscale,
            opacity: 0.25,
            window_center: 0.5,
            window_width: 1.0,
        };
        assert!(is_render_only_change(&before, &after));
    }
}
