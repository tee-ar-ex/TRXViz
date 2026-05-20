//! Human-readable labels for node ports.
//!
//! Two layers:
//! - [`port_name`] — the default label for a `PortKind` (spaces, proper
//!   casing). Used whenever an op doesn't override the label for that
//!   specific port index.
//! - [`input_port_label`] / [`output_port_label`] — per-op overrides
//!   for ops whose ports mean something more specific than the generic
//!   kind name (e.g. Purifibre's two outputs are "Scored (all)" and
//!   "Filtered", not two identical "Streamline" labels).
//!
//! Shared between the GUI (egui-snarl renders the same strings next to
//! each pin), the docs SVG generator (`trxviz-docgen::svg_layout`), and
//! the per-op reference markdown pages (`trxviz-docgen::op_pages`).
//! Keep all call sites in lockstep by routing them through this module.

use super::PortKind;
use super::ops::WorkflowNodeKind;

/// Default, op-agnostic label for a port kind. Used as the fallback
/// when an op doesn't have a per-port override.
pub fn port_name(port: PortKind) -> &'static str {
    match port {
        PortKind::Streamline => "Streamline",
        PortKind::Volume => "Volume",
        PortKind::Surface => "Surface",
        PortKind::Parcellation => "Parcellation",
        PortKind::ParcelSelection => "Parcel Set",
        PortKind::GroupSelection => "Group Set",
        PortKind::Cifti => "CIFTI",
        PortKind::SurfaceScalars => "Surface Scalars",
        PortKind::SurfaceAppearance => "Surface Appearance",
        PortKind::BundleSurface => "Bundle Surface",
        PortKind::BoundaryField => "Boundary Field",
        PortKind::Fixels => "Fixels",
        PortKind::FixelScalars => "Fixel Scalars",
        PortKind::OdfField => "ODF Field",
        PortKind::OdxCatalog => "ODX Catalog",
        PortKind::VoxelMask => "Voxel Mask",
        PortKind::TrackingPlan => "Tracking Plan",
    }
}

pub fn input_port_label(
    node_kind: &WorkflowNodeKind,
    input_index: usize,
    port: PortKind,
) -> String {
    match node_kind {
        WorkflowNodeKind::OdfGlyphRenderer { .. } => match input_index {
            0 => "ODF Field".to_string(),
            1 => "Opacity Scalars".to_string(),
            2 => "Size Scalars".to_string(),
            _ => port_name(port).to_string(),
        },
        WorkflowNodeKind::Purifibre { .. } => match input_index {
            0 => "Streamlines".to_string(),
            1 => "Direction field".to_string(),
            _ => port_name(port).to_string(),
        },
        WorkflowNodeKind::SurfaceOverlayStack { layers } => {
            if input_index == 0 {
                "Surface".to_string()
            } else {
                let layer_index = input_index - 1;
                let layer_name = if layer_index == 0 {
                    "Layer 0: Base".to_string()
                } else {
                    format!("Layer {layer_index}")
                };
                let legend = layers
                    .get(layer_index)
                    .map(|layer| layer.legend_label.trim())
                    .filter(|legend| !legend.is_empty())
                    .unwrap_or("");
                if legend.is_empty() {
                    format!("{layer_name} Scalars")
                } else {
                    format!("{layer_name} Scalars ({legend})")
                }
            }
        }
        _ => port_name(port).to_string(),
    }
}

pub fn output_port_label(
    node_kind: &WorkflowNodeKind,
    output_index: usize,
    port: PortKind,
) -> String {
    match node_kind {
        WorkflowNodeKind::PrepareHausdorffPlan { .. } => match output_index {
            0 => "Plan".to_string(),
            1 => "Seed Mask".to_string(),
            2 => "Limiting Mask".to_string(),
            3 => "No-End Mask".to_string(),
            _ => port_name(port).to_string(),
        },
        WorkflowNodeKind::PrepareSimplePlan { .. } => match output_index {
            0 => "Plan".to_string(),
            _ => port_name(port).to_string(),
        },
        WorkflowNodeKind::PreparePyafqPlan { .. } => match output_index {
            0 => "Plan".to_string(),
            1 => "Include".to_string(),
            2 => "Exclude".to_string(),
            3 => "Start Mask".to_string(),
            4 => "End Mask".to_string(),
            5 => "Probability Map".to_string(),
            _ => port_name(port).to_string(),
        },
        WorkflowNodeKind::Purifibre { .. } => match output_index {
            // Output 0 is the input streamlines passthrough with the
            // FICO DPS field attached — useful for visualizing the
            // score distribution before any filtering happens.
            0 => "Scored (all)".to_string(),
            // Output 1 has only streamlines that survived the
            // discard-fraction cutoff.
            1 => "Filtered".to_string(),
            _ => port_name(port).to_string(),
        },
        _ => port_name(port).to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::ops::WorkflowNodeKind;

    #[test]
    fn port_name_uses_human_labels() {
        assert_eq!(port_name(PortKind::Cifti), "CIFTI");
        assert_eq!(port_name(PortKind::ParcelSelection), "Parcel Set");
        assert_eq!(port_name(PortKind::SurfaceScalars), "Surface Scalars");
        assert_eq!(port_name(PortKind::OdxCatalog), "ODX Catalog");
    }

    #[test]
    fn purifibre_outputs_have_distinct_labels() {
        let kind = WorkflowNodeKind::Purifibre {
            trim_fraction: 0.0,
            puri_fraction: 0.0,
            spherical_smoothing_deg: 0.0,
        };
        assert_eq!(
            output_port_label(&kind, 0, PortKind::Streamline),
            "Scored (all)",
        );
        assert_eq!(
            output_port_label(&kind, 1, PortKind::Streamline),
            "Filtered",
        );
    }
}
