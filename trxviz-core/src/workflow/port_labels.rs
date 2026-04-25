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
//! each pin) and the docs generator (SVG workflow diagrams). Keep the
//! two call sites in lockstep by routing both through this module.

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
        PortKind::Cifti => "CIFTI",
        PortKind::SurfaceScalars => "Surface Scalars",
        PortKind::VolumeScalars => "Volume Scalars",
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
        WorkflowNodeKind::OdxVolumeSelect { .. } => match output_index {
            0 => "Volume".to_string(),
            1 => "Volume Scalars".to_string(),
            _ => port_name(port).to_string(),
        },
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
