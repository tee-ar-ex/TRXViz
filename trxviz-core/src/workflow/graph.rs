//! Pure (no-egui) workflow graph data model.
//!
//! `WorkflowGraph` is the canonical, serializable representation of a workflow:
//! a keyed set of nodes and a list of wires between them. It has no dependency
//! on `egui-snarl` or any other UI crate — the GUI editor maintains its own
//! view state and syncs to/from this type at frame boundaries.
//!
//! All topology is keyed by the stable `WorkflowNodeUuid`, so round-tripping
//! through the Snarl editor (or any other view) preserves identity.

use std::collections::{BTreeMap, btree_map};

use serde::{Deserialize, Serialize};

use super::{WorkflowNode, WorkflowNodeUuid};

/// Canvas position of a node in the graph editor. Kept in graph data so it
/// round-trips through saves; the GUI converts to/from `egui::Pos2`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GraphPos {
    pub x: f32,
    pub y: f32,
}

impl GraphPos {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// Axis-aligned rectangle in canvas coordinates. Used by seeding helpers to
/// report where new nodes were placed so the viewport can focus on them.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphRect {
    pub min: GraphPos,
    pub max: GraphPos,
}

impl GraphRect {
    pub const EMPTY: Self = Self {
        min: GraphPos::new(f32::INFINITY, f32::INFINITY),
        max: GraphPos::new(f32::NEG_INFINITY, f32::NEG_INFINITY),
    };

    pub fn from_points(points: impl IntoIterator<Item = GraphPos>) -> Self {
        let mut rect = Self::EMPTY;
        for p in points {
            rect.extend(p);
        }
        rect
    }

    pub fn extend(&mut self, p: GraphPos) {
        self.min.x = self.min.x.min(p.x);
        self.min.y = self.min.y.min(p.y);
        self.max.x = self.max.x.max(p.x);
        self.max.y = self.max.y.max(p.y);
    }

    pub fn is_finite(&self) -> bool {
        self.min.x.is_finite()
            && self.min.y.is_finite()
            && self.max.x.is_finite()
            && self.max.y.is_finite()
    }

    pub fn expanded(&self, dx: f32, dy: f32) -> Self {
        Self {
            min: GraphPos::new(self.min.x - dx, self.min.y - dy),
            max: GraphPos::new(self.max.x + dx, self.max.y + dy),
        }
    }
}

/// Output port on a node, identified by the producing node's UUID and the
/// output index on that node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OutPort {
    pub node: WorkflowNodeUuid,
    pub output: usize,
}

/// Input port on a node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InPort {
    pub node: WorkflowNodeUuid,
    pub input: usize,
}

/// A directed connection between an output port and an input port.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Wire {
    pub from: OutPort,
    pub to: InPort,
}

/// A node plus its editor-canvas position.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkflowNodeEntry {
    pub node: WorkflowNode,
    #[serde(default)]
    pub pos: GraphPos,
}

/// Canonical, serializable workflow graph.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WorkflowGraph {
    nodes: BTreeMap<WorkflowNodeUuid, WorkflowNodeEntry>,
    wires: Vec<Wire>,
}

impl WorkflowGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Insert a node at the given canvas position. Returns the node's UUID
    /// (taken from `node.uuid`). Panics in debug builds if a node with the
    /// same UUID already exists — callers are expected to allocate fresh
    /// UUIDs for new nodes via `WorkflowDocument::next_node_uuid`.
    pub fn insert_node(&mut self, node: WorkflowNode, pos: GraphPos) -> WorkflowNodeUuid {
        let uuid = node.uuid;
        debug_assert!(
            !self.nodes.contains_key(&uuid),
            "duplicate WorkflowNodeUuid inserted into WorkflowGraph"
        );
        self.nodes.insert(uuid, WorkflowNodeEntry { node, pos });
        uuid
    }

    /// Remove a node and every wire that touches it. Returns the removed
    /// node, or `None` if it was not present.
    pub fn remove_node(&mut self, uuid: WorkflowNodeUuid) -> Option<WorkflowNode> {
        let removed = self.nodes.remove(&uuid)?;
        self.wires
            .retain(|w| w.from.node != uuid && w.to.node != uuid);
        Some(removed.node)
    }

    /// Connect an output port to an input port. If a wire already exists into
    /// the same input, it is replaced (matching the editor's single-incoming
    /// semantics for typed inputs).
    pub fn connect(&mut self, from: OutPort, to: InPort) {
        self.wires.retain(|w| w.to != to);
        self.wires.push(Wire { from, to });
    }

    pub fn disconnect(&mut self, from: OutPort, to: InPort) {
        self.wires.retain(|w| !(w.from == from && w.to == to));
    }

    pub fn get(&self, uuid: WorkflowNodeUuid) -> Option<&WorkflowNode> {
        self.nodes.get(&uuid).map(|e| &e.node)
    }

    pub fn get_mut(&mut self, uuid: WorkflowNodeUuid) -> Option<&mut WorkflowNode> {
        self.nodes.get_mut(&uuid).map(|e| &mut e.node)
    }

    pub fn contains(&self, uuid: WorkflowNodeUuid) -> bool {
        self.nodes.contains_key(&uuid)
    }

    pub fn pos(&self, uuid: WorkflowNodeUuid) -> Option<GraphPos> {
        self.nodes.get(&uuid).map(|e| e.pos)
    }

    pub fn set_pos(&mut self, uuid: WorkflowNodeUuid, pos: GraphPos) {
        if let Some(entry) = self.nodes.get_mut(&uuid) {
            entry.pos = pos;
        }
    }

    /// Iterate (uuid, node) pairs in UUID order.
    pub fn nodes(&self) -> NodesIter<'_> {
        NodesIter {
            inner: self.nodes.iter(),
        }
    }

    /// Iterate (uuid, &mut node) pairs in UUID order.
    pub fn nodes_mut(&mut self) -> NodesIterMut<'_> {
        NodesIterMut {
            inner: self.nodes.iter_mut(),
        }
    }

    /// Iterate (uuid, entry) pairs in UUID order.
    pub fn entries(&self) -> btree_map::Iter<'_, WorkflowNodeUuid, WorkflowNodeEntry> {
        self.nodes.iter()
    }

    /// Iterate positions as (pos, uuid).
    pub fn nodes_pos(&self) -> impl Iterator<Item = (GraphPos, WorkflowNodeUuid)> + '_ {
        self.nodes.iter().map(|(uuid, entry)| (entry.pos, *uuid))
    }

    pub fn wires(&self) -> impl Iterator<Item = Wire> + '_ {
        self.wires.iter().copied()
    }

    /// Largest UUID currently present (as a raw `u64`), or zero if empty.
    /// Used when rebuilding `next_node_uuid` after loading a project.
    pub fn max_uuid(&self) -> u64 {
        self.nodes.keys().map(|u| u.0).max().unwrap_or(0)
    }
}

pub struct NodesIter<'a> {
    inner: btree_map::Iter<'a, WorkflowNodeUuid, WorkflowNodeEntry>,
}

impl<'a> Iterator for NodesIter<'a> {
    type Item = (WorkflowNodeUuid, &'a WorkflowNode);
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(uuid, entry)| (*uuid, &entry.node))
    }
}

pub struct NodesIterMut<'a> {
    inner: btree_map::IterMut<'a, WorkflowNodeUuid, WorkflowNodeEntry>,
}

impl<'a> Iterator for NodesIterMut<'a> {
    type Item = (WorkflowNodeUuid, &'a mut WorkflowNode);
    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|(uuid, entry)| (*uuid, &mut entry.node))
    }
}
