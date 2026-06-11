//! Generic draw-primitive registry — the first slice of the migration
//! away from `SceneFramePlan`'s parallel `Vec<FooDrawPlan>` fields.
//!
//! A display op contributes a unit of rendering work by pushing a value
//! that implements [`DrawPrimitive`] into the frame's [`DrawList`],
//! instead of `SceneFramePlan` growing a bespoke typed field per
//! visualization kind (and that field then having to be threaded through
//! the `Default` impl and every consumer). The render backend — GUI
//! (the sibling `trxviz` crate) or headless — iterates the list and
//! downcasts to the concrete plan types it knows how to upload.
//!
//! Only fixel draws use this today (see `ops::fixel_display`). The other
//! draw types still live in typed `SceneFramePlan` fields and can move
//! over one at a time behind the same trait, without a big-bang rewrite.
//!
//! The trait is deliberately a pure storage/downcast marker (`as_any` +
//! `clone_box`). Fingerprinting — "what triggers a re-upload" — is NOT on
//! the trait, because it varies too much across op families to fit one
//! signature: a voxel-mask plan carries its mesh hash in a field, a fixel
//! plan hashes a couple of fields, and a bundle's upload identity is a
//! multi-gate check folding in a *separate* boundary-field cache plus
//! per-node run/runtime state (a single `u64` can't express it). So each
//! backend computes the fingerprint it needs per draw kind — often paired
//! with [`UploadCache`] — and the trait stays out of it.

use std::any::Any;
use std::collections::HashMap;

/// A unit of rendering work emitted during workflow evaluation — a
/// heterogeneous handle the render backend downcasts to a concrete
/// `*DrawPlan` in order to upload it.
///
/// This is intentionally just a storage/downcast marker. The GPU upload,
/// and the fingerprinting that decides whether to skip it, stay in the
/// (renderer- and backend-specific) sync loop, which recovers the
/// concrete type via [`DrawPrimitive::as_any`]. See the module docs for
/// why fingerprinting is deliberately not a trait method.
pub trait DrawPrimitive: 'static {
    /// Recover the concrete plan type for renderer-specific upload.
    fn as_any(&self) -> &dyn Any;

    /// Clone into a boxed trait object. Required because `SceneFramePlan`
    /// (which owns the `DrawList`) derives `Clone`. Every implementor is
    /// itself `Clone`, so the body is always `Box::new(self.clone())`.
    fn clone_box(&self) -> Box<dyn DrawPrimitive>;
}

impl Clone for Box<dyn DrawPrimitive> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Ordered, heterogeneous list of draw primitives for one evaluated
/// frame — the registry that replaces `SceneFramePlan`'s per-kind
/// `*_draws` fields. Display ops push in evaluation order; backends
/// recover the typed plans they know how to upload with
/// [`DrawList::of_type`].
#[derive(Default, Clone)]
pub struct DrawList {
    items: Vec<Box<dyn DrawPrimitive>>,
}

impl DrawList {
    /// Append a draw primitive. Called by display ops during `evaluate`.
    pub fn push<P: DrawPrimitive>(&mut self, plan: P) {
        self.items.push(Box::new(plan));
    }

    /// Every primitive of concrete type `P`, in push order. This is how
    /// a render backend recovers the typed plans it knows how to upload.
    pub fn of_type<P: DrawPrimitive>(&self) -> impl Iterator<Item = &P> {
        self.items
            .iter()
            .filter_map(|b| b.as_any().downcast_ref::<P>())
    }
}

/// Opaque key for a slot in the GPU [`UploadCache`]. `kind` namespaces a
/// draw family (e.g. `"voxel_mask"`); `id` distinguishes draws within it
/// (typically a `draw_id`), or is a fixed constant for single-slot
/// families. A new draw family picks its own `kind` — there is no central
/// registry to edit.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct UploadSlot {
    pub kind: &'static str,
    pub id: u64,
}

impl UploadSlot {
    pub fn new(kind: &'static str, id: u64) -> Self {
        Self { kind, id }
    }
}

/// Tracks the fingerprint of what each draw last uploaded to the GPU so a
/// render backend can skip re-uploading unchanged geometry. One of these
/// on the backend state replaces the per-draw-type `uploaded_*` fields
/// that otherwise accumulate (a scalar per single-slot family, a HashMap
/// per keyed family); a new drawing op reuses it via its own
/// [`UploadSlot`] kind instead of growing the state struct.
#[derive(Default, Clone)]
pub struct UploadCache {
    seen: HashMap<UploadSlot, u64>,
}

impl UploadCache {
    /// True when `slot` was last uploaded at exactly `fingerprint` — i.e.
    /// the upload can be skipped this frame.
    pub fn is_current(&self, slot: UploadSlot, fingerprint: u64) -> bool {
        self.seen.get(&slot) == Some(&fingerprint)
    }

    /// Record that `slot` has been uploaded at `fingerprint`.
    pub fn record(&mut self, slot: UploadSlot, fingerprint: u64) {
        self.seen.insert(slot, fingerprint);
    }

    /// Forget a slot whose GPU resource was cleared (so a later draw with
    /// the same key re-uploads).
    pub fn forget(&mut self, slot: UploadSlot) {
        self.seen.remove(&slot);
    }
}
