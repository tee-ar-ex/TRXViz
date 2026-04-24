//! Shared CPU-tractography scaffolding: trait + outer loop that every
//! direction getter plugs into. Keeps the algorithm-specific code in
//! each `dg_*.rs` focused on "how do you pick the next direction" while
//! the outer seed-attempt loop, per-thread accumulator, per-step and
//! post-hoc mask orchestration, SplitMix+LCG RNG, and bidirectional
//! assembly all live here once.
//!
//! Non-goals for this module:
//!   - **Seeding strategy.** Yeh (rejection-sample with a target cap) and
//!     Dipy (enumerate `seed_mask` × `seeds_per_voxel`) have different
//!     outer-loop shapes. Each CPU runner owns its own seeding driver and
//!     calls `try_one_attempt` per seed.
//!   - **GPU.** The GPU pipelines live in `crate::gpu::dipy` and talk
//!     directly to wgpu; they don't use this trait (the shader is the
//!     DG on that path).

pub mod accum;
pub mod cancel;
pub mod direction_getter;
pub mod masks;
pub mod rng;
pub mod tracker;

pub mod dg_prob;
pub mod dg_yeh;

pub use cancel::CancelFlag;
