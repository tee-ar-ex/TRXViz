//! Deterministic per-attempt RNG used by both the Yeh and Dipy CPU trackers.
//!
//! We don't bring in `rand` here: the tracker needs (a) cheap per-attempt
//! seeding from `(plan.rng_seed, attempt_idx)` with a well-dispersed state
//! so adjacent indices don't produce correlated streamlines, and (b) two
//! trivial uniform-number primitives (`f32` in `[0, 1)` and `u32`). A
//! vanilla LCG satisfies both with no dependency footprint. The SplitMix
//! constants below are the canonical mixing avalanche from Stafford /
//! Murmur3's finalizer — they're what you'd find in any split-mix RNG.
//!
//! Why not `SmallRng`: it's overkill for this usage, and the existing Yeh
//! benchmarks baseline against exactly this LCG sequence. Preserving the
//! sequence means the refactor is bit-identical for any given seed.

/// Initial state for attempt `attempt_idx` of a run seeded by `rng_seed`.
///
/// Two multiply-adds (SplitMix-style) produce a u64 state uncorrelated
/// with the input order so `(seed=42, idx=0)` and `(seed=42, idx=1)`
/// produce completely different streamlines. This matches what the old
/// `cpu_yeh.rs` / `cpu_dipy.rs` did inline at the top of each attempt;
/// lifting it here means the constants live in one place.
#[inline]
pub fn split_mix_init(rng_seed: u64, attempt_idx: u64) -> u64 {
    rng_seed
        .wrapping_add(0x9E3779B97F4A7C15)
        .wrapping_mul(0xBF58476D1CE4E5B9)
        .wrapping_add(attempt_idx.wrapping_mul(0x94D049BB133111EB))
}

/// Advance the LCG and return the high 32 bits as a `u32`. Truncated
/// high-bits output — low bits of a power-of-2-modulus LCG are known to
/// have short cycles, so we always shift right.
#[inline]
pub fn lcg_u32(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

/// Advance the LCG and return a uniform `f32` in `[0, 1)`. The high 32
/// bits of the state are divided by `2^32` so the returned value spans
/// the full unit interval.
#[inline]
pub fn lcg_f32(state: &mut u64) -> f32 {
    (lcg_u32(state) as f32) / 4_294_967_296.0
}
