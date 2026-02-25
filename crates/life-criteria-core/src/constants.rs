/// Largest valid world dimension (world units). Prevents overflow in spatial index.
pub const MAX_WORLD_SIZE: f64 = 2048.0;

/// Prime multiplier used to derive per-organism RNG streams from a base seed.
/// Chosen so streams for consecutive organism IDs have minimal overlap.
pub const RNG_DERIVATION_PRIME: u64 = 7919;

/// Maximum number of genome pairs sampled when estimating population diversity.
/// Caps the O(n²) pairwise computation; sampling is deterministic per step.
pub const GENOME_DIVERSITY_MAX_PAIRS: usize = 50;
