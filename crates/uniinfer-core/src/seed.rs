//! Seed generation for stochastic inference pipelines.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

/// Returns a process-unique seed suitable for image or video sampling.
///
/// `RandomState` is seeded from OS entropy; the counter keeps concurrent
/// requests from colliding even when the OS entropy source is sampled at the
/// same instant.
pub fn random_seed() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut hasher = RandomState::new().build_hasher();
    hasher.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
    // Keep the value inside JavaScript's safe integer range so desktop
    // clients can display and reuse it without losing precision.
    hasher.finish() & ((1 << 53) - 1)
}

#[cfg(test)]
mod tests {
    use super::random_seed;

    #[test]
    fn seeds_differ_between_calls() {
        assert_ne!(random_seed(), random_seed());
    }
}
