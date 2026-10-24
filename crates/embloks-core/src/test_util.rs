//! Helpers shared by the unit tests.

use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64;

use crate::materials::Cell;

pub use crate::scenes::outline;

/// pytest.approx(expected, rel=tolerance).
pub fn assert_rel(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs(),
        "{actual} is not within {tolerance} (relative) of {expected}"
    );
}

pub fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Cells (c, r) for c in c0..c1.
pub fn row(c0: usize, c1: usize, r: usize) -> Vec<Cell> {
    (c0..c1).map(|c| (c, r)).collect()
}

/// A rectangle's border cells, sorted (the battery tests' loop).
pub fn rectangle(c0: usize, c1: usize, r0: usize, r1: usize) -> Vec<Cell> {
    outline(c0, c1, r0, r1)
}

/// `n` points uniform in [0, size)^2.
pub fn uniform(seed: u64, n: usize, size: f64) -> Vec<[f64; 2]> {
    let mut rng = Pcg64::seed_from_u64(seed);
    (0..n).map(|_| [rng.random_range(0.0..size), rng.random_range(0.0..size)]).collect()
}

/// `n` standard normal samples (Box-Muller).
pub fn normal(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = Pcg64::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let (u, v): (f64, f64) = (rng.random_range(f64::EPSILON..1.0), rng.random());
            (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
        })
        .collect()
}
