//! Parity tests for descriptive statistics.
//!
//! inferust's `Summary` uses the population definition of skewness/kurtosis
//! (`bias=True` in scipy), which is what we record in the fixture. Excess
//! kurtosis (Fisher) is used on both sides.

mod common;

use common::{as_f64, as_f64_vec, assert_parity, check_scalar, load_fixture};
use inferust::descriptive::Summary;

#[test]
fn parity_descriptive() {
    let fx = load_fixture("descriptive");
    let data = as_f64_vec(&fx["dataset"]["data"]);
    let s = Summary::new(&data).expect("Summary::new failed");

    assert_parity(
        "descriptive",
        vec![
            check_scalar("mean", s.mean, as_f64(&fx["mean"]), 1e-12),
            check_scalar("std", s.std, as_f64(&fx["std"]), 1e-12),
            check_scalar("variance", s.variance, as_f64(&fx["variance"]), 1e-12),
            check_scalar("min", s.min, as_f64(&fx["min"]), 0.0),
            check_scalar("max", s.max, as_f64(&fx["max"]), 0.0),
            check_scalar("q1", s.q1, as_f64(&fx["q1"]), 1e-12),
            check_scalar("median", s.median, as_f64(&fx["median"]), 1e-12),
            check_scalar("q3", s.q3, as_f64(&fx["q3"]), 1e-12),
            check_scalar("skewness", s.skewness, as_f64(&fx["skewness"]), 1e-10),
            check_scalar("kurtosis", s.kurtosis, as_f64(&fx["kurtosis"]), 1e-10),
        ],
    );
}
