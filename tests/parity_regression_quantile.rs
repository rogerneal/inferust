//! Parity tests for quantile regression against statsmodels QuantReg fixtures.
//!
//! Tolerances:
//! * params: 1e-4 — IRLS convergence creates per-iteration float drift.
//! * pseudo_r1: 1e-4 — derived from converged params.
//!
//! Note: statsmodels QuantReg uses a different bandwidth formula for standard
//! errors (Hall-Sheather by default), so we only compare params and pseudo_r1
//! here; bse / z / p are not in the fixture.

mod common;

use common::{as_f64, as_f64_vec, assert_parity, check_scalar, check_vec, load_fixture, xy};
use inferust::regression::QuantileRegression;

fn run_quantreg_fixture(name: &str, q: f64) {
    let fx = load_fixture(name);
    let (x, y) = xy(&fx);
    let k = x[0].len();
    let feature_names: Vec<String> = (1..=k).map(|i| format!("x{i}")).collect();

    let result = QuantileRegression::new(q)
        .with_feature_names(feature_names)
        .fit(&x, &y)
        .expect("QuantileRegression fit failed");

    assert_parity(
        name,
        vec![
            check_vec(
                "params",
                &result.coefficients,
                &as_f64_vec(&fx["params"]),
                1e-4,
            ),
            check_scalar("pseudor2", result.pseudo_r1, as_f64(&fx["pseudor2"]), 1e-4),
        ],
    );
}

#[test]
fn parity_quantreg_median() {
    run_quantreg_fixture("quantreg_median", 0.5);
}

#[test]
fn parity_quantreg_q25() {
    run_quantreg_fixture("quantreg_q25", 0.25);
}
