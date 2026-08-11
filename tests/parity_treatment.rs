//! Parity tests for ``PropensityScore::ipw``.
//!
//! Propensity params match statsmodels ``Logit``; ATE/ATT use the same
//! clamped-IPW formulas as ``src/treatment.rs``.

mod common;

use common::{
    as_f64, as_f64_matrix, as_f64_vec, assert_parity, check_scalar, check_vec, load_fixture,
};
use inferust::treatment::PropensityScore;

#[test]
fn parity_treatment_ipw_small() {
    let fx = load_fixture("treatment_ipw_small");
    let ds = &fx["dataset"];
    let x = as_f64_matrix(&ds["x"]);
    let treatment = as_f64_vec(&ds["treatment"]);
    let outcome = as_f64_vec(&ds["outcome"]);
    let k = x[0].len();

    let result = PropensityScore::new()
        .with_feature_names((1..=k).map(|i| format!("x{i}")).collect())
        .ipw(&x, &treatment, &outcome)
        .expect("IPW failed");

    assert_parity(
        "treatment_ipw_small",
        vec![
            check_vec(
                "params",
                &result.propensity_model.coefficients,
                &as_f64_vec(&fx["params"]),
                1e-5,
            ),
            check_vec(
                "propensity_scores",
                &result.propensity_scores,
                &as_f64_vec(&fx["propensity_scores"]),
                1e-5,
            ),
            check_scalar("ate", result.ate, as_f64(&fx["ate"]), 1e-5),
            check_scalar("att", result.att, as_f64(&fx["att"]), 1e-5),
            check_scalar(
                "treated_mean",
                result.treated_mean,
                as_f64(&fx["treated_mean"]),
                1e-5,
            ),
            check_scalar(
                "control_mean",
                result.control_mean,
                as_f64(&fx["control_mean"]),
                1e-5,
            ),
        ],
    );
}
