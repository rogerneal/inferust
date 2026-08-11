//! Parity tests for IV2SLS (`Iv2Sls`).
//!
//! Reference is the same two-stage projection formula as ``src/gmm.rs``
//! (numpy / closed-form), including the intercept convention.

mod common;

use common::{
    as_f64, as_f64_matrix, as_f64_vec, assert_parity, check_scalar, check_vec, load_fixture,
};
use inferust::gmm::Iv2Sls;

#[test]
fn parity_iv2sls_small() {
    let fx = load_fixture("iv2sls_small");
    let ds = &fx["dataset"];
    let x = as_f64_matrix(&ds["x"]);
    let y = as_f64_vec(&ds["y"]);
    let z = as_f64_matrix(&ds["instruments"]);

    let result = Iv2Sls::new()
        .with_feature_names(vec!["x_endog".into()])
        .fit(&x, &y, &z)
        .expect("IV2SLS fit failed");

    assert_parity(
        "iv2sls_small",
        vec![
            check_vec(
                "params",
                &result.coefficients,
                &as_f64_vec(&fx["params"]),
                1e-8,
            ),
            check_vec("bse", &result.std_errors, &as_f64_vec(&fx["bse"]), 1e-8),
            check_scalar("ssr", result.ssr, as_f64(&fx["ssr"]), 1e-8),
            check_scalar("rsquared", result.r_squared, as_f64(&fx["rsquared"]), 1e-8),
        ],
    );
}
