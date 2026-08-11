//! Parity tests for GaussianGam truncated-power splines.
//!
//! The fixture builds the same design matrix as ``src/gam.rs`` ``build_design``
//! and fits OLS with an intercept via statsmodels. Params / bse are closed-form.
//! Penalized GCV uses the same truncated-power design + log λ grid as Rust
//! (not statsmodels GLMGam, which uses different bases).

mod common;

use common::{as_f64, as_f64_vec, assert_parity, check_scalar, check_vec, load_fixture, xy};
use inferust::gam::{GaussianGam, SplineTerm};

#[test]
fn parity_gam_small() {
    let fx = load_fixture("gam_small");
    let (x, y) = xy(&fx);
    let knots = as_f64_vec(&fx["dataset"]["knots"]);

    let result = GaussianGam::new()
        .smooth(SplineTerm::cubic(0, knots).named("s(x)"))
        .fit(&x, &y)
        .expect("GAM fit failed");

    assert_parity(
        "gam_small",
        vec![
            check_vec(
                "params",
                &result.ols.coefficients,
                &as_f64_vec(&fx["params"]),
                1e-8,
            ),
            check_vec("bse", &result.ols.std_errors, &as_f64_vec(&fx["bse"]), 1e-8),
            check_scalar(
                "rsquared",
                result.ols.r_squared,
                as_f64(&fx["rsquared"]),
                1e-10,
            ),
        ],
    );
}

#[test]
fn parity_gam_penalized_gcv() {
    let fx = load_fixture("gam_penalized_gcv");
    let (x, y) = xy(&fx);
    let knots = as_f64_vec(&fx["dataset"]["knots"]);

    let result = GaussianGam::new()
        .smooth(SplineTerm::cubic(0, knots).named("s(x)"))
        .penalized()
        .fit(&x, &y)
        .expect("penalized GAM fit failed");

    assert_parity(
        "gam_penalized_gcv",
        vec![
            check_vec(
                "params",
                &result.ols.coefficients,
                &as_f64_vec(&fx["params"]),
                1e-8,
            ),
            check_scalar(
                "lambda",
                result.lambda.expect("lambda"),
                as_f64(&fx["lambda"]),
                1e-10,
            ),
            check_scalar("edf", result.edf.expect("edf"), as_f64(&fx["edf"]), 1e-8),
            check_scalar("gcv", result.gcv.expect("gcv"), as_f64(&fx["gcv"]), 1e-8),
            check_scalar("ssr", result.ols.ssr, as_f64(&fx["ssr"]), 1e-8),
        ],
    );
}
