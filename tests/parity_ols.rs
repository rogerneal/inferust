//! Parity tests for OLS / WLS against statsmodels reference fixtures.
//!
//! Tolerances:
//! * params, fitted, residuals, R², SSR, etc.: 1e-8 — closed-form linear algebra,
//!   any larger diff is a real bug.
//! * F, AIC/BIC: 1e-6 — accumulates extra subtractions and logs.
//! * HC1–HC3 standard errors: 1e-8 — also closed form.
//! * Influence (Cook's, studentized): 1e-7 — extra divisions, slightly looser.

mod common;

use common::{
    as_f64, as_f64_matrix, as_f64_vec, assert_parity, check_scalar, check_vec, load_fixture, xy,
};
use inferust::regression::{Ols, OlsCovariance, Wls};

fn feature_names(k: usize) -> Vec<String> {
    (1..=k).map(|i| format!("x{i}")).collect()
}

fn run_ols_fixture(name: &str) {
    let fx = load_fixture(name);
    let (x, y) = xy(&fx);
    let k = x[0].len();

    let result = Ols::new()
        .with_feature_names(feature_names(k))
        .fit(&x, &y)
        .expect("OLS fit failed");

    let expected_params = as_f64_vec(&fx["params"]);
    let expected_bse = as_f64_vec(&fx["bse"]);
    let expected_t = as_f64_vec(&fx["tvalues"]);
    let expected_p = as_f64_vec(&fx["pvalues"]);
    let expected_fitted = as_f64_vec(&fx["fittedvalues"]);
    let expected_resid = as_f64_vec(&fx["resid"]);

    let ci_actual: Vec<f64> = result
        .confidence_intervals(0.05)
        .expect("CI")
        .into_iter()
        .flat_map(|(lo, hi)| [lo, hi])
        .collect();
    let ci_expected: Vec<f64> = as_f64_matrix(&fx["conf_int_05"])
        .into_iter()
        .flatten()
        .collect();

    let diagnostics = result.diagnostics().ok();
    let influence = result.influence();

    let mut checks = vec![
        check_vec("params", &result.coefficients, &expected_params, 1e-8),
        check_vec("bse", &result.std_errors, &expected_bse, 1e-8),
        check_vec("tvalues", &result.t_statistics, &expected_t, 1e-7),
        check_vec("pvalues", &result.p_values, &expected_p, 1e-7),
        check_vec("conf_int_05", &ci_actual, &ci_expected, 1e-7),
        check_vec("fittedvalues", &result.fitted_values, &expected_fitted, 1e-8),
        check_vec("resid", &result.residuals, &expected_resid, 1e-8),
        check_scalar("rsquared", result.r_squared, as_f64(&fx["rsquared"]), 1e-10),
        check_scalar(
            "rsquared_adj",
            result.adj_r_squared,
            as_f64(&fx["rsquared_adj"]),
            1e-10,
        ),
        check_scalar("fvalue", result.f_statistic, as_f64(&fx["fvalue"]), 1e-6),
        check_scalar("f_pvalue", result.f_p_value, as_f64(&fx["f_pvalue"]), 1e-7),
        check_scalar("aic", result.aic, as_f64(&fx["aic"]), 1e-6),
        check_scalar("bic", result.bic, as_f64(&fx["bic"]), 1e-6),
        check_scalar("ssr", result.ssr, as_f64(&fx["ssr"]), 1e-8),
        check_scalar("ess", result.ess, as_f64(&fx["ess"]), 1e-8),
        check_scalar(
            "centered_tss",
            result.centered_tss,
            as_f64(&fx["centered_tss"]),
            1e-8,
        ),
        check_scalar(
            "mse_resid",
            result.mse_resid,
            as_f64(&fx["mse_resid"]),
            1e-10,
        ),
    ];

    // Influence diagnostics
    let inf_expected = &fx["influence"];
    let expected_hat = as_f64_vec(&inf_expected["hat_diag"]);
    let expected_cooks = as_f64_vec(&inf_expected["cooks_d"]);
    let expected_studentized = as_f64_vec(&inf_expected["resid_studentized_internal"]);
    checks.push(check_vec("hat_diag", &influence.leverage, &expected_hat, 1e-8));
    checks.push(check_vec(
        "cooks_d",
        &influence.cooks_distance,
        &expected_cooks,
        1e-7,
    ));
    checks.push(check_vec(
        "resid_studentized_internal",
        &influence.studentized_residuals,
        &expected_studentized,
        1e-7,
    ));

    // Diagnostics scalars (loose tolerance — JB, skew, kurtosis use slightly
    // different small-sample definitions in some implementations).
    if let Some(d) = diagnostics {
        // statsmodels reports these but we don't pull them into the fixture
        // currently; just check they're finite.
        assert!(d.durbin_watson.is_finite(), "DW should be finite");
        assert!(d.skewness.is_finite());
        assert!(d.kurtosis.is_finite());
    }

    assert_parity(name, checks);
}

#[test]
fn parity_ols_small() {
    run_ols_fixture("ols_small");
}

#[test]
fn parity_ols_medium() {
    run_ols_fixture("ols_medium");
}

fn run_hc_fixture(name: &str, cov: OlsCovariance) {
    let fx = load_fixture(name);
    let (x, y) = xy(&fx);
    let k = x[0].len();
    let result = Ols::new()
        .with_feature_names(feature_names(k))
        .with_covariance(cov)
        .fit(&x, &y)
        .expect("OLS HC fit failed");

    assert_parity(
        name,
        vec![
            check_vec(
                "params",
                &result.coefficients,
                &as_f64_vec(&fx["params"]),
                1e-9,
            ),
            check_vec("bse", &result.std_errors, &as_f64_vec(&fx["bse"]), 1e-8),
            // statsmodels switches to z (normal) for HC; inferust does too via
            // `uses_t_distribution = false`. Compare directly.
            check_vec(
                "tvalues_or_zvalues",
                &result.t_statistics,
                &as_f64_vec(&fx["tvalues"]),
                1e-7,
            ),
            check_vec(
                "pvalues",
                &result.p_values,
                &as_f64_vec(&fx["pvalues"]),
                1e-7,
            ),
        ],
    );
}

#[test]
fn parity_ols_hc0() {
    run_hc_fixture("ols_hc0", OlsCovariance::Hc0);
}

#[test]
fn parity_ols_hc1() {
    run_hc_fixture("ols_hc1", OlsCovariance::Hc1);
}

#[test]
fn parity_ols_hc2() {
    run_hc_fixture("ols_hc2", OlsCovariance::Hc2);
}

#[test]
fn parity_ols_hc3() {
    run_hc_fixture("ols_hc3", OlsCovariance::Hc3);
}

#[test]
fn parity_wls() {
    let fx = load_fixture("wls_small");
    let (x, y) = xy(&fx);
    let weights = as_f64_vec(&fx["dataset"]["weights"]);
    let k = x[0].len();
    let result = Wls::new()
        .with_feature_names(feature_names(k))
        .fit(&x, &y, &weights)
        .expect("WLS fit failed");

    assert_parity(
        "wls_small",
        vec![
            check_vec(
                "params",
                &result.coefficients,
                &as_f64_vec(&fx["params"]),
                1e-8,
            ),
            check_vec("bse", &result.std_errors, &as_f64_vec(&fx["bse"]), 1e-8),
            check_vec(
                "tvalues",
                &result.t_statistics,
                &as_f64_vec(&fx["tvalues"]),
                1e-7,
            ),
            check_vec(
                "pvalues",
                &result.p_values,
                &as_f64_vec(&fx["pvalues"]),
                1e-7,
            ),
            check_scalar(
                "rsquared",
                result.r_squared,
                as_f64(&fx["rsquared"]),
                1e-9,
            ),
            check_scalar(
                "rsquared_adj",
                result.adj_r_squared,
                as_f64(&fx["rsquared_adj"]),
                1e-9,
            ),
            check_scalar("fvalue", result.f_statistic, as_f64(&fx["fvalue"]), 1e-6),
            check_scalar("ssr", result.ssr, as_f64(&fx["ssr"]), 1e-8),
        ],
    );
}
