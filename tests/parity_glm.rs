//! Parity tests for GLM (Logit, Poisson) against statsmodels.
//!
//! GLMs use Newton/IRLS iteration so we use a looser tolerance (1e-5 on
//! coefficients) than OLS. statsmodels also reports `tvalues` on Logit/Poisson
//! result objects, but those are actually z-statistics (normal-distribution
//! inference); inferust returns `z_statistics` directly.

mod common;

use common::{as_f64, as_f64_vec, assert_parity, check_scalar, check_vec, load_fixture, xy};
use inferust::glm::{Gamma, GammaLink, Logistic, Poisson};

fn feature_names(k: usize) -> Vec<String> {
    (1..=k).map(|i| format!("x{i}")).collect()
}

#[test]
fn parity_logit_small() {
    let fx = load_fixture("logit_small");
    let (x, y) = xy(&fx);
    let k = x[0].len();
    let result = Logistic::new()
        .with_feature_names(feature_names(k))
        .max_iter(200)
        .tolerance(1e-10)
        .fit(&x, &y)
        .expect("Logit fit failed");

    assert_parity(
        "logit_small",
        vec![
            check_vec(
                "params",
                &result.coefficients,
                &as_f64_vec(&fx["params"]),
                1e-5,
            ),
            check_vec("bse", &result.std_errors, &as_f64_vec(&fx["bse"]), 1e-5),
            check_vec(
                "zvalues",
                &result.z_statistics,
                &as_f64_vec(&fx["zvalues"]),
                1e-4,
            ),
            check_vec(
                "pvalues",
                &result.p_values,
                &as_f64_vec(&fx["pvalues"]),
                1e-4,
            ),
            check_scalar("llf", result.log_likelihood, as_f64(&fx["llf"]), 1e-6),
            check_scalar(
                "llnull",
                result.null_log_likelihood,
                as_f64(&fx["llnull"]),
                1e-8,
            ),
            check_scalar(
                "prsquared",
                result.pseudo_r_squared,
                as_f64(&fx["prsquared"]),
                1e-7,
            ),
            check_scalar("aic", result.aic, as_f64(&fx["aic"]), 1e-5),
            check_scalar("bic", result.bic, as_f64(&fx["bic"]), 1e-5),
        ],
    );
}

#[test]
fn parity_poisson_small() {
    let fx = load_fixture("poisson_small");
    let (x, y) = xy(&fx);
    let k = x[0].len();
    let result = Poisson::new()
        .with_feature_names(feature_names(k))
        .max_iter(200)
        .tolerance(1e-10)
        .fit(&x, &y)
        .expect("Poisson fit failed");

    assert_parity(
        "poisson_small",
        vec![
            check_vec(
                "params",
                &result.coefficients,
                &as_f64_vec(&fx["params"]),
                1e-5,
            ),
            check_vec("bse", &result.std_errors, &as_f64_vec(&fx["bse"]), 1e-5),
            check_vec(
                "zvalues",
                &result.z_statistics,
                &as_f64_vec(&fx["zvalues"]),
                1e-4,
            ),
            check_vec(
                "pvalues",
                &result.p_values,
                &as_f64_vec(&fx["pvalues"]),
                1e-4,
            ),
            check_scalar("llf", result.log_likelihood, as_f64(&fx["llf"]), 1e-6),
            check_scalar(
                "llnull",
                result.null_log_likelihood,
                as_f64(&fx["llnull"]),
                1e-7,
            ),
            check_scalar("aic", result.aic, as_f64(&fx["aic"]), 1e-5),
            check_scalar("bic", result.bic, as_f64(&fx["bic"]), 1e-5),
            check_vec(
                "fittedvalues",
                &result.fitted_values,
                &as_f64_vec(&fx["fittedvalues"]),
                1e-5,
            ),
        ],
    );
}

/// Fit both the `InversePower` (canonical) and `Log` links on the same
/// fixture dataset and compare each against its sub-object in `gamma_glm`.
#[test]
fn parity_gamma_glm() {
    let fx = load_fixture("gamma_glm");
    let (x, y) = xy(&fx);
    let k = x[0].len();

    let inverse_power = Gamma::new()
        .with_feature_names(feature_names(k))
        .max_iter(200)
        .tolerance(1e-10)
        .fit(&x, &y)
        .expect("Gamma (InversePower) fit failed");
    check_gamma_link(&inverse_power, &fx["inverse_power"], "inverse_power");

    let log_link = Gamma::new()
        .with_feature_names(feature_names(k))
        .with_link(GammaLink::Log)
        .max_iter(200)
        .tolerance(1e-10)
        .fit(&x, &y)
        .expect("Gamma (Log) fit failed");
    check_gamma_link(&log_link, &fx["log"], "log");
}

fn check_gamma_link(
    result: &inferust::glm::GammaResult,
    expected: &serde_json::Value,
    label: &str,
) {
    let intervals = result
        .fitted_mean_intervals(0.05)
        .expect("fitted_mean_intervals failed");
    let actual_mean: Vec<f64> = intervals.iter().map(|p| p.mean).collect();
    let actual_lower: Vec<f64> = intervals.iter().map(|p| p.lower).collect();
    let actual_upper: Vec<f64> = intervals.iter().map(|p| p.upper).collect();

    // statsmodels' `summary_frame()` reports the two raw Wald bounds
    // unordered for the decreasing `InversePower` link (lower > upper); sort
    // them here the same way inferust's `gamma_prediction_intervals` does.
    let raw_lower = as_f64_vec(&expected["mean_ci_lower"]);
    let raw_upper = as_f64_vec(&expected["mean_ci_upper"]);
    let expected_lower: Vec<f64> = raw_lower
        .iter()
        .zip(raw_upper.iter())
        .map(|(&a, &b)| a.min(b))
        .collect();
    let expected_upper: Vec<f64> = raw_lower
        .iter()
        .zip(raw_upper.iter())
        .map(|(&a, &b)| a.max(b))
        .collect();

    assert_parity(
        &format!("gamma_glm.{label}"),
        vec![
            check_vec(
                "params",
                &result.coefficients,
                &as_f64_vec(&expected["params"]),
                1e-5,
            ),
            check_vec(
                "bse",
                &result.std_errors,
                &as_f64_vec(&expected["bse"]),
                1e-5,
            ),
            check_scalar("llf", result.log_likelihood, as_f64(&expected["llf"]), 1e-5),
            check_scalar(
                "llnull",
                result.null_log_likelihood,
                as_f64(&expected["llnull"]),
                1e-5,
            ),
            check_scalar(
                "deviance",
                result.deviance,
                as_f64(&expected["deviance"]),
                1e-4,
            ),
            check_scalar(
                "pearson_chi2",
                result.pearson_chi_squared,
                as_f64(&expected["pearson_chi2"]),
                1e-4,
            ),
            check_scalar("scale", result.dispersion, as_f64(&expected["scale"]), 1e-4),
            check_scalar("aic", result.aic, as_f64(&expected["aic"]), 1e-4),
            check_scalar("bic_llf", result.bic, as_f64(&expected["bic_llf"]), 1e-4),
            check_vec("mean", &actual_mean, &as_f64_vec(&expected["mean"]), 1e-4),
            check_vec("mean_ci_lower", &actual_lower, &expected_lower, 1e-3),
            check_vec("mean_ci_upper", &actual_upper, &expected_upper, 1e-3),
        ],
    );
}
