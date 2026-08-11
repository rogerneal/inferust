//! Parity for panel FE / RE estimators and the Hausman test.
//!
//! * Entity FE matches `linearmodels.panel.PanelOLS(entity_effects=True)`.
//!   Within SEs match demean-then-OLS (inferust's path).
//! * Time FE and two-way FE match linearmodels with the same within-OLS SEs.
//! * RE matches `linearmodels.panel.RandomEffects` with an intercept
//!   (`cov_type="unadjusted"`, Swamy–Arora variance components).
//! * Hausman uses FE-within OLS covariance (same as inferust), not
//!   linearmodels' within-df-corrected FE SEs.

mod common;

use common::{as_f64, as_f64_vec, assert_parity, check_scalar, check_vec, load_fixture, xy};
use inferust::panel::{hausman_fe_re, PanelOls};

fn ids_from(fx: &serde_json::Value, key: &str) -> Vec<usize> {
    fx["dataset"][key]
        .as_array()
        .unwrap_or_else(|| panic!("{key}"))
        .iter()
        .map(|v| v.as_u64().expect("id") as usize)
        .collect()
}

fn check_within_ols(name: &str, result: &inferust::regression::OlsResult, fx: &serde_json::Value) {
    assert_parity(
        name,
        vec![
            check_vec(
                "params",
                &result.coefficients,
                &as_f64_vec(&fx["params"]),
                1e-10,
            ),
            check_vec("bse", &result.std_errors, &as_f64_vec(&fx["bse"]), 1e-10),
            check_vec(
                "tvalues",
                &result.t_statistics,
                &as_f64_vec(&fx["tvalues"]),
                1e-8,
            ),
            check_vec(
                "pvalues",
                &result.p_values,
                &as_f64_vec(&fx["pvalues"]),
                1e-8,
            ),
            check_scalar("rsquared", result.r_squared, as_f64(&fx["rsquared"]), 1e-10),
        ],
    );
}

#[test]
fn parity_panel_entity_fe() {
    let fx = load_fixture("panel_fe");
    let (x, y) = xy(&fx);
    let entities = ids_from(&fx, "entities");
    let k = x[0].len();
    let names: Vec<String> = (1..=k).map(|i| format!("x{i}")).collect();

    let result = PanelOls::new()
        .with_feature_names(names)
        .fit_entity_fe(&x, &y, &entities)
        .expect("panel FE fit failed");
    check_within_ols("panel_fe", &result, &fx);
}

#[test]
fn parity_panel_time_fe() {
    let fx = load_fixture("panel_time_fe");
    let (x, y) = xy(&fx);
    let times = ids_from(&fx, "times");
    let k = x[0].len();
    let names: Vec<String> = (1..=k).map(|i| format!("x{i}")).collect();

    let result = PanelOls::new()
        .with_feature_names(names)
        .fit_time_fe(&x, &y, &times)
        .expect("panel time FE fit failed");
    check_within_ols("panel_time_fe", &result, &fx);
}

#[test]
fn parity_panel_two_way_fe() {
    let fx = load_fixture("panel_two_way_fe");
    let (x, y) = xy(&fx);
    let entities = ids_from(&fx, "entities");
    let times = ids_from(&fx, "times");
    let k = x[0].len();
    let names: Vec<String> = (1..=k).map(|i| format!("x{i}")).collect();

    let result = PanelOls::new()
        .with_feature_names(names)
        .fit_two_way_fe(&x, &y, &entities, &times)
        .expect("panel two-way FE fit failed");
    check_within_ols("panel_two_way_fe", &result, &fx);
}

#[test]
fn parity_panel_random_effects() {
    let fx = load_fixture("panel_re");
    let (x, y) = xy(&fx);
    let entities = ids_from(&fx, "entities");
    let k = x[0].len();
    let names: Vec<String> = (1..=k).map(|i| format!("x{i}")).collect();

    let result = PanelOls::new()
        .with_feature_names(names)
        .fit_random_effects(&x, &y, &entities)
        .expect("panel RE fit failed");

    assert_parity(
        "panel_re",
        vec![
            check_vec(
                "params",
                &result.coefficients,
                &as_f64_vec(&fx["params"]),
                1e-10,
            ),
            check_vec("bse", &result.std_errors, &as_f64_vec(&fx["bse"]), 1e-10),
            check_vec(
                "tvalues",
                &result.t_statistics,
                &as_f64_vec(&fx["tvalues"]),
                1e-8,
            ),
            check_vec(
                "pvalues",
                &result.p_values,
                &as_f64_vec(&fx["pvalues"]),
                1e-8,
            ),
            check_scalar("rsquared", result.r_squared, as_f64(&fx["rsquared"]), 1e-8),
            check_scalar("sigma2_e", result.sigma2_e, as_f64(&fx["sigma2_e"]), 1e-10),
            check_scalar("sigma2_u", result.sigma2_u, as_f64(&fx["sigma2_u"]), 1e-10),
            check_vec("theta", &result.theta, &as_f64_vec(&fx["theta"]), 1e-10),
        ],
    );
}

#[test]
fn parity_hausman_fe_re() {
    let fx = load_fixture("panel_re");
    let (x, y) = xy(&fx);
    let entities = ids_from(&fx, "entities");
    let fe = PanelOls::new()
        .fit_entity_fe(&x, &y, &entities)
        .expect("FE");
    let re = PanelOls::new()
        .fit_random_effects(&x, &y, &entities)
        .expect("RE");
    let h = hausman_fe_re(&fe, &re).expect("Hausman");

    assert_parity(
        "hausman",
        vec![
            check_scalar(
                "statistic",
                h.statistic,
                as_f64(&fx["hausman_statistic"]),
                1e-8,
            ),
            check_scalar("df", h.df as f64, as_f64(&fx["hausman_df"]), 1e-12),
            check_scalar("p_value", h.p_value, as_f64(&fx["hausman_p_value"]), 1e-8),
        ],
    );
}
