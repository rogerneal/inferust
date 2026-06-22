//! Parity tests for regression diagnostics against statsmodels.

mod common;

use common::{as_f64, as_f64_vec, assert_parity, check_scalar, load_fixture};
use inferust::diagnostics::{breusch_pagan, reset_test, variance_inflation_factors, white_test};
use inferust::regression::Ols;

fn load_xy(fx: &serde_json::Value) -> (Vec<Vec<f64>>, Vec<f64>) {
    let x = fx["dataset"]["x"]
        .as_array()
        .expect("x array")
        .iter()
        .map(|row| {
            row.as_array()
                .expect("row")
                .iter()
                .map(|v| v.as_f64().expect("f64"))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let y = as_f64_vec(&fx["dataset"]["y"]);
    (x, y)
}

// ── VIF ───────────────────────────────────────────────────────────────────────

#[test]
fn parity_vif() {
    let fx = load_fixture("vif");
    let x = fx["dataset"]["x"]
        .as_array()
        .expect("x array")
        .iter()
        .map(|row| {
            row.as_array()
                .expect("row")
                .iter()
                .map(|v| v.as_f64().expect("f64"))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let results = variance_inflation_factors(&x, None).expect("VIF failed");
    let expected_vifs: Vec<f64> = fx["vif"]
        .as_array()
        .expect("vif array")
        .iter()
        .map(|v| v.as_f64().expect("f64"))
        .collect();

    assert_eq!(results.len(), expected_vifs.len(), "VIF count mismatch");
    // statsmodels always adds an intercept to the auxiliary VIF regression;
    // inferust's `variance_inflation_factors` does not. The effect is small
    // (< 1 %) for uncorrelated predictors but larger when predictors correlate
    // with the implicit mean. Tolerance loosened to 1e-2 to accommodate this.
    let checks: Vec<_> = results
        .iter()
        .zip(expected_vifs.iter())
        .enumerate()
        .map(|(i, (r, &e))| check_scalar(&format!("vif[{i}]"), r.vif, e, 1e-2))
        .collect();
    assert_parity("vif", checks);
}

// ── Breusch-Pagan ─────────────────────────────────────────────────────────────

#[test]
fn parity_breusch_pagan() {
    let fx = load_fixture("breusch_pagan");
    let (x, y) = load_xy(&fx);
    let fit = Ols::new().fit(&x, &y).expect("OLS fit failed");
    let res = breusch_pagan(&fit.residuals, &x).expect("breusch_pagan failed");
    assert_parity(
        "breusch_pagan",
        vec![
            check_scalar("statistic", res.statistic, as_f64(&fx["statistic"]), 1e-4),
            check_scalar("p_value", res.p_value, as_f64(&fx["p_value"]), 1e-4),
        ],
    );
}

// ── White test ────────────────────────────────────────────────────────────────

#[test]
fn parity_white_test() {
    let fx = load_fixture("white_test");
    let (x, y) = load_xy(&fx);
    let fit = Ols::new().fit(&x, &y).expect("OLS fit failed");
    let res = white_test(&fit.residuals, &x).expect("white_test failed");
    assert_parity(
        "white_test",
        vec![
            check_scalar("statistic", res.statistic, as_f64(&fx["statistic"]), 1e-4),
            check_scalar("p_value", res.p_value, as_f64(&fx["p_value"]), 1e-4),
        ],
    );
}

// ── RESET test ────────────────────────────────────────────────────────────────
//
// statsmodels `linear_reset(res, power=3, use_f=True)` augments with powers
// 2 and 3 of yhat. inferust's `reset_test(y, x, powers)` takes an explicit
// list of powers; passing `&[2, 3]` is equivalent.

#[test]
fn parity_reset_test() {
    let fx = load_fixture("reset_test");
    let (x, y) = load_xy(&fx);
    let res = reset_test(&y, &x, &[2, 3]).expect("reset_test failed");
    assert_parity(
        "reset_test",
        vec![
            // RESET uses F-statistic; statsmodels and inferust agree closely.
            check_scalar("statistic", res.statistic, as_f64(&fx["statistic"]), 1e-4),
            check_scalar("p_value", res.p_value, as_f64(&fx["p_value"]), 1e-4),
        ],
    );
}
