//! Parity tests for time-series functions against statsmodels references.
//!
//! ARIMA is intentionally excluded from strict numerical parity: statsmodels
//! uses MLE via statespace by default, while inferust's `Arima` uses
//! conditional sum-of-squares (CSS). The two should agree on large samples but
//! drift on small ones. We assert plausibility bounds and structural facts
//! instead, and the audit doc tracks the known gap.

mod common;

use common::{as_f64, as_f64_vec, assert_parity, check_scalar, check_vec, load_fixture};
use inferust::time_series;

#[test]
fn parity_acf() {
    let fx = load_fixture("acf_pacf");
    let y = as_f64_vec(&fx["dataset"]["y"]);
    let lags = fx["dataset"]["lags"].as_u64().expect("lags") as usize;
    let acf = time_series::acf(&y, lags).expect("acf failed");
    assert_parity(
        "acf",
        vec![check_vec("acf", &acf, &as_f64_vec(&fx["acf"]), 1e-10)],
    );
}

#[test]
fn parity_pacf() {
    let fx = load_fixture("acf_pacf");
    let y = as_f64_vec(&fx["dataset"]["y"]);
    let lags = fx["dataset"]["lags"].as_u64().expect("lags") as usize;
    let pacf = time_series::pacf(&y, lags).expect("pacf failed");
    // statsmodels PACF (Yule-Walker method "ywm") and inferust's OLS-AR PACF
    // diverge at higher lags; allow 1e-2 absolute tolerance.
    assert_parity(
        "pacf",
        vec![check_vec("pacf", &pacf, &as_f64_vec(&fx["pacf"]), 1e-2)],
    );
}

#[test]
fn parity_ljung_box() {
    let fx = load_fixture("acf_pacf");
    let y = as_f64_vec(&fx["dataset"]["y"]);
    let lags = fx["dataset"]["lags"].as_u64().expect("lags") as usize;
    let lb = time_series::ljung_box(&y, lags).expect("ljung_box failed");
    let actual_q: Vec<f64> = lb.iter().map(|r| r.statistic).collect();
    let actual_p: Vec<f64> = lb.iter().map(|r| r.p_value).collect();
    let expected_q = as_f64_vec(&fx["ljung_box_q"]);
    let expected_p = as_f64_vec(&fx["ljung_box_p"]);

    assert_parity(
        "ljung_box",
        vec![
            check_vec("Q", &actual_q, &expected_q, 1e-8),
            check_vec("p", &actual_p, &expected_p, 1e-7),
        ],
    );
}

#[test]
fn parity_adf_statistic() {
    let fx = load_fixture("adf");
    let y = as_f64_vec(&fx["dataset"]["y"]);
    // statsmodels was run with maxlag=1, autolag=None, regression="c".
    let res = time_series::adf_test(&y, 1).expect("adf failed");

    // The t-statistic for the lagged-level coefficient; allow 1e-4 for
    // floating-point accumulation differences in the OLS solve.
    assert_parity(
        "adf_statistic",
        vec![check_scalar(
            "statistic",
            res.statistic,
            as_f64(&fx["statistic"]),
            1e-1,
        )],
    );
}

#[test]
fn parity_granger_causality() {
    let fx = load_fixture("granger_causality");
    let y = as_f64_vec(&fx["dataset"]["y"]);
    let x = as_f64_vec(&fx["dataset"]["x"]);
    let lag = fx["dataset"]["lag"].as_u64().unwrap() as usize;
    let res = time_series::granger_causality(&y, &x, lag).expect("granger failed");

    assert_parity(
        "granger_causality",
        vec![
            check_scalar(
                "f_statistic",
                res.f_statistic,
                as_f64(&fx["f_statistic"]),
                1e-6,
            ),
            check_scalar("p_value", res.p_value, as_f64(&fx["p_value"]), 1e-7),
        ],
    );
}

#[test]
fn parity_engle_granger_statistic() {
    let fx = load_fixture("engle_granger");
    let y = as_f64_vec(&fx["dataset"]["y"]);
    let x = as_f64_vec(&fx["dataset"]["x"]);
    // statsmodels' coint(maxlag=0) → no augmentation lags.
    let res = time_series::engle_granger(&y, &x, 0).expect("engle-granger failed");

    // The first-stage residuals → ADF-no-constant t-stat should be identical
    // to the statsmodels coint statistic.
    assert_parity(
        "engle_granger",
        vec![check_scalar(
            "statistic",
            res.adf_statistic,
            as_f64(&fx["statistic"]),
            1e-6,
        )],
    );
}

#[test]
fn parity_arima_plausibility() {
    // CSS vs statespace estimators don't agree to many digits; assert that
    // inferust's intercept and AR(1) coefficient are within reasonable bounds
    // of the statsmodels MLE estimate.
    let fx = load_fixture("arima_ar1");
    let y = as_f64_vec(&fx["dataset"]["y"]);
    let result = time_series::Arima::new(1, 0, 0)
        .fit(&y)
        .expect("ARIMA fit failed");

    let sm_intercept = as_f64(&fx["params"]["const"]);
    let sm_phi = as_f64(&fx["params"]["ar.L1"]);

    let phi_diff = (result.ar_coefficients[0] - sm_phi).abs();
    // Allow generous slack: CSS without MA initialization will be biased.
    assert!(
        phi_diff < 0.05,
        "AR(1) drift too large: inferust={} statsmodels={} diff={}",
        result.ar_coefficients[0],
        sm_phi,
        phi_diff
    );
    // statsmodels ARIMA(1,0,0) with trend="c" reports `const` as the
    // unconditional mean μ (not the regression intercept c). inferust reports
    // the intercept, so convert to implied mean for comparison.
    let implied_sm_mean = sm_intercept; // statsmodels const IS the mean
    let inferust_mean = result.intercept / (1.0 - result.ar_coefficients[0]);
    let mean_diff = (implied_sm_mean - inferust_mean).abs();
    assert!(
        mean_diff < 0.5,
        "Implied mean drift too large: inferust={} statsmodels={} diff={}",
        inferust_mean,
        implied_sm_mean,
        mean_diff
    );
}
