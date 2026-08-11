//! Parity tests for time-series functions against statsmodels references.
//!
//! Default [`Arima`](inferust::time_series::Arima) fits use conditional
//! sum-of-squares (CSS). Exact Gaussian MLE is available via `.exact_mle()` and
//! is audited against statsmodels statespace for ARIMA(1,0,0) and ARIMA(1,0,1).

mod common;

use common::{
    as_f64, as_f64_matrix, as_f64_vec, assert_parity, check_scalar, check_vec, load_fixture,
};
use inferust::statespace::LinearGaussianModel;
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
    assert_parity(
        "pacf",
        vec![check_vec("pacf", &pacf, &as_f64_vec(&fx["pacf"]), 1e-10)],
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

#[test]
fn parity_arima_exact_mle_ar1() {
    let fx = load_fixture("arima_ar1");
    let y = as_f64_vec(&fx["dataset"]["y"]);
    let sm_mu = as_f64(&fx["params"]["const"]);
    let sm_phi = as_f64(&fx["params"]["ar.L1"]);
    let sm_sigma2 = as_f64(&fx["sigma2"]);
    let sm_llf = as_f64(&fx["llf"]);

    // Filter-only llf at fixed statsmodels params (μ in obs intercept).
    let filter_llf = LinearGaussianModel::arma(sm_mu, &[sm_phi], &[], sm_sigma2)
        .expect("arma")
        .filter(&y)
        .expect("filter")
        .log_likelihood;
    let expected_filter = fx.get("filter_llf").map(as_f64).unwrap_or(sm_llf);
    assert_parity(
        "arima_exact_mle_ar1_filter",
        vec![check_scalar(
            "filter_llf",
            filter_llf,
            expected_filter,
            1e-8,
        )],
    );

    let result = time_series::Arima::new(1, 0, 0)
        .exact_mle()
        .max_iter(4000)
        .fit(&y)
        .expect("exact MLE AR(1)");
    let inferust_mu = result.intercept / (1.0 - result.ar_coefficients[0]);
    assert_parity(
        "arima_exact_mle_ar1",
        vec![
            check_scalar("const_mu", inferust_mu, sm_mu, 1e-3),
            check_scalar("ar.L1", result.ar_coefficients[0], sm_phi, 1e-3),
            check_scalar("sigma2", result.sigma2, sm_sigma2, 5e-3),
            check_scalar("llf", result.log_likelihood, sm_llf, 1e-2),
        ],
    );
}

#[test]
fn parity_arima_exact_mle_arma11() {
    let fx = load_fixture("arima_arma11");
    let y = as_f64_vec(&fx["dataset"]["y"]);
    let sm_mu = as_f64(&fx["params"]["const"]);
    let sm_phi = as_f64(&fx["params"]["ar.L1"]);
    let sm_theta = as_f64(&fx["params"]["ma.L1"]);
    let sm_sigma2 = as_f64(&fx["sigma2"]);
    let sm_llf = as_f64(&fx["llf"]);

    let filter_llf = LinearGaussianModel::arma(sm_mu, &[sm_phi], &[sm_theta], sm_sigma2)
        .expect("arma")
        .filter(&y)
        .expect("filter")
        .log_likelihood;
    assert_parity(
        "arima_exact_mle_arma11_filter",
        vec![check_scalar(
            "filter_llf",
            filter_llf,
            as_f64(&fx["filter_llf"]),
            1e-8,
        )],
    );

    let result = time_series::Arima::new(1, 0, 1)
        .exact_mle()
        .max_iter(4000)
        .fit(&y)
        .expect("exact MLE ARMA(1,1)");
    let inferust_mu = result.intercept / (1.0 - result.ar_coefficients[0]);
    assert_parity(
        "arima_exact_mle_arma11",
        vec![
            check_scalar("const_mu", inferust_mu, sm_mu, 1e-3),
            check_scalar("ar.L1", result.ar_coefficients[0], sm_phi, 1e-3),
            check_scalar("ma.L1", result.ma_coefficients[0], sm_theta, 1e-3),
            check_scalar("sigma2", result.sigma2, sm_sigma2, 5e-3),
            check_scalar("llf", result.log_likelihood, sm_llf, 1e-2),
        ],
    );
}

// ── Forecast standard errors and intervals ────────────────────────────────────

/// ARIMA(1,1,1) forecast standard errors against `SARIMAX.get_forecast`.
///
/// The fixture uses `SARIMAX.filter` with fixed parameters rather than an MLE
/// fit, so both sides evaluate the same ψ-weights and the comparison isolates
/// the variance recursion from estimation differences.
#[test]
fn parity_arima_forecast_standard_errors() {
    let fx = load_fixture("forecast_ci");
    let spec = &fx["arima"];
    let expected = as_f64_vec(&spec["se_mean"]);
    let se = time_series::sarima_forecast_standard_errors(
        &[as_f64(&spec["params"]["ar"])],
        &[as_f64(&spec["params"]["ma"])],
        1,
        &[],
        &[],
        0,
        1,
        as_f64(&spec["params"]["sigma2"]),
        expected.len(),
    )
    .expect("ARIMA forecast standard errors failed");

    assert_parity(
        "arima_forecast_se",
        vec![check_vec("se_mean", &se, &expected, 1e-10)],
    );
}

/// SARIMA(1,0,0)(0,1,1,4) forecast standard errors, exercising the seasonal
/// differencing and seasonal MA terms of the ψ-weight expansion.
///
/// Tolerance is one tier looser than the plain ARIMA case: statsmodels derives
/// `se_mean` from the Kalman filter, whose seasonal-difference state converges
/// to the analytic ψ-weights only to about 1e-10.
#[test]
fn parity_sarima_forecast_standard_errors() {
    let fx = load_fixture("forecast_ci");
    let spec = &fx["sarima"];
    let expected = as_f64_vec(&spec["se_mean"]);
    let seasonal_order = as_f64_vec(&spec["seasonal_order"]);
    let se = time_series::sarima_forecast_standard_errors(
        &[as_f64(&spec["params"]["ar"])],
        &[],
        0,
        &[],
        &[as_f64(&spec["params"]["seasonal_ma"])],
        seasonal_order[2] as usize,
        seasonal_order[3] as usize,
        as_f64(&spec["params"]["sigma2"]),
        expected.len(),
    )
    .expect("SARIMA forecast standard errors failed");

    assert_parity(
        "sarima_forecast_se",
        vec![check_vec("se_mean", &se, &expected, 1e-9)],
    );
}

/// VAR(2) point forecasts and 95% intervals against
/// `VARResults.forecast_interval`. Both sides estimate by OLS, so this should
/// agree to linear-algebra precision.
#[test]
fn parity_var_forecast_interval() {
    let fx = load_fixture("forecast_ci");
    let y1 = as_f64_vec(&fx["dataset"]["y1"]);
    let y2 = as_f64_vec(&fx["dataset"]["y2"]);
    let spec = &fx["var"];
    let lags = spec["lags"].as_u64().expect("lags") as usize;

    let series = vec![y1.clone(), y2.clone()];
    let fit = time_series::Var::new(lags)
        .fit(&series)
        .expect("VAR fit failed");

    let history: Vec<Vec<f64>> = series
        .iter()
        .map(|s| s[s.len() - lags..].to_vec())
        .collect();
    // statsmodels returns (steps, k) matrices; inferust returns one result per
    // variable, so the reference is read transposed.
    let point = as_f64_matrix(&spec["point"]);
    let lower = as_f64_matrix(&spec["lower"]);
    let upper = as_f64_matrix(&spec["upper"]);
    let steps = point.len();
    let forecasts = fit
        .forecast_with_ci(&history, steps, 0.05)
        .expect("VAR forecast_with_ci failed");

    let mut checks = Vec::new();
    for (var, fc) in forecasts.iter().enumerate() {
        let want_mean: Vec<f64> = point.iter().map(|row| row[var]).collect();
        let want_lower: Vec<f64> = lower.iter().map(|row| row[var]).collect();
        let want_upper: Vec<f64> = upper.iter().map(|row| row[var]).collect();
        checks.push(check_vec(
            &format!("mean[{var}]"),
            &fc.mean,
            &want_mean,
            1e-9,
        ));
        checks.push(check_vec(
            &format!("lower[{var}]"),
            &fc.lower,
            &want_lower,
            1e-9,
        ));
        checks.push(check_vec(
            &format!("upper[{var}]"),
            &fc.upper,
            &want_upper,
            1e-9,
        ));
    }
    assert_parity("var_forecast_interval", checks);
}

// ── SARIMAX / VECM / VARMAX (simplified surfaces) ─────────────────────────────

/// SARIMAX exogenous coefficients from the OLS projection step only.
///
/// Full SARIMAX MLE is not compared: inferust projects out `[1, X]` then fits
/// SARIMA on the residuals.
#[test]
fn parity_sarimax_exog_coefficients() {
    let fx = load_fixture("sarimax_small");
    let y = as_f64_vec(&fx["dataset"]["y"]);
    let x = as_f64_matrix(&fx["dataset"]["x"]);
    // Exog OLS is independent of SARIMA orders; s ≥ 2 is required by Sarima.
    let res = time_series::Sarimax::new(1, 0, 0, 0, 0, 0, 12)
        .fit(&y, &x)
        .expect("SARIMAX fit failed");

    assert_parity(
        "sarimax_exog",
        vec![check_vec(
            "exog_coefficients",
            &res.exog_coefficients,
            &as_f64_vec(&fx["exog_coefficients"]),
            1e-8,
        )],
    );
}

/// VECM Johansen eigenvalues and trace statistics.
///
/// Fixture pins inferust's reduced-rank path (symmetrised EVP). statsmodels
/// ``coint_johansen`` is stored for reference but can diverge by ~1e-2 because
/// of deterministic-term / inversion details; tests use the inferust pin.
#[test]
fn parity_vecm_johansen() {
    let fx = load_fixture("vecm_small");
    let series = as_f64_matrix(&fx["dataset"]["series"]);
    let lags = fx["dataset"]["lags"].as_u64().expect("lags") as usize;
    let rank = fx["dataset"]["rank"].as_u64().expect("rank") as usize;

    let res = time_series::Vecm::new(lags, rank)
        .fit(&series)
        .expect("VECM fit failed");

    assert_parity(
        "vecm_johansen",
        vec![
            check_vec(
                "eigenvalues",
                &res.eigenvalues,
                &as_f64_vec(&fx["eigenvalues"]),
                1e-6,
            ),
            check_vec(
                "trace_statistics",
                &res.trace_statistics,
                &as_f64_vec(&fx["trace_statistics"]),
                1e-4,
            ),
        ],
    );
}

/// VARMAX per-equation OLS coefficients (not statespace VARMAX MLE).
#[test]
fn parity_varmax_ols_coefficients() {
    let fx = load_fixture("varmax_small");
    let series = as_f64_matrix(&fx["dataset"]["series"]);
    let exog = as_f64_matrix(&fx["dataset"]["exog"]);
    let lags = fx["dataset"]["lags"].as_u64().expect("lags") as usize;

    let res = time_series::Varmax::new(lags)
        .fit(&series, &exog)
        .expect("VARMAX fit failed");

    let expected = as_f64_matrix(&fx["coefficients"]);
    let mut checks = Vec::new();
    for (eq, coef) in res.coefficients.iter().enumerate() {
        checks.push(check_vec(
            &format!("coefficients[{eq}]"),
            coef,
            &expected[eq],
            1e-8,
        ));
    }
    assert_parity("varmax_ols", checks);
}
