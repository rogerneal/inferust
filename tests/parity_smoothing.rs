//! Parity tests for exponential smoothing against
//! `statsmodels.tsa.holtwinters` fixtures.

mod common;

use common::{as_f64, as_f64_vec, assert_parity, check_scalar, check_vec, load_fixture};
use inferust::smoothing::{
    ExponentialSmoothing, SeasonalComponent, SimpleExpSmoothing, TrendComponent,
};

#[test]
fn parity_simple_exp_smoothing() {
    let fx = load_fixture("holt_winters");
    let y = as_f64_vec(&fx["dataset"]["y"]);
    let spec = &fx["ses_fixed"];

    let res = SimpleExpSmoothing::new()
        .smoothing_level(as_f64(&spec["alpha"]))
        .initial_level(as_f64(&spec["initial_level"]))
        .fit(&y)
        .expect("simple exponential smoothing failed");
    let forecast = res.forecast(8).expect("forecast failed");

    assert_parity(
        "simple_exp_smoothing",
        vec![
            check_vec(
                "fitted",
                &res.fitted_values,
                &as_f64_vec(&spec["fitted"]),
                1e-10,
            ),
            check_scalar("sse", res.sse, as_f64(&spec["sse"]), 1e-10),
            check_vec("forecast", &forecast, &as_f64_vec(&spec["forecast"]), 1e-10),
        ],
    );
}

fn fit_hw_fixed(
    y: &[f64],
    period: usize,
    spec: &serde_json::Value,
) -> inferust::smoothing::ExponentialSmoothingResult {
    ExponentialSmoothing::new()
        .with_trend(TrendComponent::Additive)
        .with_seasonal(SeasonalComponent::Additive, period)
        .smoothing_level(as_f64(&spec["alpha"]))
        .smoothing_trend(as_f64(&spec["beta"]))
        .smoothing_seasonal(as_f64(&spec["gamma"]))
        .initial_level(as_f64(&spec["initial_level"]))
        .initial_trend(as_f64(&spec["initial_trend"]))
        .initial_seasonal(as_f64_vec(&spec["initial_seasonal"]))
        .fit(y)
        .expect("Holt-Winters fit failed")
}

#[test]
fn parity_holt_winters_additive_fit() {
    let fx = load_fixture("holt_winters");
    let y = as_f64_vec(&fx["dataset"]["y"]);
    let period = fx["period"].as_u64().expect("period") as usize;
    let spec = &fx["hw_fixed"];
    let res = fit_hw_fixed(&y, period, spec);

    assert_parity(
        "holt_winters_additive_fit",
        vec![
            check_vec(
                "fitted",
                &res.fitted_values,
                &as_f64_vec(&spec["fitted"]),
                1e-10,
            ),
            check_scalar("sse", res.sse, as_f64(&spec["sse"]), 1e-10),
        ],
    );
}

/// Forecast parity for every horizon except the last step of each seasonal
/// cycle.
///
/// statsmodels reads its seasonal state window one step early, so at
/// `h % period == 0` it reuses the previous cycle's state for that phase
/// instead of the one updated by the final observation. `inferust` continues
/// the recursion, which is the standard Holt-Winters definition, so those
/// horizons differ deliberately. See `docs/parity.md` and
/// `holt_winters_cycle_end_uses_latest_seasonal_state` below.
#[test]
fn parity_holt_winters_additive_forecast() {
    let fx = load_fixture("holt_winters");
    let y = as_f64_vec(&fx["dataset"]["y"]);
    let period = fx["period"].as_u64().expect("period") as usize;
    let spec = &fx["hw_fixed"];
    let res = fit_hw_fixed(&y, period, spec);
    let forecast = res.forecast(2 * period).expect("forecast failed");
    let expected = as_f64_vec(&spec["forecast"]);

    let mut checks = Vec::new();
    for (i, (&got, &want)) in forecast.iter().zip(expected.iter()).enumerate() {
        let horizon = i + 1;
        if horizon % period == 0 {
            continue;
        }
        checks.push(check_scalar(
            &format!("forecast[h={horizon}]"),
            got,
            want,
            1e-10,
        ));
    }
    assert_parity("holt_winters_additive_forecast", checks);
}

/// The deliberate deviation, pinned down: with `gamma = 1` and a frozen level
/// the seasonal state is exactly `y_t - level`, so the forecast for the last
/// step of a cycle must reproduce the final observation's state rather than the
/// one from the cycle before it.
#[test]
fn holt_winters_cycle_end_uses_latest_seasonal_state() {
    let period = 4;
    let n = 24;
    // y_t = 10 + t, so with level pinned at 10 the seasonal state is s_t = t.
    let y: Vec<f64> = (0..n).map(|t| 10.0 + t as f64).collect();
    let res = ExponentialSmoothing::new()
        .with_trend(TrendComponent::Additive)
        .with_seasonal(SeasonalComponent::Additive, period)
        .smoothing_level(0.0)
        .smoothing_trend(0.0)
        .smoothing_seasonal(1.0)
        .initial_level(10.0)
        .initial_trend(0.0)
        .initial_seasonal(vec![0.0; period])
        .fit(&y)
        .expect("Holt-Winters fit failed");
    let forecast = res.forecast(period).expect("forecast failed");

    // Horizons 1..=4 continue phases 0..=3, whose latest states are s_20..s_23.
    let expected: Vec<f64> = vec![30.0, 31.0, 32.0, 33.0];
    assert_parity(
        "holt_winters_cycle_end",
        vec![check_vec("forecast", &forecast, &expected, 1e-10)],
    );
}
