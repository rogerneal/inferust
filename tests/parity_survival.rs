//! Parity test for Cox PH against statsmodels.PHReg with Breslow ties.

mod common;

use common::{
    as_f64, as_f64_matrix, as_f64_vec, assert_parity, check_scalar, check_vec, load_fixture,
};
use inferust::survival::{log_rank_test, CoxPh, KaplanMeier};

// ── Kaplan-Meier ──────────────────────────────────────────────────────────────

#[test]
fn parity_kaplan_meier() {
    let fx = load_fixture("kaplan_meier");
    let times = as_f64_vec(&fx["dataset"]["times"]);
    let events_u: Vec<usize> = fx["dataset"]["events"]
        .as_array()
        .expect("events")
        .iter()
        .map(|v| v.as_u64().expect("event u64") as usize)
        .collect();

    let result = KaplanMeier::new()
        .fit(&times, &events_u)
        .expect("KaplanMeier::fit failed");

    // Compare total event / censored counts
    let n_events_expected = fx["n_events"].as_u64().expect("n_events") as usize;
    let n_censored_expected = fx["n_censored"].as_u64().expect("n_censored") as usize;
    assert_eq!(result.n_events, n_events_expected, "n_events mismatch");
    assert_eq!(
        result.n - result.n_events,
        n_censored_expected,
        "n_censored mismatch"
    );

    // Compare survival probabilities at fixture checkpoints
    let checkpoints = fx["checkpoints"].as_array().expect("checkpoints");
    let mut checks = Vec::new();
    for cp in checkpoints {
        let t = as_f64(&cp["time"]);
        let expected_s = as_f64(&cp["survival"]);
        let actual_s = result.survival_at(t);
        checks.push(check_scalar(
            &format!("survival_at({t:.4})"),
            actual_s,
            expected_s,
            1e-6,
        ));
    }
    assert_parity("kaplan_meier", checks);
}

// ── Log-rank test ─────────────────────────────────────────────────────────────

#[test]
fn parity_log_rank() {
    let fx = load_fixture("log_rank");
    let times = as_f64_vec(&fx["dataset"]["times"]);
    let events_all: Vec<usize> = fx["dataset"]["events"]
        .as_array()
        .expect("events")
        .iter()
        .map(|v| v.as_u64().expect("event u64") as usize)
        .collect();
    let x = as_f64_vec(&fx["dataset"]["x"]);
    let median_x = as_f64(&fx["dataset"]["median_x"]);

    // Split by median covariate (same split as Python)
    let (mut times1, mut events1, mut times2, mut events2) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..times.len() {
        if x[i] <= median_x {
            times1.push(times[i]);
            events1.push(events_all[i]);
        } else {
            times2.push(times[i]);
            events2.push(events_all[i]);
        }
    }

    let result = log_rank_test(&times1, &events1, &times2, &events2).expect("log_rank_test failed");

    assert_parity(
        "log_rank",
        vec![
            check_scalar(
                "statistic",
                result.statistic,
                as_f64(&fx["statistic"]),
                1e-4,
            ),
            check_scalar("p_value", result.p_value, as_f64(&fx["p_value"]), 1e-4),
        ],
    );
}

// ── Cox PH ────────────────────────────────────────────────────────────────────

#[test]
fn parity_cox_ph() {
    let fx = load_fixture("cox_ph");
    let times = as_f64_vec(&fx["dataset"]["times"]);
    let events_u: Vec<usize> = fx["dataset"]["events"]
        .as_array()
        .expect("events")
        .iter()
        .map(|v| v.as_u64().expect("event u64") as usize)
        .collect();
    let x = as_f64_matrix(&fx["dataset"]["x"]);

    let result = CoxPh::new()
        .fit(&times, &events_u, &x)
        .expect("CoxPh::fit failed");

    assert_parity(
        "cox_ph",
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
            check_scalar("llf", result.log_likelihood, as_f64(&fx["llf"]), 1e-5),
        ],
    );
}
