//! Parity tests for contingency table statistics against statsmodels / scipy.

mod common;

use common::{as_f64, assert_parity, check_scalar, load_fixture};
use inferust::contingency::{mcnemar, odds_ratio_ci, table2x2};

// ── McNemar ───────────────────────────────────────────────────────────────────

#[test]
fn parity_mcnemar() {
    let fx = load_fixture("mcnemar");
    let tbl_raw = fx["dataset"]["table"]
        .as_array()
        .expect("table array")
        .iter()
        .map(|row| {
            let r: Vec<f64> = row
                .as_array()
                .expect("row array")
                .iter()
                .map(|v| v.as_f64().expect("f64"))
                .collect();
            r
        })
        .collect::<Vec<_>>();
    let table = [
        [tbl_raw[0][0], tbl_raw[0][1]],
        [tbl_raw[1][0], tbl_raw[1][1]],
    ];

    // statsmodels mcnemar uses continuity correction by default (correction=True)
    let res = mcnemar(table, true).expect("mcnemar failed");

    assert_parity(
        "mcnemar",
        vec![
            check_scalar("statistic", res.statistic, as_f64(&fx["statistic"]), 1e-6),
            check_scalar("p_value", res.p_value, as_f64(&fx["p_value"]), 1e-6),
        ],
    );
}

// ── Odds ratio ────────────────────────────────────────────────────────────────

#[test]
fn parity_odds_ratio() {
    let fx = load_fixture("odds_ratio");
    let tbl_raw = fx["dataset"]["table"]
        .as_array()
        .expect("table array")
        .iter()
        .map(|row| {
            let r: Vec<f64> = row
                .as_array()
                .expect("row array")
                .iter()
                .map(|v| v.as_f64().expect("f64"))
                .collect();
            r
        })
        .collect::<Vec<_>>();
    let table = [
        [tbl_raw[0][0], tbl_raw[0][1]],
        [tbl_raw[1][0], tbl_raw[1][1]],
    ];

    let res = table2x2(table).expect("table2x2 failed");
    let (ci_lo, ci_hi) = odds_ratio_ci(table, 0.05).expect("odds_ratio_ci failed");

    // scipy uses the sample (unconditional MLE) odds ratio; inferust uses the
    // same formula (ad/bc), so they agree to machine precision.
    assert_parity(
        "odds_ratio",
        vec![
            check_scalar(
                "odds_ratio",
                res.odds_ratio,
                as_f64(&fx["odds_ratio"]),
                1e-9,
            ),
            // CI method differs (Wald log-normal vs. scipy's Fisher exact CI),
            // so tolerance is loose.
            check_scalar("ci_lower", ci_lo, as_f64(&fx["ci_lower"]), 0.5),
            check_scalar("ci_upper", ci_hi, as_f64(&fx["ci_upper"]), 0.5),
        ],
    );
}
