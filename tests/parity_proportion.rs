//! Parity tests for proportion inference against
//! `statsmodels.stats.proportion` fixtures.

mod common;

use common::{as_f64, as_f64_vec, assert_parity, check_scalar, load_fixture};
use inferust::power::Alternative;
use inferust::proportion::{
    proportion_confint, proportion_effectsize, proportions_ztest, ConfintMethod,
};

fn as_u64(v: &serde_json::Value) -> u64 {
    v.as_u64()
        .unwrap_or_else(|| panic!("expected u64, got {v:?}"))
}

fn as_u64_vec(v: &serde_json::Value) -> Vec<u64> {
    v.as_array()
        .unwrap_or_else(|| panic!("expected array, got {v:?}"))
        .iter()
        .map(as_u64)
        .collect()
}

#[test]
fn parity_proportions_ztest_one_sample() {
    let fx = load_fixture("proportion");
    let one = &fx["one_sample"];
    let res = proportions_ztest(
        &[as_u64(&one["count"])],
        &[as_u64(&one["nobs"])],
        Some(as_f64(&one["value"])),
        Alternative::TwoSided,
    )
    .expect("one-sample proportions_ztest failed");

    let larger = &fx["one_sample_larger"];
    let res_larger = proportions_ztest(
        &[as_u64(&one["count"])],
        &[as_u64(&one["nobs"])],
        Some(as_f64(&one["value"])),
        Alternative::Larger,
    )
    .expect("one-sided proportions_ztest failed");

    assert_parity(
        "proportions_ztest_one_sample",
        vec![
            check_scalar("z", res.statistic, as_f64(&one["z"]), 1e-12),
            check_scalar("p", res.p_value, as_f64(&one["p"]), 1e-12),
            check_scalar(
                "z_larger",
                res_larger.statistic,
                as_f64(&larger["z"]),
                1e-12,
            ),
            check_scalar("p_larger", res_larger.p_value, as_f64(&larger["p"]), 1e-12),
        ],
    );
}

#[test]
fn parity_proportions_ztest_two_sample() {
    let fx = load_fixture("proportion");
    let two = &fx["two_sample"];
    let counts = as_u64_vec(&two["counts"]);
    let nobs = as_u64_vec(&two["nobs"]);
    let res = proportions_ztest(&counts, &nobs, None, Alternative::TwoSided)
        .expect("two-sample proportions_ztest failed");

    assert_parity(
        "proportions_ztest_two_sample",
        vec![
            check_scalar("z", res.statistic, as_f64(&two["z"]), 1e-12),
            check_scalar("p", res.p_value, as_f64(&two["p"]), 1e-12),
        ],
    );
}

#[test]
fn parity_proportion_confint() {
    let fx = load_fixture("proportion");
    let count = as_u64(&fx["one_sample"]["count"]);
    let nobs = as_u64(&fx["one_sample"]["nobs"]);

    let mut checks = Vec::new();
    for (key, method) in [
        ("normal", ConfintMethod::Normal),
        ("wilson", ConfintMethod::Wilson),
        ("clopper_pearson", ConfintMethod::ClopperPearson),
        ("agresti_coull", ConfintMethod::AgrestiCoull),
        ("jeffreys", ConfintMethod::Jeffreys),
    ] {
        let (lo, hi) =
            proportion_confint(count, nobs, 0.05, method).expect("proportion_confint failed");
        let expected = as_f64_vec(&fx["confint"][key]);
        checks.push(check_scalar(
            &format!("{key}.lower"),
            lo,
            expected[0],
            1e-10,
        ));
        checks.push(check_scalar(
            &format!("{key}.upper"),
            hi,
            expected[1],
            1e-10,
        ));
    }
    assert_parity("proportion_confint", checks);
}

#[test]
fn parity_proportion_effectsize() {
    let fx = load_fixture("proportion");
    let got = proportion_effectsize(0.6, 0.45).expect("proportion_effectsize failed");
    assert_parity(
        "proportion_effectsize",
        vec![check_scalar("h", got, as_f64(&fx["effectsize"]), 1e-12)],
    );
}
