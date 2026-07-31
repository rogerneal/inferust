//! Parity tests for power analysis against `statsmodels.stats.power` fixtures.

mod common;

use common::{as_f64, assert_parity, check_scalar, load_fixture};
use inferust::power::{Alternative, FTestAnovaPower, NormalIndPower, TTestIndPower, TTestPower};

fn alternative(v: &serde_json::Value) -> Alternative {
    match v.as_str().expect("alternative should be a string") {
        "two-sided" => Alternative::TwoSided,
        "larger" => Alternative::Larger,
        "smaller" => Alternative::Smaller,
        other => panic!("unknown alternative {other}"),
    }
}

#[test]
fn parity_ttest_power() {
    let fx = load_fixture("power");
    let cases = fx["ttest_power"]
        .as_array()
        .expect("ttest_power should be an array");
    let mut checks = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        let got = TTestPower::new()
            .power(
                as_f64(&c["effect_size"]),
                as_f64(&c["nobs"]),
                as_f64(&c["alpha"]),
                alternative(&c["alternative"]),
            )
            .expect("ttest power failed");
        checks.push(check_scalar(
            &format!("power[{i}]"),
            got,
            as_f64(&c["power"]),
            1e-9,
        ));
    }
    assert_parity("ttest_power", checks);
}

#[test]
fn parity_ttest_ind_power() {
    let fx = load_fixture("power");
    let cases = fx["ttest_ind_power"]
        .as_array()
        .expect("ttest_ind_power should be an array");
    let mut checks = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        let got = TTestIndPower::new()
            .power(
                as_f64(&c["effect_size"]),
                as_f64(&c["nobs1"]),
                as_f64(&c["alpha"]),
                as_f64(&c["ratio"]),
                alternative(&c["alternative"]),
            )
            .expect("ttest_ind power failed");
        checks.push(check_scalar(
            &format!("power[{i}]"),
            got,
            as_f64(&c["power"]),
            1e-9,
        ));
    }
    assert_parity("ttest_ind_power", checks);
}

#[test]
fn parity_normal_ind_power() {
    let fx = load_fixture("power");
    let cases = fx["normal_ind_power"]
        .as_array()
        .expect("normal_ind_power should be an array");
    let mut checks = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        let got = NormalIndPower::new()
            .power(
                as_f64(&c["effect_size"]),
                as_f64(&c["nobs1"]),
                as_f64(&c["alpha"]),
                as_f64(&c["ratio"]),
                alternative(&c["alternative"]),
            )
            .expect("normal_ind power failed");
        checks.push(check_scalar(
            &format!("power[{i}]"),
            got,
            as_f64(&c["power"]),
            1e-12,
        ));
    }
    assert_parity("normal_ind_power", checks);
}

#[test]
fn parity_ftest_anova_power() {
    let fx = load_fixture("power");
    let cases = fx["ftest_anova_power"]
        .as_array()
        .expect("ftest_anova_power should be an array");
    let mut checks = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        let got = FTestAnovaPower::new()
            .power(
                as_f64(&c["effect_size"]),
                as_f64(&c["nobs"]),
                as_f64(&c["alpha"]),
                as_f64(&c["k_groups"]),
            )
            .expect("anova power failed");
        checks.push(check_scalar(
            &format!("power[{i}]"),
            got,
            as_f64(&c["power"]),
            1e-9,
        ));
    }
    assert_parity("ftest_anova_power", checks);
}

#[test]
fn parity_solve_ttest_ind_nobs() {
    let fx = load_fixture("power");
    let c = &fx["solve_ttest_ind_nobs"];
    let got = TTestIndPower::new()
        .solve_nobs(
            as_f64(&c["effect_size"]),
            as_f64(&c["power"]),
            as_f64(&c["alpha"]),
            as_f64(&c["ratio"]),
            alternative(&c["alternative"]),
        )
        .expect("solve_nobs failed");

    assert_parity(
        "solve_ttest_ind_nobs",
        vec![check_scalar("nobs1", got, as_f64(&c["nobs1"]), 1e-6)],
    );
}
