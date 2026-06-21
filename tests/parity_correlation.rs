//! Parity tests for correlation against scipy.stats reference fixtures.

mod common;

use common::{as_f64, as_f64_vec, assert_parity, check_scalar, load_fixture};
use inferust::correlation;

#[test]
fn parity_pearson() {
    let fx = load_fixture("pearson_spearman");
    let x = as_f64_vec(&fx["dataset"]["x"]);
    let y = as_f64_vec(&fx["dataset"]["y"]);
    let r = correlation::pearson(&x, &y).expect("pearson failed");

    assert_parity(
        "pearson",
        vec![check_scalar("r", r, as_f64(&fx["pearson_r"]), 1e-10)],
    );
}

#[test]
fn parity_spearman() {
    let fx = load_fixture("pearson_spearman");
    let x = as_f64_vec(&fx["dataset"]["x"]);
    let y = as_f64_vec(&fx["dataset"]["y"]);
    let r = correlation::spearman(&x, &y).expect("spearman failed");

    assert_parity(
        "spearman",
        vec![check_scalar("r", r, as_f64(&fx["spearman_r"]), 1e-8)],
    );
}
