//! Parity tests for GEE against statsmodels.

mod common;

use common::{as_f64_vec, assert_parity, check_vec, load_fixture};
use inferust::gee::{Gee, GeeFamily, WorkingCorrelation};

#[test]
fn parity_gee_small() {
    let fx = load_fixture("gee_small");
    let ds = &fx["dataset"];
    let x: Vec<Vec<f64>> = serde_json::from_value(ds["x"].clone()).expect("x matrix");
    let y: Vec<f64> = ds["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let groups: Vec<usize> = ds["groups"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let k = x[0].len();

    let result = Gee::new(GeeFamily::Poisson)
        .with_feature_names((1..=k).map(|i| format!("x{i}")).collect())
        .with_working_correlation(WorkingCorrelation::Exchangeable)
        .max_iter(20)
        .fit(&x, &y, &groups)
        .expect("GEE fit failed");

    assert_parity(
        "gee_small",
        vec![
            check_vec(
                "params",
                &result.coefficients,
                &as_f64_vec(&fx["params"]),
                1e-2,
            ),
            check_vec(
                "bse",
                &result.robust_std_errors,
                &as_f64_vec(&fx["bse"]),
                2e-3,
            ),
        ],
    );
}
