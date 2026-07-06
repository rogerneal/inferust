//! Parity tests for mixed linear models against statsmodels.

mod common;

use common::{as_f64_vec, assert_parity, check_vec, load_fixture};

#[test]
fn parity_mixed_small() {
    let fx = load_fixture("mixed_small");
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

    let result = inferust::mixed::MixedLinearModel::new()
        .with_feature_names((1..=k).map(|i| format!("x{i}")).collect())
        .max_iter(200)
        .fit_random_intercept(&x, &y, &groups)
        .expect("Mixed LM fit failed");

    assert_parity(
        "mixed_small",
        vec![
            check_vec(
                "params",
                &result.coefficients,
                &as_f64_vec(&fx["params"]),
                1e-3,
            ),
            check_vec("bse", &result.std_errors, &as_f64_vec(&fx["bse"]), 2e-3),
        ],
    );
}
