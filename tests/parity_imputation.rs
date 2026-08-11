//! Parity tests for ``MiceImputer`` mean-impute and chained OLS passes.
//!
//! The fixture reimplements the same algorithm in Python (mean fill, then
//! per-column OLS with intercept for a fixed iteration count).

mod common;

use common::{
    as_f64, as_f64_matrix, as_f64_vec, assert_parity, check_scalar, check_vec, load_fixture,
};
use inferust::imputation::MiceImputer;

fn option_matrix(fx: &serde_json::Value) -> Vec<Vec<Option<f64>>> {
    fx["dataset"]["data"]
        .as_array()
        .expect("data")
        .iter()
        .map(|row| {
            row.as_array()
                .expect("row")
                .iter()
                .map(|v| if v.is_null() { None } else { Some(as_f64(v)) })
                .collect()
        })
        .collect()
}

#[test]
fn parity_imputation_mean_and_mice() {
    let fx = load_fixture("imputation_mice_small");
    let data = option_matrix(&fx);
    let iterations = fx["dataset"]["iterations"].as_u64().expect("iterations") as usize;

    let mean = MiceImputer::new()
        .mean_impute(&data)
        .expect("mean_impute failed");
    let mice = MiceImputer::new()
        .iterations(iterations)
        .fit_transform(&data)
        .expect("fit_transform failed");

    let expected_means = as_f64_vec(&fx["column_means"]);
    let expected_mean_data = as_f64_matrix(&fx["mean_impute_data"]);
    let expected_mice = as_f64_matrix(&fx["fit_transform_data"]);
    let expected_cells = fx["imputed_cells"].as_u64().expect("imputed_cells") as usize;

    let mean_flat: Vec<f64> = mean.data.iter().flatten().copied().collect();
    let expected_mean_flat: Vec<f64> = expected_mean_data.iter().flatten().copied().collect();
    let mice_flat: Vec<f64> = mice.data.iter().flatten().copied().collect();
    let expected_mice_flat: Vec<f64> = expected_mice.iter().flatten().copied().collect();

    assert_parity(
        "imputation_mice_small",
        vec![
            check_vec("column_means", &mean.column_means, &expected_means, 1e-12),
            check_vec("mean_impute_data", &mean_flat, &expected_mean_flat, 1e-12),
            check_scalar(
                "imputed_cells",
                mean.imputed_cells as f64,
                expected_cells as f64,
                0.0,
            ),
            check_vec("fit_transform_data", &mice_flat, &expected_mice_flat, 1e-8),
            check_scalar(
                "mice_imputed_cells",
                mice.imputed_cells as f64,
                expected_cells as f64,
                0.0,
            ),
        ],
    );
}
