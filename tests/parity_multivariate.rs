//! Parity for PCA and one-way MANOVA against statsmodels.

mod common;

use common::{
    as_f64, as_f64_matrix, as_f64_vec, assert_parity, check_scalar, check_vec, load_fixture,
};
use inferust::multivariate::{one_way_manova, pca};

/// Align each component so its largest-magnitude entry matches the fixture sign.
fn align_components(actual: &mut [Vec<f64>], expected: &[Vec<f64>]) {
    for (a, e) in actual.iter_mut().zip(expected.iter()) {
        let mut dot = 0.0;
        for (ai, ei) in a.iter().zip(e.iter()) {
            dot += ai * ei;
        }
        if dot < 0.0 {
            for v in a.iter_mut() {
                *v = -*v;
            }
        }
    }
}

#[test]
fn parity_pca() {
    let fx = load_fixture("pca");
    let x = as_f64_matrix(&fx["dataset"]["x"]);
    let result = pca(&x).expect("pca failed");

    let expected_components = as_f64_matrix(&fx["components"]);
    let mut components = result.components.clone();
    align_components(&mut components, &expected_components);
    let actual_flat: Vec<f64> = components.into_iter().flatten().collect();
    let expected_flat: Vec<f64> = expected_components.into_iter().flatten().collect();

    let scores = result
        .transform(&x, result.components.len())
        .expect("scores");
    let mut scores_aligned = scores;
    let expected_scores = as_f64_matrix(&fx["scores"]);
    // Align score columns the same way as components.
    for j in 0..scores_aligned[0].len() {
        let mut dot = 0.0;
        for i in 0..scores_aligned.len() {
            dot += scores_aligned[i][j] * expected_scores[i][j];
        }
        if dot < 0.0 {
            for row in &mut scores_aligned {
                row[j] = -row[j];
            }
        }
    }
    let scores_flat: Vec<f64> = scores_aligned.into_iter().flatten().collect();
    let expected_scores_flat: Vec<f64> = expected_scores.into_iter().flatten().collect();

    assert_parity(
        "pca",
        vec![
            check_vec("mean", &result.mean, &as_f64_vec(&fx["mean"]), 1e-12),
            check_vec("components", &actual_flat, &expected_flat, 1e-10),
            check_vec(
                "explained_variance",
                &result.explained_variance,
                &as_f64_vec(&fx["explained_variance"]),
                1e-10,
            ),
            check_vec(
                "explained_variance_ratio",
                &result.explained_variance_ratio,
                &as_f64_vec(&fx["explained_variance_ratio"]),
                1e-10,
            ),
            check_vec("scores", &scores_flat, &expected_scores_flat, 1e-9),
        ],
    );
}

#[test]
fn parity_manova() {
    let fx = load_fixture("manova");
    let groups_json = fx["dataset"]["groups"].as_array().expect("groups");
    let groups: Vec<Vec<Vec<f64>>> = groups_json.iter().map(as_f64_matrix).collect();
    let result = one_way_manova(&groups).expect("manova failed");

    assert_parity(
        "manova",
        vec![
            check_scalar(
                "wilks_lambda",
                result.wilks_lambda,
                as_f64(&fx["wilks_lambda"]),
                1e-10,
            ),
            check_scalar(
                "df_hypothesis",
                result.df_hypothesis,
                as_f64(&fx["df_hypothesis"]),
                1e-12,
            ),
            check_scalar("df_error", result.df_error, as_f64(&fx["df_error"]), 1e-10),
            check_scalar(
                "f_statistic",
                result.f_statistic,
                as_f64(&fx["f_statistic"]),
                1e-8,
            ),
            check_scalar("p_value", result.p_value, as_f64(&fx["p_value"]), 1e-8),
        ],
    );
}
