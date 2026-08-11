//! Parity tests for robust linear regression against statsmodels RLM.

mod common;

use common::{as_f64_vec, assert_parity, check_vec, load_fixture, xy};
use inferust::robust::RobustLinearModel;

#[test]
fn parity_robust_small() {
    let fx = load_fixture("robust_small");
    let (x, y) = xy(&fx);
    let k = x[0].len();
    let result = RobustLinearModel::new()
        .with_feature_names((1..=k).map(|i| format!("x{i}")).collect())
        .max_iter(50)
        .fit(&x, &y)
        .expect("Robust LM fit failed");

    // Params match statsmodels RLM (Huber). Sandwich SEs use a different
    // convention than RLM.bse, so only coefficients are pinned here.
    assert_parity(
        "robust_small",
        vec![check_vec(
            "params",
            &result.fit.coefficients,
            &as_f64_vec(&fx["params"]),
            1e-4,
        )],
    );

    assert!(result
        .robust_std_errors
        .iter()
        .all(|s| s.is_finite() && *s > 0.0));
    assert_eq!(
        result.robust_t_statistics.len(),
        result.fit.coefficients.len()
    );
}
