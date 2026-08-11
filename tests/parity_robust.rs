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

    // Params and H1 asymptotic SEs / z-stats match statsmodels RLM (Huber).
    assert_parity(
        "robust_small",
        vec![
            check_vec(
                "params",
                &result.fit.coefficients,
                &as_f64_vec(&fx["params"]),
                1e-4,
            ),
            check_vec(
                "bse",
                &result.robust_std_errors,
                &as_f64_vec(&fx["bse"]),
                1e-4,
            ),
            check_vec(
                "tvalues",
                &result.robust_t_statistics,
                &as_f64_vec(&fx["tvalues"]),
                1e-3,
            ),
            check_vec(
                "pvalues",
                &result.robust_p_values,
                &as_f64_vec(&fx["pvalues"]),
                1e-8,
            ),
        ],
    );
}
