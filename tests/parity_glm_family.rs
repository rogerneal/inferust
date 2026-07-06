//! Parity for the generic GLM front-end against dedicated estimators.

mod common;

use common::{as_f64_vec, assert_parity, check_vec, load_fixture, xy};
use inferust::glm_family::{Glm, GlmFamily, GlmResult};

#[test]
fn parity_glm_gaussian_matches_ols() {
    let fx = load_fixture("ols_small");
    let (x, y) = xy(&fx);
    let k = x[0].len();
    let result = Glm::new(GlmFamily::Gaussian)
        .with_feature_names((1..=k).map(|i| format!("x{i}")).collect())
        .fit(&x, &y)
        .expect("GLM Gaussian fit failed");
    let GlmResult::Gaussian(ols) = result else {
        panic!("expected Gaussian result");
    };
    assert_parity(
        "glm_gaussian_ols_small",
        vec![check_vec(
            "params",
            &ols.coefficients,
            &as_f64_vec(&fx["params"]),
            1e-8,
        )],
    );
}

#[test]
fn parity_glm_poisson_matches_poisson() {
    let fx = load_fixture("poisson_small");
    let (x, y) = xy(&fx);
    let k = x[0].len();
    let result = Glm::new(GlmFamily::Poisson)
        .with_feature_names((1..=k).map(|i| format!("x{i}")).collect())
        .fit(&x, &y)
        .expect("GLM Poisson fit failed");
    let GlmResult::Poisson(poisson) = result else {
        panic!("expected Poisson result");
    };
    assert_parity(
        "glm_poisson_small",
        vec![check_vec(
            "params",
            &poisson.coefficients,
            &as_f64_vec(&fx["params"]),
            1e-5,
        )],
    );
}
