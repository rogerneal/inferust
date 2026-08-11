//! Parity for the generic GLM front-end against dedicated estimators / fixtures.
//!
//! Gaussian and Poisson were covered first. Binomial routes through `Logistic`
//! (`logit_small`); Gamma through the canonical InversePower fit (`gamma_glm`);
//! InverseGaussian through the Log-link fit (`inverse_gaussian_glm`).

mod common;

use common::{as_f64, as_f64_vec, assert_parity, check_scalar, check_vec, load_fixture, xy};
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
        vec![
            check_vec(
                "params",
                &ols.coefficients,
                &as_f64_vec(&fx["params"]),
                1e-8,
            ),
            check_vec("bse", &ols.std_errors, &as_f64_vec(&fx["bse"]), 1e-8),
        ],
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
        vec![
            check_vec(
                "params",
                &poisson.coefficients,
                &as_f64_vec(&fx["params"]),
                1e-5,
            ),
            check_vec("bse", &poisson.std_errors, &as_f64_vec(&fx["bse"]), 1e-5),
            check_scalar("llf", poisson.log_likelihood, as_f64(&fx["llf"]), 1e-5),
        ],
    );
}

#[test]
fn parity_glm_binomial_matches_logit() {
    let fx = load_fixture("logit_small");
    let (x, y) = xy(&fx);
    let k = x[0].len();
    let result = Glm::new(GlmFamily::Binomial)
        .with_feature_names((1..=k).map(|i| format!("x{i}")).collect())
        .fit(&x, &y)
        .expect("GLM Binomial fit failed");
    let GlmResult::Binomial(logit) = result else {
        panic!("expected Binomial result");
    };
    assert_parity(
        "glm_binomial_logit_small",
        vec![
            check_vec(
                "params",
                &logit.coefficients,
                &as_f64_vec(&fx["params"]),
                1e-5,
            ),
            check_vec("bse", &logit.std_errors, &as_f64_vec(&fx["bse"]), 1e-5),
            check_scalar("llf", logit.log_likelihood, as_f64(&fx["llf"]), 1e-5),
            check_scalar("aic", logit.aic, as_f64(&fx["aic"]), 1e-4),
        ],
    );
}

#[test]
fn parity_glm_gamma_matches_gamma() {
    let fx = load_fixture("gamma_glm");
    let (x, y) = xy(&fx);
    let expected = &fx["inverse_power"];
    let k = x[0].len();
    let result = Glm::new(GlmFamily::Gamma)
        .with_feature_names((1..=k).map(|i| format!("x{i}")).collect())
        .fit(&x, &y)
        .expect("GLM Gamma fit failed");
    let GlmResult::Gamma(gamma) = result else {
        panic!("expected Gamma result");
    };
    assert_parity(
        "glm_gamma_inverse_power",
        vec![
            check_vec(
                "params",
                &gamma.coefficients,
                &as_f64_vec(&expected["params"]),
                1e-5,
            ),
            check_vec(
                "bse",
                &gamma.std_errors,
                &as_f64_vec(&expected["bse"]),
                1e-5,
            ),
            check_scalar("llf", gamma.log_likelihood, as_f64(&expected["llf"]), 1e-5),
            check_scalar(
                "deviance",
                gamma.deviance,
                as_f64(&expected["deviance"]),
                1e-4,
            ),
        ],
    );
}

#[test]
fn parity_glm_inverse_gaussian_matches_fixture() {
    let fx = load_fixture("inverse_gaussian_glm");
    let (x, y) = xy(&fx);
    let k = x[0].len();
    let result = Glm::new(GlmFamily::InverseGaussian)
        .with_feature_names((1..=k).map(|i| format!("x{i}")).collect())
        .fit(&x, &y)
        .expect("GLM InverseGaussian fit failed");
    let GlmResult::InverseGaussian(ig) = result else {
        panic!("expected InverseGaussian result");
    };
    assert_parity(
        "glm_inverse_gaussian_log",
        vec![
            check_vec("params", &ig.coefficients, &as_f64_vec(&fx["params"]), 1e-5),
            check_vec("bse", &ig.std_errors, &as_f64_vec(&fx["bse"]), 1e-5),
            check_scalar("llf", ig.log_likelihood, as_f64(&fx["llf"]), 1e-5),
            check_scalar("deviance", ig.deviance, as_f64(&fx["deviance"]), 1e-4),
            check_scalar("scale", ig.dispersion, as_f64(&fx["scale"]), 1e-4),
        ],
    );
}
