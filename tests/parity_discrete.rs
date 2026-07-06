//! Parity tests for discrete-choice models against statsmodels.

mod common;

use common::{as_f64, as_f64_vec, assert_parity, check_scalar, check_vec, load_fixture, xy};
use inferust::discrete::{
    MultinomialLogit, NegativeBinomial, OrderedLogit, Probit, ZeroInflatedPoisson,
};

fn feature_names(k: usize) -> Vec<String> {
    (1..=k).map(|i| format!("x{i}")).collect()
}

fn y_usize(fx: &serde_json::Value) -> Vec<usize> {
    as_f64_vec(&fx["dataset"]["y"])
        .into_iter()
        .map(|v| v as usize)
        .collect()
}

#[test]
fn parity_probit_small() {
    let fx = load_fixture("probit_small");
    let (x, y) = xy(&fx);
    let k = x[0].len();
    let result = Probit::new()
        .with_feature_names(feature_names(k))
        .max_iter(200)
        .tolerance(1e-10)
        .fit(&x, &y)
        .expect("Probit fit failed");

    assert_parity(
        "probit_small",
        vec![
            check_vec(
                "params",
                &result.coefficients,
                &as_f64_vec(&fx["params"]),
                1e-5,
            ),
            check_scalar("llf", result.log_likelihood, as_f64(&fx["llf"]), 1e-5),
            check_scalar("aic", result.aic, as_f64(&fx["aic"]), 1e-5),
            check_scalar("bic", result.bic, as_f64(&fx["bic"]), 1e-5),
        ],
    );
}

#[test]
fn parity_negbin_small() {
    let fx = load_fixture("negbin_small");
    let (x, y) = xy(&fx);
    let k = x[0].len();
    let result = NegativeBinomial::new()
        .with_feature_names(feature_names(k))
        .max_iter(300)
        .fit(&x, &y)
        .expect("NegBin fit failed");

    assert_parity(
        "negbin_small",
        vec![
            check_vec(
                "params",
                &result.coefficients,
                &as_f64_vec(&fx["params"]),
                1e-4,
            ),
            check_scalar("alpha", result.alpha, as_f64(&fx["alpha"]), 1e-3),
            check_scalar("llf", result.log_likelihood, as_f64(&fx["llf"]), 1e-3),
            check_scalar("aic", result.aic, as_f64(&fx["aic"]), 1e-3),
            check_scalar("bic", result.bic, as_f64(&fx["bic"]), 1e-3),
        ],
    );
}

#[test]
fn parity_multinomial_small() {
    let fx = load_fixture("multinomial_small");
    let (x, _) = xy(&fx);
    let y = y_usize(&fx);
    let k = x[0].len();
    let result = MultinomialLogit::new()
        .with_feature_names(feature_names(k))
        .max_iter(400)
        .fit(&x, &y)
        .expect("Multinomial fit failed");

    assert_parity(
        "multinomial_small",
        vec![
            check_scalar("llf", result.log_likelihood, as_f64(&fx["llf"]), 1e-3),
            check_scalar("aic", result.aic, as_f64(&fx["aic"]), 1e-3),
            check_scalar("bic", result.bic, as_f64(&fx["bic"]), 1e-3),
        ],
    );
}

#[test]
fn parity_ordered_logit_small() {
    let fx = load_fixture("ordered_logit_small");
    let (x, _) = xy(&fx);
    let y = y_usize(&fx);
    let k = x[0].len();
    let result = OrderedLogit::new()
        .with_feature_names(feature_names(k))
        .max_iter(400)
        .fit(&x, &y)
        .expect("Ordered logit fit failed");

    assert_parity(
        "ordered_logit_small",
        vec![
            check_scalar("llf", result.log_likelihood, as_f64(&fx["llf"]), 1e-2),
            check_scalar("aic", result.aic, as_f64(&fx["aic"]), 1e-2),
            check_scalar("bic", result.bic, as_f64(&fx["bic"]), 1e-2),
        ],
    );
}

#[test]
fn parity_zip_small() {
    let fx = load_fixture("zip_small");
    let (x, y) = xy(&fx);
    let k = x[0].len();
    let result = ZeroInflatedPoisson::new()
        .with_feature_names(feature_names(k))
        .with_inflation_feature_names(feature_names(k))
        .max_iter(300)
        .fit(&x, &y, &x)
        .expect("ZIP fit failed");

    // statsmodels orders [inflation block, count block]; inferust reports count then inflation.
    assert_parity(
        "zip_small",
        vec![
            check_vec(
                "count_params",
                &result.count_coefficients,
                &as_f64_vec(&fx["inflation_params"]),
                5e-3,
            ),
            check_vec(
                "inflation_params",
                &result.inflation_coefficients,
                &as_f64_vec(&fx["count_params"]),
                5e-3,
            ),
            check_scalar("llf", result.log_likelihood, as_f64(&fx["llf"]), 5e-2),
            check_scalar("aic", result.aic, as_f64(&fx["aic"]), 5e-2),
            check_scalar("bic", result.bic, as_f64(&fx["bic"]), 5e-2),
        ],
    );
}
