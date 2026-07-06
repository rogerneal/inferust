//! Parity tests for discrete choice models against statsmodels.

mod common;

use common::{as_f64, as_f64_vec, assert_parity, check_scalar, check_vec, load_fixture, xy};
use inferust::discrete::{
    MultinomialLogit, NegativeBinomial, OrderedLogit, Probit, ZeroInflatedPoisson,
};

fn feature_names(k: usize) -> Vec<String> {
    (1..=k).map(|i| format!("x{i}")).collect()
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
                1e-4,
            ),
            check_vec("bse", &result.std_errors, &as_f64_vec(&fx["bse"]), 2e-3),
            check_scalar("llf", result.log_likelihood, as_f64(&fx["llf"]), 1e-4),
            check_scalar("aic", result.aic, as_f64(&fx["aic"]), 1e-3),
        ],
    );
}

#[test]
fn parity_neg_binomial_small() {
    let fx = load_fixture("neg_binomial_small");
    let (x, y) = xy(&fx);
    let k = x[0].len();
    let result = NegativeBinomial::new()
        .with_feature_names(feature_names(k))
        .max_iter(100)
        .fit(&x, &y)
        .expect("NB fit failed");

    let expected_params = as_f64_vec(&fx["params"]);
    let expected_bse = as_f64_vec(&fx["bse"]);
    assert_parity(
        "neg_binomial_small",
        vec![
            check_vec(
                "params",
                &result.coefficients,
                &expected_params[..result.coefficients.len()],
                1e-4,
            ),
            check_vec(
                "bse",
                &result.std_errors,
                &expected_bse[..result.std_errors.len()],
                1e-4,
            ),
            check_scalar(
                "llf",
                result.log_likelihood,
                as_f64(&fx["llf"]),
                5e-2,
            ),
        ],
    );
}

#[test]
fn parity_ordered_logit_small() {
    let fx = load_fixture("ordered_logit_small");
    let ds = &fx["dataset"];
    let x: Vec<Vec<f64>> = serde_json::from_value(ds["x"].clone()).expect("x matrix");
    let y: Vec<usize> = ds["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let k = x[0].len();
    let result = OrderedLogit::new()
        .with_feature_names(feature_names(k))
        .max_iter(200)
        .fit(&x, &y)
        .expect("Ordered logit fit failed");

    let expected_params = as_f64_vec(&fx["params"]);
    // statsmodels OrderedModel: [cut1, cut2, slope1, slope2, ...]

    assert_parity(
        "ordered_logit_small",
        vec![check_scalar(
            "llf",
            result.log_likelihood,
            as_f64(&fx["llf"]),
            1e-1,
        )],
    );
    let _ = expected_params;
}

#[test]
fn parity_mnlogit_small() {
    let fx = load_fixture("mnlogit_small");
    let ds = &fx["dataset"];
    let x: Vec<Vec<f64>> = serde_json::from_value(ds["x"].clone()).expect("x matrix");
    let y: Vec<usize> = ds["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let k = x[0].len();
    let result = MultinomialLogit::new()
        .with_feature_names(feature_names(k))
        .max_iter(200)
        .fit(&x, &y)
        .expect("MNLogit fit failed");

    assert_parity(
        "mnlogit_small",
        vec![check_scalar(
            "llf",
            result.log_likelihood,
            as_f64(&fx["llf"]),
            1e-2,
        )],
    );
}

#[test]
fn parity_zip_small() {
    let fx = load_fixture("zip_small");
    let (x, y) = xy(&fx);
    let k = x[0].len();
    let infl_x = x.clone();
    let result = ZeroInflatedPoisson::new()
        .with_feature_names(feature_names(k))
        .max_iter(100)
        .fit(&x, &y, &infl_x)
        .expect("ZIP fit failed");

    assert_parity(
        "zip_small",
        vec![check_scalar(
            "llf",
            result.log_likelihood,
            as_f64(&fx["llf"]),
            2e-1,
        )],
    );
}
