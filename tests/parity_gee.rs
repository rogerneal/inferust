//! Parity tests for GEE against statsmodels.

mod common;

use common::{as_f64_vec, assert_parity, check_vec, load_fixture, xy};
use inferust::gee::{Gee, GeeFamily, WorkingCorrelation};

#[test]
fn parity_gee_poisson_exchangeable() {
    let fx = load_fixture("gee_poisson");
    let (x, y) = xy(&fx);
    let clusters = as_f64_vec(&fx["dataset"]["clusters"])
        .into_iter()
        .map(|v| v as usize)
        .collect::<Vec<_>>();
    let k = x[0].len();
    let result = Gee::new(GeeFamily::Poisson)
        .with_feature_names((1..=k).map(|i| format!("x{i}")).collect())
        .with_working_correlation(WorkingCorrelation::Exchangeable)
        .max_iter(30)
        .fit(&x, &y, &clusters)
        .expect("GEE fit failed");

    assert_parity(
        "gee_poisson",
        vec![
            check_vec(
                "params",
                &result.coefficients,
                &as_f64_vec(&fx["params"]),
                5e-3,
            ),
            check_vec(
                "bse",
                &result.robust_std_errors,
                &as_f64_vec(&fx["bse"]),
                5e-3,
            ),
        ],
    );
}
