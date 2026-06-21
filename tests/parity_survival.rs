//! Parity test for Cox PH against statsmodels.PHReg with Breslow ties.

mod common;

use common::{as_f64, as_f64_matrix, as_f64_vec, assert_parity, check_scalar, check_vec, load_fixture};
use inferust::survival::CoxPh;

#[test]
fn parity_cox_ph() {
    let fx = load_fixture("cox_ph");
    let times = as_f64_vec(&fx["dataset"]["times"]);
    let events_u: Vec<usize> = fx["dataset"]["events"]
        .as_array()
        .expect("events")
        .iter()
        .map(|v| v.as_u64().expect("event u64") as usize)
        .collect();
    let x = as_f64_matrix(&fx["dataset"]["x"]);

    let result = CoxPh::new()
        .fit(&times, &events_u, &x)
        .expect("CoxPh::fit failed");

    assert_parity(
        "cox_ph",
        vec![
            check_vec(
                "params",
                &result.coefficients,
                &as_f64_vec(&fx["params"]),
                1e-5,
            ),
            check_vec("bse", &result.std_errors, &as_f64_vec(&fx["bse"]), 1e-5),
            check_vec(
                "zvalues",
                &result.z_statistics,
                &as_f64_vec(&fx["zvalues"]),
                1e-4,
            ),
            check_vec(
                "pvalues",
                &result.p_values,
                &as_f64_vec(&fx["pvalues"]),
                1e-4,
            ),
            check_scalar("llf", result.log_likelihood, as_f64(&fx["llf"]), 1e-5),
        ],
    );
}
