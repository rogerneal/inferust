//! Parity for entity fixed-effects panel OLS.
//!
//! Coefficients match `linearmodels.panel.PanelOLS(entity_effects=True)`.
//! Standard errors match statsmodels OLS on the within-transformed design
//! (inferust's demean-then-`Ols::no_intercept` path). linearmodels applies an
//! extra within-df correction to SEs that inferust does not.

mod common;

use common::{as_f64, as_f64_vec, assert_parity, check_scalar, check_vec, load_fixture, xy};
use inferust::panel::PanelOls;

#[test]
fn parity_panel_entity_fe() {
    let fx = load_fixture("panel_fe");
    let (x, y) = xy(&fx);
    let entities: Vec<usize> = fx["dataset"]["entities"]
        .as_array()
        .expect("entities")
        .iter()
        .map(|v| v.as_u64().expect("entity id") as usize)
        .collect();
    let k = x[0].len();
    let names: Vec<String> = (1..=k).map(|i| format!("x{i}")).collect();

    let result = PanelOls::new()
        .with_feature_names(names)
        .fit_entity_fe(&x, &y, &entities)
        .expect("panel FE fit failed");

    assert_parity(
        "panel_fe",
        vec![
            check_vec(
                "params",
                &result.coefficients,
                &as_f64_vec(&fx["params"]),
                1e-10,
            ),
            check_vec("bse", &result.std_errors, &as_f64_vec(&fx["bse"]), 1e-10),
            check_vec(
                "tvalues",
                &result.t_statistics,
                &as_f64_vec(&fx["tvalues"]),
                1e-8,
            ),
            check_vec(
                "pvalues",
                &result.p_values,
                &as_f64_vec(&fx["pvalues"]),
                1e-8,
            ),
            check_scalar("rsquared", result.r_squared, as_f64(&fx["rsquared"]), 1e-10),
        ],
    );
}
