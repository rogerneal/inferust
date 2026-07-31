//! Parity tests for seasonal decomposition against
//! `statsmodels.tsa.seasonal` fixtures.

mod common;

use common::{as_f64_vec, assert_parity, check_scalar, check_vec, load_fixture};
use inferust::seasonal::{seasonal_decompose, DecompositionModel, Stl};

/// Compare a Rust series against a reference array that carries JSON `null`
/// where statsmodels produced NaN (the untrended edges of a centered MA).
fn check_vec_nullable(
    label: &str,
    actual: &[f64],
    expected: &serde_json::Value,
    tol: f64,
) -> Result<(), String> {
    let expected = expected
        .as_array()
        .unwrap_or_else(|| panic!("{label}: expected array"));
    if actual.len() != expected.len() {
        return Err(format!(
            "  {label}: length mismatch actual={} expected={}",
            actual.len(),
            expected.len()
        ));
    }
    let mut diffs = Vec::new();
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        match e.as_f64() {
            None => {
                if !a.is_nan() {
                    diffs.push(format!("  {label}[{i}]: expected NaN, got {a}"));
                }
            }
            Some(e) => {
                if a.is_nan() {
                    diffs.push(format!("  {label}[{i}]: unexpected NaN, expected {e}"));
                } else if let Err(s) = check_scalar(&format!("{label}[{i}]"), *a, e, tol) {
                    diffs.push(s);
                }
            }
        }
    }
    if diffs.is_empty() {
        Ok(())
    } else {
        Err(diffs.join("\n"))
    }
}

#[test]
fn parity_seasonal_decompose_additive() {
    let fx = load_fixture("seasonal_decompose");
    let y = as_f64_vec(&fx["dataset"]["y"]);
    let period = fx["period"].as_u64().expect("period") as usize;
    let d = seasonal_decompose(&y, period, DecompositionModel::Additive)
        .expect("additive decomposition failed");
    let expected = &fx["additive"];

    assert_parity(
        "seasonal_decompose_additive",
        vec![
            check_vec_nullable("trend", &d.trend, &expected["trend"], 1e-10),
            check_vec_nullable("seasonal", &d.seasonal, &expected["seasonal"], 1e-10),
            check_vec_nullable("resid", &d.resid, &expected["resid"], 1e-10),
        ],
    );
}

#[test]
fn parity_seasonal_decompose_multiplicative() {
    let fx = load_fixture("seasonal_decompose");
    let y = as_f64_vec(&fx["dataset"]["y_positive"]);
    let period = fx["period"].as_u64().expect("period") as usize;
    let d = seasonal_decompose(&y, period, DecompositionModel::Multiplicative)
        .expect("multiplicative decomposition failed");
    let expected = &fx["multiplicative"];

    assert_parity(
        "seasonal_decompose_multiplicative",
        vec![
            check_vec_nullable("trend", &d.trend, &expected["trend"], 1e-10),
            check_vec_nullable("seasonal", &d.seasonal, &expected["seasonal"], 1e-10),
            check_vec_nullable("resid", &d.resid, &expected["resid"], 1e-10),
        ],
    );
}

/// STL agreement tier.
///
/// `Stl` is an independent reimplementation of the loess inner loop, so the
/// only drift against statsmodels is accumulated floating-point rounding.
/// Measured maxima on this fixture are ~1e-12 for the default fit and ~2e-11
/// for the robust fit, which runs 15 outer reweighting passes.
const STL_TOL: f64 = 1e-11;
const STL_ROBUST_TOL: f64 = 1e-9;

#[test]
fn parity_stl() {
    let fx = load_fixture("stl");
    let y = as_f64_vec(&fx["dataset"]["y"]);
    let period = fx["period"].as_u64().expect("period") as usize;
    let res = Stl::new(period).fit(&y).expect("STL failed");

    assert_parity(
        "stl",
        vec![
            check_vec(
                "seasonal",
                &res.seasonal,
                &as_f64_vec(&fx["seasonal"]),
                STL_TOL,
            ),
            check_vec("trend", &res.trend, &as_f64_vec(&fx["trend"]), STL_TOL),
            check_vec("resid", &res.resid, &as_f64_vec(&fx["resid"]), STL_TOL),
        ],
    );
}

#[test]
fn parity_stl_robust() {
    let fx = load_fixture("stl");
    let y = as_f64_vec(&fx["dataset"]["y"]);
    let period = fx["period"].as_u64().expect("period") as usize;
    let res = Stl::new(period).robust(true).fit(&y).expect("STL failed");

    assert_parity(
        "stl_robust",
        vec![
            check_vec(
                "seasonal",
                &res.seasonal,
                &as_f64_vec(&fx["robust_seasonal"]),
                STL_ROBUST_TOL,
            ),
            check_vec(
                "trend",
                &res.trend,
                &as_f64_vec(&fx["robust_trend"]),
                STL_ROBUST_TOL,
            ),
        ],
    );
}

/// The STL identity `observed = trend + seasonal + resid` holds by construction.
#[test]
fn stl_components_sum_to_observed() {
    let fx = load_fixture("stl");
    let y = as_f64_vec(&fx["dataset"]["y"]);
    let period = fx["period"].as_u64().expect("period") as usize;
    let res = Stl::new(period).fit(&y).expect("STL failed");

    let reconstructed: Vec<f64> = (0..y.len())
        .map(|t| res.trend[t] + res.seasonal[t] + res.resid[t])
        .collect();
    assert_parity(
        "stl_identity",
        vec![check_vec("observed", &reconstructed, &y, 1e-10)],
    );
}
