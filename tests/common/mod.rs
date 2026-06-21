//! Shared helpers for parity tests against statsmodels reference JSON fixtures.
//!
//! Tolerances are deliberately tight where the algorithm should be deterministic
//! (linear algebra, closed-form stats) and loose where it shouldn't (iterative
//! optimisers, asymptotic p-values).
//!
//! Each test prints a per-field diff summary on failure so the parity audit
//! report can be filled in mechanically from `cargo test -- --nocapture` output.

#![allow(dead_code)]

use std::fs;
use std::path::PathBuf;

pub fn fixture_path(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("statsmodels");
    p.push(format!("{name}.json"));
    p
}

pub fn load_fixture(name: &str) -> serde_json::Value {
    let path = fixture_path(name);
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {}", path.display(), e));
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("failed to parse fixture {}: {}", path.display(), e))
}

pub fn as_f64(v: &serde_json::Value) -> f64 {
    v.as_f64()
        .unwrap_or_else(|| panic!("expected f64, got {v:?}"))
}

pub fn as_f64_vec(v: &serde_json::Value) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| panic!("expected array, got {v:?}"))
        .iter()
        .map(as_f64)
        .collect()
}

pub fn as_f64_matrix(v: &serde_json::Value) -> Vec<Vec<f64>> {
    v.as_array()
        .unwrap_or_else(|| panic!("expected matrix array, got {v:?}"))
        .iter()
        .map(as_f64_vec)
        .collect()
}

/// Extract `(x, y)` from a fixture dataset.
pub fn xy(value: &serde_json::Value) -> (Vec<Vec<f64>>, Vec<f64>) {
    let ds = &value["dataset"];
    let x = as_f64_matrix(&ds["x"]);
    let y = as_f64_vec(&ds["y"]);
    (x, y)
}

/// Assert two scalars match to within an absolute *or* relative tolerance.
///
/// Returns `Ok(())` on pass or `Err(diff_string)` on fail, so tests can either
/// `?`-bubble or accumulate diffs.
pub fn check_scalar(label: &str, actual: f64, expected: f64, tol: f64) -> Result<(), String> {
    let diff = (actual - expected).abs();
    let rel = diff / expected.abs().max(1.0);
    if diff <= tol || rel <= tol {
        Ok(())
    } else {
        Err(format!(
            "  {label}: actual={actual:.10e} expected={expected:.10e} diff={diff:.3e} rel={rel:.3e} (tol={tol:.0e})"
        ))
    }
}

pub fn check_vec(label: &str, actual: &[f64], expected: &[f64], tol: f64) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "  {label}: length mismatch actual={} expected={}",
            actual.len(),
            expected.len()
        ));
    }
    let mut diffs = Vec::new();
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        if let Err(s) = check_scalar(&format!("{label}[{i}]"), *a, *e, tol) {
            diffs.push(s);
        }
    }
    if diffs.is_empty() {
        Ok(())
    } else {
        Err(diffs.join("\n"))
    }
}

/// Run a parity check and either pass the test or panic with all diffs collected.
pub fn assert_parity(name: &str, checks: Vec<Result<(), String>>) {
    let mut fails: Vec<String> = checks.into_iter().filter_map(|r| r.err()).collect();
    if !fails.is_empty() {
        fails.insert(0, format!("parity diffs for {name}:"));
        panic!("{}", fails.join("\n"));
    }
}
