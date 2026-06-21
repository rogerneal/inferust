//! Multiple-testing p-value corrections.
//!
//! Matches `statsmodels.stats.multitest.multipletests` semantics for four
//! widely used methods:
//!
//! ```text
//! Bonferroni              p_i_corrected = p_i * n
//! Holm (step-down)        p_(i)_corrected = max_{j<=i} p_(j) * (n - j + 1)
//! Benjamini-Hochberg FDR  p_(i)_corrected = min_{j>=i} p_(j) * n / j
//! Benjamini-Yekutieli FDR p_(i)_corrected = min_{j>=i} p_(j) * n * c(n) / j
//! ```
//!
//! where `p_(1) <= p_(2) <= ... <= p_(n)` are the sorted p-values, `c(n) =
//! sum_{i=1}^{n} 1/i` is the harmonic number used by Benjamini-Yekutieli to
//! stay valid under arbitrary dependence between tests, and all four
//! corrected sequences are clipped to `1.0`. Benjamini-Hochberg assumes
//! independent or positively-correlated test statistics; Benjamini-Yekutieli
//! is more conservative but valid under any dependence structure.

use crate::error::{InferustError, Result};

/// Multiple-testing correction method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiTestMethod {
    /// `p_corrected = p * n`. Controls the family-wise error rate; the most
    /// conservative of the four.
    Bonferroni,
    /// Step-down procedure (Holm, 1979). Controls the family-wise error rate
    /// with uniformly more power than Bonferroni.
    Holm,
    /// Step-up false discovery rate procedure (Benjamini & Hochberg, 1995).
    /// Valid for independent or positively-correlated test statistics.
    BenjaminiHochberg,
    /// Step-up false discovery rate procedure (Benjamini & Yekutieli, 2001).
    /// Valid under arbitrary dependence; more conservative than
    /// Benjamini-Hochberg.
    BenjaminiYekutieli,
}

/// Result of adjusting a family of p-values for multiple comparisons.
#[derive(Debug, Clone)]
pub struct MultiTestResult {
    /// Whether each hypothesis is rejected at the requested `alpha`, in the
    /// same order as the input p-values.
    pub reject: Vec<bool>,
    /// Adjusted ("corrected") p-values, same order as the input.
    pub p_values_corrected: Vec<f64>,
    /// Significance level used to populate `reject`.
    pub alpha: f64,
    /// Method used.
    pub method: MultiTestMethod,
    /// Bonferroni-equivalent per-comparison alpha threshold (`alpha / n`).
    pub alpha_bonferroni: f64,
    /// Sidak-equivalent per-comparison alpha threshold (`1 - (1-alpha)^(1/n)`).
    pub alpha_sidak: f64,
}

impl MultiTestResult {
    /// Print a small table of original index, rejection, and corrected p-value.
    pub fn print(&self) {
        println!(
            "Multiple-testing correction ({:?}, alpha = {})",
            self.method, self.alpha
        );
        println!("{:>6} {:>10} {:>12}", "idx", "reject", "p_corrected");
        for (i, (&rej, &p)) in self
            .reject
            .iter()
            .zip(self.p_values_corrected.iter())
            .enumerate()
        {
            println!("{:>6} {:>10} {:>12.6}", i, rej, p);
        }
    }
}

/// Adjust a family of p-values for multiple comparisons.
///
/// `alpha` is the significance level used to populate [`MultiTestResult::reject`];
/// it does not change the corrected p-values themselves (except for
/// Bonferroni/Sidak-style thresholds reported alongside them).
///
/// # Example
/// ```
/// use inferust::hypothesis::multicomp::{adjust, MultiTestMethod};
///
/// let p = vec![0.01, 0.02, 0.03, 0.04, 0.5];
/// let result = adjust(&p, 0.05, MultiTestMethod::Holm).unwrap();
/// assert_eq!(result.reject.len(), 5);
/// assert!(result.p_values_corrected.iter().all(|&p| (0.0..=1.0).contains(&p)));
/// ```
pub fn adjust(p_values: &[f64], alpha: f64, method: MultiTestMethod) -> Result<MultiTestResult> {
    let n = p_values.len();
    if n == 0 {
        return Err(InferustError::InsufficientData { needed: 1, got: 0 });
    }
    if !(0.0..1.0).contains(&alpha) {
        return Err(InferustError::InvalidInput(
            "alpha must be between 0 and 1".to_string(),
        ));
    }
    if let Some(&bad) = p_values
        .iter()
        .find(|p| !p.is_finite() || **p < 0.0 || **p > 1.0)
    {
        return Err(InferustError::InvalidInput(format!(
            "p-values must be finite and within [0, 1], got {bad}"
        )));
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| p_values[a].partial_cmp(&p_values[b]).unwrap());
    let sorted: Vec<f64> = order.iter().map(|&i| p_values[i]).collect();

    let corrected_sorted = match method {
        MultiTestMethod::Bonferroni => sorted.iter().map(|p| p * n as f64).collect::<Vec<_>>(),
        MultiTestMethod::Holm => {
            let mut raw: Vec<f64> = sorted
                .iter()
                .enumerate()
                .map(|(i, p)| p * (n - i) as f64)
                .collect();
            running_max(&mut raw);
            raw
        }
        MultiTestMethod::BenjaminiHochberg => {
            let mut raw: Vec<f64> = sorted
                .iter()
                .enumerate()
                .map(|(i, p)| p * n as f64 / (i + 1) as f64)
                .collect();
            running_min_from_right(&mut raw);
            raw
        }
        MultiTestMethod::BenjaminiYekutieli => {
            let harmonic: f64 = (1..=n).map(|i| 1.0 / i as f64).sum();
            let mut raw: Vec<f64> = sorted
                .iter()
                .enumerate()
                .map(|(i, p)| p * n as f64 * harmonic / (i + 1) as f64)
                .collect();
            running_min_from_right(&mut raw);
            raw
        }
    };

    let mut p_values_corrected = vec![0.0; n];
    let mut reject = vec![false; n];
    for (rank, &orig_idx) in order.iter().enumerate() {
        let corrected = corrected_sorted[rank].min(1.0);
        p_values_corrected[orig_idx] = corrected;
        reject[orig_idx] = corrected <= alpha;
    }

    let alpha_bonferroni = alpha / n as f64;
    let alpha_sidak = 1.0 - (1.0 - alpha).powf(1.0 / n as f64);

    Ok(MultiTestResult {
        reject,
        p_values_corrected,
        alpha,
        method,
        alpha_bonferroni,
        alpha_sidak,
    })
}

/// Convenience wrapper for [`MultiTestMethod::Bonferroni`].
pub fn bonferroni(p_values: &[f64], alpha: f64) -> Result<MultiTestResult> {
    adjust(p_values, alpha, MultiTestMethod::Bonferroni)
}

/// Convenience wrapper for [`MultiTestMethod::Holm`].
pub fn holm(p_values: &[f64], alpha: f64) -> Result<MultiTestResult> {
    adjust(p_values, alpha, MultiTestMethod::Holm)
}

/// Convenience wrapper for [`MultiTestMethod::BenjaminiHochberg`].
pub fn fdr_bh(p_values: &[f64], alpha: f64) -> Result<MultiTestResult> {
    adjust(p_values, alpha, MultiTestMethod::BenjaminiHochberg)
}

/// Convenience wrapper for [`MultiTestMethod::BenjaminiYekutieli`].
pub fn fdr_by(p_values: &[f64], alpha: f64) -> Result<MultiTestResult> {
    adjust(p_values, alpha, MultiTestMethod::BenjaminiYekutieli)
}

/// In-place running maximum (left to right): `values[i] = max(values[0..=i])`.
fn running_max(values: &mut [f64]) {
    for i in 1..values.len() {
        if values[i] < values[i - 1] {
            values[i] = values[i - 1];
        }
    }
}

/// In-place running minimum from the right: `values[i] = min(values[i..])`.
fn running_min_from_right(values: &mut [f64]) {
    let n = values.len();
    for i in (0..n.saturating_sub(1)).rev() {
        if values[i] > values[i + 1] {
            values[i] = values[i + 1];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() <= tol,
            "expected {expected}, got {actual} (tol {tol})"
        );
    }

    #[test]
    fn bonferroni_scales_by_n() {
        let p = vec![0.01, 0.02, 0.03];
        let r = bonferroni(&p, 0.05).unwrap();
        assert_close(r.p_values_corrected[0], 0.03, 1e-12);
        assert_close(r.p_values_corrected[1], 0.06, 1e-12);
        assert_close(r.p_values_corrected[2], 0.09, 1e-12);
        assert_close(r.alpha_bonferroni, 0.05 / 3.0, 1e-12);
    }

    #[test]
    fn holm_enforces_monotonicity() {
        // sorted p = [0.01, 0.02, 0.03]; raw = [0.03, 0.04, 0.03] -> running
        // max = [0.03, 0.04, 0.04].
        let p = vec![0.01, 0.02, 0.03];
        let r = holm(&p, 0.05).unwrap();
        assert_close(r.p_values_corrected[0], 0.03, 1e-12);
        assert_close(r.p_values_corrected[1], 0.04, 1e-12);
        assert_close(r.p_values_corrected[2], 0.04, 1e-12);
    }

    #[test]
    fn holm_is_at_least_as_powerful_as_bonferroni() {
        let p = vec![0.001, 0.01, 0.02, 0.04, 0.5];
        let holm_r = holm(&p, 0.05).unwrap();
        let bonf_r = bonferroni(&p, 0.05).unwrap();
        for i in 0..p.len() {
            assert!(holm_r.p_values_corrected[i] <= bonf_r.p_values_corrected[i] + 1e-12);
        }
    }

    #[test]
    fn fdr_bh_matches_hand_computation() {
        // sorted p = [0.01, 0.02, 0.03]; raw = [0.03, 0.03, 0.03] (already
        // monotonic), so corrected == raw here.
        let p = vec![0.01, 0.02, 0.03];
        let r = fdr_bh(&p, 0.05).unwrap();
        assert_close(r.p_values_corrected[0], 0.03, 1e-12);
        assert_close(r.p_values_corrected[1], 0.03, 1e-12);
        assert_close(r.p_values_corrected[2], 0.03, 1e-12);
    }

    #[test]
    fn fdr_by_is_more_conservative_than_bh() {
        let p = vec![0.001, 0.004, 0.01, 0.03, 0.2, 0.4, 0.6];
        let bh = fdr_bh(&p, 0.05).unwrap();
        let by = fdr_by(&p, 0.05).unwrap();
        for i in 0..p.len() {
            assert!(by.p_values_corrected[i] >= bh.p_values_corrected[i] - 1e-12);
        }
    }

    #[test]
    fn corrected_p_values_never_exceed_one() {
        let p = vec![0.5, 0.6, 0.7, 0.8, 0.9];
        for method in [
            MultiTestMethod::Bonferroni,
            MultiTestMethod::Holm,
            MultiTestMethod::BenjaminiHochberg,
            MultiTestMethod::BenjaminiYekutieli,
        ] {
            let r = adjust(&p, 0.05, method).unwrap();
            assert!(r.p_values_corrected.iter().all(|&x| x <= 1.0));
        }
    }

    #[test]
    fn original_order_is_preserved() {
        // Unsorted input; corrected p-values and rejections must map back to
        // the original positions, not the sorted ones.
        let p = vec![0.04, 0.001, 0.03];
        let r = holm(&p, 0.05).unwrap();
        // p[1] is smallest, so it gets the smallest multiplier; verify the
        // ordering relationship is preserved through the unsort step.
        assert!(r.p_values_corrected[1] <= r.p_values_corrected[2]);
        assert!(r.p_values_corrected[2] <= r.p_values_corrected[0]);
    }

    #[test]
    fn rejects_invalid_alpha() {
        let p = vec![0.01, 0.02];
        assert!(adjust(&p, 1.5, MultiTestMethod::Bonferroni).is_err());
    }

    #[test]
    fn rejects_out_of_range_p_values() {
        let p = vec![0.5, 1.2];
        assert!(adjust(&p, 0.05, MultiTestMethod::Bonferroni).is_err());
    }

    #[test]
    fn rejects_empty_input() {
        let p: Vec<f64> = vec![];
        assert!(adjust(&p, 0.05, MultiTestMethod::Bonferroni).is_err());
    }
}
