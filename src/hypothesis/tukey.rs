//! Tukey's Honestly Significant Difference (HSD) post-hoc test, and the
//! studentized range distribution it depends on.
//!
//! Given `k` group means estimated from a common within-group mean squared
//! error (e.g. the residual MSE from a one-way ANOVA, see
//! [`crate::hypothesis::anova::one_way`]), Tukey HSD tests every pairwise
//! difference while controlling the family-wise error rate at exactly
//! `alpha`, using the studentized range distribution rather than a per-pair
//! t-test. Unequal group sizes are handled via the Tukey-Kramer adjustment:
//!
//! ```text
//! q_ij = |mean_i - mean_j| / sqrt(MSE/2 * (1/n_i + 1/n_j))
//! ```
//!
//! `q_ij` is then compared against the studentized range distribution with
//! `k` groups and `nu` within-group degrees of freedom.
//!
//! The studentized range distribution has no closed form. This module
//! computes its CDF via nested Gauss-Legendre quadrature: an outer integral
//! over the scaled chi distribution governing the denominator (the
//! within-group standard deviation estimate), and an inner integral over the
//! distribution of the range of `k` iid standard normal variables. This
//! matches the textbook construction used by R's `ptukey` and is accurate to
//! roughly `1e-9` against the exact distribution; statsmodels itself uses an
//! interpolated lookup table (`statsmodels.stats.libqsturng`) that is only
//! accurate to about `1e-3`, so don't expect tighter-than-`1e-3` parity
//! against statsmodels' own Tukey HSD output (see `docs/parity.md`).

use statrs::distribution::{ChiSquared, Continuous, ContinuousCDF, Normal};

use crate::error::{InferustError, Result};

/// One pairwise group comparison in a Tukey HSD table.
#[derive(Debug, Clone)]
pub struct TukeyComparison {
    /// Label of the first group in the comparison.
    pub group_a: String,
    /// Label of the second group in the comparison.
    pub group_b: String,
    /// `mean(group_a) - mean(group_b)`.
    pub mean_diff: f64,
    /// Standard error of `mean_diff` (Tukey-Kramer adjusted for unequal `n`).
    pub std_error: f64,
    /// Studentized range statistic `|mean_diff| / std_error`.
    pub q_statistic: f64,
    /// Family-wise-error-rate-adjusted p-value from the studentized range
    /// distribution.
    pub p_value: f64,
    /// Lower bound of the simultaneous confidence interval for `mean_diff`.
    pub conf_low: f64,
    /// Upper bound of the simultaneous confidence interval for `mean_diff`.
    pub conf_high: f64,
    /// Whether the null hypothesis (`mean_a == mean_b`) is rejected at `alpha`.
    pub reject: bool,
}

/// Full Tukey HSD result: every pairwise group comparison plus the shared
/// critical value.
#[derive(Debug, Clone)]
pub struct TukeyHsdResult {
    /// One row per unordered pair of groups, in the order the groups were
    /// supplied (group 0 vs 1, 0 vs 2, ..., 1 vs 2, ...).
    pub comparisons: Vec<TukeyComparison>,
    /// Family-wise significance level used for `reject` and the confidence
    /// intervals.
    pub alpha: f64,
    /// Within-group residual degrees of freedom (`N - k`).
    pub df_within: f64,
    /// Pooled within-group mean squared error.
    pub mse_within: f64,
    /// Studentized range critical value at `alpha`, `k` groups, `df_within`
    /// degrees of freedom.
    pub q_critical: f64,
}

impl TukeyHsdResult {
    /// Print a summary table, one row per pairwise comparison.
    pub fn print(&self) {
        println!(
            "Tukey HSD (alpha = {}, df_within = {}, q_crit = {:.4})",
            self.alpha, self.df_within, self.q_critical
        );
        println!(
            "{:<10} {:<10} {:>10} {:>10} {:>10} {:>9} {:>10} {:>10}",
            "group_a", "group_b", "meandiff", "std_err", "q_stat", "p_value", "lower", "upper"
        );
        for c in &self.comparisons {
            println!(
                "{:<10} {:<10} {:>10.4} {:>10.4} {:>10.4} {:>9.4} {:>10.4} {:>10.4} {}",
                c.group_a,
                c.group_b,
                c.mean_diff,
                c.std_error,
                c.q_statistic,
                c.p_value,
                c.conf_low,
                c.conf_high,
                if c.reject { "*" } else { "" }
            );
        }
    }
}

/// Run Tukey's HSD post-hoc test on raw group samples (Tukey-Kramer
/// adjustment applied automatically for unequal group sizes).
///
/// `names` optionally labels each group (defaults to `group0`, `group1`,
/// ...); pass `None` to use the defaults. Each group needs at least 2
/// observations and there must be at least 2 groups.
///
/// # Example
/// ```
/// use inferust::hypothesis::tukey::tukey_hsd;
///
/// let a = [1.0, 2.0, 3.0, 2.5];
/// let b = [4.0, 5.0, 4.5, 5.5];
/// let c = [1.2, 1.8, 2.1, 1.5];
/// let result = tukey_hsd(&[&a, &b, &c], None, 0.05).unwrap();
/// assert_eq!(result.comparisons.len(), 3);
/// ```
pub fn tukey_hsd(
    groups: &[&[f64]],
    names: Option<&[String]>,
    alpha: f64,
) -> Result<TukeyHsdResult> {
    let k = groups.len();
    if k < 2 {
        return Err(InferustError::InsufficientData { needed: 2, got: k });
    }
    if !(0.0..1.0).contains(&alpha) {
        return Err(InferustError::InvalidInput(
            "alpha must be between 0 and 1".to_string(),
        ));
    }
    for g in groups {
        if g.len() < 2 {
            return Err(InferustError::InsufficientData {
                needed: 2,
                got: g.len(),
            });
        }
    }

    let n_total: usize = groups.iter().map(|g| g.len()).sum();
    let df_within = (n_total - k) as f64;

    let means: Vec<f64> = groups
        .iter()
        .map(|g| g.iter().sum::<f64>() / g.len() as f64)
        .collect();
    let ss_within: f64 = groups
        .iter()
        .zip(means.iter())
        .map(|(g, m)| g.iter().map(|x| (x - m).powi(2)).sum::<f64>())
        .sum();
    let mse_within = ss_within / df_within;

    let labels: Vec<String> = match names {
        Some(n) if n.len() == k => n.to_vec(),
        _ => (0..k).map(|i| format!("group{i}")).collect(),
    };

    let q_critical = studentized_range_critical_value(alpha, k as f64, df_within)?;

    let mut comparisons = Vec::with_capacity(k * (k - 1) / 2);
    for i in 0..k {
        for j in (i + 1)..k {
            let mean_diff = means[i] - means[j];
            let se = (mse_within / 2.0
                * (1.0 / groups[i].len() as f64 + 1.0 / groups[j].len() as f64))
                .sqrt();
            let q_statistic = if se > 0.0 { mean_diff.abs() / se } else { 0.0 };
            let p_value = studentized_range_p_value(q_statistic, k as f64, df_within)?;
            let margin = q_critical * se;
            comparisons.push(TukeyComparison {
                group_a: labels[i].clone(),
                group_b: labels[j].clone(),
                mean_diff,
                std_error: se,
                q_statistic,
                p_value,
                conf_low: mean_diff - margin,
                conf_high: mean_diff + margin,
                reject: p_value < alpha,
            });
        }
    }

    Ok(TukeyHsdResult {
        comparisons,
        alpha,
        df_within,
        mse_within,
        q_critical,
    })
}

// ── Studentized range distribution ──────────────────────────────────────────

/// CDF of the studentized range distribution: `P(Q <= q)` for the range of
/// `k` iid standard normal variables divided by an independent estimate of
/// scale with `nu` degrees of freedom.
///
/// Pass `nu = f64::INFINITY` for the known-variance limiting case (equivalent
/// to the plain range distribution of `k` standard normals).
pub fn studentized_range_cdf(q: f64, k: f64, nu: f64) -> Result<f64> {
    if q <= 0.0 {
        return Ok(0.0);
    }
    if k < 2.0 || nu <= 0.0 {
        return Err(InferustError::InvalidInput(
            "studentized range distribution requires k >= 2 and nu > 0".to_string(),
        ));
    }
    if nu.is_infinite() {
        return Ok(range_cdf(q, k));
    }

    let chi = ChiSquared::new(nu).map_err(|_| {
        InferustError::InvalidInput("invalid chi-squared degrees of freedom".to_string())
    })?;
    let s_upper = (chi.inverse_cdf(1.0 - 1e-10) / nu).sqrt().max(2.0);
    let s_lower = 1e-8_f64;

    let log_norm_const = (nu / 2.0) * (nu / 2.0).ln() - statrs::function::gamma::ln_gamma(nu / 2.0)
        + std::f64::consts::LN_2;

    let integrand = |s: f64| -> f64 {
        let log_density = log_norm_const + (nu - 1.0) * s.ln() - nu * s * s / 2.0;
        log_density.exp() * range_cdf(q * s, k)
    };

    Ok(gauss_legendre_composite(s_lower, s_upper, 24, integrand).clamp(0.0, 1.0))
}

/// Upper-tail p-value for an observed studentized range statistic.
pub fn studentized_range_p_value(q: f64, k: f64, nu: f64) -> Result<f64> {
    Ok((1.0 - studentized_range_cdf(q, k, nu)?).clamp(0.0, 1.0))
}

/// Critical value `q*` such that `P(Q <= q*) = 1 - alpha`, found by bisection
/// (the CDF is monotonic increasing in `q`, so this always converges).
pub fn studentized_range_critical_value(alpha: f64, k: f64, nu: f64) -> Result<f64> {
    if !(0.0..1.0).contains(&alpha) {
        return Err(InferustError::InvalidInput(
            "alpha must be between 0 and 1".to_string(),
        ));
    }
    let target = 1.0 - alpha;
    let mut lo = 0.0_f64;
    let mut hi = 2.0_f64;
    let mut tries = 0;
    while studentized_range_cdf(hi, k, nu)? < target {
        hi *= 2.0;
        tries += 1;
        if tries > 40 {
            return Err(InferustError::InvalidInput(
                "studentized range quantile search diverged".to_string(),
            ));
        }
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if studentized_range_cdf(mid, k, nu)? < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// CDF of the range of `k` iid standard normal variables: `P(R <= x)`, via
/// `k * integral(phi(z) * [Phi(z + x) - Phi(z)]^(k-1) dz)`.
fn range_cdf(x: f64, k: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let normal = Normal::new(0.0, 1.0).expect("standard normal is always valid");
    let integrand = |z: f64| -> f64 {
        let diff = normal.cdf(z + x) - normal.cdf(z);
        if diff <= 0.0 {
            0.0
        } else {
            k * normal.pdf(z) * diff.powf(k - 1.0)
        }
    };
    // The standard normal density is negligible beyond +/-10.
    gauss_legendre_composite(-10.0, 10.0, 24, integrand).clamp(0.0, 1.0)
}

/// Composite Gauss-Legendre quadrature on `[a, b]` using `segments` equal
/// panels of the 16-point rule. The integrands here (Gaussian densities and
/// CDFs) are smooth and analytic with no singularities, so this converges
/// geometrically with panel count.
fn gauss_legendre_composite<F: Fn(f64) -> f64>(a: f64, b: f64, segments: usize, f: F) -> f64 {
    let panel_width = (b - a) / segments as f64;
    let mut total = 0.0;
    for s in 0..segments {
        let panel_a = a + s as f64 * panel_width;
        let mid = panel_a + 0.5 * panel_width;
        let half = 0.5 * panel_width;
        let mut panel_sum = 0.0;
        for i in 0..16 {
            panel_sum += GL16_WEIGHTS[i] * f(mid + half * GL16_NODES[i]);
        }
        total += half * panel_sum;
    }
    total
}

/// 16-point Gauss-Legendre nodes on `[-1, 1]`.
const GL16_NODES: [f64; 16] = [
    -0.9894009349916499,
    -0.9445750230732326,
    -0.8656312023878318,
    -0.755_404_408_355_003,
    -0.6178762444026438,
    -0.4580167776572274,
    -0.2816035507792589,
    -0.0950125098376374,
    0.0950125098376374,
    0.2816035507792589,
    0.4580167776572274,
    0.6178762444026438,
    0.755_404_408_355_003,
    0.8656312023878318,
    0.9445750230732326,
    0.9894009349916499,
];

/// 16-point Gauss-Legendre weights on `[-1, 1]` (sum to 2.0).
const GL16_WEIGHTS: [f64; 16] = [
    0.0271524594117541,
    0.0622535239386479,
    0.0951585116824928,
    0.1246289712555339,
    0.1495959888165767,
    0.1691565193950025,
    0.1826034150449236,
    0.1894506104550685,
    0.1894506104550685,
    0.1826034150449236,
    0.1691565193950025,
    0.1495959888165767,
    0.1246289712555339,
    0.0951585116824928,
    0.0622535239386479,
    0.0271524594117541,
];

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
    fn studentized_range_cdf_matches_known_values() {
        // Reference values from scipy.stats.studentized_range.cdf, which
        // computes the same distribution via a different quadrature scheme.
        assert_close(
            studentized_range_cdf(3.0, 3.0, 10.0).unwrap(),
            0.8650165848,
            1e-6,
        );
        assert_close(
            studentized_range_cdf(3.0, 3.0, 30.0).unwrap(),
            0.8975534133,
            1e-6,
        );
        assert_close(
            studentized_range_cdf(4.5, 5.0, 15.0).unwrap(),
            0.9580690463,
            1e-6,
        );
        assert_close(
            studentized_range_cdf(2.0, 10.0, 5.0).unwrap(),
            0.1190843581,
            1e-6,
        );
    }

    #[test]
    fn studentized_range_cdf_is_monotonic_in_q() {
        let mut prev = 0.0;
        for q in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0] {
            let cdf = studentized_range_cdf(q, 4.0, 20.0).unwrap();
            assert!(cdf >= prev - 1e-12);
            prev = cdf;
        }
    }

    #[test]
    fn critical_value_round_trips_through_cdf() {
        let q_crit = studentized_range_critical_value(0.05, 4.0, 20.0).unwrap();
        let cdf_at_crit = studentized_range_cdf(q_crit, 4.0, 20.0).unwrap();
        assert_close(cdf_at_crit, 0.95, 1e-6);
    }

    #[test]
    fn three_balanced_groups_known_shape() {
        let a = [1.0, 2.0, 3.0, 2.0, 1.5];
        let b = [5.0, 6.0, 5.5, 6.5, 5.0];
        let c = [1.0, 1.2, 0.8, 1.5, 1.1];
        let result = tukey_hsd(&[&a, &b, &c], None, 0.05).unwrap();
        assert_eq!(result.comparisons.len(), 3);
        // group b (mean ~5.6) is clearly separated from groups a and c
        // (means ~1.9 and ~1.12); both comparisons against b should reject.
        let ab = &result.comparisons[0];
        let bc = &result.comparisons[2];
        assert_eq!(ab.group_a, "group0");
        assert_eq!(ab.group_b, "group1");
        assert!(ab.reject);
        assert!(bc.reject);
        assert!(ab.conf_low < ab.mean_diff && ab.mean_diff < ab.conf_high);
    }

    #[test]
    fn unequal_group_sizes_use_tukey_kramer() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 4.5, 5.5, 4.8, 5.2];
        let result = tukey_hsd(&[&a, &b], None, 0.05).unwrap();
        let se_manual = (result.mse_within / 2.0 * (1.0 / 3.0 + 1.0 / 6.0)).sqrt();
        assert_close(result.comparisons[0].std_error, se_manual, 1e-10);
    }

    #[test]
    fn custom_group_names_are_used() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let names = vec!["control".to_string(), "treatment".to_string()];
        let result = tukey_hsd(&[&a, &b], Some(&names), 0.05).unwrap();
        assert_eq!(result.comparisons[0].group_a, "control");
        assert_eq!(result.comparisons[0].group_b, "treatment");
    }

    #[test]
    fn rejects_too_few_groups() {
        let a = [1.0, 2.0, 3.0];
        assert!(tukey_hsd(&[&a], None, 0.05).is_err());
    }

    #[test]
    fn rejects_too_small_group() {
        let a = [1.0];
        let b = [2.0, 3.0, 4.0];
        assert!(tukey_hsd(&[&a, &b], None, 0.05).is_err());
    }
}
