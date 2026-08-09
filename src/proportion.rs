//! Inference for binomial proportions.
//!
//! Mirrors `statsmodels.stats.proportion`:
//! - [`proportions_ztest`] — one- and two-sample z-tests for proportions.
//! - [`proportion_confint`] — confidence intervals (normal/Wald, Wilson,
//!   Clopper-Pearson, Agresti-Coull, Jeffreys).
//! - [`proportion_effectsize`] — Cohen's h for use with
//!   [`crate::power::NormalIndPower`].
//!
//! # Example
//! ```rust
//! use inferust::proportion::{proportions_ztest, proportion_confint, ConfintMethod};
//! use inferust::power::Alternative;
//!
//! // One-sample: is the success rate different from 50%?
//! let z = proportions_ztest(&[62], &[100], Some(0.5), Alternative::TwoSided).unwrap();
//! assert!(z.p_value < 0.05);
//!
//! // Wilson interval for the same data:
//! let (lo, hi) = proportion_confint(62, 100, 0.05, ConfintMethod::Wilson).unwrap();
//! assert!(lo > 0.5 && hi < 0.75);
//! ```

use crate::error::{InferustError, Result};
use crate::power::Alternative;
use statrs::distribution::{Beta, Continuous, ContinuousCDF, Normal};

/// Result of a z-test for one or two proportions.
#[derive(Debug, Clone)]
pub struct ProportionsZTestResult {
    /// z statistic.
    pub statistic: f64,
    /// p-value under the chosen alternative.
    pub p_value: f64,
    /// Sample proportion(s) used.
    pub proportions: Vec<f64>,
}

/// z-test for proportions, following `statsmodels.stats.proportion.proportions_ztest`.
///
/// * One sample: pass one count/nobs pair and `value = Some(p0)`; the variance
///   is estimated from the *sample* proportion (statsmodels' default).
/// * Two samples: pass two count/nobs pairs; `value` defaults to 0 and the
///   pooled sample proportion supplies the variance.
pub fn proportions_ztest(
    counts: &[u64],
    nobs: &[u64],
    value: Option<f64>,
    alternative: Alternative,
) -> Result<ProportionsZTestResult> {
    if counts.len() != nobs.len() || counts.is_empty() || counts.len() > 2 {
        return Err(InferustError::InvalidInput(
            "proportions_ztest takes 1 or 2 count/nobs pairs".into(),
        ));
    }
    for (&c, &n) in counts.iter().zip(nobs.iter()) {
        if n == 0 || c > n {
            return Err(InferustError::InvalidInput(
                "each count must satisfy 0 <= count <= nobs, nobs > 0".into(),
            ));
        }
    }
    let props: Vec<f64> = counts
        .iter()
        .zip(nobs.iter())
        .map(|(&c, &n)| c as f64 / n as f64)
        .collect();

    let (diff, variance) = if counts.len() == 1 {
        let p0 = value.ok_or_else(|| {
            InferustError::InvalidInput("one-sample proportions_ztest requires `value`".into())
        })?;
        let n = nobs[0] as f64;
        let p_hat = props[0];
        (p_hat - p0, p_hat * (1.0 - p_hat) / n)
    } else {
        let n1 = nobs[0] as f64;
        let n2 = nobs[1] as f64;
        let p_pooled = (counts[0] + counts[1]) as f64 / (n1 + n2);
        let var = p_pooled * (1.0 - p_pooled) * (1.0 / n1 + 1.0 / n2);
        (props[0] - props[1] - value.unwrap_or(0.0), var)
    };
    if variance <= 0.0 {
        return Err(InferustError::InvalidInput(
            "zero variance: all outcomes identical".into(),
        ));
    }
    let statistic = diff / variance.sqrt();
    let normal = Normal::new(0.0, 1.0).expect("standard normal");
    let p_value = match alternative {
        Alternative::TwoSided => 2.0 * normal.cdf(-statistic.abs()),
        Alternative::Larger => 1.0 - normal.cdf(statistic),
        Alternative::Smaller => normal.cdf(statistic),
    };
    Ok(ProportionsZTestResult {
        statistic,
        p_value: p_value.clamp(0.0, 1.0),
        proportions: props,
    })
}

/// Confidence-interval method for [`proportion_confint`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConfintMethod {
    /// Wald / asymptotic normal interval (statsmodels `"normal"`).
    #[default]
    Normal,
    /// Wilson score interval (statsmodels `"wilson"`).
    Wilson,
    /// Clopper-Pearson exact interval via the beta distribution
    /// (statsmodels `"beta"`).
    ClopperPearson,
    /// Agresti-Coull interval (statsmodels `"agresti_coull"`).
    AgrestiCoull,
    /// Jeffreys equal-tailed Bayesian interval (statsmodels `"jeffreys"`).
    Jeffreys,
}

/// Confidence interval for a binomial proportion at level `1 − alpha`.
///
/// Follows `statsmodels.stats.proportion.proportion_confint`. Returns
/// `(lower, upper)`, clipped to `[0, 1]`.
pub fn proportion_confint(
    count: u64,
    nobs: u64,
    alpha: f64,
    method: ConfintMethod,
) -> Result<(f64, f64)> {
    if nobs == 0 || count > nobs {
        return Err(InferustError::InvalidInput(
            "need 0 <= count <= nobs with nobs > 0".into(),
        ));
    }
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(InferustError::InvalidInput(
            "alpha must be in (0, 1)".into(),
        ));
    }
    let n = nobs as f64;
    let c = count as f64;
    let p_hat = c / n;
    let normal = Normal::new(0.0, 1.0).expect("standard normal");
    let z = normal.inverse_cdf(1.0 - alpha / 2.0);

    let (lower, upper) = match method {
        ConfintMethod::Normal => {
            let half = z * (p_hat * (1.0 - p_hat) / n).sqrt();
            (p_hat - half, p_hat + half)
        }
        ConfintMethod::Wilson => {
            let z2 = z * z;
            let center = (p_hat + z2 / (2.0 * n)) / (1.0 + z2 / n);
            let half = z / (1.0 + z2 / n) * (p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)).sqrt();
            (center - half, center + half)
        }
        ConfintMethod::ClopperPearson => {
            let lower = if count == 0 {
                0.0
            } else {
                let beta = Beta::new(c, n - c + 1.0)
                    .map_err(|_| InferustError::InvalidInput("invalid beta parameters".into()))?;
                refine_beta_quantile(&beta, alpha / 2.0)
            };
            let upper = if count == nobs {
                1.0
            } else {
                let beta = Beta::new(c + 1.0, n - c)
                    .map_err(|_| InferustError::InvalidInput("invalid beta parameters".into()))?;
                refine_beta_quantile(&beta, 1.0 - alpha / 2.0)
            };
            (lower, upper)
        }
        ConfintMethod::AgrestiCoull => {
            let z2 = z * z;
            let n_tilde = n + z2;
            let p_tilde = (c + z2 / 2.0) / n_tilde;
            let half = z * (p_tilde * (1.0 - p_tilde) / n_tilde).sqrt();
            (p_tilde - half, p_tilde + half)
        }
        ConfintMethod::Jeffreys => {
            let beta = Beta::new(c + 0.5, n - c + 0.5)
                .map_err(|_| InferustError::InvalidInput("invalid beta parameters".into()))?;
            let lower = if count == 0 {
                0.0
            } else {
                refine_beta_quantile(&beta, alpha / 2.0)
            };
            let upper = if count == nobs {
                1.0
            } else {
                refine_beta_quantile(&beta, 1.0 - alpha / 2.0)
            };
            (lower, upper)
        }
    };
    Ok((lower.max(0.0), upper.min(1.0)))
}

/// Beta quantile at `p`, polished with Newton steps on the CDF.
///
/// Guarantees `cdf(x) == p` to machine precision regardless of how accurate the
/// backend's `inverse_cdf` is. statrs 0.17 stopped a coarse bisection around
/// 1e-5 absolute, too wide for the exact Clopper-Pearson and Jeffreys
/// intervals; 0.19 resolves that, so this now converges on the first step. It
/// is retained because `cdf` and `pdf` are the accurate primitives.
fn refine_beta_quantile(beta: &Beta, p: f64) -> f64 {
    let mut x = beta.inverse_cdf(p);
    for _ in 0..40 {
        let density = beta.pdf(x);
        if !density.is_finite() || density <= 0.0 {
            break;
        }
        let step = (beta.cdf(x) - p) / density;
        if !step.is_finite() {
            break;
        }
        // Halve toward the current iterate rather than leaving (0, 1).
        let next = (x - step).clamp(0.5 * x, 0.5 * (1.0 + x));
        let moved = (next - x).abs();
        x = next;
        if moved <= 1e-16 {
            break;
        }
    }
    x
}

/// Cohen's h effect size for two proportions:
/// `h = 2·asin(√p1) − 2·asin(√p2)`.
///
/// Matches `statsmodels.stats.proportion.proportion_effectsize`; feed the
/// result into [`crate::power::NormalIndPower`].
pub fn proportion_effectsize(prop1: f64, prop2: f64) -> Result<f64> {
    for &p in &[prop1, prop2] {
        if !(0.0..=1.0).contains(&p) {
            return Err(InferustError::InvalidInput(
                "proportions must lie in [0, 1]".into(),
            ));
        }
    }
    Ok(2.0 * prop1.sqrt().asin() - 2.0 * prop2.sqrt().asin())
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
    fn one_sample_ztest() {
        // statsmodels: proportions_ztest(62, 100, value=0.5) -> z ≈ 2.4724, p ≈ 0.01342
        let r = proportions_ztest(&[62], &[100], Some(0.5), Alternative::TwoSided).unwrap();
        assert_close(r.statistic, 2.4724, 1e-3);
        assert_close(r.p_value, 0.01342, 1e-4);
    }

    #[test]
    fn two_sample_ztest_pooled() {
        // statsmodels: proportions_ztest([45, 30], [80, 80])
        // -> z = 2.3763541031440183, p = 0.017484674410521355
        let r = proportions_ztest(&[45, 30], &[80, 80], None, Alternative::TwoSided).unwrap();
        assert_close(r.statistic, 2.3763541031440183, 1e-10);
        assert_close(r.p_value, 0.017484674410521355, 1e-10);
    }

    #[test]
    fn confint_methods_bracket_p_hat() {
        for method in [
            ConfintMethod::Normal,
            ConfintMethod::Wilson,
            ConfintMethod::ClopperPearson,
            ConfintMethod::AgrestiCoull,
            ConfintMethod::Jeffreys,
        ] {
            let (lo, hi) = proportion_confint(62, 100, 0.05, method).unwrap();
            assert!(lo < 0.62 && 0.62 < hi, "{method:?}: [{lo}, {hi}]");
            assert!((0.0..=1.0).contains(&lo) && (0.0..=1.0).contains(&hi));
        }
    }

    #[test]
    fn confint_edge_counts() {
        let (lo, _) = proportion_confint(0, 20, 0.05, ConfintMethod::ClopperPearson).unwrap();
        assert_eq!(lo, 0.0);
        let (_, hi) = proportion_confint(20, 20, 0.05, ConfintMethod::ClopperPearson).unwrap();
        assert_eq!(hi, 1.0);
    }

    #[test]
    fn effectsize_symmetry() {
        let h = proportion_effectsize(0.6, 0.4).unwrap();
        assert_close(h, -proportion_effectsize(0.4, 0.6).unwrap(), 1e-12);
    }
}
