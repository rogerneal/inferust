//! Statistical power analysis and sample-size calculation.
//!
//! Mirrors `statsmodels.stats.power`:
//! - [`TTestPower`] — one-sample / paired t-test power (`statsmodels.stats.power.TTestPower`).
//! - [`TTestIndPower`] — two-independent-sample t-test power (`TTestIndPower`).
//! - [`NormalIndPower`] — two-sample z-test power, e.g. for proportions (`NormalIndPower`).
//! - [`FTestAnovaPower`] — one-way ANOVA power (`FTestAnovaPower`).
//!
//! Every solver exposes `power(...)` (compute achieved power) and
//! `solve_nobs(...)` (smallest sample size reaching a target power).
//!
//! Effect sizes follow Cohen's conventions: standardized mean difference `d`
//! for t-tests, Cohen's `h` (see [`crate::proportion::proportion_effectsize`])
//! for the z-test, and Cohen's `f` for ANOVA.
//!
//! # Example
//! ```rust
//! use inferust::power::{TTestIndPower, Alternative};
//!
//! // Power of a two-sided two-sample t-test, d = 0.5, 60 per group:
//! let power = TTestIndPower::new()
//!     .power(0.5, 60.0, 0.05, 1.0, Alternative::TwoSided)
//!     .unwrap();
//! assert!(power > 0.7 && power < 0.9);
//!
//! // Sample size per group for 80% power:
//! let n = TTestIndPower::new()
//!     .solve_nobs(0.5, 0.8, 0.05, 1.0, Alternative::TwoSided)
//!     .unwrap();
//! assert!((n - 63.77).abs() < 0.5);
//! ```

use crate::error::{InferustError, Result};
use crate::hypothesis::tukey::gauss_legendre_composite;
use statrs::distribution::{
    ChiSquared, Continuous, ContinuousCDF, FisherSnedecor, Normal, StudentsT,
};
use statrs::function::beta::beta_reg;
use statrs::function::gamma::{gamma_lr, ln_gamma};

/// Direction of the alternative hypothesis, following statsmodels' naming.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Alternative {
    /// Two-sided test (statsmodels `"two-sided"`).
    #[default]
    TwoSided,
    /// One-sided test, effect greater than null (statsmodels `"larger"`).
    Larger,
    /// One-sided test, effect smaller than null (statsmodels `"smaller"`).
    Smaller,
}

// ── Noncentral distributions ──────────────────────────────────────────────────

/// CDF of the noncentral t distribution with `df` degrees of freedom and
/// noncentrality `nc`, evaluated at `t`.
///
/// Computed from the representation T = (Z + nc) / √(V/df) with V ~ χ²_df:
/// P(T ≤ t) = E_V[Φ(t·√(V/df) − nc)]. The expectation is integrated by
/// composite Gauss-Legendre quadrature after the substitution V = w², which
/// removes the integrable singularity at zero for df < 2 and concentrates
/// the integrand in a band of constant width around w = √df.
pub fn noncentral_t_cdf(t: f64, df: f64, nc: f64) -> Result<f64> {
    if df <= 0.0 {
        return Err(InferustError::InvalidInput(
            "noncentral t requires df > 0".into(),
        ));
    }
    if !t.is_finite() {
        return Ok(if t > 0.0 { 1.0 } else { 0.0 });
    }
    let normal = Normal::new(0.0, 1.0).expect("standard normal");
    // log-normalizing constant of chi2(df) plus the |dV/dw| = 2w Jacobian.
    let log_norm = -0.5 * df * std::f64::consts::LN_2 - ln_gamma(0.5 * df) + std::f64::consts::LN_2;
    let sqrt_df = df.sqrt();
    let lo = (sqrt_df - 14.0).max(0.0);
    let hi = (sqrt_df + 14.0).max(16.0);
    let integrand = |w: f64| -> f64 {
        if w <= 0.0 {
            return 0.0;
        }
        let log_pdf = log_norm + (df - 1.0) * w.ln() - 0.5 * w * w;
        if log_pdf < -745.0 {
            return 0.0;
        }
        log_pdf.exp() * normal.cdf(t * w / sqrt_df - nc)
    };
    Ok(gauss_legendre_composite(lo, hi, 48, integrand).clamp(0.0, 1.0))
}

/// CDF of the noncentral F distribution with `(df1, df2)` degrees of freedom
/// and noncentrality `nc`, evaluated at `f`.
///
/// Uses the Poisson mixture representation:
/// P(F ≤ f) = Σ_j Pois(j; nc/2) · I_x(df1/2 + j, df2/2), x = df1·f / (df1·f + df2).
pub fn noncentral_f_cdf(f: f64, df1: f64, df2: f64, nc: f64) -> Result<f64> {
    if df1 <= 0.0 || df2 <= 0.0 || nc < 0.0 {
        return Err(InferustError::InvalidInput(
            "noncentral F requires df1 > 0, df2 > 0, nc >= 0".into(),
        ));
    }
    if f <= 0.0 {
        return Ok(0.0);
    }
    let x = df1 * f / (df1 * f + df2);
    let half_nc = 0.5 * nc;
    // Walk the Poisson weights outward from the modal term for stability.
    let j_mode = half_nc.floor().max(0.0) as usize;
    let log_pois = |j: usize| -> f64 {
        let jf = j as f64;
        -half_nc + jf * half_nc.max(f64::MIN_POSITIVE).ln() - ln_gamma(jf + 1.0)
    };
    let mut total = 0.0;
    // Upward pass from the mode.
    let mut j = j_mode;
    loop {
        let w = log_pois(j).exp();
        let term = w * beta_reg(0.5 * df1 + j as f64, 0.5 * df2, x);
        total += term;
        if (w < 1e-16 || term < 1e-16 * total.max(1e-300)) && j > j_mode + 4 {
            break;
        }
        j += 1;
        if j > j_mode + 100_000 {
            break;
        }
    }
    // Downward pass below the mode.
    let mut j = j_mode;
    while j > 0 {
        j -= 1;
        let w = log_pois(j).exp();
        total += w * beta_reg(0.5 * df1 + j as f64, 0.5 * df2, x);
        if w < 1e-16 {
            break;
        }
    }
    Ok(total.clamp(0.0, 1.0))
}

/// CDF of the noncentral chi-square distribution (Poisson mixture of central
/// chi-squares); used as the df2 → ∞ limit of the noncentral F.
fn noncentral_chi2_cdf(x: f64, df: f64, nc: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let half_nc = 0.5 * nc;
    if half_nc <= 0.0 {
        return gamma_lr(0.5 * df, 0.5 * x);
    }
    let j_mode = half_nc.floor() as usize;
    let log_pois =
        |j: usize| -> f64 { -half_nc + j as f64 * half_nc.ln() - ln_gamma(j as f64 + 1.0) };
    let mut total = 0.0;
    let mut j = j_mode;
    loop {
        let w = log_pois(j).exp();
        total += w * gamma_lr(0.5 * df + j as f64, 0.5 * x);
        if w < 1e-16 && j > j_mode + 4 {
            break;
        }
        j += 1;
        if j > j_mode + 100_000 {
            break;
        }
    }
    let mut j = j_mode;
    while j > 0 {
        j -= 1;
        let w = log_pois(j).exp();
        total += w * gamma_lr(0.5 * df + j as f64, 0.5 * x);
        if w < 1e-16 {
            break;
        }
    }
    total.clamp(0.0, 1.0)
}

// ── t-test power ──────────────────────────────────────────────────────────────

/// Power calculations for the one-sample (or paired) t-test.
///
/// Mirrors `statsmodels.stats.power.TTestPower`.
#[derive(Debug, Clone, Copy, Default)]
pub struct TTestPower;

impl TTestPower {
    /// Create a one-sample t-test power calculator.
    pub fn new() -> Self {
        Self
    }

    /// Achieved power for standardized effect size `effect_size` (Cohen's d),
    /// `nobs` observations, and significance level `alpha`.
    pub fn power(
        &self,
        effect_size: f64,
        nobs: f64,
        alpha: f64,
        alternative: Alternative,
    ) -> Result<f64> {
        let df = nobs - 1.0;
        if df <= 0.0 {
            return Err(InferustError::InvalidInput(
                "t-test power requires nobs > 1".into(),
            ));
        }
        let nc = effect_size * nobs.sqrt();
        t_power(df, nc, alpha, alternative)
    }

    /// Smallest `nobs` achieving `power`. Fails if power is unreachable.
    pub fn solve_nobs(
        &self,
        effect_size: f64,
        power: f64,
        alpha: f64,
        alternative: Alternative,
    ) -> Result<f64> {
        solve_monotone(power, 2.0 + 1e-9, 1e8, |n| {
            self.power(effect_size, n, alpha, alternative)
        })
    }
}

/// Power calculations for the two-independent-sample t-test (pooled df).
///
/// Mirrors `statsmodels.stats.power.TTestIndPower`: `nobs1` is the size of
/// the first group and `ratio` = nobs2 / nobs1.
#[derive(Debug, Clone, Copy, Default)]
pub struct TTestIndPower;

impl TTestIndPower {
    /// Create a two-sample t-test power calculator.
    pub fn new() -> Self {
        Self
    }

    /// Achieved power for effect size `effect_size` (Cohen's d), first-group
    /// size `nobs1`, significance `alpha`, and group-size ratio
    /// `ratio` = nobs2 / nobs1.
    pub fn power(
        &self,
        effect_size: f64,
        nobs1: f64,
        alpha: f64,
        ratio: f64,
        alternative: Alternative,
    ) -> Result<f64> {
        if ratio <= 0.0 {
            return Err(InferustError::InvalidInput("ratio must be > 0".into()));
        }
        let nobs2 = nobs1 * ratio;
        let df = nobs1 + nobs2 - 2.0;
        if df <= 0.0 {
            return Err(InferustError::InvalidInput(
                "t-test power requires nobs1 + nobs2 > 2".into(),
            ));
        }
        let nc = effect_size * (nobs1 * nobs2 / (nobs1 + nobs2)).sqrt();
        t_power(df, nc, alpha, alternative)
    }

    /// Smallest `nobs1` achieving `power` (group 2 has `ratio * nobs1`).
    pub fn solve_nobs(
        &self,
        effect_size: f64,
        power: f64,
        alpha: f64,
        ratio: f64,
        alternative: Alternative,
    ) -> Result<f64> {
        solve_monotone(power, 2.0 + 1e-9, 1e8, |n| {
            self.power(effect_size, n, alpha, ratio, alternative)
        })
    }
}

/// Power calculations for the two-sample z-test.
///
/// Mirrors `statsmodels.stats.power.NormalIndPower`; combine with
/// [`crate::proportion::proportion_effectsize`] for proportion tests.
#[derive(Debug, Clone, Copy, Default)]
pub struct NormalIndPower;

impl NormalIndPower {
    /// Create a two-sample z-test power calculator.
    pub fn new() -> Self {
        Self
    }

    /// Achieved power for standardized effect size `effect_size`, first-group
    /// size `nobs1`, significance `alpha`, and `ratio` = nobs2 / nobs1.
    pub fn power(
        &self,
        effect_size: f64,
        nobs1: f64,
        alpha: f64,
        ratio: f64,
        alternative: Alternative,
    ) -> Result<f64> {
        if ratio <= 0.0 {
            return Err(InferustError::InvalidInput("ratio must be > 0".into()));
        }
        let nobs2 = nobs1 * ratio;
        let nc = effect_size * (nobs1 * nobs2 / (nobs1 + nobs2)).sqrt();
        let normal = Normal::new(0.0, 1.0).expect("standard normal");
        let power = match alternative {
            Alternative::TwoSided => {
                let crit = normal.inverse_cdf(1.0 - alpha / 2.0);
                (1.0 - normal.cdf(crit - nc)) + normal.cdf(-crit - nc)
            }
            Alternative::Larger => {
                let crit = normal.inverse_cdf(1.0 - alpha);
                1.0 - normal.cdf(crit - nc)
            }
            Alternative::Smaller => {
                let crit = normal.inverse_cdf(alpha);
                normal.cdf(crit - nc)
            }
        };
        Ok(power.clamp(0.0, 1.0))
    }

    /// Smallest `nobs1` achieving `power`.
    pub fn solve_nobs(
        &self,
        effect_size: f64,
        power: f64,
        alpha: f64,
        ratio: f64,
        alternative: Alternative,
    ) -> Result<f64> {
        solve_monotone(power, 1.0, 1e9, |n| {
            self.power(effect_size, n, alpha, ratio, alternative)
        })
    }
}

/// Power calculations for the one-way ANOVA F-test.
///
/// Mirrors `statsmodels.stats.power.FTestAnovaPower`: `effect_size` is
/// Cohen's f and `nobs` is the *total* sample size across `k_groups`.
#[derive(Debug, Clone, Copy, Default)]
pub struct FTestAnovaPower;

impl FTestAnovaPower {
    /// Create an ANOVA power calculator.
    pub fn new() -> Self {
        Self
    }

    /// Achieved power for Cohen's f `effect_size`, total sample size `nobs`,
    /// significance `alpha`, and `k_groups` groups.
    pub fn power(&self, effect_size: f64, nobs: f64, alpha: f64, k_groups: f64) -> Result<f64> {
        if k_groups < 2.0 {
            return Err(InferustError::InvalidInput(
                "ANOVA power requires k_groups >= 2".into(),
            ));
        }
        let df1 = k_groups - 1.0;
        let df2 = nobs - k_groups;
        if df2 <= 0.0 {
            return Err(InferustError::InvalidInput(
                "ANOVA power requires nobs > k_groups".into(),
            ));
        }
        let nc = effect_size * effect_size * nobs;
        // For enormous df2 the F test converges to a chi-square test and
        // statrs' F quantile becomes extremely slow, so switch limits.
        if df2 > 1e7 {
            let chi2 = ChiSquared::new(df1)
                .map_err(|_| InferustError::InvalidInput("invalid chi-square parameters".into()))?;
            let crit = refine_upper_quantile(&chi2, 1.0 - alpha, chi2.inverse_cdf(1.0 - alpha));
            return Ok((1.0 - noncentral_chi2_cdf(crit, df1, nc)).clamp(0.0, 1.0));
        }
        let f_dist = FisherSnedecor::new(df1, df2)
            .map_err(|_| InferustError::InvalidInput("invalid F-distribution parameters".into()))?;
        let crit = refine_upper_quantile(&f_dist, 1.0 - alpha, f_dist.inverse_cdf(1.0 - alpha));
        Ok((1.0 - noncentral_f_cdf(crit, df1, df2, nc)?).clamp(0.0, 1.0))
    }

    /// Smallest total `nobs` achieving `power`.
    pub fn solve_nobs(
        &self,
        effect_size: f64,
        power: f64,
        alpha: f64,
        k_groups: f64,
    ) -> Result<f64> {
        solve_monotone(power, k_groups + 1e-9, 1e8, |n| {
            self.power(effect_size, n, alpha, k_groups)
        })
    }
}

// ── shared internals ──────────────────────────────────────────────────────────

/// Polish a quantile estimate with Newton steps on the CDF.
///
/// Guarantees `cdf(x) == p` to machine precision regardless of how accurate the
/// backend's `inverse_cdf` is. statrs 0.17 ended a coarse bisection around 1e-5
/// absolute here, which showed up as ~5e-6 error in the resulting power; 0.19
/// resolves that, so this now converges on the first step. It is retained
/// because `cdf` and `pdf` are the accurate primitives, and pinning the
/// invariant to them keeps power independent of the inverse-CDF implementation.
fn refine_upper_quantile<D>(dist: &D, p: f64, start: f64) -> f64
where
    D: ContinuousCDF<f64, f64> + Continuous<f64, f64>,
{
    let mut x = start;
    for _ in 0..40 {
        let density = dist.pdf(x);
        if !density.is_finite() || density <= 0.0 {
            break;
        }
        let step = (dist.cdf(x) - p) / density;
        if !step.is_finite() {
            break;
        }
        // Keep the iterate on the positive support of F and chi-square.
        let next = (x - step).max(0.5 * x);
        let moved = (next - x).abs();
        x = next;
        if moved <= 1e-15 * x.abs().max(1.0) {
            break;
        }
    }
    x
}

/// Power of a t-test with `df` degrees of freedom and noncentrality `nc`.
fn t_power(df: f64, nc: f64, alpha: f64, alternative: Alternative) -> Result<f64> {
    if !(0.0..1.0).contains(&alpha) || alpha <= 0.0 {
        return Err(InferustError::InvalidInput(
            "alpha must be in (0, 1)".into(),
        ));
    }
    // For enormous df the t distribution is normal to ~1/df and statrs'
    // incomplete-beta evaluation becomes extremely slow, so use the z-test.
    if df > 1e7 {
        let normal = Normal::new(0.0, 1.0).expect("standard normal");
        let power = match alternative {
            Alternative::TwoSided => {
                let crit = normal.inverse_cdf(1.0 - alpha / 2.0);
                (1.0 - normal.cdf(crit - nc)) + normal.cdf(-crit - nc)
            }
            Alternative::Larger => {
                let crit = normal.inverse_cdf(1.0 - alpha);
                1.0 - normal.cdf(crit - nc)
            }
            Alternative::Smaller => {
                let crit = normal.inverse_cdf(alpha);
                normal.cdf(crit - nc)
            }
        };
        return Ok(power.clamp(0.0, 1.0));
    }
    let t_dist = StudentsT::new(0.0, 1.0, df)
        .map_err(|_| InferustError::InvalidInput("invalid t-distribution parameters".into()))?;
    let power = match alternative {
        Alternative::TwoSided => {
            let crit = t_dist.inverse_cdf(1.0 - alpha / 2.0);
            (1.0 - noncentral_t_cdf(crit, df, nc)?) + noncentral_t_cdf(-crit, df, nc)?
        }
        Alternative::Larger => {
            let crit = t_dist.inverse_cdf(1.0 - alpha);
            1.0 - noncentral_t_cdf(crit, df, nc)?
        }
        Alternative::Smaller => {
            let crit = t_dist.inverse_cdf(alpha);
            noncentral_t_cdf(crit, df, nc)?
        }
    };
    Ok(power.clamp(0.0, 1.0))
}

/// Bisection solve of `f(n) = target` for monotonically increasing `f`.
fn solve_monotone<F: Fn(f64) -> Result<f64>>(
    target: f64,
    mut lo: f64,
    mut hi: f64,
    f: F,
) -> Result<f64> {
    if !(0.0..1.0).contains(&target) || target <= 0.0 {
        return Err(InferustError::InvalidInput(
            "target power must be in (0, 1)".into(),
        ));
    }
    if f(hi)? < target {
        return Err(InferustError::InvalidInput(
            "target power is unreachable at the maximum sample size searched".into(),
        ));
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(mid)? < target {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-8 * hi.max(1.0) {
            break;
        }
    }
    Ok(0.5 * (lo + hi))
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
    fn noncentral_t_matches_central_at_zero_nc() {
        // With nc = 0 the noncentral t reduces to the central t.
        let t_dist = StudentsT::new(0.0, 1.0, 7.0).unwrap();
        for &x in &[-2.5, -0.5, 0.0, 1.0, 3.0] {
            assert_close(noncentral_t_cdf(x, 7.0, 0.0).unwrap(), t_dist.cdf(x), 1e-9);
        }
    }

    #[test]
    fn noncentral_f_matches_central_at_zero_nc() {
        let f_dist = FisherSnedecor::new(3.0, 20.0).unwrap();
        for &x in &[0.2, 1.0, 2.5, 6.0] {
            assert_close(
                noncentral_f_cdf(x, 3.0, 20.0, 0.0).unwrap(),
                f_dist.cdf(x),
                1e-10,
            );
        }
    }

    #[test]
    fn ttest_power_known_value() {
        // G*Power / statsmodels: one-sample d=0.5, n=30, alpha=0.05 two-sided
        // => power ≈ 0.7539.
        let p = TTestPower::new()
            .power(0.5, 30.0, 0.05, Alternative::TwoSided)
            .unwrap();
        assert_close(p, 0.7539, 2e-3);
    }

    #[test]
    fn ttest_ind_solve_roundtrip() {
        let solver = TTestIndPower::new();
        let n = solver
            .solve_nobs(0.4, 0.8, 0.05, 1.0, Alternative::TwoSided)
            .unwrap();
        let p = solver
            .power(0.4, n, 0.05, 1.0, Alternative::TwoSided)
            .unwrap();
        assert_close(p, 0.8, 1e-6);
    }

    #[test]
    fn anova_power_monotone_in_n() {
        let solver = FTestAnovaPower::new();
        let p1 = solver.power(0.25, 60.0, 0.05, 3.0).unwrap();
        let p2 = solver.power(0.25, 120.0, 0.05, 3.0).unwrap();
        assert!(p2 > p1);
    }
}
