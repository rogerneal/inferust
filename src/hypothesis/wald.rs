//! Wald tests on arbitrary linear restrictions of estimated coefficients.
//!
//! For an estimator with coefficient vector β ∈ ℝᵏ and covariance Σ, a linear
//! restriction is `R·β = q` for an `r × k` constraint matrix `R` and an
//! `r`-vector `q`. The Wald statistic is
//!
//! ```text
//!     W = (Rβ − q)ᵀ · (R·Σ·Rᵀ)⁻¹ · (Rβ − q)
//! ```
//!
//! Under H₀, `W ~ χ²(r)` asymptotically. For finite-sample OLS the equivalent
//! `F = W / r ~ F(r, df_resid)` form is preferred and reported alongside.

use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor};

use crate::error::{InferustError, Result};

/// Outcome of a linear-restriction Wald test.
#[derive(Debug, Clone)]
pub struct WaldTestResult {
    /// Wald chi-square statistic.
    pub chi2_statistic: f64,
    /// Two-sided p-value from χ²(r).
    pub chi2_p_value: f64,
    /// Finite-sample F-statistic equivalent (`chi2_statistic / r`).
    pub f_statistic: f64,
    /// p-value of the F-statistic. `NaN` when the caller didn't supply a
    /// residual degree-of-freedom (i.e. the estimator is asymptotic only).
    pub f_p_value: f64,
    /// Number of restrictions (rows of `R`).
    pub df_num: usize,
    /// Residual degrees of freedom; `None` for purely asymptotic estimators.
    pub df_den: Option<usize>,
    /// Estimated `Rβ` (length r), useful for reporting effect sizes.
    pub r_beta: Vec<f64>,
    /// Restriction vector `q` (length r).
    pub q: Vec<f64>,
}

impl WaldTestResult {
    /// Print a short summary.
    pub fn print(&self) {
        println!(
            "Wald χ²({}) = {:.4}  p = {:.6}",
            self.df_num, self.chi2_statistic, self.chi2_p_value
        );
        if let Some(df_den) = self.df_den {
            println!(
                "         F({}, {}) = {:.4}  p = {:.6}",
                self.df_num, df_den, self.f_statistic, self.f_p_value
            );
        }
    }
}

/// Run a Wald test of `R·β = q` given an estimated coefficient vector `beta`,
/// its covariance matrix `cov` (k × k, row-major Vec<Vec<f64>>), the
/// restrictions matrix `r` (r_rows × k), and the target vector `q` (length r).
///
/// Pass `df_resid = Some(df)` for finite-sample OLS to report the F-statistic
/// p-value; pass `None` for asymptotic estimators (GLM, Cox PH).
pub fn wald_linear(
    beta: &[f64],
    cov: &[Vec<f64>],
    r: &[Vec<f64>],
    q: &[f64],
    df_resid: Option<usize>,
) -> Result<WaldTestResult> {
    let k = beta.len();
    if cov.len() != k || cov.iter().any(|row| row.len() != k) {
        return Err(InferustError::InvalidInput(format!(
            "covariance must be {k}x{k}"
        )));
    }
    let r_rows = r.len();
    if r_rows == 0 {
        return Err(InferustError::InvalidInput(
            "restriction matrix R must have at least one row".into(),
        ));
    }
    if q.len() != r_rows {
        return Err(InferustError::DimensionMismatch {
            x_rows: r_rows,
            y_len: q.len(),
        });
    }
    if r.iter().any(|row| row.len() != k) {
        return Err(InferustError::InvalidInput(format!(
            "every row of R must have length {k}"
        )));
    }

    let beta_v = DVector::from_column_slice(beta);
    let r_mat = DMatrix::from_row_iterator(r_rows, k, r.iter().flatten().copied());
    let q_v = DVector::from_column_slice(q);
    let cov_mat = DMatrix::from_row_iterator(k, k, cov.iter().flatten().copied());

    let rb = &r_mat * &beta_v;
    let diff = &rb - &q_v;
    let middle = &r_mat * &cov_mat * r_mat.transpose();
    let inv = middle
        .clone()
        .try_inverse()
        .ok_or(InferustError::SingularMatrix)?;
    let chi2 = (diff.transpose() * inv * &diff)[(0, 0)].max(0.0);
    let chi_dist = ChiSquared::new(r_rows as f64)
        .map_err(|_| InferustError::InvalidInput("invalid χ² df".into()))?;
    let chi2_p = 1.0 - chi_dist.cdf(chi2);

    let f_stat = chi2 / r_rows as f64;
    let (df_den, f_p) = if let Some(df) = df_resid {
        let f_dist = FisherSnedecor::new(r_rows as f64, df as f64)
            .map_err(|_| InferustError::InvalidInput("invalid F df".into()))?;
        (Some(df), 1.0 - f_dist.cdf(f_stat))
    } else {
        (None, f64::NAN)
    };

    Ok(WaldTestResult {
        chi2_statistic: chi2,
        chi2_p_value: chi2_p,
        f_statistic: f_stat,
        f_p_value: f_p,
        df_num: r_rows,
        df_den,
        r_beta: rb.iter().copied().collect(),
        q: q.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wald_single_restriction_matches_squared_t() {
        // β = [2.0, 1.0], cov = diag(0.04, 0.09). Restriction: β2 = 0.
        // F-stat for this restriction = (β2/se(β2))² = (1.0/0.3)² = 11.111…
        let beta = vec![2.0, 1.0];
        let cov = vec![vec![0.04, 0.0], vec![0.0, 0.09]];
        let r = vec![vec![0.0, 1.0]];
        let q = vec![0.0];
        let res = wald_linear(&beta, &cov, &r, &q, Some(50)).unwrap();
        let expected = (1.0_f64 / 0.3).powi(2);
        assert!((res.chi2_statistic - expected).abs() < 1e-10);
        assert!((res.f_statistic - expected).abs() < 1e-10);
        assert_eq!(res.df_num, 1);
    }

    #[test]
    fn wald_joint_zero_restriction() {
        // Two-restriction test: β1 = 0 AND β2 = 0.
        let beta = vec![1.0, 0.5, -0.7];
        let cov = vec![
            vec![0.10, 0.01, 0.0],
            vec![0.01, 0.04, 0.0],
            vec![0.0, 0.0, 0.09],
        ];
        // R picks the second and third coefficients.
        let r = vec![vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let q = vec![0.0, 0.0];
        let res = wald_linear(&beta, &cov, &r, &q, Some(40)).unwrap();
        assert!(res.chi2_statistic > 0.0);
        assert_eq!(res.df_num, 2);
        assert_eq!(res.df_den, Some(40));
        // Asymptotic and finite-sample F should be related by W = r * F.
        assert!((res.chi2_statistic / 2.0 - res.f_statistic).abs() < 1e-12);
    }

    #[test]
    fn wald_rejects_bad_shape() {
        let beta = vec![1.0, 2.0];
        let cov = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let r = vec![vec![1.0, 0.0, 0.0]]; // wrong width
        let q = vec![0.0];
        assert!(wald_linear(&beta, &cov, &r, &q, None).is_err());
    }
}
