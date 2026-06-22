use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ContinuousCDF, StudentsT};

use crate::error::{InferustError, Result};
use crate::regression::{OlsResult, Wls};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RobustNorm {
    Huber { tuning: f64 },
}

#[derive(Debug, Clone)]
pub struct RobustLinearResult {
    pub fit: OlsResult,
    pub weights: Vec<f64>,
    pub iterations: usize,
    /// Sandwich (HC) standard errors — robust to heteroskedasticity and
    /// influential observations.  These reflect the M-estimator's actual
    /// variance rather than the homoskedastic WLS assumption in `fit.std_errors`.
    pub robust_std_errors: Vec<f64>,
    /// t-statistics from `robust_std_errors`.
    pub robust_t_statistics: Vec<f64>,
    /// Two-sided p-values from `robust_t_statistics` on `n - k` degrees of freedom.
    pub robust_p_values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct RobustLinearModel {
    norm: RobustNorm,
    max_iter: usize,
    tolerance: f64,
    feature_names: Vec<String>,
}

impl Default for RobustLinearModel {
    fn default() -> Self {
        Self::new()
    }
}

impl RobustLinearModel {
    pub fn new() -> Self {
        Self {
            norm: RobustNorm::Huber { tuning: 1.345 },
            max_iter: 50,
            tolerance: 1e-8,
            feature_names: Vec::new(),
        }
    }

    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    pub fn with_norm(mut self, norm: RobustNorm) -> Self {
        self.norm = norm;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn fit(&self, x: &[Vec<f64>], y: &[f64]) -> Result<RobustLinearResult> {
        if y.is_empty() {
            return Err(InferustError::InsufficientData { needed: 1, got: 0 });
        }
        let n = y.len();
        let mut weights = vec![1.0; n];
        let mut previous = Vec::new();
        let mut final_fit = None;
        let mut final_scale = 1.0;
        let mut iterations = 0;
        for iter in 0..self.max_iter {
            iterations = iter + 1;
            let fit = Wls::new()
                .with_feature_names(self.feature_names.clone())
                .fit(x, y, &weights)?;
            let scale = mad_scale(&fit.residuals).max(1e-12);
            final_scale = scale;
            weights = fit
                .residuals
                .iter()
                .map(|resid| huber_weight(*resid / scale, self.norm))
                .collect();
            let max_change = if previous.len() == fit.coefficients.len() {
                fit.coefficients
                    .iter()
                    .zip(previous.iter())
                    .map(|(a, b): (&f64, &f64)| (a - b).abs())
                    .fold(0.0_f64, f64::max)
            } else {
                f64::INFINITY
            };
            previous = fit.coefficients.clone();
            final_fit = Some(fit);
            if max_change < self.tolerance {
                break;
            }
        }
        let fit = final_fit
            .ok_or_else(|| InferustError::InvalidInput("robust fit did not run".into()))?;

        let k = fit.coefficients.len();
        let df = (n as f64) - (k as f64);
        let robust_std_errors =
            sandwich_se(x, &fit.residuals, &weights, final_scale, self.norm, n, k);

        let t_dist = StudentsT::new(0.0, 1.0, df).ok();
        let robust_t_statistics: Vec<f64> = fit
            .coefficients
            .iter()
            .zip(robust_std_errors.iter())
            .map(|(&c, &se)| if se > 0.0 { c / se } else { f64::NAN })
            .collect();
        let robust_p_values: Vec<f64> = robust_t_statistics
            .iter()
            .map(|&t| match &t_dist {
                Some(d) => 2.0 * d.cdf(-t.abs()),
                None => f64::NAN,
            })
            .collect();

        Ok(RobustLinearResult {
            fit,
            weights,
            iterations,
            robust_std_errors,
            robust_t_statistics,
            robust_p_values,
        })
    }
}

/// Compute sandwich (HC) standard errors for an M-estimator.
///
/// Bread: `A = X'WX` where W = diag of IRWLS final weights (= ψ'/σ for Huber).
/// Meat:  `B = X' diag(ψ²) X` where ψᵢ = huber_psi(eᵢ/σ).
/// Sandwich: `V = A⁻¹ B A⁻¹`.
fn sandwich_se(
    x: &[Vec<f64>],
    residuals: &[f64],
    weights: &[f64],
    scale: f64,
    norm: RobustNorm,
    n: usize,
    k: usize,
) -> Vec<f64> {
    // Build design matrix with intercept column
    let mut x_mat = DMatrix::zeros(n, k);
    for (i, row) in x.iter().enumerate() {
        x_mat[(i, 0)] = 1.0;
        for (j, &v) in row.iter().enumerate() {
            x_mat[(i, j + 1)] = v;
        }
    }

    let w_diag = DMatrix::from_diagonal(&DVector::from_vec(weights.to_vec()));
    let xtwx = x_mat.transpose() * &w_diag * &x_mat;

    let psi_sq: Vec<f64> = residuals
        .iter()
        .map(|&r| {
            let p = huber_psi(r / scale, norm);
            p * p
        })
        .collect();
    let psi_mat = DMatrix::from_diagonal(&DVector::from_vec(psi_sq));
    let meat = x_mat.transpose() * psi_mat * &x_mat;

    let bread_inv = match xtwx.try_inverse() {
        Some(inv) => inv,
        None => return vec![f64::NAN; k],
    };
    let sandwich = &bread_inv * meat * &bread_inv;

    (0..k).map(|j| sandwich[(j, j)].max(0.0).sqrt()).collect()
}

fn huber_psi(value: f64, norm: RobustNorm) -> f64 {
    match norm {
        RobustNorm::Huber { tuning } => {
            if value.abs() <= tuning {
                value
            } else {
                tuning * value.signum()
            }
        }
    }
}

fn huber_weight(value: f64, norm: RobustNorm) -> f64 {
    match norm {
        RobustNorm::Huber { tuning } => {
            let abs = value.abs();
            if abs <= tuning {
                1.0
            } else {
                tuning / abs
            }
        }
    }
}

fn mad_scale(values: &[f64]) -> f64 {
    let mut abs = values.iter().map(|v| v.abs()).collect::<Vec<_>>();
    abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if abs.len() % 2 == 0 {
        (abs[abs.len() / 2 - 1] + abs[abs.len() / 2]) / 2.0
    } else {
        abs[abs.len() / 2]
    };
    median / 0.6744897501960817
}

#[cfg(test)]
mod tests {
    use super::RobustLinearModel;

    #[test]
    fn downweights_outlier() {
        let x = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
            vec![6.0],
        ];
        let y = vec![2.0, 4.1, 6.0, 8.2, 10.0, 40.0];
        let fit = RobustLinearModel::new().fit(&x, &y).unwrap();
        assert!(fit.weights[5] < 1.0);
    }

    #[test]
    fn robust_se_finite_and_positive() {
        let x = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
            vec![6.0],
        ];
        let y = vec![2.0, 4.1, 6.0, 8.2, 10.0, 40.0];
        let fit = RobustLinearModel::new().fit(&x, &y).unwrap();
        assert!(fit
            .robust_std_errors
            .iter()
            .all(|se| se.is_finite() && *se >= 0.0));
        assert!(fit.robust_t_statistics.iter().all(|t| t.is_finite()));
        assert!(fit
            .robust_p_values
            .iter()
            .all(|p| p.is_finite() && *p >= 0.0 && *p <= 1.0));
    }
}
