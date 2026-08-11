use nalgebra::DMatrix;
use statrs::distribution::{ContinuousCDF, Normal};

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
    /// Sandwich (HC) standard errors matching statsmodels RLM `cov='H1'`
    /// (`bcov_scaled` / `bse`).
    pub robust_std_errors: Vec<f64>,
    /// z-statistics from `robust_std_errors` (RLM uses a normal reference).
    pub robust_t_statistics: Vec<f64>,
    /// Two-sided normal p-values from `robust_t_statistics`.
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
        let df_resid = (n as f64) - (k as f64);
        let df_model = (k as f64) - 1.0; // intercept + slopes → rank-1
        let robust_std_errors = sandwich_se(
            x,
            &fit.residuals,
            final_scale,
            self.norm,
            n,
            k,
            df_resid,
            df_model,
        );

        let normal = Normal::new(0.0, 1.0).ok();
        let robust_t_statistics: Vec<f64> = fit
            .coefficients
            .iter()
            .zip(robust_std_errors.iter())
            .map(|(&c, &se)| if se > 0.0 { c / se } else { f64::NAN })
            .collect();
        let robust_p_values: Vec<f64> = robust_t_statistics
            .iter()
            .map(|&t| match &normal {
                // Match statsmodels RLM: normal (not t) p-values via survival fn.
                Some(d) => 2.0 * (1.0 - d.cdf(t.abs())),
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

/// Statsmodels RLM default `cov='H1'` asymptotic covariance diagonal SEs.
///
/// ```text
/// m = mean(ψ'(u)),  v = var(ψ'(u))
/// κ = 1 + (df_model+1)/n · v / m²
/// factor = κ² · (Σψ² / df_resid) · σ² / (mean(ψ'))²
/// V = factor · (X'X)⁻¹
/// ```
/// where `u = e/σ` and `normalized_cov_params ≈ (X'X)⁻¹`.
#[allow(clippy::too_many_arguments)]
fn sandwich_se(
    x: &[Vec<f64>],
    residuals: &[f64],
    scale: f64,
    norm: RobustNorm,
    n: usize,
    k: usize,
    df_resid: f64,
    df_model: f64,
) -> Vec<f64> {
    let mut x_mat = DMatrix::zeros(n, k);
    for (i, row) in x.iter().enumerate() {
        x_mat[(i, 0)] = 1.0;
        for (j, &v) in row.iter().enumerate() {
            x_mat[(i, j + 1)] = v;
        }
    }

    let scale = scale.max(1e-12);
    let mut psi = Vec::with_capacity(n);
    let mut psi_deriv = Vec::with_capacity(n);
    for &e in residuals {
        let u = e / scale;
        psi.push(huber_psi(u, norm));
        psi_deriv.push(huber_psi_deriv(u, norm));
    }

    let mean_pd = psi_deriv.iter().sum::<f64>() / n as f64;
    let var_pd = psi_deriv.iter().map(|d| (d - mean_pd).powi(2)).sum::<f64>() / n as f64;
    let k_corr = 1.0 + (df_model + 1.0) / n as f64 * var_pd / mean_pd.max(1e-15).powi(2);
    let ss_psi = psi.iter().map(|p| p * p).sum::<f64>();
    let factor =
        k_corr.powi(2) * (ss_psi / df_resid.max(1.0)) * scale.powi(2) / mean_pd.max(1e-15).powi(2);

    let xtx = x_mat.transpose() * &x_mat;
    let xtx_inv = match xtx.try_inverse() {
        Some(inv) => inv,
        None => return vec![f64::NAN; k],
    };
    let cov = xtx_inv * factor;
    (0..k).map(|j| cov[(j, j)].max(0.0).sqrt()).collect()
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

fn huber_psi_deriv(value: f64, norm: RobustNorm) -> f64 {
    match norm {
        RobustNorm::Huber { tuning } => {
            if value.abs() <= tuning {
                1.0
            } else {
                0.0
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
    let mut abs: Vec<f64> = values.iter().map(|v| v.abs()).collect();
    let median = quickselect_median(&mut abs);
    median / 0.6744897501960817
}

fn quickselect_median(values: &mut [f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    let mid = n / 2;
    quickselect(values, mid);
    if n.is_multiple_of(2) {
        let upper = values[mid];
        quickselect(&mut values[..mid + 1], mid - 1);
        (values[mid - 1] + upper) / 2.0
    } else {
        values[mid]
    }
}

fn quickselect(slice: &mut [f64], k: usize) {
    let mut left = 0usize;
    let mut right = slice.len().saturating_sub(1);
    loop {
        if left >= right {
            return;
        }
        let pivot = slice_partition(slice, left, right);
        match k.cmp(&pivot) {
            std::cmp::Ordering::Equal => return,
            std::cmp::Ordering::Less => right = pivot.saturating_sub(1),
            std::cmp::Ordering::Greater => left = pivot + 1,
        }
    }
}

fn slice_partition(slice: &mut [f64], left: usize, right: usize) -> usize {
    let pivot_val = slice[right];
    let mut store = left;
    for i in left..right {
        if slice[i] <= pivot_val {
            slice.swap(i, store);
            store += 1;
        }
    }
    slice.swap(store, right);
    store
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
