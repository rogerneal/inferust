use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ContinuousCDF, Normal};

use crate::error::{InferustError, Result};
use crate::glm::{Logistic, Poisson};

/// Working correlation structure for GEE.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WorkingCorrelation {
    /// Each observation treated as independent (no within-cluster correlation).
    Independence,
    /// Constant correlation ρ between all pairs within a cluster (compound symmetry).
    Exchangeable,
    /// AR(1) correlation: Corr(y_i, y_j) = ρ^|i-j| within cluster.
    Ar1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeeFamily {
    Binomial,
    Poisson,
}

/// Result of a GEE fit, including population-averaged coefficients and
/// sandwich-corrected (robust) standard errors.
#[derive(Debug, Clone)]
pub struct GeeResult {
    /// Population-averaged coefficient estimates.
    pub coefficients: Vec<f64>,
    /// Sandwich (robust) standard errors.
    pub robust_std_errors: Vec<f64>,
    /// z-statistics from robust SEs.
    pub z_statistics: Vec<f64>,
    /// Two-sided p-values.
    pub p_values: Vec<f64>,
    /// Feature names (including intercept).
    pub feature_names: Vec<String>,
    /// Estimated working-correlation parameter (rho); NaN for Independence.
    pub rho: f64,
    /// Number of outer GEE iterations used.
    pub iterations: usize,
    /// Which GLM family was used.
    pub family: GeeFamily,
    /// Unique cluster count.
    pub cluster_count: usize,
}

#[derive(Debug, Clone)]
pub struct Gee {
    family: GeeFamily,
    working_corr: WorkingCorrelation,
    max_iter: usize,
    tolerance: f64,
    feature_names: Vec<String>,
}

impl Gee {
    pub fn new(family: GeeFamily) -> Self {
        Self {
            family,
            working_corr: WorkingCorrelation::Independence,
            max_iter: 20,
            tolerance: 1e-6,
            feature_names: Vec::new(),
        }
    }

    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    pub fn with_working_correlation(mut self, corr: WorkingCorrelation) -> Self {
        self.working_corr = corr;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Fit a marginal (population-averaged) GLM via GEE.
    ///
    /// Uses alternating estimation of β (via Fisher-scoring with the working
    /// covariance Vᵢ = Aᵢ^{½} Rᵢ(ρ) Aᵢ^{½}) and ρ (via Pearson residuals).
    /// Standard errors are the empirical sandwich estimator, valid even when the
    /// working correlation is mis-specified.
    pub fn fit(&self, x: &[Vec<f64>], y: &[f64], clusters: &[usize]) -> Result<GeeResult> {
        if y.is_empty() {
            return Err(InferustError::InsufficientData { needed: 1, got: 0 });
        }
        if clusters.len() != y.len() {
            return Err(InferustError::DimensionMismatch {
                x_rows: clusters.len(),
                y_len: y.len(),
            });
        }
        let n = y.len();
        let k = if x.is_empty() { 1 } else { x[0].len() + 1 };

        // Build design matrix (intercept prepended)
        let mut x_mat = DMatrix::zeros(n, k);
        for (i, row) in x.iter().enumerate() {
            x_mat[(i, 0)] = 1.0;
            for (j, &v) in row.iter().enumerate() {
                x_mat[(i, j + 1)] = v;
            }
        }

        // Identify unique clusters and per-cluster indices
        let mut cluster_ids: Vec<usize> = clusters.to_vec();
        cluster_ids.sort_unstable();
        cluster_ids.dedup();
        let cluster_count = cluster_ids.len();

        let cluster_indices: Vec<Vec<usize>> = cluster_ids
            .iter()
            .map(|&cid| {
                clusters
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &c)| if c == cid { Some(i) } else { None })
                    .collect()
            })
            .collect();

        // Initialize β via ordinary GLM (independence)
        let mut beta: DVector<f64> = self.init_beta(x, y, k)?;
        let mut rho = 0.0_f64;
        let mut iterations = 0;

        for _iter in 0..self.max_iter {
            iterations += 1;
            let (mu, var_mu) = self.family_mean_var(&x_mat, &beta, n);

            // Pearson residuals: (y - mu) / sqrt(var(mu))
            let pearson: Vec<f64> = (0..n)
                .map(|i| (y[i] - mu[i]) / var_mu[i].sqrt().max(1e-12))
                .collect();

            // Update rho from cross-products of within-cluster Pearson residuals
            let new_rho = estimate_rho(&pearson, &cluster_indices, self.working_corr);

            // GEE score equations: sum_i D_i' V_i^{-1} (y_i - mu_i) = 0
            // where D_i = d(mu_i)/d(beta) = diag(mu_i') X_i
            // V_i = A_i^{1/2} R_i(rho) A_i^{1/2}  (working cov)
            let mut bread = DMatrix::zeros(k, k);
            let mut score = DVector::zeros(k);
            for cl_idx in &cluster_indices {
                let m = cl_idx.len();
                // D_i = diag(dmu/deta) * X_i
                let mut d_i = DMatrix::zeros(m, k);
                for (li, &gi) in cl_idx.iter().enumerate() {
                    let dmu = self.family_dmu_deta(mu[gi]);
                    for j in 0..k {
                        d_i[(li, j)] = dmu * x_mat[(gi, j)];
                    }
                }
                // A_i = diag(var(mu_i))
                let a_sqrt: Vec<f64> = cl_idx
                    .iter()
                    .map(|&gi| var_mu[gi].sqrt().max(1e-12))
                    .collect();
                // R_i working correlation matrix
                let r_inv = working_corr_inv(m, new_rho, self.working_corr);
                // V_i^{-1} = A_i^{-1/2} R_i^{-1} A_i^{-1/2}
                let mut v_inv = DMatrix::zeros(m, m);
                for row in 0..m {
                    for col in 0..m {
                        v_inv[(row, col)] = r_inv[(row, col)] / (a_sqrt[row] * a_sqrt[col]);
                    }
                }
                let r_vec: DVector<f64> =
                    DVector::from_vec(cl_idx.iter().map(|&gi| y[gi] - mu[gi]).collect());
                bread += d_i.transpose() * &v_inv * &d_i;
                score += d_i.transpose() * &v_inv * &r_vec;
            }

            let delta = bread
                .clone()
                .lu()
                .solve(&score)
                .ok_or(InferustError::SingularMatrix)?;
            let max_change = delta.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            beta += delta;
            rho = new_rho;
            if max_change < self.tolerance {
                break;
            }
        }

        // Sandwich variance: (sum D'V⁻¹D)⁻¹ (sum D'V⁻¹ S S' V⁻¹D) (sum D'V⁻¹D)⁻¹
        let (mu, var_mu) = self.family_mean_var(&x_mat, &beta, n);
        let mut bread = DMatrix::zeros(k, k);
        let mut meat = DMatrix::zeros(k, k);
        for cl_idx in &cluster_indices {
            let m = cl_idx.len();
            let mut d_i = DMatrix::zeros(m, k);
            for (li, &gi) in cl_idx.iter().enumerate() {
                let dmu = self.family_dmu_deta(mu[gi]);
                for j in 0..k {
                    d_i[(li, j)] = dmu * x_mat[(gi, j)];
                }
            }
            let a_sqrt: Vec<f64> = cl_idx
                .iter()
                .map(|&gi| var_mu[gi].sqrt().max(1e-12))
                .collect();
            let r_inv = working_corr_inv(m, rho, self.working_corr);
            let mut v_inv = DMatrix::zeros(m, m);
            for row in 0..m {
                for col in 0..m {
                    v_inv[(row, col)] = r_inv[(row, col)] / (a_sqrt[row] * a_sqrt[col]);
                }
            }
            let s_i: DVector<f64> =
                DVector::from_vec(cl_idx.iter().map(|&gi| y[gi] - mu[gi]).collect());
            let dv_inv = d_i.transpose() * &v_inv;
            let u_i = &dv_inv * &s_i;
            bread += &dv_inv * &d_i;
            meat += &u_i * u_i.transpose();
        }

        let bread_inv = bread.try_inverse().ok_or(InferustError::SingularMatrix)?;
        let sandwich = &bread_inv * &meat * &bread_inv;

        let normal = Normal::new(0.0, 1.0).unwrap();
        let coefficients: Vec<f64> = beta.iter().cloned().collect();
        let robust_std_errors: Vec<f64> =
            (0..k).map(|j| sandwich[(j, j)].max(0.0).sqrt()).collect();
        let z_statistics: Vec<f64> = coefficients
            .iter()
            .zip(robust_std_errors.iter())
            .map(|(&c, &se)| if se > 0.0 { c / se } else { f64::NAN })
            .collect();
        let p_values: Vec<f64> = z_statistics
            .iter()
            .map(|&z| 2.0 * normal.cdf(-z.abs()))
            .collect();

        let mut feature_names = vec!["const".to_string()];
        feature_names.extend(self.feature_names.iter().cloned());

        Ok(GeeResult {
            coefficients,
            robust_std_errors,
            z_statistics,
            p_values,
            feature_names,
            rho,
            iterations,
            family: self.family,
            cluster_count,
        })
    }

    fn init_beta(&self, x: &[Vec<f64>], y: &[f64], _k: usize) -> Result<DVector<f64>> {
        let result = match self.family {
            GeeFamily::Binomial => Logistic::new()
                .with_feature_names(self.feature_names.clone())
                .fit(x, y)
                .map(|r| r.coefficients),
            GeeFamily::Poisson => Poisson::new()
                .with_feature_names(self.feature_names.clone())
                .fit(x, y)
                .map(|r| r.coefficients),
        }?;
        Ok(DVector::from_vec(result))
    }

    fn family_mean_var(
        &self,
        x_mat: &DMatrix<f64>,
        beta: &DVector<f64>,
        _n: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let eta = x_mat * beta;
        match self.family {
            GeeFamily::Binomial => {
                let mu: Vec<f64> = eta.iter().map(|&e| 1.0 / (1.0 + (-e).exp())).collect();
                let var_mu: Vec<f64> = mu.iter().map(|&m| (m * (1.0 - m)).max(1e-12)).collect();
                (mu, var_mu)
            }
            GeeFamily::Poisson => {
                let mu: Vec<f64> = eta.iter().map(|&e| e.exp().max(1e-12)).collect();
                let var_mu: Vec<f64> = mu.clone();
                (mu, var_mu)
            }
        }
    }

    fn family_dmu_deta(&self, mu: f64) -> f64 {
        match self.family {
            GeeFamily::Binomial => mu * (1.0 - mu),
            GeeFamily::Poisson => mu,
        }
    }
}

/// Estimate the working correlation parameter ρ from cross-product of
/// Pearson residuals within clusters.
fn estimate_rho(pearson: &[f64], cluster_indices: &[Vec<usize>], corr: WorkingCorrelation) -> f64 {
    match corr {
        WorkingCorrelation::Independence => 0.0,
        WorkingCorrelation::Exchangeable | WorkingCorrelation::Ar1 => {
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            for cl_idx in cluster_indices {
                let m = cl_idx.len();
                for a in 0..m {
                    for b in (a + 1)..m {
                        let lag = (b - a) as i32;
                        let cross = pearson[cl_idx[a]] * pearson[cl_idx[b]];
                        match corr {
                            WorkingCorrelation::Exchangeable => {
                                numerator += cross;
                                denominator += 1.0;
                            }
                            WorkingCorrelation::Ar1 => {
                                // Only use adjacent pairs for AR(1) estimation
                                if lag == 1 {
                                    numerator += cross;
                                    denominator += 1.0;
                                }
                            }
                            WorkingCorrelation::Independence => unreachable!(),
                        }
                    }
                }
            }
            if denominator > 0.0 {
                (numerator / denominator).clamp(-0.999, 0.999)
            } else {
                0.0
            }
        }
    }
}

/// Compute the inverse of the m×m working correlation matrix.
fn working_corr_inv(m: usize, rho: f64, corr: WorkingCorrelation) -> DMatrix<f64> {
    match corr {
        WorkingCorrelation::Independence => DMatrix::identity(m, m),
        WorkingCorrelation::Exchangeable => {
            // R = (1-ρ)I + ρ 11'
            // R⁻¹ via Woodbury: (1-ρ)⁻¹ [I - ρ/(1+(m-1)ρ) * 11']
            let a = 1.0 - rho;
            let d = 1.0 + (m as f64 - 1.0) * rho;
            let c = rho / (a * d);
            let diag_val = 1.0 / a;
            DMatrix::from_fn(m, m, |r, col| if r == col { diag_val - c } else { -c })
        }
        WorkingCorrelation::Ar1 => {
            // Build the AR(1) correlation matrix and invert numerically
            let mut r = DMatrix::zeros(m, m);
            for row in 0..m {
                for col in 0..m {
                    r[(row, col)] = rho.powi((row as i32 - col as i32).abs());
                }
            }
            r.try_inverse().unwrap_or_else(|| DMatrix::identity(m, m))
        }
    }
}

// Keep backward-compatible helper methods on GeeResult
impl GeeResult {
    pub fn cluster_count(&self) -> usize {
        self.cluster_count
    }
}

#[cfg(test)]
mod tests {
    use super::{Gee, GeeFamily, WorkingCorrelation};

    #[test]
    fn fits_independence_poisson_gee() {
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let y = vec![1.0, 2.0, 3.0, 5.0, 8.0];
        let clusters = vec![1, 1, 2, 2, 2];
        let result = Gee::new(GeeFamily::Poisson).fit(&x, &y, &clusters).unwrap();
        assert_eq!(result.cluster_count(), 2);
        assert_eq!(result.coefficients.len(), 2);
        assert!(result
            .robust_std_errors
            .iter()
            .all(|se| se.is_finite() && *se >= 0.0));
    }

    #[test]
    fn fits_exchangeable_poisson_gee_with_correlation() {
        // Two clusters of 3 observations each with Poisson response
        let x = vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
            vec![0.0],
            vec![1.0],
            vec![2.0],
        ];
        let y = vec![1.0, 2.0, 4.0, 2.0, 3.0, 6.0];
        let clusters = vec![1, 1, 1, 2, 2, 2];
        let result = Gee::new(GeeFamily::Poisson)
            .with_working_correlation(WorkingCorrelation::Exchangeable)
            .fit(&x, &y, &clusters)
            .unwrap();
        assert_eq!(result.cluster_count(), 2);
        assert!(result.rho.is_finite());
        assert!(result
            .robust_std_errors
            .iter()
            .all(|se| se.is_finite() && *se >= 0.0));
    }
}
