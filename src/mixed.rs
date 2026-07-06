use std::collections::BTreeMap;

use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ContinuousCDF, StudentsT};

use crate::error::{InferustError, Result};

/// Variance components for a mixed linear model.
#[derive(Debug, Clone)]
pub struct VarianceComponents {
    /// Between-group variance (σ²_b) — random intercept variance.
    pub var_random: f64,
    /// Within-group (residual) variance (σ²_e).
    pub var_residual: f64,
    /// Random slope variance (σ²_u) when a random slope is estimated.
    pub var_slope: Option<f64>,
    /// Intra-class correlation: σ²_b / (σ²_b + σ²_e).
    pub icc: f64,
}

#[derive(Debug, Clone)]
pub struct MixedLinearResult {
    /// Fixed-effect coefficient estimates (intercept first).
    pub coefficients: Vec<f64>,
    /// Standard errors for fixed effects (from REML information matrix).
    pub std_errors: Vec<f64>,
    /// t-statistics for fixed effects.
    pub t_statistics: Vec<f64>,
    /// Two-sided p-values for fixed effects (large-df approximation).
    pub p_values: Vec<f64>,
    /// Feature names (intercept prepended).
    pub feature_names: Vec<String>,
    /// Estimated random intercepts (EBLUP) per group.
    pub random_intercepts: BTreeMap<usize, f64>,
    /// Estimated random slopes (EBLUP) per group when a slope is modelled.
    pub random_slopes: BTreeMap<usize, f64>,
    /// Fitted values (fixed + random).
    pub fitted_values: Vec<f64>,
    /// Residuals (y - fitted).
    pub residuals: Vec<f64>,
    /// Variance component estimates.
    pub variance_components: VarianceComponents,
    /// Number of unique groups.
    pub group_count: usize,
    /// Number of EM iterations used.
    pub iterations: usize,
    /// REML log-likelihood at convergence.
    pub reml_loglik: f64,
}

#[derive(Debug, Clone)]
pub struct MixedLinearModel {
    max_iter: usize,
    tolerance: f64,
    feature_names: Vec<String>,
}

impl Default for MixedLinearModel {
    fn default() -> Self {
        Self::new()
    }
}

impl MixedLinearModel {
    pub fn new() -> Self {
        Self {
            max_iter: 200,
            tolerance: 1e-6,
            feature_names: Vec::new(),
        }
    }

    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Fit a random-intercept LMM via the EM algorithm (REML criterion).
    ///
    /// Model: yᵢⱼ = xᵢⱼ'β + bᵢ + εᵢⱼ,  bᵢ ~ N(0, σ²_b),  εᵢⱼ ~ N(0, σ²_e)
    ///
    /// REML EM update equations (Dempster et al. 1977):
    ///
    ///   σ²_e ← [‖y − Xβ − Zb̂‖² + σ²_e · (n − tr(H))] / n
    ///   σ²_b ← [Σ b̂ᵢ² + σ²_b · (q − Σ h_bᵢ)] / q
    ///
    /// where b̂ = EBLUP random intercepts, H is the hat-on-random component,
    /// and β is updated each M-step via GLS on the current Σ.
    pub fn fit_random_intercept(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
        groups: &[usize],
    ) -> Result<MixedLinearResult> {
        let n = y.len();
        if n == 0 {
            return Err(InferustError::InsufficientData { needed: 1, got: 0 });
        }
        if groups.len() != n {
            return Err(InferustError::DimensionMismatch {
                x_rows: groups.len(),
                y_len: n,
            });
        }

        let k = if x.is_empty() { 1 } else { x[0].len() + 1 };

        // Build design matrix (intercept prepended)
        let mut x_mat = DMatrix::zeros(n, k);
        for (i, row) in x.iter().enumerate() {
            x_mat[(i, 0)] = 1.0;
            for (j, &v) in row.iter().enumerate() {
                x_mat[(i, j + 1)] = v;
            }
        }
        let y_vec = DVector::from_vec(y.to_vec());

        // Unique groups and per-group indices
        let mut group_ids: Vec<usize> = groups.to_vec();
        group_ids.sort_unstable();
        group_ids.dedup();
        let q = group_ids.len();

        let group_indices: Vec<Vec<usize>> = group_ids
            .iter()
            .map(|&gid| {
                groups
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &g)| if g == gid { Some(i) } else { None })
                    .collect()
            })
            .collect();

        // Initial variance components from OLS residuals
        let xt = x_mat.transpose();
        let xtwx = &xt * &x_mat;
        let beta_ols = xtwx
            .clone()
            .lu()
            .solve(&(&xt * &y_vec))
            .ok_or(InferustError::SingularMatrix)?;
        let resid_ols = &y_vec - &x_mat * &beta_ols;
        let var_total = resid_ols.iter().map(|r| r * r).sum::<f64>() / (n as f64 - k as f64);
        let mut sigma2_e = var_total * 0.5;
        let mut sigma2_b = var_total * 0.5;
        let mut beta = beta_ols;

        let mut iterations = 0;

        for _iter in 0..self.max_iter {
            iterations += 1;

            // E-step: compute EBLUP b̂ᵢ and posterior variance for each group
            let mut b_hat = vec![0.0_f64; q];
            let mut var_b_post = vec![0.0_f64; q]; // posterior var of bᵢ
            for (gi, idx) in group_indices.iter().enumerate() {
                let n_i = idx.len() as f64;
                // Shrinkage factor λᵢ = σ²_b / (σ²_b + σ²_e/n_i)
                let lambda = sigma2_b / (sigma2_b + sigma2_e / n_i);
                let resid_i: f64 = idx
                    .iter()
                    .map(|&j| y[j] - (x_mat.row(j) * &beta)[(0, 0)])
                    .sum::<f64>()
                    / n_i;
                b_hat[gi] = lambda * resid_i;
                // Posterior variance of bᵢ given y
                var_b_post[gi] = sigma2_b * (1.0 - lambda);
            }

            // M-step: update β via GLS with current Σ = σ²_b ZZ' + σ²_e I
            // Use the equivalence: β_GLS = (X' V⁻¹ X)⁻¹ X' V⁻¹ y
            // V⁻¹ is block-diagonal; for group i: V_i = σ²_e I_n_i + σ²_b 11'
            // V_i⁻¹ = (1/σ²_e)[I - σ²_b/(σ²_e + n_i σ²_b) 11']
            let mut xvx = DMatrix::zeros(k, k);
            let mut xvy = DVector::zeros(k);
            for idx in &group_indices {
                let n_i = idx.len() as f64;
                let c = sigma2_b / (sigma2_e + n_i * sigma2_b);
                // X_i rows
                let x_i: DMatrix<f64> = DMatrix::from_rows(
                    &idx.iter()
                        .map(|&j| x_mat.row(j).clone_owned())
                        .collect::<Vec<_>>(),
                );
                let y_i: DVector<f64> = DVector::from_vec(idx.iter().map(|&j| y[j]).collect());
                // X_i' V_i⁻¹ X_i = (1/σ²_e)(X_i'X_i - c X_i'11'X_i)
                let xt_i = x_i.transpose();
                let ones = DVector::from_element(idx.len(), 1.0_f64);
                let xt_ones = &xt_i * &ones;
                xvx += (&xt_i * &x_i - c * &xt_ones * xt_ones.transpose()) / sigma2_e;
                let xt_y = &xt_i * &y_i;
                let ones_y: f64 = y_i.sum();
                xvy += (xt_y - c * &xt_ones * ones_y) / sigma2_e;
            }
            let new_beta = xvx
                .clone()
                .lu()
                .solve(&xvy)
                .ok_or(InferustError::SingularMatrix)?;

            // M-step: update variance components
            let mut sse = 0.0_f64;
            let mut trace_h_e = 0.0_f64; // tr(P_e * V_e) term for σ²_e
            let mut ssb = 0.0_f64;
            let mut trace_h_b = 0.0_f64; // shrinkage correction for σ²_b
            for (gi, idx) in group_indices.iter().enumerate() {
                let n_i = idx.len() as f64;
                for &j in idx {
                    let fitted_j = (x_mat.row(j) * &new_beta)[(0, 0)] + b_hat[gi];
                    sse += (y[j] - fitted_j).powi(2);
                }
                // EM correction: σ²_e contribution from posterior uncertainty of bᵢ
                trace_h_e += var_b_post[gi] * n_i;
                ssb += b_hat[gi].powi(2) + var_b_post[gi];
                trace_h_b += var_b_post[gi] / sigma2_b.max(1e-15);
            }
            let new_sigma2_e = (sse + trace_h_e) / (n as f64);
            let new_sigma2_b = ssb / (q as f64);

            let delta_beta = (&new_beta - &beta).norm();
            let delta_e = (new_sigma2_e - sigma2_e).abs();
            let delta_b = (new_sigma2_b - sigma2_b).abs();
            beta = new_beta;
            sigma2_e = new_sigma2_e.max(1e-10);
            sigma2_b = new_sigma2_b.max(0.0);

            let _ = (trace_h_b, delta_beta);
            if delta_e < self.tolerance && delta_b < self.tolerance {
                break;
            }
        }

        // Final EBLUP random intercepts
        let mut random_intercepts: BTreeMap<usize, f64> = BTreeMap::new();
        for (gi, &gid) in group_ids.iter().enumerate() {
            let idx = &group_indices[gi];
            let n_i = idx.len() as f64;
            let lambda = sigma2_b / (sigma2_b + sigma2_e / n_i);
            let resid_i: f64 = idx
                .iter()
                .map(|&j| y[j] - (x_mat.row(j) * &beta)[(0, 0)])
                .sum::<f64>()
                / n_i;
            random_intercepts.insert(gid, lambda * resid_i);
        }

        // Fitted values and residuals
        let fitted_values: Vec<f64> = (0..n)
            .map(|i| {
                let gi = group_ids.iter().position(|&gid| gid == groups[i]).unwrap();
                (x_mat.row(i) * &beta)[(0, 0)] + random_intercepts[&group_ids[gi]]
            })
            .collect();
        let residuals: Vec<f64> = y
            .iter()
            .zip(fitted_values.iter())
            .map(|(yi, fi)| yi - fi)
            .collect();

        // Fixed-effect standard errors from GLS information matrix
        let mut xvx = DMatrix::zeros(k, k);
        for idx in &group_indices {
            let n_i = idx.len() as f64;
            let c = sigma2_b / (sigma2_e + n_i * sigma2_b);
            let x_i: DMatrix<f64> = DMatrix::from_rows(
                &idx.iter()
                    .map(|&j| x_mat.row(j).clone_owned())
                    .collect::<Vec<_>>(),
            );
            let xt_i = x_i.transpose();
            let ones = DVector::from_element(idx.len(), 1.0_f64);
            let xt_ones = &xt_i * &ones;
            xvx += (&xt_i * &x_i - c * &xt_ones * xt_ones.transpose()) / sigma2_e;
        }
        let cov_beta = xvx.try_inverse().ok_or(InferustError::SingularMatrix)?;
        let std_errors: Vec<f64> = (0..k).map(|j| cov_beta[(j, j)].max(0.0).sqrt()).collect();

        let df = (n as f64) - (k as f64) - (q as f64);
        let t_dist = StudentsT::new(0.0, 1.0, df.max(1.0)).ok();
        let coefficients: Vec<f64> = beta.iter().cloned().collect();
        let t_statistics: Vec<f64> = coefficients
            .iter()
            .zip(std_errors.iter())
            .map(|(&c, &se)| if se > 0.0 { c / se } else { f64::NAN })
            .collect();
        let p_values: Vec<f64> = t_statistics
            .iter()
            .map(|&t| match &t_dist {
                Some(d) => 2.0 * d.cdf(-t.abs()),
                None => f64::NAN,
            })
            .collect();

        let icc = if sigma2_b + sigma2_e > 0.0 {
            sigma2_b / (sigma2_b + sigma2_e)
        } else {
            0.0
        };

        // Approximate REML log-likelihood
        let reml_loglik = -0.5
            * (residuals.iter().map(|r| r * r).sum::<f64>() / sigma2_e
                + (n as f64) * sigma2_e.ln()
                + (q as f64) * sigma2_b.ln());

        let mut feature_names = vec!["const".to_string()];
        feature_names.extend(self.feature_names.iter().cloned());

        Ok(MixedLinearResult {
            coefficients,
            std_errors,
            t_statistics,
            p_values,
            feature_names,
            random_intercepts,
            random_slopes: BTreeMap::new(),
            fitted_values,
            residuals,
            variance_components: VarianceComponents {
                var_random: sigma2_b,
                var_residual: sigma2_e,
                var_slope: None,
                icc,
            },
            group_count: q,
            iterations,
            reml_loglik,
        })
    }

    /// Fit a random-intercept + random-slope model on one covariate column.
    ///
    /// Model: `yᵢⱼ = xᵢⱼ'β + bᵢ + uᵢ · xᵢⱼₛ + εᵢⱼ` with independent
    /// `bᵢ ~ N(0, σ²_b)`, `uᵢ ~ N(0, σ²_u)`, `εᵢⱼ ~ N(0, σ²_e)`.
    pub fn fit_random_intercept_slope(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
        groups: &[usize],
        slope_col: usize,
    ) -> Result<MixedLinearResult> {
        let n = y.len();
        if n == 0 {
            return Err(InferustError::InsufficientData { needed: 1, got: 0 });
        }
        if !x.is_empty() && slope_col >= x[0].len() {
            return Err(InferustError::InvalidInput(format!(
                "slope_col {slope_col} out of range for {} predictors",
                x[0].len()
            )));
        }

        let k = if x.is_empty() { 1 } else { x[0].len() + 1 };
        let mut x_mat = DMatrix::zeros(n, k);
        for (i, row) in x.iter().enumerate() {
            x_mat[(i, 0)] = 1.0;
            for (j, &v) in row.iter().enumerate() {
                x_mat[(i, j + 1)] = v;
            }
        }
        let z_slope: Vec<f64> = if x.is_empty() {
            vec![0.0; n]
        } else {
            x.iter().map(|row| row[slope_col]).collect()
        };
        let y_vec = DVector::from_vec(y.to_vec());

        let mut group_ids: Vec<usize> = groups.to_vec();
        group_ids.sort_unstable();
        group_ids.dedup();
        let q = group_ids.len();
        let group_indices: Vec<Vec<usize>> = group_ids
            .iter()
            .map(|&gid| {
                groups
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &g)| if g == gid { Some(i) } else { None })
                    .collect()
            })
            .collect();

        let xt = x_mat.transpose();
        let beta_ols = (&xt * &x_mat)
            .lu()
            .solve(&(&xt * &y_vec))
            .ok_or(InferustError::SingularMatrix)?;
        let resid_ols = &y_vec - &x_mat * &beta_ols;
        let var_total = resid_ols.iter().map(|r| r * r).sum::<f64>() / (n as f64 - k as f64);
        let mut sigma2_e = var_total * 0.4;
        let mut sigma2_b = var_total * 0.3;
        let mut sigma2_u = var_total * 0.3;
        let mut beta = beta_ols;
        let mut iterations = 0;

        for _iter in 0..self.max_iter {
            iterations += 1;
            let mut b_hat = vec![0.0_f64; q];
            let mut u_hat = vec![0.0_f64; q];
            let mut var_b_post = vec![0.0_f64; q];
            let mut var_u_post = vec![0.0_f64; q];

            for (gi, idx) in group_indices.iter().enumerate() {
                let n_i = idx.len() as f64;
                let mean_z: f64 = idx.iter().map(|&j| z_slope[j]).sum::<f64>() / n_i;
                let var_z: f64 = idx
                    .iter()
                    .map(|&j| (z_slope[j] - mean_z).powi(2))
                    .sum::<f64>()
                    / n_i.max(1.0);
                let lambda_b = sigma2_b / (sigma2_b + sigma2_e / n_i);
                let lambda_u = sigma2_u / (sigma2_u + sigma2_e / var_z.max(1e-6));
                let resid_i: f64 = idx
                    .iter()
                    .map(|&j| y[j] - (x_mat.row(j) * &beta)[(0, 0)])
                    .sum::<f64>()
                    / n_i;
                let resid_slope: f64 = idx
                    .iter()
                    .map(|&j| {
                        (y[j] - (x_mat.row(j) * &beta)[(0, 0)] - lambda_b * resid_i) * z_slope[j]
                    })
                    .sum::<f64>()
                    / (idx.iter().map(|&j| z_slope[j].powi(2)).sum::<f64>() + 1e-6);
                b_hat[gi] = lambda_b * resid_i;
                u_hat[gi] = lambda_u * resid_slope;
                var_b_post[gi] = sigma2_b * (1.0 - lambda_b);
                var_u_post[gi] = sigma2_u * (1.0 - lambda_u);
            }

            let mut xvx = DMatrix::zeros(k, k);
            let mut xvy = DVector::zeros(k);
            for idx in &group_indices {
                let n_i = idx.len() as f64;
                let c = sigma2_b / (sigma2_e + n_i * sigma2_b);
                let x_i: DMatrix<f64> = DMatrix::from_rows(
                    &idx.iter()
                        .map(|&j| x_mat.row(j).clone_owned())
                        .collect::<Vec<_>>(),
                );
                let y_i: DVector<f64> = DVector::from_vec(idx.iter().map(|&j| y[j]).collect());
                let xt_i = x_i.transpose();
                let ones = DVector::from_element(idx.len(), 1.0_f64);
                let xt_ones = &xt_i * &ones;
                xvx += (&xt_i * &x_i - c * &xt_ones * xt_ones.transpose()) / sigma2_e;
                let xt_y = &xt_i * &y_i;
                let ones_y: f64 = y_i.sum();
                xvy += (xt_y - c * &xt_ones * ones_y) / sigma2_e;
            }
            beta = xvx
                .clone()
                .lu()
                .solve(&xvy)
                .ok_or(InferustError::SingularMatrix)?;

            let mut sse = 0.0_f64;
            let mut trace_h_e = 0.0_f64;
            let mut ssb = 0.0_f64;
            let mut ssu = 0.0_f64;
            for (gi, idx) in group_indices.iter().enumerate() {
                for &j in idx {
                    let fitted_j =
                        (x_mat.row(j) * &beta)[(0, 0)] + b_hat[gi] + u_hat[gi] * z_slope[j];
                    sse += (y[j] - fitted_j).powi(2);
                }
                trace_h_e += (var_b_post[gi] + var_u_post[gi]) * idx.len() as f64;
                ssb += b_hat[gi].powi(2) + var_b_post[gi];
                ssu += u_hat[gi].powi(2) + var_u_post[gi];
            }
            let new_sigma2_e = (sse + trace_h_e) / (n as f64);
            let new_sigma2_b = ssb / (q as f64);
            let new_sigma2_u = ssu / (q as f64);
            let delta = (new_sigma2_e - sigma2_e).abs()
                + (new_sigma2_b - sigma2_b).abs()
                + (new_sigma2_u - sigma2_u).abs();
            sigma2_e = new_sigma2_e.max(1e-10);
            sigma2_b = new_sigma2_b.max(0.0);
            sigma2_u = new_sigma2_u.max(0.0);
            if delta < self.tolerance {
                break;
            }
        }

        let mut random_intercepts = BTreeMap::new();
        let mut random_slopes = BTreeMap::new();
        for (gi, &gid) in group_ids.iter().enumerate() {
            let idx = &group_indices[gi];
            let n_i = idx.len() as f64;
            let lambda_b = sigma2_b / (sigma2_b + sigma2_e / n_i);
            let mean_z: f64 = idx.iter().map(|&j| z_slope[j]).sum::<f64>() / n_i;
            let var_z: f64 = idx
                .iter()
                .map(|&j| (z_slope[j] - mean_z).powi(2))
                .sum::<f64>()
                / n_i.max(1.0);
            let lambda_u = sigma2_u / (sigma2_u + sigma2_e / var_z.max(1e-6));
            let resid_i: f64 = idx
                .iter()
                .map(|&j| y[j] - (x_mat.row(j) * &beta)[(0, 0)])
                .sum::<f64>()
                / n_i;
            let resid_slope: f64 = idx
                .iter()
                .map(|&j| (y[j] - (x_mat.row(j) * &beta)[(0, 0)] - lambda_b * resid_i) * z_slope[j])
                .sum::<f64>()
                / (idx.iter().map(|&j| z_slope[j].powi(2)).sum::<f64>() + 1e-6);
            random_intercepts.insert(gid, lambda_b * resid_i);
            random_slopes.insert(gid, lambda_u * resid_slope);
        }

        let fitted_values: Vec<f64> = (0..n)
            .map(|i| {
                let gi = group_ids.iter().position(|&gid| gid == groups[i]).unwrap();
                (x_mat.row(i) * &beta)[(0, 0)]
                    + random_intercepts[&group_ids[gi]]
                    + random_slopes[&group_ids[gi]] * z_slope[i]
            })
            .collect();
        let residuals: Vec<f64> = y
            .iter()
            .zip(fitted_values.iter())
            .map(|(yi, fi)| yi - fi)
            .collect();

        let mut xvx = DMatrix::zeros(k, k);
        for idx in &group_indices {
            let n_i = idx.len() as f64;
            let c = sigma2_b / (sigma2_e + n_i * sigma2_b);
            let x_i: DMatrix<f64> = DMatrix::from_rows(
                &idx.iter()
                    .map(|&j| x_mat.row(j).clone_owned())
                    .collect::<Vec<_>>(),
            );
            let xt_i = x_i.transpose();
            let ones = DVector::from_element(idx.len(), 1.0_f64);
            let xt_ones = &xt_i * &ones;
            xvx += (&xt_i * &x_i - c * &xt_ones * xt_ones.transpose()) / sigma2_e;
        }
        let cov_beta = xvx.try_inverse().ok_or(InferustError::SingularMatrix)?;
        let std_errors: Vec<f64> = (0..k).map(|j| cov_beta[(j, j)].max(0.0).sqrt()).collect();
        let df = (n as f64) - (k as f64) - 2.0 * (q as f64);
        let t_dist = StudentsT::new(0.0, 1.0, df.max(1.0)).ok();
        let coefficients: Vec<f64> = beta.iter().cloned().collect();
        let t_statistics: Vec<f64> = coefficients
            .iter()
            .zip(std_errors.iter())
            .map(|(&c, &se)| if se > 0.0 { c / se } else { f64::NAN })
            .collect();
        let p_values: Vec<f64> = t_statistics
            .iter()
            .map(|&t| match &t_dist {
                Some(d) => 2.0 * d.cdf(-t.abs()),
                None => f64::NAN,
            })
            .collect();
        let icc = if sigma2_b + sigma2_e > 0.0 {
            sigma2_b / (sigma2_b + sigma2_e)
        } else {
            0.0
        };
        let reml_loglik = -0.5
            * (residuals.iter().map(|r| r * r).sum::<f64>() / sigma2_e
                + (n as f64) * sigma2_e.ln()
                + (q as f64) * (sigma2_b + sigma2_u).ln());

        let mut feature_names = vec!["const".to_string()];
        feature_names.extend(self.feature_names.iter().cloned());

        Ok(MixedLinearResult {
            coefficients,
            std_errors,
            t_statistics,
            p_values,
            feature_names,
            random_intercepts,
            random_slopes,
            fitted_values,
            residuals,
            variance_components: VarianceComponents {
                var_random: sigma2_b,
                var_residual: sigma2_e,
                var_slope: Some(sigma2_u),
                icc,
            },
            group_count: q,
            iterations,
            reml_loglik,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::MixedLinearModel;

    #[test]
    fn estimates_random_intercepts() {
        let x = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![1.0],
            vec![2.0],
            vec![3.0],
        ];
        let y = vec![3.0, 5.0, 7.0, 6.0, 8.0, 10.0];
        let groups = vec![1, 1, 1, 2, 2, 2];
        let fit = MixedLinearModel::new()
            .fit_random_intercept(&x, &y, &groups)
            .unwrap();
        assert_eq!(fit.group_count, 2);
        assert!(fit.random_intercepts[&1] < fit.random_intercepts[&2]);
    }

    #[test]
    fn variance_components_nonnegative() {
        let x = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![1.5],
            vec![2.5],
            vec![3.5],
            vec![0.5],
            vec![1.5],
            vec![2.5],
        ];
        let y = vec![2.0, 4.0, 6.0, 3.0, 5.0, 7.0, 1.0, 3.0, 5.0];
        let groups = vec![1, 1, 1, 2, 2, 2, 3, 3, 3];
        let fit = MixedLinearModel::new()
            .fit_random_intercept(&x, &y, &groups)
            .unwrap();
        assert!(fit.variance_components.var_random >= 0.0);
        assert!(fit.variance_components.var_residual > 0.0);
        assert!((0.0..=1.0).contains(&fit.variance_components.icc));
    }

    #[test]
    fn std_errors_and_pvalues_finite() {
        let x = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![1.5],
            vec![2.5],
            vec![3.5],
        ];
        let y = vec![2.1, 4.0, 5.9, 3.0, 5.1, 7.0];
        let groups = vec![1, 1, 1, 2, 2, 2];
        let fit = MixedLinearModel::new()
            .fit_random_intercept(&x, &y, &groups)
            .unwrap();
        assert!(fit.std_errors.iter().all(|se| se.is_finite() && *se >= 0.0));
        assert!(fit
            .p_values
            .iter()
            .all(|p| p.is_finite() && *p >= 0.0 && *p <= 1.0));
    }

    #[test]
    fn random_slope_estimates_group_specific_slopes() {
        let x = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![1.0],
            vec![2.0],
            vec![3.0],
        ];
        let y = vec![2.0, 5.0, 8.0, 5.0, 8.0, 11.0];
        let groups = vec![1, 1, 1, 2, 2, 2];
        let fit = MixedLinearModel::new()
            .fit_random_intercept_slope(&x, &y, &groups, 0)
            .unwrap();
        assert_eq!(fit.group_count, 2);
        assert!(fit.variance_components.var_slope.is_some());
        assert!(fit.random_slopes.contains_key(&1));
        assert!(fit.random_slopes.contains_key(&2));
    }
}
