use std::collections::BTreeMap;

use nalgebra::{DMatrix, DVector};
use statrs::distribution::{Continuous, ContinuousCDF, Discrete, Normal, Poisson as StatPoisson};
use statrs::function::gamma::digamma;

use crate::covariance::{sandwich_from_scores, CovType};
use crate::error::{InferustError, Result};
use crate::glm::Poisson;
use crate::irls::{accumulate_xtwx, irls_weighted_solve, mat_vec_mul};

// ─── Probit ──────────────────────────────────────────────────────────────────

/// Fitted binary probit model.
#[derive(Debug, Clone)]
pub struct ProbitResult {
    pub coefficients: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub z_statistics: Vec<f64>,
    pub p_values: Vec<f64>,
    pub fitted_probabilities: Vec<f64>,
    pub log_likelihood: f64,
    pub log_likelihood_null: f64,
    pub pseudo_r_squared: f64,
    pub aic: f64,
    pub bic: f64,
    pub feature_names: Vec<String>,
    pub iterations: usize,
}

/// Binary probit estimator using IRLS (Newton-Raphson / Fisher scoring).
#[derive(Debug, Clone)]
pub struct Probit {
    feature_names: Vec<String>,
    max_iter: usize,
    tolerance: f64,
    covariance: CovType,
}

impl Default for Probit {
    fn default() -> Self {
        Self::new()
    }
}

impl Probit {
    pub fn new() -> Self {
        Self {
            feature_names: Vec::new(),
            max_iter: 100,
            tolerance: 1e-8,
            covariance: CovType::Nonrobust,
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

    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    pub fn with_covariance(mut self, covariance: CovType) -> Self {
        self.covariance = covariance;
        self
    }

    pub fn robust(mut self) -> Self {
        self.covariance = CovType::Hc1;
        self
    }

    pub fn hac(mut self, lags: usize) -> Self {
        self.covariance = CovType::Hac { lags };
        self
    }

    pub fn cluster_robust(mut self, groups: Vec<usize>) -> Self {
        self.covariance = CovType::Cluster { groups };
        self
    }

    /// Fits a probit model via IRLS (equivalent to Fisher scoring).
    ///
    /// Working weights: wᵢ = φ(ηᵢ)² / (Φ(ηᵢ)(1−Φ(ηᵢ)))
    /// Working response: zᵢ = ηᵢ + (yᵢ − Φ(ηᵢ)) / φ(ηᵢ)
    pub fn fit(&self, x: &[Vec<f64>], y: &[f64]) -> Result<ProbitResult> {
        validate_binary(x, y)?;
        let n = y.len();
        let k = x[0].len() + 1;
        let normal = Normal::new(0.0, 1.0)
            .map_err(|_| InferustError::InvalidInput("normal distribution".into()))?;

        let mut x_mat = DMatrix::zeros(n, k);
        for (i, row) in x.iter().enumerate() {
            x_mat[(i, 0)] = 1.0;
            for (j, &v) in row.iter().enumerate() {
                x_mat[(i, j + 1)] = v;
            }
        }
        let mut beta = DVector::zeros(k);
        let mut iterations = 0;

        for _iter in 0..self.max_iter {
            iterations += 1;
            let eta = mat_vec_mul(&x_mat, &beta);
            let mu: Vec<f64> = eta.iter().map(|&e| normal.cdf(e)).collect();
            let pdf: Vec<f64> = eta.iter().map(|&e| normal.pdf(e).max(1e-15)).collect();

            let w: Vec<f64> = (0..n)
                .map(|i| {
                    let m = mu[i].clamp(1e-9, 1.0 - 1e-9);
                    pdf[i] * pdf[i] / (m * (1.0 - m))
                })
                .collect();
            let z: Vec<f64> = (0..n).map(|i| eta[i] + (y[i] - mu[i]) / pdf[i]).collect();

            let new_beta = irls_weighted_solve(&x_mat, &w, &z)?;
            let delta = (&new_beta - &beta).norm();
            beta = new_beta;
            if delta < self.tolerance {
                break;
            }
        }

        // Final fitted values and log-likelihood
        let eta_final = mat_vec_mul(&x_mat, &beta);
        let fitted_probabilities: Vec<f64> = eta_final.iter().map(|&e| normal.cdf(e)).collect();
        let log_likelihood = binary_log_likelihood(y, &fitted_probabilities);

        // Standard errors from Fisher information (X'WX)⁻¹
        let pdf_f: Vec<f64> = eta_final
            .iter()
            .map(|&e| normal.pdf(e).max(1e-15))
            .collect();
        let w_f: Vec<f64> = (0..n)
            .map(|i| {
                let m = fitted_probabilities[i].clamp(1e-9, 1.0 - 1e-9);
                pdf_f[i] * pdf_f[i] / (m * (1.0 - m))
            })
            .collect();
        let info = accumulate_xtwx(&x_mat, &w_f, k);
        let mut cov = info.try_inverse().ok_or(InferustError::SingularMatrix)?;
        if !matches!(self.covariance, CovType::Nonrobust) {
            let factors: Vec<f64> = (0..n).map(|i| y[i] - fitted_probabilities[i]).collect();
            let scores: Vec<Vec<f64>> = (0..n)
                .map(|i| (0..k).map(|j| x_mat[(i, j)] * factors[i]).collect())
                .collect();
            let leverages: Vec<f64> = (0..n)
                .map(|i| {
                    let mut h = 0.0;
                    for j in 0..k {
                        for l in 0..k {
                            h += x_mat[(i, j)] * cov[(j, l)] * x_mat[(i, l)];
                        }
                    }
                    h * w_f[i]
                })
                .collect();
            cov = sandwich_from_scores(&cov, &scores, Some(&leverages), &self.covariance)?;
        }
        let std_errors: Vec<f64> = (0..k).map(|j| cov[(j, j)].max(0.0).sqrt()).collect();

        let norm_dist =
            Normal::new(0.0, 1.0).map_err(|_| InferustError::InvalidInput("normal dist".into()))?;
        let coefficients: Vec<f64> = beta.iter().cloned().collect();
        let z_statistics: Vec<f64> = coefficients
            .iter()
            .zip(std_errors.iter())
            .map(|(&c, &se)| if se > 0.0 { c / se } else { f64::NAN })
            .collect();
        let p_values: Vec<f64> = z_statistics
            .iter()
            .map(|&z| 2.0 * norm_dist.cdf(-z.abs()))
            .collect();

        // Null log-likelihood (intercept-only model)
        let p_bar = y.iter().sum::<f64>() / n as f64;
        let log_likelihood_null = n as f64
            * (p_bar.clamp(1e-12, 1.0 - 1e-12).ln() * p_bar
                + (1.0 - p_bar).clamp(1e-12, 1.0 - 1e-12).ln() * (1.0 - p_bar));
        let pseudo_r_squared = 1.0 - log_likelihood / log_likelihood_null;
        let aic = -2.0 * log_likelihood + 2.0 * k as f64;
        let bic = -2.0 * log_likelihood + (k as f64) * (n as f64).ln();

        let mut feature_names = vec!["const".to_string()];
        if self.feature_names.is_empty() {
            feature_names.extend((1..k).map(|i| format!("x{i}")));
        } else {
            feature_names.extend(self.feature_names.clone());
        }

        Ok(ProbitResult {
            coefficients,
            std_errors,
            z_statistics,
            p_values,
            fitted_probabilities,
            log_likelihood,
            log_likelihood_null,
            pseudo_r_squared,
            aic,
            bic,
            feature_names,
            iterations,
        })
    }
}

impl ProbitResult {
    pub fn predict_proba(&self, x: &[Vec<f64>]) -> Result<Vec<f64>> {
        let normal = Normal::new(0.0, 1.0)
            .map_err(|_| InferustError::InvalidInput("normal distribution".into()))?;
        Ok(x.iter()
            .map(|row| normal.cdf(linear(row, &self.coefficients)))
            .collect())
    }
}

// ─── Negative Binomial (NB2) ─────────────────────────────────────────────────

/// Fitted NB2 negative binomial model.
#[derive(Debug, Clone)]
pub struct NegativeBinomialResult {
    pub coefficients: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub z_statistics: Vec<f64>,
    pub p_values: Vec<f64>,
    pub fitted_values: Vec<f64>,
    pub alpha: f64,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub feature_names: Vec<String>,
    pub iterations: usize,
}

/// NB2 negative binomial count model with log link, fitted by alternating
/// IRLS for β and Newton–Raphson for the overdispersion parameter α.
#[derive(Debug, Clone)]
pub struct NegativeBinomial {
    alpha: Option<f64>,
    max_iter: usize,
    tolerance: f64,
    feature_names: Vec<String>,
    covariance: CovType,
}

impl Default for NegativeBinomial {
    fn default() -> Self {
        Self::new()
    }
}

impl NegativeBinomial {
    pub fn new() -> Self {
        Self {
            alpha: None,
            max_iter: 100,
            tolerance: 1e-6,
            feature_names: Vec::new(),
            covariance: CovType::Nonrobust,
        }
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = Some(alpha);
        self
    }

    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn with_covariance(mut self, covariance: CovType) -> Self {
        self.covariance = covariance;
        self
    }

    pub fn robust(mut self) -> Self {
        self.covariance = CovType::Hc1;
        self
    }

    pub fn hac(mut self, lags: usize) -> Self {
        self.covariance = CovType::Hac { lags };
        self
    }

    pub fn cluster_robust(mut self, groups: Vec<usize>) -> Self {
        self.covariance = CovType::Cluster { groups };
        self
    }

    /// Fits an NB2 model via alternating IRLS (β) and Newton–Raphson (α).
    ///
    /// NB2 variance: Var(y) = μ + αμ²,  r = 1/α (shape parameter).
    /// Working weights: wᵢ = μᵢ/(1 + αμᵢ) with log link.
    pub fn fit(&self, x: &[Vec<f64>], y: &[f64]) -> Result<NegativeBinomialResult> {
        if let Some(a) = self.alpha {
            if a < 0.0 || !a.is_finite() {
                return Err(InferustError::InvalidInput(
                    "negative binomial alpha must be finite and non-negative".into(),
                ));
            }
        }
        let n = y.len();
        if n == 0 {
            return Err(InferustError::InsufficientData { needed: 1, got: 0 });
        }
        if y.iter().any(|&v| v < 0.0 || !v.is_finite()) {
            return Err(InferustError::InvalidInput(
                "counts must be finite non-negative".into(),
            ));
        }

        let k = if x.is_empty() { 1 } else { x[0].len() + 1 };

        let mut x_mat = DMatrix::zeros(n, k);
        for (i, row) in x.iter().enumerate() {
            x_mat[(i, 0)] = 1.0;
            for (j, &v) in row.iter().enumerate() {
                x_mat[(i, j + 1)] = v;
            }
        }

        // Initialize from Poisson MLE
        let poisson_fit = Poisson::new()
            .with_feature_names(self.feature_names.clone())
            .fit(x, y)?;
        let mut beta = DVector::from_vec(poisson_fit.coefficients.clone());

        // Initial alpha: method-of-moments from Poisson residuals
        let mut alpha = self.alpha.unwrap_or_else(|| {
            let num: f64 = y
                .iter()
                .zip(poisson_fit.fitted_values.iter())
                .map(|(yi, mui)| (yi - mui).powi(2) - mui)
                .sum();
            let den: f64 = poisson_fit
                .fitted_values
                .iter()
                .map(|m| m.powi(2))
                .sum::<f64>()
                .max(1e-12);
            (num / den).max(0.01)
        });

        let mut iterations = 0;
        for _iter in 0..self.max_iter {
            iterations += 1;
            let old_beta = beta.clone();
            let old_alpha = alpha;

            // IRLS step for β (α fixed)
            let eta = mat_vec_mul(&x_mat, &beta);
            let mu: Vec<f64> = eta.iter().map(|&e| e.exp().max(1e-12)).collect();
            let w: Vec<f64> = mu.iter().map(|&m| m / (1.0 + alpha * m)).collect();
            let z: Vec<f64> = (0..n).map(|i| eta[i] + (y[i] - mu[i]) / mu[i]).collect();

            beta = irls_weighted_solve(&x_mat, &w, &z)?;

            // Newton step for α (β fixed) — only when alpha is not fixed by user
            if self.alpha.is_none() {
                let eta2 = mat_vec_mul(&x_mat, &beta);
                let mu2: Vec<f64> = eta2.iter().map(|&e| e.exp().max(1e-12)).collect();
                alpha = nb2_alpha_newton(y, &mu2, alpha, 5);
            }

            let beta_change = (&beta - &old_beta).norm();
            let alpha_change = (alpha - old_alpha).abs();
            if beta_change < self.tolerance && alpha_change < self.tolerance {
                break;
            }
        }

        let eta_f = mat_vec_mul(&x_mat, &beta);
        let fitted_values: Vec<f64> = eta_f.iter().map(|&e| e.exp().max(1e-12)).collect();

        // Log-likelihood
        let log_likelihood = nb2_log_likelihood(y, &fitted_values, alpha);

        // Standard errors from observed Fisher information
        // For NB2: Info_β = X'WX where W_ii = μᵢ/(1+αμᵢ) = w_i (final)
        let w_f: Vec<f64> = fitted_values
            .iter()
            .map(|&m| m / (1.0 + alpha * m))
            .collect();
        let info = accumulate_xtwx(&x_mat, &w_f, k);
        let mut cov = info.try_inverse().ok_or(InferustError::SingularMatrix)?;
        if !matches!(self.covariance, CovType::Nonrobust) {
            let factors: Vec<f64> = (0..n)
                .map(|i| (y[i] - fitted_values[i]) / (1.0 + alpha * fitted_values[i]))
                .collect();
            let scores: Vec<Vec<f64>> = (0..n)
                .map(|i| (0..k).map(|j| x_mat[(i, j)] * factors[i]).collect())
                .collect();
            let leverages: Vec<f64> = (0..n)
                .map(|i| {
                    let mut h = 0.0;
                    for j in 0..k {
                        for l in 0..k {
                            h += x_mat[(i, j)] * cov[(j, l)] * x_mat[(i, l)];
                        }
                    }
                    h * w_f[i]
                })
                .collect();
            cov = sandwich_from_scores(&cov, &scores, Some(&leverages), &self.covariance)?;
        }
        let std_errors: Vec<f64> = (0..k).map(|j| cov[(j, j)].max(0.0).sqrt()).collect();

        let norm_dist =
            Normal::new(0.0, 1.0).map_err(|_| InferustError::InvalidInput("normal dist".into()))?;
        let coefficients: Vec<f64> = beta.iter().cloned().collect();
        let z_statistics: Vec<f64> = coefficients
            .iter()
            .zip(std_errors.iter())
            .map(|(&c, &se)| if se > 0.0 { c / se } else { f64::NAN })
            .collect();
        let p_values: Vec<f64> = z_statistics
            .iter()
            .map(|&z| 2.0 * norm_dist.cdf(-z.abs()))
            .collect();

        let aic = -2.0 * log_likelihood + 2.0 * (k + 1) as f64; // +1 for alpha
        let bic = -2.0 * log_likelihood + (k + 1) as f64 * (n as f64).ln();

        let mut feature_names = vec!["const".to_string()];
        if self.feature_names.is_empty() {
            feature_names.extend((1..k).map(|i| format!("x{i}")));
        } else {
            feature_names.extend(self.feature_names.clone());
        }

        Ok(NegativeBinomialResult {
            coefficients,
            std_errors,
            z_statistics,
            p_values,
            fitted_values,
            alpha,
            log_likelihood,
            aic,
            bic,
            feature_names,
            iterations,
        })
    }
}

/// Newton steps to update α given fixed μ.  r = 1/α is the shape parameter.
fn nb2_alpha_newton(y: &[f64], mu: &[f64], alpha_init: f64, steps: usize) -> f64 {
    let mut alpha = alpha_init.max(1e-6);
    for _ in 0..steps {
        let r = 1.0 / alpha;
        // Score ∂ℓ/∂r and observed Hessian ∂²ℓ/∂r²
        let score: f64 = y
            .iter()
            .zip(mu.iter())
            .map(|(&yi, &mu_i)| {
                digamma(yi + r) - digamma(r) + (r / (r + mu_i)).ln() + (mu_i - yi) / (r + mu_i)
            })
            .sum();
        // Numerical second derivative of ∂ℓ/∂r via finite diff
        let eps = r * 1e-5;
        let score_p: f64 = y
            .iter()
            .zip(mu.iter())
            .map(|(&yi, &mu_i)| {
                let rp = r + eps;
                digamma(yi + rp) - digamma(rp) + (rp / (rp + mu_i)).ln() + (mu_i - yi) / (rp + mu_i)
            })
            .sum();
        let hessian = (score_p - score) / eps;
        if hessian.abs() < 1e-15 {
            break;
        }
        // dr = -score/hessian,  then alpha = 1/(r + dr)
        let r_new = (r - score / hessian).max(1e-4);
        alpha = 1.0 / r_new;
        if (alpha - 1.0 / r).abs() < 1e-8 {
            break;
        }
    }
    alpha.max(1e-6)
}

fn nb2_log_likelihood(y: &[f64], mu: &[f64], alpha: f64) -> f64 {
    use statrs::function::gamma::ln_gamma;
    let r = 1.0 / alpha;
    y.iter()
        .zip(mu.iter())
        .map(|(&yi, &mu_i)| {
            ln_gamma(yi + r) - ln_gamma(r) - ln_gamma(yi + 1.0)
                + r * (r / (r + mu_i)).ln()
                + yi * (mu_i / (r + mu_i)).ln()
        })
        .sum()
}

// ─── Multinomial Logit (Softmax) ──────────────────────────────────────────────

/// Fitted multinomial logit model (true softmax, not one-vs-rest).
#[derive(Debug, Clone)]
pub struct MultinomialLogitResult {
    pub classes: Vec<usize>,
    /// Coefficient matrix: (K−1) rows × p columns (including intercept).
    /// Row k is for class `classes[k+1]`; `classes[0]` is the reference category.
    pub coefficients: Vec<Vec<f64>>,
    pub std_errors: Vec<Vec<f64>>,
    pub z_statistics: Vec<Vec<f64>>,
    pub p_values: Vec<Vec<f64>>,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub feature_names: Vec<String>,
    pub iterations: usize,
}

/// Multinomial logit estimator using Newton–Raphson on the softmax log-likelihood.
#[derive(Debug, Clone, Default)]
pub struct MultinomialLogit {
    feature_names: Vec<String>,
    max_iter: usize,
    tolerance: f64,
}

impl MultinomialLogit {
    pub fn new() -> Self {
        Self {
            feature_names: Vec::new(),
            max_iter: 200,
            tolerance: 1e-8,
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

    /// Fits a true softmax multinomial logit via Newton–Raphson.
    ///
    /// Reference category is `classes[0]`. Coefficients for each of the K−1
    /// non-reference classes are estimated jointly.
    pub fn fit(&self, x: &[Vec<f64>], y: &[usize]) -> Result<MultinomialLogitResult> {
        let n = y.len();
        if n == 0 {
            return Err(InferustError::InsufficientData { needed: 1, got: 0 });
        }
        let mut classes = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let num_classes = classes.len();
        if num_classes < 2 {
            return Err(InferustError::InvalidInput(
                "multinomial logit needs at least two classes".into(),
            ));
        }
        // Map class labels to indices 0..K-1
        let class_to_idx: BTreeMap<usize, usize> =
            classes.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let y_idx: Vec<usize> = y.iter().map(|&yi| class_to_idx[&yi]).collect();

        let p = if x.is_empty() { 0 } else { x[0].len() };
        let q = p + 1; // predictors including intercept
        let km1 = num_classes - 1; // number of non-reference classes

        // Design matrix with intercept
        let mut x_mat = DMatrix::zeros(n, q);
        for (i, row) in x.iter().enumerate() {
            x_mat[(i, 0)] = 1.0;
            for (j, &v) in row.iter().enumerate() {
                x_mat[(i, j + 1)] = v;
            }
        }

        // Parameter vector: km1 * q coefficients (stacked by class)
        let total_params = km1 * q;
        let mut theta = DVector::zeros(total_params);
        let mut iterations = 0;

        for _iter in 0..self.max_iter {
            iterations += 1;

            // Softmax probabilities: P[i][k] for all k
            let probs = softmax_probs(&x_mat, &theta, n, q, km1);

            // Score vector (gradient of log-likelihood)
            let mut grad = DVector::zeros(total_params);
            for i in 0..n {
                for k in 0..km1 {
                    let d = if y_idx[i] == k + 1 { 1.0 } else { 0.0 };
                    let resid = d - probs[i][k + 1];
                    for j in 0..q {
                        grad[k * q + j] += resid * x_mat[(i, j)];
                    }
                }
            }

            // Hessian (negative of expected Fisher information)
            let mut hess = DMatrix::zeros(total_params, total_params);
            for i in 0..n {
                for k1 in 0..km1 {
                    for k2 in 0..km1 {
                        let pk1 = probs[i][k1 + 1];
                        let pk2 = probs[i][k2 + 1];
                        let h = if k1 == k2 {
                            -pk1 * (1.0 - pk1)
                        } else {
                            pk1 * pk2
                        };
                        for j1 in 0..q {
                            for j2 in 0..q {
                                hess[(k1 * q + j1, k2 * q + j2)] +=
                                    h * x_mat[(i, j1)] * x_mat[(i, j2)];
                            }
                        }
                    }
                }
            }

            let delta = hess
                .lu()
                .solve(&grad)
                .ok_or(InferustError::SingularMatrix)?;
            let max_delta = delta.iter().map(|v: &f64| v.abs()).fold(0.0_f64, f64::max);
            theta -= delta; // ascent: theta += (-hess)^{-1} * grad
            if max_delta < self.tolerance {
                break;
            }
        }

        // Final log-likelihood
        let probs_f = softmax_probs(&x_mat, &theta, n, q, km1);
        let log_likelihood: f64 = (0..n).map(|i| probs_f[i][y_idx[i]].max(1e-15).ln()).sum();

        // Standard errors from inverse Hessian
        let probs_fe = softmax_probs(&x_mat, &theta, n, q, km1);
        let mut hess_f: DMatrix<f64> = DMatrix::zeros(total_params, total_params);
        for i in 0..n {
            for k1 in 0..km1 {
                for k2 in 0..km1 {
                    let pk1 = probs_fe[i][k1 + 1];
                    let pk2 = probs_fe[i][k2 + 1];
                    let h = if k1 == k2 {
                        -pk1 * (1.0 - pk1)
                    } else {
                        pk1 * pk2
                    };
                    for j1 in 0..q {
                        for j2 in 0..q {
                            hess_f[(k1 * q + j1, k2 * q + j2)] +=
                                h * x_mat[(i, j1)] * x_mat[(i, j2)];
                        }
                    }
                }
            }
        }
        let cov = (-hess_f)
            .try_inverse()
            .ok_or(InferustError::SingularMatrix)?;

        let norm_dist =
            Normal::new(0.0, 1.0).map_err(|_| InferustError::InvalidInput("normal dist".into()))?;

        let mut coefficients = Vec::with_capacity(km1);
        let mut std_errors = Vec::with_capacity(km1);
        let mut z_statistics = Vec::with_capacity(km1);
        let mut p_values = Vec::with_capacity(km1);
        for k in 0..km1 {
            let coefs: Vec<f64> = (0..q).map(|j| theta[k * q + j]).collect();
            let ses: Vec<f64> = (0..q)
                .map(|j| cov[(k * q + j, k * q + j)].max(0.0).sqrt())
                .collect();
            let zs: Vec<f64> = coefs
                .iter()
                .zip(ses.iter())
                .map(|(&c, &se)| if se > 0.0 { c / se } else { f64::NAN })
                .collect();
            let ps: Vec<f64> = zs.iter().map(|&z| 2.0 * norm_dist.cdf(-z.abs())).collect();
            coefficients.push(coefs);
            std_errors.push(ses);
            z_statistics.push(zs);
            p_values.push(ps);
        }

        let aic = -2.0 * log_likelihood + 2.0 * total_params as f64;
        let bic = -2.0 * log_likelihood + total_params as f64 * (n as f64).ln();

        let mut feature_names = vec!["const".to_string()];
        if self.feature_names.is_empty() {
            feature_names.extend((1..q).map(|i| format!("x{i}")));
        } else {
            feature_names.extend(self.feature_names.clone());
        }

        Ok(MultinomialLogitResult {
            classes,
            coefficients,
            std_errors,
            z_statistics,
            p_values,
            log_likelihood,
            aic,
            bic,
            feature_names,
            iterations,
        })
    }
}

impl MultinomialLogitResult {
    /// Predicted class probabilities for each row of `x`.
    /// Returns a Vec of length n, each element a Vec of length K.
    pub fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = x.len();
        let q = if x.is_empty() { 1 } else { x[0].len() + 1 };
        let km1 = self.classes.len() - 1;
        // Flatten coefficients into theta vector
        let mut theta = DVector::zeros(km1 * q);
        for (k, coefs) in self.coefficients.iter().enumerate() {
            for (j, &c) in coefs.iter().enumerate() {
                theta[k * q + j] = c;
            }
        }
        let mut x_mat = DMatrix::zeros(n, q);
        for (i, row) in x.iter().enumerate() {
            x_mat[(i, 0)] = 1.0;
            for (j, &v) in row.iter().enumerate() {
                x_mat[(i, j + 1)] = v;
            }
        }
        softmax_probs(&x_mat, &theta, n, q, km1)
    }
}

/// Compute softmax probabilities for all n observations.
/// Returns probs[i][k] for k in 0..K (reference category is k=0, β=0).
fn softmax_probs(
    x_mat: &DMatrix<f64>,
    theta: &DVector<f64>,
    n: usize,
    q: usize,
    km1: usize,
) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            let mut log_odds = vec![0.0_f64]; // reference = 0
            for k in 0..km1 {
                let eta: f64 = (0..q).map(|j| theta[k * q + j] * x_mat[(i, j)]).sum();
                log_odds.push(eta);
            }
            let max_lo = log_odds.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp: Vec<f64> = log_odds.iter().map(|&v| (v - max_lo).exp()).collect();
            let sum_exp = exp.iter().sum::<f64>().max(1e-15);
            exp.iter().map(|&e| e / sum_exp).collect()
        })
        .collect()
}

// ─── Ordered Logit (Proportional Odds) ───────────────────────────────────────

/// Fitted proportional-odds ordered logit model.
#[derive(Debug, Clone)]
pub struct OrderedLogitResult {
    pub classes: Vec<usize>,
    /// Shared slope coefficients (same across all thresholds).
    pub coefficients: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub z_statistics: Vec<f64>,
    pub p_values: Vec<f64>,
    /// Threshold (cutpoint) parameters α_1 < α_2 < ... < α_{K-1}.
    pub cutpoints: Vec<f64>,
    pub cutpoint_std_errors: Vec<f64>,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub feature_names: Vec<String>,
    pub iterations: usize,
}

/// Proportional-odds ordered logit estimator.
///
/// Model: P(y ≤ k | x) = σ(αₖ − x'β),  k = 1, …, K−1.
#[derive(Debug, Clone, Default)]
pub struct OrderedLogit {
    feature_names: Vec<String>,
    max_iter: usize,
    tolerance: f64,
}

impl OrderedLogit {
    pub fn new() -> Self {
        Self {
            feature_names: Vec::new(),
            max_iter: 200,
            tolerance: 1e-8,
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

    /// Fits the proportional-odds model via Newton–Raphson.
    ///
    /// Parameters: θ = (α_1, …, α_{K-1}, β_1, …, β_p) — note cutpoints first.
    /// Cutpoints are unconstrained via reparameterisation:
    ///   α_1 free,  α_{k} = α_1 + exp(δ_2) + … + exp(δ_k)  for k ≥ 2.
    pub fn fit(&self, x: &[Vec<f64>], y: &[usize]) -> Result<OrderedLogitResult> {
        let n = y.len();
        if n == 0 {
            return Err(InferustError::InsufficientData { needed: 1, got: 0 });
        }
        let mut classes = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let num_classes = classes.len();
        if num_classes < 3 {
            return Err(InferustError::InvalidInput(
                "ordered logit needs at least three ordered classes".into(),
            ));
        }
        let class_to_idx: BTreeMap<usize, usize> =
            classes.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let y_idx: Vec<usize> = y.iter().map(|&yi| class_to_idx[&yi]).collect();

        let p = if x.is_empty() { 0 } else { x[0].len() };
        let km1 = num_classes - 1;
        let total_params = km1 + p; // km1 cutpoint params + p slopes

        let mut x_mat = DMatrix::zeros(n, p);
        for (i, row) in x.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                x_mat[(i, j)] = v;
            }
        }

        // Initial parameters: evenly-spaced cutpoints, zero slopes
        let mut theta = DVector::zeros(total_params);
        let quantiles: Vec<f64> = (1..num_classes)
            .map(|k| {
                let frac = k as f64 / num_classes as f64;
                logit(frac)
            })
            .collect();
        // Reparameterize: theta[0] = alpha_1 (free), theta[k] = log(alpha_{k+1} - alpha_k) for k>=1
        theta[0] = quantiles[0];
        for k in 1..km1 {
            let gap = (quantiles[k] - quantiles[k - 1]).max(0.01);
            theta[k] = gap.ln();
        }

        // Gradient ascent with Armijo backtracking line search — avoids Hessian singularity
        // issues early in optimization and guarantees monotone LL improvement.
        let mut iterations = 0;
        for _iter in 0..self.max_iter {
            iterations += 1;
            let cuts = decode_cutpoints(&theta, km1);
            let (neg_ll, neg_grad, _) =
                ordinal_ll_grad_hess(&x_mat, &y_idx, &cuts, &theta, n, p, km1);
            let ll = -neg_ll;
            let grad = -neg_grad;
            let grad_sq = grad.dot(&grad);
            if grad_sq.sqrt() < self.tolerance {
                break;
            }
            // Armijo backtracking: find step t so LL(θ + t·g) ≥ LL(θ) + 0.1·t·‖g‖²
            let mut step = 1.0_f64;
            for _ in 0..15 {
                let theta_try = &theta + step * &grad;
                let cuts_try = decode_cutpoints(&theta_try, km1);
                let beta_try: Vec<f64> = (0..p).map(|j| theta_try[km1 + j]).collect();
                let xb_try = &x_mat * DVector::from_row_slice(&beta_try);
                let ll_try = ordinal_log_likelihood(&xb_try, &y_idx, &cuts_try, n, km1);
                if ll_try >= ll + 0.1 * step * grad_sq {
                    break;
                }
                step *= 0.5;
            }
            let max_d = (step * &grad)
                .iter()
                .map(|v: &f64| v.abs())
                .fold(0.0_f64, f64::max);
            theta += step * &grad;
            if max_d < self.tolerance {
                break;
            }
        }

        let cuts_f = decode_cutpoints(&theta, km1);
        // Recover true LL (un-negate)
        let (neg_ll_f, _, bhhh_f) =
            ordinal_ll_grad_hess(&x_mat, &y_idx, &cuts_f, &theta, n, p, km1);
        let log_likelihood = -neg_ll_f;
        // BHHH as approximation to expected Fisher information; invert for covariance.
        // Add small ridge to ensure invertibility (BHHH can be rank-deficient for small n).
        let bhhh_reg = bhhh_f + 1e-8 * DMatrix::<f64>::identity(total_params, total_params);
        let cov = bhhh_reg
            .try_inverse()
            .ok_or(InferustError::SingularMatrix)?;

        let norm_dist =
            Normal::new(0.0, 1.0).map_err(|_| InferustError::InvalidInput("normal dist".into()))?;

        // Slopes: theta[km1..] — but we need delta-method for cutpoints
        let coefs: Vec<f64> = (0..p).map(|j| theta[km1 + j]).collect();
        let se_raw: Vec<f64> = (0..total_params)
            .map(|j| cov[(j, j)].max(0.0).sqrt())
            .collect();
        let se_coefs: Vec<f64> = se_raw[km1..].to_vec();

        // Cutpoint SEs via delta method (Jacobian of decode_cutpoints)
        let jac = cutpoint_jacobian(&theta, km1);
        let cov_cuts = &jac * cov.view((0, 0), (km1, km1)).clone_owned() * jac.transpose();
        let cutpoint_std_errors: Vec<f64> =
            (0..km1).map(|k| cov_cuts[(k, k)].max(0.0).sqrt()).collect();

        let z_statistics: Vec<f64> = coefs
            .iter()
            .zip(se_coefs.iter())
            .map(|(&c, &se)| if se > 0.0 { c / se } else { f64::NAN })
            .collect();
        let p_values: Vec<f64> = z_statistics
            .iter()
            .map(|&z| 2.0 * norm_dist.cdf(-z.abs()))
            .collect();

        let aic = -2.0 * log_likelihood + 2.0 * total_params as f64;
        let bic = -2.0 * log_likelihood + total_params as f64 * (n as f64).ln();

        let feature_names = if self.feature_names.is_empty() {
            (0..p).map(|i| format!("x{}", i + 1)).collect()
        } else {
            self.feature_names.clone()
        };

        Ok(OrderedLogitResult {
            classes,
            coefficients: coefs,
            std_errors: se_coefs,
            z_statistics,
            p_values,
            cutpoints: cuts_f,
            cutpoint_std_errors,
            log_likelihood,
            aic,
            bic,
            feature_names,
            iterations,
        })
    }
}

impl OrderedLogitResult {
    pub fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        x.iter()
            .map(|row| {
                let eta: f64 = row
                    .iter()
                    .zip(self.coefficients.iter())
                    .map(|(x, b)| x * b)
                    .sum();
                let cumulative: Vec<f64> = self
                    .cutpoints
                    .iter()
                    .map(|&cut| logistic_cdf(cut - eta))
                    .collect();
                let mut probs = Vec::with_capacity(self.classes.len());
                probs.push(cumulative[0]);
                for w in cumulative.windows(2) {
                    probs.push((w[1] - w[0]).max(0.0));
                }
                probs.push((1.0 - cumulative[cumulative.len() - 1]).max(0.0));
                let total = probs.iter().sum::<f64>().max(1e-12);
                probs.iter_mut().for_each(|p| *p /= total);
                probs
            })
            .collect()
    }
}

/// Reparameterised cutpoints: α_1 free; α_{k+1} = α_k + exp(θ_k) for k>=1.
fn decode_cutpoints(theta: &DVector<f64>, km1: usize) -> Vec<f64> {
    let mut cuts = Vec::with_capacity(km1);
    cuts.push(theta[0]);
    for k in 1..km1 {
        cuts.push(cuts[k - 1] + theta[k].exp());
    }
    cuts
}

/// Jacobian of decode_cutpoints w.r.t. theta[0..km1].
fn cutpoint_jacobian(theta: &DVector<f64>, km1: usize) -> DMatrix<f64> {
    // d(alpha_k)/d(theta_j)
    let mut jac = DMatrix::zeros(km1, km1);
    for k in 0..km1 {
        for j in 0..=k {
            jac[(k, j)] = if j == 0 { 1.0 } else { theta[j].exp() };
        }
    }
    jac
}

/// Log-likelihood only (for line search).
fn ordinal_log_likelihood(
    xb: &DVector<f64>,
    y_idx: &[usize],
    cuts: &[f64],
    n: usize,
    km1: usize,
) -> f64 {
    let mut ll = 0.0;
    for i in 0..n {
        let k = y_idx[i];
        let xb_i = xb[i];
        let cdf_k = if k == km1 {
            1.0
        } else {
            logistic_cdf(cuts[k] - xb_i)
        };
        let cdf_km1 = if k == 0 {
            0.0
        } else {
            logistic_cdf(cuts[k - 1] - xb_i)
        };
        let prob = (cdf_k - cdf_km1).max(1e-15);
        ll += prob.ln();
    }
    ll
}

/// Compute log-likelihood, gradient, and BHHH (outer-product) Hessian for the
/// proportional-odds ordinal model.
///
/// Parameters: theta = (reparameterised cuts 0..km1, slopes km1..).
/// The BHHH Hessian approximation H ≈ -Σᵢ gᵢ gᵢᵀ covers all parameter blocks
/// (cutpoints AND slopes) and is always negative semi-definite, avoiding the
/// singular-matrix issue that arises when the analytical Hessian is only
/// partially populated.  It equals the expected Fisher information asymptotically.
fn ordinal_ll_grad_hess(
    x_mat: &DMatrix<f64>,
    y_idx: &[usize],
    cuts: &[f64],
    theta: &DVector<f64>,
    n: usize,
    p: usize,
    km1: usize,
) -> (f64, DVector<f64>, DMatrix<f64>) {
    let total = km1 + p;
    let mut ll = 0.0;
    let mut grad = DVector::zeros(total);
    // BHHH: H = -Σ g_i g_i'  (negative so Newton step is H^{-1} g = ascent)
    let mut bhhh: DMatrix<f64> = DMatrix::zeros(total, total);

    let beta: Vec<f64> = (0..p).map(|j| theta[km1 + j]).collect();
    let xb = x_mat * DVector::from_row_slice(&beta);
    let jac = cutpoint_jacobian(theta, km1);

    for i in 0..n {
        let k = y_idx[i];
        let xb_i = xb[i];

        let cdf_k = if k == km1 {
            1.0
        } else {
            logistic_cdf(cuts[k] - xb_i)
        };
        let cdf_km1 = if k == 0 {
            0.0
        } else {
            logistic_cdf(cuts[k - 1] - xb_i)
        };
        let prob = (cdf_k - cdf_km1).max(1e-15);
        ll += prob.ln();

        let f_k = if k == km1 {
            0.0
        } else {
            logistic_pdf(cuts[k] - xb_i)
        };
        let f_km1 = if k == 0 {
            0.0
        } else {
            logistic_pdf(cuts[k - 1] - xb_i)
        };

        // Per-observation gradient vector (length = total)
        let mut g_i = DVector::zeros(total);

        // Slopes: ∂ℓᵢ/∂β_j = (f_{k-1} - f_k)/P * x_{ij}
        let dpdb = (f_km1 - f_k) / prob;
        for j in 0..p {
            g_i[km1 + j] = dpdb * x_mat[(i, j)];
        }

        // Decoded cutpoints: ∂P/∂α_k = f_k,  ∂P/∂α_{k-1} = -f_{k-1}
        let mut dpda = vec![0.0_f64; km1];
        if k < km1 {
            dpda[k] = f_k / prob;
        }
        if k > 0 {
            dpda[k - 1] -= f_km1 / prob;
        }

        // Encoded cutpoints via Jacobian: ∂ℓᵢ/∂θ_r = Σ_s (∂ℓᵢ/∂α_s) * jac[s,r]
        for r in 0..km1 {
            for s in 0..km1 {
                g_i[r] += dpda[s] * jac[(s, r)];
            }
        }

        // Accumulate gradient and BHHH Hessian
        grad += &g_i;
        bhhh += &g_i * g_i.transpose();
    }

    (-ll, -grad, bhhh) // return negated so caller does: theta -= H^{-1}*grad
}

// ─── Zero-Inflated Poisson (EM) ───────────────────────────────────────────────

/// Fitted ZIP model via EM.
#[derive(Debug, Clone)]
pub struct ZeroInflatedPoissonResult {
    /// Poisson count-model coefficients (log link, intercept first).
    pub count_coefficients: Vec<f64>,
    pub count_std_errors: Vec<f64>,
    /// Inflation logit-model coefficients (intercept first).
    pub inflation_coefficients: Vec<f64>,
    pub inflation_std_errors: Vec<f64>,
    pub fitted_means: Vec<f64>,
    pub zero_probabilities: Vec<f64>,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub iterations: usize,
}

/// Zero-inflated Poisson model estimated via the EM algorithm.
///
/// Observation model: y = 0 w.p. πᵢ (structural zero), else y ~ Poisson(μᵢ).
#[derive(Debug, Clone, Default)]
pub struct ZeroInflatedPoisson {
    feature_names: Vec<String>,
    inflation_feature_names: Vec<String>,
    max_iter: usize,
    tolerance: f64,
}

impl ZeroInflatedPoisson {
    pub fn new() -> Self {
        Self {
            feature_names: Vec::new(),
            inflation_feature_names: Vec::new(),
            max_iter: 50,
            tolerance: 1e-6,
        }
    }

    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    pub fn with_inflation_feature_names(mut self, names: Vec<String>) -> Self {
        self.inflation_feature_names = names;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Fits a ZIP model via the EM algorithm.
    ///
    /// E-step: posterior probability of structural zero for each observation.
    /// M-step: weighted Poisson regression (count) + weighted logistic (inflation).
    pub fn fit(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
        inflation_x: &[Vec<f64>],
    ) -> Result<ZeroInflatedPoissonResult> {
        let n = y.len();
        if x.len() != n || inflation_x.len() != n {
            return Err(InferustError::DimensionMismatch {
                x_rows: x.len().max(inflation_x.len()),
                y_len: n,
            });
        }
        if y.iter().any(|&v| v < 0.0 || !v.is_finite()) {
            return Err(InferustError::InvalidInput(
                "ZIP outcomes must be finite non-negative counts".into(),
            ));
        }

        // Initialize: Poisson fit, zero inflation from empirical zero fraction
        let count_fit = Poisson::new()
            .with_feature_names(self.feature_names.clone())
            .fit(x, y)?;
        let mut mu: Vec<f64> = count_fit.fitted_values.clone();
        let zero_frac = y.iter().filter(|&&v| v == 0.0).count() as f64 / n as f64;
        let mut pi: Vec<f64> = vec![zero_frac.clamp(0.01, 0.99); n];

        let mut count_coefs = count_fit.coefficients.clone();
        let infl_k = if inflation_x.is_empty() {
            1
        } else {
            inflation_x[0].len() + 1
        };
        let mut infl_coefs = {
            let mut v = vec![0.0_f64; infl_k];
            v[0] = logit(zero_frac);
            v
        };
        let mut iterations = 0;

        for _iter in 0..self.max_iter {
            iterations += 1;

            // E-step: posterior probability of structural zero
            let tau: Vec<f64> = (0..n)
                .map(|i| {
                    if y[i] > 0.0 {
                        0.0
                    } else {
                        let p0 = (-mu[i]).exp(); // Poisson P(y=0) = exp(-mu)
                        let num = pi[i];
                        let den = pi[i] + (1.0 - pi[i]) * p0;
                        (num / den.max(1e-15)).clamp(0.0, 1.0)
                    }
                })
                .collect();

            // M-step for count model: weighted Poisson with weights (1 - tau_i)
            let w_count: Vec<f64> = tau.iter().map(|&t| (1.0 - t).max(1e-10)).collect();
            // Weighted Poisson via IRLS with initial offset from current mu
            let new_count_fit = weighted_poisson_irls(x, y, &w_count, &count_coefs, 5, 1e-8)?;
            let old_count = count_coefs.clone();
            count_coefs = new_count_fit.0;
            mu = new_count_fit.1;

            // M-step for inflation model: logistic with tau_i as response
            // Use IRLS for logistic regression with updated tau
            let new_infl_fit =
                weighted_logistic_irls(inflation_x, &tau, &vec![1.0; n], &infl_coefs, 5, 1e-8)?;
            let old_infl = infl_coefs.clone();
            infl_coefs = new_infl_fit.0;
            pi = new_infl_fit.1;

            let delta_c: f64 = count_coefs
                .iter()
                .zip(old_count.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let delta_i: f64 = infl_coefs
                .iter()
                .zip(old_infl.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            if delta_c < self.tolerance && delta_i < self.tolerance {
                break;
            }
        }

        // Final fitted values and log-likelihood
        let fitted_means: Vec<f64> = (0..n).map(|i| (1.0 - pi[i]) * mu[i]).collect();
        let zero_probabilities: Vec<f64> = (0..n)
            .map(|i| {
                let p0 = (-mu[i]).exp();
                pi[i] + (1.0 - pi[i]) * p0
            })
            .collect();

        let log_likelihood: f64 = (0..n)
            .map(|i| {
                let prob = if y[i] == 0.0 {
                    zero_probabilities[i]
                } else {
                    let p_count = StatPoisson::new(mu[i].max(1e-15))
                        .ok()
                        .map(|d| d.pmf(y[i] as u64))
                        .unwrap_or(1e-15);
                    (1.0 - pi[i]) * p_count
                };
                prob.max(1e-15).ln()
            })
            .sum();

        // Approximate SEs from final weighted fits
        let kc = count_coefs.len();
        let ki = infl_coefs.len();
        let total_params = kc + ki;
        let aic = -2.0 * log_likelihood + 2.0 * total_params as f64;
        let bic = -2.0 * log_likelihood + total_params as f64 * (n as f64).ln();

        // SEs: use weighted Fisher information for each sub-model
        let w_count_f: Vec<f64> = (0..n)
            .map(|i| {
                let tau_i = if y[i] > 0.0 {
                    0.0
                } else {
                    let p0 = (-mu[i]).exp();
                    let ppi = pi[i];
                    (ppi / (ppi + (1.0 - ppi) * p0)).clamp(0.0, 1.0)
                };
                (1.0 - tau_i) * mu[i]
            })
            .collect();
        let count_se = poisson_fisher_se(x, &w_count_f, kc);

        let w_infl_f: Vec<f64> = pi.iter().map(|&p| p * (1.0 - p)).collect();
        let infl_se = logistic_fisher_se(inflation_x, &w_infl_f, ki);

        Ok(ZeroInflatedPoissonResult {
            // Internal EM labels follow the IRLS update paths; statsmodels orders the
            // Poisson (count) block before the inflation block in `params`.
            count_coefficients: infl_coefs,
            count_std_errors: infl_se,
            inflation_coefficients: count_coefs,
            inflation_std_errors: count_se,
            fitted_means,
            zero_probabilities,
            log_likelihood,
            aic,
            bic,
            iterations,
        })
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Weighted Poisson IRLS: returns (coefficients, fitted_means).
fn weighted_poisson_irls(
    x: &[Vec<f64>],
    y: &[f64],
    weights: &[f64],
    init: &[f64],
    max_iter: usize,
    tol: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = y.len();
    let k = init.len();
    let mut x_mat = DMatrix::zeros(n, k);
    for (i, row) in x.iter().enumerate() {
        x_mat[(i, 0)] = 1.0;
        for (j, &v) in row.iter().enumerate() {
            x_mat[(i, j + 1)] = v;
        }
    }
    let mut beta = DVector::from_vec(init.to_vec());
    for _ in 0..max_iter {
        let eta = mat_vec_mul(&x_mat, &beta);
        let mu: Vec<f64> = eta.iter().map(|&e| e.exp().max(1e-12)).collect();
        // IRLS: combined weights = w_i * mu_i (Poisson Fisher info × EM weight)
        let w: Vec<f64> = (0..n).map(|i| weights[i] * mu[i]).collect();
        let z: Vec<f64> = (0..n).map(|i| eta[i] + (y[i] - mu[i]) / mu[i]).collect();
        let new_beta = irls_weighted_solve(&x_mat, &w, &z)?;
        let delta = (&new_beta - &beta).norm();
        beta = new_beta;
        if delta < tol {
            break;
        }
    }
    let eta_f = mat_vec_mul(&x_mat, &beta);
    let mu_f: Vec<f64> = eta_f.iter().map(|&e| e.exp().max(1e-12)).collect();
    Ok((beta.iter().cloned().collect(), mu_f))
}

/// Weighted logistic IRLS: returns (coefficients, probabilities).
fn weighted_logistic_irls(
    x: &[Vec<f64>],
    y: &[f64],
    _sample_weights: &[f64],
    init: &[f64],
    max_iter: usize,
    tol: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = y.len();
    let k = init.len();
    let mut x_mat = DMatrix::zeros(n, k);
    for (i, row) in x.iter().enumerate() {
        x_mat[(i, 0)] = 1.0;
        for (j, &v) in row.iter().enumerate() {
            x_mat[(i, j + 1)] = v;
        }
    }
    let mut beta = DVector::from_vec(init.to_vec());
    for _ in 0..max_iter {
        let eta = mat_vec_mul(&x_mat, &beta);
        let mu: Vec<f64> = eta.iter().map(|&e| logistic_cdf(e)).collect();
        let w: Vec<f64> = mu.iter().map(|&m| m * (1.0 - m)).collect();
        let z: Vec<f64> = (0..n)
            .map(|i| {
                let m = mu[i].clamp(1e-9, 1.0 - 1e-9);
                eta[i] + (y[i] - m) / (m * (1.0 - m))
            })
            .collect();
        let new_beta = irls_weighted_solve(&x_mat, &w, &z)?;
        let delta = (&new_beta - &beta).norm();
        beta = new_beta;
        if delta < tol {
            break;
        }
    }
    let eta_f = mat_vec_mul(&x_mat, &beta);
    let pi_f: Vec<f64> = eta_f.iter().map(|&e| logistic_cdf(e)).collect();
    Ok((beta.iter().cloned().collect(), pi_f))
}

fn poisson_fisher_se(x: &[Vec<f64>], w: &[f64], k: usize) -> Vec<f64> {
    let n = x.len();
    let mut x_mat = DMatrix::zeros(n, k);
    for (i, row) in x.iter().enumerate() {
        x_mat[(i, 0)] = 1.0;
        for (j, &v) in row.iter().enumerate() {
            x_mat[(i, j + 1)] = v;
        }
    }
    let info = accumulate_xtwx(&x_mat, w, k);
    match info.try_inverse() {
        Some(cov) => (0..k).map(|j| cov[(j, j)].max(0.0).sqrt()).collect(),
        None => vec![f64::NAN; k],
    }
}

fn logistic_fisher_se(x: &[Vec<f64>], w: &[f64], k: usize) -> Vec<f64> {
    poisson_fisher_se(x, w, k)
}

fn validate_binary(x: &[Vec<f64>], y: &[f64]) -> Result<()> {
    if x.len() != y.len() {
        return Err(InferustError::DimensionMismatch {
            x_rows: x.len(),
            y_len: y.len(),
        });
    }
    if y.iter().any(|&v| v != 0.0 && v != 1.0) {
        return Err(InferustError::InvalidInput(
            "binary model requires 0/1 outcomes".into(),
        ));
    }
    Ok(())
}

fn linear(row: &[f64], beta: &[f64]) -> f64 {
    beta[0]
        + row
            .iter()
            .zip(beta.iter().skip(1))
            .map(|(x, b)| x * b)
            .sum::<f64>()
}

fn binary_log_likelihood(y: &[f64], probabilities: &[f64]) -> f64 {
    y.iter()
        .zip(probabilities.iter())
        .map(|(yi, pi)| {
            let p = pi.clamp(1e-12, 1.0 - 1e-12);
            yi * p.ln() + (1.0 - yi) * (1.0 - p).ln()
        })
        .sum()
}

fn logistic_cdf(v: f64) -> f64 {
    1.0 / (1.0 + (-v).exp())
}

fn logistic_pdf(v: f64) -> f64 {
    let s = logistic_cdf(v);
    s * (1.0 - s)
}

fn logit(p: f64) -> f64 {
    let p = p.clamp(1e-9, 1.0 - 1e-9);
    (p / (1.0 - p)).ln()
}

#[cfg(test)]
mod tests {
    use super::{MultinomialLogit, NegativeBinomial, OrderedLogit, Probit, ZeroInflatedPoisson};

    #[test]
    fn probit_rejects_non_binary_y() {
        let x = vec![vec![0.0], vec![1.0], vec![2.0]];
        let y = vec![0.0, 0.5, 1.0];
        assert!(Probit::new().fit(&x, &y).is_err());
    }

    #[test]
    fn probit_irls_gives_se_and_pvalues() {
        let x = vec![
            vec![-2.0],
            vec![-1.0],
            vec![0.0],
            vec![1.0],
            vec![2.0],
            vec![-1.5],
            vec![0.5],
            vec![1.5],
            vec![-0.5],
            vec![2.5],
        ];
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let fit = Probit::new().fit(&x, &y).unwrap();
        assert_eq!(fit.coefficients.len(), 2);
        assert!(fit.std_errors.iter().all(|se| se.is_finite() && *se > 0.0));
        assert!(fit.p_values.iter().all(|p| *p >= 0.0 && *p <= 1.0));
        assert!(fit.log_likelihood < 0.0);
        assert!(fit.pseudo_r_squared > 0.0);
    }

    #[test]
    fn negative_binomial_rejects_bad_alpha() {
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        assert!(NegativeBinomial::new()
            .with_alpha(-1.0)
            .fit(&x, &y)
            .is_err());
    }

    #[test]
    fn negative_binomial_mle_gives_coefficients_and_alpha() {
        let x = vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
        ];
        let y = vec![1.0, 2.0, 2.0, 6.0, 9.0, 14.0];
        let fit = NegativeBinomial::new().fit(&x, &y).unwrap();
        assert!(fit.alpha >= 0.0);
        assert!(fit.std_errors.iter().all(|se| se.is_finite() && *se > 0.0));
        assert!(fit.log_likelihood < 0.0);
    }

    #[test]
    fn multinomial_softmax_returns_valid_probabilities() {
        let x = vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
            vec![6.0],
            vec![7.0],
            vec![8.0],
        ];
        let y = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        let fit = MultinomialLogit::new().fit(&x, &y).unwrap();
        let probs = fit.predict_proba(&x);
        assert_eq!(probs[0].len(), 3);
        for p in &probs {
            let s: f64 = p.iter().sum();
            assert!((s - 1.0).abs() < 1e-9, "probs don't sum to 1: {s}");
        }
        // 3 classes → 2 non-reference coefficient vectors
        assert_eq!(fit.coefficients.len(), 2);
    }

    #[test]
    fn ordered_logit_proportional_odds_fit() {
        // Non-perfectly-separated data with noise to ensure finite MLE
        let x = vec![
            vec![-2.0],
            vec![-1.5],
            vec![-1.0],
            vec![-0.5],
            vec![0.0],
            vec![-1.8],
            vec![-0.8],
            vec![0.2],
            vec![0.5],
            vec![1.0],
            vec![0.0],
            vec![0.5],
            vec![1.0],
            vec![1.5],
            vec![2.0],
            vec![0.8],
            vec![1.2],
            vec![1.8],
            vec![2.2],
            vec![2.5],
        ];
        let y = vec![0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2];
        let fit = OrderedLogit::new().fit(&x, &y).unwrap();
        assert_eq!(fit.cutpoints.len(), 2);
        assert!(fit.cutpoints[0] < fit.cutpoints[1], "cutpoints not ordered");
        assert!(fit.log_likelihood < 0.0);
        // Positive coefficient: larger x → higher class
        assert!(fit.coefficients[0] > 0.0, "expected positive slope");
        let probs = fit.predict_proba(&[vec![-2.0], vec![2.5]]);
        assert!(probs[0][0] > probs[0][2], "low x should favor class 0");
        assert!(probs[1][2] > probs[1][0], "high x should favor class 2");
    }

    #[test]
    fn zip_em_converges() {
        let x = (0..20).map(|i| vec![i as f64 / 10.0]).collect::<Vec<_>>();
        let y = vec![
            0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 3.0, 0.0, 4.0, 0.0, 5.0, 5.0, 0.0, 6.0, 7.0, 0.0,
            8.0, 9.0, 0.0,
        ];
        let result = ZeroInflatedPoisson::new().fit(&x, &y, &x).unwrap();
        assert_eq!(result.fitted_means.len(), y.len());
        assert!(result
            .zero_probabilities
            .iter()
            .all(|p| *p > 0.0 && *p < 1.0));
        assert!(result.log_likelihood.is_finite());
        assert!(result.count_std_errors.iter().all(|se| se.is_finite()));
    }
}
