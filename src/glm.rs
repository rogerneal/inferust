use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};

use crate::error::{InferustError, Result};

/// GLM residual vectors using common statsmodels-compatible definitions.
#[derive(Debug, Clone)]
pub struct GlmResiduals {
    pub response: Vec<f64>,
    pub pearson: Vec<f64>,
    pub deviance: Vec<f64>,
}

/// Binary classification metrics computed from fitted probabilities.
#[derive(Debug, Clone)]
pub struct BinaryClassificationMetrics {
    pub threshold: f64,
    pub true_positives: usize,
    pub false_positives: usize,
    pub true_negatives: usize,
    pub false_negatives: usize,
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub log_loss: f64,
    pub brier_score: f64,
}

/// Likelihood-ratio test summary for nested likelihood models.
#[derive(Debug, Clone)]
pub struct LikelihoodRatioTest {
    pub statistic: f64,
    pub df: usize,
    pub p_value: f64,
}

/// Predicted mean with a Wald confidence interval on the response scale.
#[derive(Debug, Clone)]
pub struct PredictionInterval {
    pub mean: f64,
    pub lower: f64,
    pub upper: f64,
}

/// Average marginal effect summary for a logistic regression predictor.
#[derive(Debug, Clone)]
pub struct LogisticMarginalEffect {
    pub name: String,
    pub effect: f64,
    pub std_error: f64,
    pub z_statistic: f64,
    pub p_value: f64,
    pub confidence_interval: (f64, f64),
}

/// Binary logistic regression result.
#[derive(Debug, Clone)]
pub struct LogisticResult {
    pub coefficients: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub z_statistics: Vec<f64>,
    pub p_values: Vec<f64>,
    pub covariance_matrix: Vec<Vec<f64>>,
    pub fitted_probabilities: Vec<f64>,
    pub log_likelihood: f64,
    pub null_log_likelihood: f64,
    pub pseudo_r_squared: f64,
    pub aic: f64,
    pub bic: f64,
    pub n: usize,
    pub k: usize,
    pub feature_names: Vec<String>,
    observed: Vec<f64>,
    design_matrix: Vec<Vec<f64>>,
    iterations: usize,
}

impl LogisticResult {
    /// Predict probabilities for raw X rows, without the intercept column.
    pub fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter()
            .map(|row| {
                let mut eta = if self.feature_names[0] == "const" {
                    self.coefficients[0]
                } else {
                    0.0
                };
                let offset = if self.feature_names[0] == "const" {
                    1
                } else {
                    0
                };
                for (j, &value) in row.iter().enumerate() {
                    eta += self.coefficients[offset + j] * value;
                }
                sigmoid(eta)
            })
            .collect()
    }

    /// Wald confidence intervals for coefficients at the requested alpha level.
    pub fn confidence_intervals(&self, alpha: f64) -> Result<Vec<(f64, f64)>> {
        if !(0.0..1.0).contains(&alpha) {
            return Err(InferustError::InvalidInput(
                "alpha must be between 0 and 1".into(),
            ));
        }
        let normal = Normal::new(0.0, 1.0)
            .map_err(|_| InferustError::InvalidInput("invalid normal distribution".into()))?;
        let z = normal.inverse_cdf(1.0 - alpha / 2.0);
        Ok(self
            .coefficients
            .iter()
            .zip(self.std_errors.iter())
            .map(|(coef, se)| (coef - z * se, coef + z * se))
            .collect())
    }

    /// Exponentiated coefficients, matching statsmodels' common odds-ratio view.
    pub fn odds_ratios(&self) -> Vec<f64> {
        self.coefficients.iter().map(|coef| coef.exp()).collect()
    }

    /// Average marginal effects for each non-intercept predictor.
    pub fn average_marginal_effects(&self) -> Vec<(String, f64)> {
        let scale = self.average_probability_slope();
        let offset = self.intercept_offset();

        self.feature_names
            .iter()
            .skip(offset)
            .zip(self.coefficients.iter().skip(offset))
            .map(|(name, coef)| (name.clone(), coef * scale))
            .collect()
    }

    /// Delta-method average marginal effects with standard errors and Wald intervals.
    pub fn average_marginal_effects_summary(
        &self,
        alpha: f64,
    ) -> Result<Vec<LogisticMarginalEffect>> {
        if !(0.0..1.0).contains(&alpha) {
            return Err(InferustError::InvalidInput(
                "alpha must be between 0 and 1".into(),
            ));
        }
        let normal = Normal::new(0.0, 1.0)
            .map_err(|_| InferustError::InvalidInput("invalid normal distribution".into()))?;
        let critical = normal.inverse_cdf(1.0 - alpha / 2.0);
        let scale = self.average_probability_slope();
        let offset = self.intercept_offset();

        let mut effects = Vec::with_capacity(self.coefficients.len().saturating_sub(offset));
        for j in offset..self.coefficients.len() {
            let effect = self.coefficients[j] * scale;
            let gradient = self.average_marginal_effect_gradient(j, scale);
            let variance = quadratic_form(&gradient, &self.covariance_matrix).max(0.0);
            let std_error = variance.sqrt();
            let z_statistic = effect / std_error;
            let p_value = 2.0 * (1.0 - normal.cdf(z_statistic.abs()));
            effects.push(LogisticMarginalEffect {
                name: self.feature_names[j].clone(),
                effect,
                std_error,
                z_statistic,
                p_value,
                confidence_interval: (effect - critical * std_error, effect + critical * std_error),
            });
        }

        Ok(effects)
    }

    fn intercept_offset(&self) -> usize {
        usize::from(
            self.feature_names
                .first()
                .is_some_and(|name| name == "const"),
        )
    }

    fn average_probability_slope(&self) -> f64 {
        self.fitted_probabilities
            .iter()
            .map(|p| p * (1.0 - p))
            .sum::<f64>()
            / self.n as f64
    }

    fn average_marginal_effect_gradient(&self, coefficient_index: usize, scale: f64) -> Vec<f64> {
        let mut gradient = vec![0.0; self.coefficients.len()];
        for parameter_index in 0..self.coefficients.len() {
            let slope_derivative = self
                .fitted_probabilities
                .iter()
                .zip(self.design_matrix.iter())
                .map(|(p, row)| (1.0 - 2.0 * p) * p * (1.0 - p) * row[parameter_index])
                .sum::<f64>()
                / self.n as f64;
            gradient[parameter_index] = self.coefficients[coefficient_index] * slope_derivative;
        }
        gradient[coefficient_index] += scale;
        gradient
    }

    /// Linear predictor values for raw X rows, without the intercept column.
    pub fn predict_linear(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter()
            .map(|row| linear_predict(row, &self.coefficients, &self.feature_names))
            .collect()
    }

    /// Response, Pearson, and deviance residuals for the training data.
    pub fn residuals(&self) -> GlmResiduals {
        let response = self
            .observed
            .iter()
            .zip(self.fitted_probabilities.iter())
            .map(|(yi, pi)| yi - pi)
            .collect::<Vec<_>>();
        let pearson = self
            .observed
            .iter()
            .zip(self.fitted_probabilities.iter())
            .map(|(yi, pi)| (yi - pi) / (pi * (1.0 - pi)).max(1e-12).sqrt())
            .collect::<Vec<_>>();
        let deviance = self
            .observed
            .iter()
            .zip(self.fitted_probabilities.iter())
            .map(|(yi, pi)| binomial_deviance_residual(*yi, *pi))
            .collect::<Vec<_>>();

        GlmResiduals {
            response,
            pearson,
            deviance,
        }
    }

    /// Model deviance against a saturated Bernoulli model.
    pub fn deviance(&self) -> f64 {
        -2.0 * self.log_likelihood
    }

    /// Null model deviance against a saturated Bernoulli model.
    pub fn null_deviance(&self) -> f64 {
        -2.0 * self.null_log_likelihood
    }

    /// Likelihood-ratio test against the intercept-only model.
    pub fn likelihood_ratio_test(&self) -> Result<LikelihoodRatioTest> {
        likelihood_ratio_test(self.log_likelihood, self.null_log_likelihood, self.k)
    }

    /// Classification metrics at a probability threshold.
    pub fn classification_metrics(&self, threshold: f64) -> Result<BinaryClassificationMetrics> {
        if !(0.0..1.0).contains(&threshold) {
            return Err(InferustError::InvalidInput(
                "threshold must be between 0 and 1".into(),
            ));
        }

        let mut true_positives = 0;
        let mut false_positives = 0;
        let mut true_negatives = 0;
        let mut false_negatives = 0;
        for (yi, pi) in self.observed.iter().zip(self.fitted_probabilities.iter()) {
            match (*pi >= threshold, *yi == 1.0) {
                (true, true) => true_positives += 1,
                (true, false) => false_positives += 1,
                (false, false) => true_negatives += 1,
                (false, true) => false_negatives += 1,
            }
        }

        let accuracy = (true_positives + true_negatives) as f64 / self.n as f64;
        let precision = safe_ratio(
            true_positives as f64,
            (true_positives + false_positives) as f64,
        );
        let recall = safe_ratio(
            true_positives as f64,
            (true_positives + false_negatives) as f64,
        );
        let f1_score = safe_ratio(2.0 * precision * recall, precision + recall);
        let log_loss =
            -binary_log_likelihood(&self.observed, &self.fitted_probabilities) / self.n as f64;
        let brier_score = self
            .observed
            .iter()
            .zip(self.fitted_probabilities.iter())
            .map(|(yi, pi)| (yi - pi).powi(2))
            .sum::<f64>()
            / self.n as f64;

        Ok(BinaryClassificationMetrics {
            threshold,
            true_positives,
            false_positives,
            true_negatives,
            false_negatives,
            accuracy,
            precision,
            recall,
            f1_score,
            log_loss,
            brier_score,
        })
    }

    /// Number of Newton/IRLS iterations used to fit the model.
    pub fn iterations(&self) -> usize {
        self.iterations
    }
}

/// Binary logistic regression builder.
pub struct Logistic {
    feature_names: Vec<String>,
    add_intercept: bool,
    max_iter: usize,
    tolerance: f64,
}

impl Default for Logistic {
    fn default() -> Self {
        Self::new()
    }
}

impl Logistic {
    /// Create a new logistic regression builder. An intercept term is added by default.
    pub fn new() -> Self {
        Self {
            feature_names: Vec::new(),
            add_intercept: true,
            max_iter: 100,
            tolerance: 1e-8,
        }
    }

    /// Set human-readable names for predictor columns.
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Fit without an intercept.
    pub fn no_intercept(mut self) -> Self {
        self.add_intercept = false;
        self
    }

    /// Set maximum Newton iterations.
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance on the largest coefficient update.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Fit a binary logistic regression model.
    pub fn fit(&self, x: &[Vec<f64>], y: &[f64]) -> Result<LogisticResult> {
        let n = y.len();
        if n < 2 {
            return Err(InferustError::InsufficientData { needed: 2, got: n });
        }
        if x.len() != n {
            return Err(InferustError::DimensionMismatch {
                x_rows: x.len(),
                y_len: n,
            });
        }
        if let Some(value) = y.iter().find(|value| **value != 0.0 && **value != 1.0) {
            return Err(InferustError::InvalidInput(format!(
                "logistic regression requires binary 0/1 y values, got {value}"
            )));
        }

        let p = x[0].len();
        let ncols = if self.add_intercept { p + 1 } else { p };
        if n <= ncols {
            return Err(InferustError::InsufficientData {
                needed: ncols + 1,
                got: n,
            });
        }

        let mut design = Vec::with_capacity(n * ncols);
        for row in x {
            if row.len() != p {
                return Err(InferustError::InvalidInput(
                    "all rows in X must have the same length".into(),
                ));
            }
            if self.add_intercept {
                design.push(1.0);
            }
            design.extend_from_slice(row);
        }

        let x_mat = DMatrix::from_row_slice(n, ncols, &design);
        let y_vec = DVector::from_column_slice(y);
        let mut beta = DVector::zeros(ncols);
        let mut hessian = DMatrix::zeros(ncols, ncols);
        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..self.max_iter {
            iterations = iter + 1;
            let eta = &x_mat * &beta;
            let mu = eta.map(sigmoid);
            let gradient = x_mat.transpose() * (&y_vec - &mu);
            hessian.fill(0.0);
            for i in 0..n {
                let w = (mu[i] * (1.0 - mu[i])).max(1e-12);
                for j in 0..ncols {
                    for k in 0..ncols {
                        hessian[(j, k)] += w * x_mat[(i, j)] * x_mat[(i, k)];
                    }
                }
            }

            let delta = hessian
                .clone()
                .lu()
                .solve(&gradient)
                .ok_or(InferustError::SingularMatrix)?;
            let max_delta = delta.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            beta += delta;
            if max_delta < self.tolerance {
                converged = true;
                break;
            }
        }

        if !converged {
            return Err(InferustError::InvalidInput(format!(
                "logistic regression failed to converge in {} iterations",
                self.max_iter
            )));
        }

        let eta = &x_mat * &beta;
        let probabilities = eta.map(sigmoid);
        hessian.fill(0.0);
        for i in 0..n {
            let w = (probabilities[i] * (1.0 - probabilities[i])).max(1e-12);
            for j in 0..ncols {
                for k in 0..ncols {
                    hessian[(j, k)] += w * x_mat[(i, j)] * x_mat[(i, k)];
                }
            }
        }

        let cov = hessian.try_inverse().ok_or(InferustError::SingularMatrix)?;
        let covariance_matrix: Vec<Vec<f64>> = (0..ncols)
            .map(|i| (0..ncols).map(|j| cov[(i, j)]).collect())
            .collect();
        let design_matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..ncols).map(|j| x_mat[(i, j)]).collect())
            .collect();
        let coefficients: Vec<f64> = beta.iter().cloned().collect();
        let std_errors: Vec<f64> = (0..ncols).map(|i| cov[(i, i)].sqrt()).collect();
        let z_statistics: Vec<f64> = coefficients
            .iter()
            .zip(std_errors.iter())
            .map(|(coef, se)| coef / se)
            .collect();
        let normal = Normal::new(0.0, 1.0)
            .map_err(|_| InferustError::InvalidInput("invalid normal distribution".into()))?;
        let p_values = z_statistics
            .iter()
            .map(|z| 2.0 * (1.0 - normal.cdf(z.abs())))
            .collect::<Vec<_>>();

        let log_likelihood = binary_log_likelihood(y, probabilities.as_slice());
        let y_mean = y.iter().sum::<f64>() / n as f64;
        let null_probs = vec![y_mean; n];
        let null_log_likelihood = binary_log_likelihood(y, &null_probs);
        let pseudo_r_squared = 1.0 - log_likelihood / null_log_likelihood;
        let n_params = ncols as f64;
        let aic = -2.0 * log_likelihood + 2.0 * n_params;
        let bic = -2.0 * log_likelihood + n_params * (n as f64).ln();

        let mut feature_names = Vec::with_capacity(ncols);
        if self.add_intercept {
            feature_names.push("const".to_string());
        }
        if self.feature_names.is_empty() {
            for i in 0..(ncols - usize::from(self.add_intercept)) {
                feature_names.push(format!("x{}", i + 1));
            }
        } else {
            feature_names.extend(self.feature_names.iter().cloned());
        }

        Ok(LogisticResult {
            coefficients,
            std_errors,
            z_statistics,
            p_values,
            covariance_matrix,
            fitted_probabilities: probabilities.iter().cloned().collect(),
            log_likelihood,
            null_log_likelihood,
            pseudo_r_squared,
            aic,
            bic,
            n,
            k: ncols - usize::from(self.add_intercept),
            feature_names,
            observed: y.to_vec(),
            design_matrix,
            iterations,
        })
    }
}

/// Poisson regression result for count outcomes with a log link.
#[derive(Debug, Clone)]
pub struct PoissonResult {
    pub coefficients: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub z_statistics: Vec<f64>,
    pub p_values: Vec<f64>,
    pub covariance_matrix: Vec<Vec<f64>>,
    pub fitted_values: Vec<f64>,
    pub log_likelihood: f64,
    pub null_log_likelihood: f64,
    pub pseudo_r_squared: f64,
    pub deviance: f64,
    pub pearson_chi_squared: f64,
    pub aic: f64,
    pub bic: f64,
    pub n: usize,
    pub k: usize,
    pub feature_names: Vec<String>,
    observed: Vec<f64>,
    design_matrix: Vec<Vec<f64>>,
    iterations: usize,
}

impl PoissonResult {
    /// Predict expected counts for raw X rows, without the intercept column.
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter()
            .map(|row| {
                let mut eta = if self.feature_names[0] == "const" {
                    self.coefficients[0]
                } else {
                    0.0
                };
                let offset = if self.feature_names[0] == "const" {
                    1
                } else {
                    0
                };
                for (j, &value) in row.iter().enumerate() {
                    eta += self.coefficients[offset + j] * value;
                }
                eta.exp()
            })
            .collect()
    }

    /// Wald confidence intervals for coefficients at the requested alpha level.
    pub fn confidence_intervals(&self, alpha: f64) -> Result<Vec<(f64, f64)>> {
        if !(0.0..1.0).contains(&alpha) {
            return Err(InferustError::InvalidInput(
                "alpha must be between 0 and 1".into(),
            ));
        }
        let normal = Normal::new(0.0, 1.0)
            .map_err(|_| InferustError::InvalidInput("invalid normal distribution".into()))?;
        let z = normal.inverse_cdf(1.0 - alpha / 2.0);
        Ok(self
            .coefficients
            .iter()
            .zip(self.std_errors.iter())
            .map(|(coef, se)| (coef - z * se, coef + z * se))
            .collect())
    }

    /// Incidence-rate ratios, i.e. exponentiated coefficients.
    pub fn incidence_rate_ratios(&self) -> Vec<f64> {
        self.coefficients.iter().map(|coef| coef.exp()).collect()
    }

    /// Linear predictor values for raw X rows, without the intercept column.
    pub fn predict_linear(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter()
            .map(|row| linear_predict(row, &self.coefficients, &self.feature_names))
            .collect()
    }

    /// Response, Pearson, and deviance residuals for the training data.
    pub fn residuals(&self) -> GlmResiduals {
        let response = self
            .observed
            .iter()
            .zip(self.fitted_values.iter())
            .map(|(yi, mui)| yi - mui)
            .collect::<Vec<_>>();
        let pearson = self
            .observed
            .iter()
            .zip(self.fitted_values.iter())
            .map(|(yi, mui)| (yi - mui) / mui.max(1e-12).sqrt())
            .collect::<Vec<_>>();
        let deviance = self
            .observed
            .iter()
            .zip(self.fitted_values.iter())
            .map(|(yi, mui)| poisson_deviance_residual(*yi, *mui))
            .collect::<Vec<_>>();

        GlmResiduals {
            response,
            pearson,
            deviance,
        }
    }

    /// Null model deviance against a saturated Poisson model.
    pub fn null_deviance(&self) -> f64 {
        let mean = self.observed.iter().sum::<f64>() / self.n as f64;
        poisson_deviance(&self.observed, &vec![mean; self.n])
    }

    /// Likelihood-ratio test against the intercept-only model.
    pub fn likelihood_ratio_test(&self) -> Result<LikelihoodRatioTest> {
        likelihood_ratio_test(self.log_likelihood, self.null_log_likelihood, self.k)
    }

    /// Fitted mean confidence intervals for the training rows.
    pub fn fitted_mean_intervals(&self, alpha: f64) -> Result<Vec<PredictionInterval>> {
        prediction_intervals(
            &self.design_matrix,
            &self.coefficients,
            &self.covariance_matrix,
            alpha,
        )
    }

    /// Mean confidence intervals for raw X rows, without the intercept column.
    pub fn predict_mean_intervals(
        &self,
        x: &[Vec<f64>],
        alpha: f64,
    ) -> Result<Vec<PredictionInterval>> {
        let design = build_prediction_design(x, &self.feature_names);
        prediction_intervals(&design, &self.coefficients, &self.covariance_matrix, alpha)
    }

    /// Number of Newton/IRLS iterations used to fit the model.
    pub fn iterations(&self) -> usize {
        self.iterations
    }
}

/// Poisson regression builder for non-negative count outcomes.
pub struct Poisson {
    feature_names: Vec<String>,
    add_intercept: bool,
    max_iter: usize,
    tolerance: f64,
    offset: Option<Vec<f64>>,
    exposure: Option<Vec<f64>>,
}

impl Default for Poisson {
    fn default() -> Self {
        Self::new()
    }
}

impl Poisson {
    /// Create a new Poisson regression builder. An intercept term is added by default.
    pub fn new() -> Self {
        Self {
            feature_names: Vec::new(),
            add_intercept: true,
            max_iter: 100,
            tolerance: 1e-8,
            offset: None,
            exposure: None,
        }
    }

    /// Set human-readable names for predictor columns.
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Fit without an intercept.
    pub fn no_intercept(mut self) -> Self {
        self.add_intercept = false;
        self
    }

    /// Set maximum Newton iterations.
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance on the largest coefficient update.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Add an offset term to the linear predictor.
    pub fn with_offset(mut self, offset: Vec<f64>) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Add exposure values; the log exposure is used as an offset.
    pub fn with_exposure(mut self, exposure: Vec<f64>) -> Self {
        self.exposure = Some(exposure);
        self
    }

    /// Fit a Poisson regression model using Newton/IRLS updates.
    pub fn fit(&self, x: &[Vec<f64>], y: &[f64]) -> Result<PoissonResult> {
        let n = y.len();
        if n < 2 {
            return Err(InferustError::InsufficientData { needed: 2, got: n });
        }
        if x.len() != n {
            return Err(InferustError::DimensionMismatch {
                x_rows: x.len(),
                y_len: n,
            });
        }
        if let Some(value) = y.iter().find(|value| **value < 0.0 || !value.is_finite()) {
            return Err(InferustError::InvalidInput(format!(
                "poisson regression requires finite non-negative y values, got {value}"
            )));
        }
        let offset = combine_offset_exposure(n, self.offset.as_deref(), self.exposure.as_deref())?;

        let p = x[0].len();
        let ncols = if self.add_intercept { p + 1 } else { p };
        if n <= ncols {
            return Err(InferustError::InsufficientData {
                needed: ncols + 1,
                got: n,
            });
        }

        let mut design = Vec::with_capacity(n * ncols);
        for row in x {
            if row.len() != p {
                return Err(InferustError::InvalidInput(
                    "all rows in X must have the same length".into(),
                ));
            }
            if self.add_intercept {
                design.push(1.0);
            }
            design.extend_from_slice(row);
        }

        let x_mat = DMatrix::from_row_slice(n, ncols, &design);
        let y_vec = DVector::from_column_slice(y);
        let mean = (y.iter().sum::<f64>() / n as f64).max(1e-12);
        let mut beta = DVector::zeros(ncols);
        if self.add_intercept {
            beta[0] = mean.ln();
        }
        let mut hessian = DMatrix::zeros(ncols, ncols);
        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..self.max_iter {
            iterations = iter + 1;
            let eta = &x_mat * &beta + DVector::from_column_slice(&offset);
            let mu = eta.map(|value| value.clamp(-700.0, 700.0).exp());
            let gradient = x_mat.transpose() * (&y_vec - &mu);
            hessian.fill(0.0);
            for i in 0..n {
                for j in 0..ncols {
                    for k in 0..ncols {
                        hessian[(j, k)] += mu[i].max(1e-12) * x_mat[(i, j)] * x_mat[(i, k)];
                    }
                }
            }

            let delta = hessian
                .clone()
                .lu()
                .solve(&gradient)
                .ok_or(InferustError::SingularMatrix)?;
            let max_delta = delta.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            beta += delta;
            if max_delta < self.tolerance {
                converged = true;
                break;
            }
        }

        if !converged {
            return Err(InferustError::InvalidInput(format!(
                "poisson regression failed to converge in {} iterations",
                self.max_iter
            )));
        }

        let eta = &x_mat * &beta + DVector::from_column_slice(&offset);
        let fitted = eta.map(|value| value.clamp(-700.0, 700.0).exp());
        hessian.fill(0.0);
        for i in 0..n {
            for j in 0..ncols {
                for k in 0..ncols {
                    hessian[(j, k)] += fitted[i].max(1e-12) * x_mat[(i, j)] * x_mat[(i, k)];
                }
            }
        }

        let cov = hessian.try_inverse().ok_or(InferustError::SingularMatrix)?;
        let covariance_matrix: Vec<Vec<f64>> = (0..ncols)
            .map(|i| (0..ncols).map(|j| cov[(i, j)]).collect())
            .collect();
        let design_matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..ncols).map(|j| x_mat[(i, j)]).collect())
            .collect();
        let coefficients: Vec<f64> = beta.iter().cloned().collect();
        let std_errors: Vec<f64> = (0..ncols).map(|i| cov[(i, i)].sqrt()).collect();
        let z_statistics: Vec<f64> = coefficients
            .iter()
            .zip(std_errors.iter())
            .map(|(coef, se)| coef / se)
            .collect();
        let normal = Normal::new(0.0, 1.0)
            .map_err(|_| InferustError::InvalidInput("invalid normal distribution".into()))?;
        let p_values = z_statistics
            .iter()
            .map(|z| 2.0 * (1.0 - normal.cdf(z.abs())))
            .collect::<Vec<_>>();

        let fitted_values = fitted.iter().cloned().collect::<Vec<_>>();
        let log_likelihood = poisson_log_likelihood(y, &fitted_values);
        let null_fitted = vec![mean; n];
        let null_log_likelihood = poisson_log_likelihood(y, &null_fitted);
        let pseudo_r_squared = 1.0 - log_likelihood / null_log_likelihood;
        let deviance = poisson_deviance(y, &fitted_values);
        let pearson_chi_squared = y
            .iter()
            .zip(fitted_values.iter())
            .map(|(yi, mui)| (yi - mui).powi(2) / mui.max(1e-12))
            .sum();
        let n_params = ncols as f64;
        let aic = -2.0 * log_likelihood + 2.0 * n_params;
        let bic = -2.0 * log_likelihood + n_params * (n as f64).ln();

        let mut feature_names = Vec::with_capacity(ncols);
        if self.add_intercept {
            feature_names.push("const".to_string());
        }
        if self.feature_names.is_empty() {
            for i in 0..(ncols - usize::from(self.add_intercept)) {
                feature_names.push(format!("x{}", i + 1));
            }
        } else {
            feature_names.extend(self.feature_names.iter().cloned());
        }

        Ok(PoissonResult {
            coefficients,
            std_errors,
            z_statistics,
            p_values,
            covariance_matrix,
            fitted_values,
            log_likelihood,
            null_log_likelihood,
            pseudo_r_squared,
            deviance,
            pearson_chi_squared,
            aic,
            bic,
            n,
            k: ncols - usize::from(self.add_intercept),
            feature_names,
            observed: y.to_vec(),
            design_matrix,
            iterations,
        })
    }
}

/// Likelihood-ratio test for nested likelihood models.
pub fn likelihood_ratio_test(
    full_log_likelihood: f64,
    restricted_log_likelihood: f64,
    df: usize,
) -> Result<LikelihoodRatioTest> {
    if df == 0 {
        return Err(InferustError::InvalidInput(
            "likelihood-ratio test degrees of freedom must be positive".into(),
        ));
    }
    let statistic = 2.0 * (full_log_likelihood - restricted_log_likelihood);
    let chi_squared = ChiSquared::new(df as f64)
        .map_err(|_| InferustError::InvalidInput("invalid chi-squared distribution".into()))?;
    Ok(LikelihoodRatioTest {
        statistic,
        df,
        p_value: 1.0 - chi_squared.cdf(statistic.max(0.0)),
    })
}

fn linear_predict(row: &[f64], coefficients: &[f64], feature_names: &[String]) -> f64 {
    let offset = usize::from(feature_names.first().is_some_and(|name| name == "const"));
    let mut eta = if offset == 1 { coefficients[0] } else { 0.0 };
    for (j, &value) in row.iter().enumerate() {
        eta += coefficients[offset + j] * value;
    }
    eta
}

fn build_prediction_design(x: &[Vec<f64>], feature_names: &[String]) -> Vec<Vec<f64>> {
    let has_intercept = feature_names.first().is_some_and(|name| name == "const");
    x.iter()
        .map(|row| {
            let mut design_row = Vec::with_capacity(row.len() + usize::from(has_intercept));
            if has_intercept {
                design_row.push(1.0);
            }
            design_row.extend_from_slice(row);
            design_row
        })
        .collect()
}

fn prediction_intervals(
    design: &[Vec<f64>],
    coefficients: &[f64],
    covariance_matrix: &[Vec<f64>],
    alpha: f64,
) -> Result<Vec<PredictionInterval>> {
    if !(0.0..1.0).contains(&alpha) {
        return Err(InferustError::InvalidInput(
            "alpha must be between 0 and 1".into(),
        ));
    }
    let normal = Normal::new(0.0, 1.0)
        .map_err(|_| InferustError::InvalidInput("invalid normal distribution".into()))?;
    let critical = normal.inverse_cdf(1.0 - alpha / 2.0);
    Ok(design
        .iter()
        .map(|row| {
            let eta = row
                .iter()
                .zip(coefficients.iter())
                .map(|(xij, coef)| xij * coef)
                .sum::<f64>();
            let se_eta = quadratic_form(row, covariance_matrix).max(0.0).sqrt();
            PredictionInterval {
                mean: eta.exp(),
                lower: (eta - critical * se_eta).exp(),
                upper: (eta + critical * se_eta).exp(),
            }
        })
        .collect())
}

fn safe_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn combine_offset_exposure(
    n: usize,
    offset: Option<&[f64]>,
    exposure: Option<&[f64]>,
) -> Result<Vec<f64>> {
    let mut combined = vec![0.0; n];
    if let Some(values) = offset {
        if values.len() != n {
            return Err(InferustError::DimensionMismatch {
                x_rows: values.len(),
                y_len: n,
            });
        }
        combined.copy_from_slice(values);
    }
    if let Some(values) = exposure {
        if values.len() != n {
            return Err(InferustError::DimensionMismatch {
                x_rows: values.len(),
                y_len: n,
            });
        }
        for (target, value) in combined.iter_mut().zip(values.iter()) {
            if *value <= 0.0 || !value.is_finite() {
                return Err(InferustError::InvalidInput(
                    "exposure values must be finite and positive".into(),
                ));
            }
            *target += value.ln();
        }
    }
    Ok(combined)
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

fn binomial_deviance_residual(y: f64, probability: f64) -> f64 {
    let p = probability.clamp(1e-12, 1.0 - 1e-12);
    let contribution = if y == 0.0 {
        -2.0 * (1.0 - p).ln()
    } else {
        -2.0 * p.ln()
    };
    (y - p).signum() * contribution.sqrt()
}

fn poisson_deviance_residual(y: f64, fitted: f64) -> f64 {
    let mu = fitted.max(1e-12);
    let contribution = if y == 0.0 {
        2.0 * mu
    } else {
        2.0 * (y * (y / mu).ln() - (y - mu))
    };
    (y - mu).signum() * contribution.sqrt()
}

fn poisson_log_likelihood(y: &[f64], fitted: &[f64]) -> f64 {
    y.iter()
        .zip(fitted.iter())
        .map(|(yi, mui)| {
            let mu = mui.max(1e-12);
            if *yi == 0.0 {
                -mu
            } else {
                yi * mu.ln() - mu - statrs::function::gamma::ln_gamma(yi + 1.0)
            }
        })
        .sum()
}

fn poisson_deviance(y: &[f64], fitted: &[f64]) -> f64 {
    2.0 * y
        .iter()
        .zip(fitted.iter())
        .map(|(yi, mui)| {
            let mu = mui.max(1e-12);
            if *yi == 0.0 {
                mu
            } else {
                yi * (yi / mu).ln() - (yi - mu)
            }
        })
        .sum::<f64>()
}

fn quadratic_form(vector: &[f64], matrix: &[Vec<f64>]) -> f64 {
    vector
        .iter()
        .enumerate()
        .map(|(i, left)| {
            vector
                .iter()
                .enumerate()
                .map(|(j, right)| left * matrix[i][j] * right)
                .sum::<f64>()
        })
        .sum()
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

#[cfg(test)]
mod tests {
    use super::Logistic;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual} differed from expected {expected} by more than {tolerance}"
        );
    }

    fn fixture() -> (Vec<Vec<f64>>, Vec<f64>) {
        (
            vec![
                vec![0.2, 1.0],
                vec![1.1, 0.9],
                vec![1.8, 1.5],
                vec![2.4, 1.9],
                vec![3.0, 2.5],
                vec![3.7, 2.9],
                vec![4.1, 3.4],
                vec![4.8, 3.8],
                vec![5.2, 4.2],
                vec![5.9, 4.8],
                vec![2.2, 3.6],
                vec![4.6, 1.2],
            ],
            vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
        )
    }

    #[test]
    fn logistic_matches_statsmodels_reference_values() {
        let (x, y) = fixture();
        let result = Logistic::new().fit(&x, &y).unwrap();

        let expected_coefficients = [-1.7689272112231273, 5.075491023368961, -5.333429371506223];
        let expected_std_errors = [2.0497905638092466, 9.672670992764024, 12.013832575525887];
        let expected_z = [
            -0.8629794879803845,
            0.5247248693939718,
            -0.44394071067473323,
        ];
        let expected_p = [0.38814874483671213, 0.5997744663371969, 0.6570854500881999];
        let expected_covariance = [
            [4.20164135548143, 2.328115175935611, -4.294521008650565],
            [2.3281151759356806, 93.56056413425858, -115.97606208303982],
            [-4.294521008650698, -115.97606208303978, 144.33217315276696],
        ];
        let expected_ci = [
            (-5.786442892139302, 2.248588469693047),
            (-13.882595756753815, 24.033577803491738),
            (-28.88010853583104, 18.213249792818594),
        ];
        let expected_odds = [
            0.17051581819526604,
            160.05076135215282,
            0.004827486348599407,
        ];
        let expected_ame = [0.6368929483098553, -0.6692600856510365];
        let expected_ame_se = [1.1662712154590942, 1.4683077393598531];
        let expected_ame_z = [0.5460933442133759, -0.45580368999676985];
        let expected_ame_p = [0.5850017747015581, 0.6485311504206718];
        let expected_ame_ci = [
            (-1.6489566301957226, 2.9227425268154335),
            (-3.5470903730177734, 2.2085702017157005),
        ];
        let expected_linear = [
            -6.087258378055558,
            -0.9859735198728705,
            -0.6331874264183313,
            0.2787354390005561,
            0.12397243011819903,
            1.5434443978739871,
            0.9069261214684587,
            2.3263980892242415,
            2.223222749969338,
            2.576008843423878,
            -9.803192697233815,
            15.17821625046663,
        ];
        let expected_response_residuals = [
            -0.0022664797317858963,
            -0.27170811607160383,
            0.6532118497285212,
            -0.5692361727365304,
            0.46904652640209454,
            0.1760351184308543,
            -0.7123707405174255,
            0.088960152720438,
            0.09768437569027344,
            0.07069850190439753,
            -5.5271786442116135e-05,
            2.559672049873285e-07,
        ];
        let expected_pred = [
            0.0022664797317858963,
            0.27170811607160383,
            0.3467881502714788,
            0.5692361727365304,
            0.5309534735979055,
            0.8239648815691457,
            0.7123707405174255,
            0.911039847279562,
            0.9023156243097266,
            0.9293014980956025,
            5.5271786442116135e-05,
            0.999999744032795,
        ];

        for (actual, expected) in result.coefficients.iter().zip(expected_coefficients) {
            assert_close(*actual, expected, 1e-8);
        }
        for (actual, expected) in result.std_errors.iter().zip(expected_std_errors) {
            assert_close(*actual, expected, 1e-7);
        }
        for (actual, expected) in result.z_statistics.iter().zip(expected_z) {
            assert_close(*actual, expected, 1e-8);
        }
        for (actual, expected) in result.p_values.iter().zip(expected_p) {
            assert_close(*actual, expected, 1e-8);
        }
        for (actual_row, expected_row) in result.covariance_matrix.iter().zip(expected_covariance) {
            for (actual, expected) in actual_row.iter().zip(expected_row) {
                assert_close(*actual, expected, 1e-7);
            }
        }
        let intervals = result.confidence_intervals(0.05).unwrap();
        for ((actual_low, actual_high), (expected_low, expected_high)) in
            intervals.iter().zip(expected_ci)
        {
            assert_close(*actual_low, expected_low, 1e-7);
            assert_close(*actual_high, expected_high, 1e-7);
        }
        for (actual, expected) in result.odds_ratios().iter().zip(expected_odds) {
            assert_close(*actual, expected, 1e-7);
        }
        let marginal_effects = result.average_marginal_effects();
        for ((_, actual), expected) in marginal_effects.iter().zip(expected_ame) {
            assert_close(*actual, expected, 1e-8);
        }
        let marginal_effect_summaries = result.average_marginal_effects_summary(0.05).unwrap();
        for (i, summary) in marginal_effect_summaries.iter().enumerate() {
            assert_close(summary.effect, expected_ame[i], 1e-8);
            assert_close(summary.std_error, expected_ame_se[i], 1e-7);
            assert_close(summary.z_statistic, expected_ame_z[i], 1e-8);
            assert_close(summary.p_value, expected_ame_p[i], 1e-8);
            assert_close(summary.confidence_interval.0, expected_ame_ci[i].0, 1e-7);
            assert_close(summary.confidence_interval.1, expected_ame_ci[i].1, 1e-7);
        }
        for (actual, expected) in result.predict_linear(&x).iter().zip(expected_linear) {
            assert_close(*actual, expected, 1e-8);
        }
        let residuals = result.residuals();
        for (actual, expected) in residuals.response.iter().zip(expected_response_residuals) {
            assert_close(*actual, expected, 1e-8);
        }
        for (actual, expected) in result.predict_proba(&x).iter().zip(expected_pred) {
            assert_close(*actual, expected, 1e-8);
        }
        for (actual, expected) in result.fitted_probabilities.iter().zip(expected_pred) {
            assert_close(*actual, expected, 1e-8);
        }

        assert_close(result.log_likelihood, -4.562687231980299, 1e-8);
        assert_close(result.null_log_likelihood, -8.150319193022398, 1e-8);
        assert_close(result.pseudo_r_squared, 0.4401830009447385, 1e-8);
        let lr = result.likelihood_ratio_test().unwrap();
        assert_close(lr.statistic, 7.175263922084197, 1e-8);
        assert_close(lr.p_value, 0.027663761786296846, 1e-8);
        let metrics = result.classification_metrics(0.5).unwrap();
        assert_eq!(metrics.true_positives, 6);
        assert_eq!(metrics.false_positives, 2);
        assert_eq!(metrics.true_negatives, 3);
        assert_eq!(metrics.false_negatives, 1);
        assert_close(metrics.accuracy, 0.75, 1e-12);
        assert_close(metrics.precision, 0.75, 1e-12);
        assert_close(metrics.recall, 0.8571428571428571, 1e-12);
        assert_close(metrics.f1_score, 0.8, 1e-12);
        assert_close(metrics.log_loss, 0.3802239359983583, 1e-8);
        assert_close(metrics.brier_score, 0.13378879035798572, 1e-8);
        assert_close(result.deviance(), 9.125374463960599, 1e-8);
        assert_close(result.null_deviance(), 16.300638386044795, 1e-8);
        assert_close(result.aic, 15.125374463960599, 1e-8);
        assert_close(result.bic, 16.580094413324602, 1e-8);
    }

    #[test]
    fn poisson_matches_statsmodels_reference_values() {
        let x = vec![
            vec![0.2, 1.0],
            vec![0.8, 1.4],
            vec![1.2, 1.1],
            vec![1.9, 1.7],
            vec![2.4, 2.2],
            vec![2.9, 2.0],
            vec![3.4, 2.8],
            vec![3.9, 3.1],
            vec![4.5, 3.5],
            vec![5.0, 3.8],
            vec![5.5, 4.0],
            vec![6.0, 4.4],
        ];
        let y = vec![
            1.0, 2.0, 1.0, 3.0, 4.0, 3.0, 6.0, 7.0, 8.0, 11.0, 12.0, 15.0,
        ];
        let result = super::Poisson::new().fit(&x, &y).unwrap();

        let expected_coefficients = [
            -0.2951503394477173,
            -0.03616781469762084,
            0.7449321063132242,
        ];
        let expected_std_errors = [0.6980807376733295, 0.5907644630566597, 0.9402699551201388];
        let expected_z = [
            -0.4228025835971919,
            -0.06122205542033766,
            0.7922534398305259,
        ];
        let expected_p = [0.6724393025092453, 0.951182364991655, 0.42821291425600383];
        let expected_covariance = [
            [0.4873167163105398, 0.32659752549853693, -0.5651175404455193],
            [
                0.32659752549853693,
                0.34900265081062354,
                -0.5508529831108175,
            ],
            [-0.5651175404455193, -0.5508529831108175, 0.8841075885016277],
        ];
        let expected_linear = [
            0.4425482039259827,
            0.7188203576326999,
            0.48087359985968436,
            0.902515393359284,
            1.256897539167086,
            1.0898272105556306,
            1.6676889882573995,
            1.873084712802556,
            2.1493568665092733,
            2.35475259105443,
            2.4856551049682647,
            2.7655440401447446,
        ];
        let expected_response_residuals = [
            -0.5566688785565874,
            -0.05201114305546595,
            -0.6174868214186802,
            0.5342022317773867,
            0.4854990456962174,
            0.0262398062036433,
            0.7000945103509153,
            0.4916581684379562,
            -0.5793389622864833,
            0.4644780431553919,
            -0.00898482411281698,
            -0.8876811761915047,
        ];
        let expected_pearson_residuals = [
            -0.4461684129371736,
            -0.03630835232263843,
            -0.48552021604909407,
            0.34019425128753666,
            0.25897422840882695,
            0.015216250570567455,
            0.30410433403511294,
            0.19272057444436014,
            -0.1977906800668425,
            0.14309918507623465,
            -0.002592724859328783,
            -0.22270335129661484,
        ];
        let expected_deviance_residuals = [
            -0.4777461138106821,
            -0.036463376419548,
            -0.5227106686475718,
            0.32890258177487897,
            0.25333078026564637,
            0.015193954968615907,
            0.2977528353282894,
            0.19036746219532374,
            -0.2000818015261937,
            0.14206657613918533,
            -0.002593048262055297,
            -0.22482700261973126,
        ];
        let expected_mean_intervals = [
            (1.5566688785565874, 0.7014041924113672, 3.454809685604827),
            (2.052011143055466, 0.9609052431530584, 4.382065517102281),
            (1.6174868214186802, 0.7649370758304063, 3.420233768408888),
            (2.4657977682226133, 1.5050447932288815, 4.039852276241836),
            (3.5145009543037826, 2.365579136396782, 5.221434687074627),
            (2.9737601937963567, 1.230323142905119, 7.187745545716136),
            (5.299905489649085, 3.98531735487219, 7.0481208139855465),
            (6.508341831562044, 5.078173204356604, 8.341289611807781),
            (8.579338962286483, 6.6635219182387155, 11.04596907355292),
            (10.535521956844608, 8.125372043959194, 13.660571147099128),
            (12.008984824112817, 8.838022127500595, 16.317646010075826),
            (15.887681176191505, 11.389976627471619, 22.16145137185786),
        ];
        let expected_pred = [
            1.5566688785565874,
            2.052011143055466,
            1.6174868214186802,
            2.4657977682226133,
            3.5145009543037826,
            2.9737601937963567,
            5.299905489649085,
            6.508341831562044,
            8.579338962286483,
            10.535521956844608,
            12.008984824112817,
            15.887681176191505,
        ];

        for (actual, expected) in result.coefficients.iter().zip(expected_coefficients) {
            assert_close(*actual, expected, 1e-8);
        }
        for (actual, expected) in result.std_errors.iter().zip(expected_std_errors) {
            assert_close(*actual, expected, 1e-8);
        }
        for (actual, expected) in result.z_statistics.iter().zip(expected_z) {
            assert_close(*actual, expected, 1e-8);
        }
        for (actual, expected) in result.p_values.iter().zip(expected_p) {
            assert_close(*actual, expected, 1e-8);
        }
        for (actual_row, expected_row) in result.covariance_matrix.iter().zip(expected_covariance) {
            for (actual, expected) in actual_row.iter().zip(expected_row) {
                assert_close(*actual, expected, 1e-8);
            }
        }
        for (actual, expected) in result.predict_linear(&x).iter().zip(expected_linear) {
            assert_close(*actual, expected, 1e-8);
        }
        let residuals = result.residuals();
        for (actual, expected) in residuals.response.iter().zip(expected_response_residuals) {
            assert_close(*actual, expected, 1e-8);
        }
        for (actual, expected) in residuals.pearson.iter().zip(expected_pearson_residuals) {
            assert_close(*actual, expected, 1e-8);
        }
        for (actual, expected) in residuals.deviance.iter().zip(expected_deviance_residuals) {
            assert_close(*actual, expected, 1e-8);
        }
        for (actual, expected) in result.predict(&x).iter().zip(expected_pred) {
            assert_close(*actual, expected, 1e-8);
        }
        for (actual, expected) in result.fitted_values.iter().zip(expected_pred) {
            assert_close(*actual, expected, 1e-8);
        }

        assert_close(result.log_likelihood, -20.660966448471754, 1e-8);
        assert_close(result.null_log_likelihood, -39.7471753812529, 1e-8);
        assert_close(result.pseudo_r_squared, 0.48019032169474163, 1e-8);
        let mean_intervals = result.fitted_mean_intervals(0.05).unwrap();
        for (actual, expected) in mean_intervals.iter().zip(expected_mean_intervals) {
            assert_close(actual.mean, expected.0, 1e-8);
            assert_close(actual.lower, expected.1, 1e-8);
            assert_close(actual.upper, expected.2, 1e-8);
        }
        let lr = result.likelihood_ratio_test().unwrap();
        assert_close(lr.statistic, 38.172417865562295, 1e-8);
        assert_close(lr.p_value, 5.140019658986294e-09, 1e-14);
        assert_close(result.deviance, 0.9110476849677485, 1e-8);
        assert_close(result.null_deviance(), 39.08346555053005, 1e-8);
        assert_close(result.pearson_chi_squared, 0.8579684437989112, 1e-8);
        assert_close(result.aic, 47.32193289694351, 1e-8);
        assert_close(result.bic, 48.77665284630751, 1e-8);
    }
}
