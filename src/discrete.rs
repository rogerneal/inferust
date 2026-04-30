use std::collections::BTreeMap;

use statrs::distribution::{Continuous, ContinuousCDF, Normal};

use crate::error::{InferustError, Result};
use crate::glm::{Logistic, LogisticResult, Poisson, PoissonResult};

/// Fitted binary probit model.
#[derive(Debug, Clone)]
pub struct ProbitResult {
    pub coefficients: Vec<f64>,
    pub fitted_probabilities: Vec<f64>,
    pub log_likelihood: f64,
    pub feature_names: Vec<String>,
    pub iterations: usize,
}

/// Binary probit estimator using the standard normal CDF link.
#[derive(Debug, Clone)]
pub struct Probit {
    feature_names: Vec<String>,
    max_iter: usize,
    tolerance: f64,
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

    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Fits a probit model using a stable gradient ascent starter.
    pub fn fit(&self, x: &[Vec<f64>], y: &[f64]) -> Result<ProbitResult> {
        validate_binary(x, y)?;
        let n = y.len();
        let p = x[0].len() + 1;
        let mut beta = vec![0.0; p];
        let normal = Normal::new(0.0, 1.0)
            .map_err(|_| InferustError::InvalidInput("invalid normal distribution".into()))?;
        let mut previous = f64::NEG_INFINITY;
        let mut iterations = 0;
        for iter in 0..self.max_iter {
            iterations = iter + 1;
            let probabilities = probit_predict_internal(x, &beta, &normal);
            let ll = binary_log_likelihood(y, &probabilities);
            if (ll - previous).abs() < self.tolerance {
                break;
            }
            previous = ll;
            let mut gradient = vec![0.0; p];
            for (i, row) in x.iter().enumerate() {
                let eta = linear(row, &beta);
                let pdf = normal.pdf(eta).max(1e-12);
                let prob = probabilities[i].clamp(1e-9, 1.0 - 1e-9);
                let score = (y[i] - prob) * pdf / (prob * (1.0 - prob));
                gradient[0] += score;
                for (j, value) in row.iter().enumerate() {
                    gradient[j + 1] += score * value;
                }
            }
            let step = 0.05 / n as f64;
            for (coef, grad) in beta.iter_mut().zip(gradient.iter()) {
                *coef += step * grad;
            }
        }
        let fitted_probabilities = probit_predict_internal(x, &beta, &normal);
        let log_likelihood = binary_log_likelihood(y, &fitted_probabilities);
        let mut feature_names = vec!["const".to_string()];
        if self.feature_names.is_empty() {
            feature_names.extend((1..p).map(|i| format!("x{i}")));
        } else {
            feature_names.extend(self.feature_names.clone());
        }
        Ok(ProbitResult {
            coefficients: beta,
            fitted_probabilities,
            log_likelihood,
            feature_names,
            iterations,
        })
    }
}

impl ProbitResult {
    pub fn predict_proba(&self, x: &[Vec<f64>]) -> Result<Vec<f64>> {
        let normal = Normal::new(0.0, 1.0)
            .map_err(|_| InferustError::InvalidInput("invalid normal distribution".into()))?;
        Ok(probit_predict_internal(x, &self.coefficients, &normal))
    }
}

/// Negative binomial starter result using a Poisson mean model and NB2 overdispersion.
#[derive(Debug, Clone)]
pub struct NegativeBinomialResult {
    pub poisson: PoissonResult,
    pub alpha: f64,
    pub fitted_values: Vec<f64>,
}

/// Negative binomial count-model starter with method-of-moments overdispersion.
#[derive(Debug, Clone)]
pub struct NegativeBinomial {
    alpha: Option<f64>,
    feature_names: Vec<String>,
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
            feature_names: Vec::new(),
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

    /// Fits an NB2 starter by fitting Poisson means and estimating overdispersion by moments.
    pub fn fit(&self, x: &[Vec<f64>], y: &[f64]) -> Result<NegativeBinomialResult> {
        if let Some(alpha) = self.alpha {
            if alpha < 0.0 || !alpha.is_finite() {
                return Err(InferustError::InvalidInput(
                    "negative binomial alpha must be finite and non-negative".into(),
                ));
            }
        }
        let poisson = Poisson::new()
            .with_feature_names(self.feature_names.clone())
            .fit(x, y)?;
        let alpha = self.alpha.unwrap_or_else(|| {
            let numerator = y
                .iter()
                .zip(poisson.fitted_values.iter())
                .map(|(yi, mui)| (yi - mui).powi(2) - mui)
                .sum::<f64>();
            let denominator = poisson
                .fitted_values
                .iter()
                .map(|mu| mu.powi(2))
                .sum::<f64>()
                .max(1e-12);
            (numerator / denominator).max(0.0)
        });
        Ok(NegativeBinomialResult {
            fitted_values: poisson.fitted_values.clone(),
            poisson,
            alpha,
        })
    }
}

/// One-vs-rest multinomial logit starter result.
#[derive(Debug, Clone)]
pub struct MultinomialLogitResult {
    pub classes: Vec<usize>,
    pub models: BTreeMap<usize, LogisticResult>,
}

/// Multiclass logistic starter implemented as one-vs-rest binary logits.
#[derive(Debug, Clone, Default)]
pub struct MultinomialLogit {
    feature_names: Vec<String>,
}

impl MultinomialLogit {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Fits a one-vs-rest multinomial logit starter.
    pub fn fit(&self, x: &[Vec<f64>], y: &[usize]) -> Result<MultinomialLogitResult> {
        let mut classes = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        if classes.len() < 2 {
            return Err(InferustError::InvalidInput(
                "multinomial logit needs at least two classes".into(),
            ));
        }
        let mut models = BTreeMap::new();
        for class in &classes {
            let binary = y
                .iter()
                .map(|yi| usize::from(yi == class) as f64)
                .collect::<Vec<_>>();
            let model = Logistic::new()
                .with_feature_names(self.feature_names.clone())
                .fit(x, &binary)?;
            models.insert(*class, model);
        }
        Ok(MultinomialLogitResult { classes, models })
    }
}

impl MultinomialLogitResult {
    pub fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let raw = self
            .classes
            .iter()
            .map(|class| self.models[class].predict_proba(x))
            .collect::<Vec<_>>();
        (0..x.len())
            .map(|i| {
                let denom = raw.iter().map(|values| values[i]).sum::<f64>().max(1e-12);
                raw.iter()
                    .map(|values| values[i] / denom)
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

fn validate_binary(x: &[Vec<f64>], y: &[f64]) -> Result<()> {
    if x.len() != y.len() {
        return Err(InferustError::DimensionMismatch {
            x_rows: x.len(),
            y_len: y.len(),
        });
    }
    if y.iter().any(|value| *value != 0.0 && *value != 1.0) {
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

fn probit_predict_internal(x: &[Vec<f64>], beta: &[f64], normal: &Normal) -> Vec<f64> {
    x.iter().map(|row| normal.cdf(linear(row, beta))).collect()
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
    use super::{MultinomialLogit, NegativeBinomial, Probit};

    #[test]
    fn probit_rejects_non_binary_y() {
        let x = vec![vec![0.0], vec![1.0], vec![2.0]];
        let y = vec![0.0, 0.5, 1.0];
        assert!(Probit::new().fit(&x, &y).is_err());
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
    fn fits_probit_probabilities() {
        let x = vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
        ];
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let fit = Probit::new().fit(&x, &y).unwrap();
        assert_eq!(fit.fitted_probabilities.len(), y.len());
    }

    #[test]
    fn estimates_negative_binomial_alpha() {
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
    }

    #[test]
    fn fits_multinomial_one_vs_rest() {
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
    }
}
