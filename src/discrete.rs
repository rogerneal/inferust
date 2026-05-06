use std::collections::BTreeMap;

use statrs::distribution::{Continuous, ContinuousCDF, Discrete, Normal, Poisson as StatPoisson};

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
                .max_iter(500)
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

/// Ordered logit starter result using cumulative binary logits.
#[derive(Debug, Clone)]
pub struct OrderedLogitResult {
    pub classes: Vec<usize>,
    pub cutpoints: Vec<f64>,
    pub coefficients: Vec<f64>,
    pub feature_names: Vec<String>,
}

/// Proportional-odds ordered logit starter.
#[derive(Debug, Clone, Default)]
pub struct OrderedLogit {
    feature_names: Vec<String>,
}

impl OrderedLogit {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Fits cumulative logits for `P(y <= class)` and averages slope terms
    /// into a proportional-odds coefficient vector.
    pub fn fit(&self, x: &[Vec<f64>], y: &[usize]) -> Result<OrderedLogitResult> {
        validate_multiclass_x(x, y)?;
        let mut classes = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        if classes.len() < 3 {
            return Err(InferustError::InvalidInput(
                "ordered logit needs at least three ordered classes".into(),
            ));
        }

        let mut cutpoints = Vec::with_capacity(classes.len() - 1);
        let mut slope_sums = vec![0.0; x[0].len()];
        for class in classes.iter().take(classes.len() - 1) {
            let binary = y
                .iter()
                .map(|yi| f64::from(yi <= class))
                .collect::<Vec<_>>();
            let model = Logistic::new()
                .with_feature_names(self.feature_names.clone())
                .fit(x, &binary)?;
            cutpoints.push(model.coefficients[0]);
            for (sum, slope) in slope_sums.iter_mut().zip(model.coefficients.iter().skip(1)) {
                *sum += -*slope;
            }
        }
        let denom = (classes.len() - 1) as f64;
        let coefficients = slope_sums
            .into_iter()
            .map(|value| value / denom)
            .collect::<Vec<_>>();
        let feature_names = if self.feature_names.is_empty() {
            (0..x[0].len()).map(|i| format!("x{}", i + 1)).collect()
        } else {
            self.feature_names.clone()
        };

        Ok(OrderedLogitResult {
            classes,
            cutpoints,
            coefficients,
            feature_names,
        })
    }
}

impl OrderedLogitResult {
    pub fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        x.iter()
            .map(|row| {
                let eta = row
                    .iter()
                    .zip(self.coefficients.iter())
                    .map(|(x, b)| x * b)
                    .sum::<f64>();
                let cumulative = self
                    .cutpoints
                    .iter()
                    .map(|cut| logistic_cdf(cut - eta))
                    .collect::<Vec<_>>();
                let mut probabilities = Vec::with_capacity(self.classes.len());
                probabilities.push(cumulative[0]);
                for pair in cumulative.windows(2) {
                    probabilities.push((pair[1] - pair[0]).max(0.0));
                }
                probabilities.push((1.0 - cumulative[cumulative.len() - 1]).max(0.0));
                let total = probabilities.iter().sum::<f64>().max(1e-12);
                probabilities.iter_mut().for_each(|p| *p /= total);
                probabilities
            })
            .collect()
    }
}

/// Zero-inflated Poisson starter result.
#[derive(Debug, Clone)]
pub struct ZeroInflatedPoissonResult {
    pub count_model: PoissonResult,
    pub inflation_model: LogisticResult,
    pub fitted_means: Vec<f64>,
    pub zero_probabilities: Vec<f64>,
    pub log_likelihood: f64,
}

/// Zero-inflated Poisson count model with logit inflation.
#[derive(Debug, Clone, Default)]
pub struct ZeroInflatedPoisson {
    feature_names: Vec<String>,
    inflation_feature_names: Vec<String>,
}

impl ZeroInflatedPoisson {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    pub fn with_inflation_feature_names(mut self, names: Vec<String>) -> Self {
        self.inflation_feature_names = names;
        self
    }

    /// Fits a ZIP starter by combining a Poisson mean model with a logit model
    /// for structural zeros.
    pub fn fit(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
        inflation_x: &[Vec<f64>],
    ) -> Result<ZeroInflatedPoissonResult> {
        if x.len() != y.len() || inflation_x.len() != y.len() {
            return Err(InferustError::DimensionMismatch {
                x_rows: x.len().max(inflation_x.len()),
                y_len: y.len(),
            });
        }
        if y.iter().any(|value| *value < 0.0 || !value.is_finite()) {
            return Err(InferustError::InvalidInput(
                "ZIP outcomes must be finite non-negative counts".into(),
            ));
        }
        let count_model = Poisson::new()
            .with_feature_names(self.feature_names.clone())
            .fit(x, y)?;
        let zero_indicator = y
            .iter()
            .map(|value| f64::from(*value == 0.0))
            .collect::<Vec<_>>();
        let inflation_model = Logistic::new()
            .with_feature_names(self.inflation_feature_names.clone())
            .fit(inflation_x, &zero_indicator)?;
        let pi = inflation_model.predict_proba(inflation_x);
        let mut fitted_means = Vec::with_capacity(y.len());
        let mut zero_probabilities = Vec::with_capacity(y.len());
        let mut log_likelihood = 0.0;
        for ((&yi, &mu), &inflation) in y
            .iter()
            .zip(count_model.fitted_values.iter())
            .zip(pi.iter())
        {
            let mu = mu.max(1e-12);
            let inflation = inflation.clamp(1e-9, 1.0 - 1e-9);
            let poisson = StatPoisson::new(mu)
                .map_err(|_| InferustError::InvalidInput("invalid Poisson mean".into()))?;
            fitted_means.push((1.0 - inflation) * mu);
            let p0 = inflation + (1.0 - inflation) * poisson.pmf(0);
            zero_probabilities.push(p0);
            let prob = if yi == 0.0 {
                p0
            } else {
                (1.0 - inflation) * poisson.pmf(yi as u64)
            };
            log_likelihood += prob.max(1e-12).ln();
        }
        Ok(ZeroInflatedPoissonResult {
            count_model,
            inflation_model,
            fitted_means,
            zero_probabilities,
            log_likelihood,
        })
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

fn validate_multiclass_x(x: &[Vec<f64>], y: &[usize]) -> Result<()> {
    if x.len() != y.len() {
        return Err(InferustError::DimensionMismatch {
            x_rows: x.len(),
            y_len: y.len(),
        });
    }
    if x.is_empty() {
        return Err(InferustError::InsufficientData { needed: 1, got: 0 });
    }
    let p = x[0].len();
    for row in x {
        if row.len() != p {
            return Err(InferustError::InvalidInput(
                "all rows in X must have the same length".into(),
            ));
        }
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

fn logistic_cdf(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
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

    #[test]
    fn ordered_logit_returns_ordered_probabilities() {
        let x = (0..30).map(|i| vec![i as f64 / 10.0]).collect::<Vec<_>>();
        let y = (0..30)
            .map(|i| {
                let score = i as f64 / 10.0 + ((i % 5) as f64 - 2.0) * 0.35;
                if score < 1.1 {
                    0
                } else if score < 2.1 {
                    1
                } else {
                    2
                }
            })
            .collect::<Vec<_>>();
        let result = OrderedLogit::new().fit(&x, &y).unwrap();
        assert_eq!(result.cutpoints.len(), 2);
        let probabilities = result.predict_proba(&[vec![0.0], vec![2.9]]);
        assert!(probabilities[0][0] > probabilities[0][2]);
        assert!(probabilities[1][2] > probabilities[1][0]);
    }

    #[test]
    fn zero_inflated_poisson_fits_zero_probabilities() {
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
    }
}
