//! Treatment-effect estimators and balance diagnostics.

use crate::error::{InferustError, Result};
use crate::glm::{Logistic, LogisticResult};

/// Inverse-probability weighted treatment-effect result.
#[derive(Debug, Clone)]
pub struct IpwResult {
    pub ate: f64,
    pub att: f64,
    pub treated_mean: f64,
    pub control_mean: f64,
    pub propensity_scores: Vec<f64>,
    pub propensity_model: LogisticResult,
}

/// Covariate balance diagnostic for one column.
#[derive(Debug, Clone)]
pub struct BalanceDiagnostic {
    pub column: usize,
    pub treated_mean: f64,
    pub control_mean: f64,
    pub standardized_mean_difference: f64,
}

/// Propensity-score treatment-effect workflow.
#[derive(Debug, Clone, Default)]
pub struct PropensityScore {
    feature_names: Vec<String>,
}

impl PropensityScore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Fit propensity scores `P(treatment = 1 | x)`.
    pub fn fit(&self, x: &[Vec<f64>], treatment: &[f64]) -> Result<LogisticResult> {
        validate_treatment_x(x, treatment)?;
        Logistic::new()
            .with_feature_names(self.feature_names.clone())
            .fit(x, treatment)
    }

    /// Estimate IPW ATE and ATT.
    pub fn ipw(&self, x: &[Vec<f64>], treatment: &[f64], outcome: &[f64]) -> Result<IpwResult> {
        if outcome.len() != treatment.len() {
            return Err(InferustError::DimensionMismatch {
                x_rows: outcome.len(),
                y_len: treatment.len(),
            });
        }
        if outcome.iter().any(|value| !value.is_finite()) {
            return Err(InferustError::InvalidInput(
                "outcomes must be finite".into(),
            ));
        }
        let propensity_model = self.fit(x, treatment)?;
        let propensity_scores = propensity_model
            .predict_proba(x)
            .into_iter()
            .map(|p| p.clamp(1e-3, 1.0 - 1e-3))
            .collect::<Vec<_>>();

        let mut treated_weighted = 0.0;
        let mut treated_weights = 0.0;
        let mut control_weighted = 0.0;
        let mut control_weights = 0.0;
        let mut att_control_weighted = 0.0;
        let mut att_control_weights = 0.0;
        let mut treated_raw = 0.0;
        let mut treated_raw_n = 0.0;

        for ((&t, &y), &p) in treatment
            .iter()
            .zip(outcome.iter())
            .zip(propensity_scores.iter())
        {
            if t == 1.0 {
                let weight = 1.0 / p;
                treated_weighted += weight * y;
                treated_weights += weight;
                treated_raw += y;
                treated_raw_n += 1.0;
            } else {
                let weight = 1.0 / (1.0 - p);
                control_weighted += weight * y;
                control_weights += weight;
                let att_weight = p / (1.0 - p);
                att_control_weighted += att_weight * y;
                att_control_weights += att_weight;
            }
        }

        if treated_weights == 0.0 || control_weights == 0.0 || att_control_weights == 0.0 {
            return Err(InferustError::InvalidInput(
                "IPW needs both treated and control observations".into(),
            ));
        }
        let treated_mean = treated_weighted / treated_weights;
        let control_mean = control_weighted / control_weights;
        let att_treated_mean = treated_raw / treated_raw_n;
        let att_control_mean = att_control_weighted / att_control_weights;
        Ok(IpwResult {
            ate: treated_mean - control_mean,
            att: att_treated_mean - att_control_mean,
            treated_mean,
            control_mean,
            propensity_scores,
            propensity_model,
        })
    }
}

/// Standardized mean-difference diagnostics for raw covariate balance.
pub fn balance_diagnostics(x: &[Vec<f64>], treatment: &[f64]) -> Result<Vec<BalanceDiagnostic>> {
    validate_treatment_x(x, treatment)?;
    let p = x[0].len();
    let mut out = Vec::with_capacity(p);
    for column in 0..p {
        let treated = x
            .iter()
            .zip(treatment.iter())
            .filter_map(|(row, t)| (*t == 1.0).then_some(row[column]))
            .collect::<Vec<_>>();
        let control = x
            .iter()
            .zip(treatment.iter())
            .filter_map(|(row, t)| (*t == 0.0).then_some(row[column]))
            .collect::<Vec<_>>();
        if treated.is_empty() || control.is_empty() {
            return Err(InferustError::InvalidInput(
                "balance diagnostics need both treatment groups".into(),
            ));
        }
        let treated_mean = mean(&treated);
        let control_mean = mean(&control);
        let pooled_sd = (((variance(&treated) + variance(&control)) / 2.0).max(1e-12)).sqrt();
        out.push(BalanceDiagnostic {
            column,
            treated_mean,
            control_mean,
            standardized_mean_difference: (treated_mean - control_mean) / pooled_sd,
        });
    }
    Ok(out)
}

fn validate_treatment_x(x: &[Vec<f64>], treatment: &[f64]) -> Result<()> {
    if x.len() != treatment.len() {
        return Err(InferustError::DimensionMismatch {
            x_rows: x.len(),
            y_len: treatment.len(),
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
        if row.iter().any(|value| !value.is_finite()) {
            return Err(InferustError::InvalidInput(
                "covariates must be finite".into(),
            ));
        }
    }
    if treatment.iter().any(|value| *value != 0.0 && *value != 1.0) {
        return Err(InferustError::InvalidInput(
            "treatment must be coded as 0/1".into(),
        ));
    }
    Ok(())
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let center = mean(values);
    values
        .iter()
        .map(|value| (value - center).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64
}

#[cfg(test)]
mod tests {
    use super::{balance_diagnostics, PropensityScore};

    #[test]
    fn ipw_estimates_positive_treatment_effect() {
        let x = (0..40).map(|i| vec![i as f64 / 10.0]).collect::<Vec<_>>();
        let treatment = (0..40)
            .map(|i| if i % 3 == 0 || i > 25 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let outcome = x
            .iter()
            .zip(treatment.iter())
            .map(|(row, t)| 1.0 + 0.2 * row[0] + 2.0 * t)
            .collect::<Vec<_>>();
        let result = PropensityScore::new()
            .ipw(&x, &treatment, &outcome)
            .unwrap();
        assert!(result.ate > 1.0);
        assert_eq!(result.propensity_scores.len(), x.len());
    }

    #[test]
    fn balance_reports_standardized_mean_difference() {
        let x = vec![vec![0.0], vec![1.0], vec![3.0], vec![4.0]];
        let treatment = vec![0.0, 0.0, 1.0, 1.0];
        let balance = balance_diagnostics(&x, &treatment).unwrap();
        assert!(balance[0].standardized_mean_difference > 1.0);
    }
}
