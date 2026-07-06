//! State-space models and Kalman filtering.
//!
//! This module starts with a scalar linear-Gaussian Kalman filter and a
//! local-level model. It is intentionally small, but it establishes the core
//! machinery behind statsmodels' larger `statespace` family.

use crate::error::{InferustError, Result};

/// Parameters for a scalar linear-Gaussian state-space model.
///
/// Observation equation: `y[t] = design * alpha[t] + observation_intercept + eps[t]`
///
/// State equation: `alpha[t+1] = transition * alpha[t] + state_intercept + eta[t]`
#[derive(Debug, Clone, Copy)]
pub struct ScalarStateSpace {
    pub transition: f64,
    pub design: f64,
    pub state_intercept: f64,
    pub observation_intercept: f64,
    pub state_variance: f64,
    pub observation_variance: f64,
    pub initial_state: f64,
    pub initial_variance: f64,
}

impl ScalarStateSpace {
    /// Local-level/random-walk model: `y[t] = level[t] + eps[t]`,
    /// `level[t+1] = level[t] + eta[t]`.
    pub fn local_level(
        observation_variance: f64,
        level_variance: f64,
        initial_level: f64,
        initial_variance: f64,
    ) -> Self {
        Self {
            transition: 1.0,
            design: 1.0,
            state_intercept: 0.0,
            observation_intercept: 0.0,
            state_variance: level_variance,
            observation_variance,
            initial_state: initial_level,
            initial_variance,
        }
    }

    /// Run the Kalman filter over observations.
    ///
    /// Missing observations can be represented as `f64::NAN`; prediction still
    /// advances the state, while the update step is skipped.
    pub fn filter(&self, y: &[f64]) -> Result<KalmanFilterResult> {
        validate_model(self)?;
        if y.is_empty() {
            return Err(InferustError::InsufficientData { needed: 1, got: 0 });
        }

        let mut predicted_state = Vec::with_capacity(y.len());
        let mut predicted_variance = Vec::with_capacity(y.len());
        let mut filtered_state = Vec::with_capacity(y.len());
        let mut filtered_variance = Vec::with_capacity(y.len());
        let mut forecasts = Vec::with_capacity(y.len());
        let mut forecast_errors = Vec::with_capacity(y.len());
        let mut forecast_variances = Vec::with_capacity(y.len());
        let mut kalman_gain = Vec::with_capacity(y.len());
        let mut log_likelihood = 0.0;

        let mut state = self.initial_state;
        let mut variance = self.initial_variance;

        for &obs in y {
            let pred_state = self.transition * state + self.state_intercept;
            let pred_var = self.transition.powi(2) * variance + self.state_variance;
            let forecast = self.design * pred_state + self.observation_intercept;
            let forecast_var = self.design.powi(2) * pred_var + self.observation_variance;

            predicted_state.push(pred_state);
            predicted_variance.push(pred_var);
            forecasts.push(forecast);
            forecast_variances.push(forecast_var);

            if obs.is_nan() {
                forecast_errors.push(f64::NAN);
                kalman_gain.push(0.0);
                state = pred_state;
                variance = pred_var;
            } else if !obs.is_finite() {
                return Err(InferustError::InvalidInput(
                    "observations must be finite or NaN for missing values".into(),
                ));
            } else {
                let error = obs - forecast;
                let gain = pred_var * self.design / forecast_var;
                state = pred_state + gain * error;
                variance = (1.0 - gain * self.design) * pred_var;
                variance = variance.max(0.0);
                forecast_errors.push(error);
                kalman_gain.push(gain);
                log_likelihood += -0.5
                    * ((2.0 * std::f64::consts::PI * forecast_var).ln()
                        + error.powi(2) / forecast_var);
            }

            filtered_state.push(state);
            filtered_variance.push(variance);
        }

        Ok(KalmanFilterResult {
            predicted_state,
            predicted_variance,
            filtered_state,
            filtered_variance,
            forecasts,
            forecast_errors,
            forecast_variances,
            kalman_gain,
            log_likelihood,
        })
    }
}

/// Output from scalar Kalman filtering.
#[derive(Debug, Clone)]
pub struct KalmanFilterResult {
    pub predicted_state: Vec<f64>,
    pub predicted_variance: Vec<f64>,
    pub filtered_state: Vec<f64>,
    pub filtered_variance: Vec<f64>,
    pub forecasts: Vec<f64>,
    pub forecast_errors: Vec<f64>,
    pub forecast_variances: Vec<f64>,
    pub kalman_gain: Vec<f64>,
    pub log_likelihood: f64,
}

impl KalmanFilterResult {
    /// Forecast future observations from the last filtered state.
    pub fn forecast(&self, model: &ScalarStateSpace, steps: usize) -> Result<Vec<f64>> {
        validate_model(model)?;
        let mut state = *self
            .filtered_state
            .last()
            .ok_or(InferustError::InsufficientData { needed: 1, got: 0 })?;
        let mut out = Vec::with_capacity(steps);
        for _ in 0..steps {
            state = model.transition * state + model.state_intercept;
            out.push(model.design * state + model.observation_intercept);
        }
        Ok(out)
    }
}

/// Builder for local-level state-space smoothing.
#[derive(Debug, Clone)]
pub struct LocalLevel {
    observation_variance: f64,
    level_variance: f64,
    initial_level: Option<f64>,
    initial_variance: f64,
}

impl LocalLevel {
    pub fn new(observation_variance: f64, level_variance: f64) -> Self {
        Self {
            observation_variance,
            level_variance,
            initial_level: None,
            initial_variance: 1_000_000.0,
        }
    }

    pub fn initial_level(mut self, level: f64) -> Self {
        self.initial_level = Some(level);
        self
    }

    pub fn initial_variance(mut self, variance: f64) -> Self {
        self.initial_variance = variance;
        self
    }

    pub fn fit(&self, y: &[f64]) -> Result<LocalLevelResult> {
        if y.is_empty() {
            return Err(InferustError::InsufficientData { needed: 1, got: 0 });
        }
        let initial = self
            .initial_level
            .unwrap_or_else(|| y.iter().copied().find(|v| v.is_finite()).unwrap_or(0.0));
        let model = ScalarStateSpace::local_level(
            self.observation_variance,
            self.level_variance,
            initial,
            self.initial_variance,
        );
        let filter = model.filter(y)?;
        Ok(LocalLevelResult { model, filter })
    }
}

/// Fitted local-level model.
#[derive(Debug, Clone)]
pub struct LocalLevelResult {
    pub model: ScalarStateSpace,
    pub filter: KalmanFilterResult,
}

impl LocalLevelResult {
    pub fn level(&self) -> &[f64] {
        &self.filter.filtered_state
    }

    pub fn fitted_values(&self) -> &[f64] {
        &self.filter.forecasts
    }

    pub fn forecast(&self, steps: usize) -> Result<Vec<f64>> {
        self.filter.forecast(&self.model, steps)
    }
}

fn validate_model(model: &ScalarStateSpace) -> Result<()> {
    for (name, value) in [
        ("transition", model.transition),
        ("design", model.design),
        ("state_intercept", model.state_intercept),
        ("observation_intercept", model.observation_intercept),
        ("initial_state", model.initial_state),
    ] {
        if !value.is_finite() {
            return Err(InferustError::InvalidInput(format!(
                "{name} must be finite"
            )));
        }
    }
    if model.state_variance < 0.0 || !model.state_variance.is_finite() {
        return Err(InferustError::InvalidInput(
            "state_variance must be finite and non-negative".into(),
        ));
    }
    if model.observation_variance <= 0.0 || !model.observation_variance.is_finite() {
        return Err(InferustError::InvalidInput(
            "observation_variance must be finite and positive".into(),
        ));
    }
    if model.initial_variance < 0.0 || !model.initial_variance.is_finite() {
        return Err(InferustError::InvalidInput(
            "initial_variance must be finite and non-negative".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{LocalLevel, ScalarStateSpace};

    #[test]
    fn local_level_filters_noisy_series() {
        let y = vec![10.0, 10.4, 9.8, 11.0, 11.3, 11.1];
        let result = LocalLevel::new(0.25, 0.05).fit(&y).unwrap();
        assert_eq!(result.level().len(), y.len());
        assert!(result.filter.log_likelihood.is_finite());
        let forecast = result.forecast(2).unwrap();
        assert_eq!(forecast.len(), 2);
        assert!((forecast[0] - result.level().last().unwrap()).abs() < 1e-8);
    }

    #[test]
    fn scalar_filter_skips_missing_observations() {
        let model = ScalarStateSpace::local_level(1.0, 0.1, 0.0, 1.0);
        let result = model.filter(&[1.0, f64::NAN, 1.5]).unwrap();
        assert!(result.forecast_errors[1].is_nan());
        assert_eq!(result.filtered_state.len(), 3);
    }
}
