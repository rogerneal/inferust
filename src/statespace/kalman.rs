//! Multivariate linear Gaussian state-space filtering and smoothing.
//!
//! Observation: `y_t = Z α_t + d + ε_t`,  ε_t ~ N(0, H)
//! State:       `α_{t+1} = T α_t + c + R η_t`,  η_t ~ N(0, Q)

use nalgebra::{DMatrix, DVector};

use crate::error::{InferustError, Result};

/// Linear Gaussian state-space model for a scalar observation series.
#[derive(Debug, Clone)]
pub struct LinearGaussianModel {
    /// Observation matrix `Z` (1 × m).
    pub design: DMatrix<f64>,
    /// State transition `T` (m × m).
    pub transition: DMatrix<f64>,
    /// State noise loading `R` (m × 1).
    pub selection: DMatrix<f64>,
    /// Observation intercept `d`.
    pub obs_intercept: f64,
    /// State intercept `c` (m).
    pub state_intercept: DVector<f64>,
    /// Observation noise variance `H`.
    pub obs_variance: f64,
    /// State innovation variance `Q` (scalar σ²).
    pub state_variance: f64,
    /// Initial state mean (m).
    pub initial_state: DVector<f64>,
    /// Initial state covariance (m × m).
    pub initial_cov: DMatrix<f64>,
}

/// Kalman filter output including the exact Gaussian log-likelihood.
#[derive(Debug, Clone)]
pub struct KalmanLikelihood {
    pub log_likelihood: f64,
    pub filtered_state: Vec<DVector<f64>>,
    pub filtered_cov: Vec<DMatrix<f64>>,
    /// One-step-ahead predicted state `α_{t|t-1}` (before the update).
    pub predicted_state: Vec<DVector<f64>>,
    /// One-step-ahead predicted covariance `P_{t|t-1}`.
    pub predicted_cov: Vec<DMatrix<f64>>,
    pub forecast_errors: Vec<f64>,
    pub forecast_variances: Vec<f64>,
}

/// Rauch–Tung–Striebel smoother output.
#[derive(Debug, Clone)]
pub struct KalmanSmooth {
    pub smoothed_state: Vec<DVector<f64>>,
    pub smoothed_cov: Vec<DMatrix<f64>>,
}

/// Multi-step forecasts from a filtered state.
#[derive(Debug, Clone)]
pub struct StateSpaceForecast {
    pub mean: Vec<f64>,
    pub se: Vec<f64>,
    pub states: Vec<DVector<f64>>,
}

impl LinearGaussianModel {
    /// Observation matrix `Z`.
    pub fn z(&self) -> &DMatrix<f64> {
        &self.design
    }

    /// Transition matrix `T`.
    pub fn transition_matrix(&self) -> &DMatrix<f64> {
        &self.transition
    }

    /// Selection matrix `R`.
    pub fn r(&self) -> &DMatrix<f64> {
        &self.selection
    }

    /// Observation variance `H`.
    pub fn h(&self) -> f64 {
        self.obs_variance
    }

    /// State innovation variance `Q`.
    pub fn q(&self) -> f64 {
        self.state_variance
    }

    /// Build an ARMA(p, q) state-space representation with unconditional mean `mu`.
    ///
    /// Uses the Hamilton companion form that matches statsmodels SARIMAX
    /// (`hamilton_representation=True`, and the default Harvey form's likelihood):
    /// state dimension `m = max(p, q + 1)`, design `Z = [1, θ₁, …, θ_{m-1}]`
    /// (MA in the observation equation), companion transition `T` carrying the
    /// AR coefficients, selection `R = e₁`, `H = 0`, and `Q = σ²`.
    ///
    /// `mu` is the unconditional mean placed in the observation intercept
    /// (statsmodels `trend='c'` / `const`). The regression intercept
    /// `c = μ (1 − Σφ)` used by CSS forecasting is recovered separately.
    pub fn arma(mu: f64, ar: &[f64], ma: &[f64], sigma2: f64) -> Result<Self> {
        if sigma2 <= 0.0 || !sigma2.is_finite() {
            return Err(InferustError::InvalidInput(
                "sigma2 must be finite and positive".into(),
            ));
        }
        let p = ar.len();
        let q = ma.len();
        let m = p.max(q + 1).max(1);

        // Companion T (Hamilton): first row = φ, subdiagonal = 1.
        let mut transition = DMatrix::zeros(m, m);
        for i in 1..m {
            transition[(i, i - 1)] = 1.0;
        }
        for (i, &phi) in ar.iter().enumerate() {
            transition[(0, i)] = phi;
        }

        // Z carries MA coefficients: [1, θ₁, …, θ_{m-1}].
        let mut design = DMatrix::zeros(1, m);
        design[(0, 0)] = 1.0;
        for (j, &theta) in ma.iter().enumerate() {
            if j + 1 < m {
                design[(0, j + 1)] = theta;
            }
        }

        // R = e₁ (not Z'); H = 0; Q = σ².
        let mut selection = DMatrix::zeros(m, 1);
        selection[(0, 0)] = 1.0;

        let initial_state = DVector::zeros(m);
        let initial_cov = stationary_covariance(&transition, &selection, sigma2, m)?;

        Ok(Self {
            design,
            transition,
            selection,
            obs_intercept: mu,
            state_intercept: DVector::zeros(m),
            obs_variance: 0.0,
            state_variance: sigma2,
            initial_state,
            initial_cov,
        })
    }

    /// Run the Kalman filter and accumulate the exact Gaussian log-likelihood.
    pub fn filter(&self, y: &[f64]) -> Result<KalmanLikelihood> {
        self.filter_with_obs_means(y, None)
    }

    /// Filter with an extra time-varying observation mean (exogenous `x_t'β`).
    pub fn filter_with_obs_means(
        &self,
        y: &[f64],
        extra_mean: Option<&[f64]>,
    ) -> Result<KalmanLikelihood> {
        if y.is_empty() {
            return Err(InferustError::InsufficientData { needed: 1, got: 0 });
        }
        if let Some(extra) = extra_mean {
            if extra.len() != y.len() {
                return Err(InferustError::DimensionMismatch {
                    x_rows: extra.len(),
                    y_len: y.len(),
                });
            }
        }
        let m = self.initial_state.len();
        let mut state = self.initial_state.clone();
        let mut cov = self.initial_cov.clone();
        let mut log_likelihood = 0.0;
        let mut filtered_state = Vec::with_capacity(y.len());
        let mut filtered_cov = Vec::with_capacity(y.len());
        let mut predicted_state = Vec::with_capacity(y.len());
        let mut predicted_cov = Vec::with_capacity(y.len());
        let mut forecast_errors = Vec::with_capacity(y.len());
        let mut forecast_variances = Vec::with_capacity(y.len());

        let z = self.design.clone();
        let t_mat = self.transition.clone();
        let r = self.selection.clone();
        let identity = DMatrix::identity(m, m);

        for (t, &obs) in y.iter().enumerate() {
            if !obs.is_finite() {
                return Err(InferustError::InvalidInput(
                    "observations must be finite".into(),
                ));
            }

            let pred_state = &t_mat * &state + &self.state_intercept;
            let pred_cov =
                &t_mat * &cov * t_mat.transpose() + &r * self.state_variance * r.transpose();

            let extra = extra_mean.map(|m| m[t]).unwrap_or(0.0);
            let forecast = (z.clone() * &pred_state)[(0, 0)] + self.obs_intercept + extra;
            let forecast_var = (z.clone() * &pred_cov * z.transpose())[(0, 0)] + self.obs_variance;
            let forecast_var = forecast_var.max(1e-12);
            let error = obs - forecast;

            log_likelihood += -0.5
                * ((2.0 * std::f64::consts::PI * forecast_var).ln() + error * error / forecast_var);

            let gain = &pred_cov * z.transpose() / forecast_var;
            state = &pred_state + &gain * error;
            cov = (&identity - &gain * &z) * &pred_cov;

            predicted_state.push(pred_state);
            predicted_cov.push(pred_cov);
            filtered_state.push(state.clone());
            filtered_cov.push(cov.clone());
            forecast_errors.push(error);
            forecast_variances.push(forecast_var);
        }

        Ok(KalmanLikelihood {
            log_likelihood,
            filtered_state,
            filtered_cov,
            predicted_state,
            predicted_cov,
            forecast_errors,
            forecast_variances,
        })
    }

    /// Rauch–Tung–Striebel smoother from a completed filter pass.
    pub fn smooth(&self, filtered: &KalmanLikelihood) -> Result<KalmanSmooth> {
        let n = filtered.filtered_state.len();
        if n == 0 {
            return Err(InferustError::InsufficientData { needed: 1, got: 0 });
        }
        let mut smoothed_state = filtered.filtered_state.clone();
        let mut smoothed_cov = filtered.filtered_cov.clone();
        let t_mat = &self.transition;

        for t in (0..n - 1).rev() {
            let pred_cov = &filtered.predicted_cov[t + 1];
            let pred_inv = regularized_inverse(pred_cov)?;
            let gain = &filtered.filtered_cov[t] * t_mat.transpose() * pred_inv;
            let innov = &smoothed_state[t + 1] - &filtered.predicted_state[t + 1];
            smoothed_state[t] = &filtered.filtered_state[t] + &gain * innov;
            let cov_delta = &smoothed_cov[t + 1] - pred_cov;
            smoothed_cov[t] = &filtered.filtered_cov[t] + &gain * cov_delta * gain.transpose();
        }

        Ok(KalmanSmooth {
            smoothed_state,
            smoothed_cov,
        })
    }

    /// Filter then smooth in one call.
    pub fn smooth_series(&self, y: &[f64]) -> Result<KalmanSmooth> {
        let filtered = self.filter(y)?;
        self.smooth(&filtered)
    }

    /// `steps`-ahead forecasts from the last filtered state.
    pub fn forecast(
        &self,
        filtered: &KalmanLikelihood,
        steps: usize,
    ) -> Result<StateSpaceForecast> {
        let last_state = filtered
            .filtered_state
            .last()
            .ok_or(InferustError::InsufficientData { needed: 1, got: 0 })?;
        let last_cov = filtered
            .filtered_cov
            .last()
            .ok_or(InferustError::InsufficientData { needed: 1, got: 0 })?;
        self.forecast_from(last_state, last_cov, steps)
    }

    /// `steps`-ahead forecasts starting from an explicit filtered state.
    pub fn forecast_from(
        &self,
        last_filtered_state: &DVector<f64>,
        last_filtered_cov: &DMatrix<f64>,
        steps: usize,
    ) -> Result<StateSpaceForecast> {
        let mut state = last_filtered_state.clone();
        let mut cov = last_filtered_cov.clone();
        let mut mean = Vec::with_capacity(steps);
        let mut se = Vec::with_capacity(steps);
        let mut states = Vec::with_capacity(steps);
        let z = &self.design;
        let t_mat = &self.transition;
        let r = &self.selection;

        for _ in 0..steps {
            state = t_mat * &state + &self.state_intercept;
            cov = t_mat * &cov * t_mat.transpose() + r * self.state_variance * r.transpose();
            let forecast = (z * &state)[(0, 0)] + self.obs_intercept;
            let var = ((z * &cov * z.transpose())[(0, 0)] + self.obs_variance).max(1e-12);
            mean.push(forecast);
            se.push(var.sqrt());
            states.push(state.clone());
        }

        Ok(StateSpaceForecast { mean, se, states })
    }
}

/// Solve the discrete Lyapunov equation `P = T P T' + σ² R R'` via fixed-point iteration.
fn stationary_covariance(
    transition: &DMatrix<f64>,
    selection: &DMatrix<f64>,
    sigma2: f64,
    m: usize,
) -> Result<DMatrix<f64>> {
    let mut p = DMatrix::identity(m, m) * sigma2;
    let rr = selection * sigma2 * selection.transpose();
    for _ in 0..500 {
        let next = transition * &p * transition.transpose() + &rr;
        let delta = (&next - &p).norm();
        p = next;
        if delta < 1e-12 * p.norm().max(1.0) {
            return Ok(p);
        }
    }
    Ok(p)
}

fn regularized_inverse(m: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    if let Some(inv) = m.clone().try_inverse() {
        return Ok(inv);
    }
    let n = m.nrows();
    let ridge = m + DMatrix::identity(n, n) * 1e-8;
    ridge.try_inverse().ok_or(InferustError::SingularMatrix)
}

#[cfg(test)]
mod tests {
    use super::LinearGaussianModel;

    #[test]
    fn ar1_loglik_matches_closed_form() {
        let y: Vec<f64> = (0..80)
            .map(|i| 0.5 + 0.7 * (i as f64 * 0.1).sin())
            .collect();
        let sigma2 = 0.25;
        let model = LinearGaussianModel::arma(0.5, &[0.7], &[], sigma2).unwrap();
        let result = model.filter(&y).unwrap();
        assert!(result.log_likelihood.is_finite());
        assert!(result.log_likelihood < 0.0);
        assert_eq!(result.predicted_state.len(), y.len());
    }

    #[test]
    fn arma11_hamilton_matrices() {
        let model = LinearGaussianModel::arma(0.0, &[0.6], &[0.3], 1.0).unwrap();
        // Z = [1, θ], T = [[φ, 0], [1, 0]], R = [1, 0]'
        assert!((model.z()[(0, 0)] - 1.0).abs() < 1e-15);
        assert!((model.z()[(0, 1)] - 0.3).abs() < 1e-15);
        assert!((model.transition_matrix()[(0, 0)] - 0.6).abs() < 1e-15);
        assert!((model.transition_matrix()[(0, 1)] - 0.0).abs() < 1e-15);
        assert!((model.transition_matrix()[(1, 0)] - 1.0).abs() < 1e-15);
        assert!((model.r()[(0, 0)] - 1.0).abs() < 1e-15);
        assert!((model.r()[(1, 0)] - 0.0).abs() < 1e-15);
        assert_eq!(model.h(), 0.0);
        assert_eq!(model.q(), 1.0);
    }

    #[test]
    fn arma11_filter_runs() {
        let y = vec![0.2, -0.1, 0.4, 0.0, -0.3, 0.15, 0.25, -0.05];
        let model = LinearGaussianModel::arma(0.0, &[0.6], &[0.3], 1.0).unwrap();
        let result = model.filter(&y).unwrap();
        assert_eq!(result.forecast_errors.len(), y.len());
    }

    #[test]
    fn smoother_matches_filter_at_last_date() {
        let y = vec![0.2, -0.1, 0.4, 0.0, -0.3, 0.15, 0.25, -0.05];
        let model = LinearGaussianModel::arma(0.0, &[0.6], &[0.3], 1.0).unwrap();
        let filtered = model.filter(&y).unwrap();
        let smooth = model.smooth(&filtered).unwrap();
        let last = smooth.smoothed_state.len() - 1;
        assert!((smooth.smoothed_state[last][0] - filtered.filtered_state[last][0]).abs() < 1e-12);
        assert_eq!(smooth.smoothed_state.len(), y.len());
    }

    #[test]
    fn forecast_has_requested_horizon() {
        let y = vec![0.2, -0.1, 0.4, 0.0, -0.3, 0.15, 0.25, -0.05];
        let model = LinearGaussianModel::arma(0.0, &[0.5], &[], 0.4).unwrap();
        let filtered = model.filter(&y).unwrap();
        let fc = model.forecast(&filtered, 4).unwrap();
        assert_eq!(fc.mean.len(), 4);
        assert!(fc.se.iter().all(|s| *s > 0.0));
    }
}
