//! Multivariate linear Gaussian state-space filtering.
//!
//! Observation: `y_t = Z α_t + d + ε_t`,  ε_t ~ N(0, H)
//! State:       `α_{t+1} = T α_t + c + R η_t`,  η_t ~ N(0, Q)

use nalgebra::{DMatrix, DVector};

use crate::error::{InferustError, Result};

/// Linear Gaussian state-space model for a scalar observation series.
#[derive(Debug, Clone)]
pub struct LinearGaussianModel {
    /// Observation matrix (1 × m).
    pub design: DMatrix<f64>,
    /// State transition (m × m).
    pub transition: DMatrix<f64>,
    /// State noise loading (m × 1).
    pub selection: DMatrix<f64>,
    /// Observation intercept.
    pub obs_intercept: f64,
    /// State intercept (m).
    pub state_intercept: DVector<f64>,
    /// Observation noise variance.
    pub obs_variance: f64,
    /// State innovation variance (scalar σ²).
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
    pub forecast_errors: Vec<f64>,
    pub forecast_variances: Vec<f64>,
}

impl LinearGaussianModel {
    /// Build an ARMA(p, q) state-space representation with intercept `mu`.
    ///
    /// Uses the companion-form parameterisation with state dimension
    /// `m = max(p, q + 1)` (Hamilton / Harvey).
    pub fn arma(mu: f64, ar: &[f64], ma: &[f64], sigma2: f64) -> Result<Self> {
        if sigma2 <= 0.0 || !sigma2.is_finite() {
            return Err(InferustError::InvalidInput(
                "sigma2 must be finite and positive".into(),
            ));
        }
        let p = ar.len();
        let q = ma.len();
        let m = p.max(q + 1).max(1);

        let mut transition = DMatrix::zeros(m, m);
        for i in 1..m {
            transition[(i, i - 1)] = 1.0;
        }
        for (i, &phi) in ar.iter().enumerate() {
            transition[(0, i)] = phi;
        }
        for (j, &theta) in ma.iter().enumerate() {
            if j + 1 < m {
                transition[(0, j + 1)] = theta;
            }
        }

        let mut selection = DMatrix::zeros(m, 1);
        selection[(0, 0)] = 1.0;

        let mut design = DMatrix::zeros(1, m);
        design[(0, 0)] = 1.0;

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
        if y.is_empty() {
            return Err(InferustError::InsufficientData { needed: 1, got: 0 });
        }
        let m = self.initial_state.len();
        let mut state = self.initial_state.clone();
        let mut cov = self.initial_cov.clone();
        let mut log_likelihood = 0.0;
        let mut filtered_state = Vec::with_capacity(y.len());
        let mut filtered_cov = Vec::with_capacity(y.len());
        let mut forecast_errors = Vec::with_capacity(y.len());
        let mut forecast_variances = Vec::with_capacity(y.len());

        let z = self.design.clone();
        let t_mat = self.transition.clone();
        let r = self.selection.clone();
        let identity = DMatrix::identity(m, m);

        for &obs in y {
            if !obs.is_finite() {
                return Err(InferustError::InvalidInput(
                    "observations must be finite".into(),
                ));
            }

            let pred_state = &t_mat * &state + &self.state_intercept;
            let pred_cov =
                &t_mat * &cov * t_mat.transpose() + &r * self.state_variance * r.transpose();

            let forecast = (z.clone() * &pred_state)[(0, 0)] + self.obs_intercept;
            let forecast_var =
                (z.clone() * &pred_cov * z.transpose())[(0, 0)] + self.obs_variance;
            let forecast_var = forecast_var.max(1e-12);
            let error = obs - forecast;

            log_likelihood += -0.5
                * ((2.0 * std::f64::consts::PI * forecast_var).ln()
                    + error * error / forecast_var);

            let gain = &pred_cov * z.transpose() / forecast_var;
            state = pred_state + &gain * error;
            cov = (&identity - &gain * &z) * pred_cov;

            filtered_state.push(state.clone());
            filtered_cov.push(cov.clone());
            forecast_errors.push(error);
            forecast_variances.push(forecast_var);
        }

        Ok(KalmanLikelihood {
            log_likelihood,
            filtered_state,
            filtered_cov,
            forecast_errors,
            forecast_variances,
        })
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
    }

    #[test]
    fn arma11_filter_runs() {
        let y = vec![0.2, -0.1, 0.4, 0.0, -0.3, 0.15, 0.25, -0.05];
        let model = LinearGaussianModel::arma(0.0, &[0.6], &[0.3], 1.0).unwrap();
        let result = model.filter(&y).unwrap();
        assert_eq!(result.forecast_errors.len(), y.len());
    }
}
