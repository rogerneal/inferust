//! State-space models and Kalman filtering.
//!
//! This module provides scalar and multivariate linear-Gaussian Kalman filters
//! that underpin exact ARIMA MLE and statsmodels-style state-space workflows.

mod kalman;
mod scalar;

pub use kalman::{KalmanLikelihood, LinearGaussianModel};
pub use scalar::{
    KalmanFilterResult, LocalLevel, LocalLevelResult, ScalarStateSpace,
};
