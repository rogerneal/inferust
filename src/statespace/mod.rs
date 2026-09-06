//! State-space models and Kalman filtering.
//!
//! Multivariate [`LinearGaussianModel`] exposes the usual `Z`, `T`, `R`, `H`,
//! `Q` matrices, a Kalman filter, an RTS smoother, and multi-step forecasts.
//! Scalar local-level models live in [`ScalarStateSpace`].

mod kalman;
mod scalar;

pub use kalman::{KalmanLikelihood, KalmanSmooth, LinearGaussianModel, StateSpaceForecast};
pub use scalar::{KalmanFilterResult, LocalLevel, LocalLevelResult, ScalarStateSpace};
