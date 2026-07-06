//! Unified post-estimation helpers across model result types.

use crate::glm::{GammaResult, LogisticResult, PoissonResult};
use crate::regression::OlsResult;

/// Common read-only interface for fitted model summaries.
pub trait ModelResult {
    fn coefficients(&self) -> &[f64];
    fn std_errors(&self) -> &[f64];
    fn feature_names(&self) -> &[String];
}

impl ModelResult for OlsResult {
    fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
    fn std_errors(&self) -> &[f64] {
        &self.std_errors
    }
    fn feature_names(&self) -> &[String] {
        &self.feature_names
    }
}

impl ModelResult for LogisticResult {
    fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
    fn std_errors(&self) -> &[f64] {
        &self.std_errors
    }
    fn feature_names(&self) -> &[String] {
        &self.feature_names
    }
}

impl ModelResult for PoissonResult {
    fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
    fn std_errors(&self) -> &[f64] {
        &self.std_errors
    }
    fn feature_names(&self) -> &[String] {
        &self.feature_names
    }
}

impl ModelResult for GammaResult {
    fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
    fn std_errors(&self) -> &[f64] {
        &self.std_errors
    }
    fn feature_names(&self) -> &[String] {
        &self.feature_names
    }
}

/// Single linear restriction `q'β = 0` Wald test for OLS.
pub fn wald_linear_contrast(
    result: &OlsResult,
    weights: &[f64],
) -> crate::Result<crate::hypothesis::wald::WaldTestResult> {
    result.wald_test(&[weights.to_vec()], &[0.0])
}
