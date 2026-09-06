//! Unified post-estimation helpers across model result types.
//!
//! Result structs stay distinct (`OlsResult`, `GammaResult`, …). This module
//! is the shared read-only surface: coefficients, observation count, mean
//! predictions, and a coefficient-table printer.

use std::borrow::Cow;
use std::fmt;

use statrs::distribution::{ContinuousCDF, Normal};

use crate::discrete::{NegativeBinomialResult, ProbitResult};
use crate::error::{InferustError, Result};
use crate::glm::{GammaResult, InverseGaussianResult, LogisticResult, PoissonResult};
use crate::glm_family::GlmResult;
use crate::mixed::MixedLinearResult;
use crate::panel::PanelReResult;
use crate::regression::OlsResult;
use crate::robust::RobustLinearResult;
use crate::time_series::{ArimaResult, SarimaxResult};

/// Point prediction with optional standard errors and Wald interval.
#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    /// Conditional mean (response scale unless noted by the model).
    pub mean: Vec<f64>,
    /// Standard error of the mean, when the model supplies one.
    pub se: Option<Vec<f64>>,
    /// Lower interval bound (same length as [`Self::mean`] when present).
    pub lower: Option<Vec<f64>>,
    /// Upper interval bound (same length as [`Self::mean`] when present).
    pub upper: Option<Vec<f64>>,
}

impl Prediction {
    /// Mean-only prediction (no interval).
    pub fn from_mean(mean: Vec<f64>) -> Self {
        Self {
            mean,
            se: None,
            lower: None,
            upper: None,
        }
    }

    /// Mean plus a symmetric Wald interval `mean ± z · se`.
    pub fn with_wald_interval(mean: Vec<f64>, se: Vec<f64>, z: f64) -> Self {
        let lower = mean
            .iter()
            .zip(se.iter())
            .map(|(&m, &s)| m - z * s)
            .collect();
        let upper = mean
            .iter()
            .zip(se.iter())
            .map(|(&m, &s)| m + z * s)
            .collect();
        Self {
            mean,
            se: Some(se),
            lower: Some(lower),
            upper: Some(upper),
        }
    }
}

/// Common read-only interface for fitted model summaries.
///
/// Implementors keep their own result structs. `coefficients` / `std_errors` /
/// `feature_names` may be borrowed or synthesised (ARIMA packs intercept +
/// AR + MA).
pub trait ModelResult {
    fn coefficients(&self) -> Cow<'_, [f64]>;
    fn std_errors(&self) -> Cow<'_, [f64]>;
    fn feature_names(&self) -> Cow<'_, [String]>;
    fn nobs(&self) -> usize;
    /// Predict the conditional mean for new predictor rows (no intercept column).
    ///
    /// Time-series models treat an empty `x` as the in-sample fitted mean.
    fn predict(&self, x: &[Vec<f64>]) -> Result<Prediction>;
    /// Predict with a Wald interval at the given `alpha` (e.g. `0.05`).
    ///
    /// Default is the mean-only prediction when the model has no covariance.
    fn predict_interval(&self, x: &[Vec<f64>], alpha: f64) -> Result<Prediction> {
        let _ = alpha;
        self.predict(x)
    }
}

/// Structured coefficient table that can be displayed, logged, or tested.
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    pub nobs: usize,
    pub names: Vec<String>,
    pub coefficients: Vec<f64>,
    pub std_errors: Vec<f64>,
}

impl fmt::Display for Summary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f)?;
        writeln!(
            f,
            "── Model results ──────────────────────────────────────────"
        )?;
        writeln!(f, " Observations : {}", self.nobs)?;
        writeln!(f, " {:<20} {:>12} {:>12}", "variable", "coef", "std err")?;
        writeln!(f, " {}", "-".repeat(48))?;
        let n = self.names.len().max(self.coefficients.len());
        for i in 0..n {
            let name = self.names.get(i).map(String::as_str).unwrap_or("?");
            let coef = self.coefficients.get(i).copied().unwrap_or(f64::NAN);
            match self.std_errors.get(i) {
                Some(&se) => writeln!(f, " {name:<20} {coef:>12.6} {se:>12.6}")?,
                None => writeln!(f, " {name:<20} {coef:>12.6} {:>12}", "—")?,
            }
        }
        writeln!(f)?;
        Ok(())
    }
}

/// Build a [`Summary`] from any [`ModelResult`].
pub fn summary<R: ModelResult + ?Sized>(result: &R) -> Summary {
    Summary {
        nobs: result.nobs(),
        names: result.feature_names().into_owned(),
        coefficients: result.coefficients().into_owned(),
        std_errors: result.std_errors().into_owned(),
    }
}

/// Shared coefficient-table printer used by formula and matrix workflows.
pub fn print_summary<R: ModelResult + ?Sized>(result: &R) {
    print!("{}", summary(result));
}

fn linear_predictor(row: &[f64], coefficients: &[f64], names: &[String]) -> f64 {
    let offset = usize::from(names.first().is_some_and(|name| name == "const"));
    let mut eta = if offset == 1 {
        coefficients.first().copied().unwrap_or(0.0)
    } else {
        0.0
    };
    for (j, &value) in row.iter().enumerate() {
        if let Some(&coef) = coefficients.get(offset + j) {
            eta += coef * value;
        }
    }
    eta
}

fn mean_from_linear(result: &impl ModelResult, x: &[Vec<f64>]) -> Result<Prediction> {
    let coefs = result.coefficients();
    let names = result.feature_names();
    let mean = x
        .iter()
        .map(|row| linear_predictor(row, &coefs, &names))
        .collect();
    Ok(Prediction::from_mean(mean))
}

fn design_row(row: &[f64], names: &[String]) -> Vec<f64> {
    let mut design = Vec::with_capacity(row.len() + 1);
    if names.first().is_some_and(|name| name == "const") {
        design.push(1.0);
    }
    design.extend_from_slice(row);
    design
}

fn quadratic_form(row: &[f64], cov: &[Vec<f64>]) -> f64 {
    let k = row.len().min(cov.len());
    let mut acc = 0.0;
    for i in 0..k {
        for j in 0..k {
            acc += row[i] * cov[i][j] * row[j];
        }
    }
    acc.max(0.0)
}

fn normal_critical(alpha: f64) -> Result<f64> {
    if !(0.0..1.0).contains(&alpha) {
        return Err(InferustError::InvalidInput(
            "alpha must be between 0 and 1".into(),
        ));
    }
    let normal = Normal::new(0.0, 1.0)
        .map_err(|_| InferustError::InvalidInput("invalid normal distribution".into()))?;
    Ok(normal.inverse_cdf(1.0 - alpha / 2.0))
}

fn wald_from_cov(
    x: &[Vec<f64>],
    coefficients: &[f64],
    names: &[String],
    cov: &[Vec<f64>],
    alpha: f64,
    transform: impl Fn(f64) -> f64,
) -> Result<Prediction> {
    let z = normal_critical(alpha)?;
    let mut mean = Vec::with_capacity(x.len());
    let mut se = Vec::with_capacity(x.len());
    let mut lower = Vec::with_capacity(x.len());
    let mut upper = Vec::with_capacity(x.len());
    for row in x {
        let design = design_row(row, names);
        let eta = linear_predictor(row, coefficients, names);
        let se_eta = quadratic_form(&design, cov).sqrt();
        mean.push(transform(eta));
        se.push(se_eta);
        let a = transform(eta - z * se_eta);
        let b = transform(eta + z * se_eta);
        lower.push(a.min(b));
        upper.push(a.max(b));
    }
    Ok(Prediction {
        mean,
        se: Some(se),
        lower: Some(lower),
        upper: Some(upper),
    })
}

impl ModelResult for OlsResult {
    fn coefficients(&self) -> Cow<'_, [f64]> {
        Cow::Borrowed(&self.coefficients)
    }
    fn std_errors(&self) -> Cow<'_, [f64]> {
        Cow::Borrowed(&self.std_errors)
    }
    fn feature_names(&self) -> Cow<'_, [String]> {
        Cow::Borrowed(&self.feature_names)
    }
    fn nobs(&self) -> usize {
        self.n
    }
    fn predict(&self, x: &[Vec<f64>]) -> Result<Prediction> {
        Ok(Prediction::from_mean(OlsResult::predict(self, x)))
    }
    fn predict_interval(&self, x: &[Vec<f64>], alpha: f64) -> Result<Prediction> {
        wald_from_cov(
            x,
            &self.coefficients,
            &self.feature_names,
            &self.covariance_matrix,
            alpha,
            |eta| eta,
        )
    }
}

impl ModelResult for LogisticResult {
    fn coefficients(&self) -> Cow<'_, [f64]> {
        Cow::Borrowed(&self.coefficients)
    }
    fn std_errors(&self) -> Cow<'_, [f64]> {
        Cow::Borrowed(&self.std_errors)
    }
    fn feature_names(&self) -> Cow<'_, [String]> {
        Cow::Borrowed(&self.feature_names)
    }
    fn nobs(&self) -> usize {
        self.n
    }
    fn predict(&self, x: &[Vec<f64>]) -> Result<Prediction> {
        Ok(Prediction::from_mean(self.predict_proba(x)))
    }
    fn predict_interval(&self, x: &[Vec<f64>], alpha: f64) -> Result<Prediction> {
        wald_from_cov(
            x,
            &self.coefficients,
            &self.feature_names,
            &self.covariance_matrix,
            alpha,
            |eta| 1.0 / (1.0 + (-eta).exp()),
        )
    }
}

impl ModelResult for PoissonResult {
    fn coefficients(&self) -> Cow<'_, [f64]> {
        Cow::Borrowed(&self.coefficients)
    }
    fn std_errors(&self) -> Cow<'_, [f64]> {
        Cow::Borrowed(&self.std_errors)
    }
    fn feature_names(&self) -> Cow<'_, [String]> {
        Cow::Borrowed(&self.feature_names)
    }
    fn nobs(&self) -> usize {
        self.n
    }
    fn predict(&self, x: &[Vec<f64>]) -> Result<Prediction> {
        Ok(Prediction::from_mean(PoissonResult::predict(self, x)))
    }
    fn predict_interval(&self, x: &[Vec<f64>], alpha: f64) -> Result<Prediction> {
        wald_from_cov(
            x,
            &self.coefficients,
            &self.feature_names,
            &self.covariance_matrix,
            alpha,
            f64::exp,
        )
    }
}

impl ModelResult for GammaResult {
    fn coefficients(&self) -> Cow<'_, [f64]> {
        Cow::Borrowed(&self.coefficients)
    }
    fn std_errors(&self) -> Cow<'_, [f64]> {
        Cow::Borrowed(&self.std_errors)
    }
    fn feature_names(&self) -> Cow<'_, [String]> {
        Cow::Borrowed(&self.feature_names)
    }
    fn nobs(&self) -> usize {
        self.n
    }
    fn predict(&self, x: &[Vec<f64>]) -> Result<Prediction> {
        Ok(Prediction::from_mean(GammaResult::predict(self, x)))
    }
    fn predict_interval(&self, x: &[Vec<f64>], alpha: f64) -> Result<Prediction> {
        let ivs = self.predict_mean_intervals(x, alpha)?;
        Ok(Prediction {
            mean: ivs.iter().map(|i| i.mean).collect(),
            se: None,
            lower: Some(ivs.iter().map(|i| i.lower).collect()),
            upper: Some(ivs.iter().map(|i| i.upper).collect()),
        })
    }
}

impl ModelResult for InverseGaussianResult {
    fn coefficients(&self) -> Cow<'_, [f64]> {
        Cow::Borrowed(&self.coefficients)
    }
    fn std_errors(&self) -> Cow<'_, [f64]> {
        Cow::Borrowed(&self.std_errors)
    }
    fn feature_names(&self) -> Cow<'_, [String]> {
        Cow::Borrowed(&self.feature_names)
    }
    fn nobs(&self) -> usize {
        self.n
    }
    fn predict(&self, x: &[Vec<f64>]) -> Result<Prediction> {
        Ok(Prediction::from_mean(InverseGaussianResult::predict(
            self, x,
        )))
    }
}

impl ModelResult for ProbitResult {
    fn coefficients(&self) -> Cow<'_, [f64]> {
        Cow::Borrowed(&self.coefficients)
    }
    fn std_errors(&self) -> Cow<'_, [f64]> {
        Cow::Borrowed(&self.std_errors)
    }
    fn feature_names(&self) -> Cow<'_, [String]> {
        Cow::Borrowed(&self.feature_names)
    }
    fn nobs(&self) -> usize {
        self.fitted_probabilities.len()
    }
    fn predict(&self, x: &[Vec<f64>]) -> Result<Prediction> {
        Ok(Prediction::from_mean(self.predict_proba(x)?))
    }
}

impl ModelResult for NegativeBinomialResult {
    fn coefficients(&self) -> Cow<'_, [f64]> {
        Cow::Borrowed(&self.coefficients)
    }
    fn std_errors(&self) -> Cow<'_, [f64]> {
        Cow::Borrowed(&self.std_errors)
    }
    fn feature_names(&self) -> Cow<'_, [String]> {
        Cow::Borrowed(&self.feature_names)
    }
    fn nobs(&self) -> usize {
        self.fitted_values.len()
    }
    fn predict(&self, x: &[Vec<f64>]) -> Result<Prediction> {
        let mean = x
            .iter()
            .map(|row| linear_predictor(row, &self.coefficients, &self.feature_names).exp())
            .collect();
        Ok(Prediction::from_mean(mean))
    }
}

impl ModelResult for PanelReResult {
    fn coefficients(&self) -> Cow<'_, [f64]> {
        Cow::Borrowed(&self.coefficients)
    }
    fn std_errors(&self) -> Cow<'_, [f64]> {
        Cow::Borrowed(&self.std_errors)
    }
    fn feature_names(&self) -> Cow<'_, [String]> {
        Cow::Borrowed(&self.feature_names)
    }
    fn nobs(&self) -> usize {
        self.n
    }
    fn predict(&self, x: &[Vec<f64>]) -> Result<Prediction> {
        mean_from_linear(self, x)
    }
}

impl ModelResult for RobustLinearResult {
    fn coefficients(&self) -> Cow<'_, [f64]> {
        Cow::Borrowed(&self.fit.coefficients)
    }
    fn std_errors(&self) -> Cow<'_, [f64]> {
        Cow::Borrowed(&self.robust_std_errors)
    }
    fn feature_names(&self) -> Cow<'_, [String]> {
        Cow::Borrowed(&self.fit.feature_names)
    }
    fn nobs(&self) -> usize {
        self.fit.n
    }
    fn predict(&self, x: &[Vec<f64>]) -> Result<Prediction> {
        Ok(Prediction::from_mean(self.fit.predict(x)))
    }
}

impl ModelResult for MixedLinearResult {
    fn coefficients(&self) -> Cow<'_, [f64]> {
        Cow::Borrowed(&self.coefficients)
    }
    fn std_errors(&self) -> Cow<'_, [f64]> {
        Cow::Borrowed(&self.std_errors)
    }
    fn feature_names(&self) -> Cow<'_, [String]> {
        Cow::Borrowed(&self.feature_names)
    }
    fn nobs(&self) -> usize {
        self.fitted_values.len()
    }
    fn predict(&self, x: &[Vec<f64>]) -> Result<Prediction> {
        // Fixed-effects mean only; random intercepts need group ids.
        mean_from_linear(self, x)
    }
}

impl ModelResult for ArimaResult {
    fn coefficients(&self) -> Cow<'_, [f64]> {
        let mut packed = Vec::with_capacity(1 + self.p + self.q);
        packed.push(self.intercept);
        packed.extend_from_slice(&self.ar_coefficients);
        packed.extend_from_slice(&self.ma_coefficients);
        Cow::Owned(packed)
    }
    fn std_errors(&self) -> Cow<'_, [f64]> {
        Cow::Owned(vec![])
    }
    fn feature_names(&self) -> Cow<'_, [String]> {
        let mut names = vec!["const".to_string()];
        for i in 1..=self.p {
            names.push(format!("ar.L{i}"));
        }
        for i in 1..=self.q {
            names.push(format!("ma.L{i}"));
        }
        Cow::Owned(names)
    }
    fn nobs(&self) -> usize {
        self.n
    }
    fn predict(&self, x: &[Vec<f64>]) -> Result<Prediction> {
        if x.is_empty() {
            return Ok(Prediction::from_mean(self.fitted_values.clone()));
        }
        Err(InferustError::InvalidInput(
            "ARIMA predict() is in-sample only; pass empty x or use forecast()".into(),
        ))
    }
}

impl ModelResult for SarimaxResult {
    fn coefficients(&self) -> Cow<'_, [f64]> {
        Cow::Borrowed(&self.exog_coefficients)
    }
    fn std_errors(&self) -> Cow<'_, [f64]> {
        Cow::Owned(vec![])
    }
    fn feature_names(&self) -> Cow<'_, [String]> {
        Cow::Borrowed(&self.exog_names)
    }
    fn nobs(&self) -> usize {
        self.sarima.n
    }
    fn predict(&self, x: &[Vec<f64>]) -> Result<Prediction> {
        mean_from_linear(self, x)
    }
}

impl ModelResult for GlmResult {
    fn coefficients(&self) -> Cow<'_, [f64]> {
        match self {
            GlmResult::Gaussian(r) => ModelResult::coefficients(r),
            GlmResult::Binomial(r) => ModelResult::coefficients(r),
            GlmResult::Poisson(r) => ModelResult::coefficients(r),
            GlmResult::Gamma(r) => ModelResult::coefficients(r),
            GlmResult::InverseGaussian(r) => ModelResult::coefficients(r),
        }
    }
    fn std_errors(&self) -> Cow<'_, [f64]> {
        match self {
            GlmResult::Gaussian(r) => ModelResult::std_errors(r),
            GlmResult::Binomial(r) => ModelResult::std_errors(r),
            GlmResult::Poisson(r) => ModelResult::std_errors(r),
            GlmResult::Gamma(r) => ModelResult::std_errors(r),
            GlmResult::InverseGaussian(r) => ModelResult::std_errors(r),
        }
    }
    fn feature_names(&self) -> Cow<'_, [String]> {
        match self {
            GlmResult::Gaussian(r) => ModelResult::feature_names(r),
            GlmResult::Binomial(r) => ModelResult::feature_names(r),
            GlmResult::Poisson(r) => ModelResult::feature_names(r),
            GlmResult::Gamma(r) => ModelResult::feature_names(r),
            GlmResult::InverseGaussian(r) => ModelResult::feature_names(r),
        }
    }
    fn nobs(&self) -> usize {
        match self {
            GlmResult::Gaussian(r) => r.n,
            GlmResult::Binomial(r) => r.n,
            GlmResult::Poisson(r) => r.n,
            GlmResult::Gamma(r) => r.n,
            GlmResult::InverseGaussian(r) => r.n,
        }
    }
    fn predict(&self, x: &[Vec<f64>]) -> Result<Prediction> {
        match self {
            GlmResult::Gaussian(r) => ModelResult::predict(r, x),
            GlmResult::Binomial(r) => ModelResult::predict(r, x),
            GlmResult::Poisson(r) => ModelResult::predict(r, x),
            GlmResult::Gamma(r) => ModelResult::predict(r, x),
            GlmResult::InverseGaussian(r) => ModelResult::predict(r, x),
        }
    }
    fn predict_interval(&self, x: &[Vec<f64>], alpha: f64) -> Result<Prediction> {
        match self {
            GlmResult::Gaussian(r) => ModelResult::predict_interval(r, x, alpha),
            GlmResult::Binomial(r) => ModelResult::predict_interval(r, x, alpha),
            GlmResult::Poisson(r) => ModelResult::predict_interval(r, x, alpha),
            GlmResult::Gamma(r) => ModelResult::predict_interval(r, x, alpha),
            GlmResult::InverseGaussian(r) => ModelResult::predict(r, x),
        }
    }
}

/// Single linear restriction `q'β = 0` Wald test for OLS.
pub fn wald_linear_contrast(
    result: &OlsResult,
    weights: &[f64],
) -> crate::Result<crate::hypothesis::wald::WaldTestResult> {
    result.wald_test(&[weights.to_vec()], &[0.0])
}

#[cfg(test)]
mod tests {
    use super::{print_summary, ModelResult, Prediction};
    use crate::regression::Ols;

    #[test]
    fn ols_model_result_predicts_training_mean() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let y = vec![3.0, 5.0, 7.0, 9.0];
        let fit = Ols::new().fit(&x, &y).unwrap();
        let pred = ModelResult::predict(&fit, &x).unwrap();
        assert_eq!(pred.mean.len(), 4);
        assert!((pred.mean[0] - fit.fitted_values[0]).abs() < 1e-12);
        assert_eq!(fit.nobs(), 4);
        assert_eq!(fit.feature_names()[0], "const");
        let iv = ModelResult::predict_interval(&fit, &x, 0.05).unwrap();
        assert!(iv.lower.as_ref().unwrap()[0] <= iv.mean[0] + 1e-12);
        assert!(iv.upper.as_ref().unwrap()[0] >= iv.mean[0] - 1e-12);
        assert!(iv.se.as_ref().unwrap()[0] >= 0.0);
        let text = crate::post_estimation::summary(&fit).to_string();
        assert!(text.contains("const"));
        assert!(fit.summary().contains("OLS Regression Results"));
        print_summary(&fit);
    }

    #[test]
    fn prediction_wald_interval_is_symmetric() {
        let pred = Prediction::with_wald_interval(vec![10.0], vec![2.0], 1.96);
        assert!((pred.lower.unwrap()[0] - (10.0 - 1.96 * 2.0)).abs() < 1e-12);
        assert!((pred.upper.unwrap()[0] - (10.0 + 1.96 * 2.0)).abs() < 1e-12);
    }
}
