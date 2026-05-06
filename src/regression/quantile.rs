use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ContinuousCDF, Normal};

use crate::error::{InferustError, Result};
use crate::regression::{Ols, Wls};

/// Fitted quantile regression result.
///
/// This mirrors the core post-estimation surface of `statsmodels.QuantReg`:
/// coefficients, asymptotic standard errors, z statistics, p-values,
/// confidence intervals, fitted values, residuals, check loss, and pseudo R1.
#[derive(Debug, Clone)]
pub struct QuantileRegressionResult {
    /// Quantile level, e.g. `0.5` for median regression.
    pub quantile: f64,
    /// Estimated coefficients (intercept first unless `no_intercept()` was used).
    pub coefficients: Vec<f64>,
    /// Asymptotic standard errors based on a residual sparsity estimate.
    pub std_errors: Vec<f64>,
    /// z-statistics: `coef / std_err`.
    pub z_statistics: Vec<f64>,
    /// Two-sided normal p-values.
    pub p_values: Vec<f64>,
    /// Fitted values.
    pub fitted_values: Vec<f64>,
    /// Raw residuals `y - fitted`.
    pub residuals: Vec<f64>,
    /// Sum of quantile check losses.
    pub objective: f64,
    /// Koenker-Machado style pseudo R1 relative to an intercept-only quantile fit.
    pub pseudo_r1: f64,
    /// Estimated sparsity, i.e. reciprocal density at the target quantile.
    pub sparsity: f64,
    /// Number of observations.
    pub n: usize,
    /// Residual degrees of freedom.
    pub df_resid: usize,
    /// Number of IRLS iterations used.
    pub iterations: usize,
    /// Whether the IRLS coefficient path met the requested tolerance.
    pub converged: bool,
    /// Names of each term in the model.
    pub feature_names: Vec<String>,
}

/// Linear quantile regression fitted by iteratively reweighted least squares.
///
/// Quantile regression estimates conditional quantiles instead of the
/// conditional mean. `QuantileRegression::new(0.5)` gives median regression,
/// matching the common `statsmodels.regression.quantile_regression.QuantReg`
/// workflow.
#[derive(Debug, Clone)]
pub struct QuantileRegression {
    quantile: f64,
    feature_names: Vec<String>,
    add_intercept: bool,
    max_iter: usize,
    tolerance: f64,
    weight_epsilon: f64,
}

impl QuantileRegression {
    /// Create a quantile regression builder for `quantile` in `(0, 1)`.
    pub fn new(quantile: f64) -> Self {
        Self {
            quantile,
            feature_names: Vec::new(),
            add_intercept: true,
            max_iter: 1000,
            tolerance: 1e-8,
            weight_epsilon: 1e-6,
        }
    }

    /// Set human-readable names for the predictor columns.
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Fit without an intercept.
    pub fn no_intercept(mut self) -> Self {
        self.add_intercept = false;
        self
    }

    /// Override the maximum IRLS iterations.
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Override the convergence tolerance on the coefficient path.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Override the residual floor used in IRLS weights.
    pub fn weight_epsilon(mut self, epsilon: f64) -> Self {
        self.weight_epsilon = epsilon;
        self
    }

    /// Fit the quantile regression model.
    ///
    /// * `x` - slice of rows; each row is one observation and must have the same length.
    /// * `y` - dependent variable, same length as `x`.
    pub fn fit(&self, x: &[Vec<f64>], y: &[f64]) -> Result<QuantileRegressionResult> {
        self.validate_inputs(x, y)?;

        let mut initial = Ols::new()
            .stable()
            .with_feature_names(self.feature_names.clone());
        if !self.add_intercept {
            initial = initial.no_intercept();
        }
        let initial = initial.fit(x, y)?;
        let mut coefficients = initial.coefficients;
        let mut iterations = 0;
        let mut converged = false;

        for iter in 0..self.max_iter {
            iterations = iter + 1;
            let fitted = predict_with_intercept(x, &coefficients, self.add_intercept)?;
            let residuals = y
                .iter()
                .zip(fitted.iter())
                .map(|(actual, fitted)| actual - fitted)
                .collect::<Vec<_>>();
            let weights = residuals
                .iter()
                .map(|residual| {
                    let side_weight = if *residual >= 0.0 {
                        self.quantile
                    } else {
                        1.0 - self.quantile
                    };
                    (side_weight / residual.abs().max(self.weight_epsilon)).min(1.0e12)
                })
                .collect::<Vec<_>>();

            let mut wls = Wls::new()
                .stable()
                .with_feature_names(self.feature_names.clone());
            if !self.add_intercept {
                wls = wls.no_intercept();
            }
            let next = wls.fit(x, y, &weights)?;
            let max_delta = coefficients
                .iter()
                .zip(next.coefficients.iter())
                .map(|(old, new)| (old - new).abs())
                .fold(0.0, f64::max);
            coefficients = next.coefficients;
            if max_delta < self.tolerance {
                converged = true;
                break;
            }
        }

        let fitted_values = predict_with_intercept(x, &coefficients, self.add_intercept)?;
        let residuals = y
            .iter()
            .zip(fitted_values.iter())
            .map(|(actual, fitted)| actual - fitted)
            .collect::<Vec<_>>();
        let objective = check_loss_sum(&residuals, self.quantile);
        let null_quantile = empirical_quantile(y, self.quantile);
        let null_residuals = y
            .iter()
            .map(|value| value - null_quantile)
            .collect::<Vec<_>>();
        let null_objective = check_loss_sum(&null_residuals, self.quantile);
        let pseudo_r1 = if null_objective <= f64::EPSILON {
            f64::NAN
        } else {
            1.0 - objective / null_objective
        };

        let design = design_matrix(x, self.add_intercept)?;
        let xtx_inv = xtx_inverse(&design)?;
        let bandwidth = residual_bandwidth(y.len(), self.quantile);
        let lo = (self.quantile - bandwidth).max(0.01);
        let hi = (self.quantile + bandwidth).min(0.99);
        let q_lo = empirical_quantile(&residuals, lo);
        let q_hi = empirical_quantile(&residuals, hi);
        let sparsity = ((q_hi - q_lo) / (hi - lo)).abs().max(1e-12);
        let scale = self.quantile * (1.0 - self.quantile) * sparsity.powi(2);
        let std_errors = (0..coefficients.len())
            .map(|i| (scale * xtx_inv[(i, i)]).max(0.0).sqrt())
            .collect::<Vec<_>>();
        let z_statistics = coefficients
            .iter()
            .zip(std_errors.iter())
            .map(|(coef, se)| coef / se)
            .collect::<Vec<_>>();
        let normal = Normal::new(0.0, 1.0)
            .map_err(|_| InferustError::InvalidInput("invalid normal distribution".into()))?;
        let p_values = z_statistics
            .iter()
            .map(|z| 2.0 * (1.0 - normal.cdf(z.abs())))
            .collect::<Vec<_>>();

        let ncols = coefficients.len();
        let mut feature_names = Vec::with_capacity(ncols);
        if self.add_intercept {
            feature_names.push("const".to_string());
        }
        if self.feature_names.is_empty() {
            let start = if self.add_intercept { 1 } else { 0 };
            feature_names.extend((start..ncols).map(|i| format!("x{}", i - start + 1)));
        } else {
            feature_names.extend(self.feature_names.clone());
        }

        Ok(QuantileRegressionResult {
            quantile: self.quantile,
            coefficients,
            std_errors,
            z_statistics,
            p_values,
            fitted_values,
            residuals,
            objective,
            pseudo_r1,
            sparsity,
            n: y.len(),
            df_resid: y.len() - ncols,
            iterations,
            converged,
            feature_names,
        })
    }

    fn validate_inputs(&self, x: &[Vec<f64>], y: &[f64]) -> Result<()> {
        if !(0.0..1.0).contains(&self.quantile) {
            return Err(InferustError::InvalidInput(
                "quantile must be greater than 0 and less than 1".into(),
            ));
        }
        if self.max_iter == 0 {
            return Err(InferustError::InvalidInput(
                "max_iter must be at least 1".into(),
            ));
        }
        if self.tolerance <= 0.0 || !self.tolerance.is_finite() {
            return Err(InferustError::InvalidInput(
                "tolerance must be positive and finite".into(),
            ));
        }
        if self.weight_epsilon <= 0.0 || !self.weight_epsilon.is_finite() {
            return Err(InferustError::InvalidInput(
                "weight_epsilon must be positive and finite".into(),
            ));
        }
        if x.len() != y.len() {
            return Err(InferustError::DimensionMismatch {
                x_rows: x.len(),
                y_len: y.len(),
            });
        }
        if y.len() < 3 {
            return Err(InferustError::InsufficientData {
                needed: 3,
                got: y.len(),
            });
        }
        if x.is_empty() {
            return Err(InferustError::InsufficientData { needed: 1, got: 0 });
        }
        let p = x[0].len();
        let ncols = if self.add_intercept { p + 1 } else { p };
        if ncols == 0 {
            return Err(InferustError::InvalidInput(
                "quantile regression needs at least one model term".into(),
            ));
        }
        if y.len() <= ncols {
            return Err(InferustError::InsufficientData {
                needed: ncols + 1,
                got: y.len(),
            });
        }
        if !self.feature_names.is_empty() && self.feature_names.len() != p {
            return Err(InferustError::InvalidInput(format!(
                "expected {p} feature names, got {}",
                self.feature_names.len()
            )));
        }
        for row in x {
            if row.len() != p {
                return Err(InferustError::InvalidInput(
                    "all rows in X must have the same length".into(),
                ));
            }
            if row.iter().any(|value| !value.is_finite()) {
                return Err(InferustError::InvalidInput(
                    "X values must be finite".into(),
                ));
            }
        }
        if y.iter().any(|value| !value.is_finite()) {
            return Err(InferustError::InvalidInput(
                "y values must be finite".into(),
            ));
        }
        Ok(())
    }
}

impl QuantileRegressionResult {
    /// Predict conditional quantiles for new rows.
    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<f64>> {
        let has_intercept = self
            .feature_names
            .first()
            .map(|name| name == "const")
            .unwrap_or(false);
        predict_with_intercept(x, &self.coefficients, has_intercept)
    }

    /// Two-sided confidence intervals for coefficients.
    pub fn confidence_intervals(&self, alpha: f64) -> Result<Vec<(f64, f64)>> {
        if !(0.0..1.0).contains(&alpha) {
            return Err(InferustError::InvalidInput(
                "alpha must be greater than 0 and less than 1".into(),
            ));
        }
        let normal = Normal::new(0.0, 1.0)
            .map_err(|_| InferustError::InvalidInput("invalid normal distribution".into()))?;
        let critical = normal.inverse_cdf(1.0 - alpha / 2.0);
        Ok(self
            .coefficients
            .iter()
            .zip(self.std_errors.iter())
            .map(|(coef, se)| (coef - critical * se, coef + critical * se))
            .collect())
    }

    /// Print a statsmodels-style quantile regression summary.
    pub fn print_summary(&self) {
        println!();
        println!("═══════════════════════════════════════════════════════════════════");
        println!("{:^67}", "Quantile Regression Results");
        println!("═══════════════════════════════════════════════════════════════════");
        println!(
            " Quantile     : {:.3}      Observations  : {}",
            self.quantile, self.n
        );
        println!(
            " Pseudo R¹    : {:.6}   Objective     : {:.6}",
            self.pseudo_r1, self.objective
        );
        println!(
            " Sparsity     : {:.6}   Iterations    : {}{}",
            self.sparsity,
            self.iterations,
            if self.converged {
                ""
            } else {
                " (not converged)"
            }
        );
        println!("───────────────────────────────────────────────────────────────────");
        println!(
            "{:<22} {:>11} {:>11} {:>9} {:>10}",
            "Variable", "Coef", "Std Err", "z", "P>|z|"
        );
        println!("───────────────────────────────────────────────────────────────────");
        for i in 0..self.feature_names.len() {
            println!(
                "{:<22} {:>11.6} {:>11.6} {:>9.4} {:>10.6}",
                self.feature_names[i],
                self.coefficients[i],
                self.std_errors[i],
                self.z_statistics[i],
                self.p_values[i],
            );
        }
        println!("═══════════════════════════════════════════════════════════════════");
        println!();
    }
}

fn design_matrix(x: &[Vec<f64>], add_intercept: bool) -> Result<DMatrix<f64>> {
    let p = x[0].len();
    let ncols = if add_intercept { p + 1 } else { p };
    let mut design = Vec::with_capacity(x.len() * ncols);
    for row in x {
        if add_intercept {
            design.push(1.0);
        }
        design.extend_from_slice(row);
    }
    Ok(DMatrix::from_row_slice(x.len(), ncols, &design))
}

fn xtx_inverse(x: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    let xtx = x.transpose() * x;
    if let Some(cholesky) = xtx.clone().cholesky() {
        return Ok(cholesky.inverse());
    }
    xtx.try_inverse().ok_or(InferustError::SingularMatrix)
}

fn predict_with_intercept(
    x: &[Vec<f64>],
    coefficients: &[f64],
    add_intercept: bool,
) -> Result<Vec<f64>> {
    let design = design_matrix(x, add_intercept)?;
    if design.ncols() != coefficients.len() {
        return Err(InferustError::InvalidInput(format!(
            "expected {} coefficients, got {}",
            design.ncols(),
            coefficients.len()
        )));
    }
    let beta = DVector::from_column_slice(coefficients);
    Ok((design * beta).iter().cloned().collect())
}

fn check_loss_sum(residuals: &[f64], quantile: f64) -> f64 {
    residuals
        .iter()
        .map(|residual| {
            if *residual >= 0.0 {
                quantile * residual
            } else {
                (quantile - 1.0) * residual
            }
        })
        .sum()
}

fn empirical_quantile(values: &[f64], quantile: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = quantile * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let weight = pos - lo as f64;
        sorted[lo] * (1.0 - weight) + sorted[hi] * weight
    }
}

fn residual_bandwidth(n: usize, quantile: f64) -> f64 {
    let base = n.max(2) as f64;
    let tail = (quantile * (1.0 - quantile)).sqrt().max(0.05);
    (base.powf(-1.0 / 3.0) * tail).clamp(0.02, 0.20)
}

#[cfg(test)]
mod tests {
    use super::QuantileRegression;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual} differed from expected {expected} by more than {tolerance}"
        );
    }

    #[test]
    fn median_regression_resists_high_outlier() {
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
        let y = vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 60.0];
        let result = QuantileRegression::new(0.5)
            .with_feature_names(vec!["x".into()])
            .fit(&x, &y)
            .unwrap();
        assert_close(result.coefficients[0], 1.0, 1e-3);
        assert_close(result.coefficients[1], 2.0, 1e-3);
        assert!(result.objective < 30.0);
        assert_eq!(result.feature_names, vec!["const", "x"]);
    }

    #[test]
    fn upper_quantile_moves_toward_large_tail() {
        let x = vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
            vec![6.0],
            vec![7.0],
        ];
        let y = vec![1.0, 2.0, 2.5, 4.0, 4.5, 6.0, 6.5, 12.0];
        let median = QuantileRegression::new(0.5).fit(&x, &y).unwrap();
        let upper = QuantileRegression::new(0.8).fit(&x, &y).unwrap();
        assert!(upper.predict(&[vec![7.0]]).unwrap()[0] > median.predict(&[vec![7.0]]).unwrap()[0]);
        assert_eq!(upper.confidence_intervals(0.05).unwrap().len(), 2);
    }

    #[test]
    fn rejects_invalid_quantile() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![1.0, 2.0, 3.0];
        assert!(QuantileRegression::new(1.0).fit(&x, &y).is_err());
    }
}
