use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ContinuousCDF, Normal};

use crate::error::{InferustError, Result};

/// Binary logistic regression result.
#[derive(Debug, Clone)]
pub struct LogisticResult {
    pub coefficients: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub z_statistics: Vec<f64>,
    pub p_values: Vec<f64>,
    pub log_likelihood: f64,
    pub null_log_likelihood: f64,
    pub pseudo_r_squared: f64,
    pub aic: f64,
    pub bic: f64,
    pub n: usize,
    pub k: usize,
    pub feature_names: Vec<String>,
    iterations: usize,
}

impl LogisticResult {
    /// Predict probabilities for raw X rows, without the intercept column.
    pub fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter()
            .map(|row| {
                let mut eta = if self.feature_names[0] == "const" {
                    self.coefficients[0]
                } else {
                    0.0
                };
                let offset = if self.feature_names[0] == "const" {
                    1
                } else {
                    0
                };
                for (j, &value) in row.iter().enumerate() {
                    eta += self.coefficients[offset + j] * value;
                }
                sigmoid(eta)
            })
            .collect()
    }

    /// Number of Newton/IRLS iterations used to fit the model.
    pub fn iterations(&self) -> usize {
        self.iterations
    }
}

/// Binary logistic regression builder.
pub struct Logistic {
    feature_names: Vec<String>,
    add_intercept: bool,
    max_iter: usize,
    tolerance: f64,
}

impl Default for Logistic {
    fn default() -> Self {
        Self::new()
    }
}

impl Logistic {
    /// Create a new logistic regression builder. An intercept term is added by default.
    pub fn new() -> Self {
        Self {
            feature_names: Vec::new(),
            add_intercept: true,
            max_iter: 100,
            tolerance: 1e-8,
        }
    }

    /// Set human-readable names for predictor columns.
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Fit without an intercept.
    pub fn no_intercept(mut self) -> Self {
        self.add_intercept = false;
        self
    }

    /// Set maximum Newton iterations.
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance on the largest coefficient update.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Fit a binary logistic regression model.
    pub fn fit(&self, x: &[Vec<f64>], y: &[f64]) -> Result<LogisticResult> {
        let n = y.len();
        if n < 2 {
            return Err(InferustError::InsufficientData { needed: 2, got: n });
        }
        if x.len() != n {
            return Err(InferustError::DimensionMismatch {
                x_rows: x.len(),
                y_len: n,
            });
        }
        if let Some(value) = y.iter().find(|value| **value != 0.0 && **value != 1.0) {
            return Err(InferustError::InvalidInput(format!(
                "logistic regression requires binary 0/1 y values, got {value}"
            )));
        }

        let p = x[0].len();
        let ncols = if self.add_intercept { p + 1 } else { p };
        if n <= ncols {
            return Err(InferustError::InsufficientData {
                needed: ncols + 1,
                got: n,
            });
        }

        let mut design = Vec::with_capacity(n * ncols);
        for row in x {
            if row.len() != p {
                return Err(InferustError::InvalidInput(
                    "all rows in X must have the same length".into(),
                ));
            }
            if self.add_intercept {
                design.push(1.0);
            }
            design.extend_from_slice(row);
        }

        let x_mat = DMatrix::from_row_slice(n, ncols, &design);
        let y_vec = DVector::from_column_slice(y);
        let mut beta = DVector::zeros(ncols);
        let mut hessian = DMatrix::zeros(ncols, ncols);
        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..self.max_iter {
            iterations = iter + 1;
            let eta = &x_mat * &beta;
            let mu = eta.map(sigmoid);
            let gradient = x_mat.transpose() * (&y_vec - &mu);
            hessian.fill(0.0);
            for i in 0..n {
                let w = (mu[i] * (1.0 - mu[i])).max(1e-12);
                for j in 0..ncols {
                    for k in 0..ncols {
                        hessian[(j, k)] += w * x_mat[(i, j)] * x_mat[(i, k)];
                    }
                }
            }

            let delta = hessian
                .clone()
                .lu()
                .solve(&gradient)
                .ok_or(InferustError::SingularMatrix)?;
            let max_delta = delta.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            beta += delta;
            if max_delta < self.tolerance {
                converged = true;
                break;
            }
        }

        if !converged {
            return Err(InferustError::InvalidInput(format!(
                "logistic regression failed to converge in {} iterations",
                self.max_iter
            )));
        }

        let eta = &x_mat * &beta;
        let probabilities = eta.map(sigmoid);
        hessian.fill(0.0);
        for i in 0..n {
            let w = (probabilities[i] * (1.0 - probabilities[i])).max(1e-12);
            for j in 0..ncols {
                for k in 0..ncols {
                    hessian[(j, k)] += w * x_mat[(i, j)] * x_mat[(i, k)];
                }
            }
        }

        let cov = hessian.try_inverse().ok_or(InferustError::SingularMatrix)?;
        let coefficients: Vec<f64> = beta.iter().cloned().collect();
        let std_errors: Vec<f64> = (0..ncols).map(|i| cov[(i, i)].sqrt()).collect();
        let z_statistics: Vec<f64> = coefficients
            .iter()
            .zip(std_errors.iter())
            .map(|(coef, se)| coef / se)
            .collect();
        let normal = Normal::new(0.0, 1.0)
            .map_err(|_| InferustError::InvalidInput("invalid normal distribution".into()))?;
        let p_values = z_statistics
            .iter()
            .map(|z| 2.0 * (1.0 - normal.cdf(z.abs())))
            .collect::<Vec<_>>();

        let log_likelihood = binary_log_likelihood(y, probabilities.as_slice());
        let y_mean = y.iter().sum::<f64>() / n as f64;
        let null_probs = vec![y_mean; n];
        let null_log_likelihood = binary_log_likelihood(y, &null_probs);
        let pseudo_r_squared = 1.0 - log_likelihood / null_log_likelihood;
        let n_params = ncols as f64;
        let aic = -2.0 * log_likelihood + 2.0 * n_params;
        let bic = -2.0 * log_likelihood + n_params * (n as f64).ln();

        let mut feature_names = Vec::with_capacity(ncols);
        if self.add_intercept {
            feature_names.push("const".to_string());
        }
        if self.feature_names.is_empty() {
            for i in 0..(ncols - usize::from(self.add_intercept)) {
                feature_names.push(format!("x{}", i + 1));
            }
        } else {
            feature_names.extend(self.feature_names.iter().cloned());
        }

        Ok(LogisticResult {
            coefficients,
            std_errors,
            z_statistics,
            p_values,
            log_likelihood,
            null_log_likelihood,
            pseudo_r_squared,
            aic,
            bic,
            n,
            k: ncols - usize::from(self.add_intercept),
            feature_names,
            iterations,
        })
    }
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
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

#[cfg(test)]
mod tests {
    use super::Logistic;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual} differed from expected {expected} by more than {tolerance}"
        );
    }

    fn fixture() -> (Vec<Vec<f64>>, Vec<f64>) {
        (
            vec![
                vec![0.2, 1.0],
                vec![1.1, 0.9],
                vec![1.8, 1.5],
                vec![2.4, 1.9],
                vec![3.0, 2.5],
                vec![3.7, 2.9],
                vec![4.1, 3.4],
                vec![4.8, 3.8],
                vec![5.2, 4.2],
                vec![5.9, 4.8],
                vec![2.2, 3.6],
                vec![4.6, 1.2],
            ],
            vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
        )
    }

    #[test]
    fn logistic_matches_statsmodels_reference_values() {
        let (x, y) = fixture();
        let result = Logistic::new().fit(&x, &y).unwrap();

        let expected_coefficients = [-1.7689272112231273, 5.075491023368961, -5.333429371506223];
        let expected_std_errors = [2.0497905638092466, 9.672670992764024, 12.013832575525887];
        let expected_z = [
            -0.8629794879803845,
            0.5247248693939718,
            -0.44394071067473323,
        ];
        let expected_p = [0.38814874483671213, 0.5997744663371969, 0.6570854500881999];
        let expected_pred = [
            0.0022664797317858963,
            0.27170811607160383,
            0.3467881502714788,
            0.5692361727365304,
            0.5309534735979055,
            0.8239648815691457,
            0.7123707405174255,
            0.911039847279562,
            0.9023156243097266,
            0.9293014980956025,
            5.5271786442116135e-05,
            0.999999744032795,
        ];

        for (actual, expected) in result.coefficients.iter().zip(expected_coefficients) {
            assert_close(*actual, expected, 1e-8);
        }
        for (actual, expected) in result.std_errors.iter().zip(expected_std_errors) {
            assert_close(*actual, expected, 1e-7);
        }
        for (actual, expected) in result.z_statistics.iter().zip(expected_z) {
            assert_close(*actual, expected, 1e-8);
        }
        for (actual, expected) in result.p_values.iter().zip(expected_p) {
            assert_close(*actual, expected, 1e-8);
        }
        for (actual, expected) in result.predict_proba(&x).iter().zip(expected_pred) {
            assert_close(*actual, expected, 1e-8);
        }

        assert_close(result.log_likelihood, -4.562687231980299, 1e-8);
        assert_close(result.null_log_likelihood, -8.150319193022398, 1e-8);
        assert_close(result.pseudo_r_squared, 0.4401830009447385, 1e-8);
        assert_close(result.aic, 15.125374463960599, 1e-8);
        assert_close(result.bic, 16.580094413324602, 1e-8);
    }
}
