use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ContinuousCDF, FisherSnedecor, StudentsT};

use crate::error::{InferustError, Result};

// ── Result type ───────────────────────────────────────────────────────────────

/// Full output of an OLS fit, mirroring `statsmodels.OLS().fit().summary()`.
#[derive(Debug, Clone)]
pub struct OlsResult {
    /// Estimated coefficients (intercept first if `add_intercept = true`).
    pub coefficients: Vec<f64>,
    /// Standard errors of each coefficient.
    pub std_errors: Vec<f64>,
    /// t-statistics: `coef / std_err`.
    pub t_statistics: Vec<f64>,
    /// Two-sided p-values for each coefficient.
    pub p_values: Vec<f64>,
    /// Coefficient of determination R².
    pub r_squared: f64,
    /// Adjusted R².
    pub adj_r_squared: f64,
    /// Joint F-statistic (model vs. null).
    pub f_statistic: f64,
    /// p-value of the F-statistic.
    pub f_p_value: f64,
    /// Akaike Information Criterion.
    pub aic: f64,
    /// Bayesian Information Criterion.
    pub bic: f64,
    /// Raw residuals y − ŷ.
    pub residuals: Vec<f64>,
    /// Number of observations.
    pub n: usize,
    /// Number of predictors (excluding intercept).
    pub k: usize,
    /// Names of each term in the model (e.g. `["const", "x1", "x2"]`).
    pub feature_names: Vec<String>,
}

impl OlsResult {
    /// Print a statsmodels-style summary table.
    pub fn print_summary(&self) {
        println!();
        println!("═══════════════════════════════════════════════════════════════════");
        println!("                     OLS Regression Results                        ");
        println!("═══════════════════════════════════════════════════════════════════");
        println!(
            " Dep. variable: y          Observations  : {}",
            self.n
        );
        println!(
            " R²           : {:.6}   Adj. R²       : {:.6}",
            self.r_squared, self.adj_r_squared
        );
        println!(
            " F-statistic  : {:.4}    F p-value     : {:.6}",
            self.f_statistic, self.f_p_value
        );
        println!(
            " AIC          : {:.4}    BIC           : {:.4}",
            self.aic, self.bic
        );
        println!("───────────────────────────────────────────────────────────────────");
        println!(
            "{:<22} {:>11} {:>11} {:>9} {:>10}",
            "Variable", "Coef", "Std Err", "t", "P>|t|"
        );
        println!("───────────────────────────────────────────────────────────────────");
        for i in 0..self.feature_names.len() {
            println!(
                "{:<22} {:>11.6} {:>11.6} {:>9.4} {:>10.6}  {}",
                self.feature_names[i],
                self.coefficients[i],
                self.std_errors[i],
                self.t_statistics[i],
                self.p_values[i],
                sig_stars(self.p_values[i]),
            );
        }
        println!("───────────────────────────────────────────────────────────────────");
        println!(" Significance codes:  *** p<0.001  ** p<0.01  * p<0.05  . p<0.1");
        println!("═══════════════════════════════════════════════════════════════════");
        println!();
    }

    /// Predicted values ŷ = X·β (without the intercept column — pass raw X rows).
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter()
            .map(|row| {
                let mut sum = if self.feature_names[0] == "const" {
                    self.coefficients[0]
                } else {
                    0.0
                };
                let offset = if self.feature_names[0] == "const" { 1 } else { 0 };
                for (j, &xi) in row.iter().enumerate() {
                    sum += self.coefficients[offset + j] * xi;
                }
                sum
            })
            .collect()
    }
}

fn sig_stars(p: f64) -> &'static str {
    if p < 0.001 {
        "***"
    } else if p < 0.01 {
        "**"
    } else if p < 0.05 {
        "*"
    } else if p < 0.1 {
        "."
    } else {
        ""
    }
}

// ── Builder ───────────────────────────────────────────────────────────────────

/// Ordinary Least Squares regression.
///
/// # Example
/// ```
/// use inferust::regression::Ols;
///
/// let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
/// let y = vec![2.1, 3.9, 6.2, 7.8, 10.1];
///
/// let result = Ols::new()
///     .with_feature_names(vec!["hours".to_string()])
///     .fit(&x, &y)
///     .unwrap();
///
/// result.print_summary();
/// ```
pub struct Ols {
    feature_names: Vec<String>,
    add_intercept: bool,
}

impl Default for Ols {
    fn default() -> Self {
        Self::new()
    }
}

impl Ols {
    /// Create a new OLS builder. An intercept term is added by default.
    pub fn new() -> Self {
        Self {
            feature_names: Vec::new(),
            add_intercept: true,
        }
    }

    /// Set human-readable names for the predictor columns.
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Fit without an intercept (forces regression through the origin).
    pub fn no_intercept(mut self) -> Self {
        self.add_intercept = false;
        self
    }

    /// Fit the model.
    ///
    /// * `x` – slice of rows; each row is one observation and must have the same length.
    /// * `y` – dependent variable, same length as `x`.
    pub fn fit(&self, x: &[Vec<f64>], y: &[f64]) -> Result<OlsResult> {
        let n = y.len();
        if n < 3 {
            return Err(InferustError::InsufficientData { needed: 3, got: n });
        }
        if x.len() != n {
            return Err(InferustError::DimensionMismatch {
                x_rows: x.len(),
                y_len: n,
            });
        }

        let p = x[0].len(); // original feature count
        let ncols = if self.add_intercept { p + 1 } else { p }; // design matrix cols

        // Build design matrix in row-major order.
        let mut design: Vec<f64> = Vec::with_capacity(n * ncols);
        for row in x {
            if self.add_intercept {
                design.push(1.0);
            }
            design.extend_from_slice(row);
        }

        let x_mat = DMatrix::from_row_slice(n, ncols, &design);
        let y_vec = DVector::from_column_slice(y);

        // β = (X'X)⁻¹ X'y  (normal equations)
        let xtx = x_mat.transpose() * &x_mat;
        let xtx_inv = xtx
            .try_inverse()
            .ok_or(InferustError::SingularMatrix)?;
        let xty = x_mat.transpose() * &y_vec;
        let beta = &xtx_inv * xty;

        // Residuals and sums of squares.
        let y_hat = &x_mat * &beta;
        let residuals: Vec<f64> = (0..n).map(|i| y[i] - y_hat[i]).collect();

        let k = if self.add_intercept { ncols - 1 } else { ncols }; // # predictors
        let df_resid = n - ncols; // residual degrees of freedom

        let y_mean = y.iter().sum::<f64>() / n as f64;
        let ssr: f64 = residuals.iter().map(|r| r * r).sum(); // residual SS
        let sst: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum(); // total SS
        let sse = sst - ssr; // explained SS

        let s2 = ssr / df_resid as f64; // mean squared error

        // R² and adjusted R²
        let r_squared = if sst == 0.0 { 1.0 } else { 1.0 - ssr / sst };
        let adj_r_squared =
            1.0 - (1.0 - r_squared) * (n - 1) as f64 / df_resid as f64;

        // Variance-covariance matrix of coefficients: Var(β) = s² (X'X)⁻¹
        let cov_beta = &xtx_inv * s2;
        let std_errors: Vec<f64> =
            (0..ncols).map(|i| cov_beta[(i, i)].sqrt()).collect();

        // t-statistics and p-values.
        let coefficients: Vec<f64> = beta.iter().cloned().collect();
        let t_statistics: Vec<f64> = coefficients
            .iter()
            .zip(std_errors.iter())
            .map(|(b, se)| b / se)
            .collect();

        let t_dist = StudentsT::new(0.0, 1.0, df_resid as f64)
            .map_err(|_| InferustError::InvalidInput("invalid degrees of freedom".into()))?;
        let p_values: Vec<f64> = t_statistics
            .iter()
            .map(|&t| 2.0 * (1.0 - t_dist.cdf(t.abs())))
            .collect();

        // F-statistic (joint significance of all predictors).
        let df_model = k as f64;
        let f_statistic = if df_model > 0.0 && s2 > 0.0 {
            (sse / df_model) / s2
        } else {
            f64::NAN
        };
        let f_p_value = if f_statistic.is_nan() {
            f64::NAN
        } else {
            let f_dist = FisherSnedecor::new(df_model, df_resid as f64)
                .map_err(|_| InferustError::InvalidInput("invalid F distribution parameters".into()))?;
            1.0 - f_dist.cdf(f_statistic)
        };

        // Information criteria (Akaike & Bayesian).
        let n_params = ncols as f64 + 1.0; // coefficients + sigma^2
        let log_lik = -0.5 * n as f64 * (2.0 * std::f64::consts::PI * s2).ln()
            - ssr / (2.0 * s2);
        let aic = -2.0 * log_lik + 2.0 * n_params;
        let bic = -2.0 * log_lik + n_params * (n as f64).ln();

        // Feature name labels.
        let mut feature_names: Vec<String> = Vec::with_capacity(ncols);
        if self.add_intercept {
            feature_names.push("const".to_string());
        }
        if self.feature_names.is_empty() {
            for i in 0..k {
                feature_names.push(format!("x{}", i + 1));
            }
        } else {
            feature_names.extend(self.feature_names.iter().cloned());
        }

        Ok(OlsResult {
            coefficients,
            std_errors,
            t_statistics,
            p_values,
            r_squared,
            adj_r_squared,
            f_statistic,
            f_p_value,
            aic,
            bic,
            residuals,
            n,
            k,
            feature_names,
        })
    }
}
