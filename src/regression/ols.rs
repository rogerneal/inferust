use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor, Normal, StudentsT};

use crate::error::{InferustError, Result};

// ── Result type ───────────────────────────────────────────────────────────────

/// Full output of an OLS fit, mirroring `statsmodels.OLS().fit().summary()`.
#[derive(Debug, Clone)]
pub struct OlsResult {
    /// Model name used in summaries, e.g. `OLS` or `WLS`.
    pub model_name: String,
    /// Estimated coefficients (intercept first if `add_intercept = true`).
    pub coefficients: Vec<f64>,
    /// Standard errors of each coefficient.
    pub std_errors: Vec<f64>,
    /// Full covariance matrix of the coefficient estimates (k × k row-major).
    ///
    /// Populated using whatever covariance estimator was selected on the builder
    /// (Nonrobust, HC0–HC3, HAC, or Cluster). Use this for Wald tests on
    /// linear restrictions via [`OlsResult::wald_test`].
    pub covariance_matrix: Vec<Vec<f64>>,
    /// Covariance estimator used for standard errors, test statistics, and intervals.
    pub covariance: OlsCovariance,
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
    /// Explained sum of squares.
    pub ess: f64,
    /// Residual sum of squares.
    pub ssr: f64,
    /// Centered total sum of squares.
    pub centered_tss: f64,
    /// Condition number of the design matrix.
    pub condition_number: f64,
    /// Fitted values ŷ.
    pub fitted_values: Vec<f64>,
    /// Raw residuals y − ŷ.
    pub residuals: Vec<f64>,
    /// Diagonal of the projection/hat matrix.
    pub leverage: Vec<f64>,
    /// Residual mean squared error.
    pub mse_resid: f64,
    /// Number of observations.
    pub n: usize,
    /// Number of predictors (excluding intercept).
    pub k: usize,
    /// Residual degrees of freedom.
    pub df_resid: usize,
    /// Names of each term in the model (e.g. `["const", "x1", "x2"]`).
    pub feature_names: Vec<String>,
}

/// Observation-level OLS influence diagnostics.
#[derive(Debug, Clone)]
pub struct OlsInfluence {
    /// Diagonal of the projection/hat matrix.
    pub leverage: Vec<f64>,
    /// Internally studentized residuals.
    pub studentized_residuals: Vec<f64>,
    /// Cook's distance for each observation.
    pub cooks_distance: Vec<f64>,
    /// Internal DFFITS values for each observation.
    pub dffits_internal: Vec<f64>,
}

/// Summary-level OLS diagnostic statistics.
#[derive(Debug, Clone)]
pub struct OlsDiagnostics {
    /// Durbin-Watson statistic for residual autocorrelation.
    pub durbin_watson: f64,
    /// Jarque-Bera normality test statistic.
    pub jarque_bera: f64,
    /// Jarque-Bera p-value using a chi-squared distribution with two degrees of freedom.
    pub jarque_bera_p_value: f64,
    /// Residual skewness.
    pub skewness: f64,
    /// Residual kurtosis, not excess kurtosis.
    pub kurtosis: f64,
    /// Design-matrix condition number.
    pub condition_number: f64,
}

impl OlsResult {
    /// Print a statsmodels-style summary table.
    pub fn print_summary(&self) {
        println!();
        println!("═══════════════════════════════════════════════════════════════════");
        println!("{:^67}", format!("{} Regression Results", self.model_name));
        println!("═══════════════════════════════════════════════════════════════════");
        println!(" Dep. variable: y          Observations  : {}", self.n);
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
        println!(" Covariance   : {}", self.covariance.label());
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
        if let Ok(diagnostics) = self.diagnostics() {
            println!(
                " Durbin-Watson: {:>8.4}   Jarque-Bera: {:>10.4}   Prob(JB): {:>9.6}",
                diagnostics.durbin_watson, diagnostics.jarque_bera, diagnostics.jarque_bera_p_value
            );
            println!(
                " Skew          : {:>8.4}   Kurtosis    : {:>10.4}   Cond. No.: {:>9.4}",
                diagnostics.skewness, diagnostics.kurtosis, diagnostics.condition_number
            );
            if diagnostics.condition_number > 30.0 {
                println!(
                    " Note: Large condition number may indicate multicollinearity or numerical instability."
                );
            }
            println!("───────────────────────────────────────────────────────────────────");
        }
        println!(" Significance codes:  *** p<0.001  ** p<0.01  * p<0.05  . p<0.1");
        println!("═══════════════════════════════════════════════════════════════════");
        println!();
    }

    /// Wald test of a linear restriction `R·β = q`.
    ///
    /// `r` has `r_rows × k` entries (one row per restriction) and `q` is the
    /// target vector of length `r_rows`. Returns the chi-square statistic, the
    /// finite-sample F equivalent, and both p-values.
    ///
    /// # Example
    /// ```
    /// use inferust::regression::Ols;
    /// let x = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 1.0],
    ///              vec![4.0, 3.0], vec![5.0, 5.0]];
    /// let y = vec![1.0, 2.5, 3.0, 4.0, 5.0];
    /// let result = Ols::new().fit(&x, &y).unwrap();
    /// // Test whether the two slopes are jointly zero (skipping the intercept).
    /// let r = vec![vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
    /// let q = vec![0.0, 0.0];
    /// let wald = result.wald_test(&r, &q).unwrap();
    /// assert_eq!(wald.df_num, 2);
    /// ```
    pub fn wald_test(
        &self,
        r: &[Vec<f64>],
        q: &[f64],
    ) -> Result<crate::hypothesis::WaldTestResult> {
        crate::hypothesis::wald_linear(
            &self.coefficients,
            &self.covariance_matrix,
            r,
            q,
            Some(self.df_resid),
        )
    }

    /// Two-sided confidence intervals for coefficients.
    pub fn confidence_intervals(&self, alpha: f64) -> Result<Vec<(f64, f64)>> {
        if !(0.0..1.0).contains(&alpha) {
            return Err(InferustError::InvalidInput(
                "alpha must be greater than 0 and less than 1".into(),
            ));
        }

        let critical = if self.covariance.uses_t_distribution() {
            StudentsT::new(0.0, 1.0, self.df_resid as f64)
                .map_err(|_| InferustError::InvalidInput("invalid degrees of freedom".into()))?
                .inverse_cdf(1.0 - alpha / 2.0)
        } else {
            Normal::new(0.0, 1.0)
                .map_err(|_| InferustError::InvalidInput("invalid normal distribution".into()))?
                .inverse_cdf(1.0 - alpha / 2.0)
        };

        Ok(self
            .coefficients
            .iter()
            .zip(self.std_errors.iter())
            .map(|(coef, se)| (coef - critical * se, coef + critical * se))
            .collect())
    }

    /// Summary-level model diagnostics.
    pub fn diagnostics(&self) -> Result<OlsDiagnostics> {
        let durbin_watson = durbin_watson(&self.residuals, self.ssr);
        let (jarque_bera, jarque_bera_p_value, skewness, kurtosis) = jarque_bera(&self.residuals)?;

        Ok(OlsDiagnostics {
            durbin_watson,
            jarque_bera,
            jarque_bera_p_value,
            skewness,
            kurtosis,
            condition_number: self.condition_number,
        })
    }

    /// Observation-level influence diagnostics.
    pub fn influence(&self) -> OlsInfluence {
        let p = self.coefficients.len() as f64;
        let mut studentized_residuals = Vec::with_capacity(self.n);
        let mut cooks_distance = Vec::with_capacity(self.n);
        let mut dffits_internal = Vec::with_capacity(self.n);

        for (&residual, &h) in self.residuals.iter().zip(self.leverage.iter()) {
            let denom = (1.0 - h).max(f64::EPSILON);
            let studentized = residual / (self.mse_resid * denom).sqrt();
            let leverage_ratio = h / denom;
            studentized_residuals.push(studentized);
            cooks_distance.push(studentized.powi(2) * leverage_ratio / p);
            dffits_internal.push(studentized * leverage_ratio.sqrt());
        }

        OlsInfluence {
            leverage: self.leverage.clone(),
            studentized_residuals,
            cooks_distance,
            dffits_internal,
        }
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
                let offset = if self.feature_names[0] == "const" {
                    1
                } else {
                    0
                };
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

/// Linear solver used for OLS coefficient estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OlsSolver {
    /// Fast path for full-rank, well-conditioned designs.
    Cholesky,
    /// More stable path for tougher or poorly conditioned designs.
    Svd,
}

/// Covariance estimator used for OLS coefficient inference.
///
/// # Choosing an estimator
/// - [`Nonrobust`](OlsCovariance::Nonrobust) — classical OLS; assumes homoskedastic, serially
///   uncorrelated errors.
/// - [`Hc0`](OlsCovariance::Hc0)–[`Hc3`](OlsCovariance::Hc3) — White's HC family; robust to
///   heteroskedasticity. `Hc1` is statsmodels' default for `HC1`.
/// - [`Hac`](OlsCovariance::Hac) — Newey-West HAC; robust to both heteroskedasticity *and*
///   serial autocorrelation. Choose `lags` using `floor(4 * (n/100)^(2/9))` or `floor(n^(1/3))`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OlsCovariance {
    /// Classical homoskedastic covariance estimator.
    Nonrobust,
    /// White's HC0 heteroskedasticity-consistent covariance estimator.
    Hc0,
    /// HC0 with small-sample correction n / df_resid.
    Hc1,
    /// HC0 adjusted by leverage: e_i^2 / (1 - h_i).
    Hc2,
    /// HC0 adjusted by squared leverage: e_i^2 / (1 - h_i)^2.
    Hc3,
    /// Newey-West heteroskedasticity and autocorrelation consistent (HAC) estimator.
    ///
    /// Uses the Bartlett kernel with the specified number of lags.
    /// Appropriate when residuals may be serially correlated (e.g. time series data).
    Hac { lags: usize },
    /// One-way cluster-robust sandwich covariance.
    Cluster { groups: Vec<usize> },
}

impl OlsCovariance {
    fn label(&self) -> &'static str {
        match self {
            OlsCovariance::Nonrobust => "nonrobust",
            OlsCovariance::Hc0 => "HC0",
            OlsCovariance::Hc1 => "HC1",
            OlsCovariance::Hc2 => "HC2",
            OlsCovariance::Hc3 => "HC3",
            OlsCovariance::Hac { .. } => "HAC (Newey-West)",
            OlsCovariance::Cluster { .. } => "cluster",
        }
    }

    fn uses_t_distribution(&self) -> bool {
        matches!(self, OlsCovariance::Nonrobust)
    }
}

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
    solver: OlsSolver,
    covariance: OlsCovariance,
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
            solver: OlsSolver::Cholesky,
            covariance: OlsCovariance::Nonrobust,
        }
    }

    /// Set human-readable names for the predictor columns.
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Use a specific linear solver for coefficient estimation.
    pub fn with_solver(mut self, solver: OlsSolver) -> Self {
        self.solver = solver;
        self
    }

    /// Use SVD decomposition for coefficient estimation.
    pub fn stable(mut self) -> Self {
        self.solver = OlsSolver::Svd;
        self
    }

    /// Use a specific covariance estimator for coefficient inference.
    pub fn with_covariance(mut self, covariance: OlsCovariance) -> Self {
        self.covariance = covariance;
        self
    }

    /// Use HC1 robust standard errors, matching statsmodels' common robust default.
    pub fn robust(mut self) -> Self {
        self.covariance = OlsCovariance::Hc1;
        self
    }

    /// Use Newey-West HAC standard errors with the given number of Bartlett lags.
    ///
    /// Appropriate for time series regressions where residuals may be serially
    /// correlated.  A common rule-of-thumb for `lags` is `floor(n^(1/3))`.
    ///
    /// # Example
    /// ```rust
    /// use inferust::regression::Ols;
    /// let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
    /// let y = vec![2.1, 3.9, 6.2, 7.8, 10.1];
    /// let result = Ols::new().hac(2).fit(&x, &y).unwrap();
    /// ```
    pub fn hac(mut self, lags: usize) -> Self {
        self.covariance = OlsCovariance::Hac { lags };
        self
    }

    /// Use one-way cluster-robust standard errors.
    pub fn cluster_robust(mut self, groups: Vec<usize>) -> Self {
        self.covariance = OlsCovariance::Cluster { groups };
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
        fit_linear_model(FitSpec {
            model_name: "OLS",
            x,
            y,
            weights: None,
            add_intercept: self.add_intercept,
            solver: self.solver,
            covariance: self.covariance.clone(),
            feature_names: &self.feature_names,
        })
    }
}

/// Weighted Least Squares regression.
pub struct Wls {
    feature_names: Vec<String>,
    add_intercept: bool,
    solver: OlsSolver,
    covariance: OlsCovariance,
}

impl Default for Wls {
    fn default() -> Self {
        Self::new()
    }
}

impl Wls {
    /// Create a new WLS builder. An intercept term is added by default.
    pub fn new() -> Self {
        Self {
            feature_names: Vec::new(),
            add_intercept: true,
            solver: OlsSolver::Cholesky,
            covariance: OlsCovariance::Nonrobust,
        }
    }

    /// Set human-readable names for the predictor columns.
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Use a specific linear solver for coefficient estimation.
    pub fn with_solver(mut self, solver: OlsSolver) -> Self {
        self.solver = solver;
        self
    }

    /// Use SVD decomposition for coefficient estimation.
    pub fn stable(mut self) -> Self {
        self.solver = OlsSolver::Svd;
        self
    }

    /// Use a specific covariance estimator for coefficient inference.
    pub fn with_covariance(mut self, covariance: OlsCovariance) -> Self {
        self.covariance = covariance;
        self
    }

    /// Use HC1 robust standard errors.
    pub fn robust(mut self) -> Self {
        self.covariance = OlsCovariance::Hc1;
        self
    }

    /// Use Newey-West HAC standard errors with the given number of Bartlett lags.
    pub fn hac(mut self, lags: usize) -> Self {
        self.covariance = OlsCovariance::Hac { lags };
        self
    }

    /// Use one-way cluster-robust standard errors.
    pub fn cluster_robust(mut self, groups: Vec<usize>) -> Self {
        self.covariance = OlsCovariance::Cluster { groups };
        self
    }

    /// Fit without an intercept (forces regression through the origin).
    pub fn no_intercept(mut self) -> Self {
        self.add_intercept = false;
        self
    }

    /// Fit the weighted model. Weights must be positive and match `y` length.
    pub fn fit(&self, x: &[Vec<f64>], y: &[f64], weights: &[f64]) -> Result<OlsResult> {
        fit_linear_model(FitSpec {
            model_name: "WLS",
            x,
            y,
            weights: Some(weights),
            add_intercept: self.add_intercept,
            solver: self.solver,
            covariance: self.covariance.clone(),
            feature_names: &self.feature_names,
        })
    }
}

struct FitSpec<'a> {
    model_name: &'a str,
    x: &'a [Vec<f64>],
    y: &'a [f64],
    weights: Option<&'a [f64]>,
    add_intercept: bool,
    solver: OlsSolver,
    covariance: OlsCovariance,
    feature_names: &'a [String],
}

fn fit_linear_model(spec: FitSpec<'_>) -> Result<OlsResult> {
    let FitSpec {
        model_name,
        x,
        y,
        weights,
        add_intercept,
        solver,
        covariance,
        feature_names: user_feature_names,
    } = spec;
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
    if let Some(weights) = weights {
        if weights.len() != n {
            return Err(InferustError::DimensionMismatch {
                x_rows: weights.len(),
                y_len: n,
            });
        }
        if let Some(weight) = weights.iter().find(|w| **w <= 0.0 || !w.is_finite()) {
            return Err(InferustError::InvalidInput(format!(
                "weights must be positive and finite, got {weight}"
            )));
        }
    }

    let p = x[0].len();
    let ncols = if add_intercept { p + 1 } else { p };
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
        if add_intercept {
            design.push(1.0);
        }
        design.extend_from_slice(row);
    }

    let x_raw = DMatrix::from_row_slice(n, ncols, &design);
    let y_raw = DVector::from_column_slice(y);
    let (x_fit, y_fit) = if let Some(weights) = weights {
        let mut weighted_design = Vec::with_capacity(n * ncols);
        let mut weighted_y = Vec::with_capacity(n);
        for i in 0..n {
            let sqrt_w = weights[i].sqrt();
            for j in 0..ncols {
                weighted_design.push(x_raw[(i, j)] * sqrt_w);
            }
            weighted_y.push(y[i] * sqrt_w);
        }
        (
            DMatrix::from_row_slice(n, ncols, &weighted_design),
            DVector::from_column_slice(&weighted_y),
        )
    } else {
        (x_raw.clone(), y_raw.clone())
    };

    let xtx = x_fit.transpose() * &x_fit;
    let xty = x_fit.transpose() * &y_fit;
    let cholesky = xtx
        .clone()
        .cholesky()
        .ok_or(InferustError::SingularMatrix)?;
    let beta = match solver {
        OlsSolver::Cholesky => cholesky.solve(&xty),
        OlsSolver::Svd => x_fit
            .clone()
            .svd(true, true)
            .solve(&y_fit, 1e-12)
            .map_err(|_| InferustError::SingularMatrix)?,
    };
    let xtx_inv = cholesky.inverse();

    let y_hat = &x_raw * &beta;
    let fitted_values: Vec<f64> = y_hat.iter().cloned().collect();
    let residuals: Vec<f64> = (0..n).map(|i| y[i] - y_hat[i]).collect();
    let leverage = leverage_values(&x_fit, &xtx_inv);

    let k = if add_intercept { ncols - 1 } else { ncols };
    let df_resid = n - ncols;
    let y_mean = weighted_mean(y, weights);
    let ssr = weighted_sum(weights, &residuals, |r| r * r);
    let sst = weighted_sum(weights, y, |yi| (yi - y_mean).powi(2));
    let sse = sst - ssr;
    let condition_number = condition_number(&x_fit);
    let s2 = ssr / df_resid as f64;
    let r_squared = if sst == 0.0 { 1.0 } else { 1.0 - ssr / sst };
    let adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) as f64 / df_resid as f64;
    if let OlsCovariance::Cluster { groups } = &covariance {
        if groups.len() != n {
            return Err(InferustError::DimensionMismatch {
                x_rows: groups.len(),
                y_len: n,
            });
        }
        let mut unique = groups.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() < 2 {
            return Err(InferustError::InvalidInput(
                "cluster-robust covariance needs at least two clusters".into(),
            ));
        }
    }

    let cov_beta = coefficient_covariance(
        covariance.clone(),
        &x_fit,
        &xtx_inv,
        &residuals,
        &leverage,
        s2,
        df_resid,
    );
    let std_errors: Vec<f64> = (0..ncols).map(|i| cov_beta[(i, i)].sqrt()).collect();
    let covariance_matrix: Vec<Vec<f64>> = (0..ncols)
        .map(|i| (0..ncols).map(|j| cov_beta[(i, j)]).collect())
        .collect();
    let coefficients: Vec<f64> = beta.iter().cloned().collect();
    let t_statistics: Vec<f64> = coefficients
        .iter()
        .zip(std_errors.iter())
        .map(|(b, se)| b / se)
        .collect();

    let p_values: Vec<f64> = if covariance.uses_t_distribution() {
        let t_dist = StudentsT::new(0.0, 1.0, df_resid as f64)
            .map_err(|_| InferustError::InvalidInput("invalid degrees of freedom".into()))?;
        t_statistics
            .iter()
            .map(|&t| 2.0 * (1.0 - t_dist.cdf(t.abs())))
            .collect()
    } else {
        let normal = Normal::new(0.0, 1.0)
            .map_err(|_| InferustError::InvalidInput("invalid normal distribution".into()))?;
        t_statistics
            .iter()
            .map(|&t| 2.0 * (1.0 - normal.cdf(t.abs())))
            .collect()
    };

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

    let n_params = ncols as f64;
    let sigma2_mle = ssr / n as f64;
    let weight_log_det = weights
        .map(|weights| 0.5 * weights.iter().map(|w| w.ln()).sum::<f64>())
        .unwrap_or(0.0);
    let log_lik =
        -0.5 * n as f64 * ((2.0 * std::f64::consts::PI * sigma2_mle).ln() + 1.0) + weight_log_det;
    let aic = -2.0 * log_lik + 2.0 * n_params;
    let bic = -2.0 * log_lik + n_params * (n as f64).ln();

    let mut feature_names = Vec::with_capacity(ncols);
    if add_intercept {
        feature_names.push("const".to_string());
    }
    if user_feature_names.is_empty() {
        for i in 0..k {
            feature_names.push(format!("x{}", i + 1));
        }
    } else {
        feature_names.extend(user_feature_names.iter().cloned());
    }

    Ok(OlsResult {
        model_name: model_name.to_string(),
        coefficients,
        std_errors,
        covariance_matrix,
        covariance,
        t_statistics,
        p_values,
        r_squared,
        adj_r_squared,
        f_statistic,
        f_p_value,
        aic,
        bic,
        ess: sse,
        ssr,
        centered_tss: sst,
        condition_number,
        fitted_values,
        residuals,
        leverage,
        mse_resid: s2,
        n,
        k,
        df_resid,
        feature_names,
    })
}

fn weighted_mean(y: &[f64], weights: Option<&[f64]>) -> f64 {
    if let Some(weights) = weights {
        let weighted_sum: f64 = y.iter().zip(weights.iter()).map(|(yi, wi)| yi * wi).sum();
        weighted_sum / weights.iter().sum::<f64>()
    } else {
        y.iter().sum::<f64>() / y.len() as f64
    }
}

fn weighted_sum<F>(weights: Option<&[f64]>, values: &[f64], f: F) -> f64
where
    F: Fn(f64) -> f64,
{
    if let Some(weights) = weights {
        values
            .iter()
            .zip(weights.iter())
            .map(|(value, weight)| weight * f(*value))
            .sum()
    } else {
        values.iter().map(|value| f(*value)).sum()
    }
}

fn condition_number(x_mat: &DMatrix<f64>) -> f64 {
    let svd = x_mat.clone().svd(false, false);
    let mut singular_values = svd.singular_values.iter().cloned().collect::<Vec<_>>();
    singular_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let smallest = singular_values[0];
    let largest = singular_values[singular_values.len() - 1];
    largest / smallest
}

fn durbin_watson(residuals: &[f64], ssr: f64) -> f64 {
    let diff_sq_sum: f64 = residuals.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
    diff_sq_sum / ssr
}

fn jarque_bera(residuals: &[f64]) -> Result<(f64, f64, f64, f64)> {
    let n = residuals.len();
    if n == 0 {
        return Err(InferustError::InsufficientData { needed: 1, got: 0 });
    }

    let mean = residuals.iter().sum::<f64>() / n as f64;
    let centered = residuals.iter().map(|r| r - mean).collect::<Vec<_>>();
    let m2 = centered.iter().map(|r| r.powi(2)).sum::<f64>() / n as f64;
    if m2 == 0.0 {
        return Ok((0.0, 1.0, 0.0, 3.0));
    }

    let m3 = centered.iter().map(|r| r.powi(3)).sum::<f64>() / n as f64;
    let m4 = centered.iter().map(|r| r.powi(4)).sum::<f64>() / n as f64;
    let skewness = m3 / m2.powf(1.5);
    let kurtosis = m4 / m2.powi(2);
    let jarque_bera = n as f64 / 6.0 * (skewness.powi(2) + 0.25 * (kurtosis - 3.0).powi(2));
    let chi2 = ChiSquared::new(2.0)
        .map_err(|_| InferustError::InvalidInput("invalid chi-squared df".into()))?;
    let p_value = 1.0 - chi2.cdf(jarque_bera);

    Ok((jarque_bera, p_value, skewness, kurtosis))
}

fn leverage_values(x_mat: &DMatrix<f64>, xtx_inv: &DMatrix<f64>) -> Vec<f64> {
    let n = x_mat.nrows();
    let ncols = x_mat.ncols();
    let mut leverage = Vec::with_capacity(n);

    for i in 0..n {
        let mut h = 0.0;
        for j in 0..ncols {
            for k in 0..ncols {
                h += x_mat[(i, j)] * xtx_inv[(j, k)] * x_mat[(i, k)];
            }
        }
        leverage.push(h);
    }

    leverage
}

fn coefficient_covariance(
    covariance: OlsCovariance,
    x_mat: &DMatrix<f64>,
    xtx_inv: &DMatrix<f64>,
    residuals: &[f64],
    leverage: &[f64],
    s2: f64,
    df_resid: usize,
) -> DMatrix<f64> {
    if covariance == OlsCovariance::Nonrobust {
        return xtx_inv * s2;
    }

    // Newey-West HAC estimator
    if let OlsCovariance::Hac { lags } = covariance {
        return newey_west_covariance(x_mat, xtx_inv, residuals, lags);
    }
    if let OlsCovariance::Cluster { groups } = covariance {
        return cluster_covariance(x_mat, xtx_inv, residuals, &groups, df_resid);
    }

    let n = x_mat.nrows();
    let ncols = x_mat.ncols();
    let mut meat = DMatrix::zeros(ncols, ncols);

    for i in 0..n {
        let denom = (1.0 - leverage[i]).max(f64::EPSILON);
        let weight = match covariance {
            OlsCovariance::Nonrobust
            | OlsCovariance::Hac { .. }
            | OlsCovariance::Cluster { .. } => unreachable!(),
            OlsCovariance::Hc0 | OlsCovariance::Hc1 => residuals[i].powi(2),
            OlsCovariance::Hc2 => residuals[i].powi(2) / denom,
            OlsCovariance::Hc3 => residuals[i].powi(2) / denom.powi(2),
        };

        for j in 0..ncols {
            for k in 0..ncols {
                meat[(j, k)] += weight * x_mat[(i, j)] * x_mat[(i, k)];
            }
        }
    }

    let scale = if covariance == OlsCovariance::Hc1 {
        n as f64 / df_resid as f64
    } else {
        1.0
    };

    xtx_inv * meat * xtx_inv * scale
}

/// Newey-West HAC covariance matrix using the Bartlett kernel.
///
/// Ω_HAC = Σ_{t} xₜxₜ'eₜ²  +  Σ_{l=1}^{L} wₗ Σ_{t>l} ( xₜxₜ₋ₗ'eₜeₜ₋ₗ + xₜ₋ₗxₜ'eₜ₋ₗeₜ )
/// where wₗ = 1 − l / (L + 1)  (Bartlett weights).
///
/// Returns (X'X)⁻¹ Ω_HAC (X'X)⁻¹.
fn newey_west_covariance(
    x_mat: &DMatrix<f64>,
    xtx_inv: &DMatrix<f64>,
    residuals: &[f64],
    lags: usize,
) -> DMatrix<f64> {
    let n = x_mat.nrows();
    let k = x_mat.ncols();

    // Lag-0 (heteroskedasticity) term: Σ xₜxₜ'eₜ²
    let mut meat = DMatrix::zeros(k, k);
    for i in 0..n {
        let e2 = residuals[i].powi(2);
        for j in 0..k {
            for l in 0..k {
                meat[(j, l)] += e2 * x_mat[(i, j)] * x_mat[(i, l)];
            }
        }
    }

    // Lag-l cross terms with Bartlett weights
    for lag in 1..=lags.min(n - 1) {
        let w = 1.0 - lag as f64 / (lags as f64 + 1.0); // Bartlett weight
        let mut gamma = DMatrix::zeros(k, k);
        for t in lag..n {
            let et = residuals[t];
            let et_lag = residuals[t - lag];
            for j in 0..k {
                for l in 0..k {
                    gamma[(j, l)] += x_mat[(t, j)] * x_mat[(t - lag, l)] * et * et_lag;
                }
            }
        }
        // Add symmetric contribution: γ(l) + γ(l)'
        let gamma_t = gamma.transpose();
        meat += (&gamma + &gamma_t) * w;
    }

    xtx_inv * meat * xtx_inv
}

/// One-way cluster-robust covariance matrix.
fn cluster_covariance(
    x_mat: &DMatrix<f64>,
    xtx_inv: &DMatrix<f64>,
    residuals: &[f64],
    groups: &[usize],
    df_resid: usize,
) -> DMatrix<f64> {
    let n = x_mat.nrows();
    let k = x_mat.ncols();
    if groups.len() != n {
        return xtx_inv.clone() * f64::NAN;
    }
    let mut unique = groups.to_vec();
    unique.sort_unstable();
    unique.dedup();
    let g = unique.len();
    if g < 2 {
        return xtx_inv.clone() * f64::NAN;
    }
    let mut meat = DMatrix::zeros(k, k);
    for cluster in unique {
        let mut score = DVector::<f64>::zeros(k);
        for i in 0..n {
            if groups[i] == cluster {
                for j in 0..k {
                    score[j] += x_mat[(i, j)] * residuals[i];
                }
            }
        }
        meat += &score * score.transpose();
    }
    let correction = (g as f64 / (g - 1) as f64) * ((n - 1) as f64 / df_resid.max(1) as f64);
    xtx_inv * meat * xtx_inv * correction
}

#[cfg(test)]
mod tests {
    use super::{Ols, OlsCovariance, OlsSolver, Wls};

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual} differed from expected {expected} by more than {tolerance}"
        );
    }

    fn fixture() -> (Vec<Vec<f64>>, Vec<f64>) {
        (
            vec![
                vec![1.0, 2.0],
                vec![2.0, 1.0],
                vec![3.0, 4.0],
                vec![4.0, 3.0],
                vec![5.0, 5.0],
                vec![6.0, 7.0],
            ],
            vec![5.1, 5.9, 10.2, 10.8, 14.9, 19.1],
        )
    }

    #[test]
    fn matches_statsmodels_reference_values() {
        let (x, y) = fixture();
        let result = Ols::new().fit(&x, &y).unwrap();

        let expected_coefficients = [1.1666007905138316, 1.656126482213441, 1.100988142292489];
        let expected_std_errors = [
            0.33848997525229785,
            0.19115143770783555,
            0.16554200102490418,
        ];
        let expected_t_statistics = [3.4464854967840353, 8.663949913600645, 6.650808468401055];
        let expected_p_values = [
            0.04104155628322375,
            0.0032350213527919183,
            0.006927626115340223,
        ];

        for (actual, expected) in result.coefficients.iter().zip(expected_coefficients) {
            assert_close(*actual, expected, 1e-10);
        }
        for (actual, expected) in result.std_errors.iter().zip(expected_std_errors) {
            assert_close(*actual, expected, 1e-10);
        }
        for (actual, expected) in result.t_statistics.iter().zip(expected_t_statistics) {
            assert_close(*actual, expected, 1e-10);
        }
        for (actual, expected) in result.p_values.iter().zip(expected_p_values) {
            assert_close(*actual, expected, 1e-10);
        }

        assert_close(result.r_squared, 0.9972162326394675, 1e-12);
        assert_close(result.adj_r_squared, 0.9953603877324457, 1e-12);
        assert_close(result.f_statistic, 537.3381303935711, 1e-9);
        assert_close(result.f_p_value, 0.00014687551678395586, 1e-12);
        assert_close(result.aic, 6.721473225061304, 1e-10);
        assert_close(result.bic, 6.09675163274547, 1e-10);
    }

    #[test]
    fn confidence_intervals_match_statsmodels() {
        let (x, y) = fixture();
        let result = Ols::new().fit(&x, &y).unwrap();
        let intervals = result.confidence_intervals(0.05).unwrap();
        let expected = [
            (0.08937461939658253, 2.243826961631081),
            (1.0477972955304709, 2.264455668896411),
            (0.574159612761511, 1.627816671823467),
        ];

        for ((actual_low, actual_high), (expected_low, expected_high)) in
            intervals.iter().zip(expected)
        {
            assert_close(*actual_low, expected_low, 1e-10);
            assert_close(*actual_high, expected_high, 1e-10);
        }
    }

    #[test]
    fn robust_covariances_match_statsmodels() {
        let (x, y) = fixture();
        let cases = [
            (
                OlsCovariance::Hc0,
                [0.2104634179103986, 0.1225009315340049, 0.12893704138913634],
                [5.543009811854776, 13.519297049212385, 8.538959250427267],
                [
                    2.9731619484270437e-08,
                    1.20319992226485e-41,
                    1.3543612455477758e-17,
                ],
                [
                    (0.7541000713462482, 1.579101509681415),
                    (1.4160290683341843, 1.8962238960926976),
                    (0.8482761848966315, 1.3537000996883464),
                ],
            ),
            (
                OlsCovariance::Hc1,
                [0.29764021999228224, 0.1732424787787277, 0.1823445126247777],
                [3.919499826146081, 9.55958662037336, 6.037955990252719],
                [
                    8.87329252740221e-05,
                    1.1822798501555386e-21,
                    1.5607861694316155e-09,
                ],
                [
                    (0.5832366789783798, 1.7499649020492833),
                    (1.31657746321469, 1.9956755012121918),
                    (0.7435994647694155, 1.4583768198155624),
                ],
            ),
            (
                OlsCovariance::Hc2,
                [0.3193726854832724, 0.16772958709112912, 0.18509469987520621],
                [3.6527882425152916, 9.873788583964327, 5.948242402590634],
                [
                    0.00025940814824629765,
                    5.408246874584226e-23,
                    2.7103701101289776e-09,
                ],
                [
                    (0.5406418293207795, 1.7925597517068836),
                    (1.3273825323730535, 1.9848704320538284),
                    (0.7382091968078344, 1.4637670877771436),
                ],
            ),
            (
                OlsCovariance::Hc3,
                [
                    0.49474018253216606,
                    0.23476125457757344,
                    0.27128659349565304,
                ],
                [2.3580069533527, 7.05451368111597, 4.058394954596718],
                [
                    0.018373348800993426,
                    1.7320541088733153e-12,
                    4.941115052038062e-05,
                ],
                [
                    (0.19692785104601374, 2.1362737299816494),
                    (1.1960028782759582, 2.1162500861509237),
                    (0.569276189552451, 1.632700095032527),
                ],
            ),
        ];

        for (covariance, expected_se, expected_t, expected_p, expected_ci) in cases {
            let result = Ols::new().with_covariance(covariance).fit(&x, &y).unwrap();

            for (actual, expected) in result.std_errors.iter().zip(expected_se) {
                assert_close(*actual, expected, 1e-10);
            }
            for (actual, expected) in result.t_statistics.iter().zip(expected_t) {
                assert_close(*actual, expected, 1e-10);
            }
            for (actual, expected) in result.p_values.iter().zip(expected_p) {
                assert_close(*actual, expected, 1e-10);
            }

            let intervals = result.confidence_intervals(0.05).unwrap();
            for ((actual_low, actual_high), (expected_low, expected_high)) in
                intervals.iter().zip(expected_ci)
            {
                assert_close(*actual_low, expected_low, 1e-10);
                assert_close(*actual_high, expected_high, 1e-10);
            }
        }
    }

    #[test]
    fn cluster_robust_covariance_runs() {
        let (x, y) = fixture();
        let clusters = vec![0, 0, 1, 1, 2, 2];
        let result = Ols::new().cluster_robust(clusters).fit(&x, &y).unwrap();
        assert_eq!(result.covariance.label(), "cluster");
        assert_eq!(result.std_errors.len(), result.coefficients.len());
        assert!(result.std_errors.iter().all(|se| se.is_finite()));
    }

    #[test]
    fn summary_diagnostics_match_statsmodels() {
        let (x, y) = fixture();
        let result = Ols::new().fit(&x, &y).unwrap();
        let diagnostics = result.diagnostics().unwrap();

        assert_close(result.ess, 141.92381422924905, 1e-10);
        assert_close(result.ssr, 0.39618577075099115, 1e-10);
        assert_close(result.centered_tss, 142.32000000000005, 1e-10);
        assert_close(diagnostics.durbin_watson, 1.6956152054529454, 1e-10);
        assert_close(diagnostics.jarque_bera, 0.5954268251017766, 1e-10);
        assert_close(diagnostics.jarque_bera_p_value, 0.7425141044815998, 1e-10);
        assert_close(diagnostics.skewness, -0.06821777483113872, 1e-10);
        assert_close(diagnostics.kurtosis, 1.4627646109995196, 1e-10);
        assert_close(diagnostics.condition_number, 13.513151689263205, 1e-10);
    }

    #[test]
    fn influence_diagnostics_match_statsmodels() {
        let (x, y) = fixture();
        let result = Ols::new().fit(&x, &y).unwrap();
        let influence = result.influence();

        let expected_fitted = [
            5.02470355731225,
            5.5798418972332025,
            10.53893280632411,
            11.094071146245064,
            14.952173913043481,
            18.8102766798419,
        ];
        let expected_leverage = [
            0.6936758893280631,
            0.5573122529644272,
            0.33003952569169914,
            0.47035573122529617,
            0.304347826086957,
            0.6442687747035574,
        ];
        let expected_studentized = [
            0.3743648445836767,
            1.3241187566278978,
            -1.1394629471824138,
            -1.111914686008104,
            -0.17213473976427088,
            1.3366990824065803,
        ];
        let expected_cooks = [
            0.1057899181460565,
            0.7357558271225819,
            0.21320428705631478,
            0.3659854676773617,
            0.004321095425749859,
            1.0786763082065736,
        ];
        let expected_dffits = [
            0.5633557973769059,
            1.4856875449998719,
            -0.7997580016285828,
            -1.047834148628534,
            -0.11385642835277057,
            1.7988965853043695,
        ];

        for (actual, expected) in result.fitted_values.iter().zip(expected_fitted) {
            assert_close(*actual, expected, 1e-10);
        }
        for (actual, expected) in influence.leverage.iter().zip(expected_leverage) {
            assert_close(*actual, expected, 1e-10);
        }
        for (actual, expected) in influence
            .studentized_residuals
            .iter()
            .zip(expected_studentized)
        {
            assert_close(*actual, expected, 1e-10);
        }
        for (actual, expected) in influence.cooks_distance.iter().zip(expected_cooks) {
            assert_close(*actual, expected, 1e-10);
        }
        for (actual, expected) in influence.dffits_internal.iter().zip(expected_dffits) {
            assert_close(*actual, expected, 1e-10);
        }
    }

    #[test]
    fn wls_matches_statsmodels_reference_values() {
        let (x, y) = fixture();
        let weights = [1.0, 0.8, 1.2, 1.5, 0.9, 1.1];
        let result = Wls::new().fit(&x, &y, &weights).unwrap();

        assert_eq!(result.model_name, "WLS");
        let expected_coefficients = [1.0910621653414276, 1.6265313140792843, 1.139502728692733];
        let expected_std_errors = [0.3575122066358346, 0.1860769817677046, 0.1624788732811118];
        let expected_t_statistics = [3.0518179382132096, 8.741174209875238, 7.0132362791636895];
        let expected_p_values = [
            0.05534861802672169,
            0.0031525997571736127,
            0.005953985223013776,
        ];
        let expected_fitted = [
            4.996598936806178,
            5.483627522192729,
            10.528667022350213,
            11.015695607736763,
            14.921232379201513,
            18.826769150666266,
        ];
        let expected_residuals = [
            0.10340106319382159,
            0.41637247780727105,
            -0.3286670223502135,
            -0.21569560773676244,
            -0.02123237920151233,
            0.27323084933373565,
        ];

        for (actual, expected) in result.coefficients.iter().zip(expected_coefficients) {
            assert_close(*actual, expected, 1e-10);
        }
        for (actual, expected) in result.std_errors.iter().zip(expected_std_errors) {
            assert_close(*actual, expected, 1e-10);
        }
        for (actual, expected) in result.t_statistics.iter().zip(expected_t_statistics) {
            assert_close(*actual, expected, 1e-10);
        }
        for (actual, expected) in result.p_values.iter().zip(expected_p_values) {
            assert_close(*actual, expected, 1e-10);
        }
        for (actual, expected) in result.fitted_values.iter().zip(expected_fitted) {
            assert_close(*actual, expected, 1e-10);
        }
        for (actual, expected) in result.residuals.iter().zip(expected_residuals) {
            assert_close(*actual, expected, 1e-10);
        }

        assert_close(result.r_squared, 0.9969644673039065, 1e-12);
        assert_close(result.adj_r_squared, 0.9949407788398442, 1e-12);
        assert_close(result.f_statistic, 492.6472058365241, 1e-9);
        assert_close(result.f_p_value, 0.00016724470338265856, 1e-12);
        assert_close(result.aic, 6.876742008019683, 1e-10);
        assert_close(result.bic, 6.252020415703848, 1e-10);
        assert_close(result.ssr, 0.4313242580462716, 1e-10);
        assert_close(result.ess, 141.6604603573384, 1e-10);
        assert_close(result.centered_tss, 142.09178461538465, 1e-10);
        assert_close(result.condition_number, 14.263637999744354, 1e-10);

        let intervals = result.confidence_intervals(0.05).unwrap();
        let expected_intervals = [
            (-0.04670123576060958, 2.228825566443465),
            (1.034351310954309, 2.2187113172042596),
            (0.6224224387325991, 1.6565830186528672),
        ];
        for ((actual_low, actual_high), (expected_low, expected_high)) in
            intervals.iter().zip(expected_intervals)
        {
            assert_close(*actual_low, expected_low, 1e-10);
            assert_close(*actual_high, expected_high, 1e-10);
        }
    }

    #[test]
    fn cholesky_and_svd_solvers_agree() {
        let (x, y) = fixture();
        let fast = Ols::new()
            .with_solver(OlsSolver::Cholesky)
            .fit(&x, &y)
            .unwrap();
        let stable = Ols::new().stable().fit(&x, &y).unwrap();

        for (actual, expected) in fast.coefficients.iter().zip(stable.coefficients.iter()) {
            assert_close(*actual, *expected, 1e-10);
        }
        assert_close(fast.r_squared, stable.r_squared, 1e-12);
        assert_close(fast.aic, stable.aic, 1e-10);
    }
}
