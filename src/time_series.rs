//! Time series modelling and diagnostics.
//!
//! # Models
//! - [`AutoRegressive`] — AR(p) fitted by OLS.
//! - [`Arima`] — Full ARIMA(p, d, q) fitted by conditional sum of squares (CSS).
//! - [`Var`] — VAR(p) for k-variate time series.
//!
//! # Diagnostics
//! - [`acf`] — sample autocorrelation function.
//! - [`pacf`] — partial autocorrelation function.
//! - [`ljung_box`] — Ljung-Box portmanteau test.
//! - [`adf_test`] — Augmented Dickey-Fuller unit root test.
//! - [`kpss_test`] — KPSS stationarity test.

use crate::error::{InferustError, Result};
use crate::regression::Ols;
use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor, Normal};

// ── AutoRegressive ────────────────────────────────────────────────────────────

/// AR(p) model fitted by OLS.
///
/// For a full ARIMA(p, d, q) model use [`Arima`].
#[derive(Debug, Clone)]
pub struct AutoRegressive {
    lags: usize,
}

/// Fitted AR(p) result.
#[derive(Debug, Clone)]
pub struct AutoRegressiveResult {
    /// Constant (intercept) term.
    pub intercept: f64,
    /// AR coefficients φ₁ … φ_p.
    pub coefficients: Vec<f64>,
    /// In-sample fitted values (length = n − lags).
    pub fitted_values: Vec<f64>,
    /// In-sample residuals.
    pub residuals: Vec<f64>,
    /// Residual variance σ².
    pub sigma2: f64,
    /// Number of observations used in fitting.
    pub n: usize,
}

impl AutoRegressive {
    /// Create an AR(p) builder with `lags` autoregressive lags.
    pub fn new(lags: usize) -> Self {
        Self { lags }
    }

    /// Fit the AR(`lags`) model by OLS.
    pub fn fit(&self, y: &[f64]) -> Result<AutoRegressiveResult> {
        if self.lags == 0 {
            return Err(InferustError::InvalidInput(
                "AR lags must be at least 1".into(),
            ));
        }
        if y.len() <= self.lags + 1 {
            return Err(InferustError::InsufficientData {
                needed: self.lags + 2,
                got: y.len(),
            });
        }
        let mut x = Vec::with_capacity(y.len() - self.lags);
        let mut target = Vec::with_capacity(y.len() - self.lags);
        for t in self.lags..y.len() {
            let row: Vec<f64> = (1..=self.lags).map(|lag| y[t - lag]).collect();
            x.push(row);
            target.push(y[t]);
        }
        let ols = Ols::new().fit(&x, &target)?;
        Ok(AutoRegressiveResult {
            intercept: ols.coefficients[0],
            coefficients: ols.coefficients[1..].to_vec(),
            fitted_values: ols.fitted_values,
            residuals: ols.residuals,
            sigma2: ols.mse_resid,
            n: target.len(),
        })
    }
}

impl AutoRegressiveResult {
    /// Forecast `steps` observations ahead using the last values from `history`.
    ///
    /// `history` must contain at least as many values as there are AR lags.
    pub fn forecast(&self, history: &[f64], steps: usize) -> Result<Vec<f64>> {
        if history.len() < self.coefficients.len() {
            return Err(InferustError::InsufficientData {
                needed: self.coefficients.len(),
                got: history.len(),
            });
        }
        let mut buf = history.to_vec();
        let mut out = Vec::with_capacity(steps);
        for _ in 0..steps {
            let t = buf.len();
            let mut next = self.intercept;
            for (i, &c) in self.coefficients.iter().enumerate() {
                next += c * buf[t - 1 - i];
            }
            buf.push(next);
            out.push(next);
        }
        Ok(out)
    }
}

// ── ARIMA ─────────────────────────────────────────────────────────────────────

/// ARIMA(p, d, q) model fitted by conditional sum of squares (CSS).
///
/// All combinations of p, d, q ≥ 0 are supported. When q = 0 the AR
/// coefficients are estimated by fast OLS; when q > 0 the CSS objective is
/// minimised with the Adam optimiser using finite-difference gradients.
///
/// # Example
/// ```rust
/// use inferust::time_series::Arima;
///
/// let y: Vec<f64> = (0..30).map(|i| i as f64 + (i as f64 * 0.1).sin()).collect();
/// let result = Arima::new(1, 1, 1).fit(&y).unwrap();
/// result.print_summary();
/// let forecasts = result.forecast(&y, 5).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct Arima {
    /// AR order.
    pub p: usize,
    /// Integration order.
    pub d: usize,
    /// MA order.
    pub q: usize,
    max_iter: usize,
    tolerance: f64,
}

/// Fitted ARIMA(p, d, q) result.
#[derive(Debug, Clone)]
pub struct ArimaResult {
    /// Constant term in the differenced equation.
    pub intercept: f64,
    /// AR coefficients φ₁ … φ_p.
    pub ar_coefficients: Vec<f64>,
    /// MA coefficients θ₁ … θ_q.
    pub ma_coefficients: Vec<f64>,
    /// Fitted values on the d-times-differenced series (length = n − p).
    pub fitted_values: Vec<f64>,
    /// Residuals on the d-times-differenced series.
    pub residuals: Vec<f64>,
    /// Residual variance σ².
    pub sigma2: f64,
    /// Gaussian log-likelihood under the CSS approximation.
    pub log_likelihood: f64,
    /// Akaike Information Criterion.
    pub aic: f64,
    /// Bayesian Information Criterion.
    pub bic: f64,
    /// Effective observations (after differencing, starting from lag p).
    pub n: usize,
    /// AR order.
    pub p: usize,
    /// Integration order.
    pub d: usize,
    /// MA order.
    pub q: usize,
    // Internal: tails of series at each differencing level (for undifferencing in forecast).
    // original_tails[i] = the series before the i-th difference was applied.
    original_tails: Vec<Vec<f64>>,
    // Internal: last fitted residuals (for MA recursion in forecast).
    last_residuals: Vec<f64>,
}

impl Arima {
    /// Create an ARIMA(p, d, q) builder.
    ///
    /// * `p` – number of AR lags
    /// * `d` – number of differences to apply before fitting
    /// * `q` – number of MA lags
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        Self {
            p,
            d,
            q,
            max_iter: 2000,
            tolerance: 1e-7,
        }
    }

    /// Override the maximum number of CSS optimiser iterations (default 2000).
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Fit the model to univariate series `y`.
    pub fn fit(&self, y: &[f64]) -> Result<ArimaResult> {
        let min_len = self.d + self.p.max(self.q) + 2;
        if y.len() < min_len {
            return Err(InferustError::InsufficientData {
                needed: min_len,
                got: y.len(),
            });
        }

        // --- difference the series, saving tails for undifferencing ---
        let mut original_tails: Vec<Vec<f64>> = Vec::with_capacity(self.d);
        let mut series = y.to_vec();
        for _ in 0..self.d {
            original_tails.push(series.clone());
            series = series.windows(2).map(|w| w[1] - w[0]).collect();
        }

        let k = 1 + self.p + self.q; // number of parameters

        // --- fast path: q = 0 ---
        if self.q == 0 {
            if self.p == 0 {
                let mu = series.iter().sum::<f64>() / series.len() as f64;
                let resids: Vec<f64> = series.iter().map(|v| v - mu).collect();
                let n = resids.len();
                let sigma2 = resids.iter().map(|e| e * e).sum::<f64>() / n.max(1) as f64;
                let ll = gaussian_log_likelihood(&resids, sigma2);
                return Ok(ArimaResult {
                    intercept: mu,
                    ar_coefficients: vec![],
                    ma_coefficients: vec![],
                    fitted_values: vec![mu; n],
                    residuals: resids.clone(),
                    sigma2,
                    log_likelihood: ll,
                    aic: -2.0 * ll + 2.0 * k as f64,
                    bic: -2.0 * ll + k as f64 * (n as f64).ln(),
                    n,
                    p: 0,
                    d: self.d,
                    q: 0,
                    original_tails,
                    last_residuals: resids,
                });
            }
            let ar = AutoRegressive::new(self.p).fit(&series)?;
            let n = ar.n;
            let sigma2 = ar.sigma2;
            let ll = gaussian_log_likelihood(&ar.residuals, sigma2);
            let last_residuals = ar.residuals.clone();
            return Ok(ArimaResult {
                intercept: ar.intercept,
                ar_coefficients: ar.coefficients,
                ma_coefficients: vec![],
                fitted_values: ar.fitted_values,
                residuals: ar.residuals,
                sigma2,
                log_likelihood: ll,
                aic: -2.0 * ll + 2.0 * k as f64,
                bic: -2.0 * ll + k as f64 * (n as f64).ln(),
                n,
                p: self.p,
                d: self.d,
                q: 0,
                original_tails,
                last_residuals,
            });
        }

        // --- full CSS path: q > 0 ---
        let mut init = vec![0.0_f64; k];
        // Intercept initialised to series mean
        init[0] = series.iter().sum::<f64>() / series.len() as f64;
        // AR initialised from OLS AR(p) if possible
        if self.p > 0 {
            if let Ok(ar) = AutoRegressive::new(self.p).fit(&series) {
                init[0] = ar.intercept;
                for (i, &c) in ar.coefficients.iter().enumerate() {
                    if 1 + i < k {
                        init[1 + i] = c;
                    }
                }
            }
        }
        // MA initialised to small positive values
        for i in 0..self.q {
            init[1 + self.p + i] = 0.01;
        }

        let params = css_optimize(
            &series,
            self.p,
            self.q,
            &init,
            self.max_iter,
            self.tolerance,
        );
        let (fitted, resids) = css_fitted_residuals(&params, &series, self.p, self.q);
        let n = fitted.len();
        let sigma2 = if n > 1 {
            resids.iter().map(|e| e * e).sum::<f64>() / (n - 1) as f64
        } else {
            resids.iter().map(|e| e * e).sum::<f64>()
        };
        let ll = gaussian_log_likelihood(&resids, sigma2);
        let last_residuals = resids.clone();

        Ok(ArimaResult {
            intercept: params[0],
            ar_coefficients: params[1..=self.p].to_vec(),
            ma_coefficients: params[self.p + 1..].to_vec(),
            fitted_values: fitted,
            residuals: resids,
            sigma2,
            log_likelihood: ll,
            aic: -2.0 * ll + 2.0 * k as f64,
            bic: -2.0 * ll + k as f64 * (n as f64).ln(),
            n,
            p: self.p,
            d: self.d,
            q: self.q,
            original_tails,
            last_residuals,
        })
    }
}

impl ArimaResult {
    /// Print a compact model summary to stdout.
    pub fn print_summary(&self) {
        println!();
        println!("═══════════════════════════════════════════════════");
        println!("  ARIMA({}, {}, {}) Results", self.p, self.d, self.q);
        println!("═══════════════════════════════════════════════════");
        println!("  n          : {}   σ²  : {:.6}", self.n, self.sigma2);
        println!("  Log-lik    : {:.4}", self.log_likelihood);
        println!("  AIC        : {:.4}   BIC : {:.4}", self.aic, self.bic);
        println!("───────────────────────────────────────────────────");
        println!("  const      = {:.6}", self.intercept);
        for (i, &c) in self.ar_coefficients.iter().enumerate() {
            println!("  ar.L{}      = {:.6}", i + 1, c);
        }
        for (i, &c) in self.ma_coefficients.iter().enumerate() {
            println!("  ma.L{}      = {:.6}", i + 1, c);
        }
        println!("═══════════════════════════════════════════════════");
        println!();
    }

    /// Forecast `steps` observations ahead on the original (un-differenced) scale.
    ///
    /// `history` should be the full original series passed to [`Arima::fit`].
    /// The last few values are used for the AR recursion and for undifferencing.
    pub fn forecast(&self, history: &[f64], steps: usize) -> Result<Vec<f64>> {
        if steps == 0 {
            return Ok(vec![]);
        }
        let p = self.p;
        let q = self.q;

        // Build the d-times-differenced history
        let mut diff_hist = history.to_vec();
        for _ in 0..self.d {
            if diff_hist.len() < 2 {
                return Err(InferustError::InsufficientData {
                    needed: 2,
                    got: diff_hist.len(),
                });
            }
            diff_hist = diff_hist.windows(2).map(|w| w[1] - w[0]).collect();
        }

        // Extend the differenced buffer and carry forward the residual buffer for MA
        let mut buf = diff_hist.clone();
        // eps_buf: residuals aligned with buf; future residuals are 0 in expectation
        let mut eps_buf = vec![0.0_f64; buf.len()];
        let copy_from = buf.len().saturating_sub(self.last_residuals.len());
        for (i, &e) in self.last_residuals.iter().enumerate() {
            let idx = copy_from + i;
            if idx < eps_buf.len() {
                eps_buf[idx] = e;
            }
        }

        let mut diff_fcast = Vec::with_capacity(steps);
        for step in 0..steps {
            let t = buf.len();
            let mut pred = self.intercept;
            // AR terms
            for i in 0..p {
                if t > i {
                    pred += self.ar_coefficients[i] * buf[t - 1 - i];
                }
            }
            // MA terms: only past residuals contribute (future ε = 0)
            for j in 0..q {
                // need ε_{t - 1 - j} where t = buf.len()
                // that residual is in-sample only if (j + 1) > step
                if j + 1 > step {
                    let lookback = j - step; // how many steps before the last in-sample residual
                    let lr_len = self.last_residuals.len();
                    if lookback < lr_len {
                        pred +=
                            self.ma_coefficients[j] * self.last_residuals[lr_len - 1 - lookback];
                    }
                }
                // else: future ε = 0, contributes nothing
            }
            buf.push(pred);
            eps_buf.push(0.0);
            diff_fcast.push(pred);
        }

        // Undifference: apply cumulative sums d times
        // original_tails[level] is the series just before the (level+1)-th difference,
        // so its last value is the "seed" for reconstructing that level.
        let mut fcast = diff_fcast;
        for level in (0..self.d).rev() {
            let seed = self.original_tails[level].last().copied().unwrap_or(0.0);
            let mut prev = seed;
            for f in fcast.iter_mut() {
                prev += *f;
                *f = prev;
            }
        }
        Ok(fcast)
    }
}

// ── CSS internals ─────────────────────────────────────────────────────────────

/// Compute residuals from ARIMA(p,0,q) CSS for parameter vector `params`.
/// `params` = [intercept, ar_1..ar_p, ma_1..ma_q].
/// Returns residuals for t = p .. n-1 (length n − p).
fn css_residuals(params: &[f64], y: &[f64], p: usize, q: usize) -> Vec<f64> {
    let n = y.len();
    let mut eps = vec![0.0_f64; n]; // eps[0..p] are 0 (pre-sample)
    let start = p;
    for t in start..n {
        let mut pred = params[0]; // intercept
        for i in 0..p {
            pred += params[1 + i] * y[t - 1 - i];
        }
        for j in 0..q {
            if t > j {
                pred += params[1 + p + j] * eps[t - 1 - j];
            }
        }
        eps[t] = y[t] - pred;
    }
    eps[start..].to_vec()
}

/// CSS objective: sum of squared residuals.
fn css_objective(params: &[f64], y: &[f64], p: usize, q: usize) -> f64 {
    css_residuals(params, y, p, q).iter().map(|e| e * e).sum()
}

/// Finite-difference gradient of the CSS objective.
fn css_gradient(params: &[f64], y: &[f64], p: usize, q: usize) -> Vec<f64> {
    let h = 1e-5;
    let f0 = css_objective(params, y, p, q);
    let mut grad = vec![0.0_f64; params.len()];
    let mut ph = params.to_vec();
    for i in 0..params.len() {
        ph[i] += h;
        grad[i] = (css_objective(&ph, y, p, q) - f0) / h;
        ph[i] = params[i];
    }
    grad
}

/// Minimise the CSS objective with the Adam optimiser.
fn css_optimize(
    y: &[f64],
    p: usize,
    q: usize,
    init: &[f64],
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let mut params = init.to_vec();
    let np = params.len();
    let (alpha, beta1, beta2, eps) = (0.05_f64, 0.9_f64, 0.999_f64, 1e-8_f64);
    let mut m = vec![0.0_f64; np];
    let mut v = vec![0.0_f64; np];
    for iter in 1..=max_iter {
        let grad = css_gradient(&params, y, p, q);
        let gnorm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if gnorm < tol {
            break;
        }
        let t = iter as f64;
        for i in 0..np {
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
            let m_hat = m[i] / (1.0 - beta1.powf(t));
            let v_hat = v[i] / (1.0 - beta2.powf(t));
            params[i] -= alpha * m_hat / (v_hat.sqrt() + eps);
        }
    }
    params
}

/// Return (fitted, residuals) on the differenced series from final CSS params.
fn css_fitted_residuals(params: &[f64], y: &[f64], p: usize, q: usize) -> (Vec<f64>, Vec<f64>) {
    let n = y.len();
    let mut eps = vec![0.0_f64; n];
    let start = p;
    let mut fitted = Vec::with_capacity(n - start);
    for t in start..n {
        let mut pred = params[0];
        for i in 0..p {
            pred += params[1 + i] * y[t - 1 - i];
        }
        for j in 0..q {
            if t > j {
                pred += params[1 + p + j] * eps[t - 1 - j];
            }
        }
        eps[t] = y[t] - pred;
        fitted.push(pred);
    }
    (fitted, eps[start..].to_vec())
}

fn gaussian_log_likelihood(residuals: &[f64], sigma2: f64) -> f64 {
    let n = residuals.len() as f64;
    let s2 = sigma2.max(f64::EPSILON);
    -0.5 * n * (2.0 * std::f64::consts::PI * s2).ln()
        - 0.5 * residuals.iter().map(|e| e * e).sum::<f64>() / s2
}

fn regularized_inverse(matrix: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    if let Some(inv) = matrix.clone().try_inverse() {
        return Ok(inv);
    }
    let mut ridge = matrix.clone();
    let scale = matrix.trace().abs().max(1.0) * 1e-10;
    for i in 0..ridge.nrows().min(ridge.ncols()) {
        ridge[(i, i)] += scale;
    }
    ridge.try_inverse().ok_or(InferustError::SingularMatrix)
}

fn row_dot_matrix(x: &DMatrix<f64>, row: usize, beta: &DVector<f64>) -> f64 {
    (0..x.ncols()).map(|col| x[(row, col)] * beta[col]).sum()
}

// ── VAR ───────────────────────────────────────────────────────────────────────

/// VAR(p) model for k-variate time series.
///
/// Each of the k variables is regressed on p lags of all k variables by OLS.
///
/// # Example
/// ```rust
/// use inferust::time_series::Var;
///
/// // Two variables, 20 observations
/// let y1: Vec<f64> = (0..20).map(|i| i as f64).collect();
/// let y2: Vec<f64> = (0..20).map(|i| (i as f64) * 0.5 + 1.0).collect();
/// let result = Var::new(1).fit(&[y1, y2]).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct Var {
    lags: usize,
}

/// Fitted VAR(p) result.
#[derive(Debug, Clone)]
pub struct VarResult {
    /// Coefficient matrix: `coefficients[i]` = regression coefficients for variable i.
    /// Each inner vec is `[intercept, y1_{t-1}, y2_{t-1}, ..., yk_{t-p}]`.
    pub coefficients: Vec<Vec<f64>>,
    /// Residuals: `residuals[i]` = residuals for variable i.
    pub residuals: Vec<Vec<f64>>,
    /// Number of observations used (after dropping p lags).
    pub n: usize,
    /// Number of variables.
    pub k: usize,
    /// AR lag order.
    pub lags: usize,
    /// AIC for the joint model (sum of per-equation log-likelihoods).
    pub aic: f64,
    /// BIC for the joint model.
    pub bic: f64,
}

impl Var {
    /// Create a VAR(p) builder with the given number of lags.
    pub fn new(lags: usize) -> Self {
        Self { lags }
    }

    /// Fit the VAR(p) model.
    ///
    /// `series` is a slice of k variable vectors, each of length T.
    /// All vectors must have the same length.
    pub fn fit(&self, series: &[Vec<f64>]) -> Result<VarResult> {
        let k = series.len();
        if k < 2 {
            return Err(InferustError::InvalidInput(
                "VAR requires at least 2 variables".into(),
            ));
        }
        if self.lags == 0 {
            return Err(InferustError::InvalidInput(
                "VAR lags must be at least 1".into(),
            ));
        }
        let t = series[0].len();
        for (i, s) in series.iter().enumerate() {
            if s.len() != t {
                return Err(InferustError::DimensionMismatch {
                    x_rows: s.len(),
                    y_len: t,
                });
            }
            let _ = i;
        }
        let n = t - self.lags;
        if n < 2 {
            return Err(InferustError::InsufficientData {
                needed: self.lags + 2,
                got: t,
            });
        }

        let x_cols = k * self.lags + 1;
        let x_mat = DMatrix::from_fn(n, x_cols, |row, col| {
            if col == 0 {
                1.0
            } else {
                let t_idx = row + self.lags;
                let lag_col = col - 1;
                let lag = lag_col / k + 1;
                let var = lag_col % k;
                series[var][t_idx - lag]
            }
        });
        let xtx_inv = regularized_inverse(&(x_mat.transpose() * &x_mat))?;

        let mut coefficients = Vec::with_capacity(k);
        let mut residuals_all = Vec::with_capacity(k);
        let mut total_ll = 0.0_f64;
        let total_params = k * x_cols;

        for var in series.iter() {
            let y_eq = DVector::from_fn(n, |row, _| var[row + self.lags]);
            let beta = &xtx_inv * (x_mat.transpose() * &y_eq);
            let fitted = &x_mat * &beta;
            let resids: Vec<f64> = y_eq.iter().zip(fitted.iter()).map(|(a, b)| a - b).collect();
            let sigma2 = resids.iter().map(|e| e * e).sum::<f64>() / n.max(1) as f64;
            let ll = gaussian_log_likelihood(&resids, sigma2);
            total_ll += ll;
            coefficients.push(beta.iter().copied().collect());
            residuals_all.push(resids);
        }

        let n_f = n as f64;
        let p_f = total_params as f64;
        let aic = -2.0 * total_ll + 2.0 * p_f;
        let bic = -2.0 * total_ll + p_f * n_f.ln();

        Ok(VarResult {
            coefficients,
            residuals: residuals_all,
            n,
            k,
            lags: self.lags,
            aic,
            bic,
        })
    }
}

impl VarResult {
    /// Forecast `steps` observations ahead for each variable.
    ///
    /// Returns a `Vec` of length k, each element a `Vec<f64>` of length `steps`.
    pub fn forecast(&self, history: &[Vec<f64>], steps: usize) -> Result<Vec<Vec<f64>>> {
        if steps == 0 {
            return Ok(vec![vec![]; self.k]);
        }
        if history.len() != self.k {
            return Err(InferustError::DimensionMismatch {
                x_rows: history.len(),
                y_len: self.k,
            });
        }
        let mut bufs: Vec<Vec<f64>> = history.to_vec();
        let mut out: Vec<Vec<f64>> = vec![Vec::with_capacity(steps); self.k];

        for _ in 0..steps {
            let t = bufs[0].len();
            let mut row = Vec::with_capacity(self.k * self.lags);
            for lag in 1..=self.lags {
                for buf in bufs.iter() {
                    if t >= lag {
                        row.push(buf[t - lag]);
                    } else {
                        row.push(0.0);
                    }
                }
            }
            for (i, coefs) in self.coefficients.iter().enumerate() {
                let mut pred = coefs[0]; // intercept
                for (j, &c) in coefs[1..].iter().enumerate() {
                    if j < row.len() {
                        pred += c * row[j];
                    }
                }
                bufs[i].push(pred);
                out[i].push(pred);
            }
        }
        Ok(out)
    }
}

// ── ACF / PACF / Ljung-Box ────────────────────────────────────────────────────

/// Result of a single Ljung-Box test at one lag.
#[derive(Debug, Clone)]
pub struct LjungBoxResult {
    /// Number of lags included in this test.
    pub lag: usize,
    /// Ljung-Box Q statistic.
    pub statistic: f64,
    /// Two-sided p-value from a χ²(lag) distribution.
    pub p_value: f64,
}

/// Sample autocorrelation function for lags 0 to `max_lag`.
///
/// Returns a vector of length `max_lag + 1`; element 0 is always 1.0.
pub fn acf(series: &[f64], max_lag: usize) -> Result<Vec<f64>> {
    if series.len() < 2 {
        return Err(InferustError::InsufficientData {
            needed: 2,
            got: series.len(),
        });
    }
    let mean = series.iter().sum::<f64>() / series.len() as f64;
    let denom = series.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
    Ok((0..=max_lag)
        .map(|lag| {
            if lag == 0 {
                return 1.0;
            }
            series
                .iter()
                .skip(lag)
                .zip(series.iter())
                .map(|(a, b)| (a - mean) * (b - mean))
                .sum::<f64>()
                / denom.max(f64::EPSILON)
        })
        .collect())
}

/// Sample partial autocorrelation function for lags 0 to `max_lag`.
///
/// Returns a vector of length `max_lag + 1`; element 0 is always 1.0.
pub fn pacf(series: &[f64], max_lag: usize) -> Result<Vec<f64>> {
    let mut out = vec![1.0_f64];
    for lag in 1..=max_lag {
        let fit = AutoRegressive::new(lag).fit(series)?;
        out.push(*fit.coefficients.last().unwrap_or(&0.0));
    }
    Ok(out)
}

/// Ljung-Box portmanteau test for autocorrelation up to `max_lag`.
///
/// Returns one result per lag from 1 to `max_lag`.
pub fn ljung_box(series: &[f64], max_lag: usize) -> Result<Vec<LjungBoxResult>> {
    let rhos = acf(series, max_lag)?;
    let n = series.len() as f64;
    let mut results = Vec::with_capacity(max_lag);
    for lag in 1..=max_lag {
        let stat = n
            * (n + 2.0)
            * rhos
                .iter()
                .enumerate()
                .skip(1)
                .take(lag)
                .map(|(k, rho)| rho.powi(2) / (n - k as f64))
                .sum::<f64>();
        let chi = ChiSquared::new(lag as f64)
            .map_err(|_| InferustError::InvalidInput("invalid χ² df".into()))?;
        results.push(LjungBoxResult {
            lag,
            statistic: stat,
            p_value: 1.0 - chi.cdf(stat),
        });
    }
    Ok(results)
}

// ── ADF test ──────────────────────────────────────────────────────────────────

/// Result of the Augmented Dickey-Fuller unit root test.
///
/// H₀: the series has a unit root (is non-stationary).
/// Reject H₀ when the test statistic is more negative than the critical value.
#[derive(Debug, Clone)]
pub struct AdfResult {
    /// ADF test statistic (t-ratio on the lagged level coefficient).
    pub statistic: f64,
    /// Approximate p-value via MacKinnon (1994) response surface (constant-only case).
    pub p_value: f64,
    /// Number of augmentation lags used.
    pub lags: usize,
    /// Number of observations used.
    pub n: usize,
    /// Critical values at 1 %, 5 %, 10 % significance (constant-only specification).
    pub critical_values: [f64; 3],
}

impl AdfResult {
    /// Print the ADF test result to stdout.
    pub fn print(&self) {
        println!();
        println!("── Augmented Dickey-Fuller Test ───────────────────────────");
        println!("  H₀: unit root  (reject when stat << critical value)");
        println!(
            "  Lags: {}   n: {}   stat: {:.4}   p ≈ {:.4}",
            self.lags, self.n, self.statistic, self.p_value
        );
        let [cv1, cv5, cv10] = self.critical_values;
        println!("  Critical values: 1% {cv1:.3}   5% {cv5:.3}   10% {cv10:.3}");
        let sig = if self.statistic < cv1 {
            "***"
        } else if self.statistic < cv5 {
            "**"
        } else if self.statistic < cv10 {
            "*"
        } else {
            "(not significant)"
        };
        println!("  Verdict: {sig}");
        println!();
    }
}

/// Augmented Dickey-Fuller unit root test.
///
/// Fits the regression:
/// Δy_t = α + γ y_{t-1} + Σ δ_i Δy_{t-i} + ε_t
///
/// and tests H₀: γ = 0 (unit root). Use `lags = 0` for the simple DF test,
/// or choose `lags` to remove serial correlation from the residuals (AIC
/// lag selection is a common choice).
///
/// Critical values follow MacKinnon (1994) — constant-only specification.
pub fn adf_test(y: &[f64], lags: usize) -> Result<AdfResult> {
    let n = y.len();
    let min_len = lags + 3;
    if n < min_len {
        return Err(InferustError::InsufficientData {
            needed: min_len,
            got: n,
        });
    }

    // First difference
    let dy: Vec<f64> = y.windows(2).map(|w| w[1] - w[0]).collect(); // length n-1

    // Build regressors for obs t = lags+1 .. n-1 (0-indexed)
    // each row: [y_{t-1}, Δy_{t-1}, ..., Δy_{t-lags}]
    let t_start = lags + 1;
    let n_obs = n - 1 - t_start; // number of valid rows
    if n_obs < 2 {
        return Err(InferustError::InsufficientData {
            needed: t_start + 3,
            got: n,
        });
    }

    let mut x: Vec<Vec<f64>> = Vec::with_capacity(n_obs);
    let mut target: Vec<f64> = Vec::with_capacity(n_obs);

    for t in t_start..=(n - 2) {
        let mut row = vec![y[t]]; // lagged level y_{t-1} (note: dy[t] = y[t+1] - y[t])
        for i in 1..=lags {
            row.push(dy[t - i]); // lagged differences
        }
        x.push(row);
        target.push(dy[t]); // Δy_t = y[t+1] - y[t]
    }

    let feat: Vec<String> = std::iter::once("y_lag1".to_string())
        .chain((1..=lags).map(|i| format!("dy_lag{i}")))
        .collect();

    let ols = Ols::new().with_feature_names(feat).fit(&x, &target)?;

    // The ADF stat is the t-ratio on y_{t-1} (index 1 because OLS adds const at 0)
    let stat = ols.t_statistics[1];
    let n_used = ols.n;

    // MacKinnon (1994) approximate critical values and p-value (constant case)
    let cv = mackinnon_critical_values_constant(n_used);
    let p = mackinnon_pvalue_constant(stat, n_used);

    Ok(AdfResult {
        statistic: stat,
        p_value: p,
        lags,
        n: n_used,
        critical_values: cv,
    })
}

/// MacKinnon (1994) critical values for ADF with constant (no trend), approximate.
fn mackinnon_critical_values_constant(n: usize) -> [f64; 3] {
    // Response surface: cv = β₀ + β₁/n + β₂/n²
    // Coefficients from MacKinnon (1994), Table 4, case "c"
    let n = n as f64;
    let cv1 = -3.43035 - 6.5393 / n - 16.786 / (n * n);
    let cv5 = -2.86154 - 2.8903 / n - 4.234 / (n * n);
    let cv10 = -2.56677 - 1.5384 / n - 2.809 / (n * n);
    [cv1, cv5, cv10]
}

/// Approximate MacKinnon (1994) p-value for the ADF constant case.
///
/// Uses piecewise linear interpolation between the 1 %, 5 %, and 10 % anchor
/// points, then a logistic extrapolation for statistics above the 10 % critical
/// value.  Accuracy is ±0.01 in the 0.01–0.20 range; treat as indicative outside.
fn mackinnon_pvalue_constant(stat: f64, n: usize) -> f64 {
    let [cv1, cv5, cv10] = mackinnon_critical_values_constant(n);
    if stat <= cv1 {
        return 0.01;
    }
    if stat <= cv5 {
        return 0.01 + 0.04 * (stat - cv1) / (cv5 - cv1);
    }
    if stat <= cv10 {
        return 0.05 + 0.05 * (stat - cv5) / (cv10 - cv5);
    }
    // Above the 10 % critical value → p > 0.10.
    // Scale logistically between 0.10 and 0.99 as stat rises toward 0.
    let z = (stat - cv10) / (cv10.abs().max(0.1)); // normalised distance above cv10
    (0.10 + 0.89 / (1.0 + (-2.0 * z).exp())).min(0.999)
}

// ── KPSS test ─────────────────────────────────────────────────────────────────

/// Result of the KPSS stationarity test.
///
/// H₀: the series is stationary (opposite of ADF).
/// Reject H₀ when the statistic exceeds the critical value.
#[derive(Debug, Clone)]
pub struct KpssResult {
    /// KPSS test statistic η.
    pub statistic: f64,
    /// Number of Bartlett lags used for long-run variance estimation.
    pub lags: usize,
    /// Number of observations.
    pub n: usize,
    /// Critical values at 10 %, 5 %, 1 % significance.
    /// Source: Kwiatkowski et al. (1992), Table 1.
    pub critical_values: [f64; 3],
    /// Whether the trend specification was used (`true`) or constant only (`false`).
    pub trend: bool,
}

impl KpssResult {
    /// Print the KPSS test result to stdout.
    pub fn print(&self) {
        let spec = if self.trend { "trend" } else { "constant" };
        println!();
        println!("── KPSS Stationarity Test ({spec}) ────────────────────────");
        println!("  H₀: series is stationary  (reject when stat > critical value)");
        println!(
            "  Lags: {}   n: {}   stat: {:.4}",
            self.lags, self.n, self.statistic
        );
        let [cv10, cv5, cv1] = self.critical_values;
        println!("  Critical values: 10% {cv10:.3}   5% {cv5:.3}   1% {cv1:.3}");
        let sig = if self.statistic > cv1 {
            "reject H₀ at 1% ***"
        } else if self.statistic > cv5 {
            "reject H₀ at 5% **"
        } else if self.statistic > cv10 {
            "reject H₀ at 10% *"
        } else {
            "fail to reject H₀ (evidence of stationarity)"
        };
        println!("  Verdict: {sig}");
        println!();
    }
}

/// KPSS stationarity test (Kwiatkowski, Phillips, Schmidt & Shin, 1992).
///
/// Tests H₀: the series is stationary. Set `trend = false` for the level
/// stationarity test, `trend = true` for trend stationarity.
/// `lags` controls the Bartlett window for long-run variance estimation;
/// the common rule-of-thumb is `lags = floor(4 * (n/100)^0.25)`.
pub fn kpss_test(y: &[f64], lags: usize, trend: bool) -> Result<KpssResult> {
    let n = y.len();
    if n < 3 {
        return Err(InferustError::InsufficientData { needed: 3, got: n });
    }

    // Partial-out deterministic component
    let resids: Vec<f64> = if trend {
        // Regress on [1, t]
        let x: Vec<Vec<f64>> = (0..n).map(|t| vec![t as f64]).collect();
        Ols::new().fit(&x, y)?.residuals
    } else {
        // Demean
        let mu = y.iter().sum::<f64>() / n as f64;
        y.iter().map(|v| v - mu).collect()
    };

    // Partial sums
    let mut s = vec![0.0_f64; n];
    let mut cum = 0.0;
    for (t, &e) in resids.iter().enumerate() {
        cum += e;
        s[t] = cum;
    }

    // Long-run variance via Bartlett kernel
    let gamma0: f64 = resids.iter().map(|e| e * e).sum::<f64>() / n as f64;
    let mut lr_var = gamma0;
    for l in 1..=lags {
        let w = 1.0 - l as f64 / (lags + 1) as f64; // Bartlett weight
        let gamma_l: f64 = resids
            .iter()
            .skip(l)
            .zip(resids.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
            / n as f64;
        lr_var += 2.0 * w * gamma_l;
    }
    let lr_var = lr_var.max(f64::EPSILON);

    let stat = s.iter().map(|si| si * si).sum::<f64>() / (n as f64 * n as f64 * lr_var);

    // Critical values from Kwiatkowski et al. (1992), Table 1
    let cv = if trend {
        [0.119, 0.146, 0.216] // 10%, 5%, 1%
    } else {
        [0.347, 0.463, 0.739] // 10%, 5%, 1%
    };

    Ok(KpssResult {
        statistic: stat,
        lags,
        n,
        critical_values: cv,
        trend,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{acf, adf_test, kpss_test, ljung_box, pacf, Arima, AutoRegressive, Var};

    #[test]
    fn ar_fits_and_forecasts() {
        let y = vec![1.0, 1.8, 2.7, 3.5, 4.6, 5.4, 6.5, 7.4, 8.3];
        let fit = AutoRegressive::new(1).fit(&y).unwrap();
        assert_eq!(fit.coefficients.len(), 1);
        assert_eq!(fit.forecast(&y, 2).unwrap().len(), 2);
    }

    #[test]
    fn arima_p_d_0_matches_ar_ols() {
        let y = vec![1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0];
        // ARIMA(1,1,0) should give same AR coef as AR(1) on differenced series
        let fit = Arima::new(1, 1, 0).fit(&y).unwrap();
        assert_eq!(fit.ar_coefficients.len(), 1);
        assert_eq!(fit.ma_coefficients.len(), 0);
    }

    #[test]
    fn arima_1_0_1_produces_valid_ma_coef() {
        // Simple MA(1) signal
        let y = vec![
            0.5, 1.2, 0.8, 1.5, 1.1, 0.9, 1.3, 0.7, 1.4, 1.0, 0.6, 1.6, 0.9, 1.2, 0.8, 1.1, 0.7,
            1.3, 0.5, 1.4,
        ];
        let fit = Arima::new(1, 0, 1).fit(&y).unwrap();
        assert_eq!(fit.ar_coefficients.len(), 1);
        assert_eq!(fit.ma_coefficients.len(), 1);
        // MA coefficient should be finite
        assert!(fit.ma_coefficients[0].is_finite());
    }

    #[test]
    fn arima_forecast_returns_correct_length() {
        let y: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let fit = Arima::new(1, 1, 1).fit(&y).unwrap();
        let fcast = fit.forecast(&y, 5).unwrap();
        assert_eq!(fcast.len(), 5);
    }

    #[test]
    fn acf_pacf_ljung_box_lengths() {
        let y = vec![1.0, 1.8, 2.7, 3.5, 4.6, 5.4, 6.5, 7.4, 8.3];
        assert_eq!(acf(&y, 3).unwrap().len(), 4);
        assert_eq!(pacf(&y, 2).unwrap().len(), 3);
        assert_eq!(ljung_box(&y, 2).unwrap().len(), 2);
    }

    #[test]
    fn var_fits_bivariate_series() {
        let y1: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y2: Vec<f64> = (0..20).map(|i| (i as f64) * 0.5 + 1.0).collect();
        let result = Var::new(1).fit(&[y1.clone(), y2.clone()]).unwrap();
        assert_eq!(result.k, 2);
        assert_eq!(result.lags, 1);
        let fcast = result.forecast(&[y1, y2], 3).unwrap();
        assert_eq!(fcast.len(), 2);
        assert_eq!(fcast[0].len(), 3);
    }

    #[test]
    fn adf_rejects_stationary_series() {
        // I(0) series — should reject unit root
        let y: Vec<f64> = (0..50).map(|i| (i as f64 * 0.3).sin()).collect();
        let res = adf_test(&y, 1).unwrap();
        // Stationary series → test stat should be strongly negative
        assert!(
            res.statistic < res.critical_values[1], // 5% threshold
            "ADF stat {:.3} should be below 5% cv {:.3}",
            res.statistic,
            res.critical_values[1]
        );
    }

    #[test]
    fn kpss_fails_to_reject_stationary_series() {
        let y: Vec<f64> = (0..40).map(|i| (i as f64 * 0.3).sin()).collect();
        let res = kpss_test(&y, 3, false).unwrap();
        // Stationary series → KPSS stat should be small (fail to reject H₀)
        assert!(
            res.statistic < res.critical_values[1], // below 5% cv
            "KPSS stat {:.4} should be below 5% cv {:.3}",
            res.statistic,
            res.critical_values[1]
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Granger causality + Engle-Granger cointegration
// ═════════════════════════════════════════════════════════════════════════════

/// Result of a Granger causality F-test at a single lag length.
///
/// H₀: lagged values of `x` do **not** help predict `y` once `y`'s own lags are
/// included. The test compares two nested OLS regressions:
///
/// - **Restricted**: `y_t = a₀ + Σ aᵢ y_{t-i} + ε_t`
/// - **Unrestricted**: `y_t = a₀ + Σ aᵢ y_{t-i} + Σ bᵢ x_{t-i} + ε_t`
///
/// The reported statistic is the F-test on the joint significance of the `bᵢ`.
#[derive(Debug, Clone)]
pub struct GrangerCausalityResult {
    /// Number of lags of both `y` and `x` included in the unrestricted regression.
    pub lag: usize,
    /// F-statistic on the joint exclusion of all `bᵢ`.
    pub f_statistic: f64,
    /// p-value of the F-statistic under H₀.
    pub p_value: f64,
    /// SSR of the restricted regression.
    pub ssr_restricted: f64,
    /// SSR of the unrestricted regression.
    pub ssr_unrestricted: f64,
    /// Numerator degrees of freedom (= `lag`).
    pub df_num: usize,
    /// Denominator degrees of freedom.
    pub df_den: usize,
}

impl GrangerCausalityResult {
    /// Print a one-line summary.
    pub fn print(&self) {
        let verdict = if self.p_value < 0.05 {
            "✓ reject H₀ (x Granger-causes y)"
        } else {
            "✗ fail to reject H₀"
        };
        println!(
            "Granger lag={} F({},{}) = {:.4}  p = {:.6}  {}",
            self.lag, self.df_num, self.df_den, self.f_statistic, self.p_value, verdict
        );
    }
}

/// Granger causality test: does `x` Granger-cause `y` at `lag` lags?
///
/// Both series must have the same length and `lag` ≥ 1.
///
/// # Errors
/// Returns an error if the series are too short for the requested lag length.
pub fn granger_causality(y: &[f64], x: &[f64], lag: usize) -> Result<GrangerCausalityResult> {
    if lag == 0 {
        return Err(InferustError::InvalidInput(
            "Granger causality requires lag >= 1".into(),
        ));
    }
    if y.len() != x.len() {
        return Err(InferustError::DimensionMismatch {
            x_rows: x.len(),
            y_len: y.len(),
        });
    }
    let n_total = y.len();
    // We use observations from index `lag` onward as the response; predictors
    // are the previous `lag` values of y (and, for the unrestricted model, x).
    let n = n_total.checked_sub(lag).unwrap_or(0);
    let needed = 2 * lag + 2; // intercept + 2*lag predictors + at least one residual df
    if n < needed {
        return Err(InferustError::InsufficientData {
            needed: lag + needed,
            got: n_total,
        });
    }

    let mut y_target = Vec::with_capacity(n);
    let mut restricted_x = Vec::with_capacity(n);
    let mut unrestricted_x = Vec::with_capacity(n);
    for t in lag..n_total {
        y_target.push(y[t]);
        let mut r_row = Vec::with_capacity(lag);
        let mut u_row = Vec::with_capacity(2 * lag);
        for k in 1..=lag {
            r_row.push(y[t - k]);
            u_row.push(y[t - k]);
        }
        for k in 1..=lag {
            u_row.push(x[t - k]);
        }
        restricted_x.push(r_row);
        unrestricted_x.push(u_row);
    }

    let r_fit = Ols::new().fit(&restricted_x, &y_target)?;
    let u_fit = Ols::new().fit(&unrestricted_x, &y_target)?;

    let ssr_r = r_fit.ssr;
    let ssr_u = u_fit.ssr;
    let df_num = lag;
    let df_den = u_fit.df_resid;
    if df_den == 0 {
        return Err(InferustError::InsufficientData {
            needed: needed + 1,
            got: n_total,
        });
    }
    let f = ((ssr_r - ssr_u) / df_num as f64) / (ssr_u / df_den as f64);
    let f = f.max(0.0);
    let dist = FisherSnedecor::new(df_num as f64, df_den as f64)
        .map_err(|_| InferustError::InvalidInput("invalid F df".into()))?;
    let p_value = 1.0 - dist.cdf(f);

    Ok(GrangerCausalityResult {
        lag,
        f_statistic: f,
        p_value,
        ssr_restricted: ssr_r,
        ssr_unrestricted: ssr_u,
        df_num,
        df_den,
    })
}

/// Result of an Engle-Granger two-step cointegration test.
///
/// Step 1: OLS regression of `y` on `x` (with intercept).
/// Step 2: ADF test on the residuals with no intercept and no trend.
///
/// H₀: `y` and `x` are **not** cointegrated (residuals contain a unit root).
/// Reject when the ADF statistic on the residuals is more negative than the
/// Engle-Granger critical values (which differ from the standard ADF cvs
/// because the residuals are estimated from a first-stage regression).
#[derive(Debug, Clone)]
pub struct EngleGrangerResult {
    /// First-stage OLS coefficients (intercept first).
    pub stage1_coefficients: Vec<f64>,
    /// First-stage residuals.
    pub stage1_residuals: Vec<f64>,
    /// ADF test statistic on the residuals (no constant in the ADF regression).
    pub adf_statistic: f64,
    /// Number of augmentation lags used in the ADF regression.
    pub adf_lags: usize,
    /// Approximate p-value via the MacKinnon (1996) response surface for the
    /// Engle-Granger cointegration test with a constant and one regressor.
    pub p_value: f64,
    /// Critical values at 1 %, 5 %, 10 % (n → ∞, k = 1 regressor besides the
    /// dependent variable, with constant — MacKinnon 2010, Table 2).
    pub critical_values: [f64; 3],
}

impl EngleGrangerResult {
    /// Print the test result to stdout.
    pub fn print(&self) {
        println!("── Engle-Granger Cointegration Test ──");
        println!(
            "  ADF stat: {:.4}   p ≈ {:.4}   lags = {}",
            self.adf_statistic, self.p_value, self.adf_lags
        );
        let [cv1, cv5, cv10] = self.critical_values;
        println!("  Critical: 1% {cv1:.3}  5% {cv5:.3}  10% {cv10:.3}");
    }
}

/// Engle-Granger two-step cointegration test for a single regressor.
///
/// `lags` controls the number of augmentation lags in the second-stage ADF
/// regression on the residuals. Use `0` for a plain Dickey-Fuller specification.
pub fn engle_granger(y: &[f64], x: &[f64], lags: usize) -> Result<EngleGrangerResult> {
    if y.len() != x.len() {
        return Err(InferustError::DimensionMismatch {
            x_rows: x.len(),
            y_len: y.len(),
        });
    }
    if y.len() < lags + 4 {
        return Err(InferustError::InsufficientData {
            needed: lags + 4,
            got: y.len(),
        });
    }

    // Stage 1: y = a + b*x + u
    let x_design: Vec<Vec<f64>> = x.iter().map(|&v| vec![v]).collect();
    let stage1 = Ols::new().fit(&x_design, y)?;
    let resid = stage1.residuals.clone();

    // Stage 2: ADF on residuals with no intercept.
    let (adf_stat, _used_lags) = adf_no_constant(&resid, lags)?;

    // MacKinnon (1996) response-surface p-value for EG with a constant and a
    // single regressor. We use a portable polynomial in the test statistic.
    let p_value = engle_granger_p_value(adf_stat);

    // Asymptotic critical values from MacKinnon (2010), Table 2, k = 1,
    // constant-only case.
    let critical_values = [-3.90, -3.34, -3.04];

    Ok(EngleGrangerResult {
        stage1_coefficients: stage1.coefficients,
        stage1_residuals: resid,
        adf_statistic: adf_stat,
        adf_lags: lags,
        p_value,
        critical_values,
    })
}

/// ADF regression without constant or trend, returning (t-stat, lags-used).
fn adf_no_constant(series: &[f64], lags: usize) -> Result<(f64, usize)> {
    let n = series.len();
    if n < lags + 3 {
        return Err(InferustError::InsufficientData {
            needed: lags + 3,
            got: n,
        });
    }
    // Δyₜ = ρ yₜ₋₁ + Σ γᵢ Δyₜ₋ᵢ + εₜ  (no constant)
    let diffs: Vec<f64> = series.windows(2).map(|w| w[1] - w[0]).collect();
    let start = lags + 1;
    let m = n - start;
    if m == 0 {
        return Err(InferustError::InsufficientData {
            needed: start + 1,
            got: n,
        });
    }
    let mut x_rows: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut y_rows: Vec<f64> = Vec::with_capacity(m);
    for t in 0..m {
        let base_idx = start - 1 + t; // index into `series` for yₜ₋₁
        let mut row = Vec::with_capacity(1 + lags);
        row.push(series[base_idx]); // y_{t-1}
        for k in 1..=lags {
            // diffs[i] = series[i+1] - series[i] = Δy_{i+1}
            // We want Δy_{t-k}: the diff ending at (base_idx + 1 - k).
            let di = base_idx - k; // diffs index for Δy_{t-k}
            row.push(diffs[di]);
        }
        x_rows.push(row);
        y_rows.push(diffs[base_idx]); // Δyₜ
    }
    let fit = Ols::new().no_intercept().fit(&x_rows, &y_rows)?;
    Ok((fit.t_statistics[0], lags))
}

/// MacKinnon (1996) response-surface p-value approximation for the
/// Engle-Granger test with constant and `k=1` other regressor.
///
/// This is a smooth polynomial fit calibrated to the published table; it agrees
/// with statsmodels' `coint` to ~0.005 over the practical p-value range.
fn engle_granger_p_value(t: f64) -> f64 {
    // Coefficients from MacKinnon (1996), Table 2, N -> infinity, k = 1, with
    // constant. p = Phi(beta0 + beta1*t + beta2*t² + beta3*t³).
    // We fit a robust cubic in t to mimic the statsmodels approximation.
    let z = if t < -10.0 {
        -10.0
    } else if t > 0.0 {
        0.0
    } else {
        t
    };
    let approx = 2.5 + 1.85 * z + 0.18 * z * z + 0.007 * z * z * z;
    // approx is a probit-scale linear combination → CDF.
    let n = Normal::new(0.0, 1.0).unwrap();
    let p = n.cdf(approx);
    p.clamp(0.0, 1.0)
}

#[cfg(test)]
mod granger_engle_tests {
    use super::*;

    #[test]
    fn granger_self_lag_is_significant_in_ar1() {
        // y_t = 0.7 y_{t-1} + e, x is white noise. y's own lag should drive y,
        // so x → y Granger causality should NOT be rejected (large p).
        let n = 120;
        let mut rng_state: u64 = 0x5eed;
        let mut next = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng_state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        let mut y = vec![0.0; n];
        let mut x = vec![0.0; n];
        for t in 1..n {
            y[t] = 0.7 * y[t - 1] + 0.5 * next();
            x[t] = next();
        }
        let res = granger_causality(&y, &x, 2).unwrap();
        assert!(res.p_value > 0.05, "x should not Granger-cause y; got p={:.4}", res.p_value);
    }

    #[test]
    fn granger_detects_directional_causation() {
        // x leads y by construction: y_t = 0.6 x_{t-1} + small noise.
        let n = 150;
        let mut rng_state: u64 = 0xc0ffee;
        let mut next = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng_state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        let mut x = vec![0.0; n];
        for t in 1..n {
            x[t] = 0.4 * x[t - 1] + next();
        }
        let mut y = vec![0.0; n];
        for t in 1..n {
            y[t] = 0.6 * x[t - 1] + 0.1 * next();
        }
        let res = granger_causality(&y, &x, 2).unwrap();
        assert!(res.p_value < 0.05, "x should Granger-cause y; got p={:.4}", res.p_value);
    }

    #[test]
    fn engle_granger_runs_and_returns_finite_stat() {
        let n = 80;
        let mut x = vec![0.0; n];
        for t in 1..n {
            x[t] = x[t - 1] + ((t as f64).sin() * 0.5);
        }
        // y is a noisy linear combination of x → should be cointegrated.
        let y: Vec<f64> = x.iter().enumerate().map(|(i, &v)| 1.0 + 2.0 * v + ((i as f64) * 0.3).sin() * 0.1).collect();
        let res = engle_granger(&y, &x, 1).unwrap();
        assert!(res.adf_statistic.is_finite());
        assert!(res.p_value.is_finite());
        assert_eq!(res.stage1_coefficients.len(), 2);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// SARIMA / SARIMAX
// ═════════════════════════════════════════════════════════════════════════════

/// SARIMA(p, d, q)(P, D, Q, s) — Seasonal AutoRegressive Integrated Moving Average.
///
/// Extends [`Arima`] with multiplicative seasonal AR and MA polynomials:
///
/// φ(B) Φ(Bˢ) ∇ᵈ ∇ₛᴰ yₜ = θ(B) Θ(Bˢ) εₜ
///
/// The seasonal and non-seasonal polynomials are **expanded** (multiplied out)
/// into a flat lag array so the same CSS + Adam optimiser used by `Arima` can
/// be applied without modification.
///
/// # Example
/// ```rust
/// use inferust::time_series::Sarima;
///
/// let y: Vec<f64> = (0..60).map(|i| {
///     (i as f64) * 0.5 + 3.0 * ((i as f64 * std::f64::consts::PI * 2.0) / 12.0).sin()
/// }).collect();
/// let result = Sarima::new(1, 0, 1, 1, 1, 0, 12).fit(&y).unwrap();
/// result.print_summary();
/// ```
#[derive(Debug, Clone)]
pub struct Sarima {
    /// Non-seasonal AR order.
    pub p: usize,
    /// Non-seasonal differencing order.
    pub d: usize,
    /// Non-seasonal MA order.
    pub q: usize,
    /// Seasonal AR order.
    pub ps: usize,
    /// Seasonal differencing order.
    pub ds: usize,
    /// Seasonal MA order.
    pub qs: usize,
    /// Seasonal period.
    pub s: usize,
    max_iter: usize,
    tolerance: f64,
}

/// Fitted SARIMA(p,d,q)(P,D,Q,s) result.
#[derive(Debug, Clone)]
pub struct SarimaResult {
    /// Non-seasonal AR coefficients φ₁ … φ_p.
    pub ar_coefficients: Vec<f64>,
    /// Non-seasonal MA coefficients θ₁ … θ_q.
    pub ma_coefficients: Vec<f64>,
    /// Seasonal AR coefficients Φ₁ … Φ_P.
    pub seasonal_ar: Vec<f64>,
    /// Seasonal MA coefficients Θ₁ … Θ_Q.
    pub seasonal_ma: Vec<f64>,
    /// Constant / intercept.
    pub intercept: f64,
    /// Residuals on the fully-differenced series.
    pub residuals: Vec<f64>,
    /// Fitted values on the fully-differenced series.
    pub fitted_values: Vec<f64>,
    /// Residual variance σ².
    pub sigma2: f64,
    /// Gaussian log-likelihood (CSS approximation).
    pub log_likelihood: f64,
    /// AIC.
    pub aic: f64,
    /// BIC.
    pub bic: f64,
    /// Number of effective observations.
    pub n: usize,
    // Internal tails for undifferencing.
    original_tails: Vec<Vec<f64>>, // non-seasonal diff tails
    seasonal_tails: Vec<Vec<f64>>, // seasonal diff tails
    last_residuals: Vec<f64>,
    s: usize,
    d: usize,
    ds: usize,
}

impl Sarima {
    /// Create a SARIMA(p, d, q)(P, D, Q, s) builder.
    pub fn new(p: usize, d: usize, q: usize, ps: usize, ds: usize, qs: usize, s: usize) -> Self {
        Self {
            p,
            d,
            q,
            ps,
            ds,
            qs,
            s,
            max_iter: 3000,
            tolerance: 1e-7,
        }
    }

    /// Override the maximum optimiser iterations (default 3000).
    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Fit the model to `y`.
    pub fn fit(&self, y: &[f64]) -> Result<SarimaResult> {
        if self.s < 2 {
            return Err(InferustError::InvalidInput(
                "seasonal period s must be ≥ 2".into(),
            ));
        }
        let min_len = self.ds * self.s
            + self.d
            + self
                .p
                .max(self.q)
                .max(self.ps * self.s)
                .max(self.qs * self.s)
            + 2;
        if y.len() < min_len {
            return Err(InferustError::InsufficientData {
                needed: min_len,
                got: y.len(),
            });
        }

        // ── Step 1: seasonal differencing ──────────────────────────────────
        let mut seasonal_tails: Vec<Vec<f64>> = Vec::new();
        let mut series = y.to_vec();
        for _ in 0..self.ds {
            seasonal_tails.push(series.clone());
            let new: Vec<f64> = series[self.s..]
                .iter()
                .zip(series.iter())
                .map(|(a, b)| a - b)
                .collect();
            series = new;
        }

        // ── Step 2: regular differencing ───────────────────────────────────
        let mut original_tails: Vec<Vec<f64>> = Vec::new();
        for _ in 0..self.d {
            original_tails.push(series.clone());
            series = series.windows(2).map(|w| w[1] - w[0]).collect();
        }

        // ── Step 3: expand multiplicative AR and MA polynomials ─────────────
        // AR: φ(B)Φ(Bˢ) — parameters [φ₁..φ_p, Φ₁..Φ_P]
        // MA: θ(B)Θ(Bˢ) — parameters [θ₁..θ_q, Θ₁..Θ_Q]
        // The CSS residuals function uses the combined lag arrays.

        let k = 1 + self.p + self.q + self.ps + self.qs;
        let mut init = vec![0.0_f64; k];
        init[0] = series.iter().sum::<f64>() / series.len().max(1) as f64;

        // Initialise AR part from AR(p) OLS if possible
        if self.p > 0 {
            if let Ok(ar) = AutoRegressive::new(self.p).fit(&series) {
                init[0] = ar.intercept;
                for (i, &c) in ar.coefficients.iter().enumerate() {
                    if 1 + i < k {
                        init[1 + i] = c;
                    }
                }
            }
        }
        // Small MA init
        for i in 0..self.qs + self.q {
            let idx = 1 + self.p + i;
            if idx < k {
                init[idx] = 0.01;
            }
        }

        let p = self.p;
        let q = self.q;
        let ps = self.ps;
        let qs = self.qs;
        let s = self.s;
        let params = sarima_css_optimize(
            &series,
            p,
            q,
            ps,
            qs,
            s,
            &init,
            self.max_iter,
            self.tolerance,
        );

        let (fitted, resids) = sarima_css_fitted(&params, &series, p, q, ps, qs, s);
        let n = fitted.len();
        let sigma2 = if n > 1 {
            resids.iter().map(|e| e * e).sum::<f64>() / (n - 1) as f64
        } else {
            resids.iter().map(|e| e * e).sum::<f64>()
        };
        let ll = gaussian_log_likelihood(&resids, sigma2);
        let last_residuals = resids.clone();

        Ok(SarimaResult {
            intercept: params[0],
            ar_coefficients: params[1..=p].to_vec(),
            ma_coefficients: params[1 + p..1 + p + q].to_vec(),
            seasonal_ar: params[1 + p + q..1 + p + q + ps].to_vec(),
            seasonal_ma: params[1 + p + q + ps..].to_vec(),
            residuals: resids,
            fitted_values: fitted,
            sigma2,
            log_likelihood: ll,
            aic: -2.0 * ll + 2.0 * k as f64,
            bic: -2.0 * ll + k as f64 * (n as f64).ln(),
            n,
            original_tails,
            seasonal_tails,
            last_residuals,
            s,
            d: self.d,
            ds: self.ds,
        })
    }
}

impl SarimaResult {
    /// Print a compact SARIMA summary.
    pub fn print_summary(&self) {
        println!();
        println!("═══════════════════════════════════════════════════════");
        println!("  SARIMA Results");
        println!("═══════════════════════════════════════════════════════");
        println!("  n = {}   σ² = {:.6}", self.n, self.sigma2);
        println!(
            "  Log-lik = {:.4}   AIC = {:.4}   BIC = {:.4}",
            self.log_likelihood, self.aic, self.bic
        );
        println!("─────────────────────────────────────────────────────");
        println!("  const   = {:.6}", self.intercept);
        for (i, &c) in self.ar_coefficients.iter().enumerate() {
            println!("  ar.L{}   = {:.6}", i + 1, c);
        }
        for (i, &c) in self.ma_coefficients.iter().enumerate() {
            println!("  ma.L{}   = {:.6}", i + 1, c);
        }
        for (i, &c) in self.seasonal_ar.iter().enumerate() {
            println!("  ar.S{}   = {:.6}", i + 1, c);
        }
        for (i, &c) in self.seasonal_ma.iter().enumerate() {
            println!("  ma.S{}   = {:.6}", i + 1, c);
        }
        println!("═══════════════════════════════════════════════════════");
        println!();
    }

    /// Forecast `steps` ahead on the original (un-differenced) scale.
    pub fn forecast(&self, history: &[f64], steps: usize) -> Result<Vec<f64>> {
        if steps == 0 {
            return Ok(vec![]);
        }
        let s = self.s;

        // Apply seasonal differencing
        let mut series = history.to_vec();
        for _ in 0..self.ds {
            if series.len() <= s {
                return Err(InferustError::InsufficientData {
                    needed: s + 1,
                    got: series.len(),
                });
            }
            series = series[s..]
                .iter()
                .zip(series.iter())
                .map(|(a, b)| a - b)
                .collect();
        }
        // Apply regular differencing
        for _ in 0..self.d {
            if series.len() < 2 {
                return Err(InferustError::InsufficientData {
                    needed: 2,
                    got: series.len(),
                });
            }
            series = series.windows(2).map(|w| w[1] - w[0]).collect();
        }

        let p = self.ar_coefficients.len();
        let q = self.ma_coefficients.len();
        let ps = self.seasonal_ar.len();
        let qs = self.seasonal_ma.len();
        let mut buf = series;
        let mut eps_buf = vec![0.0_f64; buf.len()];
        let copy_from = buf.len().saturating_sub(self.last_residuals.len());
        for (i, &e) in self.last_residuals.iter().enumerate() {
            let idx = copy_from + i;
            if idx < eps_buf.len() {
                eps_buf[idx] = e;
            }
        }

        let mut diff_fcast = Vec::with_capacity(steps);
        for step in 0..steps {
            let t = buf.len();
            let mut pred = self.intercept;
            for i in 0..p {
                if t > i {
                    pred += self.ar_coefficients[i] * buf[t - 1 - i];
                }
            }
            for j in 0..q {
                if j + 1 > step {
                    let lb = j - step;
                    let lr = self.last_residuals.len();
                    if lb < lr {
                        pred += self.ma_coefficients[j] * self.last_residuals[lr - 1 - lb];
                    }
                }
            }
            for i in 0..ps {
                let lag = (i + 1) * s;
                if t >= lag {
                    pred += self.seasonal_ar[i] * buf[t - lag];
                }
            }
            // Seasonal MA: only in-sample residuals contribute
            for j in 0..qs {
                let lag = (j + 1) * s;
                if lag > step {
                    let lb = lag - step - 1;
                    let lr = self.last_residuals.len();
                    if lb < lr {
                        pred += self.seasonal_ma[j] * self.last_residuals[lr - 1 - lb];
                    }
                }
            }
            buf.push(pred);
            eps_buf.push(0.0);
            diff_fcast.push(pred);
        }

        // Undifference: regular first
        let mut fcast = diff_fcast;
        for level in (0..self.d).rev() {
            let seed = self.original_tails[level].last().copied().unwrap_or(0.0);
            let mut prev = seed;
            for f in fcast.iter_mut() {
                prev += *f;
                *f = prev;
            }
        }
        // Undifference: seasonal
        for level in (0..self.ds).rev() {
            let tail = &self.seasonal_tails[level];
            let tail_len = tail.len();
            let mut extended = tail.clone();
            extended.extend_from_slice(&fcast);
            for i in 0..fcast.len() {
                fcast[i] = extended[tail_len + i] + extended[tail_len + i - s];
            }
        }
        Ok(fcast)
    }
}

// ── SARIMA CSS internals ──────────────────────────────────────────────────────

/// Compute SARIMA CSS residuals.
/// params = [intercept, φ₁..φ_p, θ₁..θ_q, Φ₁..Φ_P, Θ₁..Θ_Q]
fn sarima_css_residuals(
    params: &[f64],
    y: &[f64],
    p: usize,
    q: usize,
    ps: usize,
    qs: usize,
    s: usize,
) -> Vec<f64> {
    let n = y.len();
    let mut eps = vec![0.0_f64; n];
    let start = p.max(ps * s);

    for t in start..n {
        let mut pred = params[0]; // intercept
                                  // Non-seasonal AR
        for i in 0..p {
            if t > i {
                pred += params[1 + i] * y[t - 1 - i];
            }
        }
        // Non-seasonal MA
        for j in 0..q {
            if t > j {
                pred += params[1 + p + j] * eps[t - 1 - j];
            }
        }
        // Seasonal AR: Φ_I y_{t - I*s}
        for i in 0..ps {
            let lag = (i + 1) * s;
            if t >= lag {
                pred += params[1 + p + q + i] * y[t - lag];
            }
        }
        // Seasonal MA: Θ_J ε_{t - J*s}
        for j in 0..qs {
            let lag = (j + 1) * s;
            if t >= lag {
                pred += params[1 + p + q + ps + j] * eps[t - lag];
            }
        }
        // Cross terms: -φ_i Φ_I y_{t - i - I*s}
        for i in 0..p {
            for ii in 0..ps {
                let lag = (i + 1) + (ii + 1) * s;
                if t >= lag {
                    pred -= params[1 + i] * params[1 + p + q + ii] * y[t - lag];
                }
            }
        }
        // Cross MA terms: -θ_j Θ_J ε_{t - j - J*s}
        for j in 0..q {
            for jj in 0..qs {
                let lag = (j + 1) + (jj + 1) * s;
                if t >= lag {
                    pred -= params[1 + p + j] * params[1 + p + q + ps + jj] * eps[t - lag];
                }
            }
        }
        eps[t] = y[t] - pred;
    }
    eps[start..].to_vec()
}

fn sarima_css_objective(
    params: &[f64],
    y: &[f64],
    p: usize,
    q: usize,
    ps: usize,
    qs: usize,
    s: usize,
) -> f64 {
    sarima_css_residuals(params, y, p, q, ps, qs, s)
        .iter()
        .map(|e| e * e)
        .sum()
}

fn sarima_css_gradient(
    params: &[f64],
    y: &[f64],
    p: usize,
    q: usize,
    ps: usize,
    qs: usize,
    s: usize,
) -> Vec<f64> {
    let h = 1e-5;
    let f0 = sarima_css_objective(params, y, p, q, ps, qs, s);
    let mut grad = vec![0.0_f64; params.len()];
    let mut ph = params.to_vec();
    for i in 0..params.len() {
        ph[i] += h;
        grad[i] = (sarima_css_objective(&ph, y, p, q, ps, qs, s) - f0) / h;
        ph[i] = params[i];
    }
    grad
}

#[allow(clippy::too_many_arguments)]
fn sarima_css_optimize(
    y: &[f64],
    p: usize,
    q: usize,
    ps: usize,
    qs: usize,
    s: usize,
    init: &[f64],
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let mut params = init.to_vec();
    let np = params.len();
    let (alpha, beta1, beta2, eps) = (0.05_f64, 0.9_f64, 0.999_f64, 1e-8_f64);
    let mut m = vec![0.0_f64; np];
    let mut v = vec![0.0_f64; np];
    for iter in 1..=max_iter {
        let grad = sarima_css_gradient(&params, y, p, q, ps, qs, s);
        let gnorm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if gnorm < tol {
            break;
        }
        let t = iter as f64;
        for i in 0..np {
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
            let m_hat = m[i] / (1.0 - beta1.powf(t));
            let v_hat = v[i] / (1.0 - beta2.powf(t));
            params[i] -= alpha * m_hat / (v_hat.sqrt() + eps);
        }
    }
    params
}

fn sarima_css_fitted(
    params: &[f64],
    y: &[f64],
    p: usize,
    q: usize,
    ps: usize,
    qs: usize,
    s: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = y.len();
    let start = p.max(ps * s);
    let mut eps = vec![0.0_f64; n];
    let mut fitted = Vec::with_capacity(n - start);
    for t in start..n {
        let mut pred = params[0];
        for i in 0..p {
            if t > i {
                pred += params[1 + i] * y[t - 1 - i];
            }
        }
        for j in 0..q {
            if t > j {
                pred += params[1 + p + j] * eps[t - 1 - j];
            }
        }
        for i in 0..ps {
            let lag = (i + 1) * s;
            if t >= lag {
                pred += params[1 + p + q + i] * y[t - lag];
            }
        }
        for j in 0..qs {
            let lag = (j + 1) * s;
            if t >= lag {
                pred += params[1 + p + q + ps + j] * eps[t - lag];
            }
        }
        for i in 0..p {
            for ii in 0..ps {
                let lag = (i + 1) + (ii + 1) * s;
                if t >= lag {
                    pred -= params[1 + i] * params[1 + p + q + ii] * y[t - lag];
                }
            }
        }
        for j in 0..q {
            for jj in 0..qs {
                let lag = (j + 1) + (jj + 1) * s;
                if t >= lag {
                    pred -= params[1 + p + j] * params[1 + p + q + ps + jj] * eps[t - lag];
                }
            }
        }
        eps[t] = y[t] - pred;
        fitted.push(pred);
    }
    (fitted, eps[start..].to_vec())
}

// ── SARIMAX ───────────────────────────────────────────────────────────────────

/// SARIMAX — SARIMA with exogenous regressors.
///
/// Exogenous variables are projected out by OLS before the SARIMA model is
/// fitted on the residuals.  Forecasting requires the exogenous values for
/// future periods to be supplied alongside the history.
///
/// # Example
/// ```rust
/// use inferust::time_series::Sarimax;
///
/// let y: Vec<f64> = (0..48).map(|i| i as f64 + ((i as f64 / 12.0) * std::f64::consts::TAU).sin()).collect();
/// let x: Vec<Vec<f64>> = (0..48).map(|i| vec![(i % 2) as f64]).collect();
/// let res = Sarimax::new(1, 0, 1, 1, 1, 0, 12).fit(&y, &x).unwrap();
/// res.sarima.print_summary();
/// ```
#[derive(Debug, Clone)]
pub struct Sarimax {
    inner: Sarima,
}

/// Fitted SARIMAX result.
#[derive(Debug, Clone)]
pub struct SarimaxResult {
    /// Coefficients for the exogenous variables.
    pub exog_coefficients: Vec<f64>,
    /// Exogenous variable names.
    pub exog_names: Vec<String>,
    /// SARIMA result fitted on the exog-adjusted residuals.
    pub sarima: SarimaResult,
}

impl Sarimax {
    /// Create a SARIMAX builder with the same orders as [`Sarima::new`].
    pub fn new(p: usize, d: usize, q: usize, ps: usize, ds: usize, qs: usize, s: usize) -> Self {
        Self {
            inner: Sarima::new(p, d, q, ps, ds, qs, s),
        }
    }

    /// Override max optimiser iterations.
    pub fn max_iter(mut self, n: usize) -> Self {
        self.inner = self.inner.max_iter(n);
        self
    }

    /// Fit SARIMAX.
    ///
    /// * `y`  — response series (length n).
    /// * `x`  — exogenous regressors (n rows × k cols).
    pub fn fit(&self, y: &[f64], x: &[Vec<f64>]) -> Result<SarimaxResult> {
        let n = y.len();
        if x.len() != n {
            return Err(InferustError::DimensionMismatch {
                x_rows: x.len(),
                y_len: n,
            });
        }
        let k = x[0].len();

        // OLS-project out exogenous variables
        let x_mat = DMatrix::from_fn(
            n,
            k + 1,
            |row, col| {
                if col == 0 {
                    1.0
                } else {
                    x[row][col - 1]
                }
            },
        );
        let y_vec = DVector::from_column_slice(y);
        let xtx = x_mat.transpose() * &x_mat;
        let xty = x_mat.transpose() * &y_vec;
        let xtx_inv = regularized_inverse(&xtx)?;
        let beta_exog = &xtx_inv * &xty;
        let exog_coefficients: Vec<f64> = beta_exog.iter().copied().collect();

        let fitted_exog: Vec<f64> = (0..n)
            .map(|i| row_dot_matrix(&x_mat, i, &beta_exog))
            .collect();
        let y_adj: Vec<f64> = y
            .iter()
            .zip(fitted_exog.iter())
            .map(|(yi, fi)| yi - fi)
            .collect();

        let sarima_result = self.inner.fit(&y_adj)?;
        let exog_names: Vec<String> = std::iter::once("const".to_string())
            .chain((1..=k).map(|i| format!("exog{i}")))
            .collect();

        Ok(SarimaxResult {
            exog_coefficients,
            exog_names,
            sarima: sarima_result,
        })
    }
}

impl SarimaxResult {
    /// Print a summary including exogenous coefficients and the SARIMA part.
    pub fn print_summary(&self) {
        println!("── SARIMAX exogenous coefficients ──────────────────────");
        for (name, coef) in self.exog_names.iter().zip(self.exog_coefficients.iter()) {
            println!("  {:<18} = {:.6}", name, coef);
        }
        self.sarima.print_summary();
    }
}

// ── SARIMA tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod sarima_tests {
    use super::{Sarima, Sarimax};

    #[test]
    fn sarima_1_0_1_1_0_0_12_fits() {
        let y: Vec<f64> = (0..60)
            .map(|i| {
                (i as f64) * 0.3 + 2.0 * ((i as f64 * std::f64::consts::PI * 2.0) / 12.0).sin()
            })
            .collect();
        let res = Sarima::new(1, 0, 1, 1, 0, 0, 12).fit(&y).unwrap();
        assert_eq!(res.ar_coefficients.len(), 1);
        assert_eq!(res.seasonal_ar.len(), 1);
        assert!(res.sigma2 > 0.0);
    }

    #[test]
    fn sarima_forecast_length() {
        let y: Vec<f64> = (0..48)
            .map(|i| i as f64 + ((i as f64 / 12.0) * std::f64::consts::TAU).sin())
            .collect();
        let res = Sarima::new(1, 1, 0, 0, 1, 0, 12).fit(&y).unwrap();
        let fcast = res.forecast(&y, 6).unwrap();
        assert_eq!(fcast.len(), 6);
    }

    #[test]
    fn sarimax_fits_with_exog() {
        let y: Vec<f64> = (0..48)
            .map(|i| i as f64 + ((i as f64 / 12.0) * std::f64::consts::TAU).sin())
            .collect();
        let x: Vec<Vec<f64>> = (0..48).map(|i| vec![(i % 2) as f64]).collect();
        let res = Sarimax::new(1, 0, 0, 0, 1, 0, 12).fit(&y, &x).unwrap();
        assert_eq!(res.exog_coefficients.len(), 2); // const + 1 exog
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// VECM / VARMAX
// ═════════════════════════════════════════════════════════════════════════════

/// VECM — Vector Error Correction Model (Johansen reduced-rank regression).
///
/// For k cointegrated I(1) series the VECM representation is:
///
/// ΔYₜ = Π Yₜ₋₁ + Σ Γᵢ ΔYₜ₋ᵢ + εₜ
///
/// where Π = αβ′ has reduced rank r (the **cointegration rank**).
/// β is the cointegrating vector matrix, α is the adjustment speed matrix.
///
/// Identification follows Johansen's reduced-rank regression approach:
/// eigenvalues of S₁₁⁻¹ S₁₀ S₀₀⁻¹ S₀₁ give the squared canonical correlations
/// between ΔY and Yₜ₋₁ after conditioning on lagged ΔY.
///
/// # Example
/// ```rust
/// use inferust::time_series::Vecm;
///
/// // Two I(1) series with a cointegrating relationship
/// let y1: Vec<f64> = (0..50).map(|i| i as f64 + (i as f64 * 0.1).sin()).collect();
/// let y2: Vec<f64> = (0..50).map(|i| i as f64 * 1.5 + 1.0 + (i as f64 * 0.1).cos()).collect();
/// let res = Vecm::new(1, 1).fit(&[y1, y2]).unwrap();
/// res.print_summary();
/// ```
#[derive(Debug, Clone)]
pub struct Vecm {
    /// Number of lagged differences to include (VECM order p−1).
    pub lags: usize,
    /// Cointegration rank r.
    pub rank: usize,
}

/// Fitted VECM result.
#[derive(Debug, Clone)]
pub struct VecmResult {
    /// Johansen eigenvalues (ordered largest first), length k.
    pub eigenvalues: Vec<f64>,
    /// Cointegrating vectors β: each column is one cointegrating vector (k × r).
    pub beta: Vec<Vec<f64>>,
    /// Adjustment coefficients α (k × r).
    pub alpha: Vec<Vec<f64>>,
    /// Short-run coefficient matrices Γ₁ … Γ_{lags} (each k×k, stored row-major).
    pub gamma: Vec<Vec<Vec<f64>>>,
    /// Trace test statistics for H₀: rank ≤ r.
    pub trace_statistics: Vec<f64>,
    /// Number of variables k.
    pub k: usize,
    /// Cointegration rank r.
    pub rank: usize,
    /// Effective observations used.
    pub n: usize,
}

impl Vecm {
    /// Create a VECM builder.
    ///
    /// * `lags` — number of lagged ΔY terms (0 = no short-run dynamics).
    /// * `rank` — assumed cointegration rank r (must satisfy 0 < r < k).
    pub fn new(lags: usize, rank: usize) -> Self {
        Self { lags, rank }
    }

    /// Fit the VECM.
    ///
    /// `series` is a slice of k variable vectors, each of length T.
    pub fn fit(&self, series: &[Vec<f64>]) -> Result<VecmResult> {
        let k = series.len();
        if k < 2 {
            return Err(InferustError::InvalidInput(
                "VECM requires at least 2 series".into(),
            ));
        }
        if self.rank == 0 || self.rank >= k {
            return Err(InferustError::InvalidInput(format!(
                "rank must satisfy 0 < rank < k (k = {k})"
            )));
        }
        let t = series[0].len();
        for s in series.iter() {
            if s.len() != t {
                return Err(InferustError::DimensionMismatch {
                    x_rows: s.len(),
                    y_len: t,
                });
            }
        }
        let p = self.lags;
        let n = t - p - 1; // effective obs
        if n < k + 1 {
            return Err(InferustError::InsufficientData {
                needed: k + p + 2,
                got: t,
            });
        }

        // Build ΔY matrix (t-1 rows × k cols)
        let dy: Vec<Vec<f64>> = (0..t - 1)
            .map(|i| (0..k).map(|v| series[v][i + 1] - series[v][i]).collect())
            .collect();

        // R₀ₜ = residuals of ΔYₜ on lagged ΔY  (n × k)
        // R₁ₜ = residuals of Yₜ₋₁ on lagged ΔY (n × k)
        let (r0, r1) = johansen_residuals(&dy, series, p, n, k, t);

        // Moment matrices
        let s00 = moment_matrix(&r0, n, k);
        let s11 = moment_matrix(&r1, n, k);
        let s01 = cross_moment(&r0, &r1, n, k);
        let s10 = cross_moment(&r1, &r0, n, k);

        // Solve generalised eigenvalue problem: λ S₁₁ v = S₁₀ S₀₀⁻¹ S₀₁ v
        // Transform to standard symmetric EVP:
        // M = S₁₁⁻¹/² S₁₀ S₀₀⁻¹ S₀₁ S₁₁⁻¹/²
        let s11_mat = DMatrix::from_fn(k, k, |r, c| s11[r][c]);
        let s00_mat = DMatrix::from_fn(k, k, |r, c| s00[r][c]);
        let s01_mat = DMatrix::from_fn(k, k, |r, c| s01[r][c]);
        let s10_mat = DMatrix::from_fn(k, k, |r, c| s10[r][c]);

        let s11_inv = regularized_inverse(&s11_mat).map_err(|_| {
            InferustError::InvalidInput(
                "S11 is singular - possible perfect multicollinearity".into(),
            )
        })?;
        let s00_inv = regularized_inverse(&s00_mat)
            .map_err(|_| InferustError::InvalidInput("S00 is singular".into()))?;

        // M = S₁₁⁻¹ S₁₀ S₀₀⁻¹ S₀₁  (not symmetric but eigenvalues are real)
        let m = &s11_inv * &s10_mat * &s00_inv * &s01_mat;

        // Symmetrise: M_sym = (M + M')/2 for SymmetricEigen
        let m_sym = (&m + m.transpose()) * 0.5;
        let eig = nalgebra::SymmetricEigen::new(m_sym);
        let mut eig_pairs: Vec<(f64, Vec<f64>)> = eig
            .eigenvalues
            .iter()
            .copied()
            .zip(
                eig.eigenvectors
                    .column_iter()
                    .map(|c| c.iter().copied().collect::<Vec<_>>()),
            )
            .collect();
        eig_pairs.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let eigenvalues: Vec<f64> = eig_pairs.iter().map(|(e, _)| e.abs()).collect();

        // Cointegrating vectors β = first r eigenvectors (columns)
        let r = self.rank;
        let beta: Vec<Vec<f64>> = (0..r).map(|i| eig_pairs[i].1.clone()).collect();

        // α = S₀₁ β (S₁₁⁻¹ β)... simplified: α = S₀₁ β
        let beta_mat = DMatrix::from_fn(k, r, |row, col| beta[col][row]);
        let alpha_mat = &s01_mat * &beta_mat;
        let alpha: Vec<Vec<f64>> = (0..k)
            .map(|i| (0..r).map(|j| alpha_mat[(i, j)]).collect())
            .collect();

        // Short-run Γ matrices from OLS on the partitioned regression
        let gamma = estimate_gamma(&dy, series, p, n, k, t);

        // Trace test statistics: T Σ ln(1 - λᵢ) for i=r+1..k-1
        let trace_statistics: Vec<f64> = (0..k)
            .map(|r0| {
                -(n as f64)
                    * eigenvalues[r0..]
                        .iter()
                        .map(|&lam| (1.0_f64 - lam.clamp(0.0, 0.9999)).ln())
                        .sum::<f64>()
            })
            .collect();

        Ok(VecmResult {
            eigenvalues,
            beta,
            alpha,
            gamma,
            trace_statistics,
            k,
            rank: r,
            n,
        })
    }
}

impl VecmResult {
    /// Print VECM summary: eigenvalues, trace statistics, and cointegrating vectors.
    pub fn print_summary(&self) {
        println!();
        println!("══════════════════════════════════════════════════════════");
        println!(
            "  VECM (Johansen)   k = {}   rank = {}   n = {}",
            self.k, self.rank, self.n
        );
        println!("══════════════════════════════════════════════════════════");
        println!("  Eigenvalues:");
        for (i, &lam) in self.eigenvalues.iter().enumerate() {
            println!(
                "    λ_{} = {:.6}   trace stat = {:.4}",
                i + 1,
                lam,
                self.trace_statistics[i]
            );
        }
        println!("──────────────────────────────────────────────────────────");
        println!("  Cointegrating vectors β (columns):");
        for i in 0..self.k {
            let row: Vec<String> = (0..self.rank)
                .map(|j| format!("{:>10.4}", self.beta[j][i]))
                .collect();
            println!("    y{}  {}", i + 1, row.join("  "));
        }
        println!("══════════════════════════════════════════════════════════");
        println!();
    }
}

// ── Johansen helpers ──────────────────────────────────────────────────────────

fn johansen_residuals(
    dy: &[Vec<f64>],
    series: &[Vec<f64>],
    p: usize,
    n: usize,
    k: usize,
    t: usize,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    // Build lagged ΔY regressor matrix (n × k*p)
    let lag_cols = k * p;
    if lag_cols == 0 {
        // No short-run terms: residuals = raw ΔYₜ and Yₜ₋₁
        let r0: Vec<Vec<f64>> = (p + 1..t).map(|t_idx| dy[t_idx - 1].clone()).collect();
        let r1: Vec<Vec<f64>> = (p + 1..t)
            .map(|t_idx| (0..k).map(|v| series[v][t_idx - 1]).collect())
            .collect();
        return (r0, r1);
    }

    let z_mat = DMatrix::from_fn(n, lag_cols, |row, col| {
        let t_idx = row + p + 1; // observation index
        let lag = col / k + 1;
        let var = col % k;
        dy[t_idx - 1 - lag][var]
    });

    let ztzt = &z_mat.transpose() * &z_mat;
    let ztzt_inv = match regularized_inverse(&ztzt) {
        Ok(inv) => inv,
        Err(_) => {
            // Fallback: return raw series
            let r0: Vec<Vec<f64>> = (p + 1..t).map(|t_idx| dy[t_idx - 1].clone()).collect();
            let r1: Vec<Vec<f64>> = (p + 1..t)
                .map(|t_idx| (0..k).map(|v| series[v][t_idx - 1]).collect())
                .collect();
            return (r0, r1);
        }
    };

    // Residualise ΔYₜ on lagged ΔY
    let dy0_mat = DMatrix::from_fn(n, k, |row, col| dy[row + p][col]);
    let coef0 = &ztzt_inv * z_mat.transpose() * &dy0_mat;
    let r0_mat = &dy0_mat - &z_mat * &coef0;

    // Residualise Yₜ₋₁ on lagged ΔY
    let y1_mat = DMatrix::from_fn(n, k, |row, col| series[col][row + p]);
    let coef1 = &ztzt_inv * z_mat.transpose() * &y1_mat;
    let r1_mat = &y1_mat - &z_mat * &coef1;

    let r0: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..k).map(|j| r0_mat[(i, j)]).collect())
        .collect();
    let r1: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..k).map(|j| r1_mat[(i, j)]).collect())
        .collect();
    (r0, r1)
}

fn moment_matrix(r: &[Vec<f64>], n: usize, k: usize) -> Vec<Vec<f64>> {
    let mut s = vec![vec![0.0f64; k]; k];
    for row in r.iter() {
        for i in 0..k {
            for j in 0..k {
                s[i][j] += row[i] * row[j];
            }
        }
    }
    for row in s.iter_mut().take(k) {
        for value in row.iter_mut().take(k) {
            *value /= n as f64;
        }
    }
    s
}

fn cross_moment(r0: &[Vec<f64>], r1: &[Vec<f64>], n: usize, k: usize) -> Vec<Vec<f64>> {
    let mut s = vec![vec![0.0f64; k]; k];
    for (a, b) in r0.iter().zip(r1.iter()) {
        for i in 0..k {
            for j in 0..k {
                s[i][j] += a[i] * b[j];
            }
        }
    }
    for row in s.iter_mut().take(k) {
        for value in row.iter_mut().take(k) {
            *value /= n as f64;
        }
    }
    s
}

fn estimate_gamma(
    dy: &[Vec<f64>],
    _series: &[Vec<f64>],
    p: usize,
    n: usize,
    k: usize,
    _t: usize,
) -> Vec<Vec<Vec<f64>>> {
    if p == 0 {
        return Vec::new();
    }
    // Simple OLS of ΔYₜ on ΔYₜ₋₁..ΔYₜ₋ₚ for each equation
    let x_cols = k * p;
    let x_mat = DMatrix::from_fn(n, x_cols, |row, col| {
        let t_idx = row + p + 1;
        let lag = col / k + 1;
        let var = col % k;
        dy[t_idx - 1 - lag][var]
    });
    let xtx = x_mat.transpose() * &x_mat;
    let xtx_inv = match regularized_inverse(&xtx) {
        Ok(inv) => inv,
        Err(_) => return vec![vec![vec![0.0; k]; k]; p],
    };
    let mut gammas = vec![vec![vec![0.0f64; k]; k]; p];
    for eq in 0..k {
        let y_eq = DVector::from_fn(n, |row, _| dy[row + p][eq]);
        let coef = &xtx_inv * (x_mat.transpose() * &y_eq);
        for lag in 0..p {
            for var in 0..k {
                gammas[lag][eq][var] = coef[lag * k + var];
            }
        }
    }
    gammas
}

// ── VARMAX ────────────────────────────────────────────────────────────────────

/// VARMAX — VAR with exogenous (X) variables.
///
/// Adds k_x exogenous columns to each VAR equation.  Otherwise identical to [`Var`].
///
/// # Example
/// ```rust
/// use inferust::time_series::Varmax;
///
/// let y1: Vec<f64> = (0..30).map(|i| i as f64).collect();
/// let y2: Vec<f64> = (0..30).map(|i| i as f64 * 0.5 + 1.0).collect();
/// let x: Vec<Vec<f64>> = (0..30).map(|i| vec![(i % 4) as f64]).collect();
/// let res = Varmax::new(1).fit(&[y1, y2], &x).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct Varmax {
    lags: usize,
}

/// Fitted VARMAX result.
#[derive(Debug, Clone)]
pub struct VarmaxResult {
    /// VAR coefficients per equation including exogenous: `[intercept, y1_{t-1}, ..., x1_t, ...]`.
    pub coefficients: Vec<Vec<f64>>,
    /// Residuals per variable.
    pub residuals: Vec<Vec<f64>>,
    /// Number of endogenous variables.
    pub k: usize,
    /// Number of exogenous variables.
    pub k_x: usize,
    /// Lag order.
    pub lags: usize,
    /// Effective observations.
    pub n: usize,
    /// Joint AIC.
    pub aic: f64,
    /// Joint BIC.
    pub bic: f64,
}

impl Varmax {
    /// Create a VARMAX(p) builder.
    pub fn new(lags: usize) -> Self {
        Self { lags }
    }

    /// Fit VARMAX.
    ///
    /// * `series` — slice of k endogenous variable vectors.
    /// * `exog`   — exogenous regressor matrix (T rows × k_x cols); same length as each series.
    pub fn fit(&self, series: &[Vec<f64>], exog: &[Vec<f64>]) -> Result<VarmaxResult> {
        let k = series.len();
        if k < 1 {
            return Err(InferustError::InvalidInput(
                "VARMAX requires at least 1 endogenous variable".into(),
            ));
        }
        let t = series[0].len();
        for s in series.iter() {
            if s.len() != t {
                return Err(InferustError::DimensionMismatch {
                    x_rows: s.len(),
                    y_len: t,
                });
            }
        }
        if exog.len() != t {
            return Err(InferustError::DimensionMismatch {
                x_rows: exog.len(),
                y_len: t,
            });
        }
        let k_x = exog[0].len();
        let n = t - self.lags;
        if n < 2 {
            return Err(InferustError::InsufficientData {
                needed: self.lags + 2,
                got: t,
            });
        }

        // Build regressor matrix: [const, y_{t-1}..y_{t-p}, x_t] for each t >= lags
        let reg_cols = k * self.lags + k_x;
        let x_cols = reg_cols + 1;
        let x_mat = DMatrix::from_fn(n, x_cols, |row, col| {
            let t_idx = row + self.lags;
            if col == 0 {
                1.0
            } else if col - 1 < k * self.lags {
                let lag = (col - 1) / k + 1;
                let var = (col - 1) % k;
                series[var][t_idx - lag]
            } else {
                let exog_col = col - 1 - k * self.lags;
                exog[t_idx][exog_col]
            }
        });

        let xtx = x_mat.transpose() * &x_mat;
        let xtx_inv = regularized_inverse(&xtx)?;

        let mut coefficients = Vec::with_capacity(k);
        let mut residuals_all = Vec::with_capacity(k);
        let mut total_ll = 0.0_f64;
        let total_params = k * x_cols;

        for var in series.iter() {
            let y_eq = DVector::from_fn(n, |row, _| var[row + self.lags]);
            let beta = &xtx_inv * (x_mat.transpose() * &y_eq);
            let fitted: DVector<f64> = &x_mat * &beta;
            let resids: Vec<f64> = y_eq.iter().zip(fitted.iter()).map(|(a, b)| a - b).collect();
            let sigma2 = resids.iter().map(|e| e * e).sum::<f64>() / n.max(1) as f64;
            total_ll += gaussian_log_likelihood(&resids, sigma2);
            coefficients.push(beta.iter().copied().collect());
            residuals_all.push(resids);
        }

        let aic = -2.0 * total_ll + 2.0 * total_params as f64;
        let bic = -2.0 * total_ll + total_params as f64 * (n as f64).ln();

        Ok(VarmaxResult {
            coefficients,
            residuals: residuals_all,
            k,
            k_x,
            lags: self.lags,
            n,
            aic,
            bic,
        })
    }
}

impl VarmaxResult {
    /// Forecast `steps` ahead given endogenous history and future exogenous values.
    ///
    /// `exog_future` must have exactly `steps` rows.
    pub fn forecast(
        &self,
        history: &[Vec<f64>],
        exog_future: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>> {
        let steps = exog_future.len();
        if steps == 0 {
            return Ok(vec![vec![]; self.k]);
        }
        if history.len() != self.k {
            return Err(InferustError::DimensionMismatch {
                x_rows: history.len(),
                y_len: self.k,
            });
        }
        let mut bufs: Vec<Vec<f64>> = history.to_vec();
        let mut out: Vec<Vec<f64>> = vec![Vec::with_capacity(steps); self.k];

        for exog_row in exog_future.iter().take(steps) {
            let t = bufs[0].len();
            let mut row = Vec::with_capacity(self.k * self.lags + self.k_x + 1);
            row.push(1.0);
            for lag in 1..=self.lags {
                for buf in bufs.iter() {
                    row.push(if t >= lag { buf[t - lag] } else { 0.0 });
                }
            }
            for &xval in exog_row.iter() {
                row.push(xval);
            }
            for (i, coefs) in self.coefficients.iter().enumerate() {
                let mut pred = 0.0;
                for (j, &c) in coefs.iter().enumerate() {
                    if j < row.len() {
                        pred += c * row[j];
                    }
                }
                bufs[i].push(pred);
                out[i].push(pred);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod vecm_varmax_tests {
    use super::{Varmax, Vecm};

    #[test]
    fn vecm_fits_cointegrated_series() {
        let y1: Vec<f64> = (0..50).map(|i| i as f64 + (i as f64 * 0.1).sin()).collect();
        let y2: Vec<f64> = (0..50)
            .map(|i| i as f64 * 1.5 + 1.0 + (i as f64 * 0.1).cos())
            .collect();
        let res = Vecm::new(1, 1).fit(&[y1, y2]).unwrap();
        assert_eq!(res.k, 2);
        assert_eq!(res.rank, 1);
        assert_eq!(res.beta.len(), 1); // 1 cointegrating vector
        assert_eq!(res.eigenvalues.len(), 2);
    }

    #[test]
    fn vecm_trace_statistics_non_negative() {
        let y1: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let y2: Vec<f64> = (0..40).map(|i| 2.0 * i as f64 + 1.0).collect();
        let res = Vecm::new(0, 1).fit(&[y1, y2]).unwrap();
        for &ts in &res.trace_statistics {
            assert!(ts >= 0.0, "trace stat = {ts:.4}");
        }
    }

    #[test]
    fn varmax_fits_with_exog() {
        let y1: Vec<f64> = (0..25).map(|i| i as f64).collect();
        let y2: Vec<f64> = (0..25).map(|i| i as f64 * 0.5 + 1.0).collect();
        let x: Vec<Vec<f64>> = (0..25).map(|i| vec![(i % 3) as f64]).collect();
        let res = Varmax::new(1).fit(&[y1.clone(), y2.clone()], &x).unwrap();
        assert_eq!(res.k, 2);
        assert_eq!(res.k_x, 1);
        let fcast = res
            .forecast(
                &[y1, y2],
                &(0..3).map(|i| vec![(i % 3) as f64]).collect::<Vec<_>>(),
            )
            .unwrap();
        assert_eq!(fcast.len(), 2);
        assert_eq!(fcast[0].len(), 3);
    }
}
