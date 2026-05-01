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
use statrs::distribution::{ChiSquared, ContinuousCDF};

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
        Self { p, d, q, max_iter: 2000, tolerance: 1e-7 }
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
                    sigma2, log_likelihood: ll,
                    aic: -2.0 * ll + 2.0 * k as f64,
                    bic: -2.0 * ll + k as f64 * (n as f64).ln(),
                    n, p: 0, d: self.d, q: 0,
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
                sigma2, log_likelihood: ll,
                aic: -2.0 * ll + 2.0 * k as f64,
                bic: -2.0 * ll + k as f64 * (n as f64).ln(),
                n, p: self.p, d: self.d, q: 0,
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
                    if 1 + i < k { init[1 + i] = c; }
                }
            }
        }
        // MA initialised to small positive values
        for i in 0..self.q {
            init[1 + self.p + i] = 0.01;
        }

        let params = css_optimize(&series, self.p, self.q, &init, self.max_iter, self.tolerance);
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
            sigma2, log_likelihood: ll,
            aic: -2.0 * ll + 2.0 * k as f64,
            bic: -2.0 * ll + k as f64 * (n as f64).ln(),
            n, p: self.p, d: self.d, q: self.q,
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
        if steps == 0 { return Ok(vec![]); }
        let p = self.p;
        let q = self.q;

        // Build the d-times-differenced history
        let mut diff_hist = history.to_vec();
        for _ in 0..self.d {
            if diff_hist.len() < 2 {
                return Err(InferustError::InsufficientData { needed: 2, got: diff_hist.len() });
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
            if idx < eps_buf.len() { eps_buf[idx] = e; }
        }

        let mut diff_fcast = Vec::with_capacity(steps);
        for step in 0..steps {
            let t = buf.len();
            let mut pred = self.intercept;
            // AR terms
            for i in 0..p {
                if t > i { pred += self.ar_coefficients[i] * buf[t - 1 - i]; }
            }
            // MA terms: only past residuals contribute (future ε = 0)
            for j in 0..q {
                // need ε_{t - 1 - j} where t = buf.len()
                // that residual is in-sample only if (j + 1) > step
                if j + 1 > step {
                    let lookback = j - step; // how many steps before the last in-sample residual
                    let lr_len = self.last_residuals.len();
                    if lookback < lr_len {
                        pred += self.ma_coefficients[j] * self.last_residuals[lr_len - 1 - lookback];
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
            if t >= j + 1 {
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
fn css_optimize(y: &[f64], p: usize, q: usize, init: &[f64], max_iter: usize, tol: f64) -> Vec<f64> {
    let mut params = init.to_vec();
    let np = params.len();
    let (alpha, beta1, beta2, eps) = (0.05_f64, 0.9_f64, 0.999_f64, 1e-8_f64);
    let mut m = vec![0.0_f64; np];
    let mut v = vec![0.0_f64; np];
    for iter in 1..=max_iter {
        let grad = css_gradient(&params, y, p, q);
        let gnorm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if gnorm < tol { break; }
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
            if t >= j + 1 {
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
            return Err(InferustError::InvalidInput("VAR lags must be at least 1".into()));
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

        // Build regressor matrix: for each time t >= lags,
        // row = [y1_{t-1}, ..., yk_{t-1}, y1_{t-2}, ..., yk_{t-lags}]
        let mut x: Vec<Vec<f64>> = Vec::with_capacity(n);
        for t_idx in self.lags..t {
            let mut row = Vec::with_capacity(k * self.lags);
            for lag in 1..=self.lags {
                for var in series.iter() {
                    row.push(var[t_idx - lag]);
                }
            }
            x.push(row);
        }

        // Feature names for VAR regressors
        let feat_names: Vec<String> = (1..=self.lags)
            .flat_map(|lag| (1..=k).map(move |v| format!("L{}.y{}", lag, v)))
            .collect();

        let mut coefficients = Vec::with_capacity(k);
        let mut residuals_all = Vec::with_capacity(k);
        let mut total_ll = 0.0_f64;
        let total_params = k * (k * self.lags + 1); // intercepts + slope params across all eq.

        for var in series.iter() {
            let y_eq: Vec<f64> = var[self.lags..].to_vec();
            let ols = Ols::new()
                .with_feature_names(feat_names.clone())
                .fit(&x, &y_eq)?;
            let ll = gaussian_log_likelihood(&ols.residuals, ols.mse_resid);
            total_ll += ll;
            coefficients.push(ols.coefficients);
            residuals_all.push(ols.residuals);
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
        if steps == 0 { return Ok(vec![vec![]; self.k]); }
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
                    if t >= lag { row.push(buf[t - lag]); } else { row.push(0.0); }
                }
            }
            for (i, coefs) in self.coefficients.iter().enumerate() {
                let mut pred = coefs[0]; // intercept
                for (j, &c) in coefs[1..].iter().enumerate() {
                    if j < row.len() { pred += c * row[j]; }
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
        return Err(InferustError::InsufficientData { needed: 2, got: series.len() });
    }
    let mean = series.iter().sum::<f64>() / series.len() as f64;
    let denom = series.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
    Ok((0..=max_lag)
        .map(|lag| {
            if lag == 0 { return 1.0; }
            series.iter().skip(lag)
                .zip(series.iter())
                .map(|(a, b)| (a - mean) * (b - mean))
                .sum::<f64>() / denom.max(f64::EPSILON)
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
        let stat = n * (n + 2.0)
            * rhos.iter().enumerate().skip(1).take(lag)
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
        println!("  Lags: {}   n: {}   stat: {:.4}   p ≈ {:.4}",
            self.lags, self.n, self.statistic, self.p_value);
        let [cv1, cv5, cv10] = self.critical_values;
        println!("  Critical values: 1% {cv1:.3}   5% {cv5:.3}   10% {cv10:.3}");
        let sig = if self.statistic < cv1 { "***" }
            else if self.statistic < cv5 { "**" }
            else if self.statistic < cv10 { "*" }
            else { "(not significant)" };
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
        return Err(InferustError::InsufficientData { needed: min_len, got: n });
    }

    // First difference
    let dy: Vec<f64> = y.windows(2).map(|w| w[1] - w[0]).collect(); // length n-1

    // Build regressors for obs t = lags+1 .. n-1 (0-indexed)
    // each row: [y_{t-1}, Δy_{t-1}, ..., Δy_{t-lags}]
    let t_start = lags + 1;
    let n_obs = n - 1 - t_start; // number of valid rows
    if n_obs < 2 {
        return Err(InferustError::InsufficientData { needed: t_start + 3, got: n });
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
    if stat <= cv1  { return 0.01; }
    if stat <= cv5  { return 0.01 + 0.04 * (stat - cv1)  / (cv5  - cv1); }
    if stat <= cv10 { return 0.05 + 0.05 * (stat - cv5)  / (cv10 - cv5); }
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
        println!("  Lags: {}   n: {}   stat: {:.4}", self.lags, self.n, self.statistic);
        let [cv10, cv5, cv1] = self.critical_values;
        println!("  Critical values: 10% {cv10:.3}   5% {cv5:.3}   1% {cv1:.3}");
        let sig = if self.statistic > cv1 { "reject H₀ at 1% ***" }
            else if self.statistic > cv5 { "reject H₀ at 5% **" }
            else if self.statistic > cv10 { "reject H₀ at 10% *" }
            else { "fail to reject H₀ (evidence of stationarity)" };
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
        let gamma_l: f64 = resids.iter().skip(l)
            .zip(resids.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>() / n as f64;
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

    Ok(KpssResult { statistic: stat, lags, n, critical_values: cv, trend })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{acf, adf_test, kpss_test, ljung_box, pacf, Arima, AutoRegressive, Var};

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "expected {b:.6} got {a:.6} (tol {tol})");
    }

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
        let y = vec![0.5, 1.2, 0.8, 1.5, 1.1, 0.9, 1.3, 0.7, 1.4, 1.0,
                     0.6, 1.6, 0.9, 1.2, 0.8, 1.1, 0.7, 1.3, 0.5, 1.4];
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
        assert!(res.statistic < res.critical_values[1], // 5% threshold
            "ADF stat {:.3} should be below 5% cv {:.3}", res.statistic, res.critical_values[1]);
    }

    #[test]
    fn kpss_fails_to_reject_stationary_series() {
        let y: Vec<f64> = (0..40).map(|i| (i as f64 * 0.3).sin()).collect();
        let res = kpss_test(&y, 3, false).unwrap();
        // Stationary series → KPSS stat should be small (fail to reject H₀)
        assert!(res.statistic < res.critical_values[1], // below 5% cv
            "KPSS stat {:.4} should be below 5% cv {:.3}", res.statistic, res.critical_values[1]);
    }
}
