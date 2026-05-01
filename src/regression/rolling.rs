//! Rolling OLS and Recursive OLS (CUSUM).
//!
//! | Model | Description |
//! |---|---|
//! | [`RollingOls`] | Fits OLS inside a sliding window of fixed width, producing per-window coefficient paths. |
//! | [`RecursiveOls`] | Fits OLS incrementally (one observation at a time) using the Sherman-Morrison rank-1 update; computes CUSUM stability test. |
//!
//! # Quick start
//! ```rust
//! use inferust::regression::{RollingOls, RecursiveOls};
//!
//! let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
//! let y: Vec<f64> = (0..20).map(|i| 2.0 * i as f64 + 1.0).collect();
//!
//! // Rolling coefficients
//! let rolling = RollingOls::new(8).fit(&x, &y).unwrap();
//! println!("slopes over time: {:?}", rolling.slopes());
//!
//! // Recursive OLS + CUSUM
//! let rec = RecursiveOls::new().fit(&x, &y).unwrap();
//! rec.print_cusum();
//! ```

use nalgebra::{DMatrix, DVector};

use crate::error::{InferustError, Result};

// ── Rolling OLS ───────────────────────────────────────────────────────────────

/// OLS fitted inside a sliding window of fixed width.
///
/// For each window position the full OLS problem is solved independently,
/// yielding a **path** of coefficients, standard errors, and R² values that
/// reveal structural changes over time.
#[derive(Debug, Clone)]
pub struct RollingOls {
    window: usize,
    add_intercept: bool,
    feature_names: Vec<String>,
}

/// Fitted rolling OLS result.
#[derive(Debug, Clone)]
pub struct RollingOlsResult {
    /// Coefficient matrix: `coefficients[t]` = coefficients estimated in window ending at index `t + window - 1`.
    /// Each inner vec is `[const?, x1, x2, ...]`.
    pub coefficients: Vec<Vec<f64>>,
    /// Standard errors matching `coefficients`.
    pub std_errors: Vec<Vec<f64>>,
    /// R² for each window.
    pub r_squared: Vec<f64>,
    /// Residual MSE for each window.
    pub mse_resid: Vec<f64>,
    /// Window width.
    pub window: usize,
    /// Number of windows (= n − window + 1).
    pub n_windows: usize,
    /// Feature names.
    pub feature_names: Vec<String>,
}

impl RollingOls {
    /// Create a rolling OLS builder with the given window size.
    pub fn new(window: usize) -> Self {
        Self {
            window,
            add_intercept: true,
            feature_names: Vec::new(),
        }
    }

    /// Do not add an intercept column.
    pub fn no_intercept(mut self) -> Self {
        self.add_intercept = false;
        self
    }

    /// Set human-readable predictor names.
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Fit rolling OLS.
    pub fn fit(&self, x: &[Vec<f64>], y: &[f64]) -> Result<RollingOlsResult> {
        let n = y.len();
        if x.len() != n {
            return Err(InferustError::DimensionMismatch {
                x_rows: x.len(),
                y_len: n,
            });
        }
        if self.window < 2 {
            return Err(InferustError::InvalidInput(
                "window must be at least 2".into(),
            ));
        }
        if n < self.window {
            return Err(InferustError::InsufficientData {
                needed: self.window,
                got: n,
            });
        }
        let p = x[0].len();
        let ncols = if self.add_intercept { p + 1 } else { p };
        if self.window <= ncols {
            return Err(InferustError::InvalidInput(format!(
                "window ({}) must be larger than number of parameters ({})",
                self.window, ncols
            )));
        }

        let n_windows = n - self.window + 1;
        let mut coefficients = Vec::with_capacity(n_windows);
        let mut std_errors = Vec::with_capacity(n_windows);
        let mut r_squared = Vec::with_capacity(n_windows);
        let mut mse_resid = Vec::with_capacity(n_windows);

        for start in 0..n_windows {
            let end = start + self.window;
            let x_w = &x[start..end];
            let y_w = &y[start..end];

            let x_mat = build_x(x_w, self.window, p, self.add_intercept);
            let y_vec = DVector::from_column_slice(y_w);

            let xtx = x_mat.transpose() * &x_mat;
            let xty = x_mat.transpose() * &y_vec;
            let xtx_inv = match xtx.try_inverse() {
                Some(inv) => inv,
                None => {
                    // Singular window — fill with NaN
                    coefficients.push(vec![f64::NAN; ncols]);
                    std_errors.push(vec![f64::NAN; ncols]);
                    r_squared.push(f64::NAN);
                    mse_resid.push(f64::NAN);
                    continue;
                }
            };
            let beta = &xtx_inv * &xty;
            let fitted: DVector<f64> = &x_mat * &beta;
            let resids: DVector<f64> = &y_vec - &fitted;
            let df = self.window - ncols;
            let sigma2 = resids.dot(&resids) / df as f64;
            let var_beta = &xtx_inv * sigma2;
            let se: Vec<f64> = (0..ncols).map(|j| var_beta[(j, j)].abs().sqrt()).collect();

            let y_mean = y_w.iter().sum::<f64>() / self.window as f64;
            let sst = y_w
                .iter()
                .map(|yi| (yi - y_mean).powi(2))
                .sum::<f64>()
                .max(f64::EPSILON);
            let ssr = resids.dot(&resids);
            let r2 = (1.0 - ssr / sst).max(0.0);

            coefficients.push(beta.iter().copied().collect());
            std_errors.push(se);
            r_squared.push(r2);
            mse_resid.push(sigma2);
        }

        let feat = make_feat_names(&self.feature_names, p, self.add_intercept);
        Ok(RollingOlsResult {
            coefficients,
            std_errors,
            r_squared,
            mse_resid,
            window: self.window,
            n_windows,
            feature_names: feat,
        })
    }
}

impl RollingOlsResult {
    /// Convenience: extract the slope (first non-intercept coefficient) path.
    ///
    /// Returns the second column if an intercept is present, or the first column otherwise.
    pub fn slopes(&self) -> Vec<f64> {
        let col = if self.feature_names.first().is_some_and(|n| n == "const") {
            1
        } else {
            0
        };
        self.coefficients
            .iter()
            .map(|c| c.get(col).copied().unwrap_or(f64::NAN))
            .collect()
    }

    /// Print a brief summary of coefficient paths.
    pub fn print_summary(&self) {
        println!();
        println!(
            "── Rolling OLS (window = {}, {} windows) ──",
            self.window, self.n_windows
        );
        println!("{:>8} {:>12} {:>10}", "Window", "Const/Slope", "R²");
        println!("{}", "─".repeat(32));
        for (i, coefs) in self.coefficients.iter().enumerate() {
            println!(
                "{:>8} {:>12.4} {:>10.4}",
                i + 1,
                coefs.last().copied().unwrap_or(f64::NAN),
                self.r_squared[i]
            );
        }
        println!();
    }
}

// ── Recursive OLS ─────────────────────────────────────────────────────────────

/// Recursive OLS using the Sherman-Morrison rank-1 covariance update.
///
/// Fits OLS progressively from an initial batch of `init_obs` observations, then
/// adds one observation at a time.  At each step it records:
/// - the coefficient vector
/// - the recursive residual (one-step-ahead prediction error, scaled)
/// - the CUSUM statistic (cumulative sum of scaled recursive residuals)
///
/// The **CUSUM test** (Brown, Durbin & Evans 1975) rejects parameter stability
/// if the CUSUM path crosses the 5 % significance boundaries ±c·√(T−k) where
/// c ≈ 0.948.
#[derive(Debug, Clone)]
pub struct RecursiveOls {
    add_intercept: bool,
    feature_names: Vec<String>,
    init_obs: Option<usize>,
}

/// Fitted recursive OLS result.
#[derive(Debug, Clone)]
pub struct RecursiveOlsResult {
    /// Coefficient path: `coefficients[t]` = β estimated after observing the t-th data point.
    /// The first `init_obs - 1` entries contain the initial OLS fit replicated.
    pub coefficients: Vec<Vec<f64>>,
    /// Recursive residuals w_t = (y_t − x_t′β_{t-1}) / sqrt(1 + x_t′P_{t-1}x_t).
    /// NaN for the first `init_obs` observations (used for initialisation).
    pub recursive_residuals: Vec<f64>,
    /// CUSUM statistic path W_t = Σ w_s / σ̂.
    pub cusum: Vec<f64>,
    /// 5 % significance boundaries at each time step: ±boundary[t].
    pub cusum_boundary: Vec<f64>,
    /// Estimated residual standard deviation.
    pub sigma: f64,
    /// Number of observations used for initialisation.
    pub init_obs: usize,
    /// Total observations.
    pub n: usize,
    /// Feature names.
    pub feature_names: Vec<String>,
}

impl Default for RecursiveOls {
    fn default() -> Self {
        Self::new()
    }
}

impl RecursiveOls {
    /// Create a recursive OLS builder.
    pub fn new() -> Self {
        Self {
            add_intercept: true,
            feature_names: Vec::new(),
            init_obs: None,
        }
    }

    /// Do not add an intercept column.
    pub fn no_intercept(mut self) -> Self {
        self.add_intercept = false;
        self
    }

    /// Override the number of observations used for the initial OLS estimate
    /// (default = number of parameters + 1).
    pub fn init_obs(mut self, k: usize) -> Self {
        self.init_obs = Some(k);
        self
    }

    /// Set human-readable predictor names.
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Fit recursive OLS.
    pub fn fit(&self, x: &[Vec<f64>], y: &[f64]) -> Result<RecursiveOlsResult> {
        let n = y.len();
        if x.len() != n {
            return Err(InferustError::DimensionMismatch {
                x_rows: x.len(),
                y_len: n,
            });
        }
        let p = x[0].len();
        let ncols = if self.add_intercept { p + 1 } else { p };
        let k0 = self.init_obs.unwrap_or(ncols + 1).max(ncols + 1);
        if n < k0 + 1 {
            return Err(InferustError::InsufficientData {
                needed: k0 + 1,
                got: n,
            });
        }

        // Initial OLS on first k0 observations
        let x_init = build_x(&x[..k0], k0, p, self.add_intercept);
        let y_init = DVector::from_column_slice(&y[..k0]);
        let xtx = x_init.transpose() * &x_init;
        let xty = x_init.transpose() * &y_init;
        let mut p_mat = xtx.try_inverse().ok_or(InferustError::SingularMatrix)?;
        let mut beta = &p_mat * &xty;

        let fitted_init: DVector<f64> = &x_init * &beta;
        let resid_init: DVector<f64> = &y_init - &fitted_init;
        let sigma2_est = resid_init.dot(&resid_init) / (k0 - ncols).max(1) as f64;
        let sigma = sigma2_est.sqrt().max(f64::EPSILON);

        let mut coef_path: Vec<Vec<f64>> = vec![beta.iter().copied().collect(); k0];
        let mut rec_resids: Vec<f64> = vec![f64::NAN; k0];
        let mut cusum: Vec<f64> = vec![0.0; k0];
        let mut cusum_sum = 0.0_f64;

        // Recursive update for t = k0 .. n-1
        for t in k0..n {
            let xt = DVector::from_fn(ncols, |r, _| {
                if self.add_intercept {
                    if r == 0 {
                        1.0
                    } else {
                        x[t][r - 1]
                    }
                } else {
                    x[t][r]
                }
            });

            // Innovation and scaling
            let innov = y[t] - beta.dot(&xt);
            let ft = 1.0 + xt.dot(&(&p_mat * &xt));
            let w = innov / (sigma * ft.sqrt().max(f64::EPSILON));

            // Sherman-Morrison update
            let k_gain = &p_mat * &xt / ft;
            beta += &k_gain * innov;
            p_mat -= &k_gain * xt.transpose() * &p_mat;

            cusum_sum += w;
            rec_resids.push(w);
            coef_path.push(beta.iter().copied().collect());
            cusum.push(cusum_sum);
        }

        // CUSUM 5% boundaries: ±0.948 * sqrt(T - k) at each step
        let t_total = (n - k0) as f64;
        // Simpler symmetric boundary: ±0.948 * sqrt(n - k0)
        let boundary_val = 0.948 * t_total.sqrt();
        let cusum_boundary: Vec<f64> = (0..n).map(|_| boundary_val).collect();

        let feat = make_feat_names(&self.feature_names, p, self.add_intercept);
        Ok(RecursiveOlsResult {
            coefficients: coef_path,
            recursive_residuals: rec_resids,
            cusum,
            cusum_boundary,
            sigma,
            init_obs: k0,
            n,
            feature_names: feat,
        })
    }
}

impl RecursiveOlsResult {
    /// Return `true` if the CUSUM statistic ever exceeds the 5 % boundary.
    pub fn cusum_reject(&self) -> bool {
        let boundary = self.cusum_boundary.last().copied().unwrap_or(f64::INFINITY);
        self.cusum.iter().any(|&c| c.abs() > boundary)
    }

    /// Print the CUSUM path and stability verdict.
    pub fn print_cusum(&self) {
        println!();
        println!("── Recursive OLS CUSUM Test ─────────────────────────────");
        println!(
            "  n = {}   init_obs = {}   σ̂ = {:.6}",
            self.n, self.init_obs, self.sigma
        );
        println!(
            "  5% boundary = ±{:.4}",
            self.cusum_boundary.last().copied().unwrap_or(0.0)
        );
        println!();
        println!("{:>6}  {:>10}  {:>10}", "t", "CUSUM", "Boundary");
        println!("{}", "─".repeat(30));
        for (t, (&c, &b)) in self
            .cusum
            .iter()
            .zip(self.cusum_boundary.iter())
            .enumerate()
        {
            if t < self.init_obs {
                continue;
            }
            let flag = if c.abs() > b { " *** " } else { "" };
            println!("{:>6}  {:>10.4}  {:>10.4}{}", t, c, b, flag);
        }
        let verdict = if self.cusum_reject() {
            "REJECT parameter stability (CUSUM exceeds 5% boundary)"
        } else {
            "Fail to reject parameter stability"
        };
        println!("  Verdict: {verdict}");
        println!();
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn build_x(x: &[Vec<f64>], n: usize, p: usize, add_intercept: bool) -> DMatrix<f64> {
    let ncols = if add_intercept { p + 1 } else { p };
    DMatrix::from_fn(n, ncols, |r, c| {
        if add_intercept {
            if c == 0 {
                1.0
            } else {
                x[r][c - 1]
            }
        } else {
            x[r][c]
        }
    })
}

fn make_feat_names(names: &[String], p: usize, add_intercept: bool) -> Vec<String> {
    let ncols = if add_intercept { p + 1 } else { p };
    let mut out = Vec::with_capacity(ncols);
    if add_intercept {
        out.push("const".to_string());
    }
    if names.len() == p {
        out.extend_from_slice(names);
    } else {
        for i in 1..=p {
            out.push(format!("x{i}"));
        }
    }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{RecursiveOls, RollingOls};

    #[test]
    fn rolling_ols_correct_window_count() {
        let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..20).map(|i| 2.0 * i as f64 + 1.0).collect();
        let res = RollingOls::new(8).fit(&x, &y).unwrap();
        assert_eq!(res.n_windows, 13); // 20 - 8 + 1
        assert_eq!(res.coefficients.len(), 13);
    }

    #[test]
    fn rolling_ols_stable_slope_for_linear_data() {
        let x: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..30).map(|i| 3.0 * i as f64 + 5.0).collect();
        let res = RollingOls::new(10).fit(&x, &y).unwrap();
        for slope in res.slopes() {
            assert!((slope - 3.0).abs() < 0.01, "slope = {slope:.4}");
        }
    }

    #[test]
    fn recursive_ols_coefficient_length() {
        let x: Vec<Vec<f64>> = (0..25).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..25).map(|i| 2.0 * i as f64 + 1.0).collect();
        let res = RecursiveOls::new().fit(&x, &y).unwrap();
        assert_eq!(res.coefficients.len(), 25);
        assert_eq!(res.cusum.len(), 25);
    }

    #[test]
    fn recursive_ols_stable_linear_data_passes_cusum() {
        let x: Vec<Vec<f64>> = (0..40).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..40).map(|i| 2.5 * i as f64 + 0.5).collect();
        let res = RecursiveOls::new().fit(&x, &y).unwrap();
        assert!(!res.cusum_reject(), "stable linear data should pass CUSUM");
    }
}
