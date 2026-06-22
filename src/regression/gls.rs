//! Generalised Least Squares (GLS) and Feasible GLS (Cochrane-Orcutt / Prais-Winsten).
//!
//! | Model | When to use |
//! |---|---|
//! | [`Gls`] | Error covariance matrix Ω is **known** (e.g. from domain knowledge or a prior stage). |
//! | [`Fgls`] | Ω is **unknown** but assumed to follow an AR(1) structure; estimated by Cochrane-Orcutt iteration with Prais-Winsten first-observation correction. |
//!
//! # Quick start
//! ```rust
//! use inferust::regression::Fgls;
//!
//! let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0],
//!              vec![6.0], vec![7.0], vec![8.0], vec![9.0], vec![10.0]];
//! let y = vec![2.1, 3.8, 6.3, 7.9, 10.2, 11.8, 14.1, 15.9, 18.2, 20.0];
//! let result = Fgls::new().fit(&x, &y).unwrap();
//! result.print_summary();
//! ```

use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ContinuousCDF, StudentsT};

use crate::error::{InferustError, Result};

// ── GLS ───────────────────────────────────────────────────────────────────────

/// Generalised Least Squares with a **known** error covariance matrix Ω.
///
/// Solves β̂ = (X′Ω⁻¹X)⁻¹ X′Ω⁻¹y via Cholesky factorisation of Ω.
///
/// For an **unknown** Ω with AR(1) errors, use [`Fgls`] instead.
#[derive(Debug, Clone, Default)]
pub struct Gls {
    add_intercept: bool,
    feature_names: Vec<String>,
}

/// Fitted GLS / FGLS result.
#[derive(Debug, Clone)]
pub struct GlsResult {
    /// Estimated coefficients (intercept first when `add_intercept = true`).
    pub coefficients: Vec<f64>,
    /// Standard errors of each coefficient.
    pub std_errors: Vec<f64>,
    /// t-statistics.
    pub t_statistics: Vec<f64>,
    /// Two-sided p-values.
    pub p_values: Vec<f64>,
    /// Coefficient of determination R² (on the transformed scale).
    pub r_squared: f64,
    /// Adjusted R².
    pub adj_r_squared: f64,
    /// Residual mean squared error (on the transformed scale).
    pub mse_resid: f64,
    /// Fitted values on the **original** scale.
    pub fitted_values: Vec<f64>,
    /// Residuals on the original scale.
    pub residuals: Vec<f64>,
    /// Number of observations.
    pub n: usize,
    /// Number of predictors (excluding intercept).
    pub k: usize,
    /// Residual degrees of freedom.
    pub df_resid: usize,
    /// Feature names (including "const" when intercept is present).
    pub feature_names: Vec<String>,
    /// Estimated AR(1) coefficient ρ (0.0 for plain GLS).
    pub rho: f64,
    /// Method string shown in `print_summary`.
    pub method: String,
}

impl Gls {
    /// Create a GLS builder.  By default an intercept is added.
    pub fn new() -> Self {
        Self {
            add_intercept: true,
            feature_names: Vec::new(),
        }
    }

    /// Do not add an intercept column.
    pub fn no_intercept(mut self) -> Self {
        self.add_intercept = false;
        self
    }

    /// Set human-readable names for the predictor columns.
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Fit GLS given predictor matrix `x`, response `y`, and the **n×n** error
    /// covariance matrix `omega` (row-major, length n²).
    ///
    /// `omega` must be symmetric positive-definite.
    pub fn fit(&self, x: &[Vec<f64>], y: &[f64], omega: &[f64]) -> Result<GlsResult> {
        let n = y.len();
        if x.len() != n {
            return Err(InferustError::DimensionMismatch {
                x_rows: x.len(),
                y_len: n,
            });
        }
        if omega.len() != n * n {
            return Err(InferustError::InvalidInput(format!(
                "omega must have n² = {} elements, got {}",
                n * n,
                omega.len()
            )));
        }
        let p = x[0].len();
        let omega_mat = DMatrix::from_row_slice(n, n, omega);
        let y_vec = DVector::from_column_slice(y);
        let x_mat = build_x_matrix(x, n, p, self.add_intercept);
        let ncols = x_mat.ncols();

        let chol = omega_mat
            .clone()
            .cholesky()
            .ok_or(InferustError::InvalidInput(
                "omega is not positive-definite".into(),
            ))?;

        // Ω⁻¹ y  and  Ω⁻¹ X
        let omega_inv_y = chol.solve(&y_vec);
        let omega_inv_x = chol.solve(&x_mat);

        let xto_x = x_mat.transpose() * &omega_inv_x;
        let xto_y = x_mat.transpose() * &omega_inv_y;

        let xto_x_chol = xto_x
            .clone()
            .cholesky()
            .ok_or(InferustError::SingularMatrix)?;
        let beta = xto_x_chol.solve(&xto_y);

        let beta_inv = xto_x
            .clone()
            .try_inverse()
            .ok_or(InferustError::SingularMatrix)?;

        // Residuals on original scale
        let fitted: Vec<f64> = (0..n).map(|i| row_dot(&x_mat, i, &beta)).collect();
        let resids: Vec<f64> = y
            .iter()
            .zip(fitted.iter())
            .map(|(yi, fi)| yi - fi)
            .collect();

        // σ² = e'Ω⁻¹e / df  (direct computation avoids cancellation errors)
        let e_vec = DVector::from_column_slice(&resids);
        let omega_inv_e = chol.solve(&e_vec);
        let ssr_t = e_vec.dot(&omega_inv_e);
        let df_resid = n.saturating_sub(ncols);
        let sigma2 = if df_resid > 0 {
            ssr_t.abs() / df_resid as f64
        } else {
            0.0
        };

        let var_beta = beta_inv * sigma2;
        let se: Vec<f64> = (0..ncols).map(|j| var_beta[(j, j)].abs().sqrt()).collect();
        let t_stat: Vec<f64> = beta
            .iter()
            .zip(se.iter())
            .map(|(b, s)| b / s.max(f64::EPSILON))
            .collect();
        let pv: Vec<f64> = if df_resid > 0 {
            let tdist = StudentsT::new(0.0, 1.0, df_resid as f64).unwrap();
            t_stat
                .iter()
                .map(|&t| 2.0 * (1.0 - tdist.cdf(t.abs())))
                .collect()
        } else {
            vec![1.0; ncols]
        };

        // R² (centred, on original scale)
        let y_mean = y.iter().sum::<f64>() / n as f64;
        let sst = y
            .iter()
            .map(|yi| (yi - y_mean).powi(2))
            .sum::<f64>()
            .max(f64::EPSILON);
        let ssr = resids.iter().map(|e| e * e).sum::<f64>();
        let r2 = 1.0 - ssr / sst;
        let adj_r2 = 1.0 - (1.0 - r2) * (n as f64 - 1.0) / df_resid.max(1) as f64;

        let feat = make_feature_names(&self.feature_names, p, self.add_intercept);

        Ok(GlsResult {
            coefficients: beta.iter().copied().collect(),
            std_errors: se,
            t_statistics: t_stat,
            p_values: pv,
            r_squared: r2.max(0.0),
            adj_r_squared: adj_r2,
            mse_resid: sigma2,
            fitted_values: fitted,
            residuals: resids,
            n,
            k: p,
            df_resid,
            feature_names: feat,
            rho: 0.0,
            method: "GLS".into(),
        })
    }
}

// ── FGLS (Cochrane-Orcutt / Prais-Winsten) ───────────────────────────────────

/// Feasible GLS for **AR(1) serially correlated errors**.
///
/// Estimates ρ from OLS residuals (Cochrane-Orcutt iteration) then applies the
/// Prais-Winsten transformation so the first observation is retained.
///
/// The iteration repeats until |Δρ| < `tolerance` or `max_iter` is reached.
#[derive(Debug, Clone)]
pub struct Fgls {
    add_intercept: bool,
    feature_names: Vec<String>,
    max_iter: usize,
    tolerance: f64,
}

impl Default for Fgls {
    fn default() -> Self {
        Self::new()
    }
}

impl Fgls {
    /// Create an FGLS builder with default settings (up to 50 iterations, tol 1e-8).
    pub fn new() -> Self {
        Self {
            add_intercept: true,
            feature_names: Vec::new(),
            max_iter: 50,
            tolerance: 1e-8,
        }
    }

    /// Do not add an intercept column.
    pub fn no_intercept(mut self) -> Self {
        self.add_intercept = false;
        self
    }

    /// Set human-readable names for the predictor columns.
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Override convergence tolerance (default 1e-8).
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Override maximum number of Cochrane-Orcutt iterations (default 50).
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Fit FGLS by iterating the Cochrane-Orcutt / Prais-Winsten procedure.
    pub fn fit(&self, x: &[Vec<f64>], y: &[f64]) -> Result<GlsResult> {
        let n = y.len();
        if x.len() != n {
            return Err(InferustError::DimensionMismatch {
                x_rows: x.len(),
                y_len: n,
            });
        }
        if n < 3 {
            return Err(InferustError::InsufficientData { needed: 3, got: n });
        }
        let p = x[0].len();

        // Initial OLS to get residuals
        let x_mat = build_x_matrix(x, n, p, self.add_intercept);
        let ncols = x_mat.ncols();
        let y_vec = DVector::from_column_slice(y);

        let mut resids = ols_residuals(&x_mat, &y_vec)?;
        let mut rho = estimate_rho(&resids);

        for _iter in 0..self.max_iter {
            let (y_t, x_t) = prais_winsten_transform(y, x, rho, self.add_intercept);
            let y_t_vec = DVector::from_column_slice(&y_t);
            let x_t_rows: Vec<Vec<f64>> = (0..y_t.len()).map(|i| x_t[i].clone()).collect();
            let xt_mat = DMatrix::from_fn(y_t.len(), ncols, |r, c| x_t_rows[r][c]);
            let beta_t = ols_beta(&xt_mat, &y_t_vec)?;
            resids = (0..n).map(|i| y[i] - row_dot(&x_mat, i, &beta_t)).collect();
            let rho_new = estimate_rho(&resids);
            if (rho_new - rho).abs() < self.tolerance {
                rho = rho_new;
                break;
            }
            rho = rho_new;
        }

        // Final fit on transformed data
        let (y_t, x_t) = prais_winsten_transform(y, x, rho, self.add_intercept);
        let nt = y_t.len();
        let y_t_vec = DVector::from_column_slice(&y_t);
        let x_t_rows: Vec<Vec<f64>> = (0..nt).map(|i| x_t[i].clone()).collect();
        let xt_mat = DMatrix::from_fn(nt, ncols, |r, c| x_t_rows[r][c]);

        let xtx = xt_mat.transpose() * &xt_mat;
        let xty = xt_mat.transpose() * &y_t_vec;
        let xtx_inv = xtx
            .clone()
            .try_inverse()
            .ok_or(InferustError::SingularMatrix)?;
        let beta = &xtx_inv * &xty;

        let fitted_t: Vec<f64> = (0..nt).map(|i| row_dot(&xt_mat, i, &beta)).collect();
        let resids_t: Vec<f64> = y_t
            .iter()
            .zip(fitted_t.iter())
            .map(|(yi, fi)| yi - fi)
            .collect();
        let df_resid = nt.saturating_sub(ncols);
        let sigma2 = resids_t.iter().map(|e| e * e).sum::<f64>() / df_resid.max(1) as f64;
        let var_beta = &xtx_inv * sigma2;

        let se: Vec<f64> = (0..ncols).map(|j| var_beta[(j, j)].abs().sqrt()).collect();
        let t_stat: Vec<f64> = beta
            .iter()
            .zip(se.iter())
            .map(|(b, s)| b / s.max(f64::EPSILON))
            .collect();
        let pv: Vec<f64> = if df_resid > 0 {
            let tdist = StudentsT::new(0.0, 1.0, df_resid as f64).unwrap();
            t_stat
                .iter()
                .map(|&t| 2.0 * (1.0 - tdist.cdf(t.abs())))
                .collect()
        } else {
            vec![1.0; ncols]
        };

        // Fitted values / residuals back on original scale
        let fitted_orig: Vec<f64> = (0..n).map(|i| row_dot(&x_mat, i, &beta)).collect();
        let resids_orig: Vec<f64> = y
            .iter()
            .zip(fitted_orig.iter())
            .map(|(yi, fi)| yi - fi)
            .collect();

        let y_mean = y.iter().sum::<f64>() / n as f64;
        let sst = y
            .iter()
            .map(|yi| (yi - y_mean).powi(2))
            .sum::<f64>()
            .max(f64::EPSILON);
        let ssr = resids_orig.iter().map(|e| e * e).sum::<f64>();
        let r2 = (1.0 - ssr / sst).max(0.0);
        let adj_r2 = 1.0 - (1.0 - r2) * (n as f64 - 1.0) / df_resid.max(1) as f64;

        let feat = make_feature_names(&self.feature_names, p, self.add_intercept);

        Ok(GlsResult {
            coefficients: beta.iter().copied().collect(),
            std_errors: se,
            t_statistics: t_stat,
            p_values: pv,
            r_squared: r2,
            adj_r_squared: adj_r2,
            mse_resid: sigma2,
            fitted_values: fitted_orig,
            residuals: resids_orig,
            n,
            k: p,
            df_resid,
            feature_names: feat,
            rho,
            method: "FGLS (Cochrane-Orcutt / Prais-Winsten)".into(),
        })
    }
}

impl GlsResult {
    /// Print a statsmodels-style summary.
    pub fn print_summary(&self) {
        println!();
        println!("═══════════════════════════════════════════════════════════════════");
        println!("{:^67}", format!("{} Regression Results", self.method));
        println!("═══════════════════════════════════════════════════════════════════");
        println!(
            " n = {}   df_resid = {}   R² = {:.6}   Adj.R² = {:.6}",
            self.n, self.df_resid, self.r_squared, self.adj_r_squared
        );
        if self.rho != 0.0 {
            println!(" Estimated AR(1) ρ = {:.6}", self.rho);
        }
        println!("───────────────────────────────────────────────────────────────────");
        println!(
            "{:<22} {:>11} {:>11} {:>9} {:>10}",
            "Variable", "Coef", "Std Err", "t", "P>|t|"
        );
        println!("───────────────────────────────────────────────────────────────────");
        for i in 0..self.coefficients.len() {
            println!(
                "{:<22} {:>11.6} {:>11.6} {:>9.4} {:>10.6}  {}",
                self.feature_names[i],
                self.coefficients[i],
                self.std_errors[i],
                self.t_statistics[i],
                self.p_values[i],
                sig_stars(self.p_values[i])
            );
        }
        println!("═══════════════════════════════════════════════════════════════════");
        println!();
    }

    /// Predict for new observations (rows of `x_new`).
    pub fn predict(&self, x_new: &[Vec<f64>]) -> Vec<f64> {
        let has_const = self.feature_names.first().is_some_and(|n| n == "const");
        x_new
            .iter()
            .map(|row| {
                let offset = if has_const { 1 } else { 0 };
                let mut p = if has_const { self.coefficients[0] } else { 0.0 };
                for (j, &xj) in row.iter().enumerate() {
                    if offset + j < self.coefficients.len() {
                        p += self.coefficients[offset + j] * xj;
                    }
                }
                p
            })
            .collect()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn build_x_matrix(x: &[Vec<f64>], n: usize, p: usize, add_intercept: bool) -> DMatrix<f64> {
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

fn ols_residuals(x: &DMatrix<f64>, y: &DVector<f64>) -> Result<Vec<f64>> {
    let beta = ols_beta(x, y)?;
    let fitted: DVector<f64> = x * &beta;
    Ok((y - fitted).iter().copied().collect())
}

fn ols_beta(x: &DMatrix<f64>, y: &DVector<f64>) -> Result<DVector<f64>> {
    let xtx = x.transpose() * x;
    let xty = x.transpose() * y;
    let xtx_inv = xtx.try_inverse().ok_or(InferustError::SingularMatrix)?;
    Ok(xtx_inv * xty)
}

fn row_dot(x: &DMatrix<f64>, row: usize, beta: &DVector<f64>) -> f64 {
    (0..x.ncols()).map(|col| x[(row, col)] * beta[col]).sum()
}

fn estimate_rho(resids: &[f64]) -> f64 {
    let n = resids.len();
    if n < 2 {
        return 0.0;
    }
    let num: f64 = resids
        .iter()
        .skip(1)
        .zip(resids.iter())
        .map(|(a, b)| a * b)
        .sum();
    let den: f64 = resids.iter().map(|e| e * e).sum::<f64>().max(f64::EPSILON);
    (num / den).clamp(-0.999, 0.999)
}

/// Prais-Winsten transformation.
/// Returns (y_transformed, x_transformed) where the first row uses the
/// sqrt(1-ρ²) multiplier and subsequent rows use the Cochrane-Orcutt quasi-difference.
fn prais_winsten_transform(
    y: &[f64],
    x: &[Vec<f64>],
    rho: f64,
    add_intercept: bool,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = y.len();
    let p = x[0].len();
    let ncols = if add_intercept { p + 1 } else { p };
    let scale = (1.0 - rho * rho).sqrt();

    let mut yt = Vec::with_capacity(n);
    let mut xt: Vec<Vec<f64>> = Vec::with_capacity(n);

    // First observation (Prais-Winsten)
    yt.push(y[0] * scale);
    let row0: Vec<f64> = (0..ncols)
        .map(|c| {
            if add_intercept {
                if c == 0 {
                    scale
                } else {
                    x[0][c - 1] * scale
                }
            } else {
                x[0][c] * scale
            }
        })
        .collect();
    xt.push(row0);

    // Subsequent observations (Cochrane-Orcutt quasi-difference)
    for t in 1..n {
        yt.push(y[t] - rho * y[t - 1]);
        let row: Vec<f64> = (0..ncols)
            .map(|c| {
                if add_intercept {
                    if c == 0 {
                        1.0 - rho
                    } else {
                        x[t][c - 1] - rho * x[t - 1][c - 1]
                    }
                } else {
                    x[t][c] - rho * x[t - 1][c]
                }
            })
            .collect();
        xt.push(row);
    }
    (yt, xt)
}

fn make_feature_names(names: &[String], p: usize, add_intercept: bool) -> Vec<String> {
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{Fgls, Gls};

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "expected ≈{b:.6} got {a:.6}");
    }

    #[test]
    fn gls_identity_matches_ols() {
        // With Ω = I, GLS should give same coefficients as OLS
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        let y = vec![2.1, 3.9, 6.2, 7.8, 10.1];
        let n = y.len();
        let mut omega = vec![0.0f64; n * n];
        for i in 0..n {
            omega[i * n + i] = 1.0;
        }
        let res = Gls::new().fit(&x, &y, &omega).unwrap();
        // Slope should be close to ~2
        assert!(
            res.coefficients[1] > 1.5 && res.coefficients[1] < 2.5,
            "slope = {}",
            res.coefficients[1]
        );
    }

    #[test]
    fn fgls_converges_on_ar1_data() {
        // Generate AR(1) residual data: y = 2x + ε, ε_t = 0.7 ε_{t-1} + u
        let n = 30;
        let x: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
        let mut eps = vec![0.0f64; n];
        for t in 1..n {
            eps[t] = 0.7 * eps[t - 1] + (t as f64 * 0.1).sin();
        }
        let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * i as f64 + eps[i]).collect();
        let res = Fgls::new().fit(&x, &y).unwrap();
        assert!(res.coefficients.len() == 2);
        assert!(res.rho.abs() < 1.0, "rho = {}", res.rho);
        // slope should be close to 2
        assert_close(res.coefficients[1], 2.0, 0.2);
    }

    #[test]
    fn fgls_rho_reflects_ar1_structure() {
        let n = 40;
        let x: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
        let mut eps = vec![0.0f64; n];
        for t in 1..n {
            eps[t] = 0.8 * eps[t - 1] + ((t * 7) as f64 * 0.3).sin() * 0.5;
        }
        let y: Vec<f64> = (0..n).map(|i| 1.0 + i as f64 + eps[i]).collect();
        let res = Fgls::new().fit(&x, &y).unwrap();
        // The fitted residual path should show substantial AR(1) structure.
        assert!(
            res.rho.abs() > 0.3,
            "expected substantial rho, got {:.4}",
            res.rho
        );
    }
}
