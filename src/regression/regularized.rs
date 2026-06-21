//! Ridge, Lasso, and Elastic Net regularized linear regression.
//!
//! Mirrors `statsmodels.regression.linear_model.OLS.fit_regularized` and
//! scikit-learn's `Ridge` / `Lasso` / `ElasticNet`. The fitted objective is:
//!
//! ```text
//! (1 / 2n) * ||y - X*beta||^2  +  alpha * [ l1_ratio * ||beta||_1
//!                                            + (1 - l1_ratio) / 2 * ||beta||_2^2 ]
//! ```
//!
//! `l1_ratio = 0` is pure ridge, solved in closed form. `l1_ratio = 1` is
//! pure lasso, and anything in between is elastic net; both of those are
//! solved by cyclical coordinate descent with soft-thresholding (Friedman,
//! Hastie & Tibshirani, 2010) — the same algorithm `statsmodels`' own
//! `fit_regularized` and scikit-learn use, so for a strictly convex objective
//! (`alpha > 0`) both converge to the same unique minimizer.
//!
//! Unlike `statsmodels.OLS.fit_regularized`'s default behavior with a scalar
//! `alpha` (which penalizes every column of the design matrix, including any
//! constant the caller added), inferust never penalizes the intercept — the
//! standard convention used by scikit-learn and glmnet. To reproduce
//! inferust's defaults in statsmodels, pass `alpha` as a per-column array
//! with `0` in the intercept's slot.

use nalgebra::{DMatrix, DVector};

use crate::error::{InferustError, Result};

/// Shared output of [`Ridge::fit`], [`Lasso::fit`], and [`ElasticNet::fit`].
#[derive(Debug, Clone)]
pub struct RegularizedResult {
    /// Estimated coefficients (intercept first, named `"const"`, if the
    /// model includes one).
    pub coefficients: Vec<f64>,
    /// Names parallel to `coefficients`.
    pub feature_names: Vec<String>,
    /// In-sample fitted values `X * beta`.
    pub fitted_values: Vec<f64>,
    /// `y - fitted_values`.
    pub residuals: Vec<f64>,
    /// Unpenalized R-squared of the fit.
    pub r_squared: f64,
    /// Number of observations.
    pub n: usize,
    /// Number of predictors (excluding intercept).
    pub k: usize,
    /// Overall regularization strength used.
    pub alpha: f64,
    /// L1/L2 mixing weight used (`0` = ridge, `1` = lasso).
    pub l1_ratio: f64,
    /// Coordinate-descent sweeps used (`1` for the closed-form ridge path).
    pub n_iter: usize,
    /// Whether coordinate descent converged within `max_iter` (always `true`
    /// for the closed-form ridge path).
    pub converged: bool,
}

impl RegularizedResult {
    /// Print a compact coefficient table.
    pub fn print_summary(&self) {
        println!(
            "Regularized regression (alpha = {}, l1_ratio = {})",
            self.alpha, self.l1_ratio
        );
        println!("{:<12} {:>12}", "term", "coef");
        for (name, coef) in self.feature_names.iter().zip(self.coefficients.iter()) {
            println!("{name:<12} {coef:>12.6}");
        }
        println!("R-squared: {:.6}", self.r_squared);
        if !self.converged {
            println!(
                "warning: coordinate descent did not converge within {} iterations",
                self.n_iter
            );
        }
    }

    /// Predict on new rows (raw feature rows, without an intercept column).
    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<f64>> {
        let has_intercept = self.feature_names.first().map(|n| n == "const").unwrap_or(false);
        let offset = usize::from(has_intercept);
        x.iter()
            .map(|row| {
                if row.len() != self.k {
                    return Err(InferustError::DimensionMismatch {
                        x_rows: row.len(),
                        y_len: self.k,
                    });
                }
                let mut sum = if has_intercept { self.coefficients[0] } else { 0.0 };
                for (j, &v) in row.iter().enumerate() {
                    sum += self.coefficients[offset + j] * v;
                }
                Ok(sum)
            })
            .collect()
    }
}

/// Ridge regression: closed-form L2-penalized least squares.
///
/// # Example
/// ```
/// use inferust::regression::Ridge;
///
/// let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
/// let y = vec![2.1, 3.9, 6.2, 7.8, 10.1];
/// let result = Ridge::new(0.5).fit(&x, &y).unwrap();
/// assert_eq!(result.coefficients.len(), 2); // intercept + slope
/// ```
#[derive(Debug, Clone)]
pub struct Ridge {
    alpha: f64,
    add_intercept: bool,
    feature_names: Vec<String>,
}

impl Ridge {
    /// Create a ridge builder with the given overall penalty strength.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            add_intercept: true,
            feature_names: Vec::new(),
        }
    }

    /// Supply custom feature names (must match the number of predictor
    /// columns in `x`, excluding the intercept).
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
    pub fn fit(&self, x: &[Vec<f64>], y: &[f64]) -> Result<RegularizedResult> {
        if !self.alpha.is_finite() || self.alpha < 0.0 {
            return Err(InferustError::InvalidInput(
                "alpha must be a finite, non-negative number".to_string(),
            ));
        }
        let (design, n, ncols) = build_design(x, y, self.add_intercept, &self.feature_names)?;
        let k = if self.add_intercept { ncols - 1 } else { ncols };

        let xtx = design.transpose() * &design;
        let y_vec = DVector::from_column_slice(y);
        let xty = design.transpose() * &y_vec;

        let lambda = self.alpha * n as f64;
        let mut penalized = xtx;
        for j in 0..ncols {
            if self.add_intercept && j == 0 {
                continue;
            }
            penalized[(j, j)] += lambda;
        }

        let beta = penalized
            .cholesky()
            .ok_or(InferustError::SingularMatrix)?
            .solve(&xty);

        let coefficients: Vec<f64> = beta.iter().cloned().collect();
        let y_hat = &design * &beta;
        let fitted_values: Vec<f64> = y_hat.iter().cloned().collect();
        let residuals: Vec<f64> = (0..n).map(|i| y[i] - y_hat[i]).collect();
        let r_squared = r_squared_of(y, &residuals);
        let feature_names = build_feature_names(self.add_intercept, k, &self.feature_names);

        Ok(RegularizedResult {
            coefficients,
            feature_names,
            fitted_values,
            residuals,
            r_squared,
            n,
            k,
            alpha: self.alpha,
            l1_ratio: 0.0,
            n_iter: 1,
            converged: true,
        })
    }
}

/// Lasso regression: L1-penalized least squares via coordinate descent.
///
/// # Example
/// ```
/// use inferust::regression::Lasso;
///
/// let x = vec![vec![1.0, 0.1], vec![2.0, 0.2], vec![3.0, 0.1], vec![4.0, 0.3], vec![5.0, 0.2], vec![6.0, 0.4]];
/// let y = vec![2.1, 3.9, 6.2, 7.8, 10.1, 12.0];
/// let result = Lasso::new(0.1).fit(&x, &y).unwrap();
/// assert!(result.converged);
/// ```
#[derive(Debug, Clone)]
pub struct Lasso {
    alpha: f64,
    add_intercept: bool,
    max_iter: usize,
    tol: f64,
    feature_names: Vec<String>,
}

impl Lasso {
    /// Create a lasso builder with the given overall penalty strength.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            add_intercept: true,
            max_iter: 5000,
            tol: 1e-10,
            feature_names: Vec::new(),
        }
    }

    /// Supply custom feature names.
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Fit without an intercept.
    pub fn no_intercept(mut self) -> Self {
        self.add_intercept = false;
        self
    }

    /// Maximum number of coordinate-descent sweeps (default 5000).
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Convergence tolerance on the largest per-sweep coefficient change
    /// (default `1e-10`).
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Fit the model.
    pub fn fit(&self, x: &[Vec<f64>], y: &[f64]) -> Result<RegularizedResult> {
        fit_elastic_net(
            x,
            y,
            self.add_intercept,
            self.alpha,
            1.0,
            self.max_iter,
            self.tol,
            &self.feature_names,
        )
    }
}

/// Elastic Net regression: a convex combination of L1 and L2 penalties, via
/// coordinate descent.
///
/// `l1_ratio` of `0.0` reduces to ridge (though [`Ridge`] is faster, via its
/// closed form) and `1.0` reduces to lasso (equivalent to [`Lasso`]).
///
/// # Example
/// ```
/// use inferust::regression::ElasticNet;
///
/// let x = vec![vec![1.0, 0.1], vec![2.0, 0.2], vec![3.0, 0.1], vec![4.0, 0.3], vec![5.0, 0.2], vec![6.0, 0.4]];
/// let y = vec![2.1, 3.9, 6.2, 7.8, 10.1, 12.0];
/// let result = ElasticNet::new(0.1, 0.5).fit(&x, &y).unwrap();
/// assert!(result.converged);
/// ```
#[derive(Debug, Clone)]
pub struct ElasticNet {
    alpha: f64,
    l1_ratio: f64,
    add_intercept: bool,
    max_iter: usize,
    tol: f64,
    feature_names: Vec<String>,
}

impl ElasticNet {
    /// Create an elastic net builder. `l1_ratio` must lie in `[0, 1]`.
    pub fn new(alpha: f64, l1_ratio: f64) -> Self {
        Self {
            alpha,
            l1_ratio,
            add_intercept: true,
            max_iter: 5000,
            tol: 1e-10,
            feature_names: Vec::new(),
        }
    }

    /// Supply custom feature names.
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Fit without an intercept.
    pub fn no_intercept(mut self) -> Self {
        self.add_intercept = false;
        self
    }

    /// Maximum number of coordinate-descent sweeps (default 5000).
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Convergence tolerance on the largest per-sweep coefficient change
    /// (default `1e-10`).
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Fit the model.
    pub fn fit(&self, x: &[Vec<f64>], y: &[f64]) -> Result<RegularizedResult> {
        fit_elastic_net(
            x,
            y,
            self.add_intercept,
            self.alpha,
            self.l1_ratio,
            self.max_iter,
            self.tol,
            &self.feature_names,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn fit_elastic_net(
    x: &[Vec<f64>],
    y: &[f64],
    add_intercept: bool,
    alpha: f64,
    l1_ratio: f64,
    max_iter: usize,
    tol: f64,
    user_feature_names: &[String],
) -> Result<RegularizedResult> {
    if !alpha.is_finite() || alpha < 0.0 {
        return Err(InferustError::InvalidInput(
            "alpha must be a finite, non-negative number".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&l1_ratio) {
        return Err(InferustError::InvalidInput(
            "l1_ratio must be between 0 and 1".to_string(),
        ));
    }
    if max_iter == 0 {
        return Err(InferustError::InvalidInput(
            "max_iter must be at least 1".to_string(),
        ));
    }

    let (design, n, ncols) = build_design(x, y, add_intercept, user_feature_names)?;
    let k = if add_intercept { ncols - 1 } else { ncols };
    let n_f = n as f64;

    let l1 = alpha * l1_ratio;
    let l2 = alpha * (1.0 - l1_ratio);

    let col_sq: Vec<f64> = (0..ncols)
        .map(|j| design.column(j).iter().map(|v| v * v).sum::<f64>() / n_f)
        .collect();

    let mut beta = DVector::<f64>::zeros(ncols);
    let mut residual = DVector::from_column_slice(y);
    let mut converged = false;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        let mut max_change: f64 = 0.0;
        for j in 0..ncols {
            let penalized = !(add_intercept && j == 0);
            let xj = design.column(j);
            let old_bj = beta[j];
            let rho_j = xj.dot(&residual) / n_f + col_sq[j] * old_bj;
            let denom = (if penalized { col_sq[j] + l2 } else { col_sq[j] }).max(1e-12);
            let new_bj = if penalized && l1 > 0.0 {
                soft_threshold(rho_j, l1) / denom
            } else {
                rho_j / denom
            };
            let delta = new_bj - old_bj;
            if delta != 0.0 {
                for i in 0..n {
                    residual[i] -= xj[i] * delta;
                }
                beta[j] = new_bj;
                max_change = max_change.max(delta.abs());
            }
        }
        n_iter = iter + 1;
        if max_change < tol {
            converged = true;
            break;
        }
    }

    let coefficients: Vec<f64> = beta.iter().cloned().collect();
    let y_hat = &design * &beta;
    let fitted_values: Vec<f64> = y_hat.iter().cloned().collect();
    let residuals: Vec<f64> = (0..n).map(|i| y[i] - y_hat[i]).collect();
    let r_squared = r_squared_of(y, &residuals);
    let feature_names = build_feature_names(add_intercept, k, user_feature_names);

    Ok(RegularizedResult {
        coefficients,
        feature_names,
        fitted_values,
        residuals,
        r_squared,
        n,
        k,
        alpha,
        l1_ratio,
        n_iter,
        converged,
    })
}

fn soft_threshold(x: f64, lambda: f64) -> f64 {
    if x > lambda {
        x - lambda
    } else if x < -lambda {
        x + lambda
    } else {
        0.0
    }
}

fn build_design(
    x: &[Vec<f64>],
    y: &[f64],
    add_intercept: bool,
    user_feature_names: &[String],
) -> Result<(DMatrix<f64>, usize, usize)> {
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
    let p = x[0].len();
    if !user_feature_names.is_empty() && user_feature_names.len() != p {
        return Err(InferustError::InvalidInput(format!(
            "feature_names has {} entries but x has {p} columns",
            user_feature_names.len()
        )));
    }
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
                "all rows in X must have the same length".to_string(),
            ));
        }
        if add_intercept {
            design.push(1.0);
        }
        design.extend_from_slice(row);
    }

    Ok((DMatrix::from_row_slice(n, ncols, &design), n, ncols))
}

fn build_feature_names(add_intercept: bool, k: usize, user_names: &[String]) -> Vec<String> {
    let mut names = Vec::with_capacity(k + usize::from(add_intercept));
    if add_intercept {
        names.push("const".to_string());
    }
    if user_names.is_empty() {
        for i in 0..k {
            names.push(format!("x{}", i + 1));
        }
    } else {
        names.extend(user_names.iter().cloned());
    }
    names
}

fn r_squared_of(y: &[f64], residuals: &[f64]) -> f64 {
    let n = y.len() as f64;
    let y_mean = y.iter().sum::<f64>() / n;
    let sst: f64 = y.iter().map(|v| (v - y_mean).powi(2)).sum();
    let ssr: f64 = residuals.iter().map(|r| r * r).sum();
    if sst == 0.0 {
        1.0
    } else {
        1.0 - ssr / sst
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() <= tol,
            "expected {expected}, got {actual} (tol {tol})"
        );
    }

    fn fixture() -> (Vec<Vec<f64>>, Vec<f64>) {
        // y = 3 + 2*x1 - 1*x2 + small noise.
        let x = vec![
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![3.0, 3.0],
            vec![4.0, 2.0],
            vec![5.0, 4.0],
            vec![6.0, 1.0],
            vec![7.0, 3.0],
            vec![8.0, 5.0],
        ];
        let y = vec![5.1, 6.0, 6.9, 9.1, 9.9, 13.9, 14.0, 13.1];
        (x, y)
    }

    #[test]
    fn ridge_shrinks_coefficients_toward_zero() {
        let (x, y) = fixture();
        let unpenalized = Ridge::new(0.0).fit(&x, &y).unwrap();
        let penalized = Ridge::new(50.0).fit(&x, &y).unwrap();
        let unpenalized_norm: f64 = unpenalized.coefficients[1..].iter().map(|c| c * c).sum();
        let penalized_norm: f64 = penalized.coefficients[1..].iter().map(|c| c * c).sum();
        assert!(penalized_norm < unpenalized_norm);
    }

    #[test]
    fn ridge_matches_ols_at_alpha_zero() {
        use crate::regression::Ols;
        let (x, y) = fixture();
        let ridge = Ridge::new(0.0).fit(&x, &y).unwrap();
        let ols = Ols::new().fit(&x, &y).unwrap();
        for (a, b) in ridge.coefficients.iter().zip(ols.coefficients.iter()) {
            assert_close(*a, *b, 1e-8);
        }
    }

    #[test]
    fn lasso_can_zero_out_coefficients() {
        let (x, y) = fixture();
        let result = Lasso::new(5.0).fit(&x, &y).unwrap();
        assert!(result.converged);
        // A strong enough penalty should shrink at least one slope to exactly 0.
        let any_zero = result.coefficients[1..].iter().any(|&c| c == 0.0);
        assert!(any_zero, "expected at least one zeroed coefficient, got {:?}", result.coefficients);
    }

    #[test]
    fn lasso_matches_independently_verified_solution() {
        // Cross-checked against an independent Python coordinate-descent
        // implementation (same algorithm, separately written) for alpha=0.1,
        // l1_ratio=1.0 on this exact dataset; also matches statsmodels'
        // `OLS.fit_regularized` when the intercept's penalty is zeroed via
        // an alpha array (see module docs).
        let (x, y) = fixture();
        let result = Lasso::new(0.1).fit(&x, &y).unwrap();
        assert!(result.converged);
        assert_close(result.coefficients[0], 4.33146067, 1e-6);
        assert_close(result.coefficients[1], 1.60327715, 1e-6);
        assert_close(result.coefficients[2], -0.68426966, 1e-6);
    }

    #[test]
    fn elastic_net_matches_ridge_at_l1_ratio_zero() {
        let (x, y) = fixture();
        let ridge_like = ElasticNet::new(1.0, 0.0).fit(&x, &y).unwrap();
        let ridge = Ridge::new(1.0).fit(&x, &y).unwrap();
        for (a, b) in ridge_like.coefficients.iter().zip(ridge.coefficients.iter()) {
            assert_close(*a, *b, 1e-6);
        }
    }

    #[test]
    fn predict_matches_fitted_values_on_training_data() {
        let (x, y) = fixture();
        let result = Lasso::new(0.05).fit(&x, &y).unwrap();
        let preds = result.predict(&x).unwrap();
        for (p, f) in preds.iter().zip(result.fitted_values.iter()) {
            assert_close(*p, *f, 1e-8);
        }
    }

    #[test]
    fn no_intercept_drops_const_term() {
        let (x, y) = fixture();
        let result = Ridge::new(1.0).no_intercept().fit(&x, &y).unwrap();
        assert_eq!(result.coefficients.len(), 2);
        assert_eq!(result.feature_names[0], "x1");
    }

    #[test]
    fn rejects_negative_alpha() {
        let (x, y) = fixture();
        assert!(Ridge::new(-1.0).fit(&x, &y).is_err());
        assert!(Lasso::new(-1.0).fit(&x, &y).is_err());
    }

    #[test]
    fn rejects_invalid_l1_ratio() {
        let (x, y) = fixture();
        assert!(ElasticNet::new(0.1, 1.5).fit(&x, &y).is_err());
        assert!(ElasticNet::new(0.1, -0.1).fit(&x, &y).is_err());
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let (x, _) = fixture();
        let y = vec![1.0, 2.0, 3.0];
        assert!(Ridge::new(0.1).fit(&x, &y).is_err());
    }
}
