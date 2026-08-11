//! Generalized Additive Model starters.
//!
//! The first supported model is a Gaussian additive regression using truncated
//! power spline bases. The default path is unpenalized OLS on the expanded
//! design; optional identity penalties on the truncated-power knot columns
//! support fixed-λ or GCV smoothing.

use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ContinuousCDF, FisherSnedecor, StudentsT};

use crate::error::{InferustError, Result};
use crate::regression::{Ols, OlsCovariance, OlsResult};

/// Log₁₀ grid used for GCV λ search (matches `scripts/parity_statsmodels.py`).
const GCV_LOG10_LAMBDA_MIN: f64 = -8.0;
const GCV_LOG10_LAMBDA_MAX: f64 = 8.0;
const GCV_N_LAMBDAS: usize = 81;

/// One smooth term in a Gaussian additive model.
#[derive(Debug, Clone)]
pub struct SplineTerm {
    pub column: usize,
    pub knots: Vec<f64>,
    pub degree: usize,
    pub name: String,
}

impl SplineTerm {
    /// Create a cubic spline term for a predictor column.
    pub fn cubic(column: usize, knots: Vec<f64>) -> Self {
        Self {
            column,
            knots,
            degree: 3,
            name: format!("s(x{})", column + 1),
        }
    }

    /// Create a spline term with an explicit degree.
    pub fn new(column: usize, knots: Vec<f64>, degree: usize) -> Self {
        Self {
            column,
            knots,
            degree,
            name: format!("s(x{})", column + 1),
        }
    }

    /// Set the display name used in summaries.
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

/// Gaussian additive model builder.
///
/// By default the expanded truncated-power design is fit with unpenalized OLS.
/// Call [`GaussianGam::penalized`] or [`GaussianGam::smoothing`] to enable an
/// identity penalty on the truncated-power knot columns (polynomial basis
/// columns and the intercept remain unpenalized).
#[derive(Debug, Clone, Default)]
pub struct GaussianGam {
    terms: Vec<SplineTerm>,
    linear_columns: Vec<usize>,
    add_intercept: bool,
    /// `None` = unpenalized OLS. `Some(None)` = GCV for λ. `Some(Some(λ))` = fixed λ.
    smoothing: Option<Option<f64>>,
}

impl GaussianGam {
    pub fn new() -> Self {
        Self {
            terms: Vec::new(),
            linear_columns: Vec::new(),
            add_intercept: true,
            smoothing: None,
        }
    }

    /// Add an un-smoothed linear predictor column.
    pub fn linear(mut self, column: usize) -> Self {
        self.linear_columns.push(column);
        self
    }

    /// Add a smooth spline term.
    pub fn smooth(mut self, term: SplineTerm) -> Self {
        self.terms.push(term);
        self
    }

    /// Fit without an intercept.
    pub fn no_intercept(mut self) -> Self {
        self.add_intercept = false;
        self
    }

    /// Enable the knot-column penalty and select λ by GCV.
    pub fn penalized(self) -> Self {
        self.smoothing(None)
    }

    /// Enable the knot-column penalty with a fixed λ, or `None` to select λ by GCV.
    pub fn smoothing(mut self, lambda: Option<f64>) -> Self {
        self.smoothing = Some(lambda);
        self
    }

    /// Fit the additive model to raw predictor rows and response values.
    pub fn fit(&self, x: &[Vec<f64>], y: &[f64]) -> Result<GamResult> {
        validate_inputs(x, y, &self.terms, &self.linear_columns)?;
        let design = build_design(x, &self.terms, &self.linear_columns)?;

        match self.smoothing {
            None => {
                let mut builder = Ols::new().stable().with_feature_names(design.names.clone());
                if !self.add_intercept {
                    builder = builder.no_intercept();
                }
                let ols = builder.fit(&design.x, y)?;
                Ok(GamResult {
                    ols,
                    terms: self.terms.clone(),
                    linear_columns: self.linear_columns.clone(),
                    add_intercept: self.add_intercept,
                    design_feature_names: design.names,
                    lambda: None,
                    edf: None,
                    gcv: None,
                })
            }
            Some(lambda_opt) => {
                if let Some(lambda) = lambda_opt {
                    if !lambda.is_finite() || lambda < 0.0 {
                        return Err(InferustError::InvalidInput(
                            "smoothing lambda must be finite and non-negative".into(),
                        ));
                    }
                }
                fit_penalized(
                    &design,
                    y,
                    self.add_intercept,
                    lambda_opt,
                    self.terms.clone(),
                    self.linear_columns.clone(),
                )
            }
        }
    }
}

/// Fitted Gaussian additive model.
#[derive(Debug, Clone)]
pub struct GamResult {
    pub ols: OlsResult,
    pub terms: Vec<SplineTerm>,
    pub linear_columns: Vec<usize>,
    pub add_intercept: bool,
    pub design_feature_names: Vec<String>,
    /// Selected (or fixed) penalty strength; `None` on the unpenalized OLS path.
    pub lambda: Option<f64>,
    /// Effective degrees of freedom `tr(H)`; `None` on the unpenalized OLS path.
    pub edf: Option<f64>,
    /// Generalized cross-validation score at the selected λ; `None` on OLS / fixed-λ without GCV search.
    pub gcv: Option<f64>,
}

impl GamResult {
    /// Predict response means for new rows.
    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<f64>> {
        let design = build_design(x, &self.terms, &self.linear_columns)?;
        Ok(self.ols.predict(&design.x))
    }

    /// Print the underlying regression summary on the expanded spline basis.
    pub fn print_summary(&self) {
        self.ols.print_summary();
        if let (Some(lambda), Some(edf), Some(gcv)) = (self.lambda, self.edf, self.gcv) {
            println!(" λ (smoothing): {lambda:.6e}   edf: {edf:.4}   GCV: {gcv:.6e}");
        } else if let (Some(lambda), Some(edf)) = (self.lambda, self.edf) {
            println!(" λ (smoothing): {lambda:.6e}   edf: {edf:.4}");
        }
    }
}

#[derive(Debug, Clone)]
struct GamDesign {
    x: Vec<Vec<f64>>,
    names: Vec<String>,
    /// True for truncated-power knot columns (penalized under smoothing).
    knot_penalty: Vec<bool>,
}

fn validate_inputs(
    x: &[Vec<f64>],
    y: &[f64],
    terms: &[SplineTerm],
    linear_columns: &[usize],
) -> Result<()> {
    if x.len() != y.len() {
        return Err(InferustError::DimensionMismatch {
            x_rows: x.len(),
            y_len: y.len(),
        });
    }
    if x.is_empty() {
        return Err(InferustError::InsufficientData { needed: 1, got: 0 });
    }
    let p = x[0].len();
    if terms.is_empty() && linear_columns.is_empty() {
        return Err(InferustError::InvalidInput(
            "GAM needs at least one linear or smooth term".into(),
        ));
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
    for &column in linear_columns {
        if column >= p {
            return Err(InferustError::InvalidInput(format!(
                "linear column {column} is out of bounds for {p} predictors"
            )));
        }
    }
    for term in terms {
        if term.column >= p {
            return Err(InferustError::InvalidInput(format!(
                "smooth column {} is out of bounds for {p} predictors",
                term.column
            )));
        }
        if term.degree == 0 || term.degree > 5 {
            return Err(InferustError::InvalidInput(
                "spline degree must be between 1 and 5".into(),
            ));
        }
        if term.knots.iter().any(|k| !k.is_finite()) {
            return Err(InferustError::InvalidInput(
                "spline knots must be finite".into(),
            ));
        }
    }
    Ok(())
}

fn build_design(
    x: &[Vec<f64>],
    terms: &[SplineTerm],
    linear_columns: &[usize],
) -> Result<GamDesign> {
    if x.is_empty() {
        return Err(InferustError::InsufficientData { needed: 1, got: 0 });
    }
    let mut rows = vec![Vec::new(); x.len()];
    let mut names = Vec::new();
    let mut knot_penalty = Vec::new();

    for &column in linear_columns {
        names.push(format!("x{}", column + 1));
        knot_penalty.push(false);
        for (i, row) in x.iter().enumerate() {
            rows[i].push(row[column]);
        }
    }

    for term in terms {
        for power in 1..=term.degree {
            names.push(format!("{}^{}", term.name, power));
            knot_penalty.push(false);
            for (i, row) in x.iter().enumerate() {
                rows[i].push(row[term.column].powi(power as i32));
            }
        }
        for knot in &term.knots {
            names.push(format!("{}[>{:.3}]", term.name, knot));
            knot_penalty.push(true);
            for (i, row) in x.iter().enumerate() {
                rows[i].push((row[term.column] - knot).max(0.0).powi(term.degree as i32));
            }
        }
    }

    Ok(GamDesign {
        x: rows,
        names,
        knot_penalty,
    })
}

fn gcv_lambda_grid() -> Vec<f64> {
    let mut lambdas = Vec::with_capacity(GCV_N_LAMBDAS);
    if GCV_N_LAMBDAS == 1 {
        lambdas.push(10f64.powf(GCV_LOG10_LAMBDA_MIN));
        return lambdas;
    }
    let step = (GCV_LOG10_LAMBDA_MAX - GCV_LOG10_LAMBDA_MIN) / (GCV_N_LAMBDAS - 1) as f64;
    for i in 0..GCV_N_LAMBDAS {
        lambdas.push(10f64.powf(GCV_LOG10_LAMBDA_MIN + step * i as f64));
    }
    lambdas
}

fn fit_penalized(
    design: &GamDesign,
    y: &[f64],
    add_intercept: bool,
    lambda_opt: Option<f64>,
    terms: Vec<SplineTerm>,
    linear_columns: Vec<usize>,
) -> Result<GamResult> {
    let n = y.len();
    let p = design.names.len();
    let ncols = if add_intercept { p + 1 } else { p };
    if n <= ncols {
        return Err(InferustError::InsufficientData {
            needed: ncols + 1,
            got: n,
        });
    }

    let mut flat = Vec::with_capacity(n * ncols);
    for row in &design.x {
        if add_intercept {
            flat.push(1.0);
        }
        flat.extend_from_slice(row);
    }
    let x_mat = DMatrix::from_row_slice(n, ncols, &flat);
    let y_vec = DVector::from_column_slice(y);
    let xtx = x_mat.transpose() * &x_mat;
    let xty = x_mat.transpose() * &y_vec;

    let mut penalty_diag = vec![0.0; ncols];
    let offset = if add_intercept { 1 } else { 0 };
    for (j, &is_knot) in design.knot_penalty.iter().enumerate() {
        if is_knot {
            penalty_diag[offset + j] = 1.0;
        }
    }

    let (lambda, edf, gcv, beta) = if let Some(lambda) = lambda_opt {
        let (beta, edf, ssr) = solve_penalized(&xtx, &xty, &x_mat, y, &penalty_diag, lambda)?;
        let denom = (n as f64 - edf).max(1e-12);
        let gcv = n as f64 * ssr / (denom * denom);
        (lambda, edf, Some(gcv), beta)
    } else {
        let mut best: Option<(f64, f64, f64, DVector<f64>)> = None;
        for &lambda in &gcv_lambda_grid() {
            let Ok((beta, edf, ssr)) =
                solve_penalized(&xtx, &xty, &x_mat, y, &penalty_diag, lambda)
            else {
                continue;
            };
            if !(edf.is_finite() && ssr.is_finite()) || edf >= n as f64 - 1e-8 {
                continue;
            }
            let denom = n as f64 - edf;
            if denom <= 1e-12 {
                continue;
            }
            let gcv = n as f64 * ssr / (denom * denom);
            if !gcv.is_finite() {
                continue;
            }
            match &best {
                Some((_, _, best_gcv, _)) if gcv >= *best_gcv => {}
                _ => best = Some((lambda, edf, gcv, beta)),
            }
        }
        let (lambda, edf, gcv, beta) = best.ok_or(InferustError::SingularMatrix)?;
        (lambda, edf, Some(gcv), beta)
    };

    let ols = build_penalized_ols_result(
        &x_mat,
        &xtx,
        y,
        &beta,
        edf,
        &penalty_diag,
        lambda,
        add_intercept,
        &design.names,
    )?;

    Ok(GamResult {
        ols,
        terms,
        linear_columns,
        add_intercept,
        design_feature_names: design.names.clone(),
        lambda: Some(lambda),
        edf: Some(edf),
        gcv,
    })
}

fn solve_penalized(
    xtx: &DMatrix<f64>,
    xty: &DVector<f64>,
    x_mat: &DMatrix<f64>,
    y: &[f64],
    penalty_diag: &[f64],
    lambda: f64,
) -> Result<(DVector<f64>, f64, f64)> {
    let ncols = xtx.nrows();
    let mut a = xtx.clone();
    for j in 0..ncols {
        a[(j, j)] += lambda * penalty_diag[j];
    }
    let chol = a.cholesky().ok_or(InferustError::SingularMatrix)?;
    let beta = chol.solve(xty);
    let a_inv = chol.inverse();
    // edf = tr(H) = tr(A^{-1} X'X)
    let mut edf = 0.0;
    for i in 0..ncols {
        for j in 0..ncols {
            edf += a_inv[(i, j)] * xtx[(j, i)];
        }
    }
    let y_hat = x_mat * &beta;
    let ssr: f64 = (0..y.len())
        .map(|i| {
            let r = y[i] - y_hat[i];
            r * r
        })
        .sum();
    Ok((beta, edf, ssr))
}

#[allow(clippy::too_many_arguments)]
fn build_penalized_ols_result(
    x_mat: &DMatrix<f64>,
    xtx: &DMatrix<f64>,
    y: &[f64],
    beta: &DVector<f64>,
    edf: f64,
    penalty_diag: &[f64],
    lambda: f64,
    add_intercept: bool,
    design_names: &[String],
) -> Result<OlsResult> {
    let n = y.len();
    let ncols = beta.len();
    let k = if add_intercept { ncols - 1 } else { ncols };
    let y_hat = x_mat * beta;
    let fitted_values: Vec<f64> = y_hat.iter().cloned().collect();
    let residuals: Vec<f64> = (0..n).map(|i| y[i] - y_hat[i]).collect();
    let ssr: f64 = residuals.iter().map(|r| r * r).sum();
    let y_mean = y.iter().sum::<f64>() / n as f64;
    let sst: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
    let sse = sst - ssr;
    let r_squared = if sst == 0.0 { 1.0 } else { 1.0 - ssr / sst };

    let df_resid_f = (n as f64 - edf).max(1.0);
    let df_resid = df_resid_f.floor().max(1.0) as usize;
    let adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) as f64 / df_resid_f;
    let s2 = ssr / df_resid_f;

    let mut a = xtx.clone();
    for j in 0..ncols {
        a[(j, j)] += lambda * penalty_diag[j];
    }
    let a_inv = a.cholesky().ok_or(InferustError::SingularMatrix)?.inverse();
    // Frequentist sandwich: σ² A^{-1} X'X A^{-1}
    let cov_beta = s2 * (&a_inv * xtx * &a_inv);
    let std_errors: Vec<f64> = (0..ncols)
        .map(|i| cov_beta[(i, i)].max(0.0).sqrt())
        .collect();
    let covariance_matrix: Vec<Vec<f64>> = (0..ncols)
        .map(|i| (0..ncols).map(|j| cov_beta[(i, j)]).collect())
        .collect();
    let coefficients: Vec<f64> = beta.iter().cloned().collect();
    let t_statistics: Vec<f64> = coefficients
        .iter()
        .zip(std_errors.iter())
        .map(|(b, se)| if *se > 0.0 { b / se } else { f64::NAN })
        .collect();
    let t_dist = StudentsT::new(0.0, 1.0, df_resid as f64)
        .map_err(|_| InferustError::InvalidInput("invalid degrees of freedom".into()))?;
    let p_values: Vec<f64> = t_statistics
        .iter()
        .map(|&t| {
            if t.is_finite() {
                2.0 * (1.0 - t_dist.cdf(t.abs()))
            } else {
                f64::NAN
            }
        })
        .collect();

    let df_model = (edf - if add_intercept { 1.0 } else { 0.0 }).max(0.0);
    let f_statistic = if df_model > 0.0 && s2 > 0.0 {
        (sse / df_model) / s2
    } else {
        f64::NAN
    };
    let f_p_value = if f_statistic.is_nan() {
        f64::NAN
    } else {
        let f_dist = FisherSnedecor::new(df_model.max(1e-8), df_resid_f)
            .map_err(|_| InferustError::InvalidInput("invalid F distribution parameters".into()))?;
        1.0 - f_dist.cdf(f_statistic)
    };

    let sigma2_mle = ssr / n as f64;
    let log_lik = -0.5 * n as f64 * ((2.0 * std::f64::consts::PI * sigma2_mle).ln() + 1.0);
    let aic = -2.0 * log_lik + 2.0 * edf;
    let bic = -2.0 * log_lik + edf * (n as f64).ln();

    let mut feature_names = Vec::with_capacity(ncols);
    if add_intercept {
        feature_names.push("const".to_string());
    }
    feature_names.extend(design_names.iter().cloned());

    // Leverage from unpenalized formula is approximate; leave zeros for penalized.
    let leverage = vec![0.0; n];

    Ok(OlsResult {
        model_name: "GaussianGAM".to_string(),
        coefficients,
        std_errors,
        covariance_matrix,
        covariance: OlsCovariance::Nonrobust,
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
        condition_number: f64::NAN,
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

#[cfg(test)]
mod tests {
    use super::{GaussianGam, SplineTerm};

    #[test]
    fn gaussian_gam_fits_nonlinear_signal() {
        let x = (0..40).map(|i| vec![i as f64 / 10.0]).collect::<Vec<_>>();
        let y = x
            .iter()
            .map(|row| 1.0 + 0.5 * row[0] + (row[0] - 2.0).max(0.0).powi(3))
            .collect::<Vec<_>>();
        let result = GaussianGam::new()
            .smooth(SplineTerm::cubic(0, vec![2.0]).named("s(x)"))
            .fit(&x, &y)
            .unwrap();
        let pred = result.predict(&[vec![3.0]]).unwrap();
        assert!((pred[0] - (1.0 + 1.5 + 1.0)).abs() < 1e-6);
        assert!(result.ols.r_squared > 0.99);
        assert!(result.lambda.is_none());
        assert!(result.edf.is_none());
        assert!(result.gcv.is_none());
    }

    #[test]
    fn rejects_empty_model() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![1.0, 2.0, 3.0];
        assert!(GaussianGam::new().fit(&x, &y).is_err());
    }

    #[test]
    fn penalized_gcv_recovers_nonlinear_signal() {
        let n = 50;
        let x: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64 / 10.0]).collect();
        // High noise + many knots so unpenalized OLS overfits; GCV should shrink.
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let t = row[0];
                let signal = (t * 0.8).sin() + 0.3 * t;
                let noise = ((i as f64 * 17.0).sin()) * 1.2;
                signal + noise
            })
            .collect();
        let knots: Vec<f64> = (1..=12).map(|k| k as f64 * 0.35).collect();
        let true_signal: Vec<f64> = x
            .iter()
            .map(|row| (row[0] * 0.8).sin() + 0.3 * row[0])
            .collect();

        let ols = GaussianGam::new()
            .smooth(SplineTerm::cubic(0, knots.clone()).named("s(x)"))
            .fit(&x, &y)
            .unwrap();
        let pen = GaussianGam::new()
            .smooth(SplineTerm::cubic(0, knots).named("s(x)"))
            .penalized()
            .fit(&x, &y)
            .unwrap();

        let gcv = pen.gcv.expect("GCV score");
        let edf = pen.edf.expect("edf");
        let lambda = pen.lambda.expect("lambda");
        assert!(gcv.is_finite() && gcv > 0.0);
        assert!(edf.is_finite() && edf > 1.0);
        assert!(lambda.is_finite() && lambda > 0.0);
        // Penalty should reduce effective df below the full unpenalized column count.
        assert!(edf < ols.ols.coefficients.len() as f64);

        let mse = |fitted: &[f64]| -> f64 {
            fitted
                .iter()
                .zip(true_signal.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                / n as f64
        };
        let mse_pen = mse(&pen.ols.fitted_values);
        let mse_ols = mse(&ols.ols.fitted_values);
        assert!(
            mse_pen < mse_ols,
            "penalized MSE {mse_pen} should beat unpenalized {mse_ols}"
        );
    }

    #[test]
    fn fixed_smoothing_lambda_sets_fields() {
        let x = (0..40).map(|i| vec![i as f64 / 10.0]).collect::<Vec<_>>();
        let y = x
            .iter()
            .map(|row| 1.0 + 0.5 * row[0] + (row[0] - 2.0).max(0.0).powi(3))
            .collect::<Vec<_>>();
        let result = GaussianGam::new()
            .smooth(SplineTerm::cubic(0, vec![2.0]).named("s(x)"))
            .smoothing(Some(1.0))
            .fit(&x, &y)
            .unwrap();
        assert_eq!(result.lambda, Some(1.0));
        assert!(result.edf.unwrap() > 0.0);
        assert!(result.gcv.unwrap().is_finite());
    }
}
