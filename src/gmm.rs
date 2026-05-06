//! Generalized Method of Moments and instrumental-variable estimators.
//!
//! This module starts with a linear IV/2SLS estimator, the most common GMM
//! entry point in statsmodels-style econometrics workflows.

use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ContinuousCDF, StudentsT};

use crate::error::{InferustError, Result};

/// Linear instrumental-variables regression fitted by two-stage least squares.
#[derive(Debug, Clone, Default)]
pub struct Iv2Sls {
    add_intercept: bool,
    feature_names: Vec<String>,
}

impl Iv2Sls {
    pub fn new() -> Self {
        Self {
            add_intercept: true,
            feature_names: Vec::new(),
        }
    }

    /// Fit without an intercept.
    pub fn no_intercept(mut self) -> Self {
        self.add_intercept = false;
        self
    }

    /// Set names for regressors in `x`.
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Fit IV regression.
    ///
    /// `x` contains included regressors, including any endogenous regressors.
    /// `instruments` is the full instrument matrix excluding the intercept; for
    /// models with exogenous controls, include those controls in `instruments`
    /// alongside excluded instruments.
    pub fn fit(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
        instruments: &[Vec<f64>],
    ) -> Result<IvRegressionResult> {
        validate_inputs(x, y, instruments, self.add_intercept, &self.feature_names)?;
        let x_mat = build_matrix(x, self.add_intercept);
        let z_mat = build_instrument_matrix(instruments, self.add_intercept);
        let y_vec = DVector::from_column_slice(y);
        let n = y.len();
        let k = x_mat.ncols();
        let l = z_mat.ncols();
        if l < k {
            return Err(InferustError::InvalidInput(format!(
                "model is underidentified: {l} instruments for {k} regressors"
            )));
        }

        let ztz_inv = invert(&(z_mat.transpose() * &z_mat))?;
        let xz = x_mat.transpose() * &z_mat;
        let ztx = z_mat.transpose() * &x_mat;
        let zty = z_mat.transpose() * &y_vec;
        let x_pz_x = &xz * &ztz_inv * &ztx;
        let x_pz_y = &xz * &ztz_inv * &zty;
        let x_pz_x_inv = invert(&x_pz_x)?;
        let beta = &x_pz_x_inv * x_pz_y;

        let fitted_vec = &x_mat * &beta;
        let residuals = (0..n).map(|i| y[i] - fitted_vec[i]).collect::<Vec<_>>();
        let ssr = residuals.iter().map(|e| e * e).sum::<f64>();
        let df_resid = n - k;
        let sigma2 = ssr / df_resid as f64;
        let cov = x_pz_x_inv * sigma2;
        let std_errors = (0..k)
            .map(|i| cov[(i, i)].max(0.0).sqrt())
            .collect::<Vec<_>>();
        let coefficients = beta.iter().copied().collect::<Vec<_>>();
        let t_statistics = coefficients
            .iter()
            .zip(std_errors.iter())
            .map(|(coef, se)| coef / se.max(f64::EPSILON))
            .collect::<Vec<_>>();
        let tdist = StudentsT::new(0.0, 1.0, df_resid as f64)
            .map_err(|_| InferustError::InvalidInput("invalid degrees of freedom".into()))?;
        let p_values = t_statistics
            .iter()
            .map(|t| 2.0 * (1.0 - tdist.cdf(t.abs())))
            .collect::<Vec<_>>();
        let y_mean = y.iter().sum::<f64>() / n as f64;
        let centered_tss = y.iter().map(|yi| (yi - y_mean).powi(2)).sum::<f64>();
        let r_squared = 1.0 - ssr / centered_tss.max(1e-12);
        let feature_names = feature_names(&self.feature_names, x[0].len(), self.add_intercept);

        Ok(IvRegressionResult {
            coefficients,
            std_errors,
            t_statistics,
            p_values,
            fitted_values: fitted_vec.iter().copied().collect(),
            residuals,
            r_squared,
            ssr,
            mse_resid: sigma2,
            n,
            k: if self.add_intercept { k - 1 } else { k },
            instruments: l,
            df_resid,
            overidentification_df: l.saturating_sub(k),
            feature_names,
        })
    }
}

/// Fitted IV regression result.
#[derive(Debug, Clone)]
pub struct IvRegressionResult {
    pub coefficients: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub t_statistics: Vec<f64>,
    pub p_values: Vec<f64>,
    pub fitted_values: Vec<f64>,
    pub residuals: Vec<f64>,
    pub r_squared: f64,
    pub ssr: f64,
    pub mse_resid: f64,
    pub n: usize,
    pub k: usize,
    pub instruments: usize,
    pub df_resid: usize,
    pub overidentification_df: usize,
    pub feature_names: Vec<String>,
}

impl IvRegressionResult {
    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<f64>> {
        let has_intercept = self
            .feature_names
            .first()
            .map(|name| name == "const")
            .unwrap_or(false);
        let x_mat = build_matrix(x, has_intercept);
        if x_mat.ncols() != self.coefficients.len() {
            return Err(InferustError::InvalidInput(format!(
                "expected {} columns after intercept handling, got {}",
                self.coefficients.len(),
                x_mat.ncols()
            )));
        }
        let beta = DVector::from_column_slice(&self.coefficients);
        Ok((x_mat * beta).iter().copied().collect())
    }

    pub fn print_summary(&self) {
        println!();
        println!("═══════════════════════════════════════════════════════════════════");
        println!("{:^67}", "IV2SLS Regression Results");
        println!("═══════════════════════════════════════════════════════════════════");
        println!(
            " Observations : {}       Instruments   : {}",
            self.n, self.instruments
        );
        println!(
            " R²           : {:.6}   Over-ID df    : {}",
            self.r_squared, self.overidentification_df
        );
        println!("───────────────────────────────────────────────────────────────────");
        println!(
            "{:<22} {:>11} {:>11} {:>9} {:>10}",
            "Variable", "Coef", "Std Err", "t", "P>|t|"
        );
        println!("───────────────────────────────────────────────────────────────────");
        for i in 0..self.feature_names.len() {
            println!(
                "{:<22} {:>11.6} {:>11.6} {:>9.4} {:>10.6}",
                self.feature_names[i],
                self.coefficients[i],
                self.std_errors[i],
                self.t_statistics[i],
                self.p_values[i],
            );
        }
        println!("═══════════════════════════════════════════════════════════════════");
        println!();
    }
}

fn validate_inputs(
    x: &[Vec<f64>],
    y: &[f64],
    instruments: &[Vec<f64>],
    add_intercept: bool,
    feature_names: &[String],
) -> Result<()> {
    if x.len() != y.len() {
        return Err(InferustError::DimensionMismatch {
            x_rows: x.len(),
            y_len: y.len(),
        });
    }
    if instruments.len() != y.len() {
        return Err(InferustError::DimensionMismatch {
            x_rows: instruments.len(),
            y_len: y.len(),
        });
    }
    if y.len() < 3 {
        return Err(InferustError::InsufficientData {
            needed: 3,
            got: y.len(),
        });
    }
    let p = x[0].len();
    let q = instruments[0].len();
    let k = if add_intercept { p + 1 } else { p };
    if y.len() <= k {
        return Err(InferustError::InsufficientData {
            needed: k + 1,
            got: y.len(),
        });
    }
    if !feature_names.is_empty() && feature_names.len() != p {
        return Err(InferustError::InvalidInput(format!(
            "expected {p} feature names, got {}",
            feature_names.len()
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
    for row in instruments {
        if row.len() != q {
            return Err(InferustError::InvalidInput(
                "all rows in instruments must have the same length".into(),
            ));
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(InferustError::InvalidInput(
                "instrument values must be finite".into(),
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

fn build_matrix(rows: &[Vec<f64>], add_intercept: bool) -> DMatrix<f64> {
    let p = rows[0].len();
    let ncols = if add_intercept { p + 1 } else { p };
    let mut data = Vec::with_capacity(rows.len() * ncols);
    for row in rows {
        if add_intercept {
            data.push(1.0);
        }
        data.extend_from_slice(row);
    }
    DMatrix::from_row_slice(rows.len(), ncols, &data)
}

fn build_instrument_matrix(instruments: &[Vec<f64>], add_intercept: bool) -> DMatrix<f64> {
    let ncols = usize::from(add_intercept) + instruments[0].len();
    let mut data = Vec::with_capacity(instruments.len() * ncols);
    for z_row in instruments {
        if add_intercept {
            data.push(1.0);
        }
        data.extend_from_slice(z_row);
    }
    DMatrix::from_row_slice(instruments.len(), ncols, &data)
}

fn invert(matrix: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    if let Some(cholesky) = matrix.clone().cholesky() {
        return Ok(cholesky.inverse());
    }
    matrix
        .clone()
        .try_inverse()
        .ok_or(InferustError::SingularMatrix)
}

fn feature_names(user_names: &[String], p: usize, add_intercept: bool) -> Vec<String> {
    let mut names = Vec::with_capacity(p + usize::from(add_intercept));
    if add_intercept {
        names.push("const".to_string());
    }
    if user_names.is_empty() {
        names.extend((0..p).map(|i| format!("x{}", i + 1)));
    } else {
        names.extend(user_names.iter().cloned());
    }
    names
}

#[cfg(test)]
mod tests {
    use super::Iv2Sls;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual} differed from expected {expected} by more than {tolerance}"
        );
    }

    #[test]
    fn iv2sls_recovers_linear_effect_with_valid_instrument() {
        let mut x = Vec::new();
        let mut z = Vec::new();
        let mut y = Vec::new();
        for i in 0..80 {
            let instrument = i as f64 / 10.0;
            let noise = ((i % 5) as f64 - 2.0) * 0.02;
            let endogenous = 0.5 + 1.5 * instrument + noise;
            x.push(vec![endogenous]);
            z.push(vec![instrument]);
            y.push(2.0 + 3.0 * endogenous + noise);
        }
        let result = Iv2Sls::new()
            .with_feature_names(vec!["demand".into()])
            .fit(&x, &y, &z)
            .unwrap();
        assert_close(result.coefficients[0], 2.0, 1e-1);
        assert_close(result.coefficients[1], 3.0, 1e-2);
        assert_eq!(result.feature_names, vec!["const", "demand"]);
        assert_eq!(result.predict(&[vec![2.0]]).unwrap().len(), 1);
    }

    #[test]
    fn rejects_underidentified_model() {
        let x = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
            vec![4.0, 5.0],
        ];
        let z = vec![vec![], vec![], vec![], vec![]];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        assert!(Iv2Sls::new().fit(&x, &y, &z).is_err());
    }
}
