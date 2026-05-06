//! Generalized Additive Model starters.
//!
//! The first supported model is a Gaussian additive regression using truncated
//! power spline bases and OLS inference on the expanded design.

use crate::error::{InferustError, Result};
use crate::regression::{Ols, OlsResult};

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
#[derive(Debug, Clone, Default)]
pub struct GaussianGam {
    terms: Vec<SplineTerm>,
    linear_columns: Vec<usize>,
    add_intercept: bool,
}

impl GaussianGam {
    pub fn new() -> Self {
        Self {
            terms: Vec::new(),
            linear_columns: Vec::new(),
            add_intercept: true,
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

    /// Fit the additive model to raw predictor rows and response values.
    pub fn fit(&self, x: &[Vec<f64>], y: &[f64]) -> Result<GamResult> {
        validate_inputs(x, y, &self.terms, &self.linear_columns)?;
        let design = build_design(x, &self.terms, &self.linear_columns)?;
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
        })
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
    }
}

#[derive(Debug, Clone)]
struct GamDesign {
    x: Vec<Vec<f64>>,
    names: Vec<String>,
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

    for &column in linear_columns {
        names.push(format!("x{}", column + 1));
        for (i, row) in x.iter().enumerate() {
            rows[i].push(row[column]);
        }
    }

    for term in terms {
        for power in 1..=term.degree {
            names.push(format!("{}^{}", term.name, power));
            for (i, row) in x.iter().enumerate() {
                rows[i].push(row[term.column].powi(power as i32));
            }
        }
        for knot in &term.knots {
            names.push(format!("{}[>{:.3}]", term.name, knot));
            for (i, row) in x.iter().enumerate() {
                rows[i].push((row[term.column] - knot).max(0.0).powi(term.degree as i32));
            }
        }
    }

    Ok(GamDesign { x: rows, names })
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
    }

    #[test]
    fn rejects_empty_model() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![1.0, 2.0, 3.0];
        assert!(GaussianGam::new().fit(&x, &y).is_err());
    }
}
