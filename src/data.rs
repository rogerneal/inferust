use std::collections::BTreeMap;

use crate::error::{InferustError, Result};
use crate::glm::{Logistic, LogisticResult};
use crate::regression::{Ols, OlsResult, Wls};

/// Parsed formula of the form `y ~ x1 + x2`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Formula {
    pub response: String,
    pub predictors: Vec<String>,
}

impl Formula {
    /// Parse a simple regression formula like `score ~ hours + gpa`.
    pub fn parse(input: &str) -> Result<Self> {
        let (lhs, rhs) = input
            .split_once('~')
            .ok_or_else(|| InferustError::InvalidInput("formula must contain `~`".into()))?;
        let response = lhs.trim();
        if response.is_empty() {
            return Err(InferustError::InvalidInput(
                "formula response cannot be empty".into(),
            ));
        }

        let predictors = rhs
            .split('+')
            .map(str::trim)
            .filter(|term| !term.is_empty())
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        if predictors.is_empty() {
            return Err(InferustError::InvalidInput(
                "formula must contain at least one predictor".into(),
            ));
        }
        if predictors.iter().any(|term| term == "1") {
            return Err(InferustError::InvalidInput(
                "explicit intercept terms are not supported yet; intercept is added by model builders"
                    .into(),
            ));
        }

        Ok(Self {
            response: response.to_string(),
            predictors,
        })
    }
}

/// Built matrices from a formula and named columns.
#[derive(Debug, Clone)]
pub struct DesignMatrices {
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
    pub predictor_names: Vec<String>,
}

/// Minimal named-column numeric data frame for formula-based fitting.
#[derive(Debug, Clone, Default)]
pub struct DataFrame {
    columns: BTreeMap<String, Vec<f64>>,
    nrows: Option<usize>,
}

impl DataFrame {
    /// Create an empty frame.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a numeric column. All columns must have the same length.
    pub fn with_column(mut self, name: impl Into<String>, values: Vec<f64>) -> Result<Self> {
        self.add_column(name, values)?;
        Ok(self)
    }

    /// Add a numeric column in-place. All columns must have the same length.
    pub fn add_column(&mut self, name: impl Into<String>, values: Vec<f64>) -> Result<()> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(InferustError::InvalidInput(
                "column name cannot be empty".into(),
            ));
        }
        if values.is_empty() {
            return Err(InferustError::InsufficientData { needed: 1, got: 0 });
        }
        if let Some(nrows) = self.nrows {
            if values.len() != nrows {
                return Err(InferustError::DimensionMismatch {
                    x_rows: values.len(),
                    y_len: nrows,
                });
            }
        } else {
            self.nrows = Some(values.len());
        }
        self.columns.insert(name, values);
        Ok(())
    }

    /// Number of rows in the frame.
    pub fn nrows(&self) -> usize {
        self.nrows.unwrap_or(0)
    }

    /// Borrow a column by name.
    pub fn column(&self, name: &str) -> Result<&[f64]> {
        self.columns
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| InferustError::InvalidInput(format!("unknown column `{name}`")))
    }

    /// Build design matrices from a formula.
    pub fn design_matrices(&self, formula: &str) -> Result<DesignMatrices> {
        let formula = Formula::parse(formula)?;
        let y = self.column(&formula.response)?.to_vec();
        let predictor_columns = formula
            .predictors
            .iter()
            .map(|name| self.column(name))
            .collect::<Result<Vec<_>>>()?;

        let nrows = self.nrows();
        let mut x = Vec::with_capacity(nrows);
        for row_idx in 0..nrows {
            let row = predictor_columns
                .iter()
                .map(|column| column[row_idx])
                .collect::<Vec<_>>();
            x.push(row);
        }

        Ok(DesignMatrices {
            x,
            y,
            predictor_names: formula.predictors,
        })
    }

    /// Fit OLS from a formula like `score ~ hours + gpa`.
    pub fn ols(&self, formula: &str) -> Result<OlsResult> {
        let design = self.design_matrices(formula)?;
        Ols::new()
            .with_feature_names(design.predictor_names)
            .fit(&design.x, &design.y)
    }

    /// Fit WLS from a formula and a named weight column.
    pub fn wls(&self, formula: &str, weights: &str) -> Result<OlsResult> {
        let design = self.design_matrices(formula)?;
        let weights = self.column(weights)?;
        Wls::new()
            .with_feature_names(design.predictor_names)
            .fit(&design.x, &design.y, weights)
    }

    /// Fit binary logistic regression from a formula like `clicked ~ age + visits`.
    pub fn logistic(&self, formula: &str) -> Result<LogisticResult> {
        let design = self.design_matrices(formula)?;
        Logistic::new()
            .with_feature_names(design.predictor_names)
            .fit(&design.x, &design.y)
    }
}

#[cfg(test)]
mod tests {
    use super::{DataFrame, Formula};

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual} differed from expected {expected} by more than {tolerance}"
        );
    }

    fn frame() -> DataFrame {
        DataFrame::new()
            .with_column("x1", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .with_column("x2", vec![2.0, 1.0, 4.0, 3.0, 5.0, 7.0])
            .unwrap()
            .with_column("y", vec![5.1, 5.9, 10.2, 10.8, 14.9, 19.1])
            .unwrap()
            .with_column("weights", vec![1.0, 0.8, 1.2, 1.5, 0.9, 1.1])
            .unwrap()
    }

    #[test]
    fn parses_basic_formula() {
        let formula = Formula::parse("y ~ x1 + x2").unwrap();
        assert_eq!(formula.response, "y");
        assert_eq!(formula.predictors, vec!["x1", "x2"]);
    }

    #[test]
    fn builds_design_matrices_from_named_columns() {
        let design = frame().design_matrices("y ~ x1 + x2").unwrap();
        assert_eq!(design.predictor_names, vec!["x1", "x2"]);
        assert_eq!(design.y[0], 5.1);
        assert_eq!(design.x[0], vec![1.0, 2.0]);
        assert_eq!(design.x[5], vec![6.0, 7.0]);
    }

    #[test]
    fn formula_ols_matches_matrix_ols_reference() {
        let result = frame().ols("y ~ x1 + x2").unwrap();
        assert_close(result.coefficients[0], 1.1666007905138316, 1e-10);
        assert_close(result.coefficients[1], 1.656126482213441, 1e-10);
        assert_close(result.coefficients[2], 1.100988142292489, 1e-10);
        assert_eq!(result.feature_names, vec!["const", "x1", "x2"]);
    }

    #[test]
    fn formula_wls_matches_matrix_wls_reference() {
        let result = frame().wls("y ~ x1 + x2", "weights").unwrap();
        assert_close(result.coefficients[0], 1.0910621653414276, 1e-10);
        assert_close(result.coefficients[1], 1.6265313140792843, 1e-10);
        assert_close(result.coefficients[2], 1.139502728692733, 1e-10);
        assert_eq!(result.feature_names, vec!["const", "x1", "x2"]);
    }

    #[test]
    fn formula_logistic_fits_named_columns() {
        let frame = DataFrame::new()
            .with_column(
                "x1",
                vec![0.2, 1.1, 1.8, 2.4, 3.0, 3.7, 4.1, 4.8, 5.2, 5.9, 2.2, 4.6],
            )
            .unwrap()
            .with_column(
                "x2",
                vec![1.0, 0.9, 1.5, 1.9, 2.5, 2.9, 3.4, 3.8, 4.2, 4.8, 3.6, 1.2],
            )
            .unwrap()
            .with_column(
                "y",
                vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            )
            .unwrap();
        let result = frame.logistic("y ~ x1 + x2").unwrap();
        assert_close(result.coefficients[0], -1.7689272112231273, 1e-8);
        assert_eq!(result.feature_names, vec!["const", "x1", "x2"]);
    }
}
