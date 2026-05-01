use std::collections::BTreeMap;

use crate::error::{InferustError, Result};
use crate::glm::{Logistic, LogisticResult, Poisson, PoissonResult};
use crate::regression::{Ols, OlsResult, Wls};

/// A parsed formula term — what appears between `+` / `-` signs in the RHS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormulaTerm {
    /// Plain numeric predictor, e.g. `hours`.
    Numeric(String),
    /// Inline categorical expansion, e.g. `C(group)` — one-hot encodes the column.
    Categorical(String),
    /// Two-way interaction, e.g. `x1:x2` — adds x1 * x2 as a column.
    Interaction(String, String),
    /// Poisson/GLM offset term, e.g. `offset(exposure)` — added to linear predictor as-is.
    Offset(String),
}

/// Parsed formula of the form `y ~ terms`.
///
/// Supported syntax:
/// - `y ~ x1 + x2`                  — main effects with intercept
/// - `y ~ x1 + x2 - 1`              — suppress intercept (`- 1` or `+ 0`)
/// - `y ~ C(group) + x`             — inline one-hot encoding for `group`
/// - `y ~ x1:x2`                    — interaction term (x1 × x2 column added)
/// - `y ~ x1 * x2`                  — main effects + interaction (expands to `x1 + x2 + x1:x2`)
/// - `y ~ x + offset(exposure)`     — exposure offset for Poisson/GLM models
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Formula {
    /// Response variable name.
    pub response: String,
    /// Ordered list of RHS terms after expansion.
    pub terms: Vec<FormulaTerm>,
    /// Whether an intercept should be included (default `true`; set `false` by `- 1` or `+ 0`).
    pub intercept: bool,
}

impl Formula {
    /// Parse a formula string.
    ///
    /// # Examples
    /// ```rust
    /// use inferust::data::Formula;
    ///
    /// let f = Formula::parse("y ~ x1 + C(group) + x1:x2 - 1").unwrap();
    /// assert!(!f.intercept);
    /// ```
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

        let mut intercept = true;
        let mut terms: Vec<FormulaTerm> = Vec::new();

        // Tokenise the RHS by splitting on `+` first, then handling `-` as subtraction
        // We split by `+` but also watch for tokens that start with `-`
        for raw_token in rhs.split('+') {
            // Each chunk may itself contain `- 1` or `- 0` at the end
            // Split by `-` to find suppressed intercept tokens
            let sub_parts: Vec<&str> = raw_token.split('-').collect();
            for (idx, part) in sub_parts.iter().enumerate() {
                let tok = part.trim();
                if tok.is_empty() {
                    continue;
                }
                if idx > 0 {
                    // This was after a `-`
                    if tok == "1" || tok == "0" {
                        intercept = false;
                        continue;
                    }
                    // Negative term is unusual in standard formulas — skip with error
                    return Err(InferustError::InvalidInput(
                        format!("unsupported negative term `{tok}` — only `- 1` / `- 0` is supported to drop the intercept"),
                    ));
                }
                // Positive token
                if tok == "0" {
                    intercept = false;
                    continue;
                }
                if tok == "1" {
                    // Explicit `+ 1` keeps intercept; just ignore
                    continue;
                }
                // offset(col)
                if let Some(inner) = strip_fn_call(tok, "offset") {
                    terms.push(FormulaTerm::Offset(inner.to_string()));
                    continue;
                }
                // C(col)
                if let Some(inner) = strip_fn_call(tok, "C") {
                    terms.push(FormulaTerm::Categorical(inner.to_string()));
                    continue;
                }
                // x1 * x2  →  x1 + x2 + x1:x2
                if let Some((a, b)) = tok.split_once('*') {
                    let a = a.trim().to_string();
                    let b = b.trim().to_string();
                    if a.is_empty() || b.is_empty() {
                        return Err(InferustError::InvalidInput(format!(
                            "invalid interaction term `{tok}`"
                        )));
                    }
                    terms.push(FormulaTerm::Numeric(a.clone()));
                    terms.push(FormulaTerm::Numeric(b.clone()));
                    terms.push(FormulaTerm::Interaction(a, b));
                    continue;
                }
                // x1:x2
                if let Some((a, b)) = tok.split_once(':') {
                    let a = a.trim().to_string();
                    let b = b.trim().to_string();
                    if a.is_empty() || b.is_empty() {
                        return Err(InferustError::InvalidInput(format!(
                            "invalid interaction term `{tok}`"
                        )));
                    }
                    terms.push(FormulaTerm::Interaction(a, b));
                    continue;
                }
                // Plain numeric
                terms.push(FormulaTerm::Numeric(tok.to_string()));
            }
        }

        // Deduplicate while preserving order
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        terms.retain(|t| {
            let key = format!("{t:?}");
            seen.insert(key)
        });

        if terms.is_empty() && intercept {
            return Err(InferustError::InvalidInput(
                "formula must contain at least one predictor term".into(),
            ));
        }

        Ok(Self {
            response: response.to_string(),
            terms,
            intercept,
        })
    }

    /// Convenience: predictor column names (numeric + interaction, not offsets).
    pub fn predictor_names(&self) -> Vec<String> {
        self.terms
            .iter()
            .filter_map(|t| match t {
                FormulaTerm::Numeric(n) => Some(n.clone()),
                FormulaTerm::Categorical(n) => Some(format!("C({n})")),
                FormulaTerm::Interaction(a, b) => Some(format!("{a}:{b}")),
                FormulaTerm::Offset(_) => None,
            })
            .collect()
    }
}

/// Strip `func(inner)` → `inner` if the token matches `func(...)`.
fn strip_fn_call<'a>(tok: &'a str, func: &str) -> Option<&'a str> {
    let tok = tok.trim();
    if let Some(rest) = tok.strip_prefix(func) {
        let rest = rest.trim_start();
        if rest.starts_with('(') && rest.ends_with(')') {
            return Some(rest[1..rest.len() - 1].trim());
        }
        None
    } else {
        None
    }
}

/// Built design matrices from a formula and a [`DataFrame`].
#[derive(Debug, Clone)]
pub struct DesignMatrices {
    /// Predictor matrix (n_obs × n_predictors), no intercept column.
    pub x: Vec<Vec<f64>>,
    /// Response vector.
    pub y: Vec<f64>,
    /// Human-readable names for each column in `x`.
    pub predictor_names: Vec<String>,
    /// Whether the fitted model should include an intercept (from `Formula::intercept`).
    pub intercept: bool,
    /// Offset vector for GLM models (from `offset(...)` terms); `None` if not present.
    pub offset: Option<Vec<f64>>,
}

/// Minimal named-column data frame for formula-based fitting.
#[derive(Debug, Clone, Default)]
pub struct DataFrame {
    columns: BTreeMap<String, Vec<f64>>,
    categorical_columns: BTreeMap<String, Vec<String>>,
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

    /// Add a string/categorical column. All columns must have the same length.
    ///
    /// Use `C(name)` in a formula to one-hot encode the labels with treatment
    /// coding. The lexicographically smallest level is used as the reference.
    ///
    /// This keeps `inferust` dependency-light while making it easy to pass
    /// labels collected from a Polars/String/Categorical column.
    pub fn with_categorical_column<S: Into<String>>(
        mut self,
        name: impl Into<String>,
        values: Vec<S>,
    ) -> Result<Self> {
        self.add_categorical_column(name, values)?;
        Ok(self)
    }

    /// Add a numeric column in-place. All columns must have the same length.
    pub fn add_column(&mut self, name: impl Into<String>, values: Vec<f64>) -> Result<()> {
        let name = name.into();
        self.validate_column(&name, values.len())?;
        if self.categorical_columns.contains_key(&name) {
            return Err(InferustError::InvalidInput(format!(
                "column `{name}` already exists as categorical"
            )));
        }
        self.columns.insert(name, values);
        Ok(())
    }

    /// Add a string/categorical column in-place. All columns must have the same length.
    pub fn add_categorical_column<S: Into<String>>(
        &mut self,
        name: impl Into<String>,
        values: Vec<S>,
    ) -> Result<()> {
        let name = name.into();
        self.validate_column(&name, values.len())?;
        if self.columns.contains_key(&name) {
            return Err(InferustError::InvalidInput(format!(
                "column `{name}` already exists as numeric"
            )));
        }
        let values = values.into_iter().map(Into::into).collect();
        self.categorical_columns.insert(name, values);
        Ok(())
    }

    fn validate_column(&mut self, name: &str, len: usize) -> Result<()> {
        if name.trim().is_empty() {
            return Err(InferustError::InvalidInput(
                "column name cannot be empty".into(),
            ));
        }
        if len == 0 {
            return Err(InferustError::InsufficientData { needed: 1, got: 0 });
        }
        if let Some(nrows) = self.nrows {
            if len != nrows {
                return Err(InferustError::DimensionMismatch {
                    x_rows: len,
                    y_len: nrows,
                });
            }
        } else {
            self.nrows = Some(len);
        }
        Ok(())
    }

    /// Number of rows in the frame.
    pub fn nrows(&self) -> usize {
        self.nrows.unwrap_or(0)
    }

    /// Borrow a column by name.
    pub fn column(&self, name: &str) -> Result<&[f64]> {
        if let Some(column) = self.columns.get(name) {
            return Ok(column.as_slice());
        }
        if self.categorical_columns.contains_key(name) {
            return Err(InferustError::InvalidInput(format!(
                "column `{name}` is categorical; use C({name}) in a formula"
            )));
        }
        Err(InferustError::InvalidInput(format!(
            "unknown column `{name}`"
        )))
    }

    /// Borrow a categorical column by name.
    pub fn categorical_column(&self, name: &str) -> Result<&[String]> {
        if let Some(column) = self.categorical_columns.get(name) {
            return Ok(column.as_slice());
        }
        if self.columns.contains_key(name) {
            return Err(InferustError::InvalidInput(format!(
                "column `{name}` is numeric"
            )));
        }
        Err(InferustError::InvalidInput(format!(
            "unknown column `{name}`"
        )))
    }

    /// Build design matrices from a formula string.
    ///
    /// Supports all [`Formula`] syntax: `- 1`, `C(var)`, `x1:x2`, `x1*x2`, `offset(var)`.
    pub fn design_matrices(&self, formula: &str) -> Result<DesignMatrices> {
        self.design_matrices_with_categorical(formula, &[])
    }

    /// Build design matrices, treating columns in `categorical` as one-hot encoded
    /// in addition to any `C(...)` terms in the formula.
    ///
    /// The smallest sorted category level is used as the reference (dropped) level.
    pub fn design_matrices_with_categorical(
        &self,
        formula: &str,
        extra_categorical: &[&str],
    ) -> Result<DesignMatrices> {
        let f = Formula::parse(formula)?;
        let y = self.column(&f.response)?.to_vec();
        let nrows = self.nrows();
        let mut x: Vec<Vec<f64>> = vec![Vec::new(); nrows];
        let mut predictor_names: Vec<String> = Vec::new();
        let mut offset_col: Option<Vec<f64>> = None;

        for term in &f.terms {
            match term {
                FormulaTerm::Numeric(col) => {
                    let is_cat = extra_categorical.contains(&col.as_str());
                    if is_cat {
                        self.expand_categorical(col, nrows, &mut x, &mut predictor_names)?;
                    } else {
                        let col_data = self.column(col)?;
                        predictor_names.push(col.clone());
                        for row in 0..nrows {
                            x[row].push(col_data[row]);
                        }
                    }
                }
                FormulaTerm::Categorical(col) => {
                    self.expand_categorical(col, nrows, &mut x, &mut predictor_names)?;
                }
                FormulaTerm::Interaction(a, b) => {
                    let col_a = self.column(a)?;
                    let col_b = self.column(b)?;
                    predictor_names.push(format!("{a}:{b}"));
                    for row in 0..nrows {
                        x[row].push(col_a[row] * col_b[row]);
                    }
                }
                FormulaTerm::Offset(col) => {
                    offset_col = Some(self.column(col)?.to_vec());
                }
            }
        }

        Ok(DesignMatrices {
            x,
            y,
            predictor_names,
            intercept: f.intercept,
            offset: offset_col,
        })
    }

    /// One-hot expand a categorical column (reference = smallest sorted level).
    fn expand_categorical(
        &self,
        col: &str,
        nrows: usize,
        x: &mut [Vec<f64>],
        names: &mut Vec<String>,
    ) -> Result<()> {
        if let Some(data) = self.columns.get(col) {
            let mut levels = data.clone();
            levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            levels.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);
            for &level in levels.iter().skip(1) {
                names.push(format!("{col}[T.{level}]"));
                for row in 0..nrows {
                    x[row].push(f64::from((data[row] - level).abs() < f64::EPSILON));
                }
            }
            return Ok(());
        }

        if let Some(data) = self.categorical_columns.get(col) {
            let mut levels = data.clone();
            levels.sort();
            levels.dedup();
            for level in levels.iter().skip(1) {
                names.push(format!("{col}[T.{level}]"));
                for row in 0..nrows {
                    x[row].push(f64::from(data[row] == *level));
                }
            }
            return Ok(());
        }

        Err(InferustError::InvalidInput(format!(
            "unknown column `{col}`"
        )))
    }

    /// Fit OLS from a formula.
    ///
    /// Supports all formula syntax including `- 1`, `C()`, `:`, `*`.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use inferust::data::DataFrame;
    /// # let df = DataFrame::new();
    /// // No intercept
    /// let r = df.ols("y ~ x1 + x2 - 1");
    /// // Interaction
    /// let r = df.ols("y ~ x1 * x2");
    /// // Inline categorical
    /// let r = df.ols("y ~ C(group) + x1");
    /// ```
    pub fn ols(&self, formula: &str) -> Result<OlsResult> {
        let d = self.design_matrices(formula)?;
        let mut builder = Ols::new().with_feature_names(d.predictor_names);
        if !d.intercept {
            builder = builder.no_intercept();
        }
        builder.fit(&d.x, &d.y)
    }

    /// Fit OLS with explicit extra categorical columns (in addition to `C()` in the formula).
    pub fn ols_with_categorical(&self, formula: &str, categorical: &[&str]) -> Result<OlsResult> {
        let d = self.design_matrices_with_categorical(formula, categorical)?;
        let mut builder = Ols::new().with_feature_names(d.predictor_names);
        if !d.intercept {
            builder = builder.no_intercept();
        }
        builder.fit(&d.x, &d.y)
    }

    /// Fit WLS from a formula and a named weight column.
    pub fn wls(&self, formula: &str, weights: &str) -> Result<OlsResult> {
        let d = self.design_matrices(formula)?;
        let wts = self.column(weights)?;
        let mut builder = Wls::new().with_feature_names(d.predictor_names);
        if !d.intercept {
            builder = builder.no_intercept();
        }
        builder.fit(&d.x, &d.y, wts)
    }

    /// Fit binary logistic regression from a formula.
    pub fn logistic(&self, formula: &str) -> Result<LogisticResult> {
        let d = self.design_matrices(formula)?;
        let mut builder = Logistic::new().with_feature_names(d.predictor_names);
        if !d.intercept {
            builder = builder.no_intercept();
        }
        builder.fit(&d.x, &d.y)
    }

    /// Fit Poisson regression from a formula.
    ///
    /// If the formula contains an `offset(col)` term, the column is added to the
    /// linear predictor as a fixed offset (equivalent to statsmodels' `exposure`).
    pub fn poisson(&self, formula: &str) -> Result<PoissonResult> {
        let d = self.design_matrices(formula)?;
        let mut builder = Poisson::new().with_feature_names(d.predictor_names);
        if !d.intercept {
            builder = builder.no_intercept();
        }
        if let Some(offset) = d.offset {
            builder = builder.with_offset(offset);
        }
        builder.fit(&d.x, &d.y)
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
        use super::FormulaTerm;
        let formula = Formula::parse("y ~ x1 + x2").unwrap();
        assert_eq!(formula.response, "y");
        assert!(formula.intercept);
        assert_eq!(
            formula.terms,
            vec![
                FormulaTerm::Numeric("x1".to_string()),
                FormulaTerm::Numeric("x2".to_string()),
            ]
        );
    }

    #[test]
    fn formula_no_intercept() {
        let f = Formula::parse("y ~ x1 + x2 - 1").unwrap();
        assert!(!f.intercept);
    }

    #[test]
    fn formula_interaction_colon() {
        use super::FormulaTerm;
        let f = Formula::parse("y ~ x1:x2").unwrap();
        assert_eq!(
            f.terms,
            vec![FormulaTerm::Interaction("x1".into(), "x2".into())]
        );
    }

    #[test]
    fn formula_star_expands_to_main_plus_interaction() {
        use super::FormulaTerm;
        let f = Formula::parse("y ~ x1 * x2").unwrap();
        assert!(f.terms.contains(&FormulaTerm::Numeric("x1".into())));
        assert!(f.terms.contains(&FormulaTerm::Numeric("x2".into())));
        assert!(f
            .terms
            .contains(&FormulaTerm::Interaction("x1".into(), "x2".into())));
    }

    #[test]
    fn formula_c_is_categorical() {
        use super::FormulaTerm;
        let f = Formula::parse("y ~ C(group) + x1").unwrap();
        assert!(f.terms.contains(&FormulaTerm::Categorical("group".into())));
    }

    #[test]
    fn formula_accepts_macro_stringified_function_spacing() {
        use super::FormulaTerm;
        let f = Formula::parse("y ~ x1 + C ( group )").unwrap();
        assert!(f.terms.contains(&FormulaTerm::Categorical("group".into())));
    }

    #[test]
    fn formula_offset() {
        use super::FormulaTerm;
        let f = Formula::parse("y ~ x1 + offset(exposure)").unwrap();
        assert!(f.terms.contains(&FormulaTerm::Offset("exposure".into())));
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
    fn categorical_formula_expands_treatment_dummies() {
        let frame = DataFrame::new()
            .with_column("group", vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
            .unwrap()
            .with_column("x", vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
            .unwrap()
            .with_column("y", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let design = frame
            .design_matrices_with_categorical("y ~ group + x", &["group"])
            .unwrap();
        assert_eq!(
            design.predictor_names,
            vec!["group[T.2]", "group[T.3]", "x"]
        );
        assert_eq!(design.x[2], vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn string_categorical_formula_expands_treatment_dummies() {
        let frame = DataFrame::new()
            .with_categorical_column("group", vec!["a", "a", "b", "b", "c", "c"])
            .unwrap()
            .with_column("x", vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
            .unwrap()
            .with_column("y", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let design = frame.design_matrices("y ~ C(group) + x").unwrap();
        assert_eq!(
            design.predictor_names,
            vec!["group[T.b]", "group[T.c]", "x"]
        );
        assert_eq!(design.x[2], vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn categorical_column_requires_c_formula_term() {
        let frame = DataFrame::new()
            .with_categorical_column("group", vec!["a", "b"])
            .unwrap()
            .with_column("y", vec![1.0, 2.0])
            .unwrap();
        let err = frame.design_matrices("y ~ group").unwrap_err();
        assert!(format!("{err}").contains("use C(group)"));
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
    fn formula_poisson_fits_named_columns() {
        let frame = DataFrame::new()
            .with_column(
                "x1",
                vec![0.2, 0.8, 1.2, 1.9, 2.4, 2.9, 3.4, 3.9, 4.5, 5.0, 5.5, 6.0],
            )
            .unwrap()
            .with_column(
                "x2",
                vec![1.0, 1.4, 1.1, 1.7, 2.2, 2.0, 2.8, 3.1, 3.5, 3.8, 4.0, 4.4],
            )
            .unwrap()
            .with_column(
                "y",
                vec![
                    1.0, 2.0, 1.0, 3.0, 4.0, 3.0, 6.0, 7.0, 8.0, 11.0, 12.0, 15.0,
                ],
            )
            .unwrap();
        let result = frame.poisson("y ~ x1 + x2").unwrap();
        assert_close(result.coefficients[0], -0.2951503394477173, 1e-8);
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
