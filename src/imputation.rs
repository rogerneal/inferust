//! Multiple-imputation starters.
//!
//! Uses `Option<f64>` matrices so missing values are explicit and Rust-native.

use crate::error::{InferustError, Result};
use crate::regression::Ols;

/// Result of a single imputation pass.
#[derive(Debug, Clone)]
pub struct ImputationResult {
    pub data: Vec<Vec<f64>>,
    pub column_means: Vec<f64>,
    pub imputed_cells: usize,
}

/// Chained-equations imputer using linear regressions with mean-imputation fallback.
#[derive(Debug, Clone)]
pub struct MiceImputer {
    iterations: usize,
}

impl Default for MiceImputer {
    fn default() -> Self {
        Self::new()
    }
}

impl MiceImputer {
    pub fn new() -> Self {
        Self { iterations: 5 }
    }

    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Mean-impute missing cells.
    pub fn mean_impute(&self, data: &[Vec<Option<f64>>]) -> Result<ImputationResult> {
        validate_option_matrix(data)?;
        let means = column_means(data)?;
        let mut imputed = Vec::with_capacity(data.len());
        let mut imputed_cells = 0;
        for row in data {
            let mut out = Vec::with_capacity(row.len());
            for (j, value) in row.iter().enumerate() {
                match value {
                    Some(value) => out.push(*value),
                    None => {
                        out.push(means[j]);
                        imputed_cells += 1;
                    }
                }
            }
            imputed.push(out);
        }
        Ok(ImputationResult {
            data: imputed,
            column_means: means,
            imputed_cells,
        })
    }

    /// Lightweight MICE pass. Missing cells are initialized with column means,
    /// then each incomplete column is refit from the other columns.
    pub fn fit_transform(&self, data: &[Vec<Option<f64>>]) -> Result<ImputationResult> {
        if self.iterations == 0 {
            return Err(InferustError::InvalidInput(
                "iterations must be at least 1".into(),
            ));
        }
        let initial = self.mean_impute(data)?;
        let means = initial.column_means.clone();
        let mut filled = initial.data;
        let missing = missing_mask(data);
        let p = filled[0].len();

        for _ in 0..self.iterations {
            for target_col in 0..p {
                if !missing.iter().any(|row| row[target_col]) {
                    continue;
                }
                let observed_rows = (0..filled.len())
                    .filter(|i| !missing[*i][target_col])
                    .collect::<Vec<_>>();
                if observed_rows.len() <= p {
                    continue;
                }
                let x_train = observed_rows
                    .iter()
                    .map(|&i| predictors_without(&filled[i], target_col))
                    .collect::<Vec<_>>();
                let y_train = observed_rows
                    .iter()
                    .map(|&i| filled[i][target_col])
                    .collect::<Vec<_>>();
                let Ok(model) = Ols::new().stable().fit(&x_train, &y_train) else {
                    continue;
                };
                for i in 0..filled.len() {
                    if missing[i][target_col] {
                        let pred = model.predict(&[predictors_without(&filled[i], target_col)]);
                        filled[i][target_col] = pred[0];
                    }
                }
            }
        }

        Ok(ImputationResult {
            data: filled,
            column_means: means,
            imputed_cells: missing.iter().flatten().filter(|cell| **cell).count(),
        })
    }
}

fn validate_option_matrix(data: &[Vec<Option<f64>>]) -> Result<()> {
    if data.is_empty() {
        return Err(InferustError::InsufficientData { needed: 1, got: 0 });
    }
    let p = data[0].len();
    if p == 0 {
        return Err(InferustError::InvalidInput(
            "imputation matrix needs at least one column".into(),
        ));
    }
    for row in data {
        if row.len() != p {
            return Err(InferustError::InvalidInput(
                "all imputation rows must have the same width".into(),
            ));
        }
        for value in row.iter().flatten() {
            if !value.is_finite() {
                return Err(InferustError::InvalidInput(
                    "observed values must be finite".into(),
                ));
            }
        }
    }
    Ok(())
}

fn column_means(data: &[Vec<Option<f64>>]) -> Result<Vec<f64>> {
    let p = data[0].len();
    let mut sums = vec![0.0; p];
    let mut counts = vec![0; p];
    for row in data {
        for (j, value) in row.iter().enumerate() {
            if let Some(value) = value {
                sums[j] += value;
                counts[j] += 1;
            }
        }
    }
    if let Some((j, _)) = counts.iter().enumerate().find(|(_, count)| **count == 0) {
        return Err(InferustError::InvalidInput(format!(
            "column {j} has no observed values"
        )));
    }
    Ok(sums
        .iter()
        .zip(counts.iter())
        .map(|(sum, count)| sum / *count as f64)
        .collect())
}

fn missing_mask(data: &[Vec<Option<f64>>]) -> Vec<Vec<bool>> {
    data.iter()
        .map(|row| row.iter().map(Option::is_none).collect())
        .collect()
}

fn predictors_without(row: &[f64], target_col: usize) -> Vec<f64> {
    row.iter()
        .enumerate()
        .filter_map(|(j, value)| (j != target_col).then_some(*value))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::MiceImputer;

    #[test]
    fn mean_imputation_fills_missing_cells() {
        let data = vec![
            vec![Some(1.0), Some(2.0)],
            vec![None, Some(4.0)],
            vec![Some(3.0), None],
        ];
        let result = MiceImputer::new().mean_impute(&data).unwrap();
        assert_eq!(result.imputed_cells, 2);
        assert_eq!(result.data[1][0], 2.0);
        assert_eq!(result.data[2][1], 3.0);
    }

    #[test]
    fn mice_regression_pass_runs() {
        let data = vec![
            vec![Some(1.0), Some(2.0), Some(3.0)],
            vec![Some(2.0), None, Some(5.0)],
            vec![Some(3.0), Some(6.0), Some(9.0)],
            vec![Some(4.0), None, Some(11.0)],
            vec![Some(5.0), Some(10.0), Some(15.0)],
        ];
        let result = MiceImputer::new()
            .iterations(2)
            .fit_transform(&data)
            .unwrap();
        assert_eq!(result.imputed_cells, 2);
        assert!(result.data.iter().flatten().all(|value| value.is_finite()));
    }
}
