use std::collections::BTreeMap;

use crate::error::{InferustError, Result};
use crate::regression::{Ols, OlsResult};

#[derive(Debug, Clone)]
pub struct MixedLinearModel {
    feature_names: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct MixedLinearResult {
    pub fixed_effects: OlsResult,
    pub random_intercepts: BTreeMap<usize, f64>,
    pub fitted_values: Vec<f64>,
    pub residuals: Vec<f64>,
    pub group_count: usize,
}

impl Default for MixedLinearModel {
    fn default() -> Self {
        Self::new()
    }
}

impl MixedLinearModel {
    pub fn new() -> Self {
        Self {
            feature_names: Vec::new(),
        }
    }

    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Fits fixed effects plus empirical random intercepts from group mean residuals.
    pub fn fit_random_intercept(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
        groups: &[usize],
    ) -> Result<MixedLinearResult> {
        if groups.len() != y.len() {
            return Err(InferustError::DimensionMismatch {
                x_rows: groups.len(),
                y_len: y.len(),
            });
        }
        let fixed_effects = Ols::new()
            .with_feature_names(self.feature_names.clone())
            .fit(x, y)?;
        let mut sums: BTreeMap<usize, (f64, usize)> = BTreeMap::new();
        for (&group, &resid) in groups.iter().zip(fixed_effects.residuals.iter()) {
            let entry = sums.entry(group).or_insert((0.0, 0));
            entry.0 += resid;
            entry.1 += 1;
        }
        let random_intercepts = sums
            .iter()
            .map(|(group, (sum, n))| (*group, sum / *n as f64))
            .collect::<BTreeMap<_, _>>();
        let fitted_values = fixed_effects
            .fitted_values
            .iter()
            .zip(groups.iter())
            .map(|(fitted, group)| fitted + random_intercepts[group])
            .collect::<Vec<_>>();
        let residuals = y
            .iter()
            .zip(fitted_values.iter())
            .map(|(yi, fitted)| yi - fitted)
            .collect::<Vec<_>>();
        Ok(MixedLinearResult {
            fixed_effects,
            group_count: random_intercepts.len(),
            random_intercepts,
            fitted_values,
            residuals,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::MixedLinearModel;

    #[test]
    fn estimates_random_intercepts() {
        let x = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![1.0],
            vec![2.0],
            vec![3.0],
        ];
        let y = vec![3.0, 5.0, 7.0, 6.0, 8.0, 10.0];
        let groups = vec![1, 1, 1, 2, 2, 2];
        let fit = MixedLinearModel::new()
            .fit_random_intercept(&x, &y, &groups)
            .unwrap();
        assert_eq!(fit.group_count, 2);
        assert!(fit.random_intercepts[&1] < fit.random_intercepts[&2]);
    }
}
