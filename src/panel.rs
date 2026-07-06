//! Panel-data estimators.

use std::collections::BTreeMap;

use crate::error::{InferustError, Result};
use crate::regression::{Ols, OlsResult};

/// One-way fixed-effects panel OLS via within-group demeaning.
#[derive(Debug, Clone)]
pub struct PanelOls {
    feature_names: Vec<String>,
}

impl Default for PanelOls {
    fn default() -> Self {
        Self::new()
    }
}

impl PanelOls {
    pub fn new() -> Self {
        Self {
            feature_names: Vec::new(),
        }
    }

    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Fit `y ~ x` with entity fixed effects removed by within transformation.
    pub fn fit_entity_fe(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
        entities: &[usize],
    ) -> Result<OlsResult> {
        if y.is_empty() {
            return Err(InferustError::InsufficientData { needed: 1, got: 0 });
        }
        if entities.len() != y.len() {
            return Err(InferustError::DimensionMismatch {
                x_rows: entities.len(),
                y_len: y.len(),
            });
        }
        let p = if x.is_empty() { 0 } else { x[0].len() };
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, &e) in entities.iter().enumerate() {
            groups.entry(e).or_default().push(i);
        }
        let mut y_dm = y.to_vec();
        let mut x_dm = x.to_vec();
        for idx in groups.values() {
            let n_i = idx.len() as f64;
            let mean_y: f64 = idx.iter().map(|&i| y[i]).sum::<f64>() / n_i;
            for &i in idx {
                y_dm[i] -= mean_y;
            }
            for j in 0..p {
                let mean_x: f64 = idx.iter().map(|&i| x[i][j]).sum::<f64>() / n_i;
                for &i in idx {
                    x_dm[i][j] -= mean_x;
                }
            }
        }
        Ols::new()
            .with_feature_names(self.feature_names.clone())
            .no_intercept()
            .fit(&x_dm, &y_dm)
    }
}

/// Build composite cluster ids for two-way clustering (entity × time).
pub fn two_way_cluster_ids(entities: &[usize], times: &[usize]) -> Vec<usize> {
    entities
        .iter()
        .zip(times)
        .map(|(&e, &t)| e.wrapping_mul(1_000_003).wrapping_add(t))
        .collect()
}
