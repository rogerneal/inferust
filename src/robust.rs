use crate::error::{InferustError, Result};
use crate::regression::{OlsResult, Wls};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RobustNorm {
    Huber { tuning: f64 },
}

#[derive(Debug, Clone)]
pub struct RobustLinearModel {
    norm: RobustNorm,
    max_iter: usize,
    tolerance: f64,
    feature_names: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct RobustLinearResult {
    pub fit: OlsResult,
    pub weights: Vec<f64>,
    pub iterations: usize,
}

impl Default for RobustLinearModel {
    fn default() -> Self {
        Self::new()
    }
}

impl RobustLinearModel {
    pub fn new() -> Self {
        Self {
            norm: RobustNorm::Huber { tuning: 1.345 },
            max_iter: 50,
            tolerance: 1e-8,
            feature_names: Vec::new(),
        }
    }

    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    pub fn with_norm(mut self, norm: RobustNorm) -> Self {
        self.norm = norm;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn fit(&self, x: &[Vec<f64>], y: &[f64]) -> Result<RobustLinearResult> {
        if y.is_empty() {
            return Err(InferustError::InsufficientData { needed: 1, got: 0 });
        }
        let mut weights = vec![1.0; y.len()];
        let mut previous = Vec::new();
        let mut final_fit = None;
        let mut iterations = 0;
        for iter in 0..self.max_iter {
            iterations = iter + 1;
            let fit = Wls::new()
                .with_feature_names(self.feature_names.clone())
                .fit(x, y, &weights)?;
            let scale = mad_scale(&fit.residuals).max(1e-12);
            weights = fit
                .residuals
                .iter()
                .map(|resid| huber_weight(*resid / scale, self.norm))
                .collect();
            let max_change = if previous.len() == fit.coefficients.len() {
                fit.coefficients
                    .iter()
                    .zip(previous.iter())
                    .map(|(a, b): (&f64, &f64)| (a - b).abs())
                    .fold(0.0_f64, f64::max)
            } else {
                f64::INFINITY
            };
            previous = fit.coefficients.clone();
            final_fit = Some(fit);
            if max_change < self.tolerance {
                break;
            }
        }
        Ok(RobustLinearResult {
            fit: final_fit
                .ok_or_else(|| InferustError::InvalidInput("robust fit did not run".into()))?,
            weights,
            iterations,
        })
    }
}

fn huber_weight(value: f64, norm: RobustNorm) -> f64 {
    match norm {
        RobustNorm::Huber { tuning } => {
            let abs = value.abs();
            if abs <= tuning {
                1.0
            } else {
                tuning / abs
            }
        }
    }
}

fn mad_scale(values: &[f64]) -> f64 {
    let mut abs = values.iter().map(|v| v.abs()).collect::<Vec<_>>();
    abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if abs.len() % 2 == 0 {
        (abs[abs.len() / 2 - 1] + abs[abs.len() / 2]) / 2.0
    } else {
        abs[abs.len() / 2]
    };
    median / 0.6744897501960817
}

#[cfg(test)]
mod tests {
    use super::RobustLinearModel;

    #[test]
    fn downweights_outlier() {
        let x = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
            vec![6.0],
        ];
        let y = vec![2.0, 4.1, 6.0, 8.2, 10.0, 40.0];
        let fit = RobustLinearModel::new().fit(&x, &y).unwrap();
        assert!(fit.weights[5] < 1.0);
    }
}
