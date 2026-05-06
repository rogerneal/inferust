//! Multivariate statistics.
//!
//! Includes one-way MANOVA and PCA starters for statsmodels-style workflows.

use nalgebra::{DMatrix, SymmetricEigen};
use statrs::distribution::{ContinuousCDF, FisherSnedecor};

use crate::error::{InferustError, Result};

/// One-way MANOVA result using Wilks' lambda with Rao's F approximation.
#[derive(Debug, Clone)]
pub struct ManovaResult {
    pub wilks_lambda: f64,
    pub f_statistic: f64,
    pub p_value: f64,
    pub df_hypothesis: f64,
    pub df_error: f64,
    pub groups: usize,
    pub responses: usize,
}

impl ManovaResult {
    pub fn print(&self) {
        println!();
        println!("── One-Way MANOVA ─────────────────────────────────────────────────");
        println!(" Wilks' lambda : {:.6}", self.wilks_lambda);
        println!(
            " F({:.0}, {:.0})    : {:.6}   p = {:.6}",
            self.df_hypothesis, self.df_error, self.f_statistic, self.p_value
        );
    }
}

/// One-way MANOVA over response matrices by group.
///
/// Each group is an `n × p` matrix represented as rows.
pub fn one_way_manova(groups: &[Vec<Vec<f64>>]) -> Result<ManovaResult> {
    validate_groups(groups)?;
    let g = groups.len();
    let p = groups[0][0].len();
    let n_total = groups.iter().map(Vec::len).sum::<usize>();
    let grand = mean_rows(&groups.iter().flatten().cloned().collect::<Vec<_>>(), p);
    let mut h = DMatrix::<f64>::zeros(p, p);
    let mut e = DMatrix::<f64>::zeros(p, p);

    for group in groups {
        let mean = mean_rows(group, p);
        let diff = DMatrix::from_column_slice(p, 1, &subtract(&mean, &grand));
        h += group.len() as f64 * (&diff * diff.transpose());
        for row in group {
            let centered = DMatrix::from_column_slice(p, 1, &subtract(row, &mean));
            e += &centered * centered.transpose();
        }
    }

    let det_e = regularized_determinant(&e);
    let det_total = regularized_determinant(&(e + h));
    let wilks_lambda = (det_e / det_total).clamp(0.0, 1.0);
    let df_hypothesis = (p * (g - 1)) as f64;
    let df_error = (n_total - g - p + 1).max(1) as f64;
    let f_statistic = ((1.0 - wilks_lambda) / wilks_lambda.max(1e-12)) * df_error / df_hypothesis;
    let f_dist = FisherSnedecor::new(df_hypothesis, df_error)
        .map_err(|_| InferustError::InvalidInput("invalid F distribution".into()))?;
    let p_value = 1.0 - f_dist.cdf(f_statistic);
    Ok(ManovaResult {
        wilks_lambda,
        f_statistic,
        p_value,
        df_hypothesis,
        df_error,
        groups: g,
        responses: p,
    })
}

/// Principal component analysis result.
#[derive(Debug, Clone)]
pub struct PcaResult {
    pub components: Vec<Vec<f64>>,
    pub explained_variance: Vec<f64>,
    pub explained_variance_ratio: Vec<f64>,
    pub mean: Vec<f64>,
}

impl PcaResult {
    pub fn transform(&self, x: &[Vec<f64>], components: usize) -> Result<Vec<Vec<f64>>> {
        if components == 0 || components > self.components.len() {
            return Err(InferustError::InvalidInput(format!(
                "components must be in 1..={}",
                self.components.len()
            )));
        }
        validate_matrix(x)?;
        Ok(x.iter()
            .map(|row| {
                self.components
                    .iter()
                    .take(components)
                    .map(|component| {
                        row.iter()
                            .zip(self.mean.iter())
                            .zip(component.iter())
                            .map(|((value, mean), loading)| (value - mean) * loading)
                            .sum()
                    })
                    .collect()
            })
            .collect())
    }
}

/// Fit PCA using the covariance eigendecomposition.
pub fn pca(x: &[Vec<f64>]) -> Result<PcaResult> {
    validate_matrix(x)?;
    let n = x.len();
    let p = x[0].len();
    let mean = mean_rows(x, p);
    let mut centered = Vec::with_capacity(n * p);
    for row in x {
        for j in 0..p {
            centered.push(row[j] - mean[j]);
        }
    }
    let matrix = DMatrix::from_row_slice(n, p, &centered);
    let covariance = matrix.transpose() * matrix / (n - 1) as f64;
    let eigen = SymmetricEigen::new(covariance);
    let mut pairs = (0..p)
        .map(|i| {
            let component = eigen
                .eigenvectors
                .column(i)
                .iter()
                .copied()
                .collect::<Vec<_>>();
            (eigen.eigenvalues[i], component)
        })
        .collect::<Vec<_>>();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let total = pairs.iter().map(|(value, _)| value.max(0.0)).sum::<f64>();
    let explained_variance = pairs
        .iter()
        .map(|(value, _)| value.max(0.0))
        .collect::<Vec<_>>();
    let explained_variance_ratio = explained_variance
        .iter()
        .map(|value| value / total.max(1e-12))
        .collect::<Vec<_>>();
    let components = pairs.into_iter().map(|(_, component)| component).collect();
    Ok(PcaResult {
        components,
        explained_variance,
        explained_variance_ratio,
        mean,
    })
}

fn validate_groups(groups: &[Vec<Vec<f64>>]) -> Result<()> {
    if groups.len() < 2 {
        return Err(InferustError::InsufficientData {
            needed: 2,
            got: groups.len(),
        });
    }
    let p = groups
        .first()
        .and_then(|group| group.first())
        .map(Vec::len)
        .ok_or(InferustError::InsufficientData { needed: 1, got: 0 })?;
    if p == 0 {
        return Err(InferustError::InvalidInput(
            "MANOVA needs at least one response".into(),
        ));
    }
    for group in groups {
        if group.len() < 2 {
            return Err(InferustError::InsufficientData {
                needed: 2,
                got: group.len(),
            });
        }
        for row in group {
            if row.len() != p {
                return Err(InferustError::InvalidInput(
                    "all MANOVA response rows must have the same width".into(),
                ));
            }
        }
    }
    Ok(())
}

fn validate_matrix(x: &[Vec<f64>]) -> Result<()> {
    if x.len() < 2 {
        return Err(InferustError::InsufficientData {
            needed: 2,
            got: x.len(),
        });
    }
    let p = x[0].len();
    if p == 0 {
        return Err(InferustError::InvalidInput(
            "matrix needs at least one column".into(),
        ));
    }
    for row in x {
        if row.len() != p {
            return Err(InferustError::InvalidInput(
                "all rows must have the same width".into(),
            ));
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(InferustError::InvalidInput(
                "matrix values must be finite".into(),
            ));
        }
    }
    Ok(())
}

fn mean_rows(rows: &[Vec<f64>], p: usize) -> Vec<f64> {
    let mut mean = vec![0.0; p];
    for row in rows {
        for j in 0..p {
            mean[j] += row[j];
        }
    }
    for value in &mut mean {
        *value /= rows.len() as f64;
    }
    mean
}

fn subtract(left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter().zip(right.iter()).map(|(a, b)| a - b).collect()
}

fn regularized_determinant(matrix: &DMatrix<f64>) -> f64 {
    let mut regularized = matrix.clone();
    for i in 0..regularized.nrows().min(regularized.ncols()) {
        regularized[(i, i)] += 1e-10;
    }
    regularized.determinant().abs().max(1e-12)
}

#[cfg(test)]
mod tests {
    use super::{one_way_manova, pca};

    #[test]
    fn manova_detects_group_separation() {
        let a = vec![vec![1.0, 1.1], vec![1.2, 0.9], vec![0.8, 1.0]];
        let b = vec![vec![3.0, 3.1], vec![3.2, 2.9], vec![2.8, 3.0]];
        let result = one_way_manova(&[a, b]).unwrap();
        assert!(result.wilks_lambda < 0.2);
        assert_eq!(result.responses, 2);
    }

    #[test]
    fn pca_explains_correlated_variance() {
        let x = vec![
            vec![1.0, 1.1],
            vec![2.0, 2.1],
            vec![3.0, 2.9],
            vec![4.0, 4.2],
        ];
        let result = pca(&x).unwrap();
        assert!(result.explained_variance_ratio[0] > 0.95);
        let scores = result.transform(&x, 1).unwrap();
        assert_eq!(scores[0].len(), 1);
    }
}
