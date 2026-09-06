//! Shared covariance estimators for OLS, GLM, discrete, and panel models.
//!
//! [`CovType`] is the crate-wide switch (`cov_type` in statsmodels). OLS keeps
//! the [`crate::regression::OlsCovariance`] alias so existing code still
//! compiles.

use nalgebra::DMatrix;
use statrs::distribution::{ContinuousCDF, Normal};

use crate::error::{InferustError, Result};

/// Covariance estimator used for coefficient standard errors.
///
/// OLS defaults to [`Self::Nonrobust`]. `.robust()` on builders selects
/// [`Self::Hc1`], matching the common statsmodels robust default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CovType {
    /// Model-based / homoskedastic covariance.
    Nonrobust,
    /// White HC0.
    Hc0,
    /// HC0 with small-sample correction `n / (n − k)`.
    Hc1,
    /// HC0 with leverage adjustment `e² / (1 − h)`.
    Hc2,
    /// HC0 with squared leverage adjustment `e² / (1 − h)²`.
    Hc3,
    /// Newey–West HAC with a Bartlett kernel.
    Hac { lags: usize },
    /// One-way cluster-robust sandwich.
    Cluster { groups: Vec<usize> },
}

impl CovType {
    /// Short label for summaries (`HC1`, `cluster`, …).
    pub fn label(&self) -> &'static str {
        match self {
            CovType::Nonrobust => "nonrobust",
            CovType::Hc0 => "HC0",
            CovType::Hc1 => "HC1",
            CovType::Hc2 => "HC2",
            CovType::Hc3 => "HC3",
            CovType::Hac { .. } => "HAC (Newey-West)",
            CovType::Cluster { .. } => "cluster",
        }
    }

    pub(crate) fn uses_t_distribution(&self) -> bool {
        matches!(self, CovType::Nonrobust)
    }
}

/// Score-based sandwich covariance `bread · meat · bread`.
///
/// `bread` is the model-based covariance (usually `(X'WX)⁻¹` or `φ(X'WX)⁻¹`).
/// Each `scores[i]` is the observation score `s_i` (length `k`).
/// `leverages` is used for HC2/HC3; if missing those variants fall back to HC0.
pub fn sandwich_from_scores(
    bread: &DMatrix<f64>,
    scores: &[Vec<f64>],
    leverages: Option<&[f64]>,
    cov: &CovType,
) -> Result<DMatrix<f64>> {
    if matches!(cov, CovType::Nonrobust) {
        return Ok(bread.clone());
    }
    let n = scores.len();
    if n == 0 {
        return Err(InferustError::InsufficientData { needed: 1, got: 0 });
    }
    let k = bread.nrows();
    if bread.ncols() != k {
        return Err(InferustError::InvalidInput(
            "sandwich bread must be square".into(),
        ));
    }
    for row in scores {
        if row.len() != k {
            return Err(InferustError::InvalidInput(
                "score row length must match bread dimension".into(),
            ));
        }
    }

    if let CovType::Hac { lags } = cov {
        return Ok(score_hac(bread, scores, *lags));
    }
    if let CovType::Cluster { groups } = cov {
        return Ok(score_cluster(bread, scores, groups));
    }

    let mut meat = square_zeros(k);
    for (i, score) in scores.iter().enumerate() {
        let h = leverages.and_then(|lev| lev.get(i).copied()).unwrap_or(0.0);
        let denom = (1.0 - h).max(f64::EPSILON);
        let scale = match cov {
            CovType::Hc0 | CovType::Hc1 => 1.0,
            CovType::Hc2 => 1.0 / denom.sqrt(),
            CovType::Hc3 => 1.0 / denom,
            CovType::Nonrobust | CovType::Hac { .. } | CovType::Cluster { .. } => unreachable!(),
        };
        add_outer(&mut meat, score, scale);
    }
    let hc1 = if matches!(cov, CovType::Hc1) {
        n as f64 / (n.saturating_sub(k).max(1) as f64)
    } else {
        1.0
    };
    Ok(sandwich_product(bread, &meat, hc1))
}

/// Convert a covariance matrix into standard errors, z-statistics, and two-sided
/// normal p-values.
pub fn z_inference(
    coefficients: &[f64],
    cov: &DMatrix<f64>,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let k = coefficients.len();
    let std_errors: Vec<f64> = (0..k).map(|i| cov[(i, i)].max(0.0).sqrt()).collect();
    let z_statistics: Vec<f64> = coefficients
        .iter()
        .zip(std_errors.iter())
        .map(|(&c, &se)| if se > 0.0 { c / se } else { f64::NAN })
        .collect();
    let normal = Normal::new(0.0, 1.0)
        .map_err(|_| InferustError::InvalidInput("invalid normal distribution".into()))?;
    let p_values: Vec<f64> = z_statistics
        .iter()
        .map(|&z| 2.0 * (1.0 - normal.cdf(z.abs())))
        .collect();
    Ok((std_errors, z_statistics, p_values))
}

/// Copy a `k × k` nalgebra matrix into row-major nested vectors.
pub fn matrix_rows(cov: &DMatrix<f64>) -> Vec<Vec<f64>> {
    let k = cov.nrows();
    (0..k)
        .map(|i| (0..cov.ncols()).map(|j| cov[(i, j)]).collect())
        .collect()
}

/// Build a matrix from row-major nested vectors.
pub fn rows_matrix(rows: &[Vec<f64>]) -> Result<DMatrix<f64>> {
    if rows.is_empty() {
        return Err(InferustError::InsufficientData { needed: 1, got: 0 });
    }
    let k = rows[0].len();
    let mut data = Vec::with_capacity(rows.len() * k);
    for row in rows {
        if row.len() != k {
            return Err(InferustError::InvalidInput(
                "matrix rows must have equal length".into(),
            ));
        }
        data.extend_from_slice(row);
    }
    Ok(DMatrix::<f64>::from_row_slice(rows.len(), k, &data))
}

fn square_zeros(k: usize) -> DMatrix<f64> {
    DMatrix::<f64>::from_vec(k, k, vec![0.0_f64; k.saturating_mul(k)])
}

fn square_nans(k: usize) -> DMatrix<f64> {
    DMatrix::<f64>::from_vec(k, k, vec![f64::NAN; k.saturating_mul(k)])
}

/// `meat += (scale * score) (scale * score)'` without nalgebra operator inference.
fn add_outer(meat: &mut DMatrix<f64>, score: &[f64], scale: f64) {
    let k = score.len();
    for i in 0..k {
        let si = score[i] * scale;
        for j in 0..k {
            meat[(i, j)] += si * (score[j] * scale);
        }
    }
}

fn sandwich_product(bread: &DMatrix<f64>, meat: &DMatrix<f64>, scale: f64) -> DMatrix<f64> {
    let k = bread.nrows();
    let mut mid = square_zeros(k);
    for i in 0..k {
        for j in 0..k {
            let mut sum = 0.0_f64;
            for t in 0..k {
                sum += bread[(i, t)] * meat[(t, j)];
            }
            mid[(i, j)] = sum;
        }
    }
    let mut out = square_zeros(k);
    for i in 0..k {
        for j in 0..k {
            let mut sum = 0.0_f64;
            for t in 0..k {
                sum += mid[(i, t)] * bread[(t, j)];
            }
            out[(i, j)] = sum * scale;
        }
    }
    out
}

fn score_hac(bread: &DMatrix<f64>, scores: &[Vec<f64>], lags: usize) -> DMatrix<f64> {
    let n = scores.len();
    let k = bread.nrows();
    let mut meat = square_zeros(k);
    for score in scores {
        add_outer(&mut meat, score, 1.0);
    }
    for lag in 1..=lags.min(n.saturating_sub(1)) {
        let w = 1.0 - lag as f64 / (lags as f64 + 1.0);
        let mut gamma = square_zeros(k);
        for t in lag..n {
            let st = &scores[t];
            let sl = &scores[t - lag];
            for i in 0..k {
                for j in 0..k {
                    gamma[(i, j)] += st[i] * sl[j];
                }
            }
        }
        for i in 0..k {
            for j in 0..k {
                meat[(i, j)] += (gamma[(i, j)] + gamma[(j, i)]) * w;
            }
        }
    }
    sandwich_product(bread, &meat, 1.0)
}

fn score_cluster(bread: &DMatrix<f64>, scores: &[Vec<f64>], groups: &[usize]) -> DMatrix<f64> {
    let n = scores.len();
    let k = bread.nrows();
    if groups.len() != n {
        return square_nans(k);
    }
    let mut unique = groups.to_vec();
    unique.sort_unstable();
    unique.dedup();
    let g = unique.len();
    if g < 2 {
        return square_nans(k);
    }
    let mut meat = square_zeros(k);
    for cluster in unique {
        let mut score = vec![0.0_f64; k];
        for (i, row) in scores.iter().enumerate() {
            if groups[i] == cluster {
                for (j, &v) in row.iter().enumerate() {
                    score[j] += v;
                }
            }
        }
        add_outer(&mut meat, &score, 1.0);
    }
    let df = n.saturating_sub(k).max(1) as f64;
    let correction = (g as f64 / (g - 1) as f64) * ((n - 1) as f64 / df);
    sandwich_product(bread, &meat, correction)
}

#[cfg(test)]
mod tests {
    use super::{sandwich_from_scores, CovType};
    use nalgebra::DMatrix;

    #[test]
    fn hc1_differs_from_bread_and_stays_finite() {
        let bread = DMatrix::<f64>::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let scores = vec![vec![0.5, -0.2], vec![-0.1, 0.4], vec![0.3, 0.1]];
        let robust = sandwich_from_scores(&bread, &scores, None, &CovType::Hc1).unwrap();
        assert!(robust[(0, 0)].is_finite());
        assert!(robust[(0, 0)] > 0.0);
        assert!((robust[(0, 0)] - 1.0).abs() > 1e-12);
    }
}
