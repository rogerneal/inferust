//! Shared IRLS / weighted least-squares kernels used across GLM and discrete models.

use nalgebra::{DMatrix, DVector};

use crate::error::{InferustError, Result};

/// `X * beta` without cloning the design matrix.
pub fn mat_vec_mul(x_mat: &DMatrix<f64>, beta: &DVector<f64>) -> Vec<f64> {
    (x_mat * beta).iter().cloned().collect()
}

/// Incrementally accumulate `X' diag(w) X` for an `n × k` design matrix.
pub fn accumulate_xtwx(x_mat: &DMatrix<f64>, weights: &[f64], k: usize) -> DMatrix<f64> {
    let n = weights.len();
    let mut xtwx = DMatrix::zeros(k, k);
    for i in 0..n {
        let wi = weights[i];
        if wi == 0.0 {
            continue;
        }
        for j in 0..k {
            let xij = x_mat[(i, j)];
            for l in 0..=j {
                xtwx[(j, l)] += wi * xij * x_mat[(i, l)];
            }
        }
    }
    for j in 0..k {
        for l in 0..j {
            xtwx[(l, j)] = xtwx[(j, l)];
        }
    }
    xtwx
}

/// Incrementally accumulate `X' diag(w) z`.
pub fn accumulate_xtwz(x_mat: &DMatrix<f64>, weights: &[f64], z: &[f64], k: usize) -> DVector<f64> {
    let n = weights.len();
    let mut xtwz = DVector::zeros(k);
    for i in 0..n {
        let wi = weights[i];
        if wi == 0.0 {
            continue;
        }
        let wiz = wi * z[i];
        for j in 0..k {
            xtwz[j] += x_mat[(i, j)] * wiz;
        }
    }
    xtwz
}

/// Solve weighted normal equations via incremental `X'WX` accumulation.
pub fn irls_weighted_solve(
    x_mat: &DMatrix<f64>,
    weights: &[f64],
    z: &[f64],
) -> Result<DVector<f64>> {
    let k = x_mat.ncols();
    let xtwx = accumulate_xtwx(x_mat, weights, k);
    let xtwz = accumulate_xtwz(x_mat, weights, z, k);
    xtwx.lu().solve(&xtwz).ok_or(InferustError::SingularMatrix)
}
