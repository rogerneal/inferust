use crate::error::{InferustError, Result};

/// Pearson correlation coefficient between two equal-length slices.
pub fn pearson(x: &[f64], y: &[f64]) -> Result<f64> {
    let n = x.len();
    if n != y.len() {
        return Err(InferustError::DimensionMismatch {
            x_rows: n,
            y_len: y.len(),
        });
    }
    if n < 2 {
        return Err(InferustError::InsufficientData { needed: 2, got: n });
    }

    let mx = x.iter().sum::<f64>() / n as f64;
    let my = y.iter().sum::<f64>() / n as f64;

    let num: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - mx) * (yi - my))
        .sum();
    let sx: f64 = x.iter().map(|xi| (xi - mx).powi(2)).sum::<f64>().sqrt();
    let sy: f64 = y.iter().map(|yi| (yi - my).powi(2)).sum::<f64>().sqrt();

    if sx == 0.0 || sy == 0.0 {
        return Err(InferustError::InvalidInput(
            "zero variance in one or both inputs — correlation is undefined".into(),
        ));
    }
    Ok(num / (sx * sy))
}

/// Spearman rank correlation coefficient.
pub fn spearman(x: &[f64], y: &[f64]) -> Result<f64> {
    let n = x.len();
    if n != y.len() {
        return Err(InferustError::DimensionMismatch {
            x_rows: n,
            y_len: y.len(),
        });
    }
    if n < 2 {
        return Err(InferustError::InsufficientData { needed: 2, got: n });
    }
    pearson(&rank(x), &rank(y))
}

/// Symmetric correlation matrix for k variables (each a `Vec<f64>` of the same length).
///
/// Returns a k×k matrix where entry `[i][j]` is the Pearson correlation of variable i and j.
pub fn correlation_matrix(variables: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let k = variables.len();
    let mut matrix = vec![vec![0.0f64; k]; k];
    for i in 0..k {
        for j in 0..k {
            matrix[i][j] = if i == j {
                1.0
            } else if j > i {
                pearson(&variables[i], &variables[j])?
            } else {
                matrix[j][i]
            };
        }
    }
    Ok(matrix)
}

/// Print a correlation matrix with column/row labels.
pub fn print_correlation_matrix(matrix: &[Vec<f64>], names: &[&str]) {
    let col_w = 10usize;
    let label_w = 14usize;
    print!("{:>width$}", "", width = label_w);
    for name in names {
        print!("{:>width$}", name, width = col_w);
    }
    println!();
    for (i, row) in matrix.iter().enumerate() {
        print!("{:<width$}", names[i], width = label_w);
        for val in row {
            print!("{:>width$.4}", val, width = col_w);
        }
        println!();
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

/// Convert a slice to its rank vector (average ranks for ties).
fn rank(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut indexed: Vec<(usize, f64)> = data.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && indexed[j].1 == indexed[i].1 {
            j += 1;
        }
        let avg_rank = (i + j - 1) as f64 / 2.0 + 1.0;
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    ranks
}
