use statrs::distribution::{ChiSquared, ContinuousCDF};

use crate::error::{InferustError, Result};

/// Output of a chi-squared test.
#[derive(Debug, Clone)]
pub struct ChiSqResult {
    /// χ² test statistic.
    pub statistic: f64,
    pub p_value: f64,
    pub df: f64,
}

impl ChiSqResult {
    pub fn print(&self) {
        println!();
        println!("── Chi-Squared Test ──────────────────────────────────");
        println!(
            " χ² = {:.4}   df = {:.0}   p = {:.6}",
            self.statistic, self.df, self.p_value
        );
        let verdict = if self.p_value < 0.05 {
            "✓ reject H₀ (p < 0.05)"
        } else {
            "✗ fail to reject H₀ (p ≥ 0.05)"
        };
        println!(" {}", verdict);
    }
}

// ── goodness-of-fit ───────────────────────────────────────────────────────────

/// Chi-squared goodness-of-fit test.
///
/// * `observed` – observed counts per category.
/// * `expected` – expected counts (uniform distribution if `None`).
pub fn goodness_of_fit(observed: &[f64], expected: Option<&[f64]>) -> Result<ChiSqResult> {
    let n = observed.len();
    if n < 2 {
        return Err(InferustError::InsufficientData { needed: 2, got: n });
    }

    let total: f64 = observed.iter().sum();
    let exp: Vec<f64> = match expected {
        Some(e) => {
            if e.len() != n {
                return Err(InferustError::DimensionMismatch {
                    x_rows: n,
                    y_len: e.len(),
                });
            }
            e.to_vec()
        }
        None => vec![total / n as f64; n],
    };

    let chi2: f64 = observed
        .iter()
        .zip(exp.iter())
        .map(|(o, e)| (o - e).powi(2) / e)
        .sum();

    let df = (n - 1) as f64;
    chi_sq_result(chi2, df)
}

// ── independence ──────────────────────────────────────────────────────────────

/// Chi-squared test of independence for a 2-D contingency table.
///
/// `table` is a slice of rows, each row a slice of observed cell counts.
pub fn independence(table: &[Vec<f64>]) -> Result<ChiSqResult> {
    let rows = table.len();
    if rows < 2 {
        return Err(InferustError::InsufficientData { needed: 2, got: rows });
    }
    let cols = table[0].len();
    if cols < 2 {
        return Err(InferustError::InvalidInput(
            "contingency table must have at least 2 columns".into(),
        ));
    }

    let row_sums: Vec<f64> = table.iter().map(|r| r.iter().sum()).collect();
    let col_sums: Vec<f64> =
        (0..cols).map(|j| table.iter().map(|r| r[j]).sum()).collect();
    let total: f64 = row_sums.iter().sum();

    let chi2: f64 = (0..rows)
        .flat_map(|i| (0..cols).map(move |j| (i, j)))
        .map(|(i, j)| {
            let expected = row_sums[i] * col_sums[j] / total;
            (table[i][j] - expected).powi(2) / expected
        })
        .sum();

    let df = ((rows - 1) * (cols - 1)) as f64;
    chi_sq_result(chi2, df)
}

// ── helper ────────────────────────────────────────────────────────────────────

fn chi_sq_result(chi2: f64, df: f64) -> Result<ChiSqResult> {
    let dist = ChiSquared::new(df)
        .map_err(|_| InferustError::InvalidInput(format!("invalid df = {df}")))?;
    let p_value = 1.0 - dist.cdf(chi2);
    Ok(ChiSqResult {
        statistic: chi2,
        p_value,
        df,
    })
}
