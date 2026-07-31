use statrs::distribution::{ContinuousCDF, FisherSnedecor};

use crate::error::{InferustError, Result};
use crate::regression::Ols;

/// Output of a one-way ANOVA.
#[derive(Debug, Clone)]
pub struct AnovaResult {
    pub f_statistic: f64,
    pub p_value: f64,
    pub df_between: f64,
    pub df_within: f64,
    pub ss_between: f64,
    pub ss_within: f64,
    pub ms_between: f64,
    pub ms_within: f64,
}

impl AnovaResult {
    /// Print an ANOVA table to stdout.
    pub fn print(&self) {
        println!();
        println!("── One-Way ANOVA ──────────────────────────────────────────────────");
        println!(
            "{:<16} {:>6} {:>14} {:>14} {:>10}",
            "Source", "df", "SS", "MS", "F"
        );
        println!("──────────────────────────────────────────────────────────────────");
        println!(
            "{:<16} {:>6.0} {:>14.4} {:>14.4} {:>10.4}",
            "Between groups", self.df_between, self.ss_between, self.ms_between, self.f_statistic
        );
        println!(
            "{:<16} {:>6.0} {:>14.4} {:>14.4}",
            "Within groups", self.df_within, self.ss_within, self.ms_within
        );
        println!("──────────────────────────────────────────────────────────────────");
        println!(
            " F({:.0}, {:.0}) = {:.4}   p = {:.6}",
            self.df_between, self.df_within, self.f_statistic, self.p_value
        );
        let verdict = if self.p_value < 0.05 {
            "✓ reject H₀ — group means differ (p < 0.05)"
        } else {
            "✗ fail to reject H₀ — no significant difference (p ≥ 0.05)"
        };
        println!(" {}", verdict);
    }
}

/// One-way ANOVA: tests whether the means of two or more groups are equal.
///
/// `groups` is a slice of slices; each inner slice is one group's observations.
pub fn one_way(groups: &[&[f64]]) -> Result<AnovaResult> {
    if groups.len() < 2 {
        return Err(InferustError::InsufficientData {
            needed: 2,
            got: groups.len(),
        });
    }
    for g in groups {
        if g.len() < 2 {
            return Err(InferustError::InsufficientData {
                needed: 2,
                got: g.len(),
            });
        }
    }

    let k = groups.len();
    let n_total: usize = groups.iter().map(|g| g.len()).sum();
    let grand_mean: f64 = groups.iter().flat_map(|g| g.iter()).sum::<f64>() / n_total as f64;

    let ss_between: f64 = groups
        .iter()
        .map(|g| {
            let gm = g.iter().sum::<f64>() / g.len() as f64;
            g.len() as f64 * (gm - grand_mean).powi(2)
        })
        .sum();

    let ss_within: f64 = groups
        .iter()
        .map(|g| {
            let gm = g.iter().sum::<f64>() / g.len() as f64;
            g.iter().map(|x| (x - gm).powi(2)).sum::<f64>()
        })
        .sum();

    let df_between = (k - 1) as f64;
    let df_within = (n_total - k) as f64;
    let ms_between = ss_between / df_between;
    let ms_within = ss_within / df_within;
    let f_statistic = ms_between / ms_within;

    let f_dist = FisherSnedecor::new(df_between, df_within)
        .map_err(|_| InferustError::InvalidInput("invalid F-distribution parameters".into()))?;
    let p_value = 1.0 - f_dist.cdf(f_statistic);

    Ok(AnovaResult {
        f_statistic,
        p_value,
        df_between,
        df_within,
        ss_between,
        ss_within,
        ms_between,
        ms_within,
    })
}

// ── Two-way ANOVA ─────────────────────────────────────────────────────────────

/// Sums-of-squares decomposition for [`two_way`], matching
/// `statsmodels.stats.anova.anova_lm`'s `typ` argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SsType {
    /// Sequential sums of squares (A, then B | A, then A×B | A, B).
    TypeI,
    /// Partial sums of squares for main effects, ignoring the interaction
    /// (recommended for unbalanced designs without significant interaction).
    #[default]
    TypeII,
}

/// One effect row of a two-way ANOVA table.
#[derive(Debug, Clone)]
pub struct TwoWayAnovaRow {
    /// Effect name: `"A"`, `"B"`, or `"A:B"`.
    pub source: String,
    /// Degrees of freedom.
    pub df: f64,
    /// Sum of squares.
    pub sum_sq: f64,
    /// Mean square (SS / df).
    pub mean_sq: f64,
    /// F statistic against the full-model residual mean square.
    pub f_statistic: f64,
    /// Upper-tail p-value.
    pub p_value: f64,
}

/// Output of a two-way (factorial) ANOVA with interaction.
#[derive(Debug, Clone)]
pub struct TwoWayAnovaResult {
    /// Effect rows: A, B, A:B.
    pub rows: Vec<TwoWayAnovaRow>,
    /// Residual degrees of freedom.
    pub residual_df: f64,
    /// Residual sum of squares.
    pub residual_ss: f64,
    /// Residual mean square (the F-test denominator).
    pub residual_ms: f64,
    /// Sums-of-squares type used.
    pub ss_type: SsType,
}

impl TwoWayAnovaResult {
    /// Print an ANOVA table to stdout.
    pub fn print(&self) {
        println!();
        println!(
            "── Two-Way ANOVA ({:?}) ────────────────────────────────────────────",
            self.ss_type
        );
        println!(
            "{:<10} {:>6} {:>14} {:>14} {:>10} {:>12}",
            "Source", "df", "SS", "MS", "F", "p"
        );
        println!("──────────────────────────────────────────────────────────────────");
        for row in &self.rows {
            println!(
                "{:<10} {:>6.0} {:>14.4} {:>14.4} {:>10.4} {:>12.6}",
                row.source, row.df, row.sum_sq, row.mean_sq, row.f_statistic, row.p_value
            );
        }
        println!(
            "{:<10} {:>6.0} {:>14.4} {:>14.4}",
            "Residual", self.residual_df, self.residual_ss, self.residual_ms
        );
        println!("──────────────────────────────────────────────────────────────────");
    }
}

/// Two-way ANOVA with interaction for a completely crossed design.
///
/// `factor_a` and `factor_b` give the level label of each observation.
/// Balanced and unbalanced designs are supported; sums of squares are
/// computed by nested OLS model comparison, so Type I and Type II match
/// `statsmodels.stats.anova.anova_lm(ols("y ~ C(a) * C(b)").fit(), typ=…)`.
///
/// # Example
/// ```rust
/// use inferust::hypothesis::anova::{two_way, SsType};
///
/// let y = [12.0, 14.0, 11.0, 13.0, 22.0, 25.0, 21.0, 24.0];
/// let a = ["ctl", "ctl", "ctl", "ctl", "trt", "trt", "trt", "trt"];
/// let b = ["m", "m", "f", "f", "m", "m", "f", "f"];
/// let table = two_way(&y, &a, &b, SsType::TypeII).unwrap();
/// table.print();
/// ```
pub fn two_way(
    y: &[f64],
    factor_a: &[&str],
    factor_b: &[&str],
    ss_type: SsType,
) -> Result<TwoWayAnovaResult> {
    let n = y.len();
    if factor_a.len() != n || factor_b.len() != n {
        return Err(InferustError::DimensionMismatch {
            x_rows: factor_a.len().min(factor_b.len()),
            y_len: n,
        });
    }
    let levels_a = sorted_levels(factor_a);
    let levels_b = sorted_levels(factor_b);
    if levels_a.len() < 2 || levels_b.len() < 2 {
        return Err(InferustError::InvalidInput(
            "each factor needs at least 2 levels".into(),
        ));
    }
    let df_a = (levels_a.len() - 1) as f64;
    let df_b = (levels_b.len() - 1) as f64;
    let df_ab = df_a * df_b;
    let df_resid = n as f64 - 1.0 - df_a - df_b - df_ab;
    if df_resid < 1.0 {
        return Err(InferustError::InsufficientData {
            needed: (1.0 + df_a + df_b + df_ab) as usize + 2,
            got: n,
        });
    }

    // Treatment-coded dummy columns (first level dropped), matching patsy.
    let dummies_a = dummy_columns(factor_a, &levels_a);
    let dummies_b = dummy_columns(factor_b, &levels_b);
    let mut dummies_ab: Vec<Vec<f64>> = Vec::with_capacity((dummies_a.len()) * (dummies_b.len()));
    for col_a in &dummies_a {
        for col_b in &dummies_b {
            dummies_ab.push(col_a.iter().zip(col_b.iter()).map(|(x, z)| x * z).collect());
        }
    }

    let sse_a = ols_sse(&[&dummies_a], y)?;
    let sse_b = ols_sse(&[&dummies_b], y)?;
    let sse_ab_main = ols_sse(&[&dummies_a, &dummies_b], y)?;
    let sse_full = ols_sse(&[&dummies_a, &dummies_b, &dummies_ab], y)?;

    let mean = y.iter().sum::<f64>() / n as f64;
    let sse_null: f64 = y.iter().map(|v| (v - mean).powi(2)).sum();

    let (ss_a, ss_b) = match ss_type {
        SsType::TypeI => (sse_null - sse_a, sse_a - sse_ab_main),
        SsType::TypeII => (sse_b - sse_ab_main, sse_a - sse_ab_main),
    };
    let ss_ab = sse_ab_main - sse_full;
    let ms_resid = sse_full / df_resid;

    let mut rows = Vec::with_capacity(3);
    for (source, df, ss) in [("A", df_a, ss_a), ("B", df_b, ss_b), ("A:B", df_ab, ss_ab)] {
        let ms = ss / df;
        let f_stat = ms / ms_resid;
        let f_dist = FisherSnedecor::new(df, df_resid)
            .map_err(|_| InferustError::InvalidInput("invalid F-distribution parameters".into()))?;
        rows.push(TwoWayAnovaRow {
            source: source.to_string(),
            df,
            sum_sq: ss,
            mean_sq: ms,
            f_statistic: f_stat,
            p_value: (1.0 - f_dist.cdf(f_stat)).clamp(0.0, 1.0),
        });
    }

    Ok(TwoWayAnovaResult {
        rows,
        residual_df: df_resid,
        residual_ss: sse_full,
        residual_ms: ms_resid,
        ss_type,
    })
}

/// Distinct factor levels in sorted order (deterministic dummy coding).
fn sorted_levels(labels: &[&str]) -> Vec<String> {
    let mut levels: Vec<String> = Vec::new();
    for &l in labels {
        if !levels.iter().any(|existing| existing == l) {
            levels.push(l.to_string());
        }
    }
    levels.sort();
    levels
}

/// Treatment-coded dummy columns for `labels`, dropping the first level.
fn dummy_columns(labels: &[&str], levels: &[String]) -> Vec<Vec<f64>> {
    levels[1..]
        .iter()
        .map(|level| {
            labels
                .iter()
                .map(|&l| if l == level.as_str() { 1.0 } else { 0.0 })
                .collect()
        })
        .collect()
}

/// Residual sum of squares of an OLS fit of `y` on the given column groups
/// (plus the intercept `Ols` adds automatically).
fn ols_sse(column_groups: &[&Vec<Vec<f64>>], y: &[f64]) -> Result<f64> {
    let n = y.len();
    let mut x: Vec<Vec<f64>> = vec![Vec::new(); n];
    for group in column_groups {
        for col in group.iter() {
            for (row, &v) in col.iter().enumerate() {
                x[row].push(v);
            }
        }
    }
    let fit = Ols::new().fit(&x, y)?;
    Ok(fit.residuals.iter().map(|e| e * e).sum())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_way_balanced_matches_hand_computation() {
        // 2x2 balanced design, 2 replicates per cell.
        let y = [12.0, 14.0, 11.0, 13.0, 22.0, 25.0, 21.0, 24.0];
        let a = ["ctl", "ctl", "ctl", "ctl", "trt", "trt", "trt", "trt"];
        let b = ["m", "m", "f", "f", "m", "m", "f", "f"];
        let t2 = two_way(&y, &a, &b, SsType::TypeII).unwrap();
        // In a balanced design Type I == Type II.
        let t1 = two_way(&y, &a, &b, SsType::TypeI).unwrap();
        for (r1, r2) in t1.rows.iter().zip(t2.rows.iter()) {
            assert!((r1.sum_sq - r2.sum_sq).abs() < 1e-9);
        }
        // Factor A (treatment) dominates: mean difference is 10.5.
        assert!(t2.rows[0].sum_sq > 200.0);
        // Hand computation: SS_A = 220.5, MSE = 3.25, F(1,4) ≈ 67.8, p ≈ 0.0012.
        assert!((t2.rows[0].sum_sq - 220.5).abs() < 1e-9);
        assert!(t2.rows[0].p_value < 0.01);
        assert_eq!(t2.rows[2].source, "A:B");
    }

    #[test]
    fn two_way_rejects_mismatched_lengths() {
        let y = [1.0, 2.0];
        assert!(two_way(&y, &["a"], &["x", "y"], SsType::TypeII).is_err());
    }
}
