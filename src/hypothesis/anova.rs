use statrs::distribution::{ContinuousCDF, FisherSnedecor};

use crate::error::{InferustError, Result};

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
            "Between groups",
            self.df_between,
            self.ss_between,
            self.ms_between,
            self.f_statistic
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
    let grand_mean: f64 =
        groups.iter().flat_map(|g| g.iter()).sum::<f64>() / n_total as f64;

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

    let f_dist = FisherSnedecor::new(df_between, df_within).map_err(|_| {
        InferustError::InvalidInput("invalid F-distribution parameters".into())
    })?;
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
