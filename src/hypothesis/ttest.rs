use statrs::distribution::{ContinuousCDF, StudentsT};

use crate::error::{InferustError, Result};

/// Output of any t-test variant.
#[derive(Debug, Clone)]
pub struct TTestResult {
    pub statistic: f64,
    pub p_value: f64,
    pub df: f64,
    pub mean_diff: f64,
    /// 95 % confidence interval on the mean difference.
    pub conf_interval: (f64, f64),
    pub test_name: String,
}

impl TTestResult {
    pub fn print(&self) {
        println!();
        println!("── {} ──────────────────────────────", self.test_name);
        println!(
            " t = {:.4}   df = {:.2}   p = {:.6}",
            self.statistic, self.df, self.p_value
        );
        println!(" mean difference = {:.6}", self.mean_diff);
        println!(
            " 95% CI: ({:.6},  {:.6})",
            self.conf_interval.0, self.conf_interval.1
        );
        let verdict = if self.p_value < 0.05 {
            "✓ reject H₀ (p < 0.05)"
        } else {
            "✗ fail to reject H₀ (p ≥ 0.05)"
        };
        println!(" {}", verdict);
    }
}

// ── one-sample ────────────────────────────────────────────────────────────────

/// One-sample t-test: H₀: μ = `mu`.
pub fn one_sample(data: &[f64], mu: f64) -> Result<TTestResult> {
    let n = data.len();
    if n < 2 {
        return Err(InferustError::InsufficientData { needed: 2, got: n });
    }
    let mean = data.iter().sum::<f64>() / n as f64;
    let var =
        data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let se = (var / n as f64).sqrt();
    let t = (mean - mu) / se;
    let df = (n - 1) as f64;

    let dist = t_dist(df)?;
    let p = two_sided_p(&dist, t);
    let t_crit = dist.inverse_cdf(0.975);
    let diff = mean - mu;
    let ci = (diff - t_crit * se, diff + t_crit * se);

    Ok(TTestResult {
        statistic: t,
        p_value: p,
        df,
        mean_diff: diff,
        conf_interval: ci,
        test_name: "One-Sample t-Test".to_string(),
    })
}

// ── two-sample (Welch) ────────────────────────────────────────────────────────

/// Independent two-sample Welch t-test: H₀: μ₁ = μ₂.
///
/// Uses the Welch-Satterthwaite degrees of freedom — does not assume equal variance.
pub fn two_sample(a: &[f64], b: &[f64]) -> Result<TTestResult> {
    check_len(a, 2)?;
    check_len(b, 2)?;

    let (na, nb) = (a.len() as f64, b.len() as f64);
    let mean_a = mean(a);
    let mean_b = mean(b);
    let var_a = sample_var(a, mean_a);
    let var_b = sample_var(b, mean_b);

    let se = (var_a / na + var_b / nb).sqrt();
    let t = (mean_a - mean_b) / se;

    // Welch-Satterthwaite df
    let term_a = var_a / na;
    let term_b = var_b / nb;
    let df = (term_a + term_b).powi(2)
        / (term_a.powi(2) / (na - 1.0) + term_b.powi(2) / (nb - 1.0));

    let dist = t_dist(df)?;
    let p = two_sided_p(&dist, t);
    let t_crit = dist.inverse_cdf(0.975);
    let diff = mean_a - mean_b;
    let ci = (diff - t_crit * se, diff + t_crit * se);

    Ok(TTestResult {
        statistic: t,
        p_value: p,
        df,
        mean_diff: diff,
        conf_interval: ci,
        test_name: "Two-Sample Welch t-Test".to_string(),
    })
}

// ── paired ────────────────────────────────────────────────────────────────────

/// Paired t-test: H₀: mean(before − after) = 0.
pub fn paired(before: &[f64], after: &[f64]) -> Result<TTestResult> {
    if before.len() != after.len() {
        return Err(InferustError::DimensionMismatch {
            x_rows: before.len(),
            y_len: after.len(),
        });
    }
    let diffs: Vec<f64> = before.iter().zip(after.iter()).map(|(a, b)| a - b).collect();
    one_sample(&diffs, 0.0).map(|mut r| {
        r.test_name = "Paired t-Test".to_string();
        r
    })
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn check_len(data: &[f64], min: usize) -> Result<()> {
    if data.len() < min {
        Err(InferustError::InsufficientData { needed: min, got: data.len() })
    } else {
        Ok(())
    }
}

fn mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

fn sample_var(data: &[f64], m: f64) -> f64 {
    data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64
}

fn t_dist(df: f64) -> Result<StudentsT> {
    StudentsT::new(0.0, 1.0, df)
        .map_err(|_| InferustError::InvalidInput(format!("invalid df = {df}")))
}

fn two_sided_p(dist: &StudentsT, t: f64) -> f64 {
    2.0 * (1.0 - dist.cdf(t.abs()))
}
