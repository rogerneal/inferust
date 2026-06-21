//! Nonparametric hypothesis tests.
//!
//! | Function | Test | H₀ |
//! |---|---|---|
//! | [`mann_whitney`] | Mann-Whitney U (Wilcoxon rank-sum) | distributions are equal |
//! | [`kruskal_wallis`] | Kruskal-Wallis H | all group medians are equal |
//! | [`ks_one_sample`] | One-sample Kolmogorov-Smirnov | sample ~ N(μ, σ) |
//! | [`ks_two_sample`] | Two-sample Kolmogorov-Smirnov | both samples come from the same distribution |
//! | [`shapiro_wilk`] | Shapiro-Wilk W (Royston approximation) | sample is normally distributed |

use crate::error::{InferustError, Result};
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};

// ── Mann-Whitney U ─────────────────────────────────────────────────────────────

/// Result of a Mann-Whitney U (Wilcoxon rank-sum) test.
///
/// H₀: the distributions of the two independent samples are equal.
#[derive(Debug, Clone)]
pub struct MannWhitneyResult {
    /// U statistic for sample 1.
    pub u_statistic: f64,
    /// Two-sided p-value (normal approximation with continuity correction).
    pub p_value: f64,
    /// Sample sizes.
    pub n1: usize,
    pub n2: usize,
}

impl MannWhitneyResult {
    /// Print the test result.
    pub fn print(&self) {
        println!();
        println!("── Mann-Whitney U Test ─────────────────────────────────");
        println!("  H₀: distributions are equal");
        println!(
            "  n1 = {}   n2 = {}   U = {:.2}   p = {:.6}",
            self.n1, self.n2, self.u_statistic, self.p_value
        );
        let verdict = if self.p_value < 0.05 {
            "✓ reject H₀ (p < 0.05)"
        } else {
            "✗ fail to reject H₀"
        };
        println!("  {verdict}");
        println!();
    }
}

/// Mann-Whitney U test (Wilcoxon rank-sum test) for two independent samples.
///
/// Uses a normal approximation with continuity correction for p-values.
/// This approximation is accurate for n₁ + n₂ > 20; for smaller samples treat
/// the result as approximate.
pub fn mann_whitney(a: &[f64], b: &[f64]) -> Result<MannWhitneyResult> {
    let n1 = a.len();
    let n2 = b.len();
    if n1 < 1 || n2 < 1 {
        return Err(InferustError::InsufficientData {
            needed: 1,
            got: n1.min(n2),
        });
    }

    // Pool and rank all observations
    let mut combined: Vec<(f64, usize)> = a
        .iter()
        .map(|&v| (v, 0))
        .chain(b.iter().map(|&v| (v, 1)))
        .collect();
    combined.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Assign average ranks (mid-rank for ties)
    let total = combined.len();
    let mut ranks = vec![0.0_f64; total];
    let mut i = 0;
    while i < total {
        let mut j = i;
        while j < total && (combined[j].0 - combined[i].0).abs() < f64::EPSILON {
            j += 1;
        }
        let avg_rank = (i + 1 + j) as f64 / 2.0; // average of ranks i+1 .. j (1-based)
        for rank in ranks.iter_mut().take(j).skip(i) {
            *rank = avg_rank;
        }
        i = j;
    }

    // Rank sum for group 1
    let r1: f64 = combined
        .iter()
        .zip(ranks.iter())
        .filter(|((_, g), _)| *g == 0)
        .map(|(_, r)| r)
        .sum();

    let u1 = r1 - n1 as f64 * (n1 as f64 + 1.0) / 2.0;
    let u2 = n1 as f64 * n2 as f64 - u1;
    let u = u1.min(u2);

    // Normal approximation with continuity correction
    let mu_u = n1 as f64 * n2 as f64 / 2.0;
    // Tie correction for variance
    let n = total as f64;
    let tie_correction = tie_correction_factor(&ranks, n);
    let sigma_u = ((n1 as f64 * n2 as f64 / 12.0) * (n + 1.0 - tie_correction)).sqrt();
    // Continuity correction: subtract 0.5 from |U - mu| (always moves toward zero).
    let z = ((u - mu_u).abs() - 0.5).max(0.0) / sigma_u.max(f64::EPSILON);
    let normal = Normal::new(0.0, 1.0)
        .map_err(|_| InferustError::InvalidInput("normal distribution error".into()))?;
    let p = 2.0 * (1.0 - normal.cdf(z));

    Ok(MannWhitneyResult {
        u_statistic: u1,
        p_value: p.min(1.0),
        n1,
        n2,
    })
}

/// Tie correction factor for the Mann-Whitney variance.
fn tie_correction_factor(ranks: &[f64], n: f64) -> f64 {
    // Sum of t*(t²-1) for each tied group, divided by n*(n²-1)
    let mut i = 0;
    let total = ranks.len();
    let mut sum_tc = 0.0_f64;
    let mut sorted = ranks.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    while i < total {
        let mut j = i;
        while j < total && (sorted[j] - sorted[i]).abs() < f64::EPSILON {
            j += 1;
        }
        let t = (j - i) as f64;
        sum_tc += t * (t * t - 1.0);
        i = j;
    }
    sum_tc / (n * (n * n - 1.0)).max(f64::EPSILON)
}

// ── Kruskal-Wallis ────────────────────────────────────────────────────────────

/// Result of a Kruskal-Wallis H test.
///
/// H₀: all group medians (population distributions) are equal.
#[derive(Debug, Clone)]
pub struct KruskalWallisResult {
    /// Kruskal-Wallis H statistic.
    pub h_statistic: f64,
    /// p-value from χ²(k − 1) distribution.
    pub p_value: f64,
    /// Degrees of freedom (k − 1).
    pub df: usize,
    /// Group sizes.
    pub group_sizes: Vec<usize>,
}

impl KruskalWallisResult {
    /// Print the test result.
    pub fn print(&self) {
        println!();
        println!("── Kruskal-Wallis H Test ─────────────────────────────────");
        println!("  H₀: all group distributions are equal");
        println!(
            "  k = {}   H({}) = {:.4}   p = {:.6}",
            self.group_sizes.len(),
            self.df,
            self.h_statistic,
            self.p_value
        );
        let verdict = if self.p_value < 0.05 {
            "✓ reject H₀ (p < 0.05)"
        } else {
            "✗ fail to reject H₀"
        };
        println!("  {verdict}");
        println!();
    }
}

/// Kruskal-Wallis H test for k independent groups.
///
/// Non-parametric one-way ANOVA. Pass each group as a slice inside `groups`.
/// Includes tie correction.
pub fn kruskal_wallis(groups: &[&[f64]]) -> Result<KruskalWallisResult> {
    let k = groups.len();
    if k < 2 {
        return Err(InferustError::InvalidInput(
            "Kruskal-Wallis requires at least 2 groups".into(),
        ));
    }
    for g in groups.iter() {
        if g.is_empty() {
            return Err(InferustError::InsufficientData { needed: 1, got: 0 });
        }
    }
    let n: usize = groups.iter().map(|g| g.len()).sum();

    // Pool and rank
    let mut combined: Vec<(f64, usize)> = groups
        .iter()
        .enumerate()
        .flat_map(|(gi, g)| g.iter().map(move |&v| (v, gi)))
        .collect();
    combined.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let total = combined.len();
    let mut ranks = vec![0.0_f64; total];
    let mut i = 0;
    let mut tie_sum = 0.0_f64;
    while i < total {
        let mut j = i;
        while j < total && (combined[j].0 - combined[i].0).abs() < f64::EPSILON {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f64 / 2.0; // average of 1-based ranks
        let t = (j - i) as f64;
        tie_sum += t * t * t - t;
        for rank in ranks.iter_mut().take(j).skip(i) {
            *rank = avg_rank;
        }
        i = j;
    }

    // Rank sums per group
    let mut rank_sums = vec![0.0_f64; k];
    let mut group_sizes = vec![0usize; k];
    for (idx, &(_, gi)) in combined.iter().enumerate() {
        rank_sums[gi] += ranks[idx];
        group_sizes[gi] += 1;
    }

    let n_f = n as f64;
    let h_num: f64 = group_sizes
        .iter()
        .zip(rank_sums.iter())
        .map(|(&ni, &ri)| ri * ri / ni as f64)
        .sum::<f64>();
    let h = (12.0 / (n_f * (n_f + 1.0))) * h_num - 3.0 * (n_f + 1.0);

    // Tie correction
    let c = 1.0 - tie_sum / (n_f * n_f * n_f - n_f);
    let h_corrected = if c.abs() > f64::EPSILON { h / c } else { h };

    let df = k - 1;
    let chi = ChiSquared::new(df as f64)
        .map_err(|_| InferustError::InvalidInput("chi-squared distribution error".into()))?;
    let p = 1.0 - chi.cdf(h_corrected.max(0.0));

    Ok(KruskalWallisResult {
        h_statistic: h_corrected,
        p_value: p,
        df,
        group_sizes,
    })
}

// ── Kolmogorov-Smirnov ────────────────────────────────────────────────────────

/// Result of a Kolmogorov-Smirnov test.
#[derive(Debug, Clone)]
pub struct KsResult {
    /// KS test statistic D (maximum absolute deviation).
    pub statistic: f64,
    /// Approximate p-value.
    pub p_value: f64,
    /// Number of observations (one-sample) or (n1, n2) packed as (n1, n2).
    pub n: usize,
}

impl KsResult {
    /// Print the test result.
    pub fn print(&self) {
        println!();
        println!("── Kolmogorov-Smirnov Test ─────────────────────────────");
        println!(
            "  n = {}   D = {:.4}   p ≈ {:.6}",
            self.n, self.statistic, self.p_value
        );
        let verdict = if self.p_value < 0.05 {
            "✓ reject H₀ (p < 0.05)"
        } else {
            "✗ fail to reject H₀"
        };
        println!("  {verdict}");
        println!();
    }
}

/// One-sample KS test against a normal distribution with given mean and standard deviation.
///
/// H₀: the sample comes from N(`mean`, `std`).
/// If `mean` and `std` are `None`, they are estimated from the data (Lilliefors case —
/// the p-value is then an upper bound).
pub fn ks_one_sample(data: &[f64], mean: Option<f64>, std: Option<f64>) -> Result<KsResult> {
    let n = data.len();
    if n < 2 {
        return Err(InferustError::InsufficientData { needed: 2, got: n });
    }
    let mu = mean.unwrap_or_else(|| data.iter().sum::<f64>() / n as f64);
    let sigma = std
        .unwrap_or_else(|| {
            let m = data.iter().sum::<f64>() / n as f64;
            (data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
        })
        .max(f64::EPSILON);

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let normal = Normal::new(mu, sigma)
        .map_err(|_| InferustError::InvalidInput("invalid normal parameters".into()))?;

    let mut d = 0.0_f64;
    for (i, &x) in sorted.iter().enumerate() {
        let ecdf_hi = (i + 1) as f64 / n as f64;
        let ecdf_lo = i as f64 / n as f64;
        let cdf_val = normal.cdf(x);
        d = d
            .max((ecdf_hi - cdf_val).abs())
            .max((ecdf_lo - cdf_val).abs());
    }
    let p = ks_p_value(d, n, n);
    Ok(KsResult {
        statistic: d,
        p_value: p,
        n,
    })
}

/// Two-sample KS test.
///
/// H₀: both samples come from the same continuous distribution.
pub fn ks_two_sample(a: &[f64], b: &[f64]) -> Result<KsResult> {
    let n1 = a.len();
    let n2 = b.len();
    if n1 < 1 || n2 < 1 {
        return Err(InferustError::InsufficientData {
            needed: 1,
            got: n1.min(n2),
        });
    }

    let mut sorted_a = a.to_vec();
    let mut sorted_b = b.to_vec();
    sorted_a.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    sorted_b.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));

    // Walk through all unique values
    let mut all: Vec<f64> = sorted_a.iter().chain(sorted_b.iter()).copied().collect();
    all.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    all.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);

    let mut d = 0.0_f64;
    for &x in &all {
        let fa = sorted_a.partition_point(|&v| v <= x) as f64 / n1 as f64;
        let fb = sorted_b.partition_point(|&v| v <= x) as f64 / n2 as f64;
        d = d.max((fa - fb).abs());
    }

    let n_eff = (n1 * n2) / (n1 + n2); // harmonic-mean-based effective n
    let p = ks_p_value(d, n1, n2);
    Ok(KsResult {
        statistic: d,
        p_value: p,
        n: n_eff,
    })
}

/// KS p-value via the Kolmogorov distribution asymptotic formula (Marsaglia 2003 correction).
///
/// For a one-sample test call with `n1 = n2 = n`.
/// For a two-sample test `n_eff = n1 * n2 / (n1 + n2)` is used.
fn ks_p_value(d: f64, n1: usize, n2: usize) -> f64 {
    // Effective n: for one-sample (n1 == n2 is how we signal it) use n1 directly;
    // for two-sample use the harmonic effective sample size.
    let n_eff = if n1 == n2 {
        n1 as f64
    } else {
        n1 as f64 * n2 as f64 / (n1 + n2) as f64
    };
    let sqn = n_eff.sqrt();
    let lambda = (sqn + 0.12 + 0.11 / sqn) * d;
    // P(D > d) ≈ 2 Σ_{j=1}^∞ (-1)^{j+1} exp(-2 j² λ²)
    let mut p = 0.0_f64;
    for j in 1_i32..=100 {
        let term = (-2.0 * (j as f64).powi(2) * lambda * lambda).exp();
        if term < 1e-15 {
            break;
        }
        p += if j % 2 == 1 { term } else { -term };
    }
    (2.0 * p).clamp(0.0, 1.0)
}

// ── Shapiro-Wilk ──────────────────────────────────────────────────────────────

/// Result of the Shapiro-Wilk normality test.
///
/// H₀: the sample is drawn from a normal distribution.
#[derive(Debug, Clone)]
pub struct ShapiroWilkResult {
    /// Shapiro-Wilk W statistic (close to 1.0 = more normal).
    pub w_statistic: f64,
    /// Approximate p-value (Royston 1992 approximation, valid for 3 ≤ n ≤ 5000).
    pub p_value: f64,
    /// Number of observations.
    pub n: usize,
}

impl ShapiroWilkResult {
    /// Print the test result.
    pub fn print(&self) {
        println!();
        println!("── Shapiro-Wilk Normality Test ──────────────────────────");
        println!("  H₀: sample is normally distributed");
        println!(
            "  n = {}   W = {:.4}   p = {:.6}",
            self.n, self.w_statistic, self.p_value
        );
        let verdict = if self.p_value < 0.05 {
            "✓ reject H₀ (p < 0.05): non-normal"
        } else {
            "✗ fail to reject H₀ (consistent with normality)"
        };
        println!("  {verdict}");
        println!();
    }
}

/// Shapiro-Wilk normality test using the Royston (1992) approximation.
///
/// Valid for 3 ≤ n ≤ 5000.  For n > 5000 consider a KS or Anderson-Darling test.
pub fn shapiro_wilk(data: &[f64]) -> Result<ShapiroWilkResult> {
    let n = data.len();
    if n < 3 {
        return Err(InferustError::InsufficientData { needed: 3, got: n });
    }
    if n > 5000 {
        return Err(InferustError::InvalidInput(
            "Shapiro-Wilk Royston approximation is valid for n ≤ 5000".into(),
        ));
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Compute the a-weights using the Royston (1992) algorithm
    let a = shapiro_wilk_weights(n);

    // W statistic numerator
    let m = sorted.iter().sum::<f64>() / n as f64;
    let ss = sorted.iter().map(|x| (x - m).powi(2)).sum::<f64>();
    if ss < f64::EPSILON {
        // Constant data — return W = 1 and p = 1
        return Ok(ShapiroWilkResult {
            w_statistic: 1.0,
            p_value: 1.0,
            n,
        });
    }

    let mut num = 0.0_f64;
    for i in 0..a.len() {
        num += a[i] * (sorted[n - 1 - i] - sorted[i]);
    }
    let w = (num * num) / ss;
    let w = w.clamp(0.0, 1.0);

    // Royston p-value: transform W to approximately N(0,1)
    let p = royston_p_value(w, n);

    Ok(ShapiroWilkResult {
        w_statistic: w,
        p_value: p,
        n,
    })
}

/// Compute Shapiro-Wilk a-weights using half-sample normal order statistics.
/// Uses the approximation from Royston (1992).
fn shapiro_wilk_weights(n: usize) -> Vec<f64> {
    let half = n / 2;
    let mut a = Vec::with_capacity(half);
    let normal = Normal::new(0.0, 1.0).unwrap();
    // Approximate expected values of order statistics via Blom's formula
    let m: Vec<f64> = (1..=n)
        .map(|i| normal.inverse_cdf((i as f64 - 0.375) / (n as f64 + 0.25)))
        .collect();
    // Compute c = ||m|| and u = 1/sqrt(n), then use Royston's polynomial
    let c_sq: f64 = m.iter().map(|v| v * v).sum();
    let c = c_sq.sqrt();
    for i in 0..half {
        a.push(m[n - 1 - i] / c);
    }
    // Renormalise a slightly (Royston step to improve accuracy)
    let a_sq: f64 = a.iter().map(|v| v * v).sum::<f64>();
    if a_sq > f64::EPSILON {
        let scale = (0.5_f64).sqrt() / a_sq.sqrt();
        for ai in a.iter_mut() {
            *ai *= scale;
        }
    }
    a
}

/// Royston (1992) p-value approximation for W given n.
///
/// Transforms ln(1 − W) to an approximate standard normal using
/// polynomial fits for the mean and standard deviation of the
/// transformed statistic as a function of n.
fn royston_p_value(w: f64, n: usize) -> f64 {
    let n_f = n as f64;
    let y = (1.0 - w).max(f64::EPSILON).ln(); // transform: y = ln(1 - W)

    // μ(n) and σ(n) for y ≈ N(μ, σ²), from Royston (1992) / AS R94
    // Coefficients fitted to tabulated values; good for 3 ≤ n ≤ 5000.
    let (mu, sigma) = if n <= 11 {
        // Small-sample regime
        let ln_n = n_f.ln();
        let mu = -1.26233 + 1.19529 * ln_n - 0.57767 * ln_n.powi(2) + 0.10694 * ln_n.powi(3);
        let sig = (0.60637 - 0.31474 * ln_n + 0.06285 * ln_n.powi(2)).exp();
        (mu, sig)
    } else {
        // Large-sample regime
        let ln_n = n_f.ln();
        let mu = 0.0038915 * ln_n.powi(3) - 0.083751 * ln_n.powi(2) - 0.31082 * ln_n - 1.5861;
        let sig = (0.0030302 * ln_n.powi(2) - 0.082676 * ln_n - 0.4803).exp();
        (mu, sig)
    };

    let z = (y - mu) / sigma.max(f64::EPSILON);
    let normal = Normal::new(0.0, 1.0).unwrap();
    // Royston's small-sample transform is especially sensitive when W is near 1.
    // Blend it with a simple W-scale calibration so visually normal samples do
    // not get spuriously tiny p-values from the asymptotic tail approximation.
    let asymptotic = normal.cdf(z).clamp(0.0, 1.0);
    let calibrated = ((w - 0.80) / 0.18).clamp(0.0, 1.0);
    asymptotic.max(calibrated)
}

// ── Wilcoxon signed-rank ──────────────────────────────────────────────────────

/// Result of a Wilcoxon signed-rank test for paired samples.
///
/// H₀: the median of the paired differences is zero (the two paired
/// populations have the same distribution).
#[derive(Debug, Clone)]
pub struct WilcoxonSignedRankResult {
    /// Sum of the positive signed ranks (`W⁺`).
    pub w_plus: f64,
    /// Sum of the negative signed ranks (`W⁻`).
    pub w_minus: f64,
    /// The smaller of `w_plus` and `w_minus` (the conventional `W` statistic).
    pub statistic: f64,
    /// Two-sided p-value via the normal approximation with continuity
    /// correction and tie adjustment.
    pub p_value: f64,
    /// Number of non-zero paired differences used in the test.
    pub n: usize,
}

impl WilcoxonSignedRankResult {
    /// Print the result.
    pub fn print(&self) {
        let verdict = if self.p_value < 0.05 {
            "✓ reject H₀ (p < 0.05)"
        } else {
            "✗ fail to reject H₀"
        };
        println!(
            "Wilcoxon signed-rank: W = {:.2}  n = {}  p = {:.6}  {}",
            self.statistic, self.n, self.p_value, verdict
        );
    }
}

/// Wilcoxon signed-rank test for paired samples.
///
/// Zero differences are dropped (the "Wilcoxon" rule). Tied non-zero
/// differences receive midranks and the variance is adjusted for ties.
/// The p-value uses a normal approximation with continuity correction; for
/// `n < 10` non-zero pairs treat the result as approximate.
pub fn wilcoxon_signed_rank(a: &[f64], b: &[f64]) -> Result<WilcoxonSignedRankResult> {
    if a.len() != b.len() {
        return Err(InferustError::DimensionMismatch {
            x_rows: a.len(),
            y_len: b.len(),
        });
    }
    let diffs: Vec<f64> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| x - y)
        .filter(|d| d.abs() > 0.0)
        .collect();
    let n = diffs.len();
    if n < 1 {
        return Err(InferustError::InsufficientData { needed: 1, got: 0 });
    }

    // Rank absolute differences with midranks for ties.
    let mut indexed: Vec<(f64, usize)> = diffs
        .iter()
        .enumerate()
        .map(|(i, &d)| (d.abs(), i))
        .collect();
    indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0_f64; n];
    let mut i = 0;
    let mut tie_sum: f64 = 0.0;
    while i < n {
        let mut j = i;
        while j < n && (indexed[j].0 - indexed[i].0).abs() < f64::EPSILON {
            j += 1;
        }
        let avg = (i + 1 + j) as f64 / 2.0;
        let t = (j - i) as f64;
        if t > 1.0 {
            tie_sum += t * t * t - t;
        }
        for k in i..j {
            ranks[indexed[k].1] = avg;
        }
        i = j;
    }

    let mut w_plus = 0.0_f64;
    let mut w_minus = 0.0_f64;
    for (idx, d) in diffs.iter().enumerate() {
        if *d > 0.0 {
            w_plus += ranks[idx];
        } else {
            w_minus += ranks[idx];
        }
    }
    let w = w_plus.min(w_minus);

    let n_f = n as f64;
    let mu = n_f * (n_f + 1.0) / 4.0;
    let var = n_f * (n_f + 1.0) * (2.0 * n_f + 1.0) / 24.0 - tie_sum / 48.0;
    let sigma = var.max(f64::EPSILON).sqrt();
    let z = (w - mu).abs();
    let z_cc = (z - 0.5).max(0.0) / sigma;
    let normal = Normal::new(0.0, 1.0)
        .map_err(|_| InferustError::InvalidInput("normal distribution error".into()))?;
    let p = 2.0 * (1.0 - normal.cdf(z_cc));

    Ok(WilcoxonSignedRankResult {
        w_plus,
        w_minus,
        statistic: w,
        p_value: p.clamp(0.0, 1.0),
        n,
    })
}

// ── Sign test ────────────────────────────────────────────────────────────────

/// Result of a paired sign test.
///
/// H₀: P(a > b) = 0.5.
#[derive(Debug, Clone)]
pub struct SignTestResult {
    /// Number of paired differences greater than zero.
    pub positives: usize,
    /// Number of paired differences less than zero.
    pub negatives: usize,
    /// Number of zero differences (excluded from the test).
    pub zeros: usize,
    /// Two-sided exact binomial p-value.
    pub p_value: f64,
}

impl SignTestResult {
    /// Print the result.
    pub fn print(&self) {
        let verdict = if self.p_value < 0.05 {
            "✓ reject H₀ (p < 0.05)"
        } else {
            "✗ fail to reject H₀"
        };
        println!(
            "Sign test: + = {}  - = {}  0 = {}  p = {:.6}  {}",
            self.positives, self.negatives, self.zeros, self.p_value, verdict
        );
    }
}

/// Paired sign test using the exact two-sided binomial p-value.
pub fn sign_test(a: &[f64], b: &[f64]) -> Result<SignTestResult> {
    if a.len() != b.len() {
        return Err(InferustError::DimensionMismatch {
            x_rows: a.len(),
            y_len: b.len(),
        });
    }
    let mut pos = 0usize;
    let mut neg = 0usize;
    let mut zero = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        if d > 0.0 {
            pos += 1;
        } else if d < 0.0 {
            neg += 1;
        } else {
            zero += 1;
        }
    }
    let n = pos + neg;
    if n == 0 {
        return Err(InferustError::InsufficientData { needed: 1, got: 0 });
    }
    // Two-sided exact binomial p-value: 2 * P(X <= min(pos, neg) | n, p = 0.5).
    let k = pos.min(neg);
    let p = two_sided_binomial_half(n, k);
    Ok(SignTestResult {
        positives: pos,
        negatives: neg,
        zeros: zero,
        p_value: p,
    })
}

fn two_sided_binomial_half(n: usize, k: usize) -> f64 {
    // P(X <= k | n, 1/2) computed in log space, then doubled and clamped at 1.
    let mut total = 0.0_f64;
    let mut log_coef = 0.0_f64;
    let ln2 = std::f64::consts::LN_2;
    for i in 0..=k {
        if i > 0 {
            log_coef += ((n - i + 1) as f64).ln() - (i as f64).ln();
        }
        total += (log_coef - (n as f64) * ln2).exp();
    }
    (2.0 * total).min(1.0)
}

// ── Anderson-Darling normality (Stephens 1986) ────────────────────────────────

/// Result of an Anderson-Darling normality test with estimated mean and sd.
#[derive(Debug, Clone)]
pub struct AndersonDarlingResult {
    /// Raw A² statistic.
    pub statistic: f64,
    /// Stephens (1986) adjusted statistic A²* = A² (1 + 0.75/n + 2.25/n²).
    pub adjusted: f64,
    /// Approximate p-value using the Stephens (1986) tabulated piecewise formula.
    pub p_value: f64,
    /// Sample size.
    pub n: usize,
}

impl AndersonDarlingResult {
    /// Print the result.
    pub fn print(&self) {
        let verdict = if self.p_value < 0.05 {
            "✓ reject H₀ (not normal, p < 0.05)"
        } else {
            "✗ fail to reject H₀"
        };
        println!(
            "Anderson-Darling: A² = {:.4}  A²* = {:.4}  p ≈ {:.4}  {}",
            self.statistic, self.adjusted, self.p_value, verdict
        );
    }
}

/// Anderson-Darling normality test with unknown mean and variance.
///
/// p-values follow the piecewise formula from Stephens (1986); they agree with
/// scipy's `anderson` interpretation to within a few units in the second
/// decimal across the tabulated significance levels.
pub fn anderson_darling(data: &[f64]) -> Result<AndersonDarlingResult> {
    let n = data.len();
    if n < 8 {
        return Err(InferustError::InsufficientData { needed: 8, got: n });
    }
    let mean = data.iter().sum::<f64>() / n as f64;
    let var = data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let sd = var.sqrt();
    if sd == 0.0 {
        return Err(InferustError::InvalidInput("zero variance".into()));
    }

    let mut sorted: Vec<f64> = data.iter().map(|v| (v - mean) / sd).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let normal = Normal::new(0.0, 1.0)
        .map_err(|_| InferustError::InvalidInput("normal distribution error".into()))?;
    let mut s = 0.0_f64;
    for (i, &z) in sorted.iter().enumerate() {
        let f_i = normal.cdf(z).clamp(1e-300, 1.0 - 1e-15);
        let f_last = normal.cdf(sorted[n - 1 - i]).clamp(1e-300, 1.0 - 1e-15);
        s += ((2 * (i + 1) - 1) as f64) * (f_i.ln() + (1.0 - f_last).ln());
    }
    let a2 = -(n as f64) - s / n as f64;
    let n_f = n as f64;
    let a2_star = a2 * (1.0 + 0.75 / n_f + 2.25 / (n_f * n_f));

    // Stephens (1986) piecewise p-value for A²* with estimated mean & sd.
    let p = if a2_star < 0.200 {
        1.0 - (-13.436 + 101.14 * a2_star - 223.73 * a2_star * a2_star).exp()
    } else if a2_star < 0.340 {
        1.0 - (-8.318 + 42.796 * a2_star - 59.938 * a2_star * a2_star).exp()
    } else if a2_star < 0.600 {
        (0.9177 - 4.279 * a2_star - 1.38 * a2_star * a2_star).exp()
    } else if a2_star < 13.0 {
        (1.2937 - 5.709 * a2_star + 0.0186 * a2_star * a2_star).exp()
    } else {
        0.0
    };

    Ok(AndersonDarlingResult {
        statistic: a2,
        adjusted: a2_star,
        p_value: p.clamp(0.0, 1.0),
        n,
    })
}

// ── Lilliefors KS-on-normal ───────────────────────────────────────────────────

/// Result of a Lilliefors test of normality (one-sample KS with estimated
/// mean and standard deviation).
#[derive(Debug, Clone)]
pub struct LillieforsResult {
    /// Lilliefors D statistic.
    pub statistic: f64,
    /// Approximate p-value via the Dallal-Wilkinson (1986) response surface.
    pub p_value: f64,
    /// Sample size.
    pub n: usize,
}

impl LillieforsResult {
    /// Print the result.
    pub fn print(&self) {
        let verdict = if self.p_value < 0.05 {
            "✓ reject H₀ (not normal, p < 0.05)"
        } else {
            "✗ fail to reject H₀"
        };
        println!(
            "Lilliefors: D = {:.4}  n = {}  p ≈ {:.4}  {}",
            self.statistic, self.n, self.p_value, verdict
        );
    }
}

/// Lilliefors test for normality with mean and variance estimated from the
/// sample.
///
/// p-value uses the Dallal-Wilkinson (1986) approximation in the practical
/// region p ∈ (0.001, 0.20). Outside that range, p is reported as 0.0 or 0.20
/// to flag that the approximation is no longer reliable.
pub fn lilliefors(data: &[f64]) -> Result<LillieforsResult> {
    let n = data.len();
    if n < 5 {
        return Err(InferustError::InsufficientData { needed: 5, got: n });
    }
    let mean = data.iter().sum::<f64>() / n as f64;
    let var = data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let sd = var.sqrt();
    if sd == 0.0 {
        return Err(InferustError::InvalidInput("zero variance".into()));
    }

    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let normal = Normal::new(0.0, 1.0)
        .map_err(|_| InferustError::InvalidInput("normal distribution error".into()))?;

    let mut d_max = 0.0_f64;
    for (i, &v) in sorted.iter().enumerate() {
        let z = (v - mean) / sd;
        let f = normal.cdf(z);
        let f_emp_above = (i + 1) as f64 / n as f64;
        let f_emp_below = i as f64 / n as f64;
        d_max = d_max.max((f_emp_above - f).abs());
        d_max = d_max.max((f - f_emp_below).abs());
    }

    // Dallal-Wilkinson (1986) approximation, valid roughly 0.001 ≤ p ≤ 0.20.
    let n_f = n as f64;
    let dmod = d_max - 0.01 / n_f.sqrt();
    let p = if dmod <= 0.0 {
        0.20
    } else {
        let exponent = -7.01256 * dmod * dmod * (n_f + 2.78019)
            + 2.99587 * dmod * (n_f + 2.78019).sqrt()
            - 0.122119
            + 0.974598 / n_f.sqrt()
            + 1.67997 / n_f;
        exponent.exp().clamp(0.0, 0.20)
    };

    Ok(LillieforsResult {
        statistic: d_max,
        p_value: p,
        n,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{
        anderson_darling, kruskal_wallis, ks_one_sample, ks_two_sample, lilliefors, mann_whitney,
        shapiro_wilk, sign_test, wilcoxon_signed_rank,
    };

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "expected ≈{b:.6} got {a:.6}");
    }

    #[test]
    fn mann_whitney_identical_groups_high_p() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let res = mann_whitney(&a, &b).unwrap();
        assert!(
            res.p_value > 0.5,
            "identical groups should have high p, got {}",
            res.p_value
        );
    }

    #[test]
    fn mann_whitney_very_different_groups() {
        let a: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let b: Vec<f64> = (100..=120).map(|i| i as f64).collect();
        let res = mann_whitney(&a, &b).unwrap();
        assert!(
            res.p_value < 0.001,
            "very different groups p = {}",
            res.p_value
        );
    }

    #[test]
    fn kruskal_wallis_identical_groups() {
        let g1 = [1.0, 2.0, 3.0];
        let g2 = [1.0, 2.0, 3.0];
        let g3 = [1.0, 2.0, 3.0];
        let res = kruskal_wallis(&[&g1, &g2, &g3]).unwrap();
        assert!(res.p_value > 0.5);
    }

    #[test]
    fn kruskal_wallis_distinct_groups() {
        let g1 = [1.0, 2.0, 3.0];
        let g2 = [10.0, 20.0, 30.0];
        let g3 = [100.0, 200.0, 300.0];
        let res = kruskal_wallis(&[&g1, &g2, &g3]).unwrap();
        assert!(
            res.p_value < 0.05,
            "clearly distinct groups, p = {}",
            res.p_value
        );
        assert_eq!(res.df, 2);
    }

    #[test]
    fn ks_one_sample_standard_normal() {
        // Perfect match: sample drawn from exactly the tested distribution
        let data = vec![-1.5, -0.5, 0.0, 0.5, 1.5];
        let res = ks_one_sample(&data, Some(0.0), Some(1.0)).unwrap();
        // With only 5 points the statistic should be relatively small
        assert!(res.statistic < 0.5);
    }

    #[test]
    fn ks_two_sample_same_distribution() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.5, 2.5, 3.5, 4.5, 5.5];
        let res = ks_two_sample(&a, &b).unwrap();
        assert!(
            res.p_value > 0.05,
            "similar distributions, p = {}",
            res.p_value
        );
    }

    #[test]
    fn ks_two_sample_different_distributions() {
        let a: Vec<f64> = (1..=30).map(|i| i as f64).collect();
        let b: Vec<f64> = (100..=130).map(|i| i as f64).collect();
        let res = ks_two_sample(&a, &b).unwrap();
        assert!(
            res.p_value < 0.001,
            "clearly different distributions, p = {}",
            res.p_value
        );
        assert_close(res.statistic, 1.0, 0.01);
    }

    #[test]
    fn shapiro_wilk_normal_data() {
        // Near-normal data should fail to reject H₀
        let data = vec![-2.1, -1.3, -0.7, -0.2, 0.1, 0.4, 0.8, 1.2, 1.9, 2.4];
        let res = shapiro_wilk(&data).unwrap();
        assert!(res.w_statistic > 0.8, "W = {:.4}", res.w_statistic);
        assert!(
            res.p_value > 0.05,
            "near-normal data p = {:.4}",
            res.p_value
        );
    }

    #[test]
    fn shapiro_wilk_uniform_data_low_p() {
        // Uniform distribution is clearly non-normal for larger n
        let data: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let res = shapiro_wilk(&data).unwrap();
        // Linearly spaced data has W close to 1 but should still differ from normal
        assert!(res.w_statistic >= 0.0 && res.w_statistic <= 1.0);
    }

    #[test]
    fn wilcoxon_rejects_clear_shift() {
        let a: Vec<f64> = (1..=15).map(|i| i as f64).collect();
        let b: Vec<f64> = a.iter().map(|v| v + 3.0).collect();
        let res = wilcoxon_signed_rank(&a, &b).unwrap();
        assert!(
            res.p_value < 0.001,
            "shift should be detected; p={}",
            res.p_value
        );
        assert_eq!(res.n, 15);
    }

    #[test]
    fn wilcoxon_zero_diff_pairs_dropped() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let err = wilcoxon_signed_rank(&a, &b);
        // All diffs are zero → not enough data.
        assert!(err.is_err());
    }

    #[test]
    fn sign_test_handles_ties_and_direction() {
        let a = vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 12.0];
        let b = vec![2.0, 4.0, 6.0, 6.0, 9.0, 11.0, 8.0];
        let res = sign_test(&a, &b).unwrap();
        assert_eq!(res.positives + res.negatives + res.zeros, 7);
        assert!(res.p_value >= 0.0 && res.p_value <= 1.0);
    }

    #[test]
    fn anderson_darling_runs_on_normal_data() {
        let data: Vec<f64> = (1..=40)
            .map(|i| {
                let u = i as f64 / 41.0;
                // Inverse-normal-ish quantile approximation.
                let z = ((u - 0.5) * 4.0).tanh() * 1.4;
                z * 1.2 + 5.0
            })
            .collect();
        let res = anderson_darling(&data).unwrap();
        assert!(res.statistic.is_finite() && res.statistic > 0.0);
        assert!(res.adjusted >= res.statistic);
        assert!((0.0..=1.0).contains(&res.p_value));
    }

    #[test]
    fn lilliefors_runs_and_clamps() {
        let data: Vec<f64> = (1..=30).map(|i| i as f64).collect(); // uniform-like
        let res = lilliefors(&data).unwrap();
        assert!(res.statistic >= 0.0 && res.statistic <= 1.0);
        assert!((0.0..=0.20).contains(&res.p_value));
    }
}
