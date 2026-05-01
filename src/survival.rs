//! Survival analysis: Kaplan-Meier estimator and Cox proportional hazards regression.
//!
//! # Models
//! - [`KaplanMeier`] — non-parametric survival curve estimator with log-rank test.
//! - [`CoxPh`] — semi-parametric Cox proportional hazards model via Newton-Raphson.
//!
//! # Quick start
//! ```rust
//! use inferust::survival::{KaplanMeier, CoxPh};
//!
//! // Times-to-event and event indicators (1 = event, 0 = censored)
//! let times = vec![5.0, 8.0, 12.0, 20.0, 3.0, 7.0, 15.0, 22.0];
//! let events = vec![1, 1, 0, 1, 1, 0, 1, 1];
//!
//! let km = KaplanMeier::new().fit(&times, &events).unwrap();
//! km.print_summary();
//!
//! // Cox regression with one covariate
//! let x = vec![vec![0.0], vec![1.0], vec![0.0], vec![1.0],
//!              vec![0.0], vec![1.0], vec![0.0], vec![1.0]];
//! let cox = CoxPh::new().fit(&times, &events, &x).unwrap();
//! cox.print_summary();
//! ```

use crate::error::{InferustError, Result};
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};

// ── Kaplan-Meier ──────────────────────────────────────────────────────────────

/// Kaplan-Meier non-parametric survival estimator.
///
/// Handles right-censored data. Event times where `event = 0` are treated as
/// censored.
#[derive(Debug, Clone, Default)]
pub struct KaplanMeier {
    feature_names: Vec<String>,
}

/// A single step on the Kaplan-Meier survival curve.
#[derive(Debug, Clone)]
pub struct KmStep {
    /// Event time.
    pub time: f64,
    /// Number at risk just before this time.
    pub n_at_risk: usize,
    /// Number of events at this time.
    pub n_events: usize,
    /// Kaplan-Meier survival probability S(t).
    pub survival: f64,
    /// Lower bound of the 95 % Greenwood confidence interval on S(t).
    pub ci_lower: f64,
    /// Upper bound of the 95 % Greenwood confidence interval on S(t).
    pub ci_upper: f64,
}

/// Fitted Kaplan-Meier result.
#[derive(Debug, Clone)]
pub struct KaplanMeierResult {
    /// Survival curve steps, ordered by time.
    pub curve: Vec<KmStep>,
    /// Total number of subjects.
    pub n: usize,
    /// Total number of events (non-censored).
    pub n_events: usize,
    /// Restricted mean survival time (area under the curve up to the last event time).
    pub rmst: f64,
    /// Median survival time (smallest t where S(t) ≤ 0.5), or `None` if S never reaches 0.5.
    pub median_survival: Option<f64>,
}

impl KaplanMeier {
    /// Create a new Kaplan-Meier builder.
    pub fn new() -> Self { Self::default() }

    /// Fit the Kaplan-Meier estimator.
    ///
    /// * `times` — observed times (must be > 0).
    /// * `events` — event indicators: 1 = event occurred, 0 = censored.
    pub fn fit(&self, times: &[f64], events: &[usize]) -> Result<KaplanMeierResult> {
        let n = times.len();
        if n < 1 {
            return Err(InferustError::InsufficientData { needed: 1, got: 0 });
        }
        if events.len() != n {
            return Err(InferustError::DimensionMismatch { x_rows: events.len(), y_len: n });
        }

        // Sort by time, events before censoring at tied times
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            times[a].partial_cmp(&times[b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(events[b].cmp(&events[a])) // events first at ties
        });

        // Walk through unique event times
        let mut curve: Vec<KmStep> = Vec::new();
        let mut survival = 1.0_f64;
        let mut greenwood_sum = 0.0_f64; // for Greenwood's formula
        let mut n_at_risk = n;
        let total_events: usize = events.iter().sum();

        let mut i = 0;
        while i < n {
            let t = times[order[i]];
            // Count events and censored at this time
            let mut d = 0usize; // deaths/events
            let mut c = 0usize; // censored
            while i < n && (times[order[i]] - t).abs() < f64::EPSILON {
                if events[order[i]] == 1 { d += 1; } else { c += 1; }
                i += 1;
            }

            if d > 0 {
                // Kaplan-Meier update
                let factor = 1.0 - d as f64 / n_at_risk as f64;
                survival *= factor;
                // Greenwood's formula term
                if n_at_risk > d {
                    greenwood_sum += d as f64 / (n_at_risk as f64 * (n_at_risk - d) as f64);
                }
                let se = survival * (greenwood_sum.sqrt());
                let z = Normal::new(0.0, 1.0).unwrap().inverse_cdf(0.975);
                // Log-transformed CI (more accurate near 0 and 1)
                let (ci_lower, ci_upper) = if survival > 0.0 && survival < 1.0 {
                    let log_s = survival.ln();
                    let log_se = se / (survival * log_s.abs().max(f64::EPSILON));
                    let (lo, hi) = (log_s * (1.0 + z * log_se), log_s * (1.0 - z * log_se));
                    (lo.exp().max(0.0).min(1.0), hi.exp().max(0.0).min(1.0))
                } else {
                    (0.0, 1.0)
                };
                curve.push(KmStep { time: t, n_at_risk, n_events: d, survival, ci_lower, ci_upper });
            }
            n_at_risk -= d + c;
        }

        // Restricted mean survival time (trapezoidal area under curve)
        let rmst = compute_rmst(&curve);
        let median_survival = curve.iter().find(|s| s.survival <= 0.5).map(|s| s.time);

        Ok(KaplanMeierResult { curve, n, n_events: total_events, rmst, median_survival })
    }
}

impl KaplanMeierResult {
    /// S(t): interpolated survival probability at time `t`.
    pub fn survival_at(&self, t: f64) -> f64 {
        // Return last S before or at t
        let mut s = 1.0;
        for step in &self.curve {
            if step.time > t { break; }
            s = step.survival;
        }
        s
    }

    /// Print a text survival table to stdout.
    pub fn print_summary(&self) {
        println!();
        println!("── Kaplan-Meier Survival Estimate ─────────────────────────────────");
        println!("  n = {}   events = {}   median survival = {}",
            self.n, self.n_events,
            self.median_survival.map_or("undefined".to_string(), |m| format!("{m:.3}")));
        println!("  RMST = {:.4}", self.rmst);
        println!();
        println!("{:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "Time", "N.risk", "N.event", "Survival", "CI lower", "CI upper");
        println!("{}", "─".repeat(65));
        for s in &self.curve {
            println!("{:>10.3} {:>10} {:>10} {:>10.4} {:>10.4} {:>10.4}",
                s.time, s.n_at_risk, s.n_events, s.survival, s.ci_lower, s.ci_upper);
        }
        println!();
    }
}

fn compute_rmst(curve: &[KmStep]) -> f64 {
    let mut area = 0.0;
    let mut prev_t = 0.0;
    let mut prev_s = 1.0;
    for step in curve {
        area += prev_s * (step.time - prev_t);
        prev_t = step.time;
        prev_s = step.survival;
    }
    area
}

// ── Log-rank test ─────────────────────────────────────────────────────────────

/// Result of a log-rank test comparing two survival curves.
#[derive(Debug, Clone)]
pub struct LogRankResult {
    /// Log-rank χ² statistic.
    pub statistic: f64,
    /// p-value from χ²(1) distribution.
    pub p_value: f64,
}

impl LogRankResult {
    /// Print the log-rank test result.
    pub fn print(&self) {
        println!();
        println!("── Log-Rank Test ──────────────────────────────────────");
        println!("  χ²({}) = {:.4}   p = {:.6}", 1, self.statistic, self.p_value);
        if self.p_value < 0.05 {
            println!("  ✓ Reject H₀ (p < 0.05): survival curves differ.");
        } else {
            println!("  ✗ Fail to reject H₀ (p ≥ 0.05).");
        }
        println!();
    }
}

/// Log-rank test comparing two groups.
///
/// * `times1`, `events1` — group 1 observed times and event indicators.
/// * `times2`, `events2` — group 2 observed times and event indicators.
pub fn log_rank_test(
    times1: &[f64], events1: &[usize],
    times2: &[f64], events2: &[usize],
) -> Result<LogRankResult> {
    let n1 = times1.len();
    let n2 = times2.len();
    if n1 < 1 || n2 < 1 {
        return Err(InferustError::InsufficientData { needed: 1, got: n1.min(n2) });
    }

    // Collect all unique event times (not censored)
    let mut event_times: Vec<f64> = times1.iter().zip(events1.iter())
        .filter(|(_, &e)| e == 1).map(|(&t, _)| t)
        .chain(times2.iter().zip(events2.iter()).filter(|(_, &e)| e == 1).map(|(&t, _)| t))
        .collect();
    event_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    event_times.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);

    let mut o_e = 0.0_f64;    // Σ (O₁ - E₁)
    let mut var = 0.0_f64;    // Σ Var(O₁)

    for &t in &event_times {
        let n_at_risk_1 = times1.iter().filter(|&&ti| ti >= t).count();
        let n_at_risk_2 = times2.iter().filter(|&&ti| ti >= t).count();
        let d1 = times1.iter().zip(events1.iter()).filter(|(&ti, &ei)| (ti - t).abs() < f64::EPSILON && ei == 1).count();
        let d2 = times2.iter().zip(events2.iter()).filter(|(&ti, &ei)| (ti - t).abs() < f64::EPSILON && ei == 1).count();

        let n = (n_at_risk_1 + n_at_risk_2) as f64;
        let d = (d1 + d2) as f64;
        if n < 2.0 { continue; }

        let e1 = n_at_risk_1 as f64 * d / n;
        o_e += d1 as f64 - e1;

        // Hypergeometric variance
        let n1f = n_at_risk_1 as f64;
        let n2f = n_at_risk_2 as f64;
        var += n1f * n2f * d * (n - d) / (n * n * (n - 1.0));
    }

    let stat = if var > f64::EPSILON { o_e * o_e / var } else { 0.0 };
    let chi = ChiSquared::new(1.0)
        .map_err(|_| InferustError::InvalidInput("chi-squared distribution error".into()))?;
    let p = 1.0 - chi.cdf(stat);

    Ok(LogRankResult { statistic: stat, p_value: p })
}

// ── Cox PH ────────────────────────────────────────────────────────────────────

/// Cox proportional hazards model fitted by partial likelihood Newton-Raphson.
///
/// Supports right-censored data with multiple covariates.
#[derive(Debug, Clone)]
pub struct CoxPh {
    feature_names: Vec<String>,
    max_iter: usize,
    tolerance: f64,
}

/// Fitted Cox PH result.
#[derive(Debug, Clone)]
pub struct CoxPhResult {
    /// Estimated log-hazard coefficients β (one per covariate).
    pub coefficients: Vec<f64>,
    /// Hazard ratios exp(β).
    pub hazard_ratios: Vec<f64>,
    /// Standard errors of β (from the observed Fisher information).
    pub std_errors: Vec<f64>,
    /// Wald z-statistics β / SE(β).
    pub z_statistics: Vec<f64>,
    /// Two-sided p-values for each β.
    pub p_values: Vec<f64>,
    /// 95 % confidence intervals on the hazard ratios.
    pub hr_ci: Vec<(f64, f64)>,
    /// Partial log-likelihood at convergence.
    pub log_likelihood: f64,
    /// Likelihood-ratio test statistic vs. null model (−2 log LR).
    pub lr_statistic: f64,
    /// p-value of the likelihood-ratio test.
    pub lr_p_value: f64,
    /// Feature names.
    pub feature_names: Vec<String>,
    /// Number of observations.
    pub n: usize,
    /// Number of events.
    pub n_events: usize,
    /// Number of iterations until convergence.
    pub iterations: usize,
}

impl Default for CoxPh {
    fn default() -> Self { Self::new() }
}

impl CoxPh {
    /// Create a new Cox PH builder.
    pub fn new() -> Self {
        Self {
            feature_names: Vec::new(),
            max_iter: 200,
            tolerance: 1e-8,
        }
    }

    /// Set human-readable names for the covariate columns.
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Override the maximum number of Newton-Raphson iterations (default 200).
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Fit the Cox PH model using partial likelihood Newton-Raphson.
    ///
    /// * `times`  — observed times (event or censoring time for each subject).
    /// * `events` — event indicators: 1 = event, 0 = censored.
    /// * `x`      — covariate matrix (n rows × p cols).
    pub fn fit(&self, times: &[f64], events: &[usize], x: &[Vec<f64>]) -> Result<CoxPhResult> {
        let n = times.len();
        if n < 2 {
            return Err(InferustError::InsufficientData { needed: 2, got: n });
        }
        if events.len() != n || x.len() != n {
            return Err(InferustError::DimensionMismatch { x_rows: x.len(), y_len: n });
        }
        let p = x[0].len();
        if p == 0 {
            return Err(InferustError::InvalidInput("CoxPh requires at least one covariate".into()));
        }
        for row in x.iter() {
            if row.len() != p {
                return Err(InferustError::DimensionMismatch { x_rows: row.len(), y_len: p });
            }
        }

        let n_events: usize = events.iter().sum();
        if n_events == 0 {
            return Err(InferustError::InvalidInput("no events in the data".into()));
        }

        // Sort by time (events before censored at ties — Breslow tie handling)
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            times[a].partial_cmp(&times[b]).unwrap_or(std::cmp::Ordering::Equal)
                .then(events[b].cmp(&events[a]))
        });

        let t_sorted: Vec<f64> = order.iter().map(|&i| times[i]).collect();
        let e_sorted: Vec<usize> = order.iter().map(|&i| events[i]).collect();
        let x_sorted: Vec<Vec<f64>> = order.iter().map(|&i| x[i].clone()).collect();

        // Newton-Raphson on partial log-likelihood
        let mut beta = vec![0.0_f64; p];
        let null_ll = cox_partial_ll(&t_sorted, &e_sorted, &x_sorted, &vec![0.0_f64; p]);
        let mut iterations = 0;

        for iter in 0..self.max_iter {
            let (score, hessian) = cox_score_hessian(&t_sorted, &e_sorted, &x_sorted, &beta);
            let step = solve_linear(hessian, score)?;
            let max_step: f64 = step.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
            for j in 0..p { beta[j] += step[j]; }
            iterations = iter + 1;
            if max_step < self.tolerance { break; }
        }

        let ll = cox_partial_ll(&t_sorted, &e_sorted, &x_sorted, &beta);
        let lr_stat = -2.0 * (null_ll - ll);
        let chi = ChiSquared::new(p as f64)
            .map_err(|_| InferustError::InvalidInput("chi-squared distribution error".into()))?;
        let lr_p = 1.0 - chi.cdf(lr_stat.max(0.0));

        // Standard errors from observed information (negated Hessian)
        let (_, hessian) = cox_score_hessian(&t_sorted, &e_sorted, &x_sorted, &beta);
        let var_cov = invert_symmetric(hessian)?;
        let se: Vec<f64> = (0..p).map(|j| var_cov[j][j].abs().sqrt()).collect();

        let normal = Normal::new(0.0, 1.0)
            .map_err(|_| InferustError::InvalidInput("normal distribution error".into()))?;
        let z: Vec<f64> = beta.iter().zip(se.iter()).map(|(b, s)| b / s).collect();
        let pv: Vec<f64> = z.iter().map(|&z| 2.0 * (1.0 - normal.cdf(z.abs()))).collect();
        let hr: Vec<f64> = beta.iter().map(|b| b.exp()).collect();
        let z196 = normal.inverse_cdf(0.975);
        let hr_ci: Vec<(f64, f64)> = beta.iter().zip(se.iter())
            .map(|(b, s)| ((b - z196 * s).exp(), (b + z196 * s).exp()))
            .collect();

        let feat = if self.feature_names.len() == p {
            self.feature_names.clone()
        } else {
            (1..=p).map(|i| format!("x{i}")).collect()
        };

        Ok(CoxPhResult {
            coefficients: beta,
            hazard_ratios: hr,
            std_errors: se,
            z_statistics: z,
            p_values: pv,
            hr_ci,
            log_likelihood: ll,
            lr_statistic: lr_stat,
            lr_p_value: lr_p,
            feature_names: feat,
            n,
            n_events,
            iterations,
        })
    }
}

impl CoxPhResult {
    /// Print a statsmodels-style Cox PH summary.
    pub fn print_summary(&self) {
        println!();
        println!("══════════════════════════════════════════════════════════════════");
        println!("  Cox Proportional Hazards Model");
        println!("══════════════════════════════════════════════════════════════════");
        println!("  n = {}   events = {}   iterations = {}", self.n, self.n_events, self.iterations);
        println!("  Log-likelihood: {:.4}", self.log_likelihood);
        println!("  LR χ²({}) = {:.4}   p = {:.6}", self.feature_names.len(), self.lr_statistic, self.lr_p_value);
        println!("──────────────────────────────────────────────────────────────────");
        println!("{:<18} {:>9} {:>9} {:>8} {:>9} {:>12}",
            "Variable", "coef", "HR", "SE", "z", "P>|z|");
        println!("{}", "─".repeat(66));
        for i in 0..self.coefficients.len() {
            let (hr_lo, hr_hi) = self.hr_ci[i];
            println!("{:<18} {:>9.4} {:>9.4} {:>8.4} {:>9.4} {:>9.6}  {}  [{:.4}, {:.4}]",
                self.feature_names[i],
                self.coefficients[i],
                self.hazard_ratios[i],
                self.std_errors[i],
                self.z_statistics[i],
                self.p_values[i],
                sig_stars(self.p_values[i]),
                hr_lo, hr_hi);
        }
        println!("──────────────────────────────────────────────────────────────────");
        println!("  Significance:  *** p<0.001  ** p<0.01  * p<0.05  . p<0.1");
        println!("══════════════════════════════════════════════════════════════════");
        println!();
    }
}

// ── Cox internals ─────────────────────────────────────────────────────────────

/// Breslow partial log-likelihood for sorted data.
fn cox_partial_ll(times: &[f64], events: &[usize], x: &[Vec<f64>], beta: &[f64]) -> f64 {
    let n = times.len();
    let p = beta.len();
    let xb: Vec<f64> = x.iter().map(|row| row.iter().zip(beta).map(|(xi, b)| xi * b).sum()).collect();
    let exp_xb: Vec<f64> = xb.iter().map(|v| v.exp()).collect();

    let mut ll = 0.0;
    for i in 0..n {
        if events[i] != 1 { continue; }
        // Risk set: all j with t_j >= t_i
        let risk_sum: f64 = (i..n).map(|j| exp_xb[j]).sum();
        ll += xb[i] - risk_sum.ln();
    }
    ll
}

/// Score vector and observed information (negated Hessian) for sorted data.
fn cox_score_hessian(
    times: &[f64], events: &[usize], x: &[Vec<f64>], beta: &[f64],
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = times.len();
    let p = beta.len();
    let exp_xb: Vec<f64> = x.iter()
        .map(|row| row.iter().zip(beta).map(|(xi, b)| xi * b).sum::<f64>().exp())
        .collect();

    let mut score = vec![0.0_f64; p];
    let mut info = vec![vec![0.0_f64; p]; p];

    for i in 0..n {
        if events[i] != 1 { continue; }
        // Risk set: j with t_j >= t_i (data is sorted, so j >= i)
        let mut s0 = 0.0_f64;
        let mut s1 = vec![0.0_f64; p];
        let mut s2 = vec![vec![0.0_f64; p]; p];
        for j in i..n {
            let w = exp_xb[j];
            s0 += w;
            for k in 0..p {
                s1[k] += w * x[j][k];
                for l in 0..p {
                    s2[k][l] += w * x[j][k] * x[j][l];
                }
            }
        }
        if s0 < f64::EPSILON { continue; }
        for k in 0..p {
            score[k] += x[i][k] - s1[k] / s0;
            for l in 0..p {
                info[k][l] += s2[k][l] / s0 - (s1[k] / s0) * (s1[l] / s0);
            }
        }
    }
    (score, info)
}

/// Solve the linear system A·x = b using Gaussian elimination with partial pivoting.
fn solve_linear(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        // Partial pivot
        let max_row = (col..n)
            .max_by(|&r1, &r2| a[r1][col].abs().partial_cmp(&a[r2][col].abs()).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(col);
        a.swap(col, max_row);
        b.swap(col, max_row);
        let pivot = a[col][col];
        if pivot.abs() < f64::EPSILON {
            return Err(InferustError::InvalidInput("singular information matrix — model may be under-identified".into()));
        }
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            for k in col..n { a[row][k] -= factor * a[col][k]; }
            b[row] -= factor * b[col];
        }
    }
    // Back substitution
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        x[i] = b[i];
        for j in (i + 1)..n { x[i] -= a[i][j] * x[j]; }
        x[i] /= a[i][i];
    }
    Ok(x)
}

/// Invert a symmetric positive-definite matrix using Cholesky-like Gaussian elimination.
fn invert_symmetric(a: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>> {
    let n = a.len();
    // Augment [A | I] and row-reduce
    let mut aug: Vec<Vec<f64>> = a.iter().enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            r
        })
        .collect();

    for col in 0..n {
        let max_row = (col..n)
            .max_by(|&r1, &r2| aug[r1][col].abs().partial_cmp(&aug[r2][col].abs()).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(col);
        aug.swap(col, max_row);
        let pivot = aug[col][col];
        if pivot.abs() < f64::EPSILON {
            return Err(InferustError::InvalidInput("information matrix is singular".into()));
        }
        let inv_pivot = 1.0 / pivot;
        for k in 0..(2 * n) { aug[col][k] *= inv_pivot; }
        for row in 0..n {
            if row == col { continue; }
            let factor = aug[row][col];
            for k in 0..(2 * n) { aug[row][k] -= factor * aug[col][k]; }
        }
    }
    Ok(aug.iter().map(|row| row[n..].to_vec()).collect())
}

fn sig_stars(p: f64) -> &'static str {
    if p < 0.001 { "***" } else if p < 0.01 { "**" } else if p < 0.05 { "*" } else if p < 0.1 { "." } else { "" }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{log_rank_test, CoxPh, KaplanMeier};

    #[test]
    fn km_basic_curve() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let events = vec![1, 1, 0, 1, 0, 1];
        let km = KaplanMeier::new().fit(&times, &events).unwrap();
        assert_eq!(km.n, 6);
        assert_eq!(km.n_events, 4);
        assert!(km.curve[0].survival < 1.0);
        // Survival should be non-increasing
        for w in km.curve.windows(2) {
            assert!(w[0].survival >= w[1].survival);
        }
    }

    #[test]
    fn km_all_censored_still_works() {
        let times = vec![5.0, 10.0, 15.0];
        let events = vec![0, 0, 0];
        let km = KaplanMeier::new().fit(&times, &events).unwrap();
        assert_eq!(km.n_events, 0);
        assert!(km.curve.is_empty());
    }

    #[test]
    fn log_rank_different_groups() {
        // Group 1 survives longer → expect significant difference
        let t1 = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let e1 = vec![1, 1, 1, 1, 1];
        let t2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let e2 = vec![1, 1, 1, 1, 1];
        let res = log_rank_test(&t1, &e1, &t2, &e2).unwrap();
        assert!(res.p_value < 0.05, "expected p < 0.05, got {}", res.p_value);
    }

    #[test]
    fn cox_ph_fits_simple_covariate() {
        let times = vec![5.0, 8.0, 12.0, 20.0, 3.0, 7.0, 15.0, 22.0, 9.0, 11.0];
        let events = vec![1, 1, 0, 1, 1, 0, 1, 1, 1, 0];
        // Binary treatment covariate
        let x: Vec<Vec<f64>> = (0..10).map(|i| vec![if i < 5 { 0.0 } else { 1.0 }]).collect();
        let cox = CoxPh::new()
            .with_feature_names(vec!["treatment".to_string()])
            .fit(&times, &events, &x)
            .unwrap();
        assert_eq!(cox.feature_names[0], "treatment");
        assert_eq!(cox.coefficients.len(), 1);
        assert!(cox.coefficients[0].is_finite());
        assert!(cox.hazard_ratios[0] > 0.0);
    }

    #[test]
    fn cox_ph_lr_test_statistic_positive() {
        let times: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let events: Vec<usize> = (0..20).map(|i| if i % 3 != 0 { 1 } else { 0 }).collect();
        let x: Vec<Vec<f64>> = (0..20).map(|i| vec![(i as f64) / 10.0]).collect();
        let cox = CoxPh::new().fit(&times, &events, &x).unwrap();
        assert!(cox.lr_statistic >= 0.0);
    }
}
