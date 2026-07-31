//! Exponential smoothing forecasting: SES, Holt, and Holt-Winters.
//!
//! Mirrors `statsmodels.tsa.holtwinters`:
//! - [`SimpleExpSmoothing`] — simple exponential smoothing (level only).
//! - [`Holt`] — Holt's linear trend, optionally damped.
//! - [`ExponentialSmoothing`] — the full Holt-Winters model with additive
//!   or multiplicative seasonality.
//!
//! Smoothing parameters may be fixed or optimized by minimizing the
//! one-step-ahead sum of squared errors (Nelder-Mead). Initial states use
//! statsmodels' `legacy-heuristic` scheme unless supplied explicitly, so a
//! fit with *fixed* parameters and known initial states reproduces
//! statsmodels exactly; optimized fits agree up to optimizer differences
//! (statsmodels' `estimated` method also optimizes the initial states).
//!
//! # Example
//! ```rust
//! use inferust::smoothing::{ExponentialSmoothing, SeasonalComponent, TrendComponent};
//!
//! let y: Vec<f64> = (0..48)
//!     .map(|t| 100.0 + 2.0 * t as f64
//!         + 15.0 * (2.0 * std::f64::consts::PI * (t % 12) as f64 / 12.0).sin())
//!     .collect();
//! let fit = ExponentialSmoothing::new()
//!     .with_trend(TrendComponent::Additive)
//!     .with_seasonal(SeasonalComponent::Additive, 12)
//!     .fit(&y)
//!     .unwrap();
//! let forecast = fit.forecast(12).unwrap();
//! assert_eq!(forecast.len(), 12);
//! ```

use crate::error::{InferustError, Result};

/// Trend component specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrendComponent {
    /// No trend.
    #[default]
    None,
    /// Additive (linear) trend.
    Additive,
}

/// Seasonal component specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SeasonalComponent {
    /// No seasonality.
    #[default]
    None,
    /// Additive seasonality.
    Additive,
    /// Multiplicative seasonality (requires a strictly positive series).
    Multiplicative,
}

/// Holt-Winters exponential smoothing model builder.
#[derive(Debug, Clone, Default)]
pub struct ExponentialSmoothing {
    trend: TrendComponent,
    damped: bool,
    seasonal: SeasonalComponent,
    period: usize,
    alpha: Option<f64>,
    beta: Option<f64>,
    gamma: Option<f64>,
    phi: Option<f64>,
    initial_level: Option<f64>,
    initial_trend: Option<f64>,
    initial_seasonal: Option<Vec<f64>>,
}

/// Fitted exponential smoothing model.
#[derive(Debug, Clone)]
pub struct ExponentialSmoothingResult {
    /// Smoothing level α.
    pub alpha: f64,
    /// Smoothing trend β (0 when the model has no trend).
    pub beta: f64,
    /// Smoothing seasonal γ (0 when the model has no seasonality).
    pub gamma: f64,
    /// Damping parameter φ (1 when undamped).
    pub phi: f64,
    /// Initial level state.
    pub initial_level: f64,
    /// Initial trend state.
    pub initial_trend: f64,
    /// Initial seasonal states (length = period).
    pub initial_seasonal: Vec<f64>,
    /// Final level state.
    pub level: f64,
    /// Final trend state.
    pub trend: f64,
    /// Final seasonal states, ordered so index 0 applies to the first
    /// forecast step.
    pub season: Vec<f64>,
    /// One-step-ahead in-sample forecasts.
    pub fitted_values: Vec<f64>,
    /// One-step-ahead residuals.
    pub residuals: Vec<f64>,
    /// Sum of squared one-step errors.
    pub sse: f64,
    /// AIC: n·ln(SSE/n) + 2k, following statsmodels.
    pub aic: f64,
    /// BIC: n·ln(SSE/n) + k·ln(n).
    pub bic: f64,
    /// Number of observations.
    pub n: usize,
    trend_spec: TrendComponent,
    seasonal_spec: SeasonalComponent,
    period: usize,
}

impl ExponentialSmoothing {
    /// Create a model with no trend and no seasonality (i.e. SES).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the trend component.
    pub fn with_trend(mut self, trend: TrendComponent) -> Self {
        self.trend = trend;
        self
    }

    /// Damp the trend (only meaningful with an additive trend).
    pub fn damped(mut self, damped: bool) -> Self {
        self.damped = damped;
        self
    }

    /// Set the seasonal component and period (period ≥ 2).
    pub fn with_seasonal(mut self, seasonal: SeasonalComponent, period: usize) -> Self {
        self.seasonal = seasonal;
        self.period = period;
        self
    }

    /// Fix the level smoothing parameter α instead of optimizing it.
    pub fn smoothing_level(mut self, alpha: f64) -> Self {
        self.alpha = Some(alpha);
        self
    }

    /// Fix the trend smoothing parameter β.
    pub fn smoothing_trend(mut self, beta: f64) -> Self {
        self.beta = Some(beta);
        self
    }

    /// Fix the seasonal smoothing parameter γ.
    pub fn smoothing_seasonal(mut self, gamma: f64) -> Self {
        self.gamma = Some(gamma);
        self
    }

    /// Fix the damping parameter φ.
    pub fn damping_trend(mut self, phi: f64) -> Self {
        self.phi = Some(phi);
        self
    }

    /// Supply a known initial level (statsmodels `initialization_method="known"`).
    pub fn initial_level(mut self, level: f64) -> Self {
        self.initial_level = Some(level);
        self
    }

    /// Supply a known initial trend.
    pub fn initial_trend(mut self, trend: f64) -> Self {
        self.initial_trend = Some(trend);
        self
    }

    /// Supply known initial seasonal states (length must equal the period).
    pub fn initial_seasonal(mut self, seasonal: Vec<f64>) -> Self {
        self.initial_seasonal = Some(seasonal);
        self
    }

    /// Fit the model to `y`.
    pub fn fit(&self, y: &[f64]) -> Result<ExponentialSmoothingResult> {
        let m = self.period;
        let has_seasonal = !matches!(self.seasonal, SeasonalComponent::None);
        let has_trend = matches!(self.trend, TrendComponent::Additive);
        if has_seasonal && m < 2 {
            return Err(InferustError::InvalidInput(
                "seasonal models need period >= 2".into(),
            ));
        }
        let min_n = if has_seasonal {
            2 * m
        } else {
            2 + has_trend as usize
        };
        if y.len() < min_n.max(2) {
            return Err(InferustError::InsufficientData {
                needed: min_n.max(2),
                got: y.len(),
            });
        }
        if matches!(self.seasonal, SeasonalComponent::Multiplicative) && y.iter().any(|&v| v <= 0.0)
        {
            return Err(InferustError::InvalidInput(
                "multiplicative seasonality requires a strictly positive series".into(),
            ));
        }
        for (name, p) in [
            ("alpha", self.alpha),
            ("beta", self.beta),
            ("gamma", self.gamma),
            ("phi", self.phi),
        ] {
            if let Some(v) = p {
                if !(0.0..=1.0).contains(&v) {
                    return Err(InferustError::InvalidInput(format!(
                        "{name} must lie in [0, 1]"
                    )));
                }
            }
        }
        if let Some(s) = &self.initial_seasonal {
            if s.len() != m {
                return Err(InferustError::DimensionMismatch {
                    x_rows: s.len(),
                    y_len: m,
                });
            }
        }

        // Initial states: legacy-heuristic unless supplied.
        let (l0, b0, s0) = self.initial_states(y);

        // Which of [alpha, beta, gamma, phi] are free?
        let free: Vec<usize> = {
            let mut f = Vec::new();
            if self.alpha.is_none() {
                f.push(0);
            }
            if has_trend && self.beta.is_none() {
                f.push(1);
            }
            if has_seasonal && self.gamma.is_none() {
                f.push(2);
            }
            if self.damped && self.phi.is_none() {
                f.push(3);
            }
            f
        };
        let fixed = [
            self.alpha.unwrap_or(0.0),
            self.beta.unwrap_or(0.0),
            self.gamma.unwrap_or(0.0),
            if self.damped {
                self.phi.unwrap_or(0.0)
            } else {
                1.0
            },
        ];

        let assemble = |theta: &[f64]| -> [f64; 4] {
            let mut params = fixed;
            for (i, &slot) in free.iter().enumerate() {
                params[slot] = if slot == 3 {
                    // Damping is conventionally kept in [0.8, 0.995] when optimized.
                    0.8 + 0.195 * logistic(theta[i])
                } else {
                    logistic(theta[i])
                };
            }
            params
        };

        let params = if free.is_empty() {
            fixed
        } else {
            let objective = |theta: &[f64]| -> f64 {
                let p = assemble(theta);
                // Enforce statsmodels' constraint gamma <= 1 - alpha.
                if has_seasonal && p[2] > 1.0 - p[0] {
                    return f64::MAX / 4.0;
                }
                run_recursions(
                    y,
                    self.trend,
                    self.seasonal,
                    m,
                    p[0],
                    p[1],
                    p[2],
                    p[3],
                    l0,
                    b0,
                    &s0,
                )
                .2
            };
            // Multi-start Nelder-Mead: SSE surfaces can have flat ridges.
            let starts: Vec<Vec<f64>> = vec![
                vec![0.0; free.len()],
                vec![-1.5; free.len()],
                vec![1.5; free.len()],
            ];
            let mut best: Option<(f64, Vec<f64>)> = None;
            for start in starts {
                let (theta, val) = nelder_mead(&objective, &start, 500);
                if best.as_ref().map(|(v, _)| val < *v).unwrap_or(true) {
                    best = Some((val, theta));
                }
            }
            assemble(&best.unwrap().1)
        };
        let [alpha, beta, gamma, phi] = params;

        let (fitted, states, sse) = run_recursions(
            y,
            self.trend,
            self.seasonal,
            m,
            alpha,
            beta,
            gamma,
            phi,
            l0,
            b0,
            &s0,
        );
        let residuals: Vec<f64> = y.iter().zip(fitted.iter()).map(|(a, f)| a - f).collect();
        let n = y.len();
        // Free-parameter count for information criteria (statsmodels
        // convention: smoothing params + initial states).
        let k =
            (free.len().max(1) + 1 + has_trend as usize + if has_seasonal { m } else { 0 }) as f64;
        let n_f = n as f64;
        let base = n_f * (sse / n_f).max(f64::MIN_POSITIVE).ln();

        Ok(ExponentialSmoothingResult {
            alpha,
            beta,
            gamma,
            phi,
            initial_level: l0,
            initial_trend: b0,
            initial_seasonal: s0,
            level: states.0,
            trend: states.1,
            season: states.2,
            fitted_values: fitted,
            residuals,
            sse,
            aic: base + 2.0 * k,
            bic: base + k * n_f.ln(),
            n,
            trend_spec: self.trend,
            seasonal_spec: self.seasonal,
            period: m,
        })
    }

    /// statsmodels `legacy-heuristic` initial states.
    fn initial_states(&self, y: &[f64]) -> (f64, f64, Vec<f64>) {
        let m = self.period;
        let has_seasonal = !matches!(self.seasonal, SeasonalComponent::None);
        let has_trend = matches!(self.trend, TrendComponent::Additive);

        let l0 = if let Some(l) = self.initial_level {
            l
        } else if has_seasonal {
            y[..m].iter().sum::<f64>() / m as f64
        } else {
            y[0]
        };
        let b0 = if let Some(b) = self.initial_trend {
            b
        } else if !has_trend {
            0.0
        } else if has_seasonal {
            let first = y[..m].iter().sum::<f64>() / m as f64;
            let second = y[m..2 * m].iter().sum::<f64>() / m as f64;
            (second - first) / m as f64
        } else {
            y[1] - y[0]
        };
        let s0 = if let Some(s) = &self.initial_seasonal {
            s.clone()
        } else if has_seasonal {
            match self.seasonal {
                SeasonalComponent::Additive => y[..m].iter().map(|v| v - l0).collect(),
                SeasonalComponent::Multiplicative => y[..m].iter().map(|v| v / l0).collect(),
                SeasonalComponent::None => unreachable!(),
            }
        } else {
            vec![]
        };
        (l0, b0, s0)
    }
}

impl ExponentialSmoothingResult {
    /// Forecast `steps` observations beyond the end of the fitted series.
    pub fn forecast(&self, steps: usize) -> Result<Vec<f64>> {
        let m = self.period;
        let mut out = Vec::with_capacity(steps);
        let mut phi_sum = 0.0;
        for h in 1..=steps {
            phi_sum += self.phi.powi(h as i32);
            let base = self.level
                + if matches!(self.trend_spec, TrendComponent::Additive) {
                    phi_sum * self.trend
                } else {
                    0.0
                };
            let value = match self.seasonal_spec {
                SeasonalComponent::None => base,
                SeasonalComponent::Additive => base + self.season[(h - 1) % m],
                SeasonalComponent::Multiplicative => base * self.season[(h - 1) % m],
            };
            out.push(value);
        }
        Ok(out)
    }

    /// Print a summary to stdout.
    pub fn print_summary(&self) {
        println!();
        println!("═══════════════════════════════════════════════════");
        println!("  Exponential Smoothing Results");
        println!("═══════════════════════════════════════════════════");
        println!("  n          : {}   SSE : {:.6}", self.n, self.sse);
        println!("  AIC        : {:.4}   BIC : {:.4}", self.aic, self.bic);
        println!("───────────────────────────────────────────────────");
        println!("  alpha (level)    = {:.6}", self.alpha);
        if self.beta > 0.0 {
            println!("  beta (trend)     = {:.6}", self.beta);
        }
        if self.gamma > 0.0 {
            println!("  gamma (seasonal) = {:.6}", self.gamma);
        }
        if self.phi < 1.0 {
            println!("  phi (damping)    = {:.6}", self.phi);
        }
        println!("═══════════════════════════════════════════════════");
        println!();
    }
}

/// Simple exponential smoothing (level only), mirroring
/// `statsmodels.tsa.holtwinters.SimpleExpSmoothing`.
#[derive(Debug, Clone, Default)]
pub struct SimpleExpSmoothing {
    inner: ExponentialSmoothing,
}

impl SimpleExpSmoothing {
    /// Create an SES builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fix the smoothing parameter α instead of optimizing it.
    pub fn smoothing_level(mut self, alpha: f64) -> Self {
        self.inner = self.inner.smoothing_level(alpha);
        self
    }

    /// Supply a known initial level.
    pub fn initial_level(mut self, level: f64) -> Self {
        self.inner = self.inner.initial_level(level);
        self
    }

    /// Fit to `y`.
    pub fn fit(&self, y: &[f64]) -> Result<ExponentialSmoothingResult> {
        self.inner.fit(y)
    }
}

/// Holt's linear-trend method, mirroring `statsmodels.tsa.holtwinters.Holt`.
#[derive(Debug, Clone)]
pub struct Holt {
    inner: ExponentialSmoothing,
}

impl Default for Holt {
    fn default() -> Self {
        Self {
            inner: ExponentialSmoothing::new().with_trend(TrendComponent::Additive),
        }
    }
}

impl Holt {
    /// Create a Holt linear-trend builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Damp the trend.
    pub fn damped(mut self, damped: bool) -> Self {
        self.inner = self.inner.damped(damped);
        self
    }

    /// Fix the level smoothing parameter α.
    pub fn smoothing_level(mut self, alpha: f64) -> Self {
        self.inner = self.inner.smoothing_level(alpha);
        self
    }

    /// Fix the trend smoothing parameter β.
    pub fn smoothing_trend(mut self, beta: f64) -> Self {
        self.inner = self.inner.smoothing_trend(beta);
        self
    }

    /// Fix the damping parameter φ.
    pub fn damping_trend(mut self, phi: f64) -> Self {
        self.inner = self.inner.damping_trend(phi);
        self
    }

    /// Supply a known initial level.
    pub fn initial_level(mut self, level: f64) -> Self {
        self.inner = self.inner.initial_level(level);
        self
    }

    /// Supply a known initial trend.
    pub fn initial_trend(mut self, trend: f64) -> Self {
        self.inner = self.inner.initial_trend(trend);
        self
    }

    /// Fit to `y`.
    pub fn fit(&self, y: &[f64]) -> Result<ExponentialSmoothingResult> {
        self.inner.fit(y)
    }
}

// ── internals ─────────────────────────────────────────────────────────────────

/// Run the Holt-Winters recursions. Returns (fitted, (level, trend, seasons), sse).
#[allow(clippy::too_many_arguments)]
fn run_recursions(
    y: &[f64],
    trend_spec: TrendComponent,
    seasonal_spec: SeasonalComponent,
    m: usize,
    alpha: f64,
    beta: f64,
    gamma: f64,
    phi: f64,
    l0: f64,
    b0: f64,
    s0: &[f64],
) -> (Vec<f64>, (f64, f64, Vec<f64>), f64) {
    let n = y.len();
    let has_trend = matches!(trend_spec, TrendComponent::Additive);
    let mut level = l0;
    let mut trend = b0;
    let mut seasons: Vec<f64> = s0.to_vec();
    let mut fitted = Vec::with_capacity(n);
    let mut sse = 0.0;

    for (t, &yt) in y.iter().enumerate() {
        let damped_trend = if has_trend { phi * trend } else { 0.0 };
        let base = level + damped_trend;
        let s_idx = if seasons.is_empty() { 0 } else { t % m };
        let yhat = match seasonal_spec {
            SeasonalComponent::None => base,
            SeasonalComponent::Additive => base + seasons[s_idx],
            SeasonalComponent::Multiplicative => base * seasons[s_idx],
        };
        fitted.push(yhat);
        let err = yt - yhat;
        sse += err * err;

        let prev_level = level;
        match seasonal_spec {
            SeasonalComponent::None => {
                level = alpha * yt + (1.0 - alpha) * base;
            }
            SeasonalComponent::Additive => {
                level = alpha * (yt - seasons[s_idx]) + (1.0 - alpha) * base;
                seasons[s_idx] =
                    gamma * (yt - prev_level - damped_trend) + (1.0 - gamma) * seasons[s_idx];
            }
            SeasonalComponent::Multiplicative => {
                level = alpha * (yt / seasons[s_idx]) + (1.0 - alpha) * base;
                seasons[s_idx] =
                    gamma * (yt / (prev_level + damped_trend)) + (1.0 - gamma) * seasons[s_idx];
            }
        }
        if has_trend {
            trend = beta * (level - prev_level) + (1.0 - beta) * (phi * trend);
        }
    }

    // Reorder seasons so index 0 is the factor for the first forecast step.
    let ordered: Vec<f64> = if seasons.is_empty() {
        vec![]
    } else {
        (0..m).map(|h| seasons[(n + h) % m]).collect()
    };
    (fitted, (level, trend, ordered), sse)
}

fn logistic(x: f64) -> f64 {
    let p = 1.0 / (1.0 + (-x).exp());
    p.clamp(1e-6, 1.0 - 1e-6)
}

/// Compact Nelder-Mead minimizer; returns (argmin, min).
fn nelder_mead<F: Fn(&[f64]) -> f64>(f: &F, start: &[f64], max_iter: usize) -> (Vec<f64>, f64) {
    let dim = start.len();
    let (alpha, gamma, rho, sigma) = (1.0, 2.0, 0.5, 0.5);
    let mut simplex: Vec<(Vec<f64>, f64)> = Vec::with_capacity(dim + 1);
    simplex.push((start.to_vec(), f(start)));
    for i in 0..dim {
        let mut p = start.to_vec();
        p[i] += 0.75;
        let v = f(&p);
        simplex.push((p, v));
    }
    for _ in 0..max_iter {
        simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        if (simplex[dim].1 - simplex[0].1).abs() < 1e-12 * (1.0 + simplex[0].1.abs()) {
            break;
        }
        // Centroid of all but the worst vertex.
        let mut centroid = vec![0.0; dim];
        for (p, _) in simplex.iter().take(dim) {
            for (c, &pi) in centroid.iter_mut().zip(p.iter()) {
                *c += pi / dim as f64;
            }
        }
        let worst = simplex[dim].clone();
        let reflect: Vec<f64> = centroid
            .iter()
            .zip(worst.0.iter())
            .map(|(c, w)| c + alpha * (c - w))
            .collect();
        let fr = f(&reflect);
        if fr < simplex[0].1 {
            let expand: Vec<f64> = centroid
                .iter()
                .zip(worst.0.iter())
                .map(|(c, w)| c + gamma * (c - w))
                .collect();
            let fe = f(&expand);
            simplex[dim] = if fe < fr { (expand, fe) } else { (reflect, fr) };
        } else if fr < simplex[dim - 1].1 {
            simplex[dim] = (reflect, fr);
        } else {
            let contract: Vec<f64> = centroid
                .iter()
                .zip(worst.0.iter())
                .map(|(c, w)| c + rho * (w - c))
                .collect();
            let fc = f(&contract);
            if fc < worst.1 {
                simplex[dim] = (contract, fc);
            } else {
                let best = simplex[0].0.clone();
                for entry in simplex.iter_mut().skip(1) {
                    let p: Vec<f64> = best
                        .iter()
                        .zip(entry.0.iter())
                        .map(|(b, p)| b + sigma * (p - b))
                        .collect();
                    let v = f(&p);
                    *entry = (p, v);
                }
            }
        }
    }
    simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let (p, v) = simplex.swap_remove(0);
    (p, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() <= tol,
            "expected {expected}, got {actual} (tol {tol})"
        );
    }

    #[test]
    fn ses_fixed_alpha_recursion() {
        // With known alpha and initial level, SES is a deterministic recursion.
        let y = [10.0, 12.0, 11.0, 13.0, 12.5];
        let fit = SimpleExpSmoothing::new()
            .smoothing_level(0.5)
            .initial_level(10.0)
            .fit(&y)
            .unwrap();
        assert_close(fit.fitted_values[0], 10.0, 1e-12);
        assert_close(fit.fitted_values[1], 10.0, 1e-12);
        assert_close(fit.fitted_values[2], 11.0, 1e-12);
        assert_close(fit.fitted_values[3], 11.0, 1e-12);
        assert_close(fit.fitted_values[4], 12.0, 1e-12);
        // Flat forecast at the final level.
        let f = fit.forecast(3).unwrap();
        assert_close(f[0], f[2], 1e-12);
    }

    #[test]
    fn holt_tracks_linear_trend() {
        let y: Vec<f64> = (0..40).map(|t| 5.0 + 2.0 * t as f64).collect();
        let fit = Holt::new().fit(&y).unwrap();
        let f = fit.forecast(5).unwrap();
        // Forecasts of a perfect line continue the line.
        for (h, v) in f.iter().enumerate() {
            assert_close(*v, 5.0 + 2.0 * (40 + h) as f64, 0.2);
        }
    }

    #[test]
    fn holt_winters_seasonal_forecast_repeats() {
        let y: Vec<f64> = (0..60)
            .map(|t| {
                50.0 + 0.5 * t as f64
                    + 8.0 * (2.0 * std::f64::consts::PI * (t % 12) as f64 / 12.0).sin()
            })
            .collect();
        let fit = ExponentialSmoothing::new()
            .with_trend(TrendComponent::Additive)
            .with_seasonal(SeasonalComponent::Additive, 12)
            .fit(&y)
            .unwrap();
        // statsmodels (legacy-heuristic init, L-BFGS-B) reaches SSE 79.048 on
        // this series; our Nelder-Mead should land within a hair of that.
        assert!(fit.sse < 82.0, "SSE {}", fit.sse);
        let f = fit.forecast(24).unwrap();
        // Seasonal shape repeats after removing the trend increment.
        let diff = (f[12] - f[0]) - 12.0 * 0.5;
        assert!(diff.abs() < 1.0, "seasonal repeat diff {diff}");
    }

    #[test]
    fn multiplicative_requires_positive() {
        let y = [1.0, -2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        let err = ExponentialSmoothing::new()
            .with_seasonal(SeasonalComponent::Multiplicative, 4)
            .fit(&y);
        assert!(err.is_err());
    }
}
