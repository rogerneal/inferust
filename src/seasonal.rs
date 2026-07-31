//! Seasonal decomposition of time series.
//!
//! - [`seasonal_decompose`] — classical moving-average decomposition
//!   (`statsmodels.tsa.seasonal.seasonal_decompose`).
//! - [`Stl`] — Season-Trend decomposition using Loess, after Cleveland,
//!   Cleveland, McRae & Terpenning (1990) (`statsmodels.tsa.seasonal.STL`).
//!
//! Both return the observed series split into trend, seasonal, and residual
//! components. The classical method leaves `NaN` at the ends of the trend
//! (where the centered moving average is undefined); STL estimates every
//! point.
//!
//! # Example
//! ```rust
//! use inferust::seasonal::{seasonal_decompose, DecompositionModel, Stl};
//!
//! let y: Vec<f64> = (0..48)
//!     .map(|t| 10.0 + 0.2 * t as f64 + 3.0 * ((t % 12) as f64 - 5.5).abs())
//!     .collect();
//!
//! let classical = seasonal_decompose(&y, 12, DecompositionModel::Additive).unwrap();
//! assert_eq!(classical.seasonal.len(), y.len());
//!
//! let stl = Stl::new(12).fit(&y).unwrap();
//! assert_eq!(stl.trend.len(), y.len());
//! ```

use crate::error::{InferustError, Result};

// ── Classical decomposition ───────────────────────────────────────────────────

/// Composition model for [`seasonal_decompose`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecompositionModel {
    /// y = trend + seasonal + resid.
    #[default]
    Additive,
    /// y = trend × seasonal × resid.
    Multiplicative,
}

/// Result of [`seasonal_decompose`].
#[derive(Debug, Clone)]
pub struct Decomposition {
    /// The input series.
    pub observed: Vec<f64>,
    /// Centered-moving-average trend; `NaN` in the first and last
    /// `period / 2` positions where the window does not fit.
    pub trend: Vec<f64>,
    /// Periodic seasonal component (repeats every `period`).
    pub seasonal: Vec<f64>,
    /// Remainder; `NaN` wherever `trend` is `NaN`.
    pub resid: Vec<f64>,
    /// Seasonal period.
    pub period: usize,
    /// Composition model used.
    pub model: DecompositionModel,
}

/// Classical seasonal decomposition via centered moving averages, matching
/// `statsmodels.tsa.seasonal.seasonal_decompose` (`two_sided=True`,
/// `extrapolate_trend=0`).
pub fn seasonal_decompose(
    y: &[f64],
    period: usize,
    model: DecompositionModel,
) -> Result<Decomposition> {
    if period < 2 {
        return Err(InferustError::InvalidInput("period must be >= 2".into()));
    }
    if y.len() < 2 * period {
        return Err(InferustError::InsufficientData {
            needed: 2 * period,
            got: y.len(),
        });
    }
    if matches!(model, DecompositionModel::Multiplicative) && y.iter().any(|&v| v <= 0.0) {
        return Err(InferustError::InvalidInput(
            "multiplicative decomposition requires a strictly positive series".into(),
        ));
    }
    let n = y.len();

    // Centered moving-average trend. For an even period the filter is
    // [0.5, 1, …, 1, 0.5] / period of length period + 1.
    let filt: Vec<f64> = if period.is_multiple_of(2) {
        let mut f = vec![1.0 / period as f64; period + 1];
        f[0] = 0.5 / period as f64;
        f[period] = 0.5 / period as f64;
        f
    } else {
        vec![1.0 / period as f64; period]
    };
    let half = filt.len() / 2;
    let mut trend = vec![f64::NAN; n];
    for t in half..n - half {
        trend[t] = filt
            .iter()
            .enumerate()
            .map(|(j, &w)| w * y[t - half + j])
            .sum();
    }

    // Period averages of the detrended series (NaN-aware), then centered.
    let mut sums = vec![0.0_f64; period];
    let mut counts = vec![0usize; period];
    for t in 0..n {
        if trend[t].is_nan() {
            continue;
        }
        let detrended = match model {
            DecompositionModel::Additive => y[t] - trend[t],
            DecompositionModel::Multiplicative => y[t] / trend[t],
        };
        sums[t % period] += detrended;
        counts[t % period] += 1;
    }
    let mut averages: Vec<f64> = sums
        .iter()
        .zip(counts.iter())
        .map(|(&s, &c)| if c > 0 { s / c as f64 } else { f64::NAN })
        .collect();
    let mean_avg = averages.iter().sum::<f64>() / period as f64;
    match model {
        DecompositionModel::Additive => averages.iter_mut().for_each(|a| *a -= mean_avg),
        DecompositionModel::Multiplicative => averages.iter_mut().for_each(|a| *a /= mean_avg),
    }

    let seasonal: Vec<f64> = (0..n).map(|t| averages[t % period]).collect();
    let resid: Vec<f64> = (0..n)
        .map(|t| {
            if trend[t].is_nan() {
                f64::NAN
            } else {
                match model {
                    DecompositionModel::Additive => y[t] - trend[t] - seasonal[t],
                    DecompositionModel::Multiplicative => y[t] / (trend[t] * seasonal[t]),
                }
            }
        })
        .collect();

    Ok(Decomposition {
        observed: y.to_vec(),
        trend,
        seasonal,
        resid,
        period,
        model,
    })
}

// ── STL ───────────────────────────────────────────────────────────────────────

/// Season-Trend decomposition using Loess (STL).
///
/// Defaults follow `statsmodels.tsa.seasonal.STL`: `seasonal = 7`,
/// trend window = smallest odd integer ≥ `1.5·period / (1 − 1.5/seasonal)`,
/// low-pass window = smallest odd integer > `period`, all loess degrees 1,
/// 5 inner / 0 outer iterations (2 inner / 15 outer when `robust`).
///
/// Note that the iteration counts follow statsmodels' `STL.fit`, not
/// Cleveland's original Fortran, which used 2 inner passes (1 when robust).
#[derive(Debug, Clone)]
pub struct Stl {
    period: usize,
    seasonal: usize,
    trend: Option<usize>,
    low_pass: Option<usize>,
    seasonal_deg: usize,
    trend_deg: usize,
    low_pass_deg: usize,
    robust: bool,
    inner_iter: Option<usize>,
    outer_iter: Option<usize>,
}

/// Fitted STL decomposition.
#[derive(Debug, Clone)]
pub struct StlResult {
    /// The input series.
    pub observed: Vec<f64>,
    /// Trend component (defined at every point).
    pub trend: Vec<f64>,
    /// Seasonal component.
    pub seasonal: Vec<f64>,
    /// Remainder: observed − trend − seasonal.
    pub resid: Vec<f64>,
    /// Final robustness weights (all 1 when `robust` is off).
    pub weights: Vec<f64>,
    /// Seasonal period.
    pub period: usize,
}

impl Stl {
    /// Create an STL builder for the given seasonal `period` (≥ 2).
    pub fn new(period: usize) -> Self {
        Self {
            period,
            seasonal: 7,
            trend: None,
            low_pass: None,
            seasonal_deg: 1,
            trend_deg: 1,
            low_pass_deg: 1,
            robust: false,
            inner_iter: None,
            outer_iter: None,
        }
    }

    /// Seasonal loess window (odd, ≥ 3; default 7).
    pub fn seasonal(mut self, seasonal: usize) -> Self {
        self.seasonal = seasonal;
        self
    }

    /// Trend loess window (odd, > period; default from the STL heuristic).
    pub fn trend(mut self, trend: usize) -> Self {
        self.trend = Some(trend);
        self
    }

    /// Low-pass loess window (odd, > period; default smallest odd > period).
    pub fn low_pass(mut self, low_pass: usize) -> Self {
        self.low_pass = Some(low_pass);
        self
    }

    /// Use robustness iterations (bisquare-downweighted outliers).
    pub fn robust(mut self, robust: bool) -> Self {
        self.robust = robust;
        self
    }

    /// Override the number of inner-loop iterations.
    pub fn inner_iter(mut self, n: usize) -> Self {
        self.inner_iter = Some(n);
        self
    }

    /// Override the number of outer (robustness) iterations.
    pub fn outer_iter(mut self, n: usize) -> Self {
        self.outer_iter = Some(n);
        self
    }

    /// Decompose `y`.
    pub fn fit(&self, y: &[f64]) -> Result<StlResult> {
        let np = self.period;
        let n = y.len();
        if np < 2 {
            return Err(InferustError::InvalidInput("period must be >= 2".into()));
        }
        if n < 2 * np + 1 {
            return Err(InferustError::InsufficientData {
                needed: 2 * np + 1,
                got: n,
            });
        }
        if self.seasonal < 3 || self.seasonal.is_multiple_of(2) {
            return Err(InferustError::InvalidInput(
                "seasonal window must be odd and >= 3".into(),
            ));
        }
        let ns = self.seasonal;
        let nl = self.low_pass.unwrap_or_else(|| next_odd(np + 1));
        let nt = self.trend.unwrap_or_else(|| {
            next_odd((1.5 * np as f64 / (1.0 - 1.5 / ns as f64)).ceil() as usize)
        });
        for (name, v) in [("trend", nt), ("low_pass", nl)] {
            if v <= np || v % 2 == 0 {
                return Err(InferustError::InvalidInput(format!(
                    "{name} window must be odd and > period"
                )));
            }
        }
        // statsmodels' STL.fit defaults, which differ from Cleveland's Fortran
        // (2 inner / 1 robust inner): it runs 5 inner passes when not robust
        // and 2 when robust.
        let (inner, outer) = if self.robust {
            (self.inner_iter.unwrap_or(2), self.outer_iter.unwrap_or(15))
        } else {
            (self.inner_iter.unwrap_or(5), self.outer_iter.unwrap_or(0))
        };

        let mut trend = vec![0.0_f64; n];
        let mut seasonal = vec![0.0_f64; n];
        let mut rho = vec![1.0_f64; n];

        for outer_it in 0..=outer {
            for _ in 0..inner {
                // Step 1: detrend.
                let detrended: Vec<f64> = y.iter().zip(trend.iter()).map(|(a, b)| a - b).collect();

                // Step 2: cycle-subseries smoothing, extended one full period
                // before and after the series (positions −np … n+np−1).
                let mut c = vec![0.0_f64; n + 2 * np];
                for phase in 0..np {
                    let idx: Vec<usize> = (phase..n).step_by(np).collect();
                    let sub: Vec<f64> = idx.iter().map(|&i| detrended[i]).collect();
                    let w: Vec<f64> = idx.iter().map(|&i| rho[i]).collect();
                    // Smooth at subseries positions −1 ..= len (inclusive).
                    let smoothed = loess_extended(&sub, ns, self.seasonal_deg, &w)?;
                    for (k, &v) in smoothed.iter().enumerate() {
                        // k = 0 is subseries position −1, i.e. series position
                        // phase − np; the +np offset makes it index 'phase'.
                        let pos = phase + k * np;
                        if pos < c.len() {
                            c[pos] = v;
                        }
                    }
                }

                // Step 3: low-pass filter of C — MA(np), MA(np), MA(3), then loess.
                let ma1 = moving_average(&c, np);
                let ma2 = moving_average(&ma1, np);
                let ma3 = moving_average(&ma2, 3);
                let ones = vec![1.0_f64; ma3.len()];
                let low = loess_at_points(&ma3, nl, self.low_pass_deg, &ones)?;

                // Step 4: seasonal = smoothed cycle-subseries minus low-pass.
                for t in 0..n {
                    seasonal[t] = c[np + t] - low[t];
                }

                // Steps 5-6: deseasonalize and smooth the trend.
                let deseasonalized: Vec<f64> =
                    y.iter().zip(seasonal.iter()).map(|(a, s)| a - s).collect();
                trend = loess_at_points(&deseasonalized, nt, self.trend_deg, &rho)?;
            }

            // Outer loop: update robustness weights from the remainder.
            if outer_it < outer {
                let resid: Vec<f64> = (0..n).map(|t| y[t] - trend[t] - seasonal[t]).collect();
                let mut abs: Vec<f64> = resid.iter().map(|r| r.abs()).collect();
                abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = if n.is_multiple_of(2) {
                    0.5 * (abs[n / 2 - 1] + abs[n / 2])
                } else {
                    abs[n / 2]
                };
                let h = 6.0 * median;
                for t in 0..n {
                    let r = (y[t] - trend[t] - seasonal[t]).abs();
                    rho[t] = if h <= 0.0 {
                        1.0
                    } else if r >= h {
                        0.0
                    } else {
                        let u = r / h;
                        (1.0 - u * u).powi(2)
                    };
                }
            }
        }

        let resid: Vec<f64> = (0..n).map(|t| y[t] - trend[t] - seasonal[t]).collect();
        Ok(StlResult {
            observed: y.to_vec(),
            trend,
            seasonal,
            resid,
            weights: rho,
            period: np,
        })
    }
}

/// Smallest odd integer ≥ `v`.
fn next_odd(v: usize) -> usize {
    if v % 2 == 1 {
        v
    } else {
        v + 1
    }
}

/// Simple moving average with window `w`; output length is `len − w + 1`.
fn moving_average(x: &[f64], w: usize) -> Vec<f64> {
    if x.len() < w || w == 0 {
        return vec![];
    }
    let mut out = Vec::with_capacity(x.len() - w + 1);
    let mut sum: f64 = x[..w].iter().sum();
    out.push(sum / w as f64);
    for t in w..x.len() {
        sum += x[t] - x[t - w];
        out.push(sum / w as f64);
    }
    out
}

/// Loess-smooth `y` (observed at integer positions 0..n−1) evaluated at every
/// observed position, with span `q` points, local polynomial `degree` (0 or 1),
/// and multiplicative robustness `weights`.
fn loess_at_points(y: &[f64], q: usize, degree: usize, weights: &[f64]) -> Result<Vec<f64>> {
    (0..y.len() as isize)
        .map(|x| loess_point(y, q, degree, weights, x as f64))
        .collect()
}

/// Loess-smooth `y` evaluated at positions −1, 0, …, n−1, n — the observed
/// range extended one step on each side (used for cycle-subseries extension).
fn loess_extended(y: &[f64], q: usize, degree: usize, weights: &[f64]) -> Result<Vec<f64>> {
    (-1..=y.len() as isize)
        .map(|x| loess_point(y, q, degree, weights, x as f64))
        .collect()
}

/// Evaluate the loess local regression at position `x`.
fn loess_point(y: &[f64], q: usize, degree: usize, weights: &[f64], x: f64) -> Result<f64> {
    let n = y.len();
    if n == 0 {
        return Err(InferustError::InsufficientData { needed: 1, got: 0 });
    }
    if n == 1 {
        return Ok(y[0]);
    }
    let q_eff = q.min(n);
    // The q nearest observed positions to x form a contiguous block.
    let mut left = (x.ceil() as isize - q_eff as isize / 2).clamp(0, (n - q_eff) as isize) as usize;
    // Slide the block right while that reduces the span's maximum distance.
    while left + q_eff < n && (x - (left + q_eff) as f64).abs() < (x - left as f64).abs() {
        left += 1;
    }
    let right = left + q_eff - 1;
    // Cleveland's span inflation when q exceeds n.
    let mut d_max = (x - left as f64).abs().max((x - right as f64).abs());
    if q > n {
        d_max += (q - n) as f64 * 0.5;
    }
    if d_max <= 0.0 {
        d_max = 1.0;
    }

    // Tricube neighborhood weights times robustness weights.
    let mut w = vec![0.0_f64; q_eff];
    let mut w_sum = 0.0;
    for (j, wj) in w.iter_mut().enumerate() {
        let i = left + j;
        let u = ((x - i as f64).abs() / d_max).min(1.0);
        let tricube = (1.0 - u * u * u).powi(3).max(0.0);
        *wj = tricube * weights[i];
        w_sum += *wj;
    }
    if w_sum <= 0.0 {
        // All neighbors downweighted to zero: fall back to unweighted mean.
        return Ok((left..=right).map(|i| y[i]).sum::<f64>() / q_eff as f64);
    }

    if degree == 0 {
        let mut fit = 0.0;
        for (j, &wj) in w.iter().enumerate() {
            fit += wj * y[left + j];
        }
        return Ok(fit / w_sum);
    }

    // Weighted local linear fit evaluated at x.
    let (mut sw, mut swx, mut swxx, mut swy, mut swxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (j, &wj) in w.iter().enumerate() {
        let xi = (left + j) as f64;
        sw += wj;
        swx += wj * xi;
        swxx += wj * xi * xi;
        swy += wj * y[left + j];
        swxy += wj * xi * y[left + j];
    }
    let denom = sw * swxx - swx * swx;
    if denom.abs() < 1e-12 * swxx.max(1.0) {
        return Ok(swy / sw);
    }
    let slope = (sw * swxy - swx * swy) / denom;
    let intercept = (swy - slope * swx) / sw;
    Ok(intercept + slope * x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seasonal_series(n: usize, period: usize) -> Vec<f64> {
        (0..n)
            .map(|t| {
                10.0 + 0.05 * t as f64
                    + 4.0 * (2.0 * std::f64::consts::PI * (t % period) as f64 / period as f64).sin()
            })
            .collect()
    }

    #[test]
    fn classical_additive_recovers_components() {
        let period = 12;
        let y = seasonal_series(120, period);
        let d = seasonal_decompose(&y, period, DecompositionModel::Additive).unwrap();
        // Interior identity: observed = trend + seasonal + resid.
        let interior = y.len() - period;
        for (t, &obs) in y.iter().enumerate().take(interior).skip(period) {
            assert!(
                (obs - d.trend[t] - d.seasonal[t] - d.resid[t]).abs() < 1e-10,
                "identity fails at t={t}"
            );
        }
        // Seasonal averages sum to ~0 for additive.
        let s: f64 = d.seasonal[..period].iter().sum();
        assert!(s.abs() < 1e-8);
        // Trend endpoints are NaN.
        assert!(d.trend[0].is_nan() && d.trend[y.len() - 1].is_nan());
    }

    #[test]
    fn stl_recovers_smooth_seasonal_pattern() {
        let period = 12;
        let y = seasonal_series(144, period);
        let r = Stl::new(period).fit(&y).unwrap();
        // Residuals should be small for a noiseless series.
        let max_resid = r.resid.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(max_resid < 0.5, "max residual {max_resid}");
        // Seasonal component approximately repeats with the period.
        for t in 0..period {
            let diff = (r.seasonal[t + 5 * period] - r.seasonal[t + 6 * period]).abs();
            assert!(diff < 0.5, "seasonal drift {diff} at phase {t}");
        }
    }

    #[test]
    fn stl_robust_downweights_outlier() {
        let period = 12;
        let mut y = seasonal_series(144, period);
        y[60] += 50.0;
        let r = Stl::new(period).robust(true).fit(&y).unwrap();
        assert!(r.weights[60] < 0.1, "outlier weight {}", r.weights[60]);
    }
}
