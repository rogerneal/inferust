use crate::error::{InferustError, Result};
use crate::regression::Ols;

#[derive(Debug, Clone)]
pub struct AutoRegressive {
    lags: usize,
}

#[derive(Debug, Clone)]
pub struct AutoRegressiveResult {
    pub intercept: f64,
    pub coefficients: Vec<f64>,
    pub fitted_values: Vec<f64>,
    pub residuals: Vec<f64>,
    pub sigma2: f64,
    pub n: usize,
}

impl AutoRegressive {
    pub fn new(lags: usize) -> Self {
        Self { lags }
    }

    pub fn fit(&self, y: &[f64]) -> Result<AutoRegressiveResult> {
        if self.lags == 0 {
            return Err(InferustError::InvalidInput(
                "AR lags must be positive".into(),
            ));
        }
        if y.len() <= self.lags + 1 {
            return Err(InferustError::InsufficientData {
                needed: self.lags + 2,
                got: y.len(),
            });
        }
        let mut x = Vec::with_capacity(y.len() - self.lags);
        let mut target = Vec::with_capacity(y.len() - self.lags);
        for t in self.lags..y.len() {
            let mut row = Vec::with_capacity(self.lags);
            for lag in 1..=self.lags {
                row.push(y[t - lag]);
            }
            x.push(row);
            target.push(y[t]);
        }
        let result = Ols::new().fit(&x, &target)?;
        let intercept = result.coefficients[0];
        let coefficients = result.coefficients[1..].to_vec();
        Ok(AutoRegressiveResult {
            intercept,
            coefficients,
            fitted_values: result.fitted_values,
            residuals: result.residuals,
            sigma2: result.mse_resid,
            n: target.len(),
        })
    }
}

impl AutoRegressiveResult {
    pub fn forecast(&self, history: &[f64], steps: usize) -> Result<Vec<f64>> {
        if history.len() < self.coefficients.len() {
            return Err(InferustError::InsufficientData {
                needed: self.coefficients.len(),
                got: history.len(),
            });
        }
        let mut values = history.to_vec();
        let mut out = Vec::with_capacity(steps);
        for _ in 0..steps {
            let mut next = self.intercept;
            for (lag, coef) in self.coefficients.iter().enumerate() {
                next += coef * values[values.len() - lag - 1];
            }
            values.push(next);
            out.push(next);
        }
        Ok(out)
    }
}

#[derive(Debug, Clone)]
pub struct Arima {
    p: usize,
    d: usize,
    q: usize,
}

impl Arima {
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        Self { p, d, q }
    }

    pub fn fit(&self, y: &[f64]) -> Result<AutoRegressiveResult> {
        if self.q != 0 {
            return Err(InferustError::InvalidInput(
                "ARIMA currently supports q=0 as an ARIMA(p,d,0) starter".into(),
            ));
        }
        let mut series = y.to_vec();
        for _ in 0..self.d {
            if series.len() < 2 {
                return Err(InferustError::InsufficientData {
                    needed: 2,
                    got: series.len(),
                });
            }
            series = series.windows(2).map(|pair| pair[1] - pair[0]).collect();
        }
        AutoRegressive::new(self.p).fit(&series)
    }
}

#[derive(Debug, Clone)]
pub struct LjungBoxResult {
    pub lag: usize,
    pub statistic: f64,
    pub p_value: f64,
}

pub fn acf(series: &[f64], max_lag: usize) -> Result<Vec<f64>> {
    if series.len() < 2 {
        return Err(InferustError::InsufficientData {
            needed: 2,
            got: series.len(),
        });
    }
    let mean = series.iter().sum::<f64>() / series.len() as f64;
    let denom = series
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>();
    Ok((0..=max_lag)
        .map(|lag| {
            if lag == 0 {
                1.0
            } else {
                series
                    .iter()
                    .skip(lag)
                    .zip(series.iter())
                    .map(|(current, previous)| (current - mean) * (previous - mean))
                    .sum::<f64>()
                    / denom.max(1e-12)
            }
        })
        .collect())
}

pub fn pacf(series: &[f64], max_lag: usize) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(max_lag + 1);
    out.push(1.0);
    for lag in 1..=max_lag {
        let fit = AutoRegressive::new(lag).fit(series)?;
        out.push(*fit.coefficients.last().unwrap_or(&0.0));
    }
    Ok(out)
}

pub fn ljung_box(series: &[f64], max_lag: usize) -> Result<Vec<LjungBoxResult>> {
    let correlations = acf(series, max_lag)?;
    let n = series.len() as f64;
    let mut results = Vec::with_capacity(max_lag);
    for lag in 1..=max_lag {
        let statistic = n
            * (n + 2.0)
            * correlations
                .iter()
                .enumerate()
                .skip(1)
                .take(lag)
                .map(|(k, rho)| rho.powi(2) / (n - k as f64))
                .sum::<f64>();
        let chi = statrs::distribution::ChiSquared::new(lag as f64)
            .map_err(|_| InferustError::InvalidInput("invalid chi-squared distribution".into()))?;
        results.push(LjungBoxResult {
            lag,
            statistic,
            p_value: 1.0 - statrs::distribution::ContinuousCDF::cdf(&chi, statistic),
        });
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::{acf, ljung_box, pacf, Arima, AutoRegressive};

    #[test]
    fn fits_ar_and_forecasts() {
        let y = vec![1.0, 1.8, 2.7, 3.5, 4.6, 5.4, 6.5, 7.4, 8.3];
        let fit = AutoRegressive::new(1).fit(&y).unwrap();
        assert_eq!(fit.coefficients.len(), 1);
        assert_eq!(fit.forecast(&y, 2).unwrap().len(), 2);
    }

    #[test]
    fn arima_starter_differences_series() {
        let y = vec![1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0];
        let fit = Arima::new(1, 1, 0).fit(&y).unwrap();
        assert_eq!(fit.coefficients.len(), 1);
    }

    #[test]
    fn computes_autocorrelation_diagnostics() {
        let y = vec![1.0, 1.8, 2.7, 3.5, 4.6, 5.4, 6.5, 7.4, 8.3];
        assert_eq!(acf(&y, 3).unwrap().len(), 4);
        assert_eq!(pacf(&y, 2).unwrap().len(), 3);
        assert_eq!(ljung_box(&y, 2).unwrap().len(), 2);
    }
}
