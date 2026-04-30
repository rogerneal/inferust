use crate::error::{InferustError, Result};

#[derive(Debug, Clone)]
pub struct RegressionMetrics {
    pub mae: f64,
    pub mse: f64,
    pub rmse: f64,
    pub mape: f64,
    pub r_squared: f64,
}

#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    pub threshold: f64,
    pub true_positives: usize,
    pub false_positives: usize,
    pub true_negatives: usize,
    pub false_negatives: usize,
}

#[derive(Debug, Clone)]
pub struct BootstrapInterval {
    pub estimate: f64,
    pub lower: f64,
    pub upper: f64,
}

pub fn regression_metrics(y_true: &[f64], y_pred: &[f64]) -> Result<RegressionMetrics> {
    validate_same_len(y_true, y_pred)?;
    let n = y_true.len() as f64;
    let mean = y_true.iter().sum::<f64>() / n;
    let mae = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(actual, pred)| (actual - pred).abs())
        .sum::<f64>()
        / n;
    let mse = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(actual, pred)| (actual - pred).powi(2))
        .sum::<f64>()
        / n;
    let nonzero_actuals = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(actual, _)| actual.abs() > f64::EPSILON)
        .collect::<Vec<_>>();
    let mape = if nonzero_actuals.is_empty() {
        f64::NAN
    } else {
        nonzero_actuals
            .iter()
            .map(|(actual, pred)| ((*actual - *pred) / *actual).abs())
            .sum::<f64>()
            / nonzero_actuals.len() as f64
    };
    let ss_res = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(actual, pred)| (actual - pred).powi(2))
        .sum::<f64>();
    let ss_tot = y_true
        .iter()
        .map(|actual| (actual - mean).powi(2))
        .sum::<f64>();
    Ok(RegressionMetrics {
        mae,
        mse,
        rmse: mse.sqrt(),
        mape,
        r_squared: 1.0 - ss_res / ss_tot.max(1e-12),
    })
}

pub fn confusion_matrix(
    y_true: &[f64],
    probabilities: &[f64],
    threshold: f64,
) -> Result<ConfusionMatrix> {
    validate_same_len(y_true, probabilities)?;
    if !(0.0..1.0).contains(&threshold) {
        return Err(InferustError::InvalidInput(
            "threshold must be between 0 and 1".into(),
        ));
    }
    let mut matrix = ConfusionMatrix {
        threshold,
        true_positives: 0,
        false_positives: 0,
        true_negatives: 0,
        false_negatives: 0,
    };
    for (actual, probability) in y_true.iter().zip(probabilities.iter()) {
        match (*probability >= threshold, *actual == 1.0) {
            (true, true) => matrix.true_positives += 1,
            (true, false) => matrix.false_positives += 1,
            (false, false) => matrix.true_negatives += 1,
            (false, true) => matrix.false_negatives += 1,
        }
    }
    Ok(matrix)
}

pub fn bootstrap_mean_interval(
    data: &[f64],
    resamples: usize,
    alpha: f64,
) -> Result<BootstrapInterval> {
    if data.is_empty() {
        return Err(InferustError::InsufficientData { needed: 1, got: 0 });
    }
    if resamples < 2 {
        return Err(InferustError::InsufficientData {
            needed: 2,
            got: resamples,
        });
    }
    if !(0.0..1.0).contains(&alpha) {
        return Err(InferustError::InvalidInput(
            "alpha must be between 0 and 1".into(),
        ));
    }
    let estimate = data.iter().sum::<f64>() / data.len() as f64;
    let mut rng = Lcg::new(0x5eed_u64);
    let mut means = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let mut total = 0.0;
        for _ in 0..data.len() {
            total += data[rng.next_index(data.len())];
        }
        means.push(total / data.len() as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lower_idx = ((alpha / 2.0) * (resamples - 1) as f64).round() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * (resamples - 1) as f64).round() as usize;
    Ok(BootstrapInterval {
        estimate,
        lower: means[lower_idx],
        upper: means[upper_idx],
    })
}

fn validate_same_len(left: &[f64], right: &[f64]) -> Result<()> {
    if left.len() != right.len() {
        return Err(InferustError::DimensionMismatch {
            x_rows: right.len(),
            y_len: left.len(),
        });
    }
    if left.is_empty() {
        return Err(InferustError::InsufficientData { needed: 1, got: 0 });
    }
    Ok(())
}

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_index(&mut self, len: usize) -> usize {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.state >> 32) as usize) % len
    }
}

#[cfg(test)]
mod tests {
    use super::{bootstrap_mean_interval, confusion_matrix, regression_metrics};

    #[test]
    fn computes_evaluation_metrics() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let pred = vec![1.1, 1.9, 3.2, 3.8];
        let metrics = regression_metrics(&y, &pred).unwrap();
        assert!(metrics.rmse > 0.0);
        let matrix = confusion_matrix(&[0.0, 1.0, 1.0], &[0.2, 0.7, 0.4], 0.5).unwrap();
        assert_eq!(matrix.true_positives, 1);
        let interval = bootstrap_mean_interval(&y, 50, 0.05).unwrap();
        assert!(interval.lower <= interval.upper);
    }

    #[test]
    fn regression_metrics_handles_zero_actual_mape() {
        let metrics = regression_metrics(&[0.0, 0.0], &[0.1, 0.2]).unwrap();
        assert!(metrics.mape.is_nan());
    }
}
