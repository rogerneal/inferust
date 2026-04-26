use crate::error::{InferustError, Result};

/// Summary statistics for a slice of f64 values.
#[derive(Debug, Clone)]
pub struct Summary {
    pub n: usize,
    pub mean: f64,
    pub std: f64,
    pub variance: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub q1: f64,
    pub q3: f64,
    pub skewness: f64,
    /// Excess kurtosis (Fisher's definition; 0 for a normal distribution).
    pub kurtosis: f64,
}

impl Summary {
    /// Compute summary statistics. Requires at least 2 observations.
    pub fn new(data: &[f64]) -> Result<Self> {
        let n = data.len();
        if n < 2 {
            return Err(InferustError::InsufficientData { needed: 2, got: n });
        }

        let mean = data.iter().sum::<f64>() / n as f64;
        let variance =
            data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let std = variance.sqrt();

        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = sorted[0];
        let max = *sorted.last().unwrap();
        let median = percentile(&sorted, 0.50);
        let q1 = percentile(&sorted, 0.25);
        let q3 = percentile(&sorted, 0.75);

        // Skewness and excess kurtosis use population moments (divided by n).
        let skewness = if std == 0.0 {
            0.0
        } else {
            data.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>() / n as f64
        };
        let kurtosis = if std == 0.0 {
            0.0
        } else {
            data.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f64>() / n as f64 - 3.0
        };

        Ok(Summary {
            n,
            mean,
            std,
            variance,
            min,
            max,
            median,
            q1,
            q3,
            skewness,
            kurtosis,
        })
    }

    /// Print a human-readable summary to stdout.
    pub fn print(&self) {
        println!("─────────────────────────────");
        println!(" n          : {}", self.n);
        println!(" mean       : {:.6}", self.mean);
        println!(" std        : {:.6}", self.std);
        println!(" variance   : {:.6}", self.variance);
        println!(" min        : {:.6}", self.min);
        println!(" 25%        : {:.6}", self.q1);
        println!(" 50%        : {:.6}", self.median);
        println!(" 75%        : {:.6}", self.q3);
        println!(" max        : {:.6}", self.max);
        println!(" skewness   : {:.6}", self.skewness);
        println!(" kurtosis   : {:.6}", self.kurtosis);
        println!("─────────────────────────────");
    }
}

/// Linear interpolation percentile on a pre-sorted slice.
pub(crate) fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = p * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = idx - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}
