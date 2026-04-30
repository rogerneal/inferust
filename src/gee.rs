use crate::error::Result;
use crate::glm::{GlmResiduals, Logistic, LogisticResult, Poisson, PoissonResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeeFamily {
    Binomial,
    Poisson,
}

#[derive(Debug, Clone)]
pub enum GeeResult {
    Binomial {
        fit: LogisticResult,
        clusters: Vec<usize>,
    },
    Poisson {
        fit: PoissonResult,
        clusters: Vec<usize>,
    },
}

#[derive(Debug, Clone)]
pub struct Gee {
    family: GeeFamily,
    feature_names: Vec<String>,
}

impl Gee {
    pub fn new(family: GeeFamily) -> Self {
        Self {
            family,
            feature_names: Vec::new(),
        }
    }

    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Fits a first-pass independence-working-correlation GEE.
    pub fn fit(&self, x: &[Vec<f64>], y: &[f64], clusters: &[usize]) -> Result<GeeResult> {
        match self.family {
            GeeFamily::Binomial => Logistic::new()
                .with_feature_names(self.feature_names.clone())
                .fit(x, y)
                .map(|fit| GeeResult::Binomial {
                    fit,
                    clusters: clusters.to_vec(),
                }),
            GeeFamily::Poisson => Poisson::new()
                .with_feature_names(self.feature_names.clone())
                .fit(x, y)
                .map(|fit| GeeResult::Poisson {
                    fit,
                    clusters: clusters.to_vec(),
                }),
        }
    }
}

impl GeeResult {
    pub fn residuals(&self) -> GlmResiduals {
        match self {
            GeeResult::Binomial { fit, .. } => fit.residuals(),
            GeeResult::Poisson { fit, .. } => fit.residuals(),
        }
    }

    pub fn cluster_count(&self) -> usize {
        let mut clusters = match self {
            GeeResult::Binomial { clusters, .. } => clusters.clone(),
            GeeResult::Poisson { clusters, .. } => clusters.clone(),
        };
        clusters.sort_unstable();
        clusters.dedup();
        clusters.len()
    }
}

#[cfg(test)]
mod tests {
    use super::{Gee, GeeFamily, GeeResult};

    #[test]
    fn fits_independence_poisson_gee() {
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let y = vec![1.0, 2.0, 3.0, 5.0, 8.0];
        let clusters = vec![1, 1, 2, 2, 2];
        let result = Gee::new(GeeFamily::Poisson).fit(&x, &y, &clusters).unwrap();
        assert!(matches!(result, GeeResult::Poisson { .. }));
        assert_eq!(result.cluster_count(), 2);
    }
}
