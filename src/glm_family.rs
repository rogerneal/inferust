use crate::error::{InferustError, Result};
use crate::glm::{Logistic, LogisticResult, Poisson, PoissonResult};
use crate::regression::{Ols, OlsResult};

/// Common GLM families with canonical links for first-pass statsmodels-style workflows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlmFamily {
    Gaussian,
    Binomial,
    Poisson,
}

/// Result wrapper returned by [`Glm`].
#[derive(Debug, Clone)]
pub enum GlmResult {
    Gaussian(OlsResult),
    Binomial(LogisticResult),
    Poisson(PoissonResult),
}

/// Small generic GLM front-end that dispatches to the crate's concrete model engines.
#[derive(Debug, Clone)]
pub struct Glm {
    family: GlmFamily,
    feature_names: Vec<String>,
}

impl Glm {
    pub fn new(family: GlmFamily) -> Self {
        Self {
            family,
            feature_names: Vec::new(),
        }
    }

    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    pub fn fit(&self, x: &[Vec<f64>], y: &[f64]) -> Result<GlmResult> {
        match self.family {
            GlmFamily::Gaussian => Ols::new()
                .with_feature_names(self.feature_names.clone())
                .fit(x, y)
                .map(GlmResult::Gaussian),
            GlmFamily::Binomial => Logistic::new()
                .with_feature_names(self.feature_names.clone())
                .fit(x, y)
                .map(GlmResult::Binomial),
            GlmFamily::Poisson => Poisson::new()
                .with_feature_names(self.feature_names.clone())
                .fit(x, y)
                .map(GlmResult::Poisson),
        }
    }
}

impl TryFrom<&str> for GlmFamily {
    type Error = InferustError;

    fn try_from(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "gaussian" | "normal" => Ok(Self::Gaussian),
            "binomial" | "logit" | "logistic" => Ok(Self::Binomial),
            "poisson" => Ok(Self::Poisson),
            other => Err(InferustError::InvalidInput(format!(
                "unsupported GLM family `{other}`"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Glm, GlmFamily, GlmResult};

    #[test]
    fn generic_glm_dispatches_to_poisson() {
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let y = vec![1.0, 2.0, 3.0, 5.0, 8.0];
        let result = Glm::new(GlmFamily::Poisson).fit(&x, &y).unwrap();
        assert!(matches!(result, GlmResult::Poisson(_)));
    }
}
