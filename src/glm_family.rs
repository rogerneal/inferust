use crate::error::{InferustError, Result};
use crate::glm::{
    Gamma, GammaResult, InverseGaussian, InverseGaussianResult, Logistic, LogisticResult, Poisson,
    PoissonResult,
};
use crate::regression::{Ols, OlsResult};

/// Common GLM families with canonical links for first-pass statsmodels-style workflows.
///
/// `Gamma` dispatches to [`crate::glm::Gamma::new`], which uses the
/// canonical `InversePower` link. For a `Log` or `Identity` link, build the
/// model directly with [`crate::glm::Gamma::with_link`] instead of going
/// through this generic front-end.
///
/// `InverseGaussian` dispatches to [`crate::glm::InverseGaussian::new`], which
/// uses the `Log` link (matching `statsmodels` `InverseGaussian(Log())`). For
/// the canonical `InverseSquared` link, use
/// [`crate::glm::InverseGaussian::with_link`] directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlmFamily {
    Gaussian,
    Binomial,
    Poisson,
    Gamma,
    /// Positive continuous outcomes with inverse-Gaussian variance `V(μ)=μ³`.
    InverseGaussian,
}

/// Result wrapper returned by [`Glm`].
#[derive(Debug, Clone)]
pub enum GlmResult {
    Gaussian(OlsResult),
    Binomial(LogisticResult),
    Poisson(PoissonResult),
    Gamma(GammaResult),
    InverseGaussian(InverseGaussianResult),
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
            GlmFamily::Gamma => Gamma::new()
                .with_feature_names(self.feature_names.clone())
                .fit(x, y)
                .map(GlmResult::Gamma),
            GlmFamily::InverseGaussian => InverseGaussian::new()
                .with_feature_names(self.feature_names.clone())
                .fit(x, y)
                .map(GlmResult::InverseGaussian),
        }
    }
}

impl GlmResult {
    pub fn coefficients(&self) -> &[f64] {
        match self {
            GlmResult::Gaussian(r) => &r.coefficients,
            GlmResult::Binomial(r) => &r.coefficients,
            GlmResult::Poisson(r) => &r.coefficients,
            GlmResult::Gamma(r) => &r.coefficients,
            GlmResult::InverseGaussian(r) => &r.coefficients,
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
            "gamma" => Ok(Self::Gamma),
            "inversegaussian" | "inverse_gaussian" | "invgauss" => Ok(Self::InverseGaussian),
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

    #[test]
    fn generic_glm_dispatches_to_gamma() {
        let x = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
            vec![6.0],
        ];
        let y = vec![2.1, 3.4, 4.5, 6.0, 8.1, 10.2];
        let result = Glm::new(GlmFamily::Gamma).fit(&x, &y).unwrap();
        assert!(matches!(result, GlmResult::Gamma(_)));
    }

    #[test]
    fn generic_glm_dispatches_to_inverse_gaussian() {
        let x = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
            vec![6.0],
        ];
        let y = vec![2.1, 3.4, 4.5, 6.0, 8.1, 10.2];
        let result = Glm::new(GlmFamily::InverseGaussian).fit(&x, &y).unwrap();
        assert!(matches!(result, GlmResult::InverseGaussian(_)));
    }

    #[test]
    fn family_from_str_parses_gamma() {
        assert_eq!(GlmFamily::try_from("gamma").unwrap(), GlmFamily::Gamma);
    }

    #[test]
    fn family_from_str_parses_inverse_gaussian() {
        assert_eq!(
            GlmFamily::try_from("invgauss").unwrap(),
            GlmFamily::InverseGaussian
        );
    }
}
