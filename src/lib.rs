//! # inferust
//!
//! **Statistical modeling for Rust** — a `statsmodels`-inspired library.
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|---------|
//! | [`regression`] | OLS/WLS with fast/stable solvers, robust SEs, confidence intervals, influence/residual diagnostics, and full summary output |
//! | [`glm`] | Binary logistic and Poisson regression with Wald inference, covariance, residual diagnostics, likelihood-ratio tests, prediction intervals, classification metrics, and post-estimation helpers |
//! | [`discrete`] | Probit, negative binomial, and multinomial logit starters |
//! | [`glm_family`] | Generic Gaussian/Binomial/Poisson GLM front-end |
//! | [`time_series`] | AR/ARIMA starters plus ACF, PACF, and Ljung-Box diagnostics |
//! | [`diagnostics`] | VIF, Breusch-Pagan, White, and RESET diagnostics |
//! | [`evaluation`] | Regression/classification metrics and bootstrap intervals |
//! | [`robust`] | Huber robust linear regression |
//! | [`gee`] | Independence-working-correlation GEE starters |
//! | [`mixed`] | Random-intercept mixed linear model starter |
//! | [`hypothesis`] | t-tests, chi-squared, one-way ANOVA |
//! | [`descriptive`] | Summary stats (mean, std, skewness, kurtosis, quartiles) |
//! | [`data`] | Minimal named-column data frame and formula-based OLS/WLS/logistic/Poisson fitting |
//! | [`correlation`] | Pearson, Spearman, correlation matrices |
//!
//! ## OLS solver strategy
//!
//! [`regression::Ols`] defaults to a fast Cholesky solve for full-rank,
//! well-conditioned designs. Use `.stable()` or
//! [`regression::OlsSolver::Svd`] when you prefer the more robust SVD path.
//! Use `.robust()` or [`regression::OlsCovariance`] for HC robust inference.
//!
//! ## Quick start
//!
//! ```rust
//! use inferust::regression::Ols;
//!
//! let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
//! let y = vec![2.1, 3.9, 6.2, 7.8, 10.1];
//!
//! Ols::new()
//!     .with_feature_names(vec!["hours".to_string()])
//!     .fit(&x, &y)
//!     .unwrap()
//!     .print_summary();
//! ```

pub mod correlation;
pub mod data;
pub mod descriptive;
pub mod diagnostics;
pub mod discrete;
pub mod error;
pub mod evaluation;
pub mod gee;
pub mod glm;
pub mod glm_family;
pub mod hypothesis;
pub mod mixed;
pub mod regression;
pub mod robust;
pub mod time_series;

pub use error::{InferustError, Result};
