//! # inferust
//!
//! **Statistical modeling for Rust** — a `statsmodels`-inspired library.
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|---------|
//! | [`regression`] | OLS/WLS/GLS/FGLS, quantile regression, plus rolling and recursive OLS with fast/stable/HAC solvers, HC0-HC3 and Newey-West SEs, confidence intervals, influence diagnostics, full summary |
//! | [`glm`] | Binary logistic and Poisson regression with Wald inference, LRT, residual diagnostics, prediction intervals, classification metrics |
//! | [`gam`] | Gaussian additive models with spline basis expansion |
//! | [`gmm`] | Instrumental variables / 2SLS econometrics starter |
//! | [`survival`] | Kaplan-Meier estimator, log-rank test, Cox proportional hazards regression |
//! | [`statespace`] | Scalar Kalman filter and local-level state-space model |
//! | [`time_series`] | ARIMA/SARIMA/SARIMAX via CSS, VAR/VECM/VARMAX, AR, ACF/PACF, Ljung-Box, ADF unit root, KPSS stationarity |
//! | [`hypothesis`] | t-tests, chi-squared, ANOVA, Mann-Whitney U, Kruskal-Wallis, KS tests, Shapiro-Wilk |
//! | [`contingency`] | 2x2 tables, odds/risk ratios, McNemar, CMH |
//! | [`diagnostics`] | VIF, Breusch-Pagan, White, and RESET diagnostics |
//! | [`discrete`] | Probit, ordered logit, negative binomial, multinomial logit, and zero-inflated Poisson |
//! | [`glm_family`] | Generic Gaussian/Binomial/Poisson GLM front-end |
//! | [`multivariate`] | MANOVA and PCA starters |
//! | [`imputation`] | Mean imputation and MICE-style chained equations |
//! | [`treatment`] | Propensity scores, IPW treatment effects, and balance diagnostics |
//! | [`evaluation`] | Regression/classification metrics and bootstrap intervals |
//! | [`graphics`] | Dependency-light SVG plots for lines, scatter, residuals, and ACF bars |
//! | [`plot`] | Full-featured SVG/ASCII plot builder: line, scatter, bar, step, band, ACF, survival, and residual charts |
//! | [`robust`] | Huber robust linear regression |
//! | [`gee`] | Independence-working-correlation GEE |
//! | [`mixed`] | Random-intercept mixed linear model |
//! | [`descriptive`] | Summary stats (mean, std, skewness, kurtosis, quartiles) |
//! | [`data`] | Named-column DataFrame with formula API: `y ~ C(g) + x1*x2 - 1 + offset(e)` |
//! | [`correlation`] | Pearson, Spearman, correlation matrices |
//!
//! ## OLS covariance options
//!
//! [`regression::Ols`] defaults to classical (homoskedastic) standard errors.
//! Switch with `.robust()` (HC1), `.with_covariance(OlsCovariance::Hc3)`, or
//! `.hac(lags)` (Newey-West) for time series regressions.
//!
//! ## Formula syntax
//!
//! [`data::DataFrame`] accepts R-style formulas:
//! - `"y ~ x1 + x2"`            — main effects
//! - `"y ~ x1 + x2 - 1"`        — no intercept
//! - `"y ~ C(group) + x1"`      — inline one-hot encoding
//! - `"y ~ x1:x2"`              — interaction term
//! - `"y ~ x1 * x2"`            — main effects + interaction
//! - `"y ~ log(x) + sqrt(z)"`   — numeric transforms
//! - `"y ~ x + offset(exp)"`    — Poisson offset
//!
//! For Rust-native ergonomics, the [`formula!`] macro turns formula-like tokens
//! into a formula string:
//!
//! ```rust
//! let f = inferust::formula!(y ~ x1 + C(group));
//! assert_eq!(f, "y ~ x1 + C(group)");
//! ```
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

pub mod contingency;
pub mod correlation;
pub mod data;
pub mod descriptive;
pub mod diagnostics;
pub mod discrete;
pub mod error;
pub mod evaluation;
pub mod gam;
pub mod gee;
pub mod glm;
pub mod glm_family;
pub mod gmm;
pub mod graphics;
pub mod hypothesis;
pub mod imputation;
pub mod mixed;
pub mod multivariate;
pub mod plot;
pub mod regression;
pub mod robust;
pub mod statespace;
pub mod survival;
pub mod time_series;
pub mod treatment;

pub use error::{InferustError, Result};

/// Build a formula string from Rust tokens.
///
/// This is a lightweight convenience for APIs that accept `&str`, such as
/// `DataFrame::ols`, `DataFrame::wls`, `DataFrame::logistic`, and
/// `DataFrame::poisson`.
///
/// # Example
/// ```rust
/// let f = inferust::formula!(score ~ hours + C(classroom));
/// assert_eq!(f, "score ~ hours + C(classroom)");
/// ```
#[macro_export]
macro_rules! formula {
    ($lhs:ident ~ $($rhs:tt)+) => {
        concat!(stringify!($lhs), " ~ ", stringify!($($rhs)+))
    };
}
