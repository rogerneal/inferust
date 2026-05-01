//! # inferust
//!
//! **Statistical modeling for Rust** — a `statsmodels`-inspired library.
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|---------|
//! | [`regression`] | OLS/WLS with fast/stable/HAC solvers, HC0-HC3 and Newey-West SEs, confidence intervals, influence diagnostics, full summary |
//! | [`glm`] | Binary logistic and Poisson regression with Wald inference, LRT, residual diagnostics, prediction intervals, classification metrics |
//! | [`survival`] | Kaplan-Meier estimator, log-rank test, Cox proportional hazards regression |
//! | [`time_series`] | Full ARIMA(p,d,q) via CSS, VAR, AR, ACF/PACF, Ljung-Box, ADF unit root, KPSS stationarity |
//! | [`hypothesis`] | t-tests, chi-squared, ANOVA, Mann-Whitney U, Kruskal-Wallis, KS tests, Shapiro-Wilk |
//! | [`diagnostics`] | VIF, Breusch-Pagan, White, and RESET diagnostics |
//! | [`discrete`] | Probit, negative binomial, and multinomial logit |
//! | [`glm_family`] | Generic Gaussian/Binomial/Poisson GLM front-end |
//! | [`evaluation`] | Regression/classification metrics and bootstrap intervals |
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
//! - `"y ~ x + offset(exp)"`    — Poisson offset
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
pub mod survival;
pub mod time_series;

pub use error::{InferustError, Result};
