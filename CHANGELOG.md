# Changelog

All notable changes to `inferust` are documented here. This project follows semantic versioning while the crate is pre-1.0: minor releases may still refine APIs, and patch releases should stay compatible within the active public surface.

## [Unreleased]

## [0.1.7] - 2026-05-01

### Added

#### Survival Analysis (`survival`)
- **Kaplan-Meier estimator** with Greenwood confidence intervals, restricted mean survival time (RMST), and median survival time.
- **Log-rank test** for comparing two survival curves (chi-squared statistic + p-value).
- **Cox Proportional Hazards regression** via Newton-Raphson partial likelihood maximisation: coefficients, hazard ratios, standard errors, Wald z-statistics, p-values, HR confidence intervals, log-likelihood, and likelihood-ratio test.

#### ARIMA / Time Series (`time_series`)
- **Full ARIMA(p,d,q)** estimation via Conditional Sum of Squares (CSS) with Adam optimiser and finite-difference gradients; replaces prior AR-only stub.
- `ArimaResult::forecast(history, steps)` with correct multi-level undifferencing using stored tails.
- **VAR (Vector Autoregression)** for multivariate time series with `VarResult::forecast()`.
- **ADF (Augmented Dickey-Fuller) unit-root test** with MacKinnon (1994) critical values and asymptotic p-value.
- **KPSS stationarity test** (level and trend) with Bartlett-kernel long-run variance and Kwiatkowski (1992) critical values.

#### Nonparametric Hypothesis Tests (`hypothesis::nonparametric`)
- **Mann-Whitney U test** with normal approximation and tie correction.
- **Kruskal-Wallis H test** with tie correction and chi-squared p-value.
- **Kolmogorov-Smirnov one-sample test** (vs. N(μ,σ)) and **two-sample test** (Marsaglia asymptotic p-value).
- **Shapiro-Wilk normality test** — Royston (1992) algorithm, valid for n = 3 … 5000.

#### Newey-West HAC Standard Errors (`regression`)
- `OlsCovariance::Hac { lags }` variant and `.hac(lags)` builder for both `Ols` and `Wls`.
- Bartlett-kernel HAC sandwich estimator — suitable for autocorrelated residuals in time-series regressions.

#### Formula API improvements (`data`)
- `FormulaTerm` enum: `Numeric`, `Categorical`, `Interaction`, `Offset` — replacing the prior flat string list.
- `Formula::parse()` supports: `C(var)` inline one-hot encoding; `x1:x2` interaction; `x1*x2` shorthand (main effects + interaction); `offset(var)` Poisson exposure; `- 1` / `+ 0` no-intercept; duplicate-term deduplication.
- `DesignMatrices` carries `intercept: bool` and `offset: Option<Vec<f64>>` so downstream models consume them automatically.
- `DataFrame::poisson()` now threads the offset through to `Poisson::with_offset()` when present.

## [0.1.6] - 2026-04-30

### Added
- Logistic regression post-estimation helpers for coefficient confidence intervals, covariance matrices, odds ratios, fitted probabilities, and delta-method average marginal-effect inference.
- Poisson count regression with MLE estimates, Wald inference, covariance matrices, fitted values, likelihood metrics, deviance, Pearson chi-square, AIC/BIC, incidence-rate ratios, prediction, formula fitting, and statsmodels parity tests.
- GLM residual diagnostics, likelihood-ratio tests, logistic classification metrics, logistic linear predictors, Poisson linear predictors, null deviances, offsets/exposure, and response-scale Poisson mean intervals.
- First-pass coverage for the remaining statsmodels roadmap: Probit, Negative Binomial, Multinomial Logit, generic GLM dispatch, categorical formula encoding, AR/ARIMA(p,d,0), Huber robust regression, independence GEE, and random-intercept mixed linear models.
- Cross-cutting diagnostics and evaluation utilities: VIF, Breusch-Pagan, White, RESET, ACF, PACF, Ljung-Box, regression metrics, confusion matrices, and bootstrap mean intervals.
- Runnable examples for diagnostics and discrete models.

### Changed
- Hardened starter modules with additional edge-case tests, validation checks, and rustdoc coverage.
- Added changelog coverage for prior releases and a standing place to record future versions before publication.

## [0.1.5] - 2026-04-29

### Added
- Weighted least squares via `regression::Wls`.
- Formula-based `DataFrame` fitting for OLS, WLS, and logistic models.
- Binary logistic regression with statsmodels-compatible coefficient estimates, Wald inference, log-likelihood, pseudo-R², AIC, BIC, and probability prediction.
- OLS HC0-HC3 robust covariance estimators, confidence intervals, influence measures, and residual diagnostics.
- Statsmodels-derived regression parity tests covering classical and robust OLS, diagnostics, influence, WLS, formulas, and logistic regression.

## [0.1.4] - 2026-04-29

### Changed
- Expanded README and crate-level docs for solver strategy, benchmark usage, robust inference, and statsmodels parity coverage.

## [0.1.3] - 2026-04-29

### Added
- Cholesky OLS fast path and SVD stable solver option.
- Benchmark tooling for inferust-vs-statsmodels timing comparisons.

### Fixed
- Aligned OLS information criteria calculations with statsmodels references.

## [0.1.2] - 2026-04-29

### Changed
- Cleaned crate metadata, repository links, CI badge references, package contents, and release documentation for crates.io publication.

## [0.1.1] - 2026-04-29

### Added
- Initial crates.io release stream with statistical modules for regression, hypothesis tests, descriptive statistics, and correlation utilities.

## [0.1.0] - 2026-04-29

### Added
- Initial repository version of `inferust` as a statsmodels-inspired Rust statistics crate.
