# Changelog

All notable changes to `inferust` are documented here. This project follows semantic versioning while the crate is pre-1.0: minor releases may still refine APIs, and patch releases should stay compatible within the active public surface.

## [0.1.18] - 2026-07-06

### Added

- **Discrete-choice parity harness** — statsmodels reference fixtures and
  `tests/parity_discrete.rs` for Probit, Negative Binomial, Multinomial Logit,
  Ordered Logit, and Zero-Inflated Poisson.
- **Formula API for discrete models** — `DataFrame::probit`,
  `negative_binomial`, `multinomial`, `ordered_logit`, and `zip`.

## [0.1.17] - 2026-07-06

### Added

- **State-space Kalman filter** (`statespace::LinearGaussianModel`) — multivariate
  linear-Gaussian filtering with exact Gaussian log-likelihood for ARMA models.
- **Exact ARIMA MLE** — `Arima::exact_mle()` / `ArimaMethod::ExactMle` fits
  MA components via state-space MLE instead of conditional sum of squares.
  `Sarima::exact_mle()` uses the same engine when seasonal orders are zero.

## [0.1.16] - 2026-06-22

### Added

- **Discrete choice models — full MLE rewrites** (`src/discrete.rs`):
  - `Probit` — IRLS with Fisher scoring weights φ²/(Φ(1−Φ)), proper SE/z/p/AIC/BIC/pseudo-R²
  - `NegativeBinomial` — NB2 alternating IRLS (β) + Newton (overdispersion α); uses `digamma` for score; log-likelihood reported
  - `MultinomialLogit` — true K-class softmax Newton-Raphson (not one-vs-rest); log-sum-exp stable; full (K-1)p×(K-1)p Hessian
  - `OrderedLogit` — proportional-odds model; reparameterized cutpoints (log-gap encoding) for ordering constraint; gradient ascent with Armijo backtracking line search; BHHH outer-product covariance
  - `ZeroInflatedPoisson` — EM algorithm; E-step posterior structural-zero probabilities; M-step weighted IRLS for count and inflation models

## [0.1.15] - 2026-06-22

### Fixed

- **`GammaLink::Identity` IRLS divergence** — linear predictor `eta` is now
  clamped to `1e-8` after each IRLS update, keeping it in the Gamma
  distribution's support and preventing divergence when early iterates go
  negative.
- **`GEE` working correlation** — replaced the independence-only stub with a
  full GEE estimator supporting `Independence`, `Exchangeable`, and `AR(1)`
  working correlation structures. Standard errors are now the empirical
  sandwich estimator, valid even when the working correlation is
  mis-specified. Result type changed to `GeeResult` (flat struct) with
  `coefficients`, `robust_std_errors`, `z_statistics`, `p_values`, `rho`,
  and `cluster_count`.
- **`MixedLinearModel` variance-component estimation** — replaced the
  group-mean-residual stub with an EM-algorithm REML estimator for
  random-intercept LMMs. Now reports variance components (`var_random`,
  `var_residual`, `icc`), EBLUP random intercepts, GLS-based fixed-effect
  standard errors, t/p values, and approximate REML log-likelihood.
  `MixedLinearResult` now carries `coefficients`, `std_errors`,
  `t_statistics`, `p_values`, `variance_components`, `iterations`, and
  `reml_loglik`.
- **`RobustLinearModel` standard errors** — added sandwich (HC) standard
  errors computed from the M-estimator bread (`X'WX`) and meat (`X'ψ²X`).
  `RobustLinearResult` now exposes `robust_std_errors`, `robust_t_statistics`,
  and `robust_p_values` alongside the existing WLS-derived `fit.std_errors`.
- **`docs/parity.md`** — removed stale GLS "bse/t/p excluded" note (that
  sigma² bug was fixed in 0.1.14).

## [0.1.14] - 2026-06-21

### Added

- **Parity coverage — nonparametric tests**: `ks_one_sample`, `ks_two_sample`,
  `kruskal_wallis`, `shapiro_wilk` now have scipy reference fixtures and
  integration tests in `tests/parity_hypothesis.rs`.
- **Parity coverage — chi-squared goodness-of-fit**: `hypothesis::chisq::goodness_of_fit`
  now tested against `scipy.stats.chisquare`.
- **Parity coverage — survival**: `KaplanMeier` (survival probabilities, n_events,
  n_censored) and `log_rank_test` (χ² statistic, p-value) now tested against
  `statsmodels.duration.survfunc.SurvfuncRight` and `scipy.stats.logrank` in
  `tests/parity_survival.rs`.
- **Parity coverage — contingency tables**: `mcnemar` and `odds_ratio_ci` / `table2x2`
  now tested against `statsmodels.stats.contingency_tables.mcnemar` and
  `scipy.stats.contingency.odds_ratio` in `tests/parity_contingency.rs`.
- **Parity coverage — diagnostics**: `variance_inflation_factors`, `breusch_pagan`,
  `white_test`, and `reset_test` now tested against statsmodels equivalents in
  `tests/parity_diagnostics.rs`.

### Notes

- KS p-values differ by up to 3 % from scipy due to different asymptotic series
  (Marsaglia 2003 vs. scipy's implementation); tolerances documented in test file.
- Shapiro-Wilk p-values differ substantially (Royston approximation vs. AS R94);
  tests verify W-statistic agreement and directional p-value agreement only.
- VIF differs by < 1 % because statsmodels always adds an intercept to the
  auxiliary regression while inferust does not.

## [0.1.13] - 2026-06-21

### Added

- **Multiple-testing corrections** — `hypothesis::multicomp::adjust(p_values, alpha, method)`
  with `Bonferroni`, `Holm`, `BenjaminiHochberg`, and `BenjaminiYekutieli`,
  matching `statsmodels.stats.multitest.multipletests` exactly.
- **Tukey HSD post-hoc test** — `hypothesis::tukey::tukey_hsd(groups, names, alpha)`
  for family-wise-error-rate-controlled pairwise comparisons after ANOVA, with
  the Tukey-Kramer adjustment for unequal group sizes. Matches
  `statsmodels.stats.multicomp.pairwise_tukeyhsd` on mean differences and
  standard errors; the q-critical value, p-values, and confidence intervals
  depend on the studentized range distribution, which inferust computes via
  quadrature (~1e-9 accurate) rather than statsmodels' interpolated table
  (~1e-3 accurate) — see `docs/parity.md`.
- **Ridge, Lasso, and ElasticNet regression** — `regression::{Ridge, Lasso,
  ElasticNet}` with closed-form ridge and coordinate-descent lasso/elastic
  net (soft-thresholding), following the scikit-learn/glmnet convention of
  never penalizing the intercept.
- **Gamma GLM** — `glm::Gamma` for positive, right-skewed continuous outcomes
  (costs, durations, claim sizes), with `InversePower` (canonical), `Log`,
  and `Identity` links, IRLS/Fisher-scoring fitting, and the same covariance,
  residual, likelihood-ratio, and prediction-interval helpers as
  `Logistic`/`Poisson`. `glm_family::GlmFamily` gained a `Gamma` variant.
- **Granger causality F-test** — `time_series::granger_causality(y, x, lag)` for
  whether lagged values of `x` help predict `y`.
- **Engle-Granger cointegration test** — `time_series::engle_granger(y, x, lags)`
  for two-step residual-based cointegration testing with one regressor.
- **Wilcoxon signed-rank** and **sign test** — `hypothesis::nonparametric::wilcoxon_signed_rank`
  and `::sign_test` for paired-sample inference; signed-rank handles zero
  differences and ties.
- **Anderson-Darling** and **Lilliefors** normality tests — added to
  `hypothesis::nonparametric` alongside Shapiro-Wilk. Both estimate mean and
  variance from the sample.
- **Wald linear-restriction tests** — `hypothesis::wald_linear(beta, cov, R, q, df)`
  plus convenience `.wald_test(R, q)` methods on `OlsResult`, `LogisticResult`,
  and `PoissonResult`. Returns both the χ² and finite-sample F forms.
- `OlsResult.covariance_matrix` is now exposed (was previously available only as
  `std_errors`).
- Parity fixtures and Rust integration tests for every feature above.

- **statsmodels parity harness** — `scripts/parity_statsmodels.py` generates
  reference JSON fixtures from `statsmodels` / `scipy.stats` on deterministic
  LCG-built datasets; integration tests under `tests/parity_*.rs` load each
  fixture and compare every output at a per-field tolerance.
- **Parity audit doc** — `docs/parity.md` defines the parity contract, the
  tolerance policy, the per-module status matrix, and the prioritized backlog
  of estimators still needing parity coverage.
- First-pass parity coverage: OLS (Nonrobust, HC0–HC3), WLS, Logit, Poisson,
  Cox PH, ACF/PACF/Ljung-Box, ADF, one- and two-sample t-tests, one-way ANOVA,
  Mann-Whitney U, chi-squared independence, Pearson/Spearman, descriptive
  summary statistics. ARIMA covered structurally (CSS vs MLE differ by design).
- Parity coverage for the Tier 1 additions above: `multicomp`, `tukey_hsd`,
  `ridge_small`/`lasso_small`/`elastic_net_small`, and `gamma_glm` (both
  links) fixtures and `tests/parity_*.rs` assertions.

## [0.1.12] - 2026-05-06

### Added

- Added `formula!` macro for Rust-native formula strings such as `formula!(y ~ x1 + C(group))`.
- Added string categorical columns to `DataFrame` via `with_categorical_column` / `add_categorical_column`, with `C(column)` treatment-dummy expansion.
- Added `regression::QuantileRegression` with IRLS fitting, asymptotic inference, pseudo R1, confidence intervals, summary output, prediction, and formula fitting via `DataFrame::quantile`.
- Added `statespace` with scalar Kalman filtering and a local-level state-space model.
- Added `gam::GaussianGam` with spline-basis additive regression.
- Added `gmm::Iv2Sls` for instrumental-variables / two-stage least-squares regression.
- Added ordered logit and zero-inflated Poisson starters to `discrete`.
- Added `multivariate` with one-way MANOVA and PCA starters.
- Added `treatment` with propensity-score IPW treatment effects and balance diagnostics.
- Added `imputation::MiceImputer` with mean imputation and chained-equation regression passes.
- Added `contingency` with 2x2 odds/risk ratios, odds-ratio intervals, McNemar, and Cochran-Mantel-Haenszel.
- Added one-way cluster-robust covariance for `Ols` and `Wls`.
- Added formula transforms (`log`, `sqrt`, `exp`) and `DataFrame::drop_missing()`.

## [0.1.10] - 2026-05-01

### Added

#### GLS / FGLS (`regression`)
- **Generalized Least Squares** (`Gls`) for arbitrary known error covariance Ω: Cholesky-factored (XᵀΩ⁻¹X)⁻¹ XᵀΩ⁻¹y transform with full Wald inference and summary.
- **Feasible GLS** (`Fgls`) via iterative Cochrane-Orcutt with Prais-Winsten first-observation correction; converges in ≤ 50 iterations, exposes estimated AR(1) autocorrelation ρ.

#### Rolling / Recursive OLS (`regression`)
- **Rolling OLS** (`RollingOls`) — independent OLS within a sliding window; returns per-window coefficients, standard errors, R², and `.slopes()` helper for a time-path of a single coefficient.
- **Recursive OLS** (`RecursiveOls`) — Sherman-Morrison rank-1 covariance update; computes recursive residuals, CUSUM path, and Brown-Durbin-Evans (1975) boundaries; `.cusum_reject()` and `.print_cusum()` helpers.

#### Seasonal Models (`time_series`)
- **SARIMA(p,d,q)(P,D,Q,s)** — multiplicative seasonal differencing, CSS estimation, Adam optimiser; `SarimaResult::forecast(history, steps)` with correct multi-level undifferencing.
- **SARIMAX** — exogenous regressors projected out via OLS before SARIMA fit on residuals; exposes `exog_coefficients` alongside the full `SarimaResult`.
- **VECM** (`Vecm`) — Johansen reduced-rank regression via symmetrized generalized EVP; trace statistics, cointegrating vectors β, adjustment speeds α, short-run matrices Γ, and `print_summary`.
- **VARMAX** (`Varmax`) — VAR extended with exogenous columns in each equation's OLS regressor matrix; `VarmaxResult::forecast(history, exog_future)`.

#### Plot module (`plot`)
- New `Plot` builder with `line`, `scatter`, `bar`, `step`, `band`, `hline` series types.
- Convenience constructors: `Plot::acf`, `Plot::survival`, `Plot::residuals`.
- `to_svg() -> String` renders a self-contained SVG with clip-path, axis labels, title, and per-series legend.
- `save(path)` writes the SVG file; `print_ascii()` renders a 70 × 20 grid to the terminal.

## [0.1.9] - 2026-05-01

### Fixed

- Relaxed the Kruskal-Wallis small-sample test expectation to the attainable chi-squared p-value for three groups of size three.
- Stabilized the Shapiro-Wilk p-value approximation for small near-normal samples.
- Added light diagonal regularization for Cox PH information-matrix solves in near-singular fixtures.
- Added Durbin-Watson, Jarque-Bera, skew, kurtosis, and condition number diagnostics to the default OLS/WLS printed summary.
- Cleaned up clippy warnings in release-touched modules and refreshed completed README roadmap items.

## [0.1.8] - 2026-05-01

### Added

- Exposed `regression::Gls` / `Fgls` and `RollingOls` / `RecursiveOls` from the public regression module.
- Added seasonal time-series starters: `Sarima`, `Sarimax`, `Vecm`, and `Varmax`, with forecasts and summary helpers where applicable.
- Added `graphics` SVG plotting helpers for line, scatter, residual, and ACF plots.

### Fixed

- Hardened GLS/SARIMAX fitted-value calculations against nalgebra row/vector shape panics.
- Added regularized matrix-inverse fallbacks for collinear starter VAR/VECM/VARMAX examples.

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
