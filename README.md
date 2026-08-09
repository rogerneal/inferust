# inferust

[![Crates.io](https://img.shields.io/crates/v/inferust.svg)](https://crates.io/crates/inferust)
[![Docs.rs](https://docs.rs/inferust/badge.svg)](https://docs.rs/inferust)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/rogerneal/inferust/ci.yml?branch=master)](https://github.com/rogerneal/inferust/actions)

**Statistical modeling for Rust** -  a `statsmodels`-inspired library.

`inferust` fills the gap between Python's `statsmodels` / `scipy.stats` and the Rust ecosystem. It gives you regression summaries, hypothesis tests, descriptive stats, and correlation matrices with the same depth of output you'd expect from Python -  p-values, confidence intervals, AIC/BIC, significance stars, and all.

---

## Features

| Module | What you get | Python equivalent |
|--------|-------------|-------------------|
| `regression::Ols` / `Wls` / `Gls` / `Fgls` / `QuantileRegression` | OLS, weighted least squares, GLS with known covariance, AR(1) feasible GLS, and quantile regression with fast/stable solvers, robust/HAC/cluster SEs, confidence intervals, influence diagnostics, residual diagnostics, Durbin-Watson, Jarque-Bera, condition numbers, t/z stats, p-values, R², adj-R², pseudo R¹, F-stat, AIC, BIC | `statsmodels.OLS().fit()`, `statsmodels.WLS().fit()`, `statsmodels.GLS().fit()`, `statsmodels.GLSAR()`, `statsmodels.QuantReg().fit()` |
| `regression::RollingOls` / `RecursiveOls` | Rolling-window coefficient paths and recursive OLS with CUSUM stability diagnostics | `statsmodels.regression.rolling.RollingOLS`, `statsmodels.regression.recursive_ls.RecursiveLS` basics |
| `regression::Ridge` / `Lasso` / `ElasticNet` | L2/L1/mixed-penalty regularized regression (closed-form ridge, coordinate-descent lasso/elastic net), never penalizing the intercept | `statsmodels.OLS().fit_regularized()`, scikit-learn's `Ridge`/`Lasso`/`ElasticNet` |
| `hypothesis::ttest` | One-sample, two-sample Welch, paired t-tests with 95% CI | `scipy.stats.ttest_*` |
| `hypothesis::chisq` / `contingency` | Goodness-of-fit, independence, 2x2 odds/risk ratios, McNemar, and CMH | `scipy.stats.chisquare`, `chi2_contingency`, `statsmodels.stats.contingency_tables` |
| `hypothesis::anova` | One-way ANOVA table (SS, MS, F, p) and two-way factorial ANOVA with interaction, Type I and Type II sums of squares | `scipy.stats.f_oneway`, `statsmodels.stats.anova.anova_lm` |
| `hypothesis::tukey` | Tukey HSD post-hoc pairwise comparisons (Tukey-Kramer adjusted) | `statsmodels.stats.multicomp.pairwise_tukeyhsd` |
| `hypothesis::multicomp` | Multiple-testing p-value correction (Bonferroni, Holm, Benjamini-Hochberg, Benjamini-Yekutieli) | `statsmodels.stats.multitest.multipletests` |
| `descriptive::Summary` | mean, std, variance, min/max, quartiles, skewness, excess kurtosis | `pd.Series.describe()` |
| `data::DataFrame` | named numeric/string columns, `formula!` macro, transforms, missing-row dropping, and formula-based OLS/WLS/quantile/logistic/Poisson fitting with categorical dummy expansion | `statsmodels.formula.api` basics |
| `glm::Logistic` / `Poisson` / `Gamma` | binary logistic, Poisson count, and Gamma (positive continuous) regression with MLE/IRLS estimates, Wald inference, covariance, residual diagnostics, likelihood-ratio tests, prediction intervals, classification metrics, and post-estimation helpers | `statsmodels.Logit().fit()`, `statsmodels.GLM(..., Poisson()).fit()`, `statsmodels.GLM(..., Gamma()).fit()` |
| `gam::GaussianGam` | additive Gaussian regression with spline basis expansion and statsmodels-style OLS summaries on the expanded design | `statsmodels.gam.GLMGam` basics |
| `gmm::Iv2Sls` | instrumental variables regression via two-stage least squares with t inference and summary output | `statsmodels.sandbox.regression.gmm.IV2SLS`, `statsmodels.gmm` basics |
| `discrete` | Probit, ordered logit, negative binomial, multinomial logit, and zero-inflated Poisson starters | `statsmodels.discrete` basics |
| `glm_family` | generic Gaussian/Binomial/Poisson/Gamma GLM dispatch | `statsmodels.GLM` basics |
| `multivariate` | one-way MANOVA and PCA starters | `statsmodels.multivariate` basics |
| `imputation` | mean imputation and MICE-style chained equations | `statsmodels.imputation.mice` basics |
| `treatment` | propensity scores, IPW ATE/ATT, and balance diagnostics | `statsmodels.treatment` basics |
| `statespace` | scalar Kalman filter and local-level state-space smoothing/forecasting | `statsmodels.tsa.statespace` basics |
| `time_series` | AR, ARIMA, SARIMA/SARIMAX, VAR, VECM, VARMAX starters plus ACF, PACF, Ljung-Box, ADF, and KPSS diagnostics, and forecast standard errors with confidence intervals for ARIMA/SARIMA/VAR | `statsmodels.tsa` basics, `get_forecast().conf_int()`, `VARResults.forecast_interval` |
| `seasonal` | classical seasonal decomposition (additive and multiplicative centered moving averages) and STL | `statsmodels.tsa.seasonal.seasonal_decompose`, `STL` |
| `smoothing` | simple exponential smoothing, Holt's linear and damped trend, and Holt-Winters additive/multiplicative seasonality with fitted values, SSE, and forecasts | `statsmodels.tsa.holtwinters` |
| `power` | power and sample-size solving for one/two-sample t-tests, two-sample z-tests, and one-way ANOVA, plus noncentral t/F/χ² CDFs | `statsmodels.stats.power` |
| `proportion` | one- and two-sample proportion z-tests, five confidence-interval methods (normal, Wilson, Clopper-Pearson, Agresti-Coull, Jeffreys), and Cohen's h | `statsmodels.stats.proportion` |
| `graphics` | dependency-light SVG line, scatter, residual, and ACF plots | `statsmodels.graphics` basics |
| `diagnostics` | VIF, Breusch-Pagan, White, RESET diagnostics | `statsmodels.stats.diagnostic`, `outliers_influence` basics |
| `evaluation` | regression/classification metrics, bootstrap mean intervals | common model-evaluation workflow |
| `robust` | Huber robust linear regression | `statsmodels.RLM` basics |
| `gee` | independence-working-correlation GEE starters | `statsmodels.GEE` basics |
| `mixed` | random-intercept mixed linear model starter | `statsmodels.MixedLM` basics |
| `correlation` | Pearson, Spearman, full correlation matrix | `df.corr()` |

---

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
inferust = "0.1"
```

---

## Quick start

### OLS Regression

```rust
use inferust::regression::Ols;

let x = vec![
    vec![2.0, 3.1],
    vec![5.0, 3.7],
    vec![8.0, 3.5],
    vec![11.0, 3.6],
];
let y = vec![55.0, 70.0, 80.0, 90.0];

let result = Ols::new()
    .with_feature_names(vec!["hours_studied".into(), "prior_gpa".into()])
    .fit(&x, &y)
    .unwrap();

result.print_summary();
```

Output:
```
═══════════════════════════════════════════════════════════════════
                     OLS Regression Results
═══════════════════════════════════════════════════════════════════
 Dep. variable: y          Observations  : 4
 R²           : 0.998102   Adj. R²       : 0.994305
 F-statistic  : 262.7732   F p-value     : 0.039405
 AIC          : 14.7316    BIC           : 12.0167
───────────────────────────────────────────────────────────────────
Variable               Coef       Std Err         t      P>|t|
───────────────────────────────────────────────────────────────────
const              -5.654762    5.033740    -1.1234   0.460565
hours_studied       4.130952    0.177951    23.2141   0.027430  *
prior_gpa           8.166667    1.490421     5.4793   0.115581
───────────────────────────────────────────────────────────────────
 Significance codes:  *** p<0.001  ** p<0.01  * p<0.05  . p<0.1
═══════════════════════════════════════════════════════════════════
```

The printed OLS/WLS summary also includes statsmodels-style residual diagnostics
out of the box: Durbin-Watson, Jarque-Bera with `Prob(JB)`, residual skewness,
kurtosis, and the design-matrix condition number.

### Formula-based fitting

```rust
use inferust::data::DataFrame;

let frame = DataFrame::new()
    .with_column("hours", vec![2.0, 5.0, 8.0, 11.0]).unwrap()
    .with_column("gpa", vec![3.1, 3.7, 3.5, 3.6]).unwrap()
    .with_column("score", vec![55.0, 70.0, 80.0, 90.0]).unwrap();

let result = frame.ols("score ~ hours + gpa").unwrap();
```

Formula support includes numeric `response ~ x1 + x2` terms, treatment dummy expansion for numeric-coded or string categorical columns with `C(group)`, interactions, offsets, and no-intercept formulas. Intercepts are handled by the model builders.

```rust
let frame = DataFrame::new()
    .with_column("score", vec![55.0, 70.0, 80.0, 90.0]).unwrap()
    .with_column("hours", vec![2.0, 5.0, 8.0, 11.0]).unwrap()
    .with_categorical_column("classroom", vec!["A", "B", "A", "C"]).unwrap();

let result = frame.ols(inferust::formula!(score ~ hours + C(classroom))).unwrap();
```

Formula transforms support `log(x)`, `sqrt(x)`, and `exp(x)`. Use
`frame.drop_missing()` to remove rows containing `NaN` in numeric columns before
building design matrices.

For Polars users, collect a Utf8/Categorical column into `Vec<String>` or
`Vec<&str>` and pass it to `with_categorical_column`; inferust keeps Polars
optional rather than forcing it as a dependency.

### Weighted least squares

```rust
use inferust::regression::Wls;

let weights = vec![1.0, 0.8, 1.2, 1.5];
let result = Wls::new()
    .with_feature_names(vec!["hours_studied".into(), "prior_gpa".into()])
    .fit(&x, &y, &weights)
    .unwrap();

result.print_summary();
```

### Quantile regression

```rust
use inferust::regression::QuantileRegression;

let result = QuantileRegression::new(0.5)
    .with_feature_names(vec!["hours_studied".into(), "prior_gpa".into()])
    .fit(&x, &y)
    .unwrap();

let intervals = result.confidence_intervals(0.05).unwrap();
result.print_summary();
```

Formula users can call `frame.quantile("score ~ hours + gpa", 0.5)`, matching
the common `statsmodels.formula.api.quantreg(...).fit(q=0.5)` workflow.

### GAM, IV, and state-space starters

```rust
use inferust::gam::{GaussianGam, SplineTerm};

let gam = GaussianGam::new()
    .smooth(SplineTerm::cubic(0, vec![2.0, 4.0]).named("s(hours)"))
    .fit(&x, &y)
    .unwrap();

let smooth_predictions = gam.predict(&x).unwrap();
```

```rust
use inferust::gmm::Iv2Sls;

let iv = Iv2Sls::new()
    .with_feature_names(vec!["price".into()])
    .fit(&x, &y, &instruments)
    .unwrap();
```

```rust
use inferust::statespace::LocalLevel;

let state = LocalLevel::new(0.25, 0.05).fit(&y).unwrap();
let next = state.forecast(3).unwrap();
```

### Discrete, multivariate, and treatment effects

```rust
use inferust::discrete::{OrderedLogit, ZeroInflatedPoisson};

let ordered = OrderedLogit::new().fit(&x, &ordinal_y).unwrap();
let category_probabilities = ordered.predict_proba(&x);

let zip = ZeroInflatedPoisson::new().fit(&x, &counts, &inflation_x).unwrap();
```

```rust
use inferust::multivariate::{one_way_manova, pca};

let manova = one_way_manova(&[group_a, group_b]).unwrap();
let pca_result = pca(&x).unwrap();
let scores = pca_result.transform(&x, 2).unwrap();
```

```rust
use inferust::treatment::{balance_diagnostics, PropensityScore};

let balance = balance_diagnostics(&x, &treatment).unwrap();
let effects = PropensityScore::new().ipw(&x, &treatment, &outcome).unwrap();
```

```rust
use inferust::contingency::{mcnemar, odds_ratio_ci, table2x2};
use inferust::imputation::MiceImputer;

let effects = table2x2([[12.0, 5.0], [4.0, 20.0]]).unwrap();
let interval = odds_ratio_ci([[12.0, 5.0], [4.0, 20.0]], 0.05).unwrap();
let paired = mcnemar([[20.0, 12.0], [3.0, 25.0]], true).unwrap();

let filled = MiceImputer::new().fit_transform(&possibly_missing).unwrap();
```

### GLS and rolling regression

```rust
use inferust::regression::{Fgls, RollingOls};

let fgls = Fgls::new()
    .with_feature_names(vec!["x".into()])
    .fit(&x, &y)
    .unwrap();

let rolling = RollingOls::new(12).fit(&x, &y).unwrap();
let slopes = rolling.slopes();
```

### Logistic regression

```rust
use inferust::glm::Logistic;

let result = Logistic::new()
    .with_feature_names(vec!["x1".into(), "x2".into()])
    .fit(&x, &binary_y)
    .unwrap();

let probabilities = result.predict_proba(&x);
let intervals = result.confidence_intervals(0.05).unwrap();
let odds_ratios = result.odds_ratios();
let marginal_effects = result.average_marginal_effects();
let marginal_effect_table = result.average_marginal_effects_summary(0.05).unwrap();
let residuals = result.residuals();
let metrics = result.classification_metrics(0.5).unwrap();
let lr_test = result.likelihood_ratio_test().unwrap();
```

You can also use `DataFrame::logistic("clicked ~ visits + age")` for formula-based fitting. Logistic results expose fitted probabilities, covariance estimates, response/Pearson/deviance residuals, likelihood-ratio tests, classification metrics, and post-estimation helpers designed to mirror common `statsmodels.Logit` workflows.

### Poisson regression

```rust
use inferust::glm::Poisson;

let result = Poisson::new()
    .with_feature_names(vec!["exposure".into(), "age".into()])
    .fit(&x, &counts)
    .unwrap();

let expected_counts = result.predict(&x);
let intervals = result.confidence_intervals(0.05).unwrap();
let mean_intervals = result.fitted_mean_intervals(0.05).unwrap();
let residuals = result.residuals();
let incidence_rate_ratios = result.incidence_rate_ratios();
let lr_test = result.likelihood_ratio_test().unwrap();
```

Poisson results include covariance estimates, fitted values, response/Pearson/deviance residuals, log-likelihood, null log-likelihood, pseudo-R², deviance, null deviance, Pearson chi-square, AIC, BIC, likelihood-ratio tests, and response-scale mean intervals. `DataFrame::poisson("count ~ exposure + age")` provides formula-based fitting.

### Gamma regression

```rust
use inferust::glm::{Gamma, GammaLink};

let result = Gamma::new()                       // canonical InversePower link
    .with_feature_names(vec!["claim_age".into()])
    .fit(&x, &positive_costs)
    .unwrap();

let log_link = Gamma::new()
    .with_link(GammaLink::Log)
    .fit(&x, &positive_costs)
    .unwrap();

let mean_intervals = result.fitted_mean_intervals(0.05).unwrap();
```

`Gamma` fits positive, right-skewed continuous outcomes (costs, durations, claim sizes) via IRLS, exposing the same covariance, residual, likelihood-ratio, and prediction-interval helpers as `Logistic`/`Poisson`. `GammaLink::InversePower` (default), `Log`, and `Identity` are supported.

### Regularized regression

```rust
use inferust::regression::{ElasticNet, Lasso, Ridge};

let ridge = Ridge::new(0.5).fit(&x, &y).unwrap();
let lasso = Lasso::new(0.1).fit(&x, &y).unwrap();
let elastic_net = ElasticNet::new(0.2, 0.5).fit(&x, &y).unwrap(); // l1_ratio = 0.5

elastic_net.print_summary();
```

Ridge is solved in closed form; Lasso and ElasticNet use cyclical coordinate descent with soft-thresholding. None of the three penalize the intercept (the scikit-learn/glmnet convention) -  see the `regression::regularized` module docs for the exact objective and how to reproduce it in `statsmodels.OLS().fit_regularized()`.

### Hypothesis tests

```rust
use inferust::hypothesis::{ttest, anova, chisq};

// Paired t-test
let before = vec![72.0, 68.0, 75.0, 80.0, 65.0];
let after  = vec![78.0, 74.0, 80.0, 85.0, 72.0];
ttest::paired(&before, &after).unwrap().print();

// Two-sample Welch t-test
ttest::two_sample(&group_a, &group_b).unwrap().print();

// One-way ANOVA
anova::one_way(&[&group1, &group2, &group3]).unwrap().print();

// Chi-squared goodness-of-fit
chisq::goodness_of_fit(&observed, None).unwrap().print();

// Chi-squared test of independence
chisq::independence(&contingency_table).unwrap().print();
```

### Tukey HSD and multiple-testing corrections

```rust
use inferust::hypothesis::{adjust, tukey_hsd, MultiTestMethod};

// Pairwise post-hoc comparisons after a one-way ANOVA.
let tukey = tukey_hsd(&[&group1, &group2, &group3], None, 0.05).unwrap();
tukey.print();

// Correct a family of p-values for multiple comparisons.
let p_values = vec![0.001, 0.02, 0.03, 0.04, 0.5];
let corrected = adjust(&p_values, 0.05, MultiTestMethod::BenjaminiHochberg).unwrap();
corrected.print();
```

`tukey_hsd` controls the family-wise error rate across every pairwise group comparison using the studentized range distribution (Tukey-Kramer adjusted for unequal group sizes). `adjust` supports `Bonferroni`, `Holm`, `BenjaminiHochberg`, and `BenjaminiYekutieli`.

### Descriptive statistics

```rust
use inferust::descriptive::Summary;

let data = vec![4.2, 7.8, 5.1, 9.3, 3.6, 8.4];
Summary::new(&data).unwrap().print();
// ─────────────────────────────
//  n          : 6
//  mean       : 6.400000
//  std        : 2.282176
//  min        : 3.600000
//  25%        : 4.575000
//  50%        : 6.150000
//  75%        : 8.250000
//  max        : 9.300000
//  skewness   : -0.058732
//  kurtosis   : -1.504070
// ─────────────────────────────
```

### Correlation

```rust
use inferust::correlation;

let r = correlation::pearson(&x, &y).unwrap();
let rs = correlation::spearman(&x, &y).unwrap();

let matrix = correlation::correlation_matrix(&[hours, gpa, scores]).unwrap();
correlation::print_correlation_matrix(&matrix, &["hours", "gpa", "scores"]);
```

### Time series and graphics

```rust
use inferust::graphics::{acf_plot_svg, PlotOptions};
use inferust::time_series::{acf, Sarima, Varmax};

let sarima = Sarima::new(1, 1, 1, 1, 1, 0, 12).fit(&series).unwrap();
let forecast = sarima.forecast(&series, 6).unwrap();

let acf_values = acf(&series, 24).unwrap();
let svg = acf_plot_svg(&acf_values, PlotOptions::default()).unwrap();
```

---

## OLS builder options

```rust
use inferust::regression::{Ols, OlsCovariance, OlsSolver};

let result = Ols::new()                         // intercept on by default
    .with_feature_names(vec!["x1".into()])        // label columns
    .with_solver(OlsSolver::Cholesky)             // default fast path
    .with_covariance(OlsCovariance::Hc1)          // robust standard errors
    .fit(&x, &y)
    .unwrap();

let intervals = result.confidence_intervals(0.05).unwrap();
let influence = result.influence();
let diagnostics = result.diagnostics().unwrap();
let cooks_distance = influence.cooks_distance;
let durbin_watson = diagnostics.durbin_watson;

Ols::new()
    .stable()                                    // SVD solver for tougher designs
    .robust()                                    // shorthand for HC1 covariance
    .fit(&x, &y)
    .unwrap();
```

`OlsResult` also exposes `.predict(&x)` for out-of-sample predictions and all raw fields (`coefficients`, `residuals`, `r_squared`, `p_values`, etc.) for programmatic use.

### Solver strategy

`inferust` defaults to a Cholesky solve of the normal system for full-rank, well-conditioned OLS problems. This avoids the extra work of forming a full inverse for coefficient estimation and is the fastest path for typical dense data.

For tougher or poorly conditioned designs, call `.stable()` or `.with_solver(OlsSolver::Svd)` to use the SVD path. For heteroskedasticity-consistent inference, use `.with_covariance(OlsCovariance::Hc0)`, `.Hc1`, `.Hc2`, `.Hc3`, or the `.robust()` HC1 shorthand. The test suite includes statsmodels-derived reference values for coefficients, classical and robust standard errors, t/z statistics, p-values, confidence intervals, leverage, internally studentized residuals, Cook's distance, DFFITS, Durbin-Watson, Jarque-Bera, residual skew/kurtosis, condition number, R², F-statistics, AIC, and BIC.

---

## Changelog

- **v0.1.17** -  IRLS performance pass (Cox PH, Probit, Gamma, ZIP EM), PACF methods,
  discrete/GEE/mixed/robust parity fixtures, full benchmark suite.

---

## Benchmarks

The repository includes reproducible benchmark scripts for comparing `inferust`
with Python `statsmodels` on deterministic synthetic data.

**OLS only** (configurable rows, features, solver):

```bash
cargo run --release --example bench_ols -- --rows 10000 --features 8 --repeats 10 --warmups 2
cargo run --release --example bench_ols -- --solver svd --rows 10000 --features 8 --repeats 10 --warmups 2
python scripts/bench_statsmodels.py --rows 10000 --features 8 --repeats 10 --warmups 2
```

**Full estimator suite** (~25 estimators, 10,000 rows):

```bash
cargo run --release --example bench_all
python3 scripts/bench_all_statsmodels.py   # requires numpy, scipy, statsmodels, lifelines, pandas
```

**CI smoke benchmark** (5,000 rows, OLS + Logistic + Poisson):

```bash
cargo run --release --example bench_smoke
```

See [bench/README.md](bench/README.md) for Docker-based reproducible runs.

On the current local benchmark machine (Apple Silicon, release build):

| Case | Median fit time |
|------|-----------------|
| OLS 10k × 8 features (Cholesky) | 0.769 ms |
| OLS 10k × 8 features (SVD) | 2.474 ms |
| statsmodels OLS (same data) | 2.492 ms |
| Smoke OLS 5k × 4 features | 0.568 ms |
| Smoke Logistic 5k × 4 | 1.903 ms |
| Smoke Poisson 5k × 4 | 2.436 ms |

Benchmark results vary by machine and BLAS/LAPACK configuration, so treat these as a local smoke test rather than a universal claim. The checksum printed by each script is useful for confirming both implementations fit equivalent data.

---

## Error handling

All fallible functions return `inferust::Result<T>` (an alias for `Result<T, InferustError>`):

```rust
use inferust::InferustError;

match result {
    Err(InferustError::SingularMatrix)           => { /* perfect multicollinearity */ }
    Err(InferustError::InsufficientData { .. })  => { /* too few rows */ }
    Err(InferustError::DimensionMismatch { .. }) => { /* X rows ≠ y length */ }
    Err(InferustError::InvalidInput(msg))        => { /* other input problem */ }
    Ok(r) => { /* use result */ }
}
```

---

## Dependencies

| Crate | Purpose |
|-------|---------|
| [`nalgebra`](https://crates.io/crates/nalgebra) | Matrix operations for OLS normal equations -  no LAPACK required |
| [`statrs`](https://crates.io/crates/statrs) | Student's t, F, and χ² distributions for p-values and confidence intervals |
| [`thiserror`](https://crates.io/crates/thiserror) | Ergonomic error types |

---

## Roadmap

### Planned

Ordered roughly by priority. The parity gaps are tracked in more detail under
"Future work" in [docs/parity.md](docs/parity.md).

- [ ] Parity fixtures for `multivariate` (PCA against `statsmodels.multivariate.pca.PCA`, MANOVA)
- [ ] Parity fixtures for `panel` fixed- and random-effects estimators
- [ ] Binomial and Gamma dispatch through the generic `glm_family` front-end, plus bse/llf/deviance
- [ ] Fuller output sets for the `gee`, `mixed`, and `robust` fixtures, which currently pin only headline coefficients
- [ ] Parity fixtures for `gam`, `gmm`, `imputation`, and `treatment`
- [ ] Numerical parity for `Sarimax`, `Vecm`, and `Varmax`

### Shipped

- [x] Logistic regression (GLM with logit link)
- [x] Gamma regression (GLM with InversePower/Log/Identity links)
- [x] Ridge / Lasso / ElasticNet regularization
- [x] Durbin-Watson and Breusch-Pagan diagnostic tests
- [x] Tukey HSD post-hoc test (after ANOVA)
- [x] Multiple-testing corrections (Bonferroni, Holm, Benjamini-Hochberg/Yekutieli)
- [x] Time-series: ARIMA / ACF / PACF
- [x] Weighted OLS
- [x] Two-way ANOVA with Type I / Type II sums of squares
- [x] Seasonal decomposition (classical and STL)
- [x] Exponential smoothing and Holt-Winters
- [x] Power analysis and sample-size solving
- [x] Proportion tests and confidence intervals
- [x] Forecast confidence intervals for ARIMA / SARIMA / VAR

Contributions welcome -  open an issue or PR!

---

## License

MIT -  see [LICENSE](LICENSE).
