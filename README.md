# inferust

[![Crates.io](https://img.shields.io/crates/v/inferust.svg)](https://crates.io/crates/inferust)
[![Docs.rs](https://docs.rs/inferust/badge.svg)](https://docs.rs/inferust)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/rogerneal/inferust/ci.yml?branch=master)](https://github.com/rogerneal/inferust/actions)

**Statistical modeling for Rust** — a `statsmodels`-inspired library.

`inferust` fills the gap between Python's `statsmodels` / `scipy.stats` and the Rust ecosystem. It gives you regression summaries, hypothesis tests, descriptive stats, and correlation matrices with the same depth of output you'd expect from Python — p-values, confidence intervals, AIC/BIC, significance stars, and all.

---

## Features

| Module | What you get | Python equivalent |
|--------|-------------|-------------------|
| `regression::Ols` / `Wls` / `Gls` / `Fgls` | OLS, weighted least squares, GLS with known covariance, and AR(1) feasible GLS with fast/stable solvers, robust/HAC SEs, confidence intervals, influence diagnostics, residual diagnostics, Durbin-Watson, Jarque-Bera, condition numbers, t/z stats, p-values, R², adj-R², F-stat, AIC, BIC | `statsmodels.OLS().fit()`, `statsmodels.WLS().fit()`, `statsmodels.GLS().fit()`, `statsmodels.GLSAR()` |
| `regression::RollingOls` / `RecursiveOls` | Rolling-window coefficient paths and recursive OLS with CUSUM stability diagnostics | `statsmodels.regression.rolling.RollingOLS`, `statsmodels.regression.recursive_ls.RecursiveLS` basics |
| `hypothesis::ttest` | One-sample, two-sample Welch, paired t-tests with 95% CI | `scipy.stats.ttest_*` |
| `hypothesis::chisq` | Goodness-of-fit and independence (contingency table) | `scipy.stats.chisquare`, `chi2_contingency` |
| `hypothesis::anova` | One-way ANOVA table (SS, MS, F, p) | `scipy.stats.f_oneway` |
| `descriptive::Summary` | mean, std, variance, min/max, quartiles, skewness, excess kurtosis | `pd.Series.describe()` |
| `data::DataFrame` | named numeric columns and formula-based OLS/WLS/logistic/Poisson fitting | `statsmodels.formula.api` basics |
| `glm::Logistic` / `Poisson` | binary logistic and Poisson count regression with MLE estimates, Wald inference, covariance, residual diagnostics, likelihood-ratio tests, prediction intervals, classification metrics, and post-estimation helpers | `statsmodels.Logit().fit()`, `statsmodels.GLM(..., Poisson()).fit()` |
| `discrete` | Probit, negative binomial, multinomial logit starters | `statsmodels.discrete` basics |
| `glm_family` | generic Gaussian/Binomial/Poisson GLM dispatch | `statsmodels.GLM` basics |
| `time_series` | AR, ARIMA, SARIMA/SARIMAX, VAR, VECM, VARMAX starters plus ACF, PACF, Ljung-Box, ADF, and KPSS diagnostics | `statsmodels.tsa` basics |
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

Formula support includes numeric `response ~ x1 + x2` terms, plus treatment dummy expansion for numeric-coded categorical columns via `design_matrices_with_categorical` and `ols_with_categorical`. Intercepts are handled by the model builders.

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

Release history is tracked in [CHANGELOG.md](CHANGELOG.md), with an `Unreleased` section reserved for the next version before publication.

---

## Benchmarks

The repository includes reproducible OLS benchmark scripts for comparing `inferust` with Python `statsmodels` on deterministic synthetic data. Build and run the Rust benchmark in release mode:

```bash
cargo run --release --example bench_ols -- --rows 10000 --features 8 --repeats 10 --warmups 2
cargo run --release --example bench_ols -- --solver svd --rows 10000 --features 8 --repeats 10 --warmups 2
```

Additional examples:

```bash
cargo run --example diagnostics
cargo run --example discrete_models
```

Run the Python comparison after installing `numpy`, `scipy`, and `statsmodels`:

```bash
python scripts/bench_statsmodels.py --rows 10000 --features 8 --repeats 10 --warmups 2
```

On the current local benchmark machine, the 10,000 row × 8 feature case measured approximately:

| Engine | Solver | Median fit time |
|--------|--------|-----------------|
| `inferust` | Cholesky | 0.769 ms |
| `inferust` | SVD | 2.474 ms |
| `statsmodels` | default OLS | 2.492 ms |

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
| [`nalgebra`](https://crates.io/crates/nalgebra) | Matrix operations for OLS normal equations — no LAPACK required |
| [`statrs`](https://crates.io/crates/statrs) | Student's t, F, and χ² distributions for p-values and confidence intervals |
| [`thiserror`](https://crates.io/crates/thiserror) | Ergonomic error types |

---

## Roadmap

- [x] Logistic regression (GLM with logit link)
- [ ] Ridge / Lasso regularization
- [x] Durbin-Watson and Breusch-Pagan diagnostic tests
- [ ] Tukey HSD post-hoc test (after ANOVA)
- [x] Time-series: ARIMA / ACF / PACF
- [x] Weighted OLS

Contributions welcome — open an issue or PR!

---

## License

MIT — see [LICENSE](LICENSE).
