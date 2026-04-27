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
| `regression::Ols` | OLS with fast Cholesky and stable SVD solvers, coefficients, std errors, t-stats, p-values, R², adj-R², F-stat, AIC, BIC | `statsmodels.OLS().fit()` |
| `hypothesis::ttest` | One-sample, two-sample Welch, paired t-tests with 95% CI | `scipy.stats.ttest_*` |
| `hypothesis::chisq` | Goodness-of-fit and independence (contingency table) | `scipy.stats.chisquare`, `chi2_contingency` |
| `hypothesis::anova` | One-way ANOVA table (SS, MS, F, p) | `scipy.stats.f_oneway` |
| `descriptive::Summary` | mean, std, variance, min/max, quartiles, skewness, excess kurtosis | `pd.Series.describe()` |
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

---

## OLS builder options

```rust
use inferust::regression::{Ols, OlsSolver};

Ols::new()                                        // intercept on by default
    .with_feature_names(vec!["x1".into()])        // label columns
    .with_solver(OlsSolver::Cholesky)             // default fast path
    .no_intercept()                               // force through origin
    .fit(&x, &y)
    .unwrap();

Ols::new()
    .stable()                                    // SVD solver for tougher designs
    .fit(&x, &y)
    .unwrap();
```

`OlsResult` also exposes `.predict(&x)` for out-of-sample predictions and all raw fields (`coefficients`, `residuals`, `r_squared`, `p_values`, etc.) for programmatic use.

### Solver strategy

`inferust` defaults to a Cholesky solve of the normal system for full-rank, well-conditioned OLS problems. This avoids the extra work of forming a full inverse for coefficient estimation and is the fastest path for typical dense data.

For tougher or poorly conditioned designs, call `.stable()` or `.with_solver(OlsSolver::Svd)` to use the SVD path. The test suite includes statsmodels-derived reference values for coefficients, standard errors, t/p-values, R², F-statistics, AIC, and BIC.

---

## Benchmarks

The repository includes reproducible OLS benchmark scripts for comparing `inferust` with Python `statsmodels` on deterministic synthetic data. Build and run the Rust benchmark in release mode:

```bash
cargo run --release --example bench_ols -- --rows 10000 --features 8 --repeats 10 --warmups 2
cargo run --release --example bench_ols -- --solver svd --rows 10000 --features 8 --repeats 10 --warmups 2
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

- [ ] Logistic regression (GLM with logit link)
- [ ] Ridge / Lasso regularization
- [ ] Durbin-Watson and Breusch-Pagan diagnostic tests
- [ ] Tukey HSD post-hoc test (after ANOVA)
- [ ] Time-series: ARIMA / ACF / PACF
- [ ] Weighted OLS

Contributions welcome — open an issue or PR!

---

## License

MIT — see [LICENSE](LICENSE).
