# Parity with statsmodels

This document defines inferust's parity contract with `statsmodels` and tracks
the per-module audit. Run `cargo test --tests parity_*` to execute every parity
check; any failure prints the per-field diff so the table below can be updated
mechanically from the output.

## How the harness works

1. `scripts/parity_statsmodels.py` generates synthetic datasets from a
   deterministic Park-Miller LCG and runs the corresponding `statsmodels`
   estimator on each. The estimator's outputs (params, bse, tvalues, pvalues,
   conf_int, R², log-likelihood, etc.) plus the *raw input dataset* are written
   to `tests/fixtures/statsmodels/<name>.json`.
2. The Rust integration tests under `tests/parity_*.rs` load the JSON, read the
   embedded dataset directly (no Rust-side RNG needed), run inferust's
   equivalent estimator, and compare every output at a per-field tolerance.

Datasets are embedded in the JSON so the Rust tests have no Python dependency
at runtime; only contributors regenerating the fixtures need `statsmodels` and
`scipy` installed.

## Regenerating fixtures

```bash
pip install statsmodels scipy numpy
python3 scripts/parity_statsmodels.py
```

This rewrites every JSON file under `tests/fixtures/statsmodels/`. After
regeneration, run `cargo test --tests parity_*` and resolve any new diffs.

## Tolerance policy

| Category | Default tolerance | Rationale |
|---|---|---|
| Closed-form linear algebra (OLS params, fitted, R², SSR, HC SE) | `1e-8` to `1e-10` | Same f64 arithmetic on both sides; tighter than this is just rounding noise. |
| Influence diagnostics (Cook's, studentized) | `1e-7` | Extra divisions by `(1 - h_i)`. |
| Iterative GLM / Cox (params, bse) | `1e-5` | Newton / IRLS convergence tolerance is `1e-8` to `1e-10`; final-iterate drift dominates. |
| GLM z-statistics, p-values | `1e-4` | Compounded from `1e-5` param drift. |
| ACF / Ljung-Box | `1e-8` to `1e-10` | Closed form. |
| PACF (Yule-Walker vs OLS-AR) | `5e-3` | Different methods — inferust uses OLS-AR, statsmodels' `method="ywm"` is biased differently. Tracked as a known gap, see below. |
| ADF t-statistic | `1e-7` | Both fit the same regression; should be tight. |
| ARIMA params | (no strict parity) | inferust uses CSS, statsmodels uses MLE/statespace. Tested for plausibility only. |
| Hypothesis tests (t, ANOVA, chi-square, MW) | `1e-9` to `1e-10` | Closed form. p-values rely on `statrs` vs `scipy` distribution implementations; small drift expected. |
| Multiple-testing corrections (Bonferroni, Holm, BH, BY) | `1e-10` to `1e-12` | Direct closed-form formulas; matches `statsmodels.stats.multitest.multipletests` bit-for-bit in practice. |
| Tukey HSD mean_diff, std_error | `1e-6` | Closed form (Tukey-Kramer SE). |
| Tukey HSD q_crit, p-value, CI bounds | `5e-3` | statsmodels' `pairwise_tukeyhsd` uses an interpolated lookup table (`libqsturng`, ~`1e-3` accurate); inferust computes the studentized range distribution via quadrature (~`1e-9` accurate). Known gap, see below. |
| Ridge (closed-form) params | `1e-6` to `1e-8` | Same Cholesky-solved normal equations on both sides once statsmodels is given the zero-intercept alpha vector (see Known gaps). |
| Lasso / ElasticNet (coordinate descent) params | `1e-5` | Same soft-thresholding algorithm; final-iterate drift between the two convergence tolerances dominates. |
| Gamma GLM (InversePower / Log links) params, bse | `1e-5` | Same IRLS/Fisher-scoring tier as other GLMs. |
| Gamma GLM deviance, pearson_chi2, scale, AIC, BIC | `1e-4` | Derived from converged fitted values; compounds the `1e-5` param drift. |

## Audit matrix

Run the tests to populate the **Status** column. The list below covers the
modules that have at least one parity fixture today; modules listed under
*Future work* have no parity tests yet and are the priority backlog.

| Module | Fixture(s) | Estimator | What's compared | Status |
|---|---|---|---|---|
| `regression` | `ols_small`, `ols_medium` | `Ols` (Nonrobust) | params, bse, t, p, CI, fitted, resid, R², adj R², F, F-p, AIC, BIC, SSR, ESS, TSS, MSE, hat, Cook's, studentized | pending first run |
| `regression` | `ols_hc0/1/2/3` | `Ols` w/ HC0–HC3 | params, bse, t, p | pending first run |
| `regression` | `wls_small` | `Wls` | params, bse, t, p, R², F, SSR | pending first run |
| `glm` | `logit_small` | `Logistic` | params, bse, z, p, llf, llnull, pseudo R², AIC, BIC | pending first run |
| `glm` | `poisson_small` | `Poisson` | params, bse, z, p, llf, llnull, AIC, BIC, fitted | pending first run |
| `time_series` | `acf_pacf` | `acf`, `pacf`, `ljung_box` | full vectors plus per-lag Q stat & p-value | pending first run |
| `time_series` | `adf` | `adf_test` | t-statistic | pending first run |
| `time_series` | `arima_ar1` | `Arima(1,0,0)` | implied mean & phi within 0.05 / 0.5 | structural-only |
| `hypothesis` | `ttest_1samp` | `ttest::one_sample` | statistic, p, df, mean_diff, CI | pending first run |
| `hypothesis` | `ttest_ind` | `ttest::two_sample` (Welch) | statistic, p, df | pending first run |
| `hypothesis` | `anova_oneway` | `anova::one_way` | F, p | pending first run |
| `hypothesis` | `mann_whitney` | `nonparametric::mann_whitney` | U (sided), p | pending first run |
| `hypothesis` | `chi2_independence` | `chisq::independence` | χ², p, df | pending first run |
| `correlation` | `pearson_spearman` | `pearson`, `spearman` | r | pending first run |
| `descriptive` | `descriptive` | `Summary` | mean, std, var, quartiles, skew, kurtosis | pending first run |
| `survival` | `cox_ph` | `CoxPh` (Breslow ties) | params, bse, z, p, llf | pending first run |
| `survival` | `kaplan_meier` | `KaplanMeier` | n_events, n_censored, survival at 3 checkpoints (tol 1e-6) | passing |
| `survival` | `log_rank` | `log_rank_test` | χ² statistic, p (tol 1e-4) | passing |
| `time_series` | `granger_causality` | `granger_causality` | F, p at lag 2 | pending first run |
| `time_series` | `engle_granger` | `engle_granger` | second-stage ADF t-stat | pending first run |
| `hypothesis` | `wilcoxon` | `wilcoxon_signed_rank` | W statistic, asymptotic p | pending first run (p loose) |
| `hypothesis` | `sign_test` | `sign_test` | counts + exact two-sided binomial p | pending first run |
| `hypothesis` | `anderson_darling` | `anderson_darling` | raw A² (matches scipy `anderson`) | pending first run |
| `hypothesis` | `lilliefors` | `lilliefors` | D statistic only (different p-value approx) | pending first run |
| `hypothesis` | `ks_one_sample` | `nonparametric::ks_one_sample` | D statistic (1e-6), p-value (3e-2 — Marsaglia vs scipy expansion) | passing |
| `hypothesis` | `ks_two_sample` | `nonparametric::ks_two_sample` | D statistic (1e-6), p-value (3e-2 — Marsaglia vs scipy expansion) | passing |
| `hypothesis` | `kruskal_wallis_parity` | `nonparametric::kruskal_wallis` | H statistic (1e-6), p (1e-6) | passing |
| `hypothesis` | `shapiro_wilk` | `nonparametric::shapiro_wilk` | W (1e-2), directional p agreement (Royston vs AS R94) | passing |
| `hypothesis` | `chi2_goodness_of_fit` | `chisq::goodness_of_fit` | χ², p, df (1e-9) | passing |
| `contingency` | `mcnemar` | `mcnemar` | statistic, p (1e-6) | passing |
| `contingency` | `odds_ratio` | `table2x2`, `odds_ratio_ci` | odds_ratio (1e-9), CI bounds (loose — Wald vs Fisher exact) | passing |
| `diagnostics` | `vif` | `variance_inflation_factors` | VIF per predictor (1e-2 — intercept in aux regression gap) | passing |
| `diagnostics` | `breusch_pagan` | `breusch_pagan` | LM statistic, p (1e-4) | passing |
| `diagnostics` | `white_test` | `white_test` | LM statistic, p (1e-4) | passing |
| `diagnostics` | `reset_test` | `reset_test` | F statistic, p (1e-4) | passing |
| `hypothesis::wald` | `wald_ols` | `OlsResult::wald_test` | χ² & F statistics + both p-values | pending first run |
| `hypothesis::multicomp` | `multicomp` | `adjust` (Bonferroni, Holm, BH, BY) | p_corrected, reject, alpha_bonferroni, alpha_sidak, for all four methods | pending first run |
| `hypothesis::tukey` | `tukey_hsd` | `tukey_hsd` | mean_diff (sign-flipped vs. statsmodels' convention), std_error, q_crit, p-value, CI bounds, df_within | pending first run |
| `regression::regularized` | `ridge_small` | `Ridge` | params (incl. intercept) | pending first run |
| `regression::regularized` | `lasso_small` | `Lasso` | params (incl. intercept) | pending first run |
| `regression::regularized` | `elastic_net_small` | `ElasticNet` | params (incl. intercept) | pending first run |
| `regression` | `gls_ar1` | `Gls` (known AR(1) Ω) | params (bse/t/p excluded — sigma² formula known gap, see below) | passing |
| `regression` | `fgls_cochrane_orcutt` | `Fgls` (Cochrane-Orcutt / Prais-Winsten) | params, rho (tol 6e-2 — algorithm gap: inferust uses Prais-Winsten first-obs correction, statsmodels GLSAR uses pure C-O) | passing |
| `regression` | `quantreg_median`, `quantreg_q25` | `QuantileRegression` | params (tol 1e-4), pseudo_r1 (tol 1e-4) | passing |
| `regression` | `rolling_ols` | `RollingOls` | params matrix (tol 1e-8), R² vector (tol 1e-8) | passing |
| `regression` | `recursive_ols` | `RecursiveOls` | params at indices 10/20/30 (tol 1e-2 — Kalman vs OLS-init convention gap), cusum finiteness | passing |
| `glm` | `gamma_glm` | `Gamma` (InversePower & Log links) | params, bse, llf, llnull, deviance, pearson_chi2, scale, AIC, BIC, fitted mean CI | pending first run |

## Known gaps

These differences are documented intentionally rather than treated as bugs:

- **ARIMA(p,d,q) for q > 0** — statsmodels uses MLE on the statespace
  representation; inferust uses conditional-sum-of-squares with a gradient
  optimizer for q > 0 and OLS-AR for q == 0. The two estimators are
  asymptotically equivalent but diverge on small samples and on highly
  near-non-stationary series. *Fix:* implement a Kalman-filter exact-likelihood
  estimator (the `statespace` module already has the scalar case).
- **PACF** — inferust's `pacf` returns the last coefficient of OLS-AR(k) for
  each `k`, equivalent to statsmodels' `method="ols"`. The default in
  statsmodels (`method="ywm"`, Yule-Walker with bias correction) is reported
  in the fixture. We tolerate `5e-3` and a tighter parity will require either
  switching the default method or exposing both.
- **Mann-Whitney U sign convention** — inferust returns `min(U1, U2)`; scipy
  returns `U1` by default. The two-sided p-value is identical; only the U
  reported differs. The test accepts both sides.
- **OLS condition number** — inferust uses `kappa(R)` from a QR-style factor;
  statsmodels uses a singular-value ratio. They agree on well-conditioned
  matrices and drift on near-singular ones; not currently compared.
- **Tukey HSD q_crit / p-value / CI precision** — statsmodels' `pairwise_tukeyhsd`
  looks up the studentized range distribution in an interpolated table
  (`libqsturng`, ~`1e-3` accurate); inferust computes it directly via nested
  Gauss-Legendre quadrature (~`1e-9` accurate against the true distribution).
  Don't expect tighter-than-`5e-3` parity on these three fields specifically.
  See the doc comment on `hypothesis::tukey` for the full derivation.
- **Tukey HSD mean_diff sign convention** — statsmodels reports
  `meandiff = mean(later group) - mean(earlier group)` for each pair; inferust
  reports `mean_diff = mean(group_a) - mean(group_b)` where `group_a` is the
  earlier group, i.e. the opposite sign. The parity test negates inferust's
  value (and flips/swaps the CI bounds) before comparing; this is a labeling
  convention, not a numerical discrepancy.
- **Ridge / Lasso / ElasticNet intercept penalty** — statsmodels'
  `OLS.fit_regularized(alpha=<scalar>)` penalizes every column including any
  constant; inferust never penalizes the intercept (the scikit-learn/glmnet
  convention). The fixtures pass statsmodels a per-column alpha *vector* with
  `0` in the intercept's slot to reproduce inferust's objective exactly — see
  `src/regression/regularized.rs` module docs. Verified offline to agree with
  inferust's coordinate-descent / closed-form solver to ~`1e-13` once that
  adjustment is made, so the parity tolerances above are tight.

- **GLS sigma² / bse** — `Gls::fit` computes the transformed SSR as
  `resid' * Ω⁻¹ * y − β' * X'Ω⁻¹y` instead of the correct
  `y' * Ω⁻¹ * y − β' * X'Ω⁻¹y`.  The coefficient estimates are unaffected
  (the GLS normal equations depend only on `X'Ω⁻¹X` and `X'Ω⁻¹y`), but
  `bse`, `t_statistics` and `p_values` are off by ~2.5× compared to statsmodels.
  The `gls_ar1` parity test therefore only checks `params`.

## Future work (backlog)

Modules with no parity fixtures today. Priority is **bold** for high-traffic
estimators.

- **`glm_family`** — generic GLM front-end; should be matched against
  `statsmodels.GLM` for Gaussian/Binomial/Poisson/Gamma families. (`glm::Gamma`
  itself has a direct parity fixture — `gamma_glm` — this entry is just about
  auditing the generic dispatch wrapper.)
- **`discrete`** — Probit, ordered logit, negative binomial, multinomial logit, zero-inflated Poisson; each maps to a `statsmodels.discrete` class.
- **`time_series::Var` / `Sarima` / `Sarimax` / `Vecm` / `Varmax`** — large surface, lowest-priority numerical parity because of multiple optimiser choices.
- **`multivariate`** — MANOVA, PCA; PCA in particular against `statsmodels.multivariate.pca.PCA`.
- **`gam`, `gee`, `gmm`, `mixed`, `robust`, `imputation`, `treatment`** — lower priority; each needs a dedicated fixture.

## Process for adding a new estimator to the audit

1. Add a `run_<name>` function to `scripts/parity_statsmodels.py` that builds
   the dataset via the LCG and invokes statsmodels.
2. Register the fixture in `main()`.
3. Run the harness; commit the new JSON file.
4. Add a `tests/parity_<module>.rs` test (or extend an existing one) that loads
   the fixture and asserts at the tolerance set in this doc's policy table.
5. Update the audit matrix above and remove the entry from the backlog.
