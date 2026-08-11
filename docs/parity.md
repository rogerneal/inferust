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
pip install statsmodels scipy numpy pandas linearmodels
python3 scripts/parity_statsmodels.py
```

This rewrites every JSON file under `tests/fixtures/statsmodels/`. After
regeneration, run `cargo test --tests parity_*` and resolve any new diffs.
`linearmodels` is required for the `panel_*` fixtures.

`scripts/stl_reference.py` is a companion development oracle: a direct
transcription of Cleveland's Fortran STL that agrees with
`statsmodels.tsa.seasonal.STL` to `~1e-13` when given matching iteration counts.
It is useful for localizing an STL discrepancy to a specific inner step, which
is hard to do against the compiled Cython extension. It is not used by the test
suite.

## Tolerance policy

| Category | Default tolerance | Rationale |
|---|---|---|
| Closed-form linear algebra (OLS params, fitted, R², SSR, HC SE) | `1e-8` to `1e-10` | Same f64 arithmetic on both sides; tighter than this is just rounding noise. |
| Influence diagnostics (Cook's, studentized) | `1e-7` | Extra divisions by `(1 - h_i)`. |
| Iterative GLM / Cox (params, bse) | `1e-5` | Newton / IRLS convergence tolerance is `1e-8` to `1e-10`; final-iterate drift dominates. |
| GLM z-statistics, p-values | `1e-4` | Compounded from `1e-5` param drift. |
| ACF / Ljung-Box | `1e-8` to `1e-10` | Closed form. |
| PACF (Yule-Walker vs OLS-AR) | `1e-10` | Default `pacf()` uses YWM/Durbin-Levinson; OLS-AR available via `PacfMethod::Ols`. |
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
| InverseGaussian GLM (Log link) params, bse, llf | `1e-5` | Same IRLS/Fisher-scoring tier as other GLMs. |
| InverseGaussian GLM deviance, scale | `1e-4` | Derived from converged fitted values; compounds the `1e-5` param drift. |
| Two-way ANOVA df, SS, F, p | `1e-9` to `1e-10` | Closed-form projections onto the same nested designs. |
| Power (t-test, z-test, ANOVA F) | `1e-9` to `1e-12` | Closed-form critical values with noncentral CDFs by quadrature and Poisson mixture. Requires the polished quantiles described under Known gaps. |
| Proportion z-tests, normal/Wilson/Agresti-Coull intervals | `1e-12` | Closed form. |
| Proportion Clopper-Pearson / Jeffreys intervals | `1e-10` | Exact Beta quantiles, polished (see Known gaps). |
| Classical `seasonal_decompose` (trend, seasonal, resid) | `1e-10` | Identical centered moving-average filters. |
| STL (trend, seasonal, resid) | `1e-11`, `1e-9` robust | Independent loess reimplementation; only accumulated rounding differs. The robust fit runs 15 outer reweighting passes, hence the looser tier. |
| Exponential smoothing / Holt-Winters fitted, SSE, forecasts | `1e-10` | Identical recursions, with one documented horizon exception (see Known gaps). |
| ARIMA / VAR forecast standard errors and intervals | `1e-9` to `1e-10` | Analytic ψ-weights vs statsmodels' Kalman filter; the seasonal-difference state converges to about `1e-10`. |
| PCA mean, loadings, variance, scores | `1e-10` to `1e-9` | Same covariance eigendecomposition; loadings/scores compared after sign alignment. |
| One-way MANOVA Wilks' λ, Rao F, df, p | `1e-10` to `1e-8` | Identical SSCP construction; F uses Rao's approximation. |
| Panel entity FE params / within bse | `1e-10` | Within transform + OLS; params also match linearmodels entity FE. |
| Panel RE params, bse, σ²_e/σ²_u, θ | `1e-10` to `1e-8` | Swamy–Arora GLS matching linearmodels RandomEffects. |
| Hausman FE vs RE | `1e-8` | Uses within-OLS FE cov (inferust), not linearmodels FE df correction. |
| GAM truncated-power + OLS params / bse | `1e-8` | Same design expansion and OLS as the fixture. |
| IV2SLS params / bse | `1e-8` | Closed-form two-stage projection (same formula both sides). |
| MICE mean-impute / chained OLS | `1e-12` / `1e-8` | Deterministic mean fill; OLS iterate drift is tiny on the 8×3 fixture. |
| IPW propensity params, ATE/ATT | `1e-5` | Logistic IRLS + clamped IPW formulas matching `treatment.rs`. |
| SARIMAX exogenous OLS coefficients | `1e-8` | Only the `[1, X]` projection is pinned (not SARIMA MLE). |
| VECM Johansen eigenvalues / trace | `1e-6` / `1e-4` | Pins inferust's symmetrised EVP; statsmodels `coint_johansen` may differ ~`1e-2`. |
| VARMAX per-equation OLS coefficients | `1e-8` | VAR+exog OLS layout, not statespace VARMAX. |

## Audit matrix

Run the tests to populate the **Status** column. The list below covers the
modules that have at least one parity fixture today; modules listed under
*Future work* have no parity tests yet and are the priority backlog.

| Module | Fixture(s) | Estimator | What's compared | Status |
|---|---|---|---|---|
| `regression` | `ols_small`, `ols_medium` | `Ols` (Nonrobust) | params, bse, t, p, CI, fitted, resid, R², adj R², F, F-p, AIC, BIC, SSR, ESS, TSS, MSE, hat, Cook's, studentized | passing |
| `regression` | `ols_hc0/1/2/3` | `Ols` w/ HC0–HC3 | params, bse, t, p | passing |
| `regression` | `wls_small` | `Wls` | params, bse, t, p, R², F, SSR | passing |
| `glm` | `logit_small` | `Logistic` | params, bse, z, p, llf, llnull, pseudo R², AIC, BIC | passing |
| `glm` | `poisson_small` | `Poisson` | params, bse, z, p, llf, llnull, AIC, BIC, fitted | passing |
| `time_series` | `acf_pacf` | `acf`, `pacf`, `ljung_box` | full vectors plus per-lag Q stat & p-value | passing |
| `time_series` | `adf` | `adf_test` | t-statistic | passing |
| `time_series` | `arima_ar1` | `Arima(1,0,0)` | implied mean & phi within 0.05 / 0.5 | structural-only |
| `hypothesis` | `ttest_1samp` | `ttest::one_sample` | statistic, p, df, mean_diff, CI | passing |
| `hypothesis` | `ttest_ind` | `ttest::two_sample` (Welch) | statistic, p, df | passing |
| `hypothesis` | `anova_oneway` | `anova::one_way` | F, p | passing |
| `hypothesis` | `mann_whitney` | `nonparametric::mann_whitney` | U (sided), p | passing |
| `hypothesis` | `chi2_independence` | `chisq::independence` | χ², p, df | passing |
| `correlation` | `pearson_spearman` | `pearson`, `spearman` | r | passing |
| `descriptive` | `descriptive` | `Summary` | mean, std, var, quartiles, skew, kurtosis | passing |
| `survival` | `cox_ph` | `CoxPh` (Breslow ties) | params, bse, z, p, llf | passing |
| `survival` | `kaplan_meier` | `KaplanMeier` | n_events, n_censored, survival at 3 checkpoints (tol 1e-6) | passing |
| `survival` | `log_rank` | `log_rank_test` | χ² statistic, p (tol 1e-4) | passing |
| `time_series` | `granger_causality` | `granger_causality` | F, p at lag 2 | passing |
| `time_series` | `engle_granger` | `engle_granger` | second-stage ADF t-stat | passing |
| `hypothesis` | `wilcoxon` | `wilcoxon_signed_rank` | W statistic, asymptotic p | passing (p loose) |
| `hypothesis` | `sign_test` | `sign_test` | counts + exact two-sided binomial p | passing |
| `hypothesis` | `anderson_darling` | `anderson_darling` | raw A² (matches scipy `anderson`) | passing |
| `hypothesis` | `lilliefors` | `lilliefors` | D statistic only (different p-value approx) | passing |
| `hypothesis` | `ks_one_sample` | `nonparametric::ks_one_sample` | D statistic (1e-6), p-value (3e-2 -  Marsaglia vs scipy expansion) | passing |
| `hypothesis` | `ks_two_sample` | `nonparametric::ks_two_sample` | D statistic (1e-6), p-value (3e-2 -  Marsaglia vs scipy expansion) | passing |
| `hypothesis` | `kruskal_wallis_parity` | `nonparametric::kruskal_wallis` | H statistic (1e-6), p (1e-6) | passing |
| `hypothesis` | `shapiro_wilk` | `nonparametric::shapiro_wilk` | W (1e-2), directional p agreement (Royston vs AS R94) | passing |
| `hypothesis` | `chi2_goodness_of_fit` | `chisq::goodness_of_fit` | χ², p, df (1e-9) | passing |
| `contingency` | `mcnemar` | `mcnemar` | statistic, p (1e-6) | passing |
| `contingency` | `odds_ratio` | `table2x2`, `odds_ratio_ci` | odds_ratio (1e-9), CI bounds (loose -  Wald vs Fisher exact) | passing |
| `diagnostics` | `vif` | `variance_inflation_factors` | VIF per predictor (1e-2 -  intercept in aux regression gap) | passing |
| `diagnostics` | `breusch_pagan` | `breusch_pagan` | LM statistic, p (1e-4) | passing |
| `diagnostics` | `white_test` | `white_test` | LM statistic, p (1e-4) | passing |
| `diagnostics` | `reset_test` | `reset_test` | F statistic, p (1e-4) | passing |
| `hypothesis::wald` | `wald_ols` | `OlsResult::wald_test` | χ² & F statistics + both p-values | passing |
| `hypothesis::multicomp` | `multicomp` | `adjust` (Bonferroni, Holm, BH, BY) | p_corrected, reject, alpha_bonferroni, alpha_sidak, for all four methods | passing |
| `hypothesis::tukey` | `tukey_hsd` | `tukey_hsd` | mean_diff (sign-flipped vs. statsmodels' convention), std_error, q_crit, p-value, CI bounds, df_within | passing |
| `regression::regularized` | `ridge_small` | `Ridge` | params (incl. intercept) | passing |
| `regression::regularized` | `lasso_small` | `Lasso` | params (incl. intercept) | passing |
| `regression::regularized` | `elastic_net_small` | `ElasticNet` | params (incl. intercept) | passing |
| `regression` | `gls_ar1` | `Gls` (known AR(1) Ω) | params, bse, t, p | passing |
| `regression` | `fgls_cochrane_orcutt` | `Fgls` (Cochrane-Orcutt / Prais-Winsten) | params, rho (tol 6e-2 -  algorithm gap: inferust uses Prais-Winsten first-obs correction, statsmodels GLSAR uses pure C-O) | passing |
| `regression` | `quantreg_median`, `quantreg_q25` | `QuantileRegression` | params (tol 1e-4), pseudo_r1 (tol 1e-4) | passing |
| `regression` | `rolling_ols` | `RollingOls` | params matrix (tol 1e-8), R² vector (tol 1e-8) | passing |
| `regression` | `recursive_ols` | `RecursiveOls` | params at indices 10/20/30 (tol 1e-2 -  Kalman vs OLS-init convention gap), cusum finiteness | passing |
| `glm` | `gamma_glm` | `Gamma` (InversePower & Log links) | params, bse, llf, llnull, deviance, pearson_chi2, scale, AIC, BIC, fitted mean CI | passing |
| `glm` | `inverse_gaussian_glm` | `InverseGaussian` (Log link) | params, bse, llf, deviance, scale | passing |
| `hypothesis::anova` | `anova_twoway` | `two_way` (Type I and Type II) | df, sum_sq, F, p per effect plus residual df/SS | passing |
| `power` | `power` | `TTestPower`, `TTestIndPower`, `NormalIndPower`, `FTestAnovaPower`, `solve_nobs` | power across alternatives and ratios, solved `nobs1` | passing |
| `proportion` | `proportion` | `proportions_ztest`, `proportion_confint`, `proportion_effectsize` | one/two-sample z and p, five interval methods, Cohen's h | passing |
| `seasonal` | `seasonal_decompose` | `seasonal_decompose` (additive & multiplicative) | trend, seasonal, resid including NaN edges | passing |
| `seasonal` | `stl` | `Stl` (default & robust) | trend, seasonal, resid, plus the component-sum identity | passing |
| `smoothing` | `holt_winters` | `SimpleExpSmoothing`, `ExponentialSmoothing` (add trend + add seasonal) | fitted, SSE, forecasts except `h % period == 0` | passing |
| `time_series` | `forecast_ci` | `sarima_forecast_standard_errors`, `VarResult::forecast_with_ci` | ARIMA & SARIMA `se_mean`, VAR point/lower/upper | passing |
| `discrete` | `probit_small` | `Probit` | params, bse, z, p, llf | passing |
| `discrete` | `negbin_small` | `NegativeBinomial` (NB2) | params, bse, alpha, llf | passing |
| `discrete` | `multinomial_small` | `MultinomialLogit` | params per non-base outcome, llf | passing |
| `discrete` | `ordered_logit_small` | `OrderedLogit` | params, thresholds, llf | passing |
| `discrete` | `zip_small` | `ZeroInflatedPoisson` | count & inflation params, llf | passing |
| `glm_family` | `ols_small`, `poisson_small`, `logit_small`, `gamma_glm`, `inverse_gaussian_glm` | `Glm` dispatch (Gaussian, Poisson, Binomial, Gamma, InverseGaussian) | params/bse (+ llf/deviance/scale where present) via the generic front-end | passing |
| `gee` | `gee_poisson`, `gee_small` | `Gee` (Poisson, exchangeable) | params, bse, z, p, rho (poisson) | passing |
| `mixed` | `mixed_small` | `MixedLm` (random intercept) | fixed effects, bse, REML llf, var_random, var_residual | passing |
| `robust` | `robust_small` | `Rlm` (Huber) | params, sandwich bse/t/p | passing |
| `multivariate` | `pca` | `pca` | mean, loadings (sign-aligned), explained variance / ratio, scores | passing |
| `multivariate` | `manova` | `one_way_manova` | Wilks' λ, Rao F, df, p | passing |
| `panel` | `panel_fe` | `PanelOls::fit_entity_fe` | params (vs linearmodels), within bse/t/p/R² (vs demean+OLS) | passing |
| `panel` | `panel_time_fe` | `PanelOls::fit_time_fe` | params (vs linearmodels), within bse/t/p/R² (vs demean+OLS) | passing |
| `panel` | `panel_two_way_fe` | `PanelOls::fit_two_way_fe` | params (vs linearmodels), iterative-within bse/t/p/R² | passing |
| `panel` | `panel_re` | `PanelOls::fit_random_effects`, `hausman_fe_re` | RE params/bse/σ²/θ (vs linearmodels), Hausman χ² (vs within-OLS cov) | passing |
| `gam` | `gam_small` | `GaussianGam` (cubic truncated-power, knot 2.0) | params, bse, R² | passing |
| `gmm` | `iv2sls_small` | `Iv2Sls` | params, bse, SSR, R² | passing |
| `imputation` | `imputation_mice_small` | `MiceImputer` | column_means, mean_impute data, fit_transform (iters=3) | passing |
| `treatment` | `treatment_ipw_small` | `PropensityScore::ipw` | propensity params/scores, ATE, ATT | passing |
| `time_series` | `sarimax_small` | `Sarimax` | exog_coefficients (OLS projection only) | passing |
| `time_series` | `vecm_small` | `Vecm` (Johansen) | eigenvalues, trace_statistics | passing |
| `time_series` | `varmax_small` | `Varmax` | per-equation OLS coefficients | passing |

## Known gaps

These differences are documented intentionally rather than treated as bugs:

- **ARIMA(p,d,q) for q > 0** -  statsmodels uses MLE on the statespace
  representation; inferust uses conditional-sum-of-squares with a gradient
  optimizer for q > 0 and OLS-AR for q == 0. The two estimators are
  asymptotically equivalent but diverge on small samples and on highly
  near-non-stationary series. *Fix:* implement a Kalman-filter exact-likelihood
  estimator (the `statespace` module already has the scalar case).
- **PACF** -  `pacf()` defaults to Yule-Walker (`method="ywm"`). Use `pacf_with_method(..., PacfMethod::Ols)` for the legacy OLS-AR partial coefficients.
- **Mann-Whitney U sign convention** -  inferust returns `min(U1, U2)`; scipy
  returns `U1` by default. The two-sided p-value is identical; only the U
  reported differs. The test accepts both sides.
- **OLS condition number** -  inferust uses `kappa(R)` from a QR-style factor;
  statsmodels uses a singular-value ratio. They agree on well-conditioned
  matrices and drift on near-singular ones; not currently compared.
- **Tukey HSD q_crit / p-value / CI precision** -  statsmodels' `pairwise_tukeyhsd`
  looks up the studentized range distribution in an interpolated table
  (`libqsturng`, ~`1e-3` accurate); inferust computes it directly via nested
  Gauss-Legendre quadrature (~`1e-9` accurate against the true distribution).
  Don't expect tighter-than-`5e-3` parity on these three fields specifically.
  See the doc comment on `hypothesis::tukey` for the full derivation.
- **Tukey HSD mean_diff sign convention** -  statsmodels reports
  `meandiff = mean(later group) - mean(earlier group)` for each pair; inferust
  reports `mean_diff = mean(group_a) - mean(group_b)` where `group_a` is the
  earlier group, i.e. the opposite sign. The parity test negates inferust's
  value (and flips/swaps the CI bounds) before comparing; this is a labeling
  convention, not a numerical discrepancy.
- **Ridge / Lasso / ElasticNet intercept penalty** -  statsmodels'
  `OLS.fit_regularized(alpha=<scalar>)` penalizes every column including any
  constant; inferust never penalizes the intercept (the scikit-learn/glmnet
  convention). The fixtures pass statsmodels a per-column alpha *vector* with
  `0` in the intercept's slot to reproduce inferust's objective exactly -  see
  `src/regression/regularized.rs` module docs. Verified offline to agree with
  inferust's coordinate-descent / closed-form solver to ~`1e-13` once that
  adjustment is made, so the parity tolerances above are tight.
- **Holt-Winters forecast at the end of a seasonal cycle** -  when
  `h % period == 0`, statsmodels reuses the previous cycle's seasonal state for
  that phase instead of the one updated by the final observation. inferust
  continues the recursion, which is the standard Holt-Winters definition.

  This is a defect in statsmodels rather than a modelling choice. In
  `statsmodels/tsa/holtwinters/model.py`, `_predict` runs the seasonal update
  `s[i + m - 1] = ...` for `i` in `1..=nobs`, so `s[nobs + m - 1]` holds the
  state implied by the last observation. The next statement then extends the
  array cyclically starting one slot too early:

  ```python
  s[nobs + m - 1 :] = [s[(nobs - 1) + j % m] for j in range(h + 1 + 1)]
  ```

  Because the slice starts at `nobs + m - 1` rather than `nobs + m`, the state
  just computed from the final observation is overwritten with `s[nobs - 1]`,
  which was written one full cycle earlier. Forecast step `h` reads
  `s[nobs + h - 1]`, so every horizon except `h % period == 0` is unaffected.

  Confirmed independently with a `gamma = 1` frozen-level series, where the
  seasonal state is exactly `y_t - level`: for `n = 24`, `period = 4`,
  statsmodels' forecasts imply seasonal states `s20, s21, s22, s19` across
  `h = 1..4`, where the correct continuation is `s20, s21, s22, s23`. The parity
  test skips those horizons and
  `holt_winters_cycle_end_uses_latest_seasonal_state` pins the intended
  behaviour. Every other horizon matches at `1e-10`.
- **statrs coarse quantiles (resolved in 0.19)** -  under statrs 0.17,
  `FisherSnedecor::inverse_cdf`, `ChiSquared::inverse_cdf`, and
  `Beta::inverse_cdf` terminated a bisection at roughly `1e-5` absolute (the
  returned values were dyadic rationals), while the corresponding `cdf`/`pdf`
  were accurate to near machine precision. Taking the quantiles at face value
  cost about `5e-6` in ANOVA power and `2.6e-5` in Clopper-Pearson bounds.
  statrs 0.19 fixes this: raw `inverse_cdf` now round-trips through `cdf` to
  `1e-16`. `power::refine_upper_quantile` and
  `proportion::refine_beta_quantile` are retained anyway, and now converge on
  the first Newton step. They pin the invariant `cdf(q) == p` to the accurate
  primitives, so precision no longer depends on the inverse-CDF implementation.
- **VECM vs statsmodels `coint_johansen`** -  inferust's Johansen path
  symmetrises the EVP and uses a regularised inverse; `coint_johansen` can
  differ by ~`1e-2` on eigenvalues / more on trace statistics for the same
  series. The `vecm_small` fixture pins inferust's algorithm (Python
  transcription) and stores the statsmodels values only as a side reference.
- **SARIMAX / VARMAX** -  only the closed-form OLS pieces are under parity
  (exogenous projection for SARIMAX; per-equation VAR+exog OLS for VARMAX).
  Full statespace MLE is intentionally out of scope.


## Future work (backlog)

Gaps in the audit: modules with no fixture at all, plus estimators whose fixture
pins only part of the result surface.

- **SARIMAX / VARMAX full MLE surface** -  exogenous OLS and VAR+exog OLS are
  covered; statespace likelihood parity remains future work.
- **VECM β / α / Γ** -  eigenvalues and trace stats are pinned; cointegrating
  vectors and short-run matrices are sign/scale-ambiguous and not yet compared.

## Process for adding a new estimator to the audit

1. Add a `run_<name>` function to `scripts/parity_statsmodels.py` that builds
   the dataset via the LCG and invokes statsmodels.
2. Register the fixture in `main()`.
3. Run the harness; commit the new JSON file.
4. Add a `tests/parity_<module>.rs` test (or extend an existing one) that loads
   the fixture and asserts at the tolerance set in this doc's policy table.
5. Update the audit matrix above and remove the entry from the backlog.
