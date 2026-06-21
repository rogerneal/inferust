# inferust vs. statsmodels — feature gap analysis

Reviewed against inferust `0.1.12` plus the `Unreleased` section of CHANGELOG.md, and statsmodels `0.14.6` (current stable API reference, June 2026).

## Where inferust already holds up well

inferust's regression core is genuinely competitive with statsmodels: OLS/WLS/GLS/FGLS all carry classical, HC0–HC3, and Newey-West HAC standard errors plus one-way cluster-robust covariance, full influence diagnostics (Cook's distance, DFFITS, leverage), and the same residual diagnostics statsmodels prints by default (Durbin-Watson, Jarque-Bera, condition number). Quantile regression, rolling/recursive OLS with CUSUM, and IV/2SLS round out the linear-model side. Logistic and Poisson GLMs have real Wald inference, likelihood-ratio tests, classification metrics, and marginal effects — not just point estimates. Time series is the other strong area: ARIMA/SARIMA/SARIMAX, VAR/VECM/VARMAX, ACF/PACF/Ljung-Box, ADF, KPSS, Granger causality, and Engle-Granger cointegration are all present. Survival analysis (Kaplan-Meier with Greenwood CIs, log-rank, Cox PH via Newton-Raphson) and the nonparametric hypothesis-test suite (Mann-Whitney, Kruskal-Wallis, Wilcoxon, sign test, KS, Shapiro-Wilk, Anderson-Darling, Lilliefors) are also solid. The project already runs its own statsmodels-parity harness (`scripts/parity_statsmodels.py` + `tests/parity_*.rs`, tracked in `docs/parity.md`), which is a better parity-verification setup than most Rust stats crates have at all.

That said, several modules listed in the README as shipped are thin relative to their statsmodels namesake. `mixed::MixedLinearModel` fits plain OLS and then averages group residuals into "random intercepts" — it has no REML/ML variance-component estimation and no random slopes, so it isn't really a mixed model. `gee::Gee` fits an ordinary independent GLM and only tracks cluster IDs for bookkeeping — there's no working-correlation matrix (exchangeable/AR/independence) or sandwich variance, so it isn't GEE in the statistical sense. `robust::RobustLinearModel` only supports Huber's M-estimator with classical (non-robust) standard errors, where statsmodels' `RLM` offers six norms (HuberT, Hampel, TukeyBiweight, AndrewWave, RamsayE, TrimmedMean) and a robust covariance matrix. These three are worth re-scoping before counting them as "done."

## Gap matrix

| Category | statsmodels | inferust today | Gap |
|---|---|---|---|
| Regularized regression | `OLS`/`GLM`/`Logit`.fit_regularized (L1, elastic net) | none | On inferust's own README roadmap, unchecked. No ridge/lasso/elastic-net path for any model. |
| GLM families & links | Gaussian, Binomial, Poisson, Gamma, InverseGaussian, NegativeBinomial, Tweedie × identity/log/logit/probit/cloglog/power links | `glm_family` dispatches Gaussian/Binomial/Poisson only, canonical links only | Blocks Gamma/Tweedie GLMs (cost, insurance, skewed-positive outcomes) and any non-canonical link. |
| Mixed models | `MixedLM`: REML/ML, random slopes + intercepts, variance components | empirical-residual-averaging stub (see above) | Needs a real Newton/EM-based variance-component estimator. |
| GEE | `GEE`/`NominalGEE`/`OrdinalGEE`: working correlation + sandwich variance | independent-GLM stub (see above) | Needs working-correlation estimation and a sandwich covariance. |
| Robust regression | `RLM`: 6 M-estimator norms + robust covariance | Huber-only, classical SEs | Add Hampel/TukeyBiweight/AndrewWave/RamsayE/TrimmedMean and a sandwich covariance. |
| Multiple comparisons | `stats.multicomp` (Tukey HSD), `stats.multitest` (Bonferroni, Holm, FDR BH/BY) | none | Tukey HSD is explicitly unchecked on inferust's own roadmap. |
| Power / sample size | `stats.power`: TTestPower, NormalIndPower, FTestAnovaPower, GofChisquarePower | none | No design-stage sample-size tooling at all. |
| Proportion tests | `stats.proportion`: proportions_ztest, proportion_confint, binom_test | none | Only general chi-squared/2×2 contingency tools exist. |
| Weighted descriptive stats | `stats.weightstats`: DescrStatsW, CompareMeans, ztest/ztest_ind | unweighted `Summary` only | No weighted mean/variance or two-sample z-test path. |
| ANOVA depth | `stats.anova.anova_lm` (type I/II/III SS from a fitted model, factorial, repeated measures) | one-way only, from raw groups | No two-way/N-way or model-comparison ANOVA. |
| Exponential smoothing | `Holt`, `ExponentialSmoothing` (Holt-Winters), `ETSModel`, `ThetaModel` | none | A common forecasting baseline that's entirely missing. |
| Seasonal decomposition | `seasonal_decompose`, `STL`, `MSTL` | none | No trend/seasonal/residual decomposition. |
| State-space breadth | `UnobservedComponents`, `DynamicFactor`/`DynamicFactorMQ` | scalar Kalman filter + local-level only | `statespace` covers one structural case out of several. |
| Markov regime-switching | `MarkovRegression`, `MarkovAutoregression` | none | — |
| ARDL / UECM | `tsa.ardl.ARDL`, `UECM` | none | Have VAR/VECM/VARMAX but no single-equation distributed-lag model. |
| Time-series filters | Baxter-King, Christiano-Fitzgerald, Hodrick-Prescott | none | — |
| Extra unit-root/independence tests | BDS test, Zivot-Andrews, range unit-root test | ADF, KPSS, Granger, Engle-Granger present | Three tests short of statsmodels' `tsa.stattools` set. |
| Factor analysis / canonical correlation | `multivariate.factor.Factor`, `CanCorr` | PCA + one-way MANOVA only | — |
| Beta regression | `othermod.betareg.BetaModel` | none | Useful for bounded (0,1) outcomes. |
| Conditional fixed-effects panel models | `ConditionalLogit`/`ConditionalMNLogit`/`ConditionalPoisson` | none | — |
| Diagnostic/graphics depth | `graphics.gofplots` (ProbPlot, qqplot), `stats.diagnostic` (Breusch-Godfrey, Goldfeld-Quandt) | VIF, Breusch-Pagan, White, RESET; no Q-Q/P-P plots | Serial-correlation and heteroskedasticity test menu and plotting are both short. |
| Survival depth | `PHReg`: strata, time-varying covariates, robust/cluster variance | Cox PH without strata, time-varying covariates, or robust variance | — |
| Multiple imputation | `imputation.bayes_mi` (BayesGaussMI, MI) | mean + MICE only | No Bayesian joint-model imputation. |
| Niche stats | mediation analysis, meta-analysis, inter-rater kappa (Cohen's/Fleiss') | none | Low priority, but namespace gaps. |

## What's already on inferust's own radar

`docs/parity.md`'s backlog and the README roadmap already flag Tukey HSD, ridge/lasso, and numerical-parity testing for `Gls`/`Fgls`, `QuantileRegression`, `RollingOls`/`RecursiveOls`, `glm_family`, `discrete`, the seasonal time-series models, and `gam`/`gee`/`gmm`/`mixed`/`robust`/`imputation`/`treatment`. This analysis is meant to extend that list, not duplicate it — the items above that aren't already in `docs/parity.md` are: GLM family/link breadth (Gamma/Tweedie/non-canonical links), exponential smoothing and seasonal decomposition, Markov regime-switching, ARDL/UECM, time-series filters, power analysis, proportion tests, weighted descriptive stats, multi-way ANOVA, beta regression, factor analysis/canonical correlation, conditional fixed-effects models, Q-Q/P-P plots, Breusch-Godfrey/Goldfeld-Quandt, and Cox PH strata/time-varying covariates.

## Suggested priority order

The highest-leverage items are the ones statsmodels users reach for constantly: Tukey HSD plus a multiple-testing correction utility, ridge/lasso/elastic-net for OLS and the GLMs, expanding `glm_family` to Gamma/Tweedie with selectable links, replacing the `mixed` and `gee` stubs with real REML and working-correlation estimators, and adding Holt-Winters/ETS plus STL decomposition to time series. A second tier — power analysis, proportion tests, `DescrStatsW`/`CompareMeans`, multi-way ANOVA, Q-Q/P-P plots, Breusch-Godfrey and Goldfeld-Quandt, and a full RLM norm menu — rounds out the statistics and diagnostics surface. A third, lower-priority tier covers Markov-switching, ARDL/UECM, dynamic factor models, the classical time-series filters, beta regression, factor analysis/canonical correlation, conditional fixed-effects panel models, Bayesian multiple imputation, and the niche mediation/meta-analysis/inter-rater tools.

As each item lands, follow the process already documented in `docs/parity.md`: add a fixture-generating function to `scripts/parity_statsmodels.py`, register it, add a `tests/parity_*.rs` assertion at the tolerance the doc specifies, and move the item from backlog to the audit matrix.
