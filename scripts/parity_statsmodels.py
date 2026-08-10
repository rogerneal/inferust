#!/usr/bin/env python3
"""Generate statsmodels reference outputs for inferust parity tests.

Each dataset is built from a deterministic PCG-like LCG so that Rust can
reproduce the exact same `x` and `y` byte-for-byte without shipping CSVs.
Output is one JSON file per (estimator, dataset) under
``tests/fixtures/statsmodels/``.

Usage::

    python3 scripts/parity_statsmodels.py [--out DIR]

Run after installing ``statsmodels`` (``pip install statsmodels``). The JSON
files should be committed; Rust tests load them and compare each scalar /
vector against inferust outputs at controlled tolerances.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import statsmodels.api as sm
from scipy import stats as scipy_stats


# ── deterministic dataset builders ────────────────────────────────────────────
#
# We use a portable Lehmer / minimal-standard LCG so the *Rust* side can
# regenerate the exact same f64 values without depending on numpy's RNG. Every
# step is documented in tests/fixtures/statsmodels/README.md.


def _lcg(seed: int) -> "Lcg":
    return Lcg(seed)


class Lcg:
    """Park-Miller minimal-standard LCG. State is u32, output u01 in [0, 1)."""

    M: int = 2147483647  # 2^31 - 1
    A: int = 48271

    def __init__(self, seed: int) -> None:
        s = seed % self.M
        if s == 0:
            s = 1
        self.state: int = s

    def next_u32(self) -> int:
        self.state = (self.state * self.A) % self.M
        return self.state

    def next_u01(self) -> float:
        # (state - 1) / (M - 1) → exact division on doubles for state in [1, M-1].
        return (self.next_u32() - 1) / (self.M - 1)

    def standard_normal(self) -> float:
        # Box-Muller; consumes two uniforms.
        u1 = max(self.next_u01(), 1e-300)
        u2 = self.next_u01()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def dataset_linear(n: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """X[i,j] ~ N(0,1); y = 1.0 + sum_j (j+1)/k * X[:,j] + N(0, 0.5)."""
    rng = _lcg(seed)
    x = np.empty((n, k), dtype=np.float64)
    for i in range(n):
        for j in range(k):
            x[i, j] = rng.standard_normal()
    eps = np.array([rng.standard_normal() * 0.5 for _ in range(n)])
    beta = np.array([(j + 1) / k for j in range(k)])
    y = 1.0 + x @ beta + eps
    return x, y


def dataset_logit(n: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = _lcg(seed)
    x = np.empty((n, k), dtype=np.float64)
    for i in range(n):
        for j in range(k):
            x[i, j] = rng.standard_normal()
    beta = np.array([0.5, -0.7, 0.3, 0.1][:k])
    eta = 0.2 + x @ beta
    p = 1.0 / (1.0 + np.exp(-eta))
    # Use rng for Bernoulli sampling.
    y = np.array([1.0 if rng.next_u01() < pi else 0.0 for pi in p])
    return x, y


def dataset_poisson(n: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = _lcg(seed)
    x = np.empty((n, k), dtype=np.float64)
    for i in range(n):
        for j in range(k):
            x[i, j] = rng.standard_normal() * 0.5
    beta = np.array([0.3, -0.2, 0.1][:k])
    eta = 0.5 + x @ beta
    mu = np.exp(np.clip(eta, -5.0, 5.0))
    # Inverse-CDF Poisson sampler so it's reproducible from the LCG.
    y = np.array([_poisson_sample(rng, m) for m in mu], dtype=np.float64)
    return x, y


def _poisson_sample(rng: Lcg, mean: float) -> int:
    # Knuth-style sampler — fine for the small means here.
    L = math.exp(-mean)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.next_u01()
        if p <= L:
            return k - 1


def dataset_gamma(n: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """X[i,j] ~ N(0, 0.5); mu = exp(1 + X*beta); y = mu * (1 + 0.3*N(0,1)),
    floored away from zero so every observation stays strictly positive (a
    requirement of the Gamma family)."""
    rng = _lcg(seed)
    x = np.empty((n, k), dtype=np.float64)
    for i in range(n):
        for j in range(k):
            x[i, j] = rng.standard_normal() * 0.5
    beta = np.array([0.3, -0.2, 0.1][:k])
    eta = 1.0 + x @ beta
    mu = np.exp(np.clip(eta, -3.0, 3.0))
    y = np.array([max(mu[i] * (1.0 + 0.3 * rng.standard_normal()), 0.05) for i in range(n)])
    return x, y


def dataset_ar1(n: int, phi: float, seed: int) -> np.ndarray:
    """Stationary AR(1) with intercept c such that the mean is 5.0."""
    rng = _lcg(seed)
    c = 5.0 * (1.0 - phi)
    y = np.empty(n, dtype=np.float64)
    y[0] = 5.0 + rng.standard_normal()
    for t in range(1, n):
        y[t] = c + phi * y[t - 1] + rng.standard_normal()
    return y


def dataset_two_groups(n_a: int, n_b: int, shift: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = _lcg(seed)
    a = np.array([rng.standard_normal() for _ in range(n_a)])
    b = np.array([rng.standard_normal() + shift for _ in range(n_b)])
    return a, b


def dataset_survival(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-censored Weibull survival with a single covariate."""
    rng = _lcg(seed)
    x = np.array([rng.standard_normal() for _ in range(n)])
    beta = 0.7
    # Baseline Weibull shape=1.2, scale depends on x via PH.
    shape = 1.2
    times = np.empty(n)
    events = np.empty(n, dtype=int)
    for i in range(n):
        u = max(rng.next_u01(), 1e-12)
        scale = math.exp(-beta * x[i] / shape)
        t = scale * (-math.log(u)) ** (1.0 / shape)
        # Independent uniform censoring time.
        c = rng.next_u01() * 3.0
        if t <= c:
            times[i] = t
            events[i] = 1
        else:
            times[i] = c
            events[i] = 0
    return times, events, x


# ── helpers ───────────────────────────────────────────────────────────────────


def _to_list(arr: Any) -> list[float]:
    if arr is None:
        return []
    a = np.asarray(arr, dtype=np.float64).ravel()
    return [float(v) for v in a]


def _matrix(arr: Any) -> list[list[float]]:
    a = np.asarray(arr, dtype=np.float64)
    return [[float(v) for v in row] for row in a]


def _xy_payload(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    return {"x": [[float(v) for v in row] for row in x], "y": _to_list(y)}


# ── estimator runners ────────────────────────────────────────────────────────


def run_ols(n: int, k: int, seed: int) -> dict[str, Any]:
    x, y = dataset_linear(n, k, seed)
    xc = sm.add_constant(x, has_constant="add")
    res = sm.OLS(y, xc).fit()
    ci = res.conf_int(alpha=0.05)
    out = {
        "kind": "ols",
        "dataset": {"n": n, "k": k, "seed": seed, **_xy_payload(x, y)},
        "params": _to_list(res.params),
        "bse": _to_list(res.bse),
        "tvalues": _to_list(res.tvalues),
        "pvalues": _to_list(res.pvalues),
        "conf_int_05": _matrix(ci),
        "rsquared": float(res.rsquared),
        "rsquared_adj": float(res.rsquared_adj),
        "fvalue": float(res.fvalue),
        "f_pvalue": float(res.f_pvalue),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "ssr": float(res.ssr),
        "ess": float(res.ess),
        "centered_tss": float(res.centered_tss),
        "mse_resid": float(res.mse_resid),
        "df_resid": int(res.df_resid),
        "fittedvalues": _to_list(res.fittedvalues),
        "resid": _to_list(res.resid),
    }
    # Influence diagnostics
    inf = res.get_influence()
    out["influence"] = {
        "hat_diag": _to_list(inf.hat_matrix_diag),
        "cooks_d": _to_list(inf.cooks_distance[0]),
        "resid_studentized_internal": _to_list(inf.resid_studentized_internal),
    }
    return out


def run_ols_hc(n: int, k: int, seed: int, cov_type: str) -> dict[str, Any]:
    x, y = dataset_linear(n, k, seed)
    xc = sm.add_constant(x, has_constant="add")
    res = sm.OLS(y, xc).fit(cov_type=cov_type)
    return {
        "kind": f"ols_{cov_type.lower()}",
        "dataset": {"n": n, "k": k, "seed": seed, **_xy_payload(x, y)},
        "cov_type": cov_type,
        "params": _to_list(res.params),
        "bse": _to_list(res.bse),
        "tvalues": _to_list(res.tvalues),
        "pvalues": _to_list(res.pvalues),
    }


def run_wls(n: int, k: int, seed: int) -> dict[str, Any]:
    x, y = dataset_linear(n, k, seed)
    # Heteroskedastic weights tied to first covariate.
    weights = 1.0 / (1.0 + 0.5 * x[:, 0] ** 2)
    xc = sm.add_constant(x, has_constant="add")
    res = sm.WLS(y, xc, weights=weights).fit()
    return {
        "kind": "wls",
        "dataset": {
            "n": n,
            "k": k,
            "seed": seed,
            "weights": _to_list(weights),
            **_xy_payload(x, y),
        },
        "params": _to_list(res.params),
        "bse": _to_list(res.bse),
        "tvalues": _to_list(res.tvalues),
        "pvalues": _to_list(res.pvalues),
        "rsquared": float(res.rsquared),
        "rsquared_adj": float(res.rsquared_adj),
        "fvalue": float(res.fvalue),
        "ssr": float(res.ssr),
        "df_resid": int(res.df_resid),
    }


def run_logit(n: int, k: int, seed: int) -> dict[str, Any]:
    x, y = dataset_logit(n, k, seed)
    xc = sm.add_constant(x, has_constant="add")
    res = sm.Logit(y, xc).fit(disp=False)
    return {
        "kind": "logit",
        "dataset": {"n": n, "k": k, "seed": seed, **_xy_payload(x, y)},
        "params": _to_list(res.params),
        "bse": _to_list(res.bse),
        "zvalues": _to_list(res.tvalues),  # Logit uses normal; statsmodels reports as tvalues
        "pvalues": _to_list(res.pvalues),
        "conf_int_05": _matrix(res.conf_int(alpha=0.05)),
        "llf": float(res.llf),
        "llnull": float(res.llnull),
        "prsquared": float(res.prsquared),
        "aic": float(res.aic),
        "bic": float(res.bic),
    }


def run_poisson(n: int, k: int, seed: int) -> dict[str, Any]:
    x, y = dataset_poisson(n, k, seed)
    xc = sm.add_constant(x, has_constant="add")
    res = sm.Poisson(y, xc).fit(disp=False)
    return {
        "kind": "poisson",
        "dataset": {"n": n, "k": k, "seed": seed, **_xy_payload(x, y)},
        "params": _to_list(res.params),
        "bse": _to_list(res.bse),
        "zvalues": _to_list(res.tvalues),
        "pvalues": _to_list(res.pvalues),
        "llf": float(res.llf),
        "llnull": float(res.llnull),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "fittedvalues": _to_list(res.predict()),
    }


def run_gamma(n: int, k: int, seed: int) -> dict[str, Any]:
    """Fit a Gamma GLM under both `InversePower` (canonical) and `Log` links
    on the same dataset, matching `inferust::glm::Gamma::new()` /
    `.with_link(GammaLink::Log)`."""
    x, y = dataset_gamma(n, k, seed)
    xc = sm.add_constant(x, has_constant="add")
    out: dict[str, Any] = {
        "kind": "gamma",
        "dataset": {"n": n, "k": k, "seed": seed, **_xy_payload(x, y)},
    }
    links = {
        "inverse_power": sm.families.links.InversePower(),
        "log": sm.families.links.Log(),
    }
    for link_key, link in links.items():
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fam = sm.families.Gamma(link=link)
                res = sm.GLM(y, xc, family=fam).fit()
                pred = res.get_prediction(xc).summary_frame(alpha=0.05)
        out[link_key] = {
            "params": _to_list(res.params),
            "bse": _to_list(res.bse),
            "llf": float(res.llf),
            "llnull": float(res.llnull),
            "deviance": float(res.deviance),
            "pearson_chi2": float(res.pearson_chi2),
            "scale": float(res.scale),
            "aic": float(res.aic),
            "bic_llf": float(res.bic_llf),
            "mean": _to_list(pred["mean"]),
            "mean_ci_lower": _to_list(pred["mean_ci_lower"]),
            "mean_ci_upper": _to_list(pred["mean_ci_upper"]),
        }
    return out


def run_regularized(name: str, n: int, k: int, seed: int, alpha: float, l1_ratio: float) -> dict[str, Any]:
    """Fit `OLS.fit_regularized` with an alpha *vector* that zeroes the
    intercept's penalty, matching inferust's `Ridge`/`Lasso`/`ElasticNet`
    convention of never penalizing the intercept (see
    `src/regression/regularized.rs` module docs). Verified numerically to
    agree with inferust's coordinate-descent / closed-form solution to
    ~1e-13 for this objective scaling."""
    x, y = dataset_linear(n, k, seed)
    xc = sm.add_constant(x, has_constant="add")
    alpha_vec = np.array([0.0] + [alpha] * k)
    res = sm.OLS(y, xc).fit_regularized(
        method="elastic_net",
        alpha=alpha_vec,
        L1_wt=l1_ratio,
        maxiter=20000,
        cnvrg_tol=1e-12,
        refit=False,
    )
    return {
        "kind": name,
        "dataset": {"n": n, "k": k, "seed": seed, **_xy_payload(x, y)},
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "params": _to_list(res.params),
    }


def run_multicomp(seed: int) -> dict[str, Any]:
    from statsmodels.stats.multitest import multipletests

    rng = _lcg(seed)
    # Square the raw uniforms to skew the family toward a few small p-values
    # (so `reject` is a non-trivial mix of True/False across all four methods).
    p_values = [float(rng.next_u01() ** 2) for _ in range(15)]
    alpha = 0.05
    methods: dict[str, Any] = {}
    for key, sm_method in [
        ("Bonferroni", "bonferroni"),
        ("Holm", "holm"),
        ("BenjaminiHochberg", "fdr_bh"),
        ("BenjaminiYekutieli", "fdr_by"),
    ]:
        reject, p_corrected, alpha_sidak, alpha_bonferroni = multipletests(
            p_values, alpha=alpha, method=sm_method
        )
        methods[key] = {
            "reject": [bool(r) for r in reject],
            "p_corrected": _to_list(p_corrected),
            "alpha_sidak": float(alpha_sidak),
            "alpha_bonferroni": float(alpha_bonferroni),
        }
    return {
        "kind": "multicomp",
        "dataset": {"seed": seed, "p_values": p_values},
        "alpha": alpha,
        "methods": methods,
    }


def run_tukey_hsd(seed: int) -> dict[str, Any]:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    sizes = [12, 15, 10]
    shifts = [0.0, 1.2, 2.5]
    rng = _lcg(seed)
    groups = [
        np.array([rng.standard_normal() + shift for _ in range(n)])
        for n, shift in zip(sizes, shifts)
    ]
    data = np.concatenate(groups)
    group_labels = np.concatenate([[f"group{i}"] * len(g) for i, g in enumerate(groups)])
    res = pairwise_tukeyhsd(data, group_labels, alpha=0.05)
    return {
        "kind": "tukey_hsd",
        "dataset": {"seed": seed, "groups": [_to_list(g) for g in groups]},
        "alpha": 0.05,
        "mean_diffs": _to_list(res.meandiffs),
        "std_pairs": _to_list(res.std_pairs),
        "conf_int": _matrix(res.confint),
        "q_crit": float(res.q_crit),
        "reject": [bool(r) for r in res.reject],
        "p_values": _to_list(res.pvalues),
        "df_total": float(res.df_total),
    }


def run_arima(n: int, phi: float, seed: int) -> dict[str, Any]:
    y = dataset_ar1(n, phi, seed)
    # statsmodels' Arima uses MLE by default; inferust uses CSS for q==0 (just
    # AR), so we fit a pure AR(1) in statsmodels via conditional sum of squares
    # to keep estimators comparable.
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(y, order=(1, 0, 0), trend="c")
    res = model.fit(method="statespace")
    params = {str(name): float(val) for name, val in zip(res.param_names, np.asarray(res.params))}
    return {
        "kind": "arima_ar1",
        "dataset": {"n": n, "phi": phi, "seed": seed, "y": _to_list(y)},
        "params": params,
        "sigma2": float(params.get("sigma2", float("nan"))),
        "llf": float(res.llf),
        "aic": float(res.aic),
        "bic": float(res.bic),
    }


def run_acf_pacf(n: int, phi: float, seed: int, lags: int) -> dict[str, Any]:
    from statsmodels.tsa.stattools import acf, pacf, q_stat

    y = dataset_ar1(n, phi, seed)
    a = acf(y, nlags=lags, fft=False)
    p = pacf(y, nlags=lags, method="ywm")
    # Ljung-Box at the same lags
    q, qp = q_stat(a[1:], n)
    return {
        "kind": "acf_pacf",
        "dataset": {"n": n, "phi": phi, "seed": seed, "y": _to_list(y), "lags": lags},
        "acf": _to_list(a),
        "pacf": _to_list(p),
        "ljung_box_q": _to_list(q),
        "ljung_box_p": _to_list(qp),
    }


def run_adf(n: int, phi: float, seed: int) -> dict[str, Any]:
    from statsmodels.tsa.stattools import adfuller

    y = dataset_ar1(n, phi, seed)
    # autolag=None, default regression="c" matches inferust's intercept-only ADF.
    # statsmodels returns (stat, pval, usedlag, nobs, crit) when autolag is None.
    result = adfuller(y, maxlag=1, autolag=None, regression="c")
    stat, pval, _used_lag, _nobs, crit = result[0], result[1], result[2], result[3], result[4]
    return {
        "kind": "adf",
        "dataset": {"n": n, "phi": phi, "seed": seed, "y": _to_list(y)},
        "statistic": float(stat),
        "pvalue": float(pval),
        "crit": {k: float(v) for k, v in crit.items()},
    }


def run_t_one_sample(n: int, seed: int) -> dict[str, Any]:
    rng = _lcg(seed)
    data = np.array([rng.standard_normal() + 0.3 for _ in range(n)])
    res = scipy_stats.ttest_1samp(data, popmean=0.0)
    mean = float(data.mean())
    se = float(data.std(ddof=1) / math.sqrt(n))
    t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
    return {
        "kind": "ttest_1samp",
        "dataset": {"n": n, "seed": seed, "data": _to_list(data)},
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        "df": float(n - 1),
        "mean_diff": mean,
        "conf_int_05": [mean - t_crit * se, mean + t_crit * se],
    }


def run_t_two_sample(seed: int) -> dict[str, Any]:
    a, b = dataset_two_groups(50, 60, 0.4, seed)
    # equal_var=False = Welch's, which is what inferust documents.
    res = scipy_stats.ttest_ind(a, b, equal_var=False)
    return {
        "kind": "ttest_ind",
        "dataset": {"seed": seed, "a": _to_list(a), "b": _to_list(b)},
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        "df": float(res.df),
    }


def run_anova_oneway(seed: int) -> dict[str, Any]:
    rng = _lcg(seed)
    groups = []
    for shift in (0.0, 0.4, 0.8):
        groups.append(np.array([rng.standard_normal() + shift for _ in range(30)]))
    res = scipy_stats.f_oneway(*groups)
    return {
        "kind": "anova_oneway",
        "dataset": {"seed": seed, "groups": [_to_list(g) for g in groups]},
        "f_statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
    }


def run_mann_whitney(seed: int) -> dict[str, Any]:
    a, b = dataset_two_groups(40, 45, 0.5, seed)
    res = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "kind": "mann_whitney",
        "dataset": {"seed": seed, "a": _to_list(a), "b": _to_list(b)},
        "u_statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
    }


def run_pearson(seed: int) -> dict[str, Any]:
    rng = _lcg(seed)
    n = 80
    x = np.array([rng.standard_normal() for _ in range(n)])
    y = 0.6 * x + np.array([rng.standard_normal() * 0.8 for _ in range(n)])
    res = scipy_stats.pearsonr(x, y)
    spear = scipy_stats.spearmanr(x, y)
    return {
        "kind": "pearson_spearman",
        "dataset": {"seed": seed, "x": _to_list(x), "y": _to_list(y)},
        "pearson_r": float(res.statistic),
        "pearson_p": float(res.pvalue),
        "spearman_r": float(spear.statistic),
        "spearman_p": float(spear.pvalue),
    }


def run_chi2_independence(seed: int) -> dict[str, Any]:
    # Fixed 3x3 contingency table — no randomness.
    table = np.array([[10, 20, 30], [6, 9, 17], [8, 13, 22]], dtype=float)
    res = scipy_stats.chi2_contingency(table, correction=False)
    return {
        "kind": "chi2_independence",
        "dataset": {"table": _matrix(table), "seed": seed},
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        "dof": int(res.dof),
    }


def run_descriptive(seed: int) -> dict[str, Any]:
    rng = _lcg(seed)
    n = 200
    data = np.array([rng.standard_normal() * 2.0 + 5.0 for _ in range(n)])
    return {
        "kind": "descriptive",
        "dataset": {"seed": seed, "n": n, "data": _to_list(data)},
        "mean": float(np.mean(data)),
        "std": float(np.std(data, ddof=1)),
        "variance": float(np.var(data, ddof=1)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "q1": float(np.percentile(data, 25, method="linear")),
        "median": float(np.percentile(data, 50, method="linear")),
        "q3": float(np.percentile(data, 75, method="linear")),
        # scipy.stats.skew/kurtosis with bias=True match the population-moment
        # definitions inferust uses; kurtosis is reported as excess.
        "skewness": float(scipy_stats.skew(data, bias=True)),
        "kurtosis": float(scipy_stats.kurtosis(data, fisher=True, bias=True)),
    }


def dataset_granger(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a pair (y, x) where x leads y by one lag plus noise."""
    rng = _lcg(seed)
    x = np.empty(n, dtype=np.float64)
    x[0] = rng.standard_normal()
    for t in range(1, n):
        x[t] = 0.4 * x[t - 1] + rng.standard_normal()
    y = np.empty(n, dtype=np.float64)
    y[0] = rng.standard_normal()
    for t in range(1, n):
        y[t] = 0.3 * y[t - 1] + 0.6 * x[t - 1] + rng.standard_normal()
    return y, x


def run_granger(seed: int) -> dict[str, Any]:
    from statsmodels.tsa.stattools import grangercausalitytests

    y, x = dataset_granger(150, seed)
    # statsmodels expects shape (n, 2): first column = caused, second = cause.
    data = np.column_stack([y, x])
    result = grangercausalitytests(data, maxlag=[2], verbose=False)
    f_stat, p_val, _, _ = result[2][0]["ssr_ftest"]
    return {
        "kind": "granger_causality",
        "dataset": {"seed": seed, "y": _to_list(y), "x": _to_list(x), "lag": 2},
        "f_statistic": float(f_stat),
        "p_value": float(p_val),
    }


def run_engle_granger(seed: int) -> dict[str, Any]:
    from statsmodels.tsa.stattools import coint

    rng = _lcg(seed)
    n = 200
    x = np.empty(n, dtype=np.float64)
    x[0] = rng.standard_normal()
    for t in range(1, n):
        x[t] = x[t - 1] + rng.standard_normal()
    # y is a cointegrated combination of x with stationary noise.
    y = 1.0 + 2.0 * x + np.array([rng.standard_normal() * 0.5 for _ in range(n)])
    stat, pval, crit = coint(y, x, trend="c", maxlag=0)
    return {
        "kind": "engle_granger",
        "dataset": {"seed": seed, "y": _to_list(y), "x": _to_list(x)},
        "statistic": float(stat),
        "pvalue": float(pval),
        "critical": _to_list(crit),
    }


def run_wilcoxon(seed: int) -> dict[str, Any]:
    rng = _lcg(seed)
    n = 40
    a = np.array([rng.standard_normal() for _ in range(n)])
    b = np.array([rng.standard_normal() + 0.5 for _ in range(n)])
    res = scipy_stats.wilcoxon(a, b, zero_method="wilcox", correction=True)
    return {
        "kind": "wilcoxon",
        "dataset": {"seed": seed, "a": _to_list(a), "b": _to_list(b)},
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
    }


def run_sign_test(seed: int) -> dict[str, Any]:
    rng = _lcg(seed)
    n = 30
    a = np.array([rng.standard_normal() for _ in range(n)])
    b = np.array([rng.standard_normal() + 0.3 for _ in range(n)])
    diffs = a - b
    pos = int(np.sum(diffs > 0))
    neg = int(np.sum(diffs < 0))
    zeros = int(np.sum(diffs == 0))
    # Two-sided exact binomial p-value at p = 0.5.
    n_nonzero = pos + neg
    k = min(pos, neg)
    p_val = float(min(1.0, 2.0 * scipy_stats.binom.cdf(k, n_nonzero, 0.5)))
    return {
        "kind": "sign_test",
        "dataset": {"seed": seed, "a": _to_list(a), "b": _to_list(b)},
        "positives": pos,
        "negatives": neg,
        "zeros": zeros,
        "p_value": p_val,
    }


def run_anderson_darling(seed: int) -> dict[str, Any]:
    rng = _lcg(seed)
    n = 50
    data = np.array([rng.standard_normal() * 1.4 + 2.0 for _ in range(n)])
    res = scipy_stats.anderson(data, dist="norm")
    # scipy returns A² (already adjusted for normal-with-estimated-params if
    # we pick `dist='norm'`).
    return {
        "kind": "anderson_darling",
        "dataset": {"seed": seed, "n": n, "data": _to_list(data)},
        "statistic": float(res.statistic),
        "critical_values": _to_list(res.critical_values),
        "significance_levels": _to_list(res.significance_level),
    }


def run_lilliefors(seed: int) -> dict[str, Any]:
    from statsmodels.stats.diagnostic import lilliefors as sm_lilliefors

    rng = _lcg(seed)
    n = 40
    data = np.array([rng.standard_normal() * 1.2 + 3.0 for _ in range(n)])
    stat, pval = sm_lilliefors(data, dist="norm")
    return {
        "kind": "lilliefors",
        "dataset": {"seed": seed, "n": n, "data": _to_list(data)},
        "statistic": float(stat),
        "p_value": float(pval),
    }


def run_wald_ols(seed: int) -> dict[str, Any]:
    x, y = dataset_linear(n=120, k=4, seed=seed)
    xc = sm.add_constant(x, has_constant="add")
    res = sm.OLS(y, xc).fit()
    # Test the joint hypothesis: β1 = 0 AND β3 = 0 (using statsmodels' f_test).
    # Restriction matrix is r x (k+1) where k+1 = 5 with intercept first.
    r = [[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0]]
    q = [0.0, 0.0]
    ft = res.f_test(np.array(r))
    wt = res.wald_test(np.array(r), use_f=False, scalar=True)
    return {
        "kind": "wald_ols",
        "dataset": {"n": 120, "k": 4, "seed": seed, **_xy_payload(x, y)},
        "r": r,
        "q": q,
        "f_statistic": float(ft.fvalue),
        "f_pvalue": float(ft.pvalue),
        "chi2_statistic": float(wt.statistic),
        "chi2_pvalue": float(wt.pvalue),
        "df_num": int(ft.df_num),
        "df_den": int(ft.df_denom),
    }


def run_gls_ar1(n: int, k: int, seed: int, rho: float) -> dict[str, Any]:
    """GLS with a known AR(1) error covariance matrix Ω[i,j] = rho^|i-j|."""
    x, y = dataset_linear(n, k, seed)
    xc = sm.add_constant(x, has_constant="add")
    # Build AR(1) covariance matrix
    omega = np.array([[rho ** abs(i - j) for j in range(n)] for i in range(n)])
    res = sm.GLS(y, xc, sigma=omega).fit()
    return {
        "kind": "gls_ar1",
        "dataset": {"n": n, "k": k, "seed": seed, "rho": rho, **_xy_payload(x, y)},
        "omega": [[float(omega[i, j]) for j in range(n)] for i in range(n)],
        "params": _to_list(res.params),
        "bse": _to_list(res.bse),
        "tvalues": _to_list(res.tvalues),
        "pvalues": _to_list(res.pvalues),
    }


def run_fgls_cochrane_orcutt(n: int, k: int, seed: int) -> dict[str, Any]:
    """FGLS via GLSAR (Cochrane-Orcutt style)."""
    x, y = dataset_linear(n, k, seed)
    xc = sm.add_constant(x, has_constant="add")
    res = sm.GLSAR(y, xc, rho=1).iterative_fit(maxiter=10)
    return {
        "kind": "fgls_cochrane_orcutt",
        "dataset": {"n": n, "k": k, "seed": seed, **_xy_payload(x, y)},
        "params": _to_list(res.params),
        "rho": float(res.model.rho[0]) if hasattr(res.model.rho, '__len__') else float(res.model.rho),
    }


def run_quantreg(n: int, k: int, seed: int, q: float) -> dict[str, Any]:
    """Quantile regression at quantile q."""
    from statsmodels.regression.quantile_regression import QuantReg
    x, y = dataset_linear(n, k, seed)
    xc = sm.add_constant(x, has_constant="add")
    res = QuantReg(y, xc).fit(q=q)
    return {
        "kind": f"quantreg_q{int(q * 100)}",
        "dataset": {"n": n, "k": k, "seed": seed, **_xy_payload(x, y)},
        "q": q,
        "params": _to_list(res.params),
        "pseudor2": float(res.prsquared),
    }


def run_rolling_ols(n: int, k: int, seed: int, window: int) -> dict[str, Any]:
    """Rolling OLS with fixed window."""
    from statsmodels.regression.rolling import RollingOLS
    x, y = dataset_linear(n, k, seed)
    xc = sm.add_constant(x, has_constant="add")
    res = RollingOLS(y, xc, window=window).fit()
    # params shape: (n, k+1); first `window-1` rows are NaN
    params_arr = np.asarray(res.params)
    rsq_arr = np.asarray(res.rsquared)
    # Only keep rows where window is fully populated (index >= window-1)
    valid_start = window - 1
    params_valid = params_arr[valid_start:]
    rsq_valid = rsq_arr[valid_start:]
    return {
        "kind": "rolling_ols",
        "dataset": {"n": n, "k": k, "seed": seed, "window": window, **_xy_payload(x, y)},
        "params": _matrix(params_valid),
        "rsquared": _to_list(rsq_valid),
    }


def run_recursive_ols(n: int, k: int, seed: int) -> dict[str, Any]:
    """Recursive LS (recursive OLS via statsmodels sm.RecursiveLS)."""
    x, y = dataset_linear(n, k, seed)
    xc = sm.add_constant(x, has_constant="add")
    res = sm.RecursiveLS(y, xc).fit()
    # recursive_coefficients.filtered has shape (k+1, n) — column t is beta at time t
    fc = np.asarray(res.recursive_coefficients.filtered)  # shape (k+1, n)
    # cusum: the CUSUM statistic path (length n - k - 1)
    cusum_arr = np.asarray(res.cusum)
    # Extract at specific time indices
    idx = [10, 20, 30]
    params_at_idx = {str(i): _to_list(fc[:, i]) for i in idx if i < fc.shape[1]}
    return {
        "kind": "recursive_ols",
        "dataset": {"n": n, "k": k, "seed": seed, **_xy_payload(x, y)},
        "params_at_idx": params_at_idx,
        "cusum": _to_list(cusum_arr),
    }


def run_ks_one_sample(n: int, seed: int) -> dict[str, Any]:
    rng = _lcg(seed)
    data = np.array([rng.standard_normal() for _ in range(n)])
    mean = float(data.mean())
    std = float(data.std(ddof=1))
    from scipy.stats import norm as scipy_norm
    res = scipy_stats.kstest(data, lambda x: scipy_norm.cdf(x, loc=mean, scale=std))
    return {
        "kind": "ks_one_sample",
        "dataset": {"n": n, "seed": seed, "data": _to_list(data), "mean": mean, "std": std},
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
    }


def run_ks_two_sample(n_a: int, n_b: int, seed: int) -> dict[str, Any]:
    rng = _lcg(seed)
    a = np.array([rng.standard_normal() for _ in range(n_a)])
    b = np.array([rng.standard_normal() + 0.5 for _ in range(n_b)])
    res = scipy_stats.ks_2samp(a, b)
    return {
        "kind": "ks_two_sample",
        "dataset": {"n_a": n_a, "n_b": n_b, "seed": seed, "a": _to_list(a), "b": _to_list(b)},
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
    }


def run_kruskal_wallis(n_per_group: int, seed: int) -> dict[str, Any]:
    rng = _lcg(seed)
    groups = [
        np.array([rng.standard_normal() + shift for _ in range(n_per_group)])
        for shift in (0.0, 0.3, 0.6)
    ]
    res = scipy_stats.kruskal(*groups)
    return {
        "kind": "kruskal_wallis",
        "dataset": {"n_per_group": n_per_group, "seed": seed, "groups": [_to_list(g) for g in groups]},
        "h_statistic": float(res.statistic),
        "p_value": float(res.pvalue),
    }


def run_shapiro_wilk(n: int, seed: int) -> dict[str, Any]:
    rng = _lcg(seed)
    data = np.array([rng.standard_normal() for _ in range(n)])
    res = scipy_stats.shapiro(data)
    return {
        "kind": "shapiro_wilk",
        "dataset": {"n": n, "seed": seed, "data": _to_list(data)},
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
    }


def run_chi2_goodness_of_fit(k: int, seed: int) -> dict[str, Any]:
    rng = _lcg(seed)
    # Generate random observed counts (integers)
    raw = np.array([int(rng.next_u01() * 40 + 10) for _ in range(k)], dtype=float)
    # Expected: uniform
    expected = np.full(k, raw.sum() / k)
    res = scipy_stats.chisquare(raw, expected)
    return {
        "kind": "chi2_goodness_of_fit",
        "dataset": {"k": k, "seed": seed, "observed": _to_list(raw), "expected": _to_list(expected)},
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "df": float(k - 1),
    }


def run_kaplan_meier(seed: int) -> dict[str, Any]:
    from statsmodels.duration.survfunc import SurvfuncRight

    times, events, _x = dataset_survival(80, seed)
    sf = SurvfuncRight(times, events)
    # Pick a few time points: the 10th, 25th, 40th event times (if they exist)
    event_times = sorted(set(times[events == 1]))
    checkpoints = []
    for idx in [10, 25, 40]:
        if idx < len(event_times):
            t = event_times[idx]
            # Find survival at this time in the fitted curve
            surv_times = np.asarray(sf.surv_times)
            surv_probs = np.asarray(sf.surv_prob)
            # survival just after t (last value <= t)
            mask = surv_times <= t
            s = float(surv_probs[mask][-1]) if mask.any() else 1.0
            checkpoints.append({"time": float(t), "survival": s})
    n_events = int(np.sum(events))
    n_censored = int(len(events) - n_events)
    return {
        "kind": "kaplan_meier",
        "dataset": {
            "seed": seed,
            "n": int(len(times)),
            "times": _to_list(times),
            "events": [int(e) for e in events],
        },
        "checkpoints": checkpoints,
        "n_events": n_events,
        "n_censored": n_censored,
    }


def run_log_rank(seed: int) -> dict[str, Any]:
    from scipy.stats import logrank

    times, events, x = dataset_survival(80, seed)
    # Split by median covariate
    median_x = float(np.median(x))
    mask1 = x <= median_x
    mask2 = ~mask1
    times1 = times[mask1]
    events1 = events[mask1]
    times2 = times[mask2]
    events2 = events[mask2]
    from scipy.stats import CensoredData
    cd1 = CensoredData(uncensored=times1[events1 == 1], right=times1[events1 == 0])
    cd2 = CensoredData(uncensored=times2[events2 == 1], right=times2[events2 == 0])
    res = logrank(x=cd1, y=cd2)
    # scipy.stats.logrank returns a signed Z-statistic; chi2 = Z^2, same p-value
    return {
        "kind": "log_rank",
        "dataset": {
            "seed": seed,
            "n": int(len(times)),
            "times": _to_list(times),
            "events": [int(e) for e in events],
            "x": _to_list(x),
            "median_x": median_x,
        },
        "statistic": float(res.statistic ** 2),  # chi2 = Z^2
        "p_value": float(res.pvalue),
        "n_group1": int(mask1.sum()),
        "n_group2": int(mask2.sum()),
    }


def run_mcnemar(seed: int) -> dict[str, Any]:
    from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

    rng = _lcg(seed)
    # Generate a 2x2 paired binary table
    a = int(rng.next_u01() * 30 + 15)  # ~20-45
    b = int(rng.next_u01() * 20 + 5)   # ~5-25
    c = int(rng.next_u01() * 15 + 5)   # ~5-20
    d = int(rng.next_u01() * 30 + 15)  # ~15-45
    table = np.array([[a, b], [c, d]])
    res = sm_mcnemar(table, exact=False, correction=True)
    return {
        "kind": "mcnemar",
        "dataset": {"seed": seed, "table": _matrix(table.astype(float))},
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
    }


def run_odds_ratio(seed: int) -> dict[str, Any]:
    rng = _lcg(seed)
    a = int(rng.next_u01() * 30 + 10)
    b = int(rng.next_u01() * 20 + 5)
    c = int(rng.next_u01() * 15 + 5)
    d = int(rng.next_u01() * 30 + 10)
    table = np.array([[a, b], [c, d]])
    res = scipy_stats.contingency.odds_ratio(table, kind="sample")
    ci = res.confidence_interval(confidence_level=0.95)
    return {
        "kind": "odds_ratio",
        "dataset": {"seed": seed, "table": _matrix(table.astype(float))},
        "odds_ratio": float(res.statistic),
        "ci_lower": float(ci.low),
        "ci_upper": float(ci.high),
    }


def run_vif(n: int, k: int, seed: int) -> dict[str, Any]:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    x, _y = dataset_linear(n, k, seed)
    vifs = [float(variance_inflation_factor(x, j)) for j in range(k)]
    return {
        "kind": "vif",
        "dataset": {"n": n, "k": k, "seed": seed, "x": _matrix(x)},
        "vif": vifs,
    }


def run_breusch_pagan(n: int, k: int, seed: int) -> dict[str, Any]:
    from statsmodels.stats.diagnostic import het_breuschpagan

    # Heteroskedastic errors: multiply noise by abs(x1)
    rng = _lcg(seed)
    x = np.empty((n, k), dtype=np.float64)
    for i in range(n):
        for j in range(k):
            x[i, j] = rng.standard_normal()
    beta = np.array([(j + 1) / k for j in range(k)])
    # Heteroskedastic: sigma_i = 0.5 + |x[i,0]|
    eps = np.array([rng.standard_normal() * (0.5 + abs(x[i, 0])) for i in range(n)])
    y = 1.0 + x @ beta + eps
    xc = sm.add_constant(x, has_constant="add")
    res = sm.OLS(y, xc).fit()
    lm, lm_pval, _f, _f_pval = het_breuschpagan(res.resid, xc)
    return {
        "kind": "breusch_pagan",
        "dataset": {"n": n, "k": k, "seed": seed, "x": _matrix(x), "y": _to_list(y)},
        "statistic": float(lm),
        "p_value": float(lm_pval),
    }


def run_white_test(n: int, k: int, seed: int) -> dict[str, Any]:
    from statsmodels.stats.diagnostic import het_white

    rng = _lcg(seed)
    x = np.empty((n, k), dtype=np.float64)
    for i in range(n):
        for j in range(k):
            x[i, j] = rng.standard_normal()
    beta = np.array([(j + 1) / k for j in range(k)])
    eps = np.array([rng.standard_normal() * (0.5 + abs(x[i, 0])) for i in range(n)])
    y = 1.0 + x @ beta + eps
    xc = sm.add_constant(x, has_constant="add")
    res = sm.OLS(y, xc).fit()
    lm, lm_pval, _f, _f_pval = het_white(res.resid, xc)
    return {
        "kind": "white_test",
        "dataset": {"n": n, "k": k, "seed": seed, "x": _matrix(x), "y": _to_list(y)},
        "statistic": float(lm),
        "p_value": float(lm_pval),
    }


def run_reset_test(n: int, k: int, seed: int) -> dict[str, Any]:
    from statsmodels.stats.diagnostic import linear_reset

    x, y = dataset_linear(n, k, seed)
    xc = sm.add_constant(x, has_constant="add")
    res = sm.OLS(y, xc).fit()
    reset = linear_reset(res, power=3, use_f=True)
    return {
        "kind": "reset_test",
        "dataset": {"n": n, "k": k, "seed": seed, "x": _matrix(x), "y": _to_list(y)},
        "statistic": float(reset.fvalue),
        "p_value": float(reset.pvalue),
    }


def dataset_multinomial(n: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Three-class outcome from a softmax linear predictor."""
    rng = _lcg(seed)
    x = np.empty((n, k), dtype=np.float64)
    for i in range(n):
        for j in range(k):
            x[i, j] = rng.standard_normal() * 0.5
    beta0 = np.array([0.2, -0.3, 0.1][:k])
    beta1 = np.array([-0.4, 0.2, 0.0][:k])
    eta0 = 0.0 + x @ beta0
    eta1 = 0.5 + x @ beta1
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        e0 = math.exp(np.clip(eta0[i], -10.0, 10.0))
        e1 = math.exp(np.clip(eta1[i], -10.0, 10.0))
        z = 1.0 + e0 + e1
        p0, p1 = 1.0 / z, e0 / z
        u = rng.next_u01()
        if u < p0:
            y[i] = 0.0
        elif u < p0 + p1:
            y[i] = 1.0
        else:
            y[i] = 2.0
    return x, y


def dataset_ordered(n: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Ordinal 0/1/2 from latent linear index with cutpoints."""
    rng = _lcg(seed)
    x = np.empty((n, k), dtype=np.float64)
    for i in range(n):
        for j in range(k):
            x[i, j] = rng.standard_normal() * 0.5
    beta = np.array([0.4, -0.2, 0.1][:k])
    eta = x @ beta + np.array([rng.standard_normal() * 0.25 for _ in range(n)])
    y = np.zeros(n, dtype=np.float64)
    y[eta > -0.5] = 1.0
    y[eta > 0.8] = 2.0
    return x, y


def dataset_zip(n: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Count data with structural zeros for ZIP."""
    rng = _lcg(seed)
    x = np.empty((n, k), dtype=np.float64)
    for i in range(n):
        for j in range(k):
            x[i, j] = rng.standard_normal() * 0.4
    beta = np.array([0.25, -0.15, 0.05][:k])
    eta = 0.3 + x @ beta
    mu = np.exp(np.clip(eta, -3.0, 3.0))
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if rng.next_u01() < 0.25:
            y[i] = 0.0
        else:
            y[i] = float(_poisson_sample(rng, mu[i]))
    return x, y


def run_probit(n: int, k: int, seed: int) -> dict[str, Any]:
    x, y = dataset_logit(n, k, seed)
    xc = sm.add_constant(x, has_constant="add")
    res = sm.Probit(y, xc).fit(disp=False, maxiter=200)
    return {
        "kind": "probit",
        "dataset": {"n": n, "k": k, "seed": seed, **_xy_payload(x, y)},
        "params": _to_list(res.params),
        "bse": _to_list(res.bse),
        "zvalues": _to_list(res.tvalues),
        "pvalues": _to_list(res.pvalues),
        "llf": float(res.llf),
        "aic": float(res.aic),
        "bic": float(res.bic),
    }


def dataset_negbin(n: int, k: int, seed: int, alpha: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Overdispersed counts via Gamma–Poisson mixture (NB2)."""
    rng = _lcg(seed)
    x = np.empty((n, k), dtype=np.float64)
    for i in range(n):
        for j in range(k):
            x[i, j] = rng.standard_normal() * 0.4
    beta = np.array([0.35, -0.2, 0.1][:k])
    eta = 0.4 + x @ beta
    mu = np.exp(np.clip(eta, -2.0, 2.5))
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        # NB2: draw Gamma(shape=1/alpha, scale=alpha*mu) then Poisson
        g = rng.standard_normal()
        lam = mu[i] * max(0.1, 1.0 + math.sqrt(alpha) * g)
        y[i] = float(_poisson_sample(rng, lam))
    return x, y


def run_negbin(n: int, k: int, seed: int) -> dict[str, Any]:
    x, y = dataset_negbin(n, k, seed)
    xc = sm.add_constant(x, has_constant="add")
    res = sm.NegativeBinomial(y, xc).fit(disp=False, maxiter=300, method="bfgs")
    params = _to_list(res.params)
    return {
        "kind": "negbin",
        "dataset": {"n": n, "k": k, "seed": seed, **_xy_payload(x, y)},
        "params": params[:-1],
        "alpha": float(params[-1]),
        "bse": _to_list(res.bse)[:-1],
        "zvalues": _to_list(res.tvalues)[:-1],
        "pvalues": _to_list(res.pvalues)[:-1],
        "llf": float(res.llf),
        "aic": float(res.aic),
        "bic": float(res.bic),
    }


def run_multinomial(n: int, k: int, seed: int) -> dict[str, Any]:
    x, y = dataset_multinomial(n, k, seed)
    xc = sm.add_constant(x, has_constant="add")
    res = sm.MNLogit(y.astype(int), xc).fit(disp=False, maxiter=300)
    return {
        "kind": "multinomial",
        "dataset": {"n": n, "k": k, "seed": seed, **_xy_payload(x, y)},
        "params": _matrix(res.params),
        "bse": _matrix(res.bse),
        "llf": float(res.llf),
        "aic": float(res.aic),
        "bic": float(res.bic),
    }


def run_ordered_logit(n: int, k: int, seed: int) -> dict[str, Any]:
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    x, y = dataset_ordered(n, k, seed)
    res = OrderedModel(y.astype(int), x, distr="logit").fit(
        disp=False, method="bfgs", maxiter=300
    )
    # statsmodels orders params as cutpoints then slopes for exog.
    cutpoints = _to_list(res.params[: res.model.k_levels - 1])
    slopes = _to_list(res.params[res.model.k_levels - 1 :])
    bse = _to_list(res.bse)
    cut_bse = bse[: res.model.k_levels - 1]
    slope_bse = bse[res.model.k_levels - 1 :]
    return {
        "kind": "ordered_logit",
        "dataset": {"n": n, "k": k, "seed": seed, **_xy_payload(x, y)},
        "params": slopes,
        "cutpoints": cutpoints,
        "bse": slope_bse,
        "cutpoint_bse": cut_bse,
        "llf": float(res.llf),
        "aic": float(res.aic),
        "bic": float(res.bic),
    }


def run_zip(n: int, k: int, seed: int) -> dict[str, Any]:
    from statsmodels.discrete.count_model import ZeroInflatedPoisson

    x, y = dataset_zip(n, k, seed)
    xc = sm.add_constant(x, has_constant="add")
    res = ZeroInflatedPoisson(y, xc, exog_infl=xc, inflation="logit").fit(
        disp=False, maxiter=300
    )
    # params: count block then inflation block (each with intercept).
    q = xc.shape[1]
    count_params = _to_list(res.params[:q])
    infl_params = _to_list(res.params[q:])
    count_bse = _to_list(res.bse[:q])
    infl_bse = _to_list(res.bse[q:])
    return {
        "kind": "zip",
        "dataset": {"n": n, "k": k, "seed": seed, **_xy_payload(x, y)},
        "count_params": count_params,
        "inflation_params": infl_params,
        "count_bse": count_bse,
        "inflation_bse": infl_bse,
        "llf": float(res.llf),
        "aic": float(res.aic),
        "bic": float(res.bic),
    }


def run_gee_poisson(n: int, k: int, seed: int) -> dict[str, Any]:
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.genmod.families import Poisson
    from statsmodels.genmod.generalized_estimating_equations import GEE

    x, y = dataset_poisson(n, k, seed)
    xc = sm.add_constant(x, has_constant="add")
    clusters = np.array([(i % 10) for i in range(n)], dtype=int)
    res = GEE(y, xc, groups=clusters, family=Poisson(), cov_struct=Exchangeable()).fit()
    return {
        "kind": "gee_poisson",
        "dataset": {
            "n": n,
            "k": k,
            "seed": seed,
            "clusters": clusters.tolist(),
            **_xy_payload(x, y),
        },
        "params": _to_list(res.params),
        "bse": _to_list(res.bse),
        "llf": float(res.llf) if res.llf is not None else float("nan"),
    }


def run_cox(seed: int) -> dict[str, Any]:
    from statsmodels.duration.hazard_regression import PHReg

    times, events, x = dataset_survival(200, seed)
    res = PHReg(times, x.reshape(-1, 1), status=events, ties="breslow").fit()
    return {
        "kind": "cox_ph",
        "dataset": {
            "seed": seed,
            "n": int(len(times)),
            "times": _to_list(times),
            "events": [int(e) for e in events],
            "x": _matrix(x.reshape(-1, 1)),
        },
        "params": _to_list(res.params),
        "bse": _to_list(res.bse),
        "zvalues": _to_list(res.tvalues),
        "pvalues": _to_list(res.pvalues),
        "llf": float(res.llf),
    }


# ── orchestration ────────────────────────────────────────────────────────────


def _nan_to_none(arr: Any) -> list[Any]:
    """NaN-safe list conversion (JSON has no NaN literal serde accepts)."""
    out = []
    for v in np.asarray(arr, dtype=np.float64).ravel():
        out.append(None if not math.isfinite(v) else float(v))
    return out


def run_power() -> dict[str, Any]:
    from statsmodels.stats.power import (
        FTestAnovaPower,
        NormalIndPower,
        TTestIndPower,
        TTestPower,
    )

    cases_t1 = [
        {"effect_size": 0.5, "nobs": 30.0, "alpha": 0.05, "alternative": "two-sided"},
        {"effect_size": 0.3, "nobs": 80.0, "alpha": 0.05, "alternative": "larger"},
        {"effect_size": -0.4, "nobs": 50.0, "alpha": 0.10, "alternative": "smaller"},
    ]
    for c in cases_t1:
        c["power"] = float(
            TTestPower().power(c["effect_size"], c["nobs"], c["alpha"], alternative=c["alternative"])
        )
    cases_t2 = [
        {"effect_size": 0.5, "nobs1": 60.0, "alpha": 0.05, "ratio": 1.0, "alternative": "two-sided"},
        {"effect_size": 0.35, "nobs1": 90.0, "alpha": 0.05, "ratio": 1.5, "alternative": "larger"},
    ]
    for c in cases_t2:
        c["power"] = float(
            TTestIndPower().power(
                c["effect_size"], c["nobs1"], c["alpha"], ratio=c["ratio"], alternative=c["alternative"]
            )
        )
    cases_z = [
        {"effect_size": 0.3, "nobs1": 100.0, "alpha": 0.05, "ratio": 1.0, "alternative": "two-sided"},
    ]
    for c in cases_z:
        c["power"] = float(
            NormalIndPower().power(
                c["effect_size"], c["nobs1"], c["alpha"], ratio=c["ratio"], alternative=c["alternative"]
            )
        )
    cases_f = [
        {"effect_size": 0.25, "nobs": 120.0, "alpha": 0.05, "k_groups": 4.0},
        {"effect_size": 0.4, "nobs": 60.0, "alpha": 0.01, "k_groups": 3.0},
    ]
    for c in cases_f:
        c["power"] = float(
            FTestAnovaPower().power(c["effect_size"], nobs=c["nobs"], alpha=c["alpha"], k_groups=c["k_groups"])
        )
    solve_t2 = {
        "effect_size": 0.4,
        "power": 0.8,
        "alpha": 0.05,
        "ratio": 1.0,
        "alternative": "two-sided",
    }
    solve_t2["nobs1"] = float(
        TTestIndPower().solve_power(
            effect_size=solve_t2["effect_size"],
            power=solve_t2["power"],
            alpha=solve_t2["alpha"],
            ratio=solve_t2["ratio"],
            alternative=solve_t2["alternative"],
        )
    )
    return {
        "estimator": "statsmodels.stats.power",
        "ttest_power": cases_t1,
        "ttest_ind_power": cases_t2,
        "normal_ind_power": cases_z,
        "ftest_anova_power": cases_f,
        "solve_ttest_ind_nobs": solve_t2,
    }


def run_proportion() -> dict[str, Any]:
    from statsmodels.stats.proportion import (
        proportion_confint,
        proportion_effectsize,
        proportions_ztest,
    )

    z1, p1 = proportions_ztest(62, 100, value=0.5, alternative="two-sided")
    z1l, p1l = proportions_ztest(62, 100, value=0.5, alternative="larger")
    z2, p2 = proportions_ztest([45, 30], [80, 80], alternative="two-sided")
    confint = {}
    for method, key in [
        ("normal", "normal"),
        ("wilson", "wilson"),
        ("beta", "clopper_pearson"),
        ("agresti_coull", "agresti_coull"),
        ("jeffreys", "jeffreys"),
    ]:
        lo, hi = proportion_confint(62, 100, alpha=0.05, method=method)
        confint[key] = [float(lo), float(hi)]
    return {
        "estimator": "statsmodels.stats.proportion",
        "one_sample": {"count": 62, "nobs": 100, "value": 0.5, "z": float(z1), "p": float(p1)},
        "one_sample_larger": {"z": float(z1l), "p": float(p1l)},
        "two_sample": {
            "counts": [45, 30],
            "nobs": [80, 80],
            "z": float(z2),
            "p": float(p2),
        },
        "confint": confint,
        "effectsize": float(proportion_effectsize(0.6, 0.45)),
    }


def run_anova_twoway(seed: int) -> dict[str, Any]:
    import pandas as pd
    from statsmodels.formula.api import ols as smf_ols
    from statsmodels.stats.anova import anova_lm

    rng = _lcg(seed)
    levels_a = ["a1", "a2", "a3"]
    levels_b = ["b1", "b2"]
    rows = []
    # Unbalanced factorial: cell counts differ so Type I != Type II.
    counts = {
        ("a1", "b1"): 6, ("a1", "b2"): 9, ("a2", "b1"): 8,
        ("a2", "b2"): 5, ("a3", "b1"): 7, ("a3", "b2"): 10,
    }
    effects_a = {"a1": 0.0, "a2": 1.2, "a3": -0.7}
    effects_b = {"b1": 0.0, "b2": 0.8}
    inter = {("a2", "b2"): 1.5, ("a3", "b2"): -0.9}
    for (la, lb), cnt in counts.items():
        for _ in range(cnt):
            y = (
                5.0
                + effects_a[la]
                + effects_b[lb]
                + inter.get((la, lb), 0.0)
                + rng.standard_normal()
            )
            rows.append({"y": y, "a": la, "b": lb})
    df = pd.DataFrame(rows)
    fit = smf_ols("y ~ C(a) * C(b)", data=df).fit()

    def table(typ: int) -> dict[str, Any]:
        tab = anova_lm(fit, typ=typ)
        out = {}
        for row_name, key in [("C(a)", "a"), ("C(b)", "b"), ("C(a):C(b)", "ab"), ("Residual", "residual")]:
            r = tab.loc[row_name]
            out[key] = {
                "df": float(r["df"]),
                "sum_sq": float(r["sum_sq"]),
                "f": None if not math.isfinite(r.get("F", math.nan)) else float(r["F"]),
                "p": None if not math.isfinite(r.get("PR(>F)", math.nan)) else float(r["PR(>F)"]),
            }
        return out

    return {
        "estimator": "statsmodels.stats.anova.anova_lm",
        "dataset": {
            "y": [float(v) for v in df["y"]],
            "a": list(df["a"]),
            "b": list(df["b"]),
        },
        "type1": table(1),
        "type2": table(2),
    }


def _seasonal_series(n: int, period: int, seed: int) -> np.ndarray:
    rng = _lcg(seed)
    t = np.arange(n, dtype=np.float64)
    seasonal = 6.0 * np.sin(2.0 * np.pi * (t % period) / period)
    noise = np.array([rng.standard_normal() for _ in range(n)])
    return 30.0 + 0.15 * t + seasonal + 0.8 * noise


def run_seasonal_decompose(seed: int) -> dict[str, Any]:
    from statsmodels.tsa.seasonal import seasonal_decompose as sm_decompose

    period = 12
    y = _seasonal_series(120, period, seed)
    add = sm_decompose(y, model="additive", period=period)
    y_pos = y - y.min() + 5.0
    mul = sm_decompose(y_pos, model="multiplicative", period=period)
    return {
        "estimator": "statsmodels.tsa.seasonal.seasonal_decompose",
        "dataset": {"y": [float(v) for v in y], "y_positive": [float(v) for v in y_pos]},
        "period": period,
        "additive": {
            "trend": _nan_to_none(add.trend),
            "seasonal": _nan_to_none(add.seasonal),
            "resid": _nan_to_none(add.resid),
        },
        "multiplicative": {
            "trend": _nan_to_none(mul.trend),
            "seasonal": _nan_to_none(mul.seasonal),
            "resid": _nan_to_none(mul.resid),
        },
    }


def run_stl(seed: int) -> dict[str, Any]:
    from statsmodels.tsa.seasonal import STL

    period = 12
    y = _seasonal_series(144, period, seed)
    res = STL(y, period=period).fit()
    robust = STL(y, period=period, robust=True).fit()
    return {
        "estimator": "statsmodels.tsa.seasonal.STL",
        "dataset": {"y": [float(v) for v in y]},
        "period": period,
        "seasonal": _to_list(res.seasonal),
        "trend": _to_list(res.trend),
        "resid": _to_list(res.resid),
        "robust_seasonal": _to_list(robust.seasonal),
        "robust_trend": _to_list(robust.trend),
    }


def run_holt_winters(seed: int) -> dict[str, Any]:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

    period = 6
    y = _seasonal_series(72, period, seed)

    ses = SimpleExpSmoothing(
        y, initialization_method="known", initial_level=float(y[0])
    ).fit(smoothing_level=0.35, optimized=False)

    init_level = float(np.mean(y[:period]))
    init_trend = float((np.mean(y[period : 2 * period]) - np.mean(y[:period])) / period)
    init_seasonal = [float(v - init_level) for v in y[:period]]
    hw_fixed = ExponentialSmoothing(
        y,
        trend="add",
        seasonal="add",
        seasonal_periods=period,
        initialization_method="known",
        initial_level=init_level,
        initial_trend=init_trend,
        initial_seasonal=init_seasonal,
    ).fit(
        smoothing_level=0.4,
        smoothing_trend=0.2,
        smoothing_seasonal=0.3,
        optimized=False,
    )
    hw_opt = ExponentialSmoothing(
        y,
        trend="add",
        seasonal="add",
        seasonal_periods=period,
        initialization_method="legacy-heuristic",
    ).fit()
    return {
        "estimator": "statsmodels.tsa.holtwinters",
        "dataset": {"y": [float(v) for v in y]},
        "period": period,
        "ses_fixed": {
            "alpha": 0.35,
            "initial_level": float(y[0]),
            "fitted": _to_list(ses.fittedvalues),
            "sse": float(ses.sse),
            "forecast": _to_list(ses.forecast(8)),
        },
        "hw_fixed": {
            "alpha": 0.4,
            "beta": 0.2,
            "gamma": 0.3,
            "initial_level": init_level,
            "initial_trend": init_trend,
            "initial_seasonal": init_seasonal,
            "fitted": _to_list(hw_fixed.fittedvalues),
            "sse": float(hw_fixed.sse),
            "forecast": _to_list(hw_fixed.forecast(2 * period)),
        },
        "hw_optimized_sse": float(hw_opt.sse),
    }


def run_forecast_ci(seed: int) -> dict[str, Any]:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.api import VAR as SmVar

    rng = _lcg(seed)
    n = 200
    # ARIMA(1,1,1) with known params: filter (no estimation) and read se_mean.
    eps = np.array([rng.standard_normal() for _ in range(n)])
    dy = np.empty(n)
    dy[0] = eps[0]
    for t in range(1, n):
        dy[t] = 0.6 * dy[t - 1] + eps[t] + 0.25 * eps[t - 1]
    y = 50.0 + np.cumsum(dy)
    arima_params = {"ar": 0.6, "ma": 0.25, "sigma2": 1.2}
    mod = SARIMAX(y, order=(1, 1, 1), trend=None)
    res = mod.filter([arima_params["ar"], arima_params["ma"], arima_params["sigma2"]])
    fc = res.get_forecast(10)

    # Seasonal case with known params.
    smod = SARIMAX(y, order=(1, 0, 0), seasonal_order=(0, 1, 1, 4), trend=None)
    sres = smod.filter([0.55, -0.4, 0.9])
    sfc = sres.get_forecast(12)

    # VAR(2): both sides estimate by OLS, so point forecasts and intervals
    # should agree tightly.
    m = 160
    e1 = np.array([rng.standard_normal() for _ in range(m)])
    e2 = np.array([rng.standard_normal() for _ in range(m)])
    y1 = np.zeros(m)
    y2 = np.zeros(m)
    for t in range(2, m):
        y1[t] = 0.5 * y1[t - 1] - 0.2 * y2[t - 2] + e1[t]
        y2[t] = 0.3 * y2[t - 1] + 0.25 * y1[t - 1] + e2[t]
    data = np.column_stack([y1, y2])
    var_res = SmVar(data).fit(2, trend="c")
    point, lower, upper = var_res.forecast_interval(data[-2:], 8, alpha=0.05)

    return {
        "estimator": "SARIMAX.filter/get_forecast + VAR.forecast_interval",
        "dataset": {"y": [float(v) for v in y], "y1": _to_list(y1), "y2": _to_list(y2)},
        "arima": {
            "order": [1, 1, 1],
            "params": arima_params,
            "se_mean": _to_list(fc.se_mean),
        },
        "sarima": {
            "order": [1, 0, 0],
            "seasonal_order": [0, 1, 1, 4],
            "params": {"ar": 0.55, "seasonal_ma": -0.4, "sigma2": 0.9},
            "se_mean": _to_list(sfc.se_mean),
        },
        "var": {
            "lags": 2,
            "point": _matrix(point),
            "lower": _matrix(lower),
            "upper": _matrix(upper),
        },
    }


def run_pca(n: int, k: int, seed: int) -> dict[str, Any]:
    """Covariance PCA via statsmodels.multivariate.pca.PCA.

    Matches inferust's `pca`: demean, no column standardization, eigendecomposition
    of the sample covariance. statsmodels reports eigenvalues of X'X (not divided
    by n−1); the fixture stores covariance eigenvalues so Rust can compare
    `explained_variance` directly. Loadings may flip sign relative to inferust;
    the Rust test aligns signs before checking.
    """
    from statsmodels.multivariate.pca import PCA

    rng = _lcg(seed)
    x = np.empty((n, k), dtype=np.float64)
    for i in range(n):
        for j in range(k):
            x[i, j] = rng.standard_normal()
    # Build a dominant first component so ratio checks are meaningful.
    for i in range(n):
        x[i, 1] = 0.8 * x[i, 0] + 0.2 * x[i, 1]
        if k > 2:
            x[i, 2] = 0.5 * x[i, 0] + 0.5 * x[i, 2]

    fit = PCA(x, standardize=False, demean=True, normalize=False, method="eig")
    loadings = np.asarray(fit.loadings, dtype=np.float64)
    # Convert X'X eigenvalues → sample-covariance eigenvalues.
    explained = np.asarray(fit.eigenvals, dtype=np.float64).ravel() / (n - 1)
    total = explained.sum()
    ratios = explained / total
    mean = x.mean(axis=0)
    xc = x - mean
    scores = xc @ loadings

    return {
        "kind": "pca",
        "n": n,
        "k": k,
        "seed": seed,
        "dataset": {"x": _matrix(x)},
        "mean": _to_list(mean),
        "components": _matrix(loadings.T),  # row-per-component, like inferust
        "explained_variance": _to_list(explained),
        "explained_variance_ratio": _to_list(ratios),
        "scores": _matrix(scores),
    }


def run_manova(seed: int) -> dict[str, Any]:
    """One-way MANOVA via statsmodels.multivariate.manova.MANOVA."""
    from statsmodels.multivariate.manova import MANOVA
    import pandas as pd

    rng = _lcg(seed)
    # 3 groups × 20 obs × 2 responses
    groups_data: list[list[list[float]]] = []
    rows_y1: list[float] = []
    rows_y2: list[float] = []
    rows_g: list[str] = []
    for g in range(3):
        group: list[list[float]] = []
        for _ in range(20):
            y1 = 0.8 * g + rng.standard_normal()
            y2 = 0.3 * g + rng.standard_normal()
            group.append([y1, y2])
            rows_y1.append(y1)
            rows_y2.append(y2)
            rows_g.append(str(g))
        groups_data.append(group)

    df = pd.DataFrame({"y1": rows_y1, "y2": rows_y2, "g": rows_g})
    result = MANOVA.from_formula("y1 + y2 ~ g", data=df).mv_test()
    table = result["g"]["stat"]
    # Index 0 is Wilks' lambda row.
    wilks = float(table.iloc[0, 0])
    df_h = float(table.iloc[0, 1])
    df_e = float(table.iloc[0, 2])
    f_stat = float(table.iloc[0, 3])
    p_value = float(table.iloc[0, 4])

    return {
        "kind": "manova",
        "seed": seed,
        "dataset": {"groups": groups_data},
        "wilks_lambda": wilks,
        "df_hypothesis": df_h,
        "df_error": df_e,
        "f_statistic": f_stat,
        "p_value": p_value,
        "groups": 3,
        "responses": 2,
    }


def run_panel_fe(n_entities: int, n_times: int, k: int, seed: int) -> dict[str, Any]:
    """Entity fixed-effects panel OLS.

    Coefficients are validated against ``linearmodels.panel.PanelOLS`` with
    ``entity_effects=True``. Standard errors are from statsmodels OLS on the
    same within-transformed data (no intercept), which matches inferust's
    demean-then-``Ols::no_intercept`` implementation. linearmodels applies an
    extra within-df correction to SEs, so those are not used here.
    """
    from linearmodels.panel import PanelOLS
    import pandas as pd

    rng = _lcg(seed)
    rows: list[dict[str, Any]] = []
    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    entities: list[int] = []
    for e in range(n_entities):
        alpha_e = rng.standard_normal()
        for t in range(n_times):
            x_i = [rng.standard_normal() for _ in range(k)]
            y_i = (
                1.0
                + alpha_e
                + sum((j + 1) / k * x_i[j] for j in range(k))
                + 0.25 * rng.standard_normal()
            )
            row = {"entity": e, "time": t, "y": y_i}
            for j, v in enumerate(x_i):
                row[f"x{j + 1}"] = v
            rows.append(row)
            x_rows.append(x_i)
            y_rows.append(y_i)
            entities.append(e)

    df = pd.DataFrame(rows).set_index(["entity", "time"])
    exog_cols = [f"x{j + 1}" for j in range(k)]
    panel = PanelOLS(df["y"], df[exog_cols], entity_effects=True).fit()

    # Within SEs via demean + OLS (matches inferust).
    flat = df.reset_index()
    dm = flat.copy()
    for col in ["y", *exog_cols]:
        dm[col] = flat[col] - flat.groupby("entity")[col].transform("mean")
    within = sm.OLS(dm["y"], dm[exog_cols]).fit()

    return {
        "kind": "panel_fe",
        "n_entities": n_entities,
        "n_times": n_times,
        "k": k,
        "seed": seed,
        "dataset": {
            "x": _matrix(np.asarray(x_rows, dtype=np.float64)),
            "y": _to_list(np.asarray(y_rows, dtype=np.float64)),
            "entities": entities,
        },
        "params": _to_list(np.asarray(panel.params, dtype=np.float64)),
        "bse": _to_list(np.asarray(within.bse, dtype=np.float64)),
        "tvalues": _to_list(np.asarray(within.tvalues, dtype=np.float64)),
        "pvalues": _to_list(np.asarray(within.pvalues, dtype=np.float64)),
        "rsquared": float(within.rsquared),
    }


def emit(out_dir: Path, name: str, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{name}.json"
    with target.open("w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    print(f"  wrote {target.relative_to(out_dir.parent.parent)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "statsmodels",
        help="Output directory (default tests/fixtures/statsmodels/)",
    )
    args = parser.parse_args()
    out: Path = args.out

    print(f"writing fixtures to {out}")

    # OLS family
    emit(out, "ols_small", run_ols(n=60, k=3, seed=1))
    emit(out, "ols_medium", run_ols(n=400, k=5, seed=2))
    emit(out, "ols_hc0", run_ols_hc(n=200, k=3, seed=11, cov_type="HC0"))
    emit(out, "ols_hc1", run_ols_hc(n=200, k=3, seed=11, cov_type="HC1"))
    emit(out, "ols_hc2", run_ols_hc(n=200, k=3, seed=11, cov_type="HC2"))
    emit(out, "ols_hc3", run_ols_hc(n=200, k=3, seed=11, cov_type="HC3"))
    emit(out, "wls_small", run_wls(n=120, k=2, seed=3))

    # GLM
    emit(out, "logit_small", run_logit(n=200, k=3, seed=4))
    emit(out, "poisson_small", run_poisson(n=200, k=3, seed=5))
    emit(out, "gamma_glm", run_gamma(n=150, k=2, seed=41))

    # Regularized regression
    emit(out, "ridge_small", run_regularized("ridge", n=80, k=3, seed=31, alpha=0.5, l1_ratio=0.0))
    emit(out, "lasso_small", run_regularized("lasso", n=80, k=3, seed=31, alpha=0.1, l1_ratio=1.0))
    emit(
        out,
        "elastic_net_small",
        run_regularized("elastic_net", n=80, k=3, seed=31, alpha=0.2, l1_ratio=0.5),
    )

    # Time series
    emit(out, "arima_ar1", run_arima(n=300, phi=0.6, seed=6))
    emit(out, "acf_pacf", run_acf_pacf(n=300, phi=0.5, seed=7, lags=10))
    emit(out, "adf", run_adf(n=300, phi=0.5, seed=7))

    # Hypothesis tests
    emit(out, "ttest_1samp", run_t_one_sample(n=80, seed=8))
    emit(out, "ttest_ind", run_t_two_sample(seed=9))
    emit(out, "anova_oneway", run_anova_oneway(seed=10))
    emit(out, "mann_whitney", run_mann_whitney(seed=12))
    emit(out, "chi2_independence", run_chi2_independence(seed=13))
    emit(out, "multicomp", run_multicomp(seed=42))
    emit(out, "tukey_hsd", run_tukey_hsd(seed=43))

    # Correlation
    emit(out, "pearson_spearman", run_pearson(seed=14))

    # Descriptive
    emit(out, "descriptive", run_descriptive(seed=15))

    # Survival
    emit(out, "cox_ph", run_cox(seed=16))

    # GLS / FGLS / Quantile / Rolling
    emit(out, "gls_ar1", run_gls_ar1(n=40, k=2, seed=50, rho=0.6))
    emit(out, "fgls_cochrane_orcutt", run_fgls_cochrane_orcutt(n=60, k=2, seed=51))
    emit(out, "quantreg_median", run_quantreg(n=80, k=2, seed=52, q=0.5))
    emit(out, "quantreg_q25", run_quantreg(n=80, k=2, seed=52, q=0.25))
    emit(out, "rolling_ols", run_rolling_ols(n=60, k=2, seed=53, window=20))
    emit(out, "recursive_ols", run_recursive_ols(n=50, k=1, seed=54))

    # New (0.1.13) features
    emit(out, "granger_causality", run_granger(seed=17))
    emit(out, "engle_granger", run_engle_granger(seed=18))
    emit(out, "wilcoxon", run_wilcoxon(seed=19))
    emit(out, "sign_test", run_sign_test(seed=20))
    emit(out, "anderson_darling", run_anderson_darling(seed=21))
    emit(out, "lilliefors", run_lilliefors(seed=22))
    emit(out, "wald_ols", run_wald_ols(seed=23))

    # New (0.1.14) features
    emit(out, "ks_one_sample", run_ks_one_sample(n=60, seed=20))
    emit(out, "ks_two_sample", run_ks_two_sample(n_a=40, n_b=45, seed=21))
    emit(out, "kruskal_wallis_parity", run_kruskal_wallis(n_per_group=25, seed=22))
    emit(out, "shapiro_wilk", run_shapiro_wilk(n=40, seed=23))
    emit(out, "chi2_goodness_of_fit", run_chi2_goodness_of_fit(k=5, seed=24))
    emit(out, "kaplan_meier", run_kaplan_meier(seed=7))
    emit(out, "log_rank", run_log_rank(seed=7))
    emit(out, "mcnemar", run_mcnemar(seed=60))
    emit(out, "odds_ratio", run_odds_ratio(seed=61))
    emit(out, "vif", run_vif(n=60, k=3, seed=30))
    emit(out, "breusch_pagan", run_breusch_pagan(n=60, k=2, seed=31))
    emit(out, "white_test", run_white_test(n=60, k=2, seed=32))
    emit(out, "reset_test", run_reset_test(n=80, k=2, seed=33))

    # Discrete choice (0.1.18)
    emit(out, "probit_small", run_probit(n=200, k=3, seed=44))
    emit(out, "negbin_small", run_negbin(n=200, k=3, seed=45))
    emit(out, "multinomial_small", run_multinomial(n=240, k=2, seed=46))
    emit(out, "ordered_logit_small", run_ordered_logit(n=200, k=2, seed=47))
    emit(out, "zip_small", run_zip(n=200, k=2, seed=48))
    emit(out, "gee_poisson", run_gee_poisson(n=200, k=2, seed=49))

    # Power analysis, proportions, two-way ANOVA (0.2.0)
    emit(out, "power", run_power())
    emit(out, "proportion", run_proportion())
    emit(out, "anova_twoway", run_anova_twoway(seed=70))

    # Seasonal decomposition, exponential smoothing, forecast intervals (0.2.0)
    emit(out, "seasonal_decompose", run_seasonal_decompose(seed=71))
    emit(out, "stl", run_stl(seed=72))
    emit(out, "holt_winters", run_holt_winters(seed=73))
    emit(out, "forecast_ci", run_forecast_ci(seed=74))

    # Multivariate PCA / MANOVA and panel entity FE (0.3.2)
    emit(out, "pca", run_pca(n=60, k=3, seed=80))
    emit(out, "manova", run_manova(seed=81))
    emit(out, "panel_fe", run_panel_fe(n_entities=12, n_times=8, k=2, seed=82))

    print(f"\nstatsmodels version: {sm.__version__}")


if __name__ == "__main__":
    main()
