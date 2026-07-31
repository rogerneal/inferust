#!/usr/bin/env python3
"""
Full benchmark suite: statsmodels / scipy / lifelines equivalents for every
estimator covered by bench_all.rs. Outputs key=value lines.
"""
import time, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.regression.rolling import RollingOLS
from statsmodels.regression.recursive_ls import RecursiveLS
from lifelines import CoxPHFitter
from scipy import stats as spstats
warnings.filterwarnings("ignore")

RNG = np.random.default_rng(42)
N = 10_000

def ms(s): return s * 1000.0

def bench(label, fn, rows=N, repeats=20, warmups=3):
    for _ in range(warmups):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    print(f"engine=python-statsmodels estimator={label} rows={rows} repeats={repeats} "
          f"min_ms={ms(times[0]):.3f} median_ms={ms(times[len(times)//2]):.3f} "
          f"mean_ms={ms(sum(times)/len(times)):.3f}", flush=True)

# ── OLS ──────────────────────────────────────────────────────────────────────
X = RNG.standard_normal((N,8)); y = X @ np.arange(1,9,dtype=float) + RNG.standard_normal(N)
Xc = sm.add_constant(X)
bench("ols", lambda: sm.OLS(y, Xc).fit())

# ── WLS ──────────────────────────────────────────────────────────────────────
X3 = RNG.standard_normal((N,3)); y3 = X3 @ [1.,2.,3.] + RNG.standard_normal(N)
w = 1/(1+np.arange(N)*0.0001); X3c = sm.add_constant(X3)
bench("wls", lambda: sm.WLS(y3, X3c, weights=w).fit())

# ── Ridge (via statsmodels fit_regularized) ──────────────────────────────────
X4 = RNG.standard_normal((N,4)); y4 = X4 @ [1.,2.,3.,4.] + RNG.standard_normal(N)
X4c = sm.add_constant(X4)
bench("ridge", lambda: sm.OLS(y4, X4c).fit_regularized(method='elastic_net', alpha=0.1, L1_wt=0.0))

# ── Lasso ────────────────────────────────────────────────────────────────────
bench("lasso", lambda: sm.OLS(y4, X4c).fit_regularized(method='elastic_net', alpha=0.01, L1_wt=1.0))

# ── ElasticNet ───────────────────────────────────────────────────────────────
bench("elastic_net", lambda: sm.OLS(y4, X4c).fit_regularized(method='elastic_net', alpha=0.01, L1_wt=0.5))

# ── QuantileRegression ───────────────────────────────────────────────────────
bench("quantile_reg", lambda: QuantReg(y3, X3c).fit(q=0.5))

# ── RollingOLS ───────────────────────────────────────────────────────────────
X2 = RNG.standard_normal((N,2)); y2 = X2 @ [1.,2.] + RNG.standard_normal(N)
X2c = sm.add_constant(X2)
bench("rolling_ols", lambda: RollingOLS(y2, X2c, window=200).fit(), repeats=10)

# ── RecursiveOLS ─────────────────────────────────────────────────────────────
bench("recursive_ols", lambda: RecursiveLS(y2, X2c).fit(), repeats=10)

# ── FGLS (Cochrane-Orcutt via GLSAR) ─────────────────────────────────────────
bench("fgls", lambda: sm.GLSAR(y3, X3c, rho=1).iterative_fit(maxiter=20), repeats=10)

# ── RobustLM (IRLS Huber via RLM) ────────────────────────────────────────────
bench("robust_lm", lambda: sm.RLM(y3, X3c, M=sm.robust.norms.HuberT()).fit(), repeats=10)

# ── Logistic ─────────────────────────────────────────────────────────────────
Xl = RNG.standard_normal((N,4))
pl = 1/(1+np.exp(-(Xl @ [0.5,-1,1.5,-0.5])))
yl = RNG.binomial(1, pl).astype(float)
Xlc = sm.add_constant(Xl)
bench("logistic", lambda: sm.GLM(yl, Xlc, family=sm.families.Binomial()).fit())

# ── Poisson ───────────────────────────────────────────────────────────────────
Xp = RNG.standard_normal((N,4))
yp = RNG.poisson(np.exp(Xp @ [0.3,0.5,-0.2,0.4] + 1.0)).astype(float)
Xpc = sm.add_constant(Xp)
bench("poisson", lambda: sm.GLM(yp, Xpc, family=sm.families.Poisson()).fit())

# ── Gamma GLM ─────────────────────────────────────────────────────────────────
Xg = RNG.standard_normal((N,3))
yg = np.exp(Xg @ [0.3,0.5,-0.2] + 1.0) * (1 + 0.1*np.abs(RNG.standard_normal(N)))
Xgc = sm.add_constant(Xg)
bench("gamma_glm", lambda: sm.GLM(yg, Xgc, family=sm.families.Gamma(sm.families.links.Log())).fit())

# ── Probit ────────────────────────────────────────────────────────────────────
Xpr = RNG.standard_normal((N,3))
ypr = (Xpr @ [0.8,-0.5,1.0] + RNG.standard_normal(N) > 0).astype(float)
Xprc = sm.add_constant(Xpr)
bench("probit", lambda: sm.Probit(ypr, Xprc).fit(disp=False))

# ── NegativeBinomial ─────────────────────────────────────────────────────────
ynb = RNG.poisson(np.exp(Xg @ [0.3,0.5,-0.2] + 1.0)).astype(float)
bench("neg_binomial", lambda: sm.NegativeBinomial(ynb, Xgc).fit(disp=False), repeats=10, warmups=1)

# ── MultinomialLogit ─────────────────────────────────────────────────────────
Xm = RNG.standard_normal((N,3))
eta1 = Xm[:,0]*0.5 - Xm[:,1]*0.3
eta2 = -Xm[:,0]*0.2 + Xm[:,2]*0.8
denom = 1 + np.exp(eta1) + np.exp(eta2)
p = np.column_stack([1/denom, np.exp(eta1)/denom, np.exp(eta2)/denom])
cum = p.cumsum(axis=1); u = RNG.random(N)[:,None]
ym = (cum < u).sum(axis=1)
Xmc = sm.add_constant(Xm)
bench("mnlogit", lambda: sm.MNLogit(ym, Xmc).fit(disp=False), repeats=10, warmups=1)

# ── OrderedLogit ─────────────────────────────────────────────────────────────
from statsmodels.miscmodels.ordinal_model import OrderedModel
Xo = RNG.standard_normal((N,2))
eta_o = Xo[:,0]*0.8 - Xo[:,1]*0.4 + 0.3*RNG.standard_normal(N)
yo = np.where(eta_o < -0.5, 0, np.where(eta_o < 0.5, 1, 2)).astype(int)
bench("ordered_logit", lambda: OrderedModel(yo, Xo, distr='logit').fit(method='bfgs', disp=False), repeats=5, warmups=1)

# ── ZeroInflatedPoisson ──────────────────────────────────────────────────────
from statsmodels.discrete.count_model import ZeroInflatedPoisson as SM_ZIP
Xz = RNG.standard_normal((N,2))
mu_z = np.exp(Xz @ [0.5,-0.3] + 1.0)
mask = RNG.random(N) < 0.2
yz = np.where(mask, 0, RNG.poisson(mu_z)).astype(float)
Xzc = sm.add_constant(Xz)
bench("zip", lambda: SM_ZIP(yz, Xzc).fit(disp=False), repeats=5, warmups=1)

# ── GEE ──────────────────────────────────────────────────────────────────────
Xe = RNG.standard_normal((N,3))
ye = RNG.poisson(np.exp(Xe @ [0.5,-0.3,0.4] + 0.5)).astype(float)
clusters_e = np.repeat(np.arange(N//10), 10)
Xec = sm.add_constant(Xe)
bench("gee", lambda: sm.GEE(ye, Xec, groups=clusters_e, family=sm.families.Poisson(),
      cov_struct=sm.cov_struct.Exchangeable()).fit(), repeats=5, warmups=1)

# ── MixedLM ──────────────────────────────────────────────────────────────────
Xmx = RNG.standard_normal((N,2))
ymx = Xmx @ [1.,2.] + RNG.standard_normal(N)
groups_mx = np.repeat(np.arange(N//20), 20)
Xmxc = sm.add_constant(Xmx)
bench("mixed_lm", lambda: sm.MixedLM(ymx, Xmxc, groups=groups_mx).fit(), repeats=5, warmups=1)

# ── KaplanMeier (lifelines) ───────────────────────────────────────────────────
from lifelines import KaplanMeierFitter
t_km = RNG.exponential(1, N); e_km = RNG.binomial(1, 0.8, N)
kmf = KaplanMeierFitter()
bench("kaplan_meier", lambda: kmf.fit(t_km, e_km))

# ── CoxPH ────────────────────────────────────────────────────────────────────
Xcox = RNG.standard_normal((N,3))
lam = np.exp(Xcox @ [0.5,-0.3,0.8])
t_cox = RNG.exponential(1/lam); e_cox = RNG.binomial(1, 0.8, N)
df_cox = pd.DataFrame(Xcox, columns=["x1","x2","x3"])
df_cox["T"] = t_cox; df_cox["E"] = e_cox
cph = CoxPHFitter()
bench("cox_ph", lambda: cph.fit(df_cox, "T", "E"), repeats=5, warmups=1)

# ── t-test (two-sample) ──────────────────────────────────────────────────────
a_t = RNG.standard_normal(N); b_t = RNG.standard_normal(N) + 0.2
bench("ttest_2samp", lambda: spstats.ttest_ind(a_t, b_t))

# ── Mann-Whitney U ───────────────────────────────────────────────────────────
a_mw = RNG.standard_normal(1000); b_mw = RNG.standard_normal(1000) + 0.2
bench("mann_whitney", lambda: spstats.mannwhitneyu(a_mw, b_mw), rows=1000)
