use std::hint::black_box;
use std::time::{Duration, Instant};

use inferust::discrete::{
    MultinomialLogit, NegativeBinomial, OrderedLogit, Probit, ZeroInflatedPoisson,
};
use inferust::gee::{Gee, GeeFamily, WorkingCorrelation};
use inferust::glm::{Gamma, GammaLink, Logistic, Poisson};
use inferust::hypothesis::{nonparametric, ttest};
use inferust::mixed::MixedLinearModel;
use inferust::regression::{
    ElasticNet, Fgls, Lasso, Ols, QuantileRegression, RecursiveOls, Ridge, RollingOls, Wls,
};
use inferust::robust::RobustLinearModel;
use inferust::survival::{CoxPh, KaplanMeier};

fn millis(d: Duration) -> f64 {
    d.as_secs_f64() * 1_000.0
}

// black_box wraps the unit return to keep the fit call from being elided.
#[allow(clippy::unit_arg)]
fn bench<F: Fn()>(label: &str, rows: usize, repeats: usize, warmups: usize, f: F) {
    for _ in 0..warmups {
        f();
    }
    let mut times: Vec<Duration> = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let t = Instant::now();
        black_box(f());
        times.push(t.elapsed());
    }
    times.sort_unstable();
    let mean = times.iter().map(|d| millis(*d)).sum::<f64>() / times.len() as f64;
    println!(
        "engine=rust-inferust estimator={} rows={} repeats={} min_ms={:.3} median_ms={:.3} mean_ms={:.3}",
        label, rows, repeats,
        millis(times[0]),
        millis(times[times.len() / 2]),
        mean,
    );
}

// ── deterministic data generators ────────────────────────────────────────────

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / (1u64 << 53) as f64
}

fn norm(s: &mut u64) -> f64 {
    let u1 = lcg(s).max(1e-15);
    let u2 = lcg(s);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn mat(s: &mut u64, rows: usize, cols: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|_| (0..cols).map(|_| norm(s)).collect())
        .collect()
}

fn linear_y(x: &[Vec<f64>], beta: &[f64], noise: f64, s: &mut u64) -> Vec<f64> {
    x.iter()
        .map(|r| r.iter().zip(beta).map(|(xi, bi)| xi * bi).sum::<f64>() + noise * norm(s))
        .collect()
}

fn binary_y(x: &[Vec<f64>], beta: &[f64], s: &mut u64) -> Vec<f64> {
    x.iter()
        .map(|r| {
            let eta: f64 = r.iter().zip(beta).map(|(xi, bi)| xi * bi).sum();
            let p = 1.0 / (1.0 + (-eta).exp());
            if lcg(s) < p {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

fn count_y(x: &[Vec<f64>], beta: &[f64], offset: f64, s: &mut u64) -> Vec<f64> {
    x.iter()
        .map(|r| {
            let mu = (r.iter().zip(beta).map(|(xi, bi)| xi * bi).sum::<f64>() + offset).exp();
            let mut k = 0u64;
            let mut p = (-mu).exp();
            let mut cdf = p;
            let u = lcg(s);
            while cdf < u && k < 2000 {
                k += 1;
                p *= mu / k as f64;
                cdf += p;
            }
            k as f64
        })
        .collect()
}

fn main() {
    let n = 10_000_usize;
    let r20 = 20_usize;
    let r10 = 10_usize;
    let r5 = 5_usize;
    let w3 = 3_usize;
    let w1 = 1_usize;

    // ── OLS ──────────────────────────────────────────────────────────────────
    {
        let mut s = 1u64;
        let b = [1., 2., 3., 4., 5., 6., 7., 8.];
        let x = mat(&mut s, n, 8);
        let y = linear_y(&x, &b, 1.0, &mut s);
        let m = Ols::new();
        bench("ols", n, r20, w3, || {
            m.fit(&x, &y).unwrap();
        });
    }

    // ── WLS ──────────────────────────────────────────────────────────────────
    {
        let mut s = 2u64;
        let b = [1., 2., 3.];
        let x = mat(&mut s, n, 3);
        let y = linear_y(&x, &b, 1.0, &mut s);
        let w: Vec<f64> = (0..n).map(|i| 1.0 / (1.0 + i as f64 * 0.0001)).collect();
        let m = Wls::new();
        bench("wls", n, r20, w3, || {
            m.fit(&x, &y, &w).unwrap();
        });
    }

    // ── Ridge ─────────────────────────────────────────────────────────────────
    {
        let mut s = 3u64;
        let b = [1., 2., 3., 4.];
        let x = mat(&mut s, n, 4);
        let y = linear_y(&x, &b, 1.0, &mut s);
        let m = Ridge::new(0.1);
        bench("ridge", n, r20, w3, || {
            m.fit(&x, &y).unwrap();
        });
    }

    // ── Lasso ────────────────────────────────────────────────────────────────
    {
        let mut s = 4u64;
        let b = [1., 2., 3., 4.];
        let x = mat(&mut s, n, 4);
        let y = linear_y(&x, &b, 1.0, &mut s);
        let m = Lasso::new(0.01);
        bench("lasso", n, r20, w3, || {
            m.fit(&x, &y).unwrap();
        });
    }

    // ── ElasticNet ───────────────────────────────────────────────────────────
    {
        let mut s = 5u64;
        let b = [1., 2., 3., 4.];
        let x = mat(&mut s, n, 4);
        let y = linear_y(&x, &b, 1.0, &mut s);
        let m = ElasticNet::new(0.01, 0.5);
        bench("elastic_net", n, r20, w3, || {
            m.fit(&x, &y).unwrap();
        });
    }

    // ── QuantileRegression ───────────────────────────────────────────────────
    {
        let mut s = 6u64;
        let b = [1., 2., 3.];
        let x = mat(&mut s, n, 3);
        let y = linear_y(&x, &b, 1.0, &mut s);
        let m = QuantileRegression::new(0.5);
        bench("quantile_reg", n, r20, w3, || {
            m.fit(&x, &y).unwrap();
        });
    }

    // ── RollingOls ───────────────────────────────────────────────────────────
    {
        let mut s = 7u64;
        let b = [1., 2.];
        let x = mat(&mut s, n, 2);
        let y = linear_y(&x, &b, 1.0, &mut s);
        let m = RollingOls::new(200);
        bench("rolling_ols", n, r10, w3, || {
            m.fit(&x, &y).unwrap();
        });
    }

    // ── RecursiveOls ─────────────────────────────────────────────────────────
    {
        let mut s = 8u64;
        let b = [1., 2.];
        let x = mat(&mut s, n, 2);
        let y = linear_y(&x, &b, 1.0, &mut s);
        let m = RecursiveOls::new();
        bench("recursive_ols", n, r10, w3, || {
            m.fit(&x, &y).unwrap();
        });
    }

    // ── FGLS (Cochrane-Orcutt) ───────────────────────────────────────────────
    {
        let mut s = 9u64;
        let b = [1., 2., 3.];
        let x = mat(&mut s, n, 3);
        let y = linear_y(&x, &b, 1.0, &mut s);
        let m = Fgls::new();
        bench("fgls", n, r10, w3, || {
            m.fit(&x, &y).unwrap();
        });
    }

    // ── RobustLinearModel ────────────────────────────────────────────────────
    {
        let mut s = 10u64;
        let b = [1., 2., 3.];
        let x = mat(&mut s, n, 3);
        let y = linear_y(&x, &b, 1.0, &mut s);
        let m = RobustLinearModel::new();
        bench("robust_lm", n, r10, w3, || {
            m.fit(&x, &y).unwrap();
        });
    }

    // ── Logistic GLM ─────────────────────────────────────────────────────────
    {
        let mut s = 11u64;
        let b = [0.5, -1., 1.5, -0.5];
        let x = mat(&mut s, n, 4);
        let y = binary_y(&x, &b, &mut s);
        let m = Logistic::new();
        bench("logistic", n, r20, w3, || {
            m.fit(&x, &y).unwrap();
        });
    }

    // ── Poisson GLM ──────────────────────────────────────────────────────────
    {
        let mut s = 12u64;
        let b = [0.3, 0.5, -0.2, 0.4];
        let x = mat(&mut s, n, 4);
        let y = count_y(&x, &b, 1.0, &mut s);
        let m = Poisson::new();
        bench("poisson", n, r20, w3, || {
            m.fit(&x, &y).unwrap();
        });
    }

    // ── Gamma GLM ────────────────────────────────────────────────────────────
    {
        let mut s = 13u64;
        let b = [0.3, 0.5, -0.2];
        let x = mat(&mut s, n, 3);
        let y: Vec<f64> = x
            .iter()
            .map(|r| {
                let eta: f64 = r.iter().zip(b.iter()).map(|(xi, bi)| xi * bi).sum::<f64>() + 1.0;
                let mu = eta.exp();
                mu * (1.0 + 0.1 * norm(&mut s).abs())
            })
            .collect();
        let m = Gamma::new().with_link(GammaLink::Log);
        bench("gamma_glm", n, r20, w3, || {
            m.fit(&x, &y).unwrap();
        });
    }

    // ── Probit ───────────────────────────────────────────────────────────────
    {
        let mut s = 14u64;
        let b = [0.8, -0.5, 1.0];
        let x = mat(&mut s, n, 3);
        let y = binary_y(&x, &b, &mut s);
        let m = Probit::new();
        bench("probit", n, r20, w3, || {
            m.fit(&x, &y).unwrap();
        });
    }

    // ── NegativeBinomial ─────────────────────────────────────────────────────
    {
        let mut s = 15u64;
        let b = [0.3, 0.5, -0.2];
        let x = mat(&mut s, n, 3);
        let y = count_y(&x, &b, 1.0, &mut s);
        let m = NegativeBinomial::new();
        bench("neg_binomial", n, r10, w1, || {
            m.fit(&x, &y).unwrap();
        });
    }

    // ── MultinomialLogit ─────────────────────────────────────────────────────
    {
        let mut s = 16u64;
        let x = mat(&mut s, n, 3);
        let y: Vec<usize> = x
            .iter()
            .map(|r| {
                let s0: f64 = 0.0;
                let s1: f64 = r[0] * 0.5 - r[1] * 0.3;
                let s2: f64 = -r[0] * 0.2 + r[2] * 0.8;
                let denom = s0.exp() + s1.exp() + s2.exp();
                let p = [s0.exp() / denom, s1.exp() / denom, s2.exp() / denom];
                let u = lcg(&mut s);
                if u < p[0] {
                    0
                } else if u < p[0] + p[1] {
                    1
                } else {
                    2
                }
            })
            .collect();
        let m = MultinomialLogit::new();
        bench("mnlogit", n, r10, w1, || {
            m.fit(&x, &y).unwrap();
        });
    }

    // ── OrderedLogit ─────────────────────────────────────────────────────────
    {
        let mut s = 17u64;
        let x = mat(&mut s, n, 2);
        let y: Vec<usize> = x
            .iter()
            .map(|r| {
                let eta = r[0] * 0.8 - r[1] * 0.4 + 0.3 * norm(&mut s);
                if eta < -0.5 {
                    0
                } else if eta < 0.5 {
                    1
                } else {
                    2
                }
            })
            .collect();
        let m = OrderedLogit::new();
        bench("ordered_logit", n, r5, w1, || {
            m.fit(&x, &y).unwrap();
        });
    }

    // ── ZeroInflatedPoisson ──────────────────────────────────────────────────
    {
        let mut s = 18u64;
        let b = [0.5, -0.3];
        let x = mat(&mut s, n, 2);
        let y: Vec<f64> = x
            .iter()
            .map(|r| {
                if lcg(&mut s) < 0.2 {
                    0.0
                } else {
                    let mu =
                        (r.iter().zip(b.iter()).map(|(xi, bi)| xi * bi).sum::<f64>() + 1.0).exp();
                    count_y(std::slice::from_ref(r), &[1.0], mu.ln(), &mut {
                        lcg(&mut s) as u64
                    })[0]
                }
            })
            .collect();
        let infl_x: Vec<Vec<f64>> = vec![vec![]; n];
        let m = ZeroInflatedPoisson::new();
        bench("zip", n, r5, w1, || {
            m.fit(&x, &y, &infl_x).unwrap();
        });
    }

    // ── GEE ──────────────────────────────────────────────────────────────────
    {
        let mut s = 19u64;
        let b = [0.5, -0.3, 0.4];
        let x = mat(&mut s, n, 3);
        let y = count_y(&x, &b, 0.5, &mut s);
        let clusters: Vec<usize> = (0..n).map(|i| i / 10).collect();
        let m =
            Gee::new(GeeFamily::Poisson).with_working_correlation(WorkingCorrelation::Exchangeable);
        bench("gee", n, r5, w1, || {
            m.fit(&x, &y, &clusters).unwrap();
        });
    }

    // ── MixedLinearModel ─────────────────────────────────────────────────────
    {
        let mut s = 20u64;
        let b = [1.0, 2.0];
        let x = mat(&mut s, n, 2);
        let y = linear_y(&x, &b, 1.0, &mut s);
        let groups: Vec<usize> = (0..n).map(|i| i / 20).collect();
        let m = MixedLinearModel::new();
        bench("mixed_lm", n, r5, w1, || {
            m.fit_random_intercept(&x, &y, &groups).unwrap();
        });
    }

    // ── KaplanMeier ──────────────────────────────────────────────────────────
    {
        let mut s = 21u64;
        let times: Vec<f64> = (0..n).map(|_| -lcg(&mut s).max(1e-15).ln()).collect();
        let events: Vec<usize> = (0..n)
            .map(|_| if lcg(&mut s) < 0.8 { 1 } else { 0 })
            .collect();
        let m = KaplanMeier::new();
        bench("kaplan_meier", n, r20, w3, || {
            m.fit(&times, &events).unwrap();
        });
    }

    // ── CoxPH ────────────────────────────────────────────────────────────────
    {
        let mut s = 22u64;
        let b = [0.5, -0.3, 0.8];
        let x = mat(&mut s, n, 3);
        let (times, events): (Vec<f64>, Vec<usize>) = x
            .iter()
            .map(|r| {
                let lam = (r.iter().zip(b.iter()).map(|(xi, bi)| xi * bi).sum::<f64>()).exp();
                let t = -lcg(&mut s).max(1e-15).ln() / lam;
                let e = if lcg(&mut s) < 0.8 { 1 } else { 0 };
                (t, e)
            })
            .unzip();
        let m = CoxPh::new();
        bench("cox_ph", n, r5, w1, || {
            m.fit(&times, &events, &x).unwrap();
        });
    }

    // ── t-test (two-sample) ──────────────────────────────────────────────────
    {
        let mut s = 23u64;
        let a: Vec<f64> = (0..n).map(|_| norm(&mut s)).collect();
        let b: Vec<f64> = (0..n).map(|_| norm(&mut s) + 0.2).collect();
        bench("ttest_2samp", n, r20, w3, || {
            ttest::two_sample(&a, &b).unwrap();
        });
    }

    // ── Mann-Whitney U ───────────────────────────────────────────────────────
    {
        let mut s = 24u64;
        let a: Vec<f64> = (0..1000).map(|_| norm(&mut s)).collect();
        let b: Vec<f64> = (0..1000).map(|_| norm(&mut s) + 0.2).collect();
        bench("mann_whitney", 1000, r20, w3, || {
            nonparametric::mann_whitney(&a, &b).unwrap();
        });
    }
}
