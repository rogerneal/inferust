//! Fast CI smoke benchmark — OLS, Logistic, Poisson only (~1 s).

use std::hint::black_box;
use std::time::Instant;

use inferust::glm::{Logistic, Poisson};
use inferust::regression::Ols;

fn main() {
    let n = 5_000_usize;
    let mut s = 42u64;
    let x: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..4).map(|_| norm(&mut s)).collect())
        .collect();
    let y_lin: Vec<f64> = x
        .iter()
        .map(|r| r.iter().sum::<f64>() + norm(&mut s))
        .collect();
    let y_bin: Vec<f64> = x
        .iter()
        .map(|r| {
            let p = 1.0 / (1.0 + (-r[0]).exp());
            if lcg(&mut s) < p {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let y_cnt: Vec<f64> = x
        .iter()
        .map(|r| {
            let mu = (r[0] + 1.0).exp();
            poisson_sample(mu, &mut s)
        })
        .collect();

    bench("ols", || {
        Ols::new().fit(&x, &y_lin).unwrap();
    });
    bench("logistic", || {
        Logistic::new().fit(&x, &y_bin).unwrap();
    });
    bench("poisson", || {
        Poisson::new().fit(&x, &y_cnt).unwrap();
    });
}

fn bench<F: Fn()>(label: &str, f: F) {
    for _ in 0..2 {
        black_box(f());
    }
    let t0 = Instant::now();
    for _ in 0..5 {
        black_box(f());
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / 5.0;
    println!("smoke estimator={label} median_ms={ms:.3}");
}

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

fn poisson_sample(mu: f64, s: &mut u64) -> f64 {
    let mut k = 0u64;
    let mut p = (-mu).exp();
    let mut cdf = p;
    let u = lcg(s);
    while cdf < u && k < 100 {
        k += 1;
        p *= mu / k as f64;
        cdf += p;
    }
    k as f64
}
