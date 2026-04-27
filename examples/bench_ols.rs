use std::env;
use std::hint::black_box;
use std::time::{Duration, Instant};

use inferust::regression::{Ols, OlsSolver};

fn main() {
    let n = arg_usize("--rows", 10_000);
    let p = arg_usize("--features", 8);
    let repeats = arg_usize("--repeats", 10);
    let warmups = arg_usize("--warmups", 2);
    let solver = arg_solver();

    let (x, y) = dataset(n, p);
    let feature_names = (0..p).map(|i| format!("x{}", i + 1)).collect::<Vec<_>>();
    let model = Ols::new()
        .with_feature_names(feature_names)
        .with_solver(solver);

    for _ in 0..warmups {
        black_box(model.fit(&x, &y).expect("warmup fit should succeed"));
    }

    let mut timings = Vec::with_capacity(repeats);
    let mut checksum = 0.0;
    for _ in 0..repeats {
        let started = Instant::now();
        let result = model.fit(&x, &y).expect("benchmark fit should succeed");
        let elapsed = started.elapsed();
        checksum += result.coefficients.iter().sum::<f64>();
        timings.push(elapsed);
    }

    timings.sort_unstable();
    println!(
        "engine=rust-inferust solver={} rows={} features={} repeats={} warmups={} min_ms={:.3} median_ms={:.3} mean_ms={:.3} checksum={:.8}",
        solver_name(solver),
        n,
        p,
        repeats,
        warmups,
        millis(timings[0]),
        millis(timings[timings.len() / 2]),
        mean_ms(&timings),
        checksum
    );
}

fn dataset(n: usize, p: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let beta = (0..p)
        .map(|j| (j as f64 + 1.0) / p as f64)
        .collect::<Vec<_>>();

    for i in 0..n {
        let mut row = Vec::with_capacity(p);
        let mut yi = 1.5;
        for (j, beta_j) in beta.iter().enumerate() {
            let value = ((i + 1) as f64 * (j + 1) as f64 * 0.001).sin()
                + ((i + j + 3) as f64 * 0.017).cos()
                + ((i % 97) as f64) * 0.0001;
            row.push(value);
            yi += beta_j * value;
        }
        yi += ((i + 11) as f64 * 0.037).sin() * 0.01;
        x.push(row);
        y.push(yi);
    }

    (x, y)
}

fn arg_solver() -> OlsSolver {
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = if arg == "--solver" {
            args.next()
                .unwrap_or_else(|| panic!("missing value for --solver"))
        } else if let Some(value) = arg.strip_prefix("--solver=") {
            value.to_string()
        } else {
            continue;
        };

        return match value.as_str() {
            "cholesky" | "fast" => OlsSolver::Cholesky,
            "svd" | "stable" => OlsSolver::Svd,
            _ => panic!("invalid value for --solver: {value}"),
        };
    }
    OlsSolver::Cholesky
}

fn solver_name(solver: OlsSolver) -> &'static str {
    match solver {
        OlsSolver::Cholesky => "cholesky",
        OlsSolver::Svd => "svd",
    }
}

fn arg_usize(flag: &str, default: usize) -> usize {
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == flag {
            return args
                .next()
                .unwrap_or_else(|| panic!("missing value for {flag}"))
                .parse()
                .unwrap_or_else(|_| panic!("invalid value for {flag}"));
        }
        if let Some(value) = arg.strip_prefix(&format!("{flag}=")) {
            return value
                .parse()
                .unwrap_or_else(|_| panic!("invalid value for {flag}"));
        }
    }
    default
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn mean_ms(durations: &[Duration]) -> f64 {
    durations.iter().map(|d| millis(*d)).sum::<f64>() / durations.len() as f64
}
