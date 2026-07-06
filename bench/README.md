# Benchmarks

Cross-language performance comparison between `inferust` and Python `statsmodels`
on deterministic synthetic data (10,000 rows by default).

## Quick start

```bash
# Rust -  full estimator suite
cargo run --release --example bench_all

# Rust -  OLS only (configurable rows/features/solver)
cargo run --release --example bench_ols -- --rows 10000 --features 8 --repeats 10

# Python counterpart (requires numpy, scipy, statsmodels, lifelines, pandas)
python3 scripts/bench_all_statsmodels.py
python3 scripts/bench_statsmodels.py --rows 10000 --features 8 --repeats 10
```

Each script prints one `key=value` line per estimator with `min_ms`, `median_ms`,
and `mean_ms` so results can be compared side by side.

## Docker (reproducible environment)

Build from the repository root:

```bash
docker build -f bench/Dockerfile.rust -t inferust-bench-rust .
docker build -f bench/Dockerfile.python -t inferust-bench-python .
docker run --rm inferust-bench-rust
docker run --rm inferust-bench-python
```

Benchmark numbers vary by machine and BLAS/LAPACK configuration. Treat these as
local smoke tests, not universal performance claims.
