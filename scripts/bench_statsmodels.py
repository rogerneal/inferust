#!/usr/bin/env python3
import argparse
import time

import numpy as np
import statsmodels.api as sm


def dataset(n: int, p: int):
    i = np.arange(n, dtype=np.float64)[:, None]
    j = np.arange(p, dtype=np.float64)[None, :]
    x = np.sin((i + 1.0) * (j + 1.0) * 0.001)
    x += np.cos((i + j + 3.0) * 0.017)
    x += (np.mod(i, 97.0)) * 0.0001
    beta = (np.arange(p, dtype=np.float64) + 1.0) / float(p)
    y = 1.5 + x @ beta + np.sin((np.arange(n, dtype=np.float64) + 11.0) * 0.037) * 0.01
    return x, y


def millis(seconds: float) -> float:
    return seconds * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()

    x, y = dataset(args.rows, args.features)
    x_const = sm.add_constant(x, has_constant="add")

    for _ in range(args.warmups):
        sm.OLS(y, x_const).fit()

    timings = []
    checksum = 0.0
    for _ in range(args.repeats):
        started = time.perf_counter()
        result = sm.OLS(y, x_const).fit()
        elapsed = time.perf_counter() - started
        checksum += float(result.params.sum())
        timings.append(elapsed)

    timings.sort()
    print(
        "engine=python-statsmodels "
        f"rows={args.rows} features={args.features} repeats={args.repeats} warmups={args.warmups} "
        f"min_ms={millis(timings[0]):.3f} "
        f"median_ms={millis(timings[len(timings) // 2]):.3f} "
        f"mean_ms={millis(sum(timings) / len(timings)):.3f} "
        f"checksum={checksum:.8f}"
    )


if __name__ == "__main__":
    main()
