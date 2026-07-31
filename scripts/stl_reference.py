"""Faithful transcription of Cleveland's Fortran STL (netlib `stl.f`).

Used only as a development oracle: it is validated against
`statsmodels.tsa.seasonal.STL` and then compared step by step against the Rust
implementation in `src/seasonal.rs`. Not part of the shipped library.

Indexing follows the Fortran source (1-based) wherever that keeps the
transcription readable; arrays are padded so index 0 is unused.
"""

from __future__ import annotations

import numpy as np


def stlest(y, n, length, ideg, xs, nleft, nright, rw, userw):
    """Fortran `stlest`: local regression estimate at position `xs`.

    `y`, `rw` are 1-based (index 0 unused). Returns `(ys, ok)`.
    """
    w = np.zeros(n + 2)
    rng = float(n) - 1.0
    h = max(xs - float(nleft), float(nright) - xs)
    if length > n:
        h += float((length - n) // 2)
    h9 = 0.999 * h
    h1 = 0.001 * h
    a = 0.0
    for j in range(nleft, nright + 1):
        w[j] = 0.0
        r = abs(float(j) - xs)
        if r <= h9:
            if r <= h1:
                w[j] = 1.0
            else:
                w[j] = (1.0 - (r / h) ** 3) ** 3
            if userw:
                w[j] = rw[j] * w[j]
            a += w[j]
    if a <= 0.0:
        return 0.0, False
    for j in range(nleft, nright + 1):
        w[j] /= a
    if h > 0.0 and ideg > 0:
        a = 0.0
        for j in range(nleft, nright + 1):
            a += w[j] * float(j)
        b = xs - a
        c = 0.0
        for j in range(nleft, nright + 1):
            c += w[j] * (float(j) - a) ** 2
        if np.sqrt(c) > 0.001 * rng:
            b /= c
            for j in range(nleft, nright + 1):
                w[j] = w[j] * (b * (float(j) - a) + 1.0)
    ys = 0.0
    for j in range(nleft, nright + 1):
        ys += w[j] * y[j]
    return ys, True


def stless(y, n, length, ideg, njump, rw, userw):
    """Fortran `stless`: loess smooth of `y` at every position 1..n."""
    ys = np.zeros(n + 2)
    if n < 2:
        ys[1] = y[1]
        return ys
    newnj = min(njump, n - 1)
    if length >= n:
        nleft, nright = 1, n
        for i in range(1, n + 1, newnj):
            val, ok = stlest(y, n, length, ideg, float(i), nleft, nright, rw, userw)
            ys[i] = val if ok else y[i]
    elif newnj == 1:
        nsh = (length + 1) // 2
        nleft, nright = 1, length
        for i in range(1, n + 1):
            if i > nsh and nright != n:
                nleft += 1
                nright += 1
            val, ok = stlest(y, n, length, ideg, float(i), nleft, nright, rw, userw)
            ys[i] = val if ok else y[i]
    else:
        raise NotImplementedError("njump > 1 not needed for the default configuration")
    return ys


def stlma(x, n, length):
    """Fortran `stlma`: moving average of length `length`; output length n-length+1."""
    out = np.zeros(n + 2)
    newn = n - length + 1
    flen = float(length)
    v = sum(x[1 : length + 1])
    out[1] = v / flen
    if newn > 1:
        k = length
        m = 0
        for j in range(2, newn + 1):
            k += 1
            m += 1
            v = v - x[m] + x[k]
            out[j] = v / flen
    return out


def stlfts(x, n, np_):
    """Fortran `stlfts`: MA(np) -> MA(np) -> MA(3)."""
    t1 = stlma(x, n, np_)
    t2 = stlma(t1, n - np_ + 1, np_)
    return stlma(t2, n - 2 * np_ + 2, 3)


def stlss(y, n, np_, ns, isdeg, nsjump, rw, userw):
    """Fortran `stlss`: cycle-subseries smoothing, extended one period each side."""
    season = np.zeros(n + 2 * np_ + 2)
    for j in range(1, np_ + 1):
        k = (n - j) // np_ + 1
        work1 = np.zeros(k + 2)
        work3 = np.ones(k + 2)
        for i in range(1, k + 1):
            work1[i] = y[(i - 1) * np_ + j]
        if userw:
            for i in range(1, k + 1):
                work3[i] = rw[(i - 1) * np_ + j]
        smoothed = stless(work1, k, ns, isdeg, nsjump, work3, userw)
        work2 = np.zeros(k + 3)
        for i in range(1, k + 1):
            work2[i + 1] = smoothed[i]
        nright = min(ns, k)
        val, ok = stlest(work1, k, ns, isdeg, 0.0, 1, nright, work3, userw)
        work2[1] = val if ok else work2[2]
        nleft = max(1, k - ns + 1)
        val, ok = stlest(work1, k, ns, isdeg, float(k + 1), nleft, k, work3, userw)
        work2[k + 2] = val if ok else work2[k + 1]
        for m in range(1, k + 3):
            season[(m - 1) * np_ + j] = work2[m]
    return season


def stl(y_in, np_, ns=7, nt=None, nl=None, isdeg=1, itdeg=1, ildeg=1, robust=False,
        ni=None, no=None, nsjump=1, ntjump=1, nljump=1):
    """Fortran `stlstp` driver with the standard default window heuristics."""
    n = len(y_in)
    y = np.zeros(n + 2)
    y[1 : n + 1] = y_in

    if nt is None:
        nt = _next_odd(int(np.ceil(1.5 * np_ / (1.0 - 1.5 / ns))))
    if nl is None:
        nl = _next_odd(np_)
    if ni is None:
        ni = 1 if robust else 2
    if no is None:
        no = 15 if robust else 0

    trend = np.zeros(n + 2)
    season = np.zeros(n + 2)
    rw = np.ones(n + 2)

    for outer in range(no + 1):
        userw = outer > 0
        for _ in range(ni):
            # Step 1: detrend.
            work1 = np.zeros(n + 2)
            for i in range(1, n + 1):
                work1[i] = y[i] - trend[i]
            # Step 2: cycle-subseries smoothing (length n + 2*np).
            # `stlss` already writes 1-based positions 1..n+2*np.
            ext_1 = stlss(work1, n, np_, ns, isdeg, nsjump, rw, userw)
            # Step 3: low-pass filter the extended subseries, then loess.
            lp = stlfts(ext_1, n + 2 * np_, np_)
            low = stless(lp, n, nl, ildeg, nljump, rw, False)
            # Step 4: seasonal = extended subseries (centre block) - low pass.
            for i in range(1, n + 1):
                season[i] = ext_1[np_ + i] - low[i]
            # Steps 5-6: deseasonalize and smooth the trend.
            for i in range(1, n + 1):
                work1[i] = y[i] - season[i]
            trend = stless(work1, n, nt, itdeg, ntjump, rw, userw)
        if outer < no:
            resid = np.array([y[i] - trend[i] - season[i] for i in range(1, n + 1)])
            rw[1 : n + 1] = _bisquare_weights(resid)

    return (
        trend[1 : n + 1].copy(),
        season[1 : n + 1].copy(),
        np.array([y[i] - trend[i] - season[i] for i in range(1, n + 1)]),
        rw[1 : n + 1].copy(),
    )


def _bisquare_weights(resid):
    a = np.abs(resid)
    mid = np.sort(a)
    n = len(a)
    if n % 2 == 0:
        med = 0.5 * (mid[n // 2 - 1] + mid[n // 2])
    else:
        med = mid[n // 2]
    h = 6.0 * med
    h9 = 0.999 * h
    h1 = 0.001 * h
    w = np.empty(n)
    for i, r in enumerate(a):
        if r <= h1:
            w[i] = 1.0
        elif r >= h9:
            w[i] = 0.0
        else:
            w[i] = (1.0 - (r / h) ** 2) ** 2
    return w


def _next_odd(v):
    v = int(round(v))
    return v if v % 2 == 1 else v + 1


if __name__ == "__main__":
    from statsmodels.tsa.seasonal import STL

    rng = np.random.default_rng(0)
    for n, period in [(144, 12), (120, 12), (72, 6)]:
        t = np.arange(n, dtype=float)
        y = 30.0 + 0.15 * t + 6.0 * np.sin(2 * np.pi * (t % period) / period) + rng.standard_normal(n)
        for robust in (False, True):
            tr, se, re, _ = stl(y, period, robust=robust)
            ref = STL(y, period=period, robust=robust).fit()
            print(
                f"n={n} period={period} robust={robust}: "
                f"trend={np.max(np.abs(tr - ref.trend)):.3e} "
                f"seasonal={np.max(np.abs(se - ref.seasonal)):.3e}"
            )
