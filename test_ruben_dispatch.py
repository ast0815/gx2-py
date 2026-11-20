"""Tests for the up-front method choice in gx2cdf.

gx2cdf sends a call straight to Davies when it predicts that Ruben's series,
run to its cap, could not meet the requested tolerance. The prediction has two
claims, both checked here against the exact coefficient recurrence:

1. lambda = 0: the predicted leftover series mass at the cap is within a
   factor 1.4 of the true one (measured: <= 4.8% over spans 10:1 - 310:1);
2. lambda > 0: the prediction is optimistic (never above the truth), so it can
   only waste a capped run, never send a would-be-accepted value to Davies.

On top of that the decision itself is checked with a margin: wherever the TRUE
truncation bound is more than 1.4x clear of the tolerance the predictor must
agree, in both directions.

Run: python test_ruben_dispatch.py (no MATLAB needed).
"""
import time

import numpy as np
from scipy.stats import chi2

from functions import (RUBEN_N, _ruben_leftover_cap, _ruben_worth_trying,
                       gx2cdf_ruben)

ABSTOL = 1e-10
RELTOL = 1e-6


def exact_leftover(w, k, lam, N=2 * RUBEN_N):
    """True leftover(N) from the coefficient recurrence, computed to 2N terms
    so the tail past the cap is exact (tail sums right-to-left)."""
    w = np.atleast_1d(np.asarray(w, dtype=float))
    k = np.atleast_1d(np.asarray(k, dtype=float))
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    beta = 0.90625 * np.min(w)
    n = np.arange(1, N)
    g = np.sum(k * (1 - beta / w) ** n[:, None]
               + beta * n[:, None] * ((1 - beta / w) ** (n[:, None] - 1)) * (lam / w),
               axis=1)
    a = np.empty(N)
    a[0] = np.sqrt(np.exp(-np.sum(lam)) * beta ** np.sum(k) * np.prod(w ** -k))
    gflip = g[::-1]
    for j in range(1, N):
        a[j] = np.dot(gflip[-j:], a[:j]) / (2 * j)
    tail = np.cumsum(a[::-1])[::-1] - a
    return float(tail[RUBEN_N - 1])


def weights(name, span):
    if name in ("pr", "prnc"):          # the weight shape of this PR's issue
        u = np.array([0.01, 0.9, 1.0, 1.25, 2.0]) / 0.01
        return span ** (np.log(u) / np.log(np.max(u)) - 1.0)
    return span ** (-np.linspace(0, 1, 5))   # even geometric spread


FAMILIES = [
    ("pr",    np.array([1, 1, 6, 1, 2]), np.zeros(5)),
    ("prnc",  np.array([1, 1, 6, 1, 2]), np.array([1.0, 2.0, 3.0, 4.0, 5.0])),
    ("even",  np.ones(5),                np.zeros(5)),
    ("evnc",  np.ones(5),                np.array([1.0, 2.0, 3.0, 4.0, 5.0])),
]
SPANS = [10, 50, 100, 200, 310]

ratio_max, ratio_min = 0.0, np.inf
checked = 0
for name, k, lam in FAMILIES:
    for span in SPANS:
        w = weights(name, span)
        exact = exact_leftover(w, k, lam)
        pred = _ruben_leftover_cap(w, k, lam)
        if np.all(lam == 0):
            ratio = pred / exact
            ratio_max = max(ratio_max, ratio)
            ratio_min = min(ratio_min, ratio)
            assert 0.7 <= ratio <= 1.4, (
                f"{name} span{span}: predicted/true leftover = {ratio:.3f}")
        else:
            assert pred <= exact, (
                f"{name} span{span}: optimistic prediction exceeded truth "
                f"({pred:.3e} > {exact:.3e})")
        checked += 1
        # the decision itself, with a 1.4x margin around the tolerance
        beta = 0.90625 * np.min(w)
        M = float(np.sum(k))
        mu = float(np.sum(w * k))
        sd = float(np.sqrt(2.0 * np.sum(w ** 2 * k)))
        for x in (mu, mu - 2 * sd, mu + 2 * sd, mu + 5 * sd):
            F = float(chi2.cdf(x / beta, M + 2 * RUBEN_N))
            bound = exact * F
            decision = _ruben_worth_trying(x, w, k, lam, 0.0, AbsTol=ABSTOL)
            if bound <= ABSTOL / 1.4:
                assert decision, (f"{name} span{span} x={x:.3f}: true bound "
                                  f"{bound:.3e} clear of tolerance but the "
                                  "estimate said Davies")
            elif bound >= ABSTOL / 0.7:
                assert not decision, (f"{name} span{span} x={x:.3f}: true bound "
                                      f"{bound:.3e} exceeds tolerance but the "
                                      "estimate said Ruben")
print(f"leftover prediction vs exact recurrence on {checked} configs: "
      f"lambda=0 ratios in [{ratio_min:.3f}, {ratio_max:.3f}], "
      "lambda>0 never above the truth")

# the estimate must be cheap: it runs on every gx2cdf call
t0 = time.perf_counter()
for i in range(200):
    _ruben_worth_trying(10.0 + i * 0.01,
                        np.array([0.01, 0.9, 1.0, 1.25, 2.0]),
                        np.array([1, 1, 6, 1, 2]), np.zeros(5), 0.0)
per_call = (time.perf_counter() - t0) / 200 * 1e6
print(f"estimate cost: {per_call:.1f} us per call")
assert per_call < 200, f"estimate took {per_call:.1f} us, too slow to be free"

# sign flip: the bound is evaluated in gx2cdf_ruben's flipped frame
w = weights("pr", 200)
k = np.array([1, 1, 6, 1, 2])
lam = np.zeros(5)
for x in (2.0, 15.0, 60.0):
    assert (_ruben_worth_trying(x, w, k, lam, 0.0)
            == _ruben_worth_trying(-x, -w, k, lam, 0.0))
print("negative-weight sign flip consistent")

# the prediction must line up with gx2cdf_ruben's own errbnd at the cap:
# predicted accept -> the run is kept; predicted reject -> its bound at the
# cap really does exceed AbsTol (the conservative half of the acceptance test)
for x, expect_keep in ((6.0, True), (200.0, False)):
    decision = _ruben_worth_trying(x, w, k, lam, 0.0, AbsTol=ABSTOL)
    p, errbnd = gx2cdf_ruben(x, w, k, lam, 0.0, N=RUBEN_N, AbsTol=ABSTOL,
                             RelTol=RELTOL)
    kept_at_abstol = bool(np.all(errbnd <= ABSTOL))
    if expect_keep:
        assert decision and kept_at_abstol, (
            f"x={x}: predicted {decision}, ruben errbnd {errbnd:.3e} "
            f"< AbsTol: {kept_at_abstol}")
    else:
        assert not decision and not kept_at_abstol, (
            f"x={x}: predicted {decision}, ruben errbnd {errbnd:.3e} "
            f"< AbsTol: {kept_at_abstol}")
print("predictions agree with gx2cdf_ruben's own errbnd test")
print("all gx2cdf dispatch tests passed")
