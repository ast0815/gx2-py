"""Regression tests for issue #1: gx2inv returned wrong quantiles when the weights
span about two orders of magnitude.

Two causes, both in functions.py:
- gx2cdf used Ruben's series with a fixed N = 1000 terms and discarded the series'
  error bound, so for w_min/w_max ~ 0.005 the cdf it returned was wrong by up to
  0.03 in probability and the quantile built on it was ~30% too small;
- gx2inv returned the root finder's point without checking that it converged.

Run: python test_gx2inv_quantiles.py (no MATLAB needed).
"""
import warnings

import numpy as np
from scipy.integrate import quad
from scipy.stats import ncx2

from functions import gx2inv, gx2cdf

warnings.filterwarnings('ignore')


def sf_imhof(x, w, k):
    """P(sum_i w_i chi2_{k_i} > x) by Imhof's (1961) inversion: an independent reference."""
    def f(u):
        theta = 0.5 * np.sum(k * np.arctan(w * u)) - 0.5 * x * u
        return np.sin(theta) / (u * np.prod((1 + (w * u) ** 2) ** (k / 4)))
    return 0.5 + quad(f, 0, np.inf, limit=500)[0] / np.pi


p = 0.0027  # 3 sigma, upper tail
cases = [
    ([0.5, 1.0, 2.0], [1, 4, 1]),                 # moderate spread: Ruben converges
    ([0.05, 1.0, 2.0], [1, 4, 1]),
    ([0.01, 1.0, 2.0], [1, 4, 1]),                # ~2 orders of magnitude: wrong before
    ([0.01, 0.9, 1.0, 1.25, 2.0], [1, 1, 6, 1, 2]),
]

worst = 0.0
for w, k in cases:
    w, k = np.asarray(w, float), np.asarray(k, float)
    zero = np.zeros_like(w)
    x = float(np.ravel(gx2inv(p, w, k, zero, 0, 0, AbsTol=p / 1000, RelTol=p / 1000, side="upper"))[0])
    sf = sf_imhof(x, w, k)
    rel = abs(sf - p) / p
    worst = max(worst, rel)
    print(f"w={w.tolist()} -> gx2inv x={x:8.3f}, P(>x)={sf:.5f} (target {p})")
    assert rel < 0.02, f"tail probability off by {rel:.1%} for w={w.tolist()}"

# the cdf itself must also be right where Ruben's series has not converged
# (gx2cdf now falls back to Davies' method instead of returning the truncated series)
w = np.array([0.01, 0.9, 1.0, 1.25, 2.0])
k = np.array([1, 1, 6, 1, 2])
zero = np.zeros_like(w)
x = 35.0
cdf = float(np.ravel(gx2cdf(x, w, k, zero, 0, 0))[0])
ref = 1 - sf_imhof(x, w, k)
print(f"cdf({x}) = {cdf:.6f}, Imhof reference {ref:.6f}")
assert abs(cdf - ref) < 1e-3

# equal-weight branch: side must select the tail (it compared the method str.lower
# with a string, so the upper-tail quantile came back for both sides)
w = np.array([2.0, 2.0, 2.0])
k = np.array([1.0, 4.0, 1.0])
zero = np.zeros_like(w)
lo = float(np.ravel(gx2inv(0.01, w, k, zero, 0, 0, side="lower"))[0])
hi = float(np.ravel(gx2inv(0.01, w, k, zero, 0, 0, side="upper"))[0])
ref_lo = 2 * ncx2.ppf(0.01, k.sum(), 0)
ref_hi = 2 * ncx2.isf(0.01, k.sum(), 0)
print(f"equal weights: side=lower {lo:.6f} (ncx2 {ref_lo:.6f}), side=upper {hi:.6f} (ncx2 {ref_hi:.6f})")
assert abs(lo - ref_lo) < 1e-6 and abs(hi - ref_hi) < 1e-6

# a root that does not converge must raise, not come back as the quantile
try:
    gx2inv(1e-30, np.array([0.01, 1.0, 2.0]), np.array([1.0, 4.0, 1.0]), np.zeros(3), 0, 0, side="upper")
except RuntimeError as e:
    print(f"unconverged root raises: {e}")
else:
    raise AssertionError("gx2inv returned an unconverged root without raising")

print(f"worst relative tail error: {worst:.4%} - all gx2inv quantile tests passed")
