"""
hill.py
=======
Core implementation of the Hill tail-index estimator and the
Weissman tail-VaR, plus helper utilities shared by every other script.

Dependencies: numpy, pandas only — no scipy required.

References
----------
Hill (1975), "A simple general approach to inference about the tail of a
distribution", Ann. Statist. 3(5), 1163–1174.

Weissman (1978), "Estimation of parameters and large quantiles based on
the k largest observations", JASA 73(364), 812–815.

Kupiec (1995), "Techniques for verifying the accuracy of risk measurement
models", Journal of Derivatives 3(2), 73–84.
"""

from __future__ import annotations
import math as _math
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Pure-numpy replacements for the two scipy.stats functions we need
# ─────────────────────────────────────────────────────────────────────────────

def _norm_ppf(p: float) -> float:
    """
    Inverse standard-normal CDF (quantile function) via the rational
    approximation of Peter Acklam (max absolute error < 1.15e-9).
    https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/
    """
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00]
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = _math.sqrt(-2.0 * _math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = _math.sqrt(-2.0 * _math.log(1.0 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def _norm_sf(x: float) -> float:
    """Survival function of the standard normal: P(Z > x) = 0.5 * erfc(x/√2)."""
    return 0.5 * _math.erfc(x / _math.sqrt(2.0))


def _chi2_sf_df1(x: float) -> float:
    """
    Survival function of chi-squared(df=1) at x.
    Exact identity: chi2.sf(x, 1) = 2 * norm.sf(sqrt(x))  for x ≥ 0.
    """
    if x < 0:
        return 1.0
    return 2.0 * _norm_sf(_math.sqrt(x))

norm_ppf = _norm_ppf
chi2_sf_df1 = _chi2_sf_df1


# ─────────────────────────────────────────────────────────────────────────────
# Return / loss helpers
# ─────────────────────────────────────────────────────────────────────────────

def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log-returns, dropping the leading NaN."""
    return np.log(prices / prices.shift(1)).dropna()


def losses(returns: pd.Series) -> pd.Series:
    """Positive losses: L_t = max(0, -r_t).  Zeros are kept for sizing."""
    return (-returns).clip(lower=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Hill estimator  (vectorised over all k in one pass)
# ─────────────────────────────────────────────────────────────────────────────

def hill_all_k(
    loss_series: pd.Series | np.ndarray,
    k_min: int = 10,
    k_max_frac: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Hill's tail-exponent estimate α̂(k) for every k in [k_min, k_max].

    Parameters
    ----------
    loss_series : losses; zeros and negatives are dropped before sorting.
    k_min       : smallest k to evaluate (default 10 — below this the
                  variance is enormous).
    k_max_frac  : upper bound on k as a fraction of the positive losses
                  (default 25 %).

    Returns
    -------
    ks     : 1-D integer array of k values.
    alphas : corresponding Hill estimates α̂(k); NaN where the k largest
             losses are all tied with X_(k+1) (the estimate is undefined).

    Notes
    -----
    For the k largest order statistics X_(1) ≥ X_(2) ≥ … ≥ X_(n):

        α̂(k) = k / Σ_{i=1}^{k} ln( X_(i) / X_(k+1) )

    With S_k = Σ_{i≤k} ln X_(i) (a cumulative sum), the denominator is
    S_k − k·ln X_(k+1), so every k is computed at once without a loop.
    """
    x = np.asarray(loss_series, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    x = np.sort(x)[::-1]          # descending: X_(1), X_(2), …

    n = len(x)
    k_max = min(max(k_min, int(n * k_max_frac)), n - 1)   # X_(k+1) must exist
    if k_max < k_min:
        raise ValueError(
            f"Need at least {k_min + 1} positive losses for the Hill estimator, got {n}."
        )

    ks = np.arange(k_min, k_max + 1, dtype=int)
    log_x = np.log(x)
    cumsum = np.cumsum(log_x)                 # cumsum[k-1] = S_k
    denom = cumsum[ks - 1] - ks * log_x[ks]   # log_x[k] = ln X_(k+1)
    with np.errstate(divide="ignore", invalid="ignore"):
        alphas = np.where(denom > 0, ks / denom, np.nan)
    return ks, alphas


def select_k(
    ks: np.ndarray,
    alphas: np.ndarray,
    n_obs: int,
) -> tuple[int, float]:
    """
    Choose k by the square-root rule, k* = round(√n), where n is the number
    of observations (all days, including zero-loss and gain days).

    Why not "find the plateau"?  On fat-tailed but not exactly Pareto data
    (e.g. Student-t, and real returns) the Hill plot keeps drifting down as
    k grows, so the flattest stretch sits at large k where the estimate is
    badly biased.  In a 150-replication simulation on Student-t samples of
    5,700 days (reproduce with `python src/k_selection_study.py`), the old
    minimum-rolling-std rule returned α̂ ≈ 2.3 for a true α of 4.  √n had
    the lowest RMSE of the rules tried at α = 3, 4 and 5 and was within 0.03
    of the best (2 % of n) at α = 2.5.  Hill still reads somewhat low on
    Student-t data (≈ 3.45 for α = 4) because the t tail is only
    approximately Pareto; that bias is inherent to the estimator.

    If α̂ is undefined (NaN) at √n, the nearest k with a finite estimate is used.

    Returns
    -------
    best_k    : chosen number of order statistics.
    best_alpha: Hill estimate at best_k.
    """
    ks = np.asarray(ks)
    alphas = np.asarray(alphas, dtype=float)
    finite = np.isfinite(alphas)
    if not finite.any():
        raise ValueError("No finite Hill estimates to choose from.")
    target = int(round(np.sqrt(n_obs)))
    candidates = np.where(finite)[0]
    idx = candidates[np.argmin(np.abs(ks[candidates] - target))]
    return int(ks[idx]), float(alphas[idx])


def alpha_at(ks: np.ndarray, alphas: np.ndarray, k: int) -> float:
    """Hill estimate at the k in `ks` closest to the requested k."""
    idx = int(np.argmin(np.abs(np.asarray(ks) - k)))
    return float(alphas[idx])


def hill_ci(alpha: float, k: int) -> float:
    """Approximate ±1σ half-width of the Hill estimate: α / √k."""
    return alpha / np.sqrt(k)


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian VaR (rolling, no look-ahead)
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_var(
    returns: pd.Series,
    confidence: float = 0.99,
    window: int = 252,
) -> pd.Series:
    """
    Rolling Gaussian VaR (expressed as a *positive* loss fraction).

    Only uses data available strictly before day t, so there is no look-ahead.
    The first `window` observations return NaN.
    """
    z = _norm_ppf(confidence)   # pure-numpy, no scipy
    # Shift(1) so we never use today's return to forecast today's VaR
    mu  = returns.shift(1).rolling(window).mean()
    sig = returns.shift(1).rolling(window).std()
    # VaR = -(μ - z·σ)  expressed as a positive number
    var = -(mu - z * sig)
    return var.rename("gaussian_var")


# ─────────────────────────────────────────────────────────────────────────────
# Weissman tail-VaR  (full-sample, for comparison table)
# ─────────────────────────────────────────────────────────────────────────────

def weissman_var(
    loss_series: pd.Series | np.ndarray,
    alpha_hat: float,
    k: int,
    target_prob: float = 0.01,
) -> float:
    """
    Weissman (1978) estimator of the quantile at tail probability `target_prob`.

        VaR_p = X_(k+1) · ( k / (n·p) )^(1/α̂)

    where X_(k+1) is the (k+1)-th largest loss and n is the number of
    observations the probability p refers to.  p is a per-day probability,
    so n counts *every* day, including zero-loss and gain days: pass the full
    loss series (zeros included), not just the positive losses.
    """
    x = np.asarray(loss_series, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)                     # all days
    pos = np.sort(x[x > 0])[::-1]
    if not 0 < k < len(pos):
        raise ValueError(f"k={k} must be between 1 and {len(pos) - 1} (number of positive losses − 1).")
    if not (np.isfinite(alpha_hat) and alpha_hat > 0):
        raise ValueError(f"alpha_hat must be positive and finite, got {alpha_hat}.")
    if not 0 < target_prob <= k / n:
        raise ValueError(f"target_prob must be in (0, k/n] = (0, {k / n:.4g}]; Weissman only extrapolates beyond X_(k+1).")
    threshold = pos[k]             # X_(k+1)  (0-indexed → index k)
    factor = (k / (n * target_prob)) ** (1.0 / alpha_hat)
    return float(threshold * factor)


# ─────────────────────────────────────────────────────────────────────────────
# Breach analysis
# ─────────────────────────────────────────────────────────────────────────────

def find_breaches(
    loss_series: pd.Series,
    var_series: pd.Series,
    cluster_window: int = 10,
) -> pd.DataFrame:
    """
    Return a DataFrame of days where actual loss exceeded VaR.

    Columns
    -------
    date, loss, var, excess, clustered (bool — another breach within the
    prior `cluster_window` observations).
    """
    df = pd.DataFrame({"loss": loss_series, "var": var_series}).dropna()
    breaches = df[df["loss"] > df["var"]].copy()
    breaches["excess"] = breaches["loss"] - breaches["var"]

    # Clustering flag: another breach within the previous `cluster_window`
    # observations (rows of the series, i.e. trading days for equities).
    pos = df.index.get_indexer(breaches.index)
    gaps = np.diff(pos, prepend=-(cluster_window + 1))
    breaches["clustered"] = gaps <= cluster_window

    return breaches[["loss", "var", "excess", "clustered"]].rename_axis("date")


def kupiec_test(n_obs: int, n_breaches: int, confidence: float = 0.99) -> dict:
    """
    Kupiec (1995) proportion-of-failures (POF) likelihood-ratio test.

    H₀: breach probability = 1 - confidence.
    Returns LR statistic and p-value (χ²(1)).
    Uses the exact identity chi2.sf(x, df=1) = 2·norm.sf(√x) — no scipy.
    """
    if n_obs <= 0 or not 0 <= n_breaches <= n_obs:
        raise ValueError(f"Need 0 ≤ n_breaches ≤ n_obs and n_obs > 0, got {n_breaches}, {n_obs}.")
    p0 = 1.0 - confidence
    x, n = n_breaches, n_obs
    p_hat = x / n
    # Log-likelihoods; 0·log(0) is taken as 0, so zero breaches (or all
    # breaches) still give a valid statistic instead of NaN.
    def _xlogy(a, b):
        return a * np.log(b) if a > 0 else 0.0
    ll_null = _xlogy(x, p0) + _xlogy(n - x, 1 - p0)
    ll_alt = _xlogy(x, p_hat) + _xlogy(n - x, 1 - p_hat)
    lr = max(0.0, -2 * (ll_null - ll_alt))
    p_value = _chi2_sf_df1(float(lr))   # exact for df=1, no scipy needed
    return {
        "lr": float(lr),
        "p_value": float(p_value),
        "reject_h0": bool(p_value < 0.05),
    }
