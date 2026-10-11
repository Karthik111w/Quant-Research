"""
tests/test_hill.py
==================
Unit tests for the Hill estimator and related utilities.

Methodology
-----------
1.  Pareto(α=3): α̂ at the selected k recovers 3.
2.  Student-t(df=3, 4): averaged over 20 samples the size of the SPY history,
    α̂ lands near the true df.  These tests fail for the old "plateau" k rule,
    which returned ≈2.1 and ≈2.3.
3.  hill_all_k: vectorised result equals a direct loop; edge cases raise.
4.  select_k: √n rule, NaN handling.
5.  Weissman VaR: monotone in confidence, uses all days for n, and recovers a
    known quantile of a zero-inflated Pareto loss distribution.
6.  Kupiec: calibrated not rejected, miscalibrated rejected, zero breaches
    rejected (previously returned NaN), matches the textbook LR statistic.
7.  find_breaches clustering counts observations, not calendar days.
8.  log_returns / losses helpers and hill_ci.

Run with:
    pytest tests/test_hill.py -v
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
from hill import (
    hill_all_k, select_k, hill_ci, alpha_at,
    weissman_var, kupiec_test, find_breaches,
    log_returns, losses,
)

RNG = np.random.default_rng(42)
N_LARGE = 10_000   # enough data for small-sample variance to settle


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data helpers
# ─────────────────────────────────────────────────────────────────────────────

def pareto_sample(alpha: float, n: int, x_min: float = 1.0) -> np.ndarray:
    """i.i.d. Pareto(alpha, x_min) via inverse CDF."""
    u = RNG.uniform(size=n)
    return x_min / (1 - u) ** (1 / alpha)


def student_t_losses(df: float, n_days: int) -> np.ndarray:
    """
    Daily losses max(0, −r) for Student-t returns.  The t tail is
    regularly varying with index = df.  Zeros are kept, as in the real
    pipeline, so n_days is the full sample size.
    """
    return np.clip(-RNG.standard_t(df, size=n_days), 0, None)


N_DAYS = 5_700   # about the length of the SPY history


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 – Pareto(α=3)
# ─────────────────────────────────────────────────────────────────────────────

def test_hill_pareto_alpha3():
    """α̂ at the selected k should recover α=3 within ±0.5 on 10k samples."""
    true_alpha = 3.0
    x = pareto_sample(true_alpha, N_LARGE)
    ks, alphas = hill_all_k(x)
    best_k, best_alpha = select_k(ks, alphas, n_obs=len(x))
    assert abs(best_alpha - true_alpha) < 0.5, (
        f"Expected α≈{true_alpha}, got {best_alpha:.3f} at k={best_k}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 – Student-t: the realistic case, where the old k rule failed
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("df, low, high", [(3, 2.5, 3.3), (4, 3.1, 4.2)])
def test_hill_student_t(df, low, high):
    """
    Mean α̂ over 20 SPY-sized samples must land near the true tail index.
    Hill reads a little low on Student-t (its tail is only asymptotically
    Pareto), so the bands sit slightly below df.  The old minimum-rolling-std
    k rule averaged ≈2.1 (df=3) and ≈2.3 (df=4) and fails both.
    """
    est = []
    for _ in range(20):
        loss = student_t_losses(df, N_DAYS)
        ks, alphas = hill_all_k(loss)
        _, a = select_k(ks, alphas, n_obs=len(loss))
        est.append(a)
    mean = float(np.mean(est))
    assert low < mean < high, f"t({df}): mean α̂ {mean:.2f} outside ({low}, {high})"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 – hill_all_k properties
# ─────────────────────────────────────────────────────────────────────────────

def test_hill_all_k_positive_finite():
    """All returned α̂ values must be finite and positive."""
    x = pareto_sample(4.0, 2000)
    ks, alphas = hill_all_k(x)
    assert len(ks) == len(alphas)
    assert (ks > 0).all()
    finite_mask = np.isfinite(alphas)
    assert finite_mask.sum() > 0, "At least some α̂ values must be finite."
    assert (alphas[finite_mask] > 0).all(), "All finite α̂ must be positive."


def test_hill_all_k_length():
    """k array must span from k_min to at most n//4."""
    x = pareto_sample(3.5, 5000)
    ks, alphas = hill_all_k(x, k_min=10)
    assert ks[0] == 10
    assert ks[-1] <= len(x) // 4


def test_hill_all_k_matches_direct_formula():
    """The cumulative-sum version must equal α̂(k) = k / Σ ln(X_(i)/X_(k+1))."""
    x = pareto_sample(3.0, 800)
    ks, alphas = hill_all_k(x)
    xs = np.sort(x)[::-1]
    for k, a in zip(ks[::17], alphas[::17]):
        direct = k / np.sum(np.log(xs[:k] / xs[k]))
        assert a == pytest.approx(direct, rel=1e-10)


def test_hill_all_k_too_few_losses_raises():
    with pytest.raises(ValueError):
        hill_all_k(np.r_[np.zeros(100), pareto_sample(3.0, 5)])
    with pytest.raises(ValueError):
        hill_all_k(np.zeros(100))


def test_hill_all_k_ties_give_nan_not_error():
    """When the k largest losses all equal X_(k+1), α̂(k) is undefined (NaN)."""
    x = np.repeat([0.05, 0.04, 0.03, 0.02, 0.01], 40)
    ks, alphas = hill_all_k(x)
    assert np.isnan(alphas).any() and np.isfinite(alphas).any()
    k, a = select_k(ks, alphas, n_obs=len(x))
    assert np.isfinite(a)


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 – select_k
# ─────────────────────────────────────────────────────────────────────────────

def test_select_k_in_range():
    """best_k must lie within the supplied ks array."""
    x = pareto_sample(3.0, 5000)
    ks, alphas = hill_all_k(x)
    best_k, best_alpha = select_k(ks, alphas, n_obs=len(x))
    assert best_k in ks, f"best_k={best_k} not in ks range [{ks[0]}, {ks[-1]}]"
    assert np.isfinite(best_alpha) and best_alpha > 0


def test_select_k_sqrt_rule():
    ks = np.arange(10, 500)
    alphas = np.linspace(4, 2, len(ks))
    k, a = select_k(ks, alphas, n_obs=5_776)        # √5776 = 76
    assert k == 76 and a == alphas[76 - 10]
    k, _ = select_k(ks, alphas, n_obs=25)           # √25 = 5 < k_min → clamp to 10
    assert k == 10


def test_select_k_skips_nan():
    ks = np.arange(10, 60)
    alphas = np.full(len(ks), 3.0)
    alphas[ks == 49] = np.nan                       # √2401 = 49
    k, a = select_k(ks, alphas, n_obs=2_401)
    assert k in (48, 50) and a == 3.0
    with pytest.raises(ValueError):
        select_k(ks, np.full(len(ks), np.nan), n_obs=2_401)


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 – Weissman VaR monotone in confidence
# ─────────────────────────────────────────────────────────────────────────────

def test_weissman_var_monotone():
    """VaR(99.9%) should exceed VaR(99%) for the same data."""
    x = pareto_sample(3.5, 5000)
    ks, alphas = hill_all_k(x)
    best_k, alpha_hat = select_k(ks, alphas, n_obs=len(x))

    wvar_99  = weissman_var(x, alpha_hat, best_k, target_prob=0.01)
    wvar_999 = weissman_var(x, alpha_hat, best_k, target_prob=0.001)

    assert wvar_999 > wvar_99, (
        f"Expected VaR(99.9%) > VaR(99%), got {wvar_999:.4f} vs {wvar_99:.4f}"
    )


def test_weissman_var_counts_all_days():
    """
    Losses are zero on 60 % of days and Pareto(α=3, x_min=1) on the rest.
    The true per-day quantile at p is x with 0.4·x^(−3) = p.  Using only the
    positive days for n (the old behaviour) inflates it by (1/0.4)^(1/3) ≈ 1.36.
    """
    n_days, share_pos, alpha = 200_000, 0.4, 3.0
    n_pos = int(n_days * share_pos)
    loss = np.r_[np.zeros(n_days - n_pos), pareto_sample(alpha, n_pos)]
    for p in (0.01, 0.001):
        true_q = (share_pos / p) ** (1 / alpha)
        est = weissman_var(loss, alpha, k=4_000, target_prob=p)
        assert est == pytest.approx(true_q, rel=0.08), f"p={p}: {est:.3f} vs true {true_q:.3f}"


def test_weissman_var_rejects_bad_inputs():
    x = pareto_sample(3.0, 100)
    with pytest.raises(ValueError):
        weissman_var(x, 3.0, k=100)          # needs X_(k+1)
    with pytest.raises(ValueError):
        weissman_var(x, np.nan, k=10)
    with pytest.raises(ValueError):
        weissman_var(x, 3.0, k=10, target_prob=0.5)   # not beyond X_(k+1)


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 – Kupiec: well-calibrated model should NOT be rejected
# ─────────────────────────────────────────────────────────────────────────────

def test_kupiec_calibrated():
    """
    If we observe exactly the nominal 1% breach rate, the test should
    not reject H₀ at 5% significance.
    """
    n_obs      = 2520   # ~10 years of trading days
    n_breaches = 25     # ≈ 1% → calibrated
    result = kupiec_test(n_obs, n_breaches, confidence=0.99)
    assert not result["reject_h0"], (
        f"Kupiec incorrectly rejected a calibrated model "
        f"(p={result['p_value']:.4f})."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 – Kupiec: badly miscalibrated model SHOULD be rejected
# ─────────────────────────────────────────────────────────────────────────────

def test_kupiec_miscalibrated():
    """
    If we observe 10% breaches under a 99% VaR, H₀ must be rejected.
    """
    n_obs      = 2520
    n_breaches = 252    # 10× the expected 1%
    result = kupiec_test(n_obs, n_breaches, confidence=0.99)
    assert result["reject_h0"], (
        f"Kupiec failed to reject a severely miscalibrated model "
        f"(p={result['p_value']:.4f})."
    )


def test_kupiec_zero_breaches_rejects():
    """Zero breaches in 5,000 days at 99 % is far too few: LR ≈ 100, reject."""
    result = kupiec_test(5_000, 0, confidence=0.99)
    assert np.isfinite(result["lr"]) and result["reject_h0"]
    assert result["lr"] == pytest.approx(-2 * 5_000 * np.log(0.99), rel=1e-12)


def test_kupiec_matches_textbook_statistic():
    n, x, p0 = 2_520, 40, 0.01
    ph = x / n
    lr = -2 * (x * np.log(p0) + (n - x) * np.log(1 - p0) - x * np.log(ph) - (n - x) * np.log(1 - ph))
    assert kupiec_test(n, x)["lr"] == pytest.approx(lr, rel=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Test 7b – breach clustering counts observations
# ─────────────────────────────────────────────────────────────────────────────

def test_find_breaches_clustering_uses_observations():
    """Breaches 10 rows apart are clustered; 11 rows apart are not, whatever the dates."""
    idx = pd.bdate_range("2020-01-01", periods=60)
    loss = pd.Series(0.0, index=idx)
    var = pd.Series(0.02, index=idx)
    loss.iloc[[5, 15, 26]] = 0.05
    b = find_breaches(loss, var, cluster_window=10)
    assert b["clustered"].tolist() == [False, True, False]


# ─────────────────────────────────────────────────────────────────────────────
# Test 8 – log_returns and losses helpers
# ─────────────────────────────────────────────────────────────────────────────

def test_log_returns_and_losses():
    prices = pd.Series([100.0, 101.0, 99.0, 102.0], name="px")
    ret = log_returns(prices)
    assert len(ret) == 3, "Should drop one leading NaN."
    loss = losses(ret)
    assert (loss >= 0).all(), "Losses must be non-negative."
    # Day 2: price dropped from 101→99, so loss > 0
    assert loss.iloc[1] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 9 – hill_ci magnitude
# ─────────────────────────────────────────────────────────────────────────────

def test_hill_ci_decreases_with_k():
    """
    Confidence interval width (α/√k) should shrink as k grows.
    """
    alpha = 4.0
    ci_small = hill_ci(alpha, k=20)
    ci_large  = hill_ci(alpha, k=200)
    assert ci_large < ci_small, (
        f"CI should shrink with k; got ci(k=20)={ci_small:.3f}, "
        f"ci(k=200)={ci_large:.3f}."
    )
