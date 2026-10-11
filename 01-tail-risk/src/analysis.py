"""
analysis.py
===========
End-to-end pipeline:
  1. Load the saved CSV of prices.
  2. Compute Gaussian rolling VaR (one-year window), count breaches, run Kupiec test.
  3. Run the Hill estimator on each asset; select k; compute Weissman VaR.
  4. Print the comparison table and save results to results/.

Run:
    python src/analysis.py
"""

from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd

from hill import (
    log_returns, losses, hill_all_k, select_k, hill_ci, alpha_at,
    gaussian_var, weissman_var, find_breaches, kupiec_test,
    norm_ppf,
)

# ── paths ────────────────────────────────────────────────────────────────────
ROOT        = pathlib.Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
PRICES_CSV  = RESULTS_DIR / "prices_raw.csv"

CONFIDENCE_99   = 0.99
CONFIDENCE_999  = 0.999
CLUSTER_WINDOW  = 10    # observations — breaches within this are "clustered"


def periods_per_year(index: pd.DatetimeIndex) -> int:
    """
    Observations per calendar year, inferred from the dates: ~252 for
    exchange-traded assets, ~365 for Bitcoin, which trades every day.
    Rounded to the nearest of those two so short samples don't drift.
    """
    span_years = (index[-1] - index[0]).days / 365.25
    per_year = len(index) / span_years
    return 365 if per_year > 300 else 252


def load_prices() -> pd.DataFrame:
    df = pd.read_csv(PRICES_CSV, index_col="Date", parse_dates=True)
    df.sort_index(inplace=True)
    return df


def run_asset(
    ticker: str,
    prices: pd.Series,
    confidence: float = CONFIDENCE_99,
) -> dict:
    """Full analysis for one ticker.  Returns a results dict."""
    ret   = log_returns(prices)
    loss  = losses(ret)
    ppy   = periods_per_year(ret.index)

    # ── Gaussian VaR (one year of observations per window) ──────────────────
    gvar  = gaussian_var(ret, confidence=confidence, window=ppy)
    bdf   = find_breaches(loss, gvar, cluster_window=CLUSTER_WINDOW)

    n_obs      = int(gvar.dropna().shape[0])
    n_breaches = len(bdf)
    ann_years  = n_obs / ppy
    breaches_per_year = n_breaches / ann_years if ann_years > 0 else np.nan
    expected_per_year = (1.0 - confidence) * ppy

    kupiec = kupiec_test(n_obs, n_breaches, confidence)

    # ── Hill estimator (full sample) ─────────────────────────────────────────
    n_days = len(loss)
    ks, alphas = hill_all_k(loss)
    best_k, best_alpha = select_k(ks, alphas, n_obs=n_days)
    ci_half   = hill_ci(best_alpha, best_k)
    alpha_lo  = alpha_at(ks, alphas, round(best_k * 0.8))   # sensitivity: k* − 20 %
    alpha_hi  = alpha_at(ks, alphas, round(best_k * 1.2))   # sensitivity: k* + 20 %
    # Wider view: the Hill plot drifts down as k grows, so report α̂ at fixed
    # shares of the sample too (1 %, 2 %, 5 % of all days).
    alpha_by_share = {share: alpha_at(ks, alphas, round(share * n_days)) for share in (0.01, 0.02, 0.05)}

    # ── Weissman VaR (n = all days, so p is a per-day probability) ──────────
    wvar_99  = weissman_var(loss, best_alpha, best_k, target_prob=1 - CONFIDENCE_99)
    wvar_999 = weissman_var(loss, best_alpha, best_k, target_prob=1 - CONFIDENCE_999)

    # Sample Gaussian 99% and 99.9% VaR (full-sample, for comparison table)
    mu_all  = float(ret.mean())
    sig_all = float(ret.std())
    gvar_99_full  = -(mu_all - norm_ppf(CONFIDENCE_99)  * sig_all)
    gvar_999_full = -(mu_all - norm_ppf(CONFIDENCE_999) * sig_all)

    # What actually happened: empirical loss quantiles, and how many days were
    # worse than each 99.9 % estimate (expected: 0.1 % of days).
    emp_99  = float(np.quantile(-ret, CONFIDENCE_99))
    emp_999 = float(np.quantile(-ret, CONFIDENCE_999))
    days_over_g999 = int((loss > gvar_999_full).sum())
    days_over_w999 = int((loss > wvar_999).sum())

    return {
        "ticker":               ticker,
        "n_obs":                n_obs,
        "years":                round(ann_years, 1),
        "alpha_hill":           round(best_alpha, 3),
        "alpha_ci":             round(ci_half, 3),
        "alpha_k_minus_20":     round(alpha_lo, 3),
        "alpha_k_plus_20":      round(alpha_hi, 3),
        "alpha_1pct":           round(alpha_by_share[0.01], 3),
        "alpha_2pct":           round(alpha_by_share[0.02], 3),
        "alpha_5pct":           round(alpha_by_share[0.05], 3),
        "best_k":               best_k,
        "periods_per_year":     ppy,
        "breaches_99":          n_breaches,
        "breaches_per_year":    round(breaches_per_year, 2),
        "expected_per_year":    round(expected_per_year, 2),
        "kupiec_p":             float(f'{kupiec["p_value"]:.3g}'),
        "kupiec_reject":        kupiec["reject_h0"],
        "gvar_99_full_pct":     round(gvar_99_full * 100, 3),
        "gvar_999_full_pct":    round(gvar_999_full * 100, 3),
        "wvar_99_pct":          round(wvar_99 * 100, 3),
        "wvar_999_pct":         round(wvar_999 * 100, 3),
        "emp_99_pct":           round(emp_99 * 100, 3),
        "emp_999_pct":          round(emp_999 * 100, 3),
        "days_over_g999":       days_over_g999,
        "days_over_w999":       days_over_w999,
        "expected_over_999":    round((1 - CONFIDENCE_999) * n_days, 1),
        "breach_df":            bdf,
        "ks":                   ks,
        "alphas":               alphas,
        "loss_series":          loss,
        "ret_series":           ret,
        "gvar_series":          gvar,
    }


def main():
    prices = load_prices()
    results = {}
    for ticker in prices.columns:
        print(f"  Analysing {ticker} …")
        try:
            results[ticker] = run_asset(ticker, prices[ticker].dropna())
        except Exception as exc:
            print(f"    ⚠  {ticker} failed: {exc}")

    # ── Summary table ─────────────────────────────────────────────────────────
    rows = []
    for r in results.values():
        rows.append({
            "Ticker":           r["ticker"],
            "Years":            r["years"],
            "α̂ (Hill)":        r["alpha_hill"],
            "± CI":             r["alpha_ci"],
            "α̂ (k*−20%)":      r["alpha_k_minus_20"],
            "α̂ (k*+20%)":      r["alpha_k_plus_20"],
            "α̂ (k=1% of n)":   r["alpha_1pct"],
            "α̂ (k=2% of n)":   r["alpha_2pct"],
            "α̂ (k=5% of n)":   r["alpha_5pct"],
            "k*":              r["best_k"],
            "Breaches/yr (obs)":r["breaches_per_year"],
            "Expected/yr":      r["expected_per_year"],
            "Kupiec p":         r["kupiec_p"],
            "Reject H₀":       r["kupiec_reject"],
            "G-VaR 99% (%)":   r["gvar_99_full_pct"],
            "W-VaR 99% (%)":   r["wvar_99_pct"],
            "G-VaR 99.9% (%)": r["gvar_999_full_pct"],
            "W-VaR 99.9% (%)": r["wvar_999_pct"],
            "Empirical 99% (%)": r["emp_99_pct"],
            "Empirical 99.9% (%)": r["emp_999_pct"],
            "Days > G-VaR 99.9%": r["days_over_g999"],
            "Days > W-VaR 99.9%": r["days_over_w999"],
            "Expected days > 99.9%": r["expected_over_999"],
        })
    table = pd.DataFrame(rows)
    print("\n" + "═" * 80)
    print("  FAT-TAIL RISK SUMMARY TABLE")
    print("═" * 80)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(table.to_string(index=False))

    table.to_csv(RESULTS_DIR / "summary_table.csv", index=False)
    print(f"\nSaved → {(RESULTS_DIR / 'summary_table.csv').relative_to(ROOT)}")

    # ── Save breach details ───────────────────────────────────────────────────
    for ticker, r in results.items():
        out = RESULTS_DIR / f"breaches_{ticker.replace('-','_')}.csv"
        r["breach_df"].to_csv(out)

    return results


if __name__ == "__main__":
    main()
