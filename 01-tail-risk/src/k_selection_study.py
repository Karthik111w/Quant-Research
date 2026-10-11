"""
k_selection_study.py
====================
Monte Carlo check of rules for choosing k in the Hill estimator.

Losses are the positive half of Student-t returns, whose tail exponent equals
the degrees of freedom, on samples the size of the SPY history (~5,700 days).
Each rule is scored by bias and RMSE against the known α.

Run:
    python src/k_selection_study.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from hill import hill_all_k, alpha_at


def plateau_rule(ks: np.ndarray, alphas: np.ndarray, window: int = 15) -> float:
    """The original rule: centre of the window where α̂(k) has minimum rolling std."""
    roll_std = pd.Series(alphas).rolling(window).std().to_numpy()
    end = int(np.nanargmin(roll_std))
    return float(alphas[max(0, end - window // 2)])


def ks_distance_rule(losses: np.ndarray, ks: np.ndarray, alphas: np.ndarray) -> float:
    """Danielsson et al. (2016): minimise the max gap between empirical and fitted tail quantiles."""
    x = np.sort(losses[losses > 0])[::-1]
    t = int(ks[-1])
    j = np.arange(1, t + 1)
    dist = [np.max(np.abs(x[:t] - x[k] * (k / j) ** (1 / a))) for k, a in zip(ks, alphas)]
    return float(alphas[int(np.argmin(dist))])


def main(n_days: int = 5_700, reps: int = 150, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for true_alpha in (2.5, 3.0, 4.0, 5.0):
        est = {"plateau (old)": [], "sqrt(n) (used)": [], "2% of n": [], "KS distance": []}
        for _ in range(reps):
            r = rng.standard_t(true_alpha, n_days)
            loss = np.clip(-r, 0, None)
            ks, alphas = hill_all_k(loss)
            est["plateau (old)"].append(plateau_rule(ks, alphas))
            est["sqrt(n) (used)"].append(alpha_at(ks, alphas, round(np.sqrt(n_days))))
            est["2% of n"].append(alpha_at(ks, alphas, round(0.02 * n_days)))
            est["KS distance"].append(ks_distance_rule(loss, ks, alphas))
        for rule, values in est.items():
            v = np.array(values)
            rows.append({
                "true α": true_alpha,
                "rule": rule,
                "mean α̂": round(v.mean(), 2),
                "sd": round(v.std(), 2),
                "RMSE": round(float(np.sqrt(np.mean((v - true_alpha) ** 2))), 2),
            })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    main()
