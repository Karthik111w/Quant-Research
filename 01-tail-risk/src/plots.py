"""
plots.py
========
Generate every figure for the project and save them to results/.

Figures produced
----------------
hill_plots.png      — Hill plot grid (one panel per asset)
breach_timeline.png — Breach timeline per asset, colour-coded by loss
var_compare.png     — Gaussian vs Weissman VaR comparison bar chart

Run:
    python src/plots.py
(calls analysis.py internally — no need to run it separately first)
"""

from __future__ import annotations
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import pandas as pd

# We import the analysis results dict via a small helper so this file
# can also be imported by the notebook.
from analysis import load_prices, run_asset, CONFIDENCE_99
from hill import hill_ci

ROOT        = pathlib.Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

# ── Palette (matches Oracle's Edge dark tokens, works on white too) ───────────
GOLD    = "#d4a853"
DANGER  = "#ef4444"
WARNING = "#f59e0b"
SUCCESS = "#10b981"
MUTED   = "#7d8ba0"
BG      = "#0f172a"
SURFACE = "#1e293b"
TEXT    = "#e2e8f0"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    "#334155",
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "grid.color":        "#1e293b",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "text.color":        TEXT,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "legend.facecolor":  SURFACE,
    "legend.edgecolor":  "#334155",
    "legend.labelcolor": TEXT,
})


# ─────────────────────────────────────────────────────────────────────────────
# 1. Hill plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_hill(results: dict, out_path: pathlib.Path) -> None:
    """One Hill-plot panel per asset in a grid."""
    tickers = list(results.keys())
    n = len(tickers)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 5.5, nrows * 4),
        facecolor=BG,
        squeeze=False,
    )
    fig.suptitle(
        "Hill Tail-Exponent Estimates  α̂(k)",
        fontsize=14, fontweight="bold", color=TEXT, y=1.01,
    )

    for ax_idx, ticker in enumerate(tickers):
        r   = results[ticker]
        ks  = r["ks"]
        als = r["alphas"]
        k_star  = r["best_k"]
        a_star  = r["alpha_hill"]
        ci_half = hill_ci(a_star, k_star)

        row, col = divmod(ax_idx, ncols)
        ax = axes[row][col]

        # Confidence band  ±α/√k
        ci_band = np.array([hill_ci(a, k) for a, k in zip(als, ks)])
        ax.fill_between(
            ks, als - ci_band, als + ci_band,
            alpha=0.18, color=GOLD, label="±1σ band",
        )
        ax.plot(ks, als, color=GOLD, linewidth=1.6, label="α̂(k)")

        # Horizontal lines: range typically reported for equity returns (not a bound)
        for y, ls, lbl in [
            (3, ":", "α = 3"),
            (5, ":", "α = 5"),
        ]:
            ax.axhline(y, color=MUTED, linestyle=ls, linewidth=0.8, label=lbl)

        # Selected k marker
        ax.axvline(k_star, color=DANGER, linestyle="--", linewidth=1.2, alpha=0.85)
        ax.scatter([k_star], [a_star], color=DANGER, s=60, zorder=5)
        ax.annotate(
            f"k*={k_star}\nα̂={a_star:.2f}±{ci_half:.2f}",
            xy=(k_star, a_star),
            xytext=(k_star + max(1, len(ks) // 12), a_star + 0.25),
            fontsize=8, color=DANGER,
            arrowprops=dict(arrowstyle="->", color=DANGER, lw=0.8),
        )

        ax.set_title(ticker, fontweight="bold", fontsize=11)
        ax.set_xlabel("k  (order statistics)")
        ax.set_ylabel("α̂  (tail exponent)")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.4)
        if ax_idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    # Hide unused panels
    for ax_idx in range(n, nrows * ncols):
        row, col = divmod(ax_idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved → {out_path.relative_to(ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Breach heatmap / calendar dots
# ─────────────────────────────────────────────────────────────────────────────

def plot_breach_timeline(results: dict, out_path: pathlib.Path) -> None:
    """
    One row per asset.  Each breach is a dot at (year, day-of-year),
    coloured by loss magnitude (green → amber → red).
    """
    tickers = list(results.keys())
    fig, axes = plt.subplots(
        len(tickers), 1,
        figsize=(14, 2.4 * len(tickers)),
        facecolor=BG, sharex=False,
    )
    if len(tickers) == 1:
        axes = [axes]

    fig.suptitle(
        "VaR Breach Timeline  (99% Gaussian, rolling one-year window)",
        fontsize=13, fontweight="bold", color=TEXT,
    )

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "loss_severity", [SUCCESS, WARNING, DANGER]
    )

    for ax, ticker in zip(axes, tickers):
        r   = results[ticker]
        bdf = r["breach_df"]

        if bdf.empty:
            ax.text(0.5, 0.5, f"{ticker} — no breaches", ha="center",
                    va="center", transform=ax.transAxes, color=TEXT)
            ax.set_ylabel(ticker, fontweight="bold", color=GOLD)
            continue

        dates  = pd.to_datetime(bdf.index)
        years  = dates.year + dates.day_of_year / 365.25
        losses_pct = bdf["loss"].values * 100

        # Normalise for colour mapping
        lo, hi = losses_pct.min(), losses_pct.max()
        norm_l = (losses_pct - lo) / (hi - lo + 1e-9)
        colours = cmap(norm_l)

        sc = ax.scatter(years, [0.5] * len(years), c=losses_pct,
                        cmap=cmap, vmin=lo, vmax=hi,
                        s=55, alpha=0.85, marker="|", linewidths=2.5)

        # Highlight clustered breaches with a ring
        clustered_mask = bdf["clustered"].values
        if clustered_mask.any():
            ax.scatter(
                years[clustered_mask], [0.5] * clustered_mask.sum(),
                s=160, facecolors="none", edgecolors=WARNING, linewidths=1.2,
                zorder=5, label="Clustered",
            )

        # Annotate crisis years
        # Lehman collapse (15 Sep 2008) and the COVID crash (March 2020), only
        # where the asset has data (Bitcoin starts in 2014).  Each panel spans
        # its own VaR sample so short histories don't leave an empty half.
        span = r["gvar_series"].dropna().index
        x_lo = span[0].year + span[0].day_of_year / 365.25
        x_hi = span[-1].year + span[-1].day_of_year / 365.25
        ax.set_xlim(x_lo - 0.2, x_hi + 0.2)
        for crisis, label in [(pd.Timestamp("2008-09-15"), "GFC"), (pd.Timestamp("2020-03-01"), "COVID")]:
            if not span[0] <= crisis <= span[-1]:
                continue
            x_c = crisis.year + crisis.day_of_year / 365.25
            ax.axvline(x_c, color=MUTED, linestyle=":", linewidth=0.8)
            ax.text(x_c + 0.05, 0.82, label, color=MUTED, fontsize=7,
                    transform=ax.get_xaxis_transform())

        cb = fig.colorbar(sc, ax=ax, orientation="vertical", pad=0.01, shrink=0.6)
        cb.set_label("Loss (%)", color=MUTED, fontsize=7)
        cb.ax.yaxis.set_tick_params(color=MUTED, labelcolor=MUTED, labelsize=6)

        n_per_yr = round(len(bdf) / r["years"], 2)
        ax.set_ylabel(
            f"{ticker}\n{len(bdf)} breaches\n({n_per_yr}/yr)",
            fontweight="bold", color=GOLD, fontsize=9,
        )
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        ax.grid(axis="x", alpha=0.3)
        if clustered_mask.any():
            ax.legend(fontsize=7, loc="upper left")

    axes[-1].set_xlabel("Year", color=TEXT)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved → {out_path.relative_to(ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. VaR comparison bar chart (Gaussian vs Weissman)
# ─────────────────────────────────────────────────────────────────────────────

def plot_var_compare(results: dict, out_path: pathlib.Path) -> None:
    """
    Grouped bar chart: for each asset, side-by-side bars for
    Gaussian VaR 99%, Weissman VaR 99%, Gaussian 99.9%, Weissman 99.9%.
    """
    tickers = list(results.keys())
    x       = np.arange(len(tickers))
    width   = 0.2

    g99  = [results[t]["gvar_99_full_pct"]  for t in tickers]
    w99  = [results[t]["wvar_99_pct"]        for t in tickers]
    g999 = [results[t]["gvar_999_full_pct"]  for t in tickers]
    w999 = [results[t]["wvar_999_pct"]       for t in tickers]

    fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG)
    ax.set_facecolor(SURFACE)

    ax.bar(x - 1.5*width, g99,  width, label="Gaussian 99%",   color=GOLD,    alpha=0.85)
    ax.bar(x - 0.5*width, w99,  width, label="Weissman 99%",   color=DANGER,  alpha=0.85)
    ax.bar(x + 0.5*width, g999, width, label="Gaussian 99.9%", color=GOLD,    alpha=0.40)
    ax.bar(x + 1.5*width, w999, width, label="Weissman 99.9%", color=DANGER,  alpha=0.40)

    # What actually happened: empirical loss quantiles over the same sample
    e99  = [results[t]["emp_99_pct"]  for t in tickers]
    e999 = [results[t]["emp_999_pct"] for t in tickers]
    ax.scatter(x - 1.0*width, e99,  marker="_", s=420, linewidths=2.2, color=TEXT, zorder=5,
               label="Empirical (actual) quantile")
    ax.scatter(x + 1.0*width, e999, marker="_", s=420, linewidths=2.2, color=TEXT, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(tickers, fontsize=11, fontweight="bold")
    ax.set_ylabel("VaR (% of portfolio, 1-day)", color=TEXT)
    ax.set_title(
        "1-day VaR: Gaussian vs Weissman, against what actually happened",
        fontweight="bold", color=TEXT, fontsize=12,
    )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved → {out_path.relative_to(ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    prices  = load_prices()
    results = {}
    for ticker in prices.columns:
        print(f"  Running {ticker} …")
        try:
            results[ticker] = run_asset(ticker, prices[ticker].dropna())
        except Exception as exc:
            print(f"    ⚠  {ticker} failed: {exc}")

    plot_hill(         results, RESULTS_DIR / "hill_plots.png")
    plot_breach_timeline(results, RESULTS_DIR / "breach_timeline.png")
    plot_var_compare(  results, RESULTS_DIR / "var_compare.png")
    print("\nAll figures saved to results/")


if __name__ == "__main__":
    main()
