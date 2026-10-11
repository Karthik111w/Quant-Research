"""Run the reproducible Monte Carlo experiment and save its figures and data."""

from __future__ import annotations

import argparse
import csv
import pathlib

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from greeks import (
    Option,
    black_scholes_price_delta_gamma,
    delta_autodiff,
    delta_common_noise,
    delta_fresh_noise,
    gamma_finite_difference_delta,
    gamma_likelihood_ratio,
    gamma_naive_autodiff,
    gamma_smoothed_autodiff,
    price_mc,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 2026


def run(num_paths: int = 200_000) -> None:
    if num_paths < 2:
        raise ValueError("num_paths must be at least 2")

    RESULTS.mkdir(exist_ok=True)
    option = Option()
    bumps = np.geomspace(0.05, 30.0, 14)
    repeats = 30
    temperatures = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
    keys = jax.random.split(
        jax.random.PRNGKey(SEED),
        2 + repeats * (1 + 2 * len(bumps)),
    )
    z_price = jax.random.normal(keys[0], (num_paths,), dtype=jnp.float64)
    z_gamma = jax.random.normal(keys[1], (num_paths,), dtype=jnp.float64)
    exact_price, exact_delta, exact_gamma = black_scholes_price_delta_gamma(option)

    price = price_mc(option.spot, z_price, option)
    autodiff = delta_autodiff(option.spot, z_price, option)
    rows: list[dict[str, float | str]] = []

    delta_samples: dict[str, list[list[float]]] = {
        method: [[] for _ in bumps]
        for method in ("fresh noise", "common noise", "autodiff")
    }
    delta_standard_errors: dict[str, list[list[float]]] = {
        method: [[] for _ in bumps]
        for method in ("fresh noise", "common noise", "autodiff")
    }
    keys_per_repeat = 1 + 2 * len(bumps)
    for repeat in range(repeats):
        key_offset = 2 + repeat * keys_per_repeat
        z_common = jax.random.normal(keys[key_offset], (num_paths,), dtype=jnp.float64)
        autodiff_repeat = delta_autodiff(option.spot, z_common, option)
        for index, bump in enumerate(bumps):
            z_up = jax.random.normal(
                keys[key_offset + 1 + 2 * index], (num_paths,), dtype=jnp.float64
            )
            z_down = jax.random.normal(
                keys[key_offset + 2 + 2 * index], (num_paths,), dtype=jnp.float64
            )
            estimates = (
                ("fresh noise", delta_fresh_noise(option.spot, float(bump), z_up, z_down, option)),
                ("common noise", delta_common_noise(option.spot, float(bump), z_common, option)),
                ("autodiff", autodiff_repeat),
            )
            for method, estimate in estimates:
                delta_samples[method][index].append(estimate.value)
                delta_standard_errors[method][index].append(estimate.standard_error)

    for index, bump in enumerate(bumps):
        for method in ("fresh noise", "common noise", "autodiff"):
            sample_values = np.asarray(delta_samples[method][index])
            mean_estimate = float(sample_values.mean())
            rmse = float(np.sqrt(np.mean((sample_values - exact_delta) ** 2)))
            rows.append(
                {
                    "method": method,
                    "bump": float(bump),
                    "runs": repeats,
                    "mean_estimate": mean_estimate,
                    "rmse": rmse,
                    "mean_standard_error": float(
                        np.mean(delta_standard_errors[method][index])
                    ),
                }
            )

    with (RESULTS / "delta_estimates.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "method",
                "bump",
                "runs",
                "mean_estimate",
                "rmse",
                "mean_standard_error",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), layout="constrained")
    colors = {"fresh noise": "#ef4444", "common noise": "#2563eb", "autodiff": "#10b981"}
    for method in ("fresh noise", "common noise", "autodiff"):
        method_rows = [row for row in rows if row["method"] == method]
        ax.loglog(
            [row["bump"] for row in method_rows],
            [row["rmse"] for row in method_rows],
            marker="o" if method != "autodiff" else None,
            linewidth=2,
            color=colors[method],
            label=method,
        )
    ax.set(
        xlabel="Central bump size h ($)",
        ylabel=f"Delta RMSE over {repeats} independent runs",
        title=f"Monte Carlo call delta: bias versus simulation noise ({num_paths:,} paths/run)",
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.savefig(RESULTS / "delta_error.png", dpi=180)
    plt.close(fig)

    gamma_smoothing = [
        (
            temperature,
            gamma_smoothed_autodiff(option.spot, z_gamma, option, temperature),
        )
        for temperature in temperatures
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), layout="constrained")
    smoothing_values = [estimate.value for _, estimate in gamma_smoothing]
    smoothing_errors = [2.0 * estimate.standard_error for _, estimate in gamma_smoothing]
    ax.errorbar(
        temperatures,
        smoothing_values,
        yerr=smoothing_errors,
        marker="o",
        capsize=4,
        linewidth=2,
        color="#2563eb",
        label="Softplus gamma ±2 SE",
    )
    ax.axhline(exact_gamma, color="#10b981", linestyle="--", label="Black-Scholes gamma")
    ax.set_xscale("log")
    ax.set(
        xlabel="Softplus temperature τ ($)",
        ylabel="Gamma estimate",
        title=f"Softplus gamma: smoothing bias and Monte Carlo uncertainty ({num_paths:,} paths)",
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.savefig(RESULTS / "gamma_softplus_sweep.png", dpi=180)
    plt.close(fig)

    gamma_estimates = (
        ("naive pathwise autodiff", gamma_naive_autodiff(option.spot, z_gamma, option)),
        ("likelihood ratio", gamma_likelihood_ratio(option.spot, z_gamma, option)),
        ("softplus autodiff (tau=0.5)", next(
            estimate for temperature, estimate in gamma_smoothing if temperature == 0.5
        )),
        ("finite difference of delta (h=0.5)", gamma_finite_difference_delta(option.spot, 0.5, z_gamma, option)),
    )
    summary_rows = [
        {
            "greek": "price",
            "method": "Monte Carlo",
            "estimate": price.value,
            "black_scholes": exact_price,
            "absolute_error": abs(price.value - exact_price),
            "standard_error": price.standard_error,
        },
        {
            "greek": "delta",
            "method": "autodiff",
            "estimate": autodiff.value,
            "black_scholes": exact_delta,
            "absolute_error": abs(autodiff.value - exact_delta),
            "standard_error": autodiff.standard_error,
        },
    ]
    summary_rows.extend(
        {
            "greek": "gamma",
            "method": label,
            "estimate": estimate.value,
            "black_scholes": exact_gamma,
            "absolute_error": abs(estimate.value - exact_gamma),
            "standard_error": estimate.standard_error,
        }
        for label, estimate in gamma_estimates
    )
    summary_rows.extend(
        {
            "greek": "gamma",
            "method": f"softplus autodiff (tau={temperature:g})",
            "estimate": estimate.value,
            "black_scholes": exact_gamma,
            "absolute_error": abs(estimate.value - exact_gamma),
            "standard_error": estimate.standard_error,
        }
        for temperature, estimate in gamma_smoothing
    )
    with (RESULTS / "greek_estimates.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=summary_rows[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Black-Scholes price={exact_price:.6f}, delta={exact_delta:.6f}, gamma={exact_gamma:.6f}")
    print(f"Monte Carlo price={price.value:.6f} (SE {price.standard_error:.6f})")
    print(f"Autodiff delta={autodiff.value:.6f} (SE {autodiff.standard_error:.6f})")
    print("\nGamma estimates:")
    for label, estimate in gamma_estimates:
        print(
            f"  {label:38s} {estimate.value:.6f} "
            f"(SE {estimate.standard_error:.6f}, error {abs(estimate.value - exact_gamma):.6f})"
        )
    print("\nSoftplus gamma temperature sweep (estimate ± 2 SE):")
    for temperature, estimate in gamma_smoothing:
        print(
            f"  tau={temperature:4.2f}: {estimate.value:.6f} "
            f"+/- {2 * estimate.standard_error:.6f}"
        )
    print(
        f"\nWrote {RESULTS / 'delta_error.png'}, "
        f"{RESULTS / 'delta_estimates.csv'}, {RESULTS / 'greek_estimates.csv'}, "
        f"and {RESULTS / 'gamma_softplus_sweep.png'}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=200_000, help="Monte Carlo path count")
    run(parser.parse_args().paths)
