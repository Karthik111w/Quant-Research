"""Monte Carlo pricing and Greek estimators for a European call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr


class Estimate(NamedTuple):
    value: float
    standard_error: float


@dataclass(frozen=True)
class Option:
    spot: float = 100.0
    strike: float = 100.0
    rate: float = 0.05
    volatility: float = 0.2
    maturity: float = 1.0


def validate_simulation(num_paths: int, bump: float | None = None) -> None:
    if num_paths < 2:
        raise ValueError("num_paths must be at least 2 to estimate standard error")
    if bump is not None and bump <= 0:
        raise ValueError("bump must be positive")


def terminal_stock(spot: float, z: jax.Array, option: Option) -> jax.Array:
    drift = (option.rate - 0.5 * option.volatility**2) * option.maturity
    diffusion = option.volatility * jnp.sqrt(option.maturity) * z
    return spot * jnp.exp(drift + diffusion)


def call_payoff(spot: float, z: jax.Array, option: Option) -> jax.Array:
    return jnp.maximum(terminal_stock(spot, z, option) - option.strike, 0.0)


def discounted_payoffs(spot: float, z: jax.Array, option: Option) -> jax.Array:
    return jnp.exp(-option.rate * option.maturity) * call_payoff(spot, z, option)


def _estimate(samples: jax.Array) -> Estimate:
    values = np.asarray(samples, dtype=np.float64)
    if values.size < 2:
        raise ValueError("num_paths must be at least 2 to estimate standard error")
    return Estimate(float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size)))


def price_mc(spot: float, z: jax.Array, option: Option) -> Estimate:
    return _estimate(discounted_payoffs(spot, z, option))


def delta_fresh_noise(
    spot: float,
    bump: float,
    z_up: jax.Array,
    z_down: jax.Array,
    option: Option,
) -> Estimate:
    """Central bump estimate from independent paths at each bumped spot."""
    validate_simulation(z_up.size, bump)
    validate_simulation(z_down.size, bump)
    if z_up.shape != z_down.shape:
        raise ValueError("up and down normal draws must have the same shape")
    samples = (
        discounted_payoffs(spot + bump, z_up, option)
        - discounted_payoffs(spot - bump, z_down, option)
    ) / (2.0 * bump)
    return _estimate(samples)


def delta_common_noise(
    spot: float,
    bump: float,
    z: jax.Array,
    option: Option,
) -> Estimate:
    """Central bump estimate using common random numbers at both spots."""
    validate_simulation(z.size, bump)
    samples = (
        discounted_payoffs(spot + bump, z, option)
        - discounted_payoffs(spot - bump, z, option)
    ) / (2.0 * bump)
    return _estimate(samples)


def _pathwise_delta(spot: float, z: jax.Array, option: Option) -> jax.Array:
    stock = terminal_stock(spot, z, option)
    in_the_money = stock > option.strike
    return (
        jnp.exp(-option.rate * option.maturity)
        * in_the_money
        * stock
        / spot
    )


def delta_autodiff(spot: float, z: jax.Array, option: Option) -> Estimate:
    """Pathwise delta via JAX differentiation of the simulated price."""
    price_for_spot = lambda s: jnp.mean(discounted_payoffs(s, z, option))
    value = jax.grad(price_for_spot)(spot)
    pathwise = jax.vmap(jax.grad(lambda s, zi: discounted_payoffs(s, zi, option), argnums=0))(
        jnp.full(z.shape, spot), z
    )
    return Estimate(float(value), _estimate(pathwise).standard_error)


def black_scholes_price_delta_gamma(option: Option) -> tuple[float, float, float]:
    """Closed-form Black-Scholes call price, delta, and gamma."""
    s, k, r, sigma, t = (
        option.spot,
        option.strike,
        option.rate,
        option.volatility,
        option.maturity,
    )
    d1 = (jnp.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * jnp.sqrt(t))
    d2 = d1 - sigma * jnp.sqrt(t)
    density = jnp.exp(-0.5 * d1**2) / jnp.sqrt(2.0 * jnp.pi)
    price = s * ndtr(d1) - k * jnp.exp(-r * t) * ndtr(d2)
    delta = ndtr(d1)
    gamma = density / (s * sigma * jnp.sqrt(t))
    return float(price), float(delta), float(gamma)


def gamma_naive_autodiff(spot: float, z: jax.Array, option: Option) -> Estimate:
    """Second pathwise derivative; identically zero for the hard call payoff."""
    price_for_spot = lambda s: jnp.mean(discounted_payoffs(s, z, option))
    gamma = jax.grad(jax.grad(price_for_spot))(spot)
    return Estimate(float(gamma), 0.0)


def gamma_likelihood_ratio(
    spot: float, z: jax.Array, option: Option
) -> Estimate:
    """Gamma by differentiating the lognormal density rather than the payoff."""
    scaled_volatility = option.volatility * jnp.sqrt(option.maturity)
    score = (z**2 - scaled_volatility * z - 1.0) / (
        spot**2 * option.volatility**2 * option.maturity
    )
    return _estimate(discounted_payoffs(spot, z, option) * score)


def gamma_smoothed_autodiff(
    spot: float, z: jax.Array, option: Option, temperature: float
) -> Estimate:
    """Pathwise gamma with a softplus payoff approximation."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    discount = jnp.exp(-option.rate * option.maturity)

    def smooth_payoff(s: float, zi: jax.Array) -> jax.Array:
        moneyness = terminal_stock(s, zi, option) - option.strike
        return discount * temperature * jnp.logaddexp(0.0, moneyness / temperature)

    second_derivative = jax.grad(jax.grad(lambda s, zi: smooth_payoff(s, zi), argnums=0), argnums=0)
    pathwise = jax.vmap(second_derivative, in_axes=(None, 0))(spot, z)
    return _estimate(pathwise)


def gamma_finite_difference_delta(
    spot: float, bump: float, z: jax.Array, option: Option
) -> Estimate:
    """Finite difference of pathwise autodiff deltas, sharing the paths."""
    validate_simulation(z.size, bump)
    samples = (
        _pathwise_delta(spot + bump, z, option)
        - _pathwise_delta(spot - bump, z, option)
    ) / (2.0 * bump)
    return _estimate(samples)
