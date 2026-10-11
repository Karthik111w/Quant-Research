from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from greeks import (
    Option,
    black_scholes_price_delta_gamma,
    delta_autodiff,
    delta_common_noise,
    gamma_finite_difference_delta,
    gamma_likelihood_ratio,
    gamma_naive_autodiff,
    gamma_smoothed_autodiff,
    price_mc,
)


@pytest.fixture(scope="module")
def sample():
    option = Option()
    z = jax.random.normal(jax.random.PRNGKey(17), (150_000,), dtype=jnp.float64)
    return option, z


def test_price_and_pathwise_delta_match_black_scholes(sample):
    option, z = sample
    exact_price, exact_delta, _ = black_scholes_price_delta_gamma(option)
    price = price_mc(option.spot, z, option)
    delta = delta_autodiff(option.spot, z, option)
    assert abs(price.value - exact_price) < 4 * price.standard_error
    assert abs(delta.value - exact_delta) < 4 * delta.standard_error


def test_common_noise_delta_matches_black_scholes(sample):
    option, z = sample
    _, exact_delta, _ = black_scholes_price_delta_gamma(option)
    delta = delta_common_noise(option.spot, 0.5, z, option)
    assert abs(delta.value - exact_delta) < 4 * delta.standard_error + 2e-4


def test_naive_pathwise_gamma_is_zero_but_true_gamma_is_not(sample):
    option, z = sample
    _, _, exact_gamma = black_scholes_price_delta_gamma(option)
    estimate = gamma_naive_autodiff(option.spot, z, option)
    assert estimate.value == 0.0
    assert exact_gamma > 0.0


@pytest.mark.parametrize(
    "estimator",
    [
        lambda s, z, o: gamma_likelihood_ratio(s, z, o),
        lambda s, z, o: gamma_smoothed_autodiff(s, z, o, temperature=0.5),
        lambda s, z, o: gamma_finite_difference_delta(s, 0.5, z, o),
    ],
)
def test_gamma_fixes_match_black_scholes(sample, estimator):
    option, z = sample
    _, _, exact_gamma = black_scholes_price_delta_gamma(option)
    estimate = estimator(option.spot, z, option)
    assert abs(estimate.value - exact_gamma) < 4 * estimate.standard_error


def test_invalid_path_count_or_bump_raise(sample):
    option, z = sample
    with pytest.raises(ValueError, match="at least 2"):
        price_mc(option.spot, z[:1], option)
    with pytest.raises(ValueError, match="bump must be positive"):
        gamma_finite_difference_delta(option.spot, 0.0, z, option)
    with pytest.raises(ValueError, match="temperature must be positive"):
        gamma_smoothed_autodiff(option.spot, z, option, 0.0)
