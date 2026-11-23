"""Unit tests for data/generate_synthetic.py: the three underlying models of
Section 3 ("Reconstructing risk preferences from data", p. 10)."""

import numpy as np

from data.generate_synthetic import discrete_grid, simulate_gbm, simulate_heston


def test_gbm_mean_matches_closed_form():
    """E[S_1] = S_0 exp(mu t) under GBM (Sec. 3, p. 10)."""
    rng = np.random.default_rng(0)
    s1 = simulate_gbm(500_000, mu=0.01, sigma=0.01, s0=1.0, t=1.0, rng=rng)
    assert s1.shape == (500_000,)
    assert np.all(s1 > 0.0)
    expected_mean = 1.0 * np.exp(0.01 * 1.0)
    assert abs(s1.mean() - expected_mean) < 5e-4


def test_gbm_is_lognormal_positive_support():
    rng = np.random.default_rng(1)
    s1 = simulate_gbm(1000, mu=0.0, sigma=0.2, s0=2.0, t=1.0, rng=rng)
    assert np.all(s1 > 0.0)


def test_heston_variance_stays_nonnegative_and_shape():
    """The full-truncation Euler scheme must keep S_1 positive (Sec. 3, p. 10)."""
    rng = np.random.default_rng(2)
    s1 = simulate_heston(
        2000, mu=0.01, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
        s0=1.0, v0=0.04, t=1.0, n_steps=50, rng=rng,
    )
    assert s1.shape == (2000,)
    assert np.all(np.isfinite(s1))
    assert np.all(s1 > 0.0)


def test_heston_mean_close_to_gbm_like_drift():
    """With a small vol-of-vol the Heston mean should stay near S_0 exp(mu t)."""
    rng = np.random.default_rng(3)
    s1 = simulate_heston(
        200_000, mu=0.01, kappa=2.0, theta=0.04, xi=0.01, rho=0.0,
        s0=1.0, v0=0.04, t=1.0, n_steps=100, rng=rng,
    )
    expected_mean = np.exp(0.01)
    assert abs(s1.mean() - expected_mean) < 5e-3


def test_discrete_grid_bounds_and_uniform_probability():
    """Eq. 3.1: I equally likely points s_i = 1 + 9 i / I in [1, 10)."""
    grid = discrete_grid(1000, low=1.0, width=9.0)
    assert grid.shape == (1000,)
    assert grid.min() == 1.0
    assert grid.max() < 10.0
    assert np.all(np.diff(grid) > 0.0)
    # Equally likely by construction: 1/I each, i.e. a uniform empirical law.
    diffs = np.diff(grid)
    assert np.allclose(diffs, diffs[0])
