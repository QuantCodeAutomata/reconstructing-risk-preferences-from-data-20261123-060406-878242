"""Unit tests for src/distortions.py: concave distortions g and the implied
global Lipschitz constants (Sec. 3, p. 10-11, and Proposition 2.15)."""

import numpy as np

from src.distortions import (
    distortion_es,
    distortion_exponential,
    distortion_power,
    distortion_wang,
    global_lipschitz_constant,
)


def test_distortion_boundary_conditions():
    """Every concave distortion satisfies g(0) = 0, g(1) = 1 (Sec. 3, p. 10)."""
    for g, param in [
        (distortion_es, 0.7),
        (distortion_power, 3.0),
        (distortion_exponential, 2.0),
        (distortion_wang, 1.0),
    ]:
        assert np.isclose(g(np.array([0.0]), param)[0], 0.0, atol=1e-10)
        assert np.isclose(g(np.array([1.0]), param)[0], 1.0, atol=1e-6)


def test_distortion_monotone_nondecreasing():
    """g must be non-decreasing on [0, 1] for a valid distortion."""
    u = np.linspace(0.0, 1.0, 200)
    for g, param in [
        (distortion_es, 0.6),
        (distortion_power, 4.0),
        (distortion_exponential, 3.0),
        (distortion_wang, 0.8),
    ]:
        values = g(u, param)
        assert np.all(np.diff(values) >= -1e-10)


def test_es_global_lipschitz_matches_closed_form():
    """ES_alpha's global L^2_P Lipschitz constant equals sqrt(alpha/(1-alpha))
    (Proposition 2.15 / p. 9 footnote), matching the integral computation."""
    for alpha in [0.5, 0.7, 0.9]:
        expected = np.sqrt(alpha / (1.0 - alpha))
        computed = global_lipschitz_constant("ES", alpha)
        assert np.isclose(computed, expected, rtol=1e-3)


def test_lipschitz_constants_are_positive_and_finite():
    for family, param in [("ES", 0.8), ("power", 5.0), ("exponential", 4.0), ("Wang", 1.2)]:
        value = global_lipschitz_constant(family, param)
        assert np.isfinite(value)
        assert value > 0.0
