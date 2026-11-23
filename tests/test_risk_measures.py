"""Unit tests for src/risk_measures.py: discrete evaluation of the four risk
measure families used by the synthetic user (Sec. 3, p. 10-11)."""

import numpy as np
import pytest

from src.risk_measures import risk_measure


@pytest.fixture
def constant_loss():
    """A deterministic claim X = -5 (certain loss of 5) on I=1000 points."""
    return np.full(1000, -5.0)


def test_normalized_at_zero():
    """rho(0) = 0 for a cash-additive, normalized risk measure (Definition 2.1)."""
    x = np.zeros(500)
    for family, param in [("ES", 0.8), ("power", 4.0), ("exponential", 2.0), ("Wang", 1.0)]:
        assert np.isclose(risk_measure(x, family, param), 0.0, atol=1e-10)


def test_cash_additivity(constant_loss):
    """rho(X + m) = rho(X) - m for all m (Definition 2.1 (iii))."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    m = 3.0
    for family, param in [("ES", 0.7), ("power", 2.5), ("exponential", 1.5), ("Wang", 0.9)]:
        base = risk_measure(x, family, param)
        shifted = risk_measure(x + m, family, param)
        assert np.isclose(shifted, base - m, atol=1e-8)


def test_constant_claim_equals_negative_constant(constant_loss):
    """For a certain outcome X = c, every normalized risk measure gives -c."""
    for family, param in [("ES", 0.6), ("power", 3.0), ("exponential", 2.0), ("Wang", 1.1)]:
        value = risk_measure(constant_loss, family, param)
        assert np.isclose(value, 5.0, atol=1e-8)


def test_monotonicity_worse_outcome_higher_risk():
    """rho(X) <= rho(Y) whenever X >= Y (Definition 2.1 (ii)): a uniformly
    worse claim must have a higher (more conservative) risk evaluation."""
    rng = np.random.default_rng(1)
    base = rng.standard_normal(1000)
    worse = base - 1.0  # worse P-a.s.
    for family, param in [("ES", 0.75), ("power", 3.0), ("exponential", 2.0), ("Wang", 1.0)]:
        assert risk_measure(worse, family, param) >= risk_measure(base, family, param) - 1e-8


def test_batched_evaluation_matches_columnwise():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((500, 7))
    batched = risk_measure(x, "power", 3.5)
    for j in range(7):
        single = risk_measure(x[:, j], "power", 3.5)
        assert np.isclose(batched[j], single, atol=1e-10)


def test_es_matches_tail_average_definition():
    """ES_alpha(X) for the discrete sample equals minus the average of the
    (1 - alpha)*I smallest (worst) values when (1 - alpha)*I is an integer.

    Eq. (2.8)-(2.9) define ES_alpha(X) = sup_{Z} E[ZX] over densities Z with
    E[Z] = 1 and 0 <= (1 - alpha) Z <= 1 a.s.; the extremal density puts
    weight 1/(1-alpha) on the event A with P(A) = 1 - alpha where X is
    smallest, so ES_alpha(X) is minus the average of the worst (1 - alpha)
    fraction of outcomes, matching the distortion g(u) = min(u/(1-alpha), 1)
    used in src/distortions.py.
    """
    rng = np.random.default_rng(3)
    x = rng.standard_normal(1000)
    alpha = 0.9
    # round(), not int(), avoids truncating 999...8 from the float (1 - 0.9)
    # one unit short of the intended tail size.
    n_tail = round((1.0 - alpha) * 1000)
    expected = -np.mean(np.sort(x)[:n_tail])
    computed = risk_measure(x, "ES", alpha)
    assert np.isclose(computed, expected, atol=1e-6)
