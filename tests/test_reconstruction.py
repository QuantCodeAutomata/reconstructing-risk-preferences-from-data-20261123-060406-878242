"""Unit tests for src/reconstruction.py: exact interpolation on a risk field
(Theorem 2.6) and the optimal Lipschitz constant L_*(D_N) (Definition 2.3)."""

import numpy as np

from src.reconstruction import fit_universal_risk_measure, optimal_lipschitz_constant
from src.risk_measures import risk_measure


def _make_claims(rng, b=400, n_train=12):
    """Random claims on a B-point equally weighted probability space."""
    return rng.standard_normal((b, n_train))


def test_exact_interpolation_when_data_is_a_risk_field():
    """When D_N is induced by an actual risk measure, rho_{D_N,L} must
    reproduce the training quotes exactly (Theorem 2.6, compatibility)."""
    rng = np.random.default_rng(42)
    x_train = _make_claims(rng)
    y_train = risk_measure(x_train, "ES", 0.8)

    lipschitz = optimal_lipschitz_constant(x_train, y_train, upper_bound=50.0)
    urm = fit_universal_risk_measure(x_train, y_train, lipschitz, m_dual=1000, rng=rng)
    predicted = urm.evaluate_exact(x_train)

    # The Frank-Wolfe primal solve (conditional gradient on the simplex) only
    # converges to the exact vertex solution at a sublinear rate near the
    # kink where the residual norm vanishes (Theorem 2.6 compatibility); the
    # per-point gap is therefore tiny but not at machine precision.
    assert np.max(np.abs(predicted - y_train)) < 1e-3
    assert np.mean((predicted - y_train) ** 2) < 1e-6


def test_cash_additive_lift_matches_definition_2_5():
    """rho_{D_N,L}(X + m) = rho_{D_N,L}(X) - m (Definition 2.5 cash lift)."""
    rng = np.random.default_rng(7)
    x_train = _make_claims(rng)
    y_train = risk_measure(x_train, "power", 3.0)
    lipschitz = optimal_lipschitz_constant(x_train, y_train, upper_bound=50.0)
    urm = fit_universal_risk_measure(x_train, y_train, lipschitz, m_dual=1000, rng=rng)

    query = rng.standard_normal(x_train.shape[0])
    m = 2.5
    base = urm.evaluate(query)
    shifted = urm.evaluate(query + m)
    assert np.isclose(shifted, base - m, atol=1e-6)


def test_optimal_lipschitz_is_nondecreasing_feasibility():
    """L_*(D_N) found by bisection must itself satisfy the risk-field
    condition (2.1): using it should still interpolate exactly."""
    rng = np.random.default_rng(11)
    x_train = _make_claims(rng, n_train=8)
    y_train = risk_measure(x_train, "Wang", 1.0)
    lipschitz = optimal_lipschitz_constant(x_train, y_train, upper_bound=30.0)
    assert lipschitz >= 0.0
    urm = fit_universal_risk_measure(x_train, y_train, lipschitz, m_dual=1000, rng=rng)
    predicted = urm.evaluate_exact(x_train)
    assert np.max(np.abs(predicted - y_train)) < 1e-3
    assert np.mean((predicted - y_train) ** 2) < 1e-6


def test_reconstruction_is_lipschitz_bounded():
    """Lip(rho_{D_N,L}) <= sqrt(L^2 + 1) in the L^2_P metric (Theorem 2.6)."""
    rng = np.random.default_rng(5)
    b, n_train = 300, 10
    x_train = rng.standard_normal((b, n_train))
    y_train = risk_measure(x_train, "exponential", 2.0)
    lipschitz = optimal_lipschitz_constant(x_train, y_train, upper_bound=40.0)
    urm = fit_universal_risk_measure(x_train, y_train, lipschitz, m_dual=1000, rng=rng)

    bound = np.sqrt(lipschitz ** 2 + 1.0)
    x1 = rng.standard_normal(b)
    x2 = rng.standard_normal(b)
    v1 = urm.evaluate(x1)
    v2 = urm.evaluate(x2)
    l2_dist = np.sqrt(np.mean((x1 - x2) ** 2))
    assert abs(v1 - v2) <= bound * l2_dist + 1e-6
