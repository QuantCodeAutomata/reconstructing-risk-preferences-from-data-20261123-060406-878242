"""Concave distortion functions for the four risk-measure families of Sec. 3
("Reconstructing risk preferences from data", p. 10-11) and the implied
global Lipschitz constant of each family's dual representation.

Each family corresponds to a concave distortion g: [0, 1] -> [0, 1] with
g(0) = 0, g(1) = 1, used in the spectral-risk-measure representation

    rho_g(X) = -sum_i x_(i) * (g(i/I) - g((i - 1) / I)),

where x_(1) <= ... <= x_(I) are the sorted values of X on a discrete,
equally-weighted sample of size I (Sec. 3, p. 10-11).

The acceptance sets defining ES, power spectral, exponential spectral and
Wang risk measures (p. 10-11) are {Q : Q(A) <= g(P(A)) for all A}. The
extremal density attaining the dual supremum has quantile function
h(u) = g'(1 - u); by the Cauchy-Schwarz argument used in the paper to derive
the lower bound L >= sqrt(alpha / (1 - alpha)) for ES (p. 9, Proposition 2.15
footnote), the global L^2_P Lipschitz constant of the risk measure equals

    L_rho = sqrt( integral_0^1 (h(u) - 1)^2 du ).

For ES_alpha this integral evaluates in closed form to alpha / (1 - alpha),
reproducing exactly the bound stated in the paper and confirming the method.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, stats

_NORM = stats.norm


def distortion_es(u: np.ndarray, alpha: float) -> np.ndarray:
    """g(u) = min(u / (1 - alpha), 1), the ES_alpha distortion (p. 11)."""
    return np.minimum(u / (1.0 - alpha), 1.0)


def distortion_power(u: np.ndarray, r: float) -> np.ndarray:
    """g(u) = 1 - (1 - u)^r, the power-spectral distortion (p. 11)."""
    return 1.0 - (1.0 - u) ** r


def distortion_exponential(u: np.ndarray, beta: float) -> np.ndarray:
    """g(u) = (1 - exp(-beta u)) / (1 - exp(-beta)), p. 11."""
    return (1.0 - np.exp(-beta * u)) / (1.0 - np.exp(-beta))


def distortion_wang(u: np.ndarray, lam: float) -> np.ndarray:
    """g(u) = Phi(Phi^{-1}(u) + lambda), the Wang transform, p. 11."""
    u_clipped = np.clip(u, 1e-14, 1.0 - 1e-14)
    return _NORM.cdf(_NORM.ppf(u_clipped) + lam)


# Derivative g'(u) of each distortion, used to build h(u) = g'(1 - u).
def _dg_es(u: np.ndarray, alpha: float) -> np.ndarray:
    return np.where(u < 1.0 - alpha, 1.0 / (1.0 - alpha), 0.0)


def _dg_power(u: np.ndarray, r: float) -> np.ndarray:
    return r * (1.0 - u) ** (r - 1.0)


def _dg_exponential(u: np.ndarray, beta: float) -> np.ndarray:
    return beta * np.exp(-beta * u) / (1.0 - np.exp(-beta))


def _dg_wang(u: np.ndarray, lam: float) -> np.ndarray:
    u_clipped = np.clip(u, 1e-14, 1.0 - 1e-14)
    z = _NORM.ppf(u_clipped)
    return _NORM.pdf(z + lam) / _NORM.pdf(z)


DISTORTIONS = {
    "ES": distortion_es,
    "power": distortion_power,
    "exponential": distortion_exponential,
    "Wang": distortion_wang,
}

_DERIVATIVES = {
    "ES": _dg_es,
    "power": _dg_power,
    "exponential": _dg_exponential,
    "Wang": _dg_wang,
}


def global_lipschitz_constant(family: str, param: float) -> float:
    """Global L^2_P Lipschitz constant L_rho of the risk measure's family.

    Computed as sqrt(integral_0^1 (h(u) - 1)^2 du) with h(u) = g'(1 - u),
    the quantile function of the extremal dual density (see module docstring).
    """
    dg = _DERIVATIVES[family]

    def integrand(u: float) -> float:
        h = dg(1.0 - u, param)
        return (h - 1.0) ** 2

    value, _ = integrate.quad(integrand, 0.0, 1.0, limit=200)
    return float(np.sqrt(max(value, 0.0)))
