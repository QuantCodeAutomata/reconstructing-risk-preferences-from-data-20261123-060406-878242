"""Discrete evaluation of the four risk-measure families used by the
synthetic user in Section 3 ("Reconstructing risk preferences from data").

Every risk measure is evaluated on a sample of I equally likely outcomes of a
contingent claim X (a column vector of length I) through the spectral/
distortion representation

    rho(X) = - sum_{i=1}^{I} x_(i) * (g(i / I) - g((i - 1) / I)),

where x_(1) <= ... <= x_(I) are the sorted values of X and g is the concave
distortion of the family (Sec. 3, p. 10-11).
"""

from __future__ import annotations

import numpy as np

from src.distortions import DISTORTIONS


def distortion_weights(family: str, param: float, n_points: int) -> np.ndarray:
    """Weights w_i = g(i/I) - g((i-1)/I), i = 1..I, for the sorted sample."""
    g = DISTORTIONS[family]
    u = np.arange(0, n_points + 1) / n_points
    gu = g(u, param)
    return np.diff(gu)


def risk_measure(x: np.ndarray, family: str, param: float) -> float | np.ndarray:
    """Evaluate rho(X) on one or several equally-weighted discrete samples.

    Parameters
    ----------
    x : array of shape (I,) or (I, n_claims)
        Sample values of the claim(s) X, one column per claim.
    family : one of "ES", "power", "exponential", "Wang".
    param : the family's risk-aversion parameter (alpha, r, beta, lambda).

    Returns
    -------
    A scalar if x is 1-D, else an array of shape (n_claims,).
    """
    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    n_points = x.shape[0]
    weights = distortion_weights(family, param, n_points)
    x_sorted = np.sort(x, axis=0)
    values = -np.einsum("i,ij->j", weights, x_sorted)
    return values[0] if squeeze else values


def risk_measure_es(x: np.ndarray, alpha: float) -> float | np.ndarray:
    """Convenience wrapper for the ES_alpha benchmark risk measure."""
    return risk_measure(x, "ES", alpha)
