"""Synthetic sample paths of the underlying S_1 used in the numerical
illustration of "Reconstructing risk preferences from data" (Sec. 3, p. 10).

No market data is used anywhere in this repository: every experiment
evaluates the universal risk measure reconstruction on a synthetic
underlying simulated under one of three models (GBM, Heston, a finite
state-space / discrete model).
"""

from __future__ import annotations

import numpy as np


def simulate_gbm(
    n_paths: int,
    mu: float,
    sigma: float,
    s0: float,
    t: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Exact simulation of S_1 under GBM: dS_t = mu S_t dt + sigma S_t dW_t.

    Uses the closed-form solution S_t = S_0 exp((mu - sigma^2/2) t + sigma W_t)
    (Sec. 3, p. 10), so there is no discretization error.
    """
    z = rng.standard_normal(n_paths)
    w_t = np.sqrt(t) * z
    return s0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * w_t)


def simulate_heston(
    n_paths: int,
    mu: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    s0: float,
    v0: float,
    t: float,
    n_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Euler-Maruyama simulation of S_1 under the Heston model (Sec. 3, p. 10):

        dS_t = mu S_t dt + sqrt(v_t) S_t dW_t^(1)
        dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW_t^(2)
        d<W^(1), W^(2)>_t = rho dt

    Full-truncation Euler scheme (v_t replaced by max(v_t, 0) inside the
    square roots) avoids negative variances; the discretization scheme itself
    is not specified in the paper (flagged as an open detail).
    """
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)
    s = np.full(n_paths, s0, dtype=np.float64)
    v = np.full(n_paths, v0, dtype=np.float64)
    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z_indep = rng.standard_normal(n_paths)
        z2 = rho * z1 + np.sqrt(1.0 - rho ** 2) * z_indep
        v_pos = np.maximum(v, 0.0)
        sqrt_v = np.sqrt(v_pos)
        s = s * np.exp((mu - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * z1)
        v = v + kappa * (theta - v_pos) * dt + xi * sqrt_v * sqrt_dt * z2
    return s


def discrete_grid(n_points: int, low: float, width: float) -> np.ndarray:
    """The exact, equally-likely grid of Eq. 3.1: s_i = low + width * i / I."""
    i = np.arange(n_points)
    return low + width * i / n_points
