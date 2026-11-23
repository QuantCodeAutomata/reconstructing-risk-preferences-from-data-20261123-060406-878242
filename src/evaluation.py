"""Shared evaluation helpers: strike sampling and MSE computations used by
every experiment in Section 3 of "Reconstructing risk preferences from data".
"""

from __future__ import annotations

import numpy as np


def equally_spaced_strikes(k1: float, kn: float, n: int) -> np.ndarray:
    """N training strikes equally spaced in [K_1, K_N] (Sec. 3, p. 10)."""
    return np.linspace(k1, kn, n)


def uniform_strikes(low: float, high: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """n strikes drawn uniformly from (low, high)."""
    return rng.uniform(low, high, size=n)


def ood_strikes(
    k1: float,
    kn: float,
    n_each_side: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Extrapolation strikes: n_each_side below K_1 and n_each_side above K_N,
    with d = (K_N - K_1) / 4 (Sec. 3 footnote 2)."""
    d = (kn - k1) / 4.0
    low_side = rng.uniform(max(0.0, k1 - d), k1, size=n_each_side)
    high_side = rng.uniform(kn, kn + d, size=n_each_side)
    return np.concatenate([low_side, high_side])


def short_call_payoffs(s1: np.ndarray, strikes: np.ndarray) -> np.ndarray:
    """X_n = -(S_1 - K_n)_+ for every strike, shape (B, n_strikes)."""
    diff = s1[:, None] - strikes[None, :]
    return -np.maximum(diff, 0.0)


def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean squared error between a reconstruction and the true quotes."""
    return float(np.mean((np.asarray(predictions) - np.asarray(targets)) ** 2))


def mean_sd(values: np.ndarray) -> tuple[float, float]:
    """Mean and one sample standard deviation (ddof=1) across repetitions."""
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return float(values.mean()), 0.0
    return float(values.mean()), float(values.std(ddof=1))
