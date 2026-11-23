"""Shared pipeline reproducing Section 3 of "Reconstructing risk preferences
from data": fit the universal risk measure of Theorem 2.6 / Appendix B to
synthetic user quotes and measure training, interpolation-test and
out-of-distribution (OOD) extrapolation error, for each of the four
risk-measure families of p. 10-11.

Experiments exp_1 (GBM), exp_2 (Heston) and exp_3 (discrete) all call
`run_risk_field_experiment`; exp_4 calls `run_incompatible_experiment`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.config import (
    ALPHA_BENCHMARK,
    ALPHA_RANGE,
    BETA_RANGE,
    LAMBDA_RANGE,
    N_REPS_RUN,
    N_TEST_INTERP,
    N_TEST_OOD_EACH_SIDE,
    N_TRAIN_PAPER,
    R_RANGE,
    STRIKE_QUANTILE_HIGH,
    STRIKE_QUANTILE_LOW,
)
from src.distortions import global_lipschitz_constant
from src.evaluation import (
    equally_spaced_strikes,
    mean_sd,
    mse,
    ood_strikes,
    short_call_payoffs,
    uniform_strikes,
)
from src.reconstruction import fit_universal_risk_measure, optimal_lipschitz_constant
from src.risk_measures import risk_measure

FAMILY_PARAM_RANGES = {
    "ES": ALPHA_RANGE,
    "power": R_RANGE,
    "exponential": BETA_RANGE,
    "Wang": LAMBDA_RANGE,
}


def draw_family_param(family: str, rng: np.random.Generator) -> float:
    """Draw the risk-aversion parameter of a family uniformly (Sec. 3, p. 11)."""
    low, high = FAMILY_PARAM_RANGES[family]
    return float(rng.uniform(low, high))


def strike_quantiles(s1_quantile_sample: np.ndarray) -> tuple[float, float]:
    """K_1, K_N as the 0.1- and 0.9-quantiles of S_1 (Sec. 3 footnote 2)."""
    k1 = float(np.quantile(s1_quantile_sample, STRIKE_QUANTILE_LOW))
    kn = float(np.quantile(s1_quantile_sample, STRIKE_QUANTILE_HIGH))
    return k1, kn


def run_risk_field_experiment(
    s1_sample: np.ndarray,
    k1: float,
    kn: float,
    m_dual: int,
    rng: np.random.Generator,
    n_reps: int = N_REPS_RUN,
    n_train: int = N_TRAIN_PAPER,
) -> pd.DataFrame:
    """Reproduce Tables 1-3: reconstruction performance under a risk field.

    Parameters
    ----------
    s1_sample : array (B,)
        Equally weighted Monte Carlo / grid sample of S_1 used as the
        discretized probability space for the dual reconstruction.
    k1, kn : training strike range bounds (quantiles of S_1).
    m_dual : number of sampled dual directions (Appendix B).
    rng : numpy random Generator.
    n_reps : repetitions per family (paper: 30).
    n_train : number of training strikes N (paper: 100).

    Returns
    -------
    DataFrame with one row per (family, repetition) and columns
    ["family", "rep", "param", "lipschitz", "train_mse", "test_mse",
    "ood_ratio"].
    """
    k_train = equally_spaced_strikes(k1, kn, n_train)
    x_train_all = short_call_payoffs(s1_sample, k_train)  # (B, n_train), fixed across reps

    records = []
    for family in FAMILY_PARAM_RANGES:
        for rep in range(n_reps):
            param = draw_family_param(family, rng)
            y_train = risk_measure(x_train_all, family, param)
            lipschitz_bound = global_lipschitz_constant(family, param)
            lipschitz = optimal_lipschitz_constant(x_train_all, y_train, lipschitz_bound)
            urm = fit_universal_risk_measure(x_train_all, y_train, lipschitz, m_dual, rng)

            train_pred = urm.evaluate_exact(x_train_all)
            train_mse = mse(train_pred, y_train)

            k_test = uniform_strikes(k1, kn, N_TEST_INTERP, rng)
            x_test = short_call_payoffs(s1_sample, k_test)
            y_test = risk_measure(x_test, family, param)
            test_pred = urm.evaluate_exact(x_test)
            test_mse = mse(test_pred, y_test)

            k_ood = ood_strikes(k1, kn, N_TEST_OOD_EACH_SIDE, rng)
            x_ood = short_call_payoffs(s1_sample, k_ood)
            y_ood_true = risk_measure(x_ood, family, param)
            ood_pred = urm.evaluate(x_ood)
            ood_mse_recon = mse(ood_pred, y_ood_true)

            y_ood_bench = risk_measure(x_ood, "ES", ALPHA_BENCHMARK)
            ood_mse_bench = mse(y_ood_bench, y_ood_true)
            ood_ratio = ood_mse_recon / ood_mse_bench if ood_mse_bench > 0 else np.nan

            records.append(
                {
                    "family": family,
                    "rep": rep,
                    "param": param,
                    "lipschitz": lipschitz,
                    "train_mse": train_mse,
                    "test_mse": test_mse,
                    "ood_ratio": ood_ratio,
                }
            )
    return pd.DataFrame.from_records(records)


def summarize_risk_field_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-repetition results into mean +/- 1 sample SD per family."""
    rows = []
    for family, group in results.groupby("family"):
        train_mean, train_sd = mean_sd(group["train_mse"].to_numpy())
        test_mean, test_sd = mean_sd(group["test_mse"].to_numpy())
        ood_mean, ood_sd = mean_sd(group["ood_ratio"].to_numpy())
        rows.append(
            {
                "family": family,
                "train_mse_mean": train_mean,
                "train_mse_sd": train_sd,
                "test_mse_mean": test_mean,
                "test_mse_sd": test_sd,
                "ood_ratio_mean": ood_mean,
                "ood_ratio_sd": ood_sd,
            }
        )
    return pd.DataFrame(rows)


def run_incompatible_experiment(
    s1_sample: np.ndarray,
    gamma_grid: np.ndarray,
    alpha: float,
    m_dual: int,
    rng: np.random.Generator,
    n_reps: int,
    n_train: int,
    strike_low: float,
    strike_high: float,
) -> pd.DataFrame:
    """Reproduce Sec. 3.2 (Eq. 3.2): reconstruction error as a function of the
    risk-field violation strength gamma.

    The user's perturbed preference is
        y_n = ES_alpha((S_1 - K_n)_+) + gamma * (1 - K_n^2) / 2,
    which is convex and cash-additive (hence induces a risk field) iff
    gamma = 0 (Sec. 3.2, p. 13).
    """
    records = []
    lipschitz_bound = global_lipschitz_constant("ES", alpha)
    for rep in range(n_reps):
        # The strike set (and hence the Lipschitz constant L_*(D_N) computed
        # from the gamma=0, genuinely risk-field data) is held fixed across
        # the gamma grid for a given repetition (Sec. 3.2, p. 13). Footnote 3
        # shows that for N >= 2 and gamma > 0 no convex, cash-additive risk
        # measure is compatible with D_N, i.e. L_*(D_N) is not finite for
        # gamma > 0: "the best available L" (methodology step (d)) is thus
        # the L_*(D_N) computed once from the gamma = 0 data, reused as-is.
        #
        # Eq. 3.2: y_n = ES_alpha((S_1-K_n)_+) + gamma*(1-K_n^2)/2, i.e. ES is
        # applied to the *long* call payoff (S_1-K_n)_+ = -X_n, not to the
        # claim X_n = -(S_1-K_n)_+ itself (Eq. 3.3: ES_alpha(-X) + ...).
        k_train = np.sort(rng.uniform(strike_low, strike_high, size=n_train))
        x_train = short_call_payoffs(s1_sample, k_train)
        es_part = risk_measure(-x_train, "ES", alpha)
        y_train_0 = es_part  # gamma = 0 is a genuine risk field (Sec. 3.2)
        lipschitz = optimal_lipschitz_constant(x_train, y_train_0, lipschitz_bound)

        for gamma in gamma_grid:
            y_train = es_part + gamma * (1.0 - k_train ** 2) / 2.0

            urm = fit_universal_risk_measure(x_train, y_train, lipschitz, m_dual, rng)
            train_pred = urm.evaluate_exact(x_train)
            train_mse = mse(train_pred, y_train)

            k_test = np.sort(rng.uniform(strike_low, strike_high, size=n_train // 10 or 1))
            x_test = short_call_payoffs(s1_sample, k_test)
            es_part_test = risk_measure(-x_test, "ES", alpha)
            y_test = es_part_test + gamma * (1.0 - k_test ** 2) / 2.0
            test_pred = urm.evaluate_exact(x_test)
            test_mse = mse(test_pred, y_test)

            records.append(
                {
                    "gamma": gamma,
                    "rep": rep,
                    "train_mse": train_mse,
                    "test_mse": test_mse,
                }
            )
    return pd.DataFrame.from_records(records)


def summarize_incompatible_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-repetition results into mean +/- 2 sample SD per gamma
    (the paper reports +/- 2 sample standard deviations in Sec. 3.2)."""
    rows = []
    for gamma, group in results.groupby("gamma"):
        train_mean, train_sd = mean_sd(group["train_mse"].to_numpy())
        test_mean, test_sd = mean_sd(group["test_mse"].to_numpy())
        rows.append(
            {
                "gamma": gamma,
                "train_mse_mean": train_mean,
                "train_mse_sd2": 2 * train_sd,
                "test_mse_mean": test_mean,
                "test_mse_sd2": 2 * test_sd,
            }
        )
    return pd.DataFrame(rows)
