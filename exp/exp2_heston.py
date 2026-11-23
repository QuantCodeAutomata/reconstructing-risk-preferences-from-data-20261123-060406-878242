"""Experiment 2: universal risk measure reconstruction under the Heston
stochastic volatility model for S_1 (Section 3, p. 10, Table 2).

Run: python -m exp.exp2_heston
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from src.config import (
    B_GBM_HESTON,
    HESTON_KAPPA,
    HESTON_MU,
    HESTON_RHO,
    HESTON_S0,
    HESTON_T,
    HESTON_THETA,
    HESTON_V0,
    HESTON_XI,
    HESTON_MC_PATHS_QUANTILE,
    M_DUAL_RUN,
    N_HESTON_STEPS,
    N_REPS_RUN,
    N_TRAIN_PAPER,
    RANDOM_SEED,
)
from src.experiment_runner import (
    run_risk_field_experiment,
    strike_quantiles,
    summarize_risk_field_results,
)
from data.generate_synthetic import simulate_heston

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = "results"

# Expected OOD test ratios reported in Table 2 of the paper (Sec. 3, p. 13),
# used only for the printed comparison, not as inputs to the computation.
PAPER_TABLE_2 = {
    "ES": {"test_mse": 1e-7, "ood_ratio": 0.473},
    "power": {"test_mse": 1e-7, "ood_ratio": 0.461},
    "exponential": {"test_mse": 1e-7, "ood_ratio": 0.392},
    "Wang": {"test_mse": 1e-7, "ood_ratio": 0.580},
}


def _simulate(n_paths: int, rng: np.random.Generator) -> np.ndarray:
    return simulate_heston(
        n_paths, HESTON_MU, HESTON_KAPPA, HESTON_THETA, HESTON_XI, HESTON_RHO,
        HESTON_S0, HESTON_V0, HESTON_T, N_HESTON_STEPS, rng,
    )


def main(n_reps: int = N_REPS_RUN, m_dual: int = M_DUAL_RUN) -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    t0 = time.time()
    quantile_sample = _simulate(HESTON_MC_PATHS_QUANTILE, rng)
    k1, kn = strike_quantiles(quantile_sample)
    s1_sample = _simulate(B_GBM_HESTON, rng)
    log.info(
        "Heston: simulated %d quantile paths and %d discretization points "
        "(%d Euler steps); K1=%.6f, KN=%.6f (0.1-/0.9-quantiles of S_1)",
        HESTON_MC_PATHS_QUANTILE, B_GBM_HESTON, N_HESTON_STEPS, k1, kn,
    )

    results = run_risk_field_experiment(
        s1_sample, k1, kn, m_dual, rng, n_reps=n_reps, n_train=N_TRAIN_PAPER
    )
    summary = summarize_risk_field_results(results)
    elapsed = time.time() - t0
    log.info("exp_2 (Heston) finished in %.1f s", elapsed)

    results.to_csv(f"{RESULTS_DIR}/exp2_heston_raw.csv", index=False)
    summary.to_csv(f"{RESULTS_DIR}/exp2_heston_summary.csv", index=False)

    print("\nExperiment 2 (Heston) -- reconstruction vs. Table 2 of the paper")
    print(summary.to_string(index=False))
    for family, ref in PAPER_TABLE_2.items():
        row = summary[summary["family"] == family].iloc[0]
        print(
            f"{family}: test_mse={row['test_mse_mean']:.3e} (paper ~{ref['test_mse']:.0e}), "
            f"ood_ratio={row['ood_ratio_mean']:.3f} (paper ~{ref['ood_ratio']:.3f})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full", action="store_true",
        help="Run the exact paper configuration (no-op here: exp_2 already "
        "runs the full N_TRAIN_PAPER, N_REPS_PAPER, M_DUAL_PAPER configuration).",
    )
    args = parser.parse_args()
    main()
