"""Experiment 1: universal risk measure reconstruction under a Geometric
Brownian Motion model for S_1 (Section 3, p. 10, Table 1).

Run: python -m exp.exp1_gbm
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from src.config import (
    ALPHA_BENCHMARK,
    GBM_MC_PATHS_QUANTILE,
    GBM_MU,
    GBM_S0,
    GBM_SIGMA,
    GBM_T,
    B_GBM_HESTON,
    M_DUAL_RUN,
    N_REPS_RUN,
    N_TRAIN_PAPER,
    RANDOM_SEED,
)
from src.experiment_runner import (
    run_risk_field_experiment,
    strike_quantiles,
    summarize_risk_field_results,
)
from data.generate_synthetic import simulate_gbm

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = "results"

# Expected OOD test ratios reported in Table 1 of the paper (Sec. 3, p. 12),
# used only for the printed comparison, not as inputs to the computation.
PAPER_TABLE_1 = {
    "ES": {"test_mse": 1e-10, "ood_ratio": 0.382},
    "power": {"test_mse": 1e-10, "ood_ratio": 0.374},
    "exponential": {"test_mse": 1e-10, "ood_ratio": 0.319},
    "Wang": {"test_mse": 1e-10, "ood_ratio": 0.460},
}


def main(n_reps: int = N_REPS_RUN, m_dual: int = M_DUAL_RUN) -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    t0 = time.time()
    quantile_sample = simulate_gbm(
        GBM_MC_PATHS_QUANTILE, GBM_MU, GBM_SIGMA, GBM_S0, GBM_T, rng
    )
    k1, kn = strike_quantiles(quantile_sample)
    s1_sample = simulate_gbm(B_GBM_HESTON, GBM_MU, GBM_SIGMA, GBM_S0, GBM_T, rng)
    log.info(
        "GBM: simulated %d quantile paths and %d discretization points; "
        "K1=%.6f, KN=%.6f (0.1-/0.9-quantiles of S_1)",
        GBM_MC_PATHS_QUANTILE, B_GBM_HESTON, k1, kn,
    )

    results = run_risk_field_experiment(
        s1_sample, k1, kn, m_dual, rng, n_reps=n_reps, n_train=N_TRAIN_PAPER
    )
    summary = summarize_risk_field_results(results)
    elapsed = time.time() - t0
    log.info("exp_1 (GBM) finished in %.1f s", elapsed)

    results.to_csv(f"{RESULTS_DIR}/exp1_gbm_raw.csv", index=False)
    summary.to_csv(f"{RESULTS_DIR}/exp1_gbm_summary.csv", index=False)

    print("\nExperiment 1 (GBM) -- reconstruction vs. Table 1 of the paper")
    print(summary.to_string(index=False))
    for family, ref in PAPER_TABLE_1.items():
        row = summary[summary["family"] == family].iloc[0]
        print(
            f"{family}: test_mse={row['test_mse_mean']:.3e} (paper ~{ref['test_mse']:.0e}), "
            f"ood_ratio={row['ood_ratio_mean']:.3f} (paper ~{ref['ood_ratio']:.3f})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full", action="store_true",
        help="Run the exact paper configuration (no-op here: exp_1 already "
        "runs the full N_TRAIN_PAPER, N_REPS_PAPER, M_DUAL_PAPER configuration).",
    )
    args = parser.parse_args()
    main()
