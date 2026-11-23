"""Experiment 3: universal risk measure reconstruction under the finite
state-space (discrete) model for S_1 (Section 3, Eq. 3.1, Table 3).

Run: python -m exp.exp3_discrete
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from src.config import (
    DISCRETE_I,
    DISCRETE_K1,
    DISCRETE_KN,
    DISCRETE_LOW,
    DISCRETE_WIDTH,
    M_DUAL_RUN,
    N_REPS_RUN,
    N_TRAIN_PAPER,
    RANDOM_SEED,
)
from src.experiment_runner import run_risk_field_experiment, summarize_risk_field_results
from data.generate_synthetic import discrete_grid

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = "results"

# Expected orders of magnitude reported in Table 3 of the paper (Sec. 3,
# p. 12), used only for the printed comparison, not as inputs.
PAPER_TABLE_3 = {
    "ES": {"test_mse": 1e-30, "ood_ratio": 1e-30},
    "power": {"test_mse": 1e-30, "ood_ratio": 1e-30},
    "exponential": {"test_mse": 1e-30, "ood_ratio": 1e-30},
    "Wang": {"test_mse": 1e-30, "ood_ratio": 1e-30},
}


def main(n_reps: int = N_REPS_RUN, m_dual: int = M_DUAL_RUN) -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    t0 = time.time()
    s1_sample = discrete_grid(DISCRETE_I, DISCRETE_LOW, DISCRETE_WIDTH)
    k1, kn = DISCRETE_K1, DISCRETE_KN
    log.info(
        "Discrete grid: I=%d equally likely points in [%.1f, %.1f); "
        "fixed training strikes K1=%.2f, KN=%.2f (Eq. 3.1)",
        DISCRETE_I, DISCRETE_LOW, DISCRETE_LOW + DISCRETE_WIDTH, k1, kn,
    )

    results = run_risk_field_experiment(
        s1_sample, k1, kn, m_dual, rng, n_reps=n_reps, n_train=N_TRAIN_PAPER
    )
    summary = summarize_risk_field_results(results)
    elapsed = time.time() - t0
    log.info("exp_3 (discrete) finished in %.1f s", elapsed)

    results.to_csv(f"{RESULTS_DIR}/exp3_discrete_raw.csv", index=False)
    summary.to_csv(f"{RESULTS_DIR}/exp3_discrete_summary.csv", index=False)

    print("\nExperiment 3 (discrete) -- reconstruction vs. Table 3 of the paper")
    print(summary.to_string(index=False))
    for family, ref in PAPER_TABLE_3.items():
        row = summary[summary["family"] == family].iloc[0]
        print(
            f"{family}: test_mse={row['test_mse_mean']:.3e} (paper ~{ref['test_mse']:.0e}), "
            f"ood_ratio={row['ood_ratio_mean']:.3e} (paper ~{ref['ood_ratio']:.0e})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full", action="store_true",
        help="Run the exact paper configuration (no-op here: exp_3 already "
        "runs the full N_TRAIN_PAPER, N_REPS_PAPER, M_DUAL_PAPER configuration).",
    )
    args = parser.parse_args()
    main()
