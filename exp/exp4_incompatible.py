"""Experiment 4: universal risk measure reconstruction on incompatible
(non-risk-field) data (Section 3.2, Eq. 3.2, p. 13-14).

Run: python -m exp.exp4_incompatible
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd

from src.config import (
    EXP4_ALPHA,
    EXP4_GRID_LOW,
    EXP4_GRID_WIDTH,
    EXP4_I,
    EXP4_N_REPS,
    EXP4_N_TRAIN,
    EXP4_STRIKE_HIGH,
    EXP4_STRIKE_LOW,
    GAMMA_GRID,
    M_DUAL_RUN,
    RANDOM_SEED,
)
from src.experiment_runner import run_incompatible_experiment, summarize_incompatible_results
from data.generate_synthetic import discrete_grid

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = "results"

# Approximate training/test MSE reported in Sec. 3.2 (p. 14) as a function of
# gamma, used only for the printed comparison, not as inputs.
PAPER_TRAIN_MSE_1E7 = [0.0, 0.1, 0.3, 0.7, 1.2, 1.8, 2.6, 3.6, 4.7, 5.9, 7.3]
PAPER_TEST_MSE_1E6 = [3.30, 3.31, 3.33, 3.37, 3.40, 3.45, 3.50, 3.56, 3.61, 3.67, 3.74]


def main(n_reps: int = EXP4_N_REPS, m_dual: int = M_DUAL_RUN) -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    t0 = time.time()
    s1_sample = discrete_grid(EXP4_I, EXP4_GRID_LOW, EXP4_GRID_WIDTH)
    log.info(
        "exp_4: discrete grid I=%d in [%.1f, %.1f); strikes uniform in (%.2f, %.2f), "
        "gamma in %s, alpha=%.2f",
        EXP4_I, EXP4_GRID_LOW, EXP4_GRID_LOW + EXP4_GRID_WIDTH,
        EXP4_STRIKE_LOW, EXP4_STRIKE_HIGH, GAMMA_GRID.tolist(), EXP4_ALPHA,
    )

    results = run_incompatible_experiment(
        s1_sample, GAMMA_GRID, EXP4_ALPHA, m_dual, rng,
        n_reps=n_reps, n_train=EXP4_N_TRAIN,
        strike_low=EXP4_STRIKE_LOW, strike_high=EXP4_STRIKE_HIGH,
    )
    summary = summarize_incompatible_results(results)
    elapsed = time.time() - t0
    log.info("exp_4 (incompatible data) finished in %.1f s", elapsed)

    results.to_csv(f"{RESULTS_DIR}/exp4_incompatible_raw.csv", index=False)
    summary.to_csv(f"{RESULTS_DIR}/exp4_incompatible_summary.csv", index=False)

    paper_ref = pd.DataFrame(
        {
            "gamma": GAMMA_GRID,
            "paper_train_mse": np.array(PAPER_TRAIN_MSE_1E7) * 1e-7,
            "paper_test_mse": np.array(PAPER_TEST_MSE_1E6) * 1e-6,
        }
    )
    print("\nExperiment 4 (incompatible data) -- reconstruction vs. Sec. 3.2 of the paper")
    merged = summary.merge(paper_ref, on="gamma")
    print(merged.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full", action="store_true",
        help="Run the exact paper configuration (no-op here: exp_4 already "
        "runs the full configuration).",
    )
    args = parser.parse_args()
    main()
