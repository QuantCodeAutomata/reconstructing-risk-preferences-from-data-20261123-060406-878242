# Repository guide for coding agents

This repository reproduces the numerical illustrations of Section 3 of
"Reconstructing risk preferences from data" (universal risk measure
reconstruction, Theorem 2.6 / Appendix B).

## Layout

* `src/` — flat library modules: `distortions.py` (concave distortion
  functions and global Lipschitz constants for the four risk-measure
  families), `risk_measures.py` (discrete spectral risk-measure evaluation),
  `reconstruction.py` (Frank-Wolfe solver for the universal risk measure of
  Eq. 2.2 and the optimal Lipschitz constant L_*(D_N) of Definition 2.3),
  `evaluation.py` (strike sampling and MSE helpers), `experiment_runner.py`
  (the one shared pipeline called by every `exp/` script), `config.py` (all
  parameters, as module-level `UPPER_CASE` variables with a source comment).
* `exp/` — one script per experiment (`exp1_gbm.py`, `exp2_heston.py`,
  `exp3_discrete.py`, `exp4_incompatible.py`), each calling only the
  functions in `src/experiment_runner.py` with no private shortcuts.
* `data/generate_synthetic.py` — simulates S_1 under GBM, Heston, or an
  explicit finite grid. No market data is used anywhere in this repository.
* `tests/` — pytest unit tests for `src/` and `data/`.
* `results/` — CSV tables, plots and `RESULTS.md`, overwritten by the `exp/`
  scripts on every rerun; the numbers and verdicts in `RESULTS.md` are
  checked against the newly written files and updated by hand after a rerun.

## Commands

```bash
pip install -r requirements.txt
python -m pytest -m "not market_data"   # unit tests (no live data is ever used)
python -m exp.exp1_gbm                  # Table 1 (GBM)
python -m exp.exp2_heston               # Table 2 (Heston)
python -m exp.exp3_discrete             # Table 3 (finite grid)
python -m exp.exp4_incompatible         # Sec. 3.2, Eq. 3.2
bash reproduce.sh                       # tests + all four experiments
```

Each of `exp1`-`exp3` accepts `--full` to force the paper's exact
configuration (N = 100, 30 repetitions, M_dual = 10^5).

## Conventions

* All experiment parameters live only in `src/config.py` as module-level
  variables; no classes or parameter dicts.
* `exp/` scripts call only the signatures exposed by `src/experiment_runner.py`
  and `src/reconstruction.py`; they do not duplicate the reconstruction logic.
* No market data is used by any experiment, so there is no `data_loader.py`
  and no API keys are required.
* Deviations from the paper and any experiment not reproduced are recorded
  only in `results/RESULTS.md`, never in code comments or config values.
