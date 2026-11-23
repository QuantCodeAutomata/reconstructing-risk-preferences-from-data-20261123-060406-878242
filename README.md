# Reconstructing risk preferences from data

Replication of the numerical illustrations (Section 3) of **"Reconstructing
risk preferences from data"**, which constructs a universal, finite-valued,
cash-additive, convex risk measure that exactly interpolates a finite set of
elicited quotes whenever the data form a *risk field* (Definition 2.3,
Theorem 2.4, Theorem 2.6).

A robo-advisor queries a user with N = 100 short-position European call
options on an underlying S_1 at equally spaced strikes. The user evaluates
each option with one of four convex risk-measure families (Expected
Shortfall, power spectral, exponential spectral, Wang). The universal risk
measure is reconstructed from these quotes via the dual representation of
Appendix B and evaluated on:

* **Training MSE** — should vanish exactly, since the quotes are induced by a
  risk measure (Theorem 2.6).
* **Test (interpolation) MSE** — on strikes drawn uniformly inside the
  training range.
* **OOD (extrapolation) test ratio** — reconstruction MSE divided by an
  ES_0.99 benchmark MSE, on strikes outside the training range.

No market data is used anywhere in this repository: every experiment
simulates its own underlying process (Geometric Brownian motion, a Heston
stochastic-volatility model, or an explicit finite grid) in
`data/generate_synthetic.py`.

## Experiments

| Script | Paper reference | Underlying model |
|---|---|---|
| `python -m exp.exp1_gbm` | Sec. 3, Table 1 | Geometric Brownian motion |
| `python -m exp.exp2_heston` | Sec. 3, Table 2 | Heston stochastic volatility |
| `python -m exp.exp3_discrete` | Sec. 3, Eq. 3.1, Table 3 | Finite state-space grid, I = 10^3 |
| `python -m exp.exp4_incompatible` | Sec. 3.2, Eq. 3.2 | Finite grid, data that violates the risk-field condition |

Each of `exp1`-`exp3` accepts `--full` to force the exact paper configuration
(N = 100 training quotes, 30 repetitions per family, M_dual = 10^5 sampled
dual directions); by default they already run this configuration unless the
runtime estimate in `results/RESULTS.md` required a reduction, in which case
the reduced and full commands are both given there.

## Repository layout

```
src/   risk-measure families, universal-reconstruction solver, config, shared experiment pipeline
exp/   one script per experiment, calling only the src/ functions above
data/  data/generate_synthetic.py simulates S_1 under each model (no market data)
tests/ unit tests for src/ and data/
results/ CSV tables, plots and RESULTS.md written by the exp/ scripts
```

## Setup

```bash
pip install -r requirements.txt
```

## Tests

```bash
python -m pytest
```

## Running the experiments

```bash
python -m exp.exp1_gbm
python -m exp.exp2_heston
python -m exp.exp3_discrete
python -m exp.exp4_incompatible
```

or, to run everything (tests + all four experiments):

```bash
bash reproduce.sh
```

Results (CSV tables and the comparison against the paper's reported figures)
are written to `results/`; see `results/RESULTS.md` for the final numbers,
deviations from the paper, and any experiments not reproduced.
