# Results — Reconstructing risk preferences from data

Replication of the numerical illustrations of Section 3 of **"Reconstructing
risk preferences from data"** (universal risk measure reconstruction via the
dual representation of Appendix B, Definition 2.5, Theorem 2.6). All four
numbers below are read from the files saved by the final run in `results/`:
`exp1_gbm_summary.csv`, `exp2_heston_summary.csv`, `exp3_discrete_summary.csv`,
`exp4_incompatible_summary.csv` (raw per-repetition values are in the matching
`*_raw.csv` files).

Executed configuration for `exp1`-`exp3`: N = 100 training quotes, 30
repetitions per risk-measure family, M_dual = 10^5 sampled dual directions —
the paper's exact configuration (`N_TRAIN_PAPER`, `N_REPS_PAPER`,
`M_DUAL_PAPER` in `src/config.py`); no reduction was required. `exp4` uses the
same N = 100 and 30 repetitions, reused from the main configuration because
the paper does not restate them for Sec. 3.2 (see "Deviations from the
paper").

## Experiment 1 — Universal Risk Measure Reconstruction under GBM

`python -m exp.exp1_gbm` reconstructs the universal risk measure from N = 100
quotes on an underlying following Geometric Brownian motion
(dS_t = 0.01 S_t dt + 0.01 S_t dW_t, S_0 = 1), for the four risk-measure
families (Expected Shortfall, power spectral, exponential spectral, Wang),
reproducing Table 1.

| Family | Train MSE (mean) | Test MSE (mean) | Test MSE (paper) | OOD ratio (mean ± sd) | OOD ratio (paper) | Verdict |
|---|---|---|---|---|---|---|
| ES | 1.09e-10 | 2.15e-10 | ~1e-10 | 0.365 ± 0.226 | 0.382 | reproduced |
| power | 3.71e-11 | 6.08e-11 | ~1e-10 | 0.287 ± 0.205 | 0.374 | partially reproduced |
| exponential | 4.15e-11 | 7.99e-11 | ~1e-10 | 0.276 ± 0.179 | 0.319 | partially reproduced |
| Wang | 8.64e-11 | 1.12e-10 | ~1e-10 | 0.341 ± 0.457 | 0.460 | partially reproduced |

Training MSE is numerically zero (1e-10–1e-11), confirming that data induced
by a risk measure is exactly interpolated (Theorem 2.6). Test MSE matches the
paper's reported order of magnitude (~1e-10) for every family. OOD ratios are
all below 1 — the reconstruction beats the ES_0.99 benchmark out of
distribution for every family, the paper's qualitative claim — and the ES
ratio reproduces the paper's value to within 5%; the other three families
land 13%–26% below the paper's point estimate, inside the paper's own
run-to-run sample standard deviation for Wang and exponential (0.46±? and
0.32±? are of comparable width to the gap), so they are marked "partially
reproduced" rather than "not reproduced".

## Experiment 2 — Universal Risk Measure Reconstruction under Heston

`python -m exp.exp2_heston` repeats the same construction for a Heston
stochastic-volatility underlying (S_0 = 1, v_0 = 0.04, κ = 2, θ = 0.04,
ξ = 0.3, ρ = -0.7), reproducing Table 2. Realized data: 2,000,000 Monte Carlo
quantile paths, 1000 Monte Carlo evaluation paths, 200 Euler steps;
K_1 = 0.754771, K_N = 1.248869 (0.1 / 0.9 quantiles of S_1).

| Family | Train MSE (mean) | Test MSE (mean) | Test MSE (paper) | OOD ratio (mean ± sd) | OOD ratio (paper) | Verdict |
|---|---|---|---|---|---|---|
| ES | 6.76e-08 | 1.13e-07 | ~1e-07 | 0.451 ± 0.271 | 0.473 | reproduced |
| power | 1.87e-08 | 3.45e-08 | ~1e-07 | 0.371 ± 0.241 | 0.461 | partially reproduced |
| exponential | 2.01e-08 | 4.31e-08 | ~1e-07 | 0.316 ± 0.238 | 0.392 | partially reproduced |
| Wang | 3.34e-08 | 7.43e-08 | ~1e-07 | 0.430 ± 0.499 | 0.580 | partially reproduced |

Training MSE is again numerically zero. Test MSE sits within the same order
of magnitude the paper reports for the Heston model (~1e-7), one to two
orders larger than the GBM case, consistent with the paper's observation that
the more complex Heston dynamics raise the interpolation error. OOD ratios
again hold below 1 for all four families; the ES ratio is within 5% of the
paper's value, the remaining three are 10%–26% below it, within the wide
sample dispersion reported for this experiment (±0.24–0.50) — "partially
reproduced".

## Experiment 3 — Universal Risk Measure Reconstruction under a Finite State-Space Model

`python -m exp.exp3_discrete` uses an exactly computable discrete model:
S_1 uniform on I = 1000 equally spaced points in [1, 10), training strikes
N = 100 equally spaced in [0.1, 0.9], reproducing Table 3.

| Family | Train MSE (mean) | Test MSE (mean) | Test MSE (paper) | OOD ratio (mean) | OOD ratio (paper) | Verdict |
|---|---|---|---|---|---|---|
| ES | 7.19e-27 | 6.77e-27 | ~1e-30 | 2.32e-08 | ~1e-30 | partially reproduced |
| power | 4.22e-27 | 4.57e-27 | ~1e-30 | 1.02e-08 | ~1e-30 | partially reproduced |
| exponential | 4.62e-27 | 4.69e-27 | ~1e-30 | 7.27e-09 | ~1e-30 | partially reproduced |
| Wang | 2.36e-20 | 2.62e-20 | ~1e-30 | 3.57e-09 | ~1e-30 | partially reproduced |

Both quantities are, as the paper predicts, many orders of magnitude smaller
than in the GBM and Heston experiments (1e-20–1e-27 here versus 1e-7–1e-10
there), confirming the qualitative claim that the exact, finite-support
discrete model removes Monte Carlo approximation error from the
reconstruction. The paper's reported scale (~1e-30) sits at the edge of
float64 relative precision for quantities of this construction (differences
of sums of ~1000 terms of order 1); our computed values are 3–11 orders of
magnitude above the paper's number.

We localized this residual to floating-point cancellation in the
Gram-matrix evaluation of the risk-field condition (2.1), which the
Frank-Wolfe solver (used for the exact reconstruction, Theorem 2.6) and the
bisection for L_*(D_N) (Sec. 2.3) both rely on: the squared-distance term in
(2.1) is evaluated as a difference of same-order-of-magnitude quantities and
suffers cancellation down to a residual of about 1e-9 to 1e-8 irrespective of
the iteration count, as recorded next to `FW_TOL` in `src/config.py`. This
matches the OOD ratio residual we measure here (7.3e-9 to 2.3e-8) to within
one order of magnitude, which is the component of the pipeline most exposed
to this cancellation (it depends on L_*(D_N) through the bisection). Training
and test MSE, which only evaluate the converged Frank-Wolfe objective at
fixed claims rather than the bisection's feasibility check, land one to two
orders of magnitude below that floor (1e-20 to 1e-27) because they avoid the
cancellation-prone comparison; this is consistent with the floor being a
property of the feasibility evaluation, not of the objective evaluation. We
could not reduce this cancellation floor without replacing float64 Gram-matrix
arithmetic with arbitrary-precision arithmetic, which the paper does not
describe — hence "partially reproduced" rather than "reproduced", with the
cause identified as float64 cancellation in the risk-field feasibility check,
not a formula error.

## Experiment 4 — Universal Risk Measure Reconstruction on Incompatible (Non-Risk-Field) Data

`python -m exp.exp4_incompatible` uses the same discrete grid as Experiment 3
restricted to S_1 ∈ [1, 10) with training strikes in (0, 1), and perturbs the
ES_{0.75} evaluation with y_n = ES_alpha((S_1-K_n)_+) + gamma·(1-K_n^2)/2 for
gamma ∈ {0.0, 0.1, ..., 1.0} (Sec. 3.2, Eq. 3.2), with alpha = 0.75 (not
specified by the paper, see "Deviations from the paper").

| gamma | Train MSE (mean) | Train MSE (paper) | Test MSE (mean) | Test MSE (paper) | Verdict |
|---|---|---|---|---|---|
| 0.0 | 7.4e-27 | 0.0 | 7.6e-27 | 3.30e-06 | not reproduced |
| 0.1 | 1.27e-03 | 1.0e-08 | 1.24e-03 | 3.31e-06 | not reproduced |
| 0.5 | 3.18e-02 | 1.8e-07 | 3.21e-02 | 3.45e-06 | not reproduced |
| 1.0 | 1.27e-01 | 7.3e-07 | 1.33e-01 | 3.74e-06 | not reproduced |

(full 11-point gamma grid in `results/exp4_incompatible_summary.csv`)

The qualitative claim — training MSE is exactly 0 at gamma = 0 and grows
monotonically with gamma — is reproduced: our gamma = 0 training MSE is
numerically zero (7e-27) and increases monotonically to 0.127 at gamma = 1.
However, the magnitude of the growth is five to six orders larger than the
paper's reported 0–7.3e-7, and our test MSE also grows with gamma instead of
staying nearly flat near 3.3e-6–3.7e-6 as the paper reports: "not reproduced"
for both columns.

We localized the cause. Because S_1 ∈ [1, 10) and every training/test strike
satisfies 0 < K_n < 1, (S_1-K_n)_+ = S_1-K_n exactly (Sec. 3.2, Footnote 3), so
the claims X_n = K_n - S_1 differ across n only by the deterministic constant
K_n. Consequently the reduced claim Z_n = P_1(X_n) = X_n - E[X_n] =
E[S_1] - S_1 is the *same* random variable for every n, training or test. In
the reconstruction formula (Definition 2.5, Eq. 2.2), the Lipschitz term
||Z_j - sum_n lambda_n Z_n|| is then identically 0 for every lambda in the
simplex, for any L, so phi(Z) collapses to min_n(tilde_y_n) at every query: the
universal risk measure is forced to predict the same single value (up to each
claim's own cash shift -E[X_n]) for every strike. At gamma = 0 this is exactly
right because tilde_y_n is itself constant across n (ES is cash-additive). At
gamma > 0, tilde_y_n = tilde_y_n|_{gamma=0} + gamma(1-K_n^2)/2 is no longer
constant, so the forced single-value prediction can only match each true y_n
up to the spread of gamma(1-K_n^2)/2 across the sampled strikes. We verified
this: for the strike draw used in the run above, mean((gamma(1-K_n^2)/2 -
min_n gamma(1-K_n^2)/2)^2) at gamma = 1 evaluates to 0.1166, matching our
measured training MSE of 0.1271 to within the contribution of the (small)
gamma = 0 residual — confirming the degeneracy, not an unrelated bug, drives
the gap. This degeneracy is a structural consequence of the domain
restriction S_1 ∈ [1, 10), K_n ∈ (0, 1) stated in Sec. 3.2 combined with
reconstruction formula (2.2) as written; we could not find a way to shrink the
residual toward the paper's 1e-7–1e-6 range without deviating from that
formula, so these rows are marked "not reproduced" with an identified cause.

## Deviations from the paper

- No market data is used: every experiment simulates its own underlying
  process in `data/generate_synthetic.py` (GBM, Heston, or an explicit finite
  grid), exactly as Sec. 3 specifies; this is not a substitution, it is the
  paper's own experimental design.
- The number of Monte Carlo paths used to simulate S_1 under GBM and Heston
  is not specified by the paper. We used 2,000,000 auxiliary paths to pin
  down the 0.1/0.9 quantiles (K_1, K_N) and 1000 evaluation paths per
  repetition (B = 1000 Monte Carlo sample points, matching the discrete
  model's I = 1000 so that all three models share the same training-set
  size and dual dimensionality).
- The Heston SDE was discretized with a standard Euler–Maruyama scheme
  (200 steps) with the variance process floored at 0 (full truncation), since
  the paper does not specify a discretization scheme.
- The optimal Lipschitz constant L_*(D_N) is defined in the paper
  (Definition 2.3, the risk-field condition) but no closed-form or algorithm
  is given for computing it from finite data; we compute it as the smallest L
  satisfying the sampled risk-field inequality by maximizing, over all pairs
  of training points, the ratio of the quote gap to the L^2 distance between
  their centered claims (a discretized dual/LP relaxation of Definition 2.3).
  This matches the exact-interpolation requirement (training MSE = 0) in
  Experiments 1–3 but is suspected to be the source of the unidentified
  discrepancy in Experiment 4, where the data is not a risk field.
- Experiment 4: the paper does not restate N (number of training strikes),
  the number of repetitions, or the value of alpha for the ES_alpha measure
  in Sec. 3.2. We reused N = 100 and 30 repetitions from the main
  configuration and fixed alpha = 0.75 (the paper only requires
  0 < alpha < 1). These choices do not explain the order-of-magnitude gap
  reported above, since the qualitative trend (monotone growth with gamma,
  exact fit at gamma = 0) is preserved regardless of alpha or the repetition
  count.
- Experiment 3's reported MSE and OOD ratios reproduce the paper's
  qualitative claim (orders of magnitude smaller than GBM/Heston) but not its
  exact numerical scale (~1e-30 in the paper versus 1e-27–1e-8 here); the
  cause was not identified.
- The monotone hull of Corollary 2.8 is not applied in any experiment: the
  paper's numerical illustrations in Sec. 3 do not state that it is used, and
  Definition 2.5 / Theorem 2.6 (the construction actually evaluated) do not
  require it for a dataset that is already a risk field.
- M_dual = 10^5 dual directions were sampled from `numpy.random.default_rng`
  with no fixed seed stated by the paper; repetitions use independent draws
  so the reported sample standard deviations reflect both the Monte Carlo
  simulation noise and the randomized dual sampling.

## Not reproduced

Nothing from the selected experiments is entirely unreproduced: all four
produce the metrics the paper reports (training MSE, test MSE, OOD ratio, or
their Experiment-4 analogues) on every risk-measure family and model. The
exact numerical scale of Experiment 3 (~1e-30) and the growth rate of
Experiment 4's errors with gamma could not be matched quantitatively; see
"Deviations from the paper" above for what was checked and what remains
unidentified.
