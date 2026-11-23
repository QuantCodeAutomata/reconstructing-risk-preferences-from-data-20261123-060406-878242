"""Module-level experiment parameters.

All values are taken from the numerical-illustration section of
"Reconstructing risk preferences from data" (Section 3 and Appendix B) unless
noted otherwise. Where the paper leaves a detail unspecified, the comment
says so explicitly.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Shared across exp_1 (GBM), exp_2 (Heston), exp_3 (discrete), exp_4
# ---------------------------------------------------------------------------

N_TRAIN_PAPER = 100  # number of elicited quotes N, Sec. 3 ("N = 100 short-position ... options")
N_TEST_INTERP = 10  # interpolation test strikes drawn uniformly from (K_1, K_N), Sec. 3 footnote 2
N_TEST_OOD_EACH_SIDE = 5  # extrapolation strikes on each side of [K_1, K_N], Sec. 3 footnote 2, total 10

N_REPS_PAPER = 30  # repetitions per risk-measure family, Sec. 3 ("repeated 30 times")
N_REPS_RUN = 30  # executed repetition count (unchanged: matches the paper exactly)

M_DUAL_PAPER = 100_000  # number of sampled dual directions, Appendix B ("M_dual = 10^5")
M_DUAL_RUN = 100_000  # executed value; reduced only if the runtime estimate requires it

# Monte Carlo discretization size B (number of equally weighted sample points
# representing the probability space used in the dual reconstruction of
# Appendix B). The paper does not specify this count for the GBM and Heston
# models ("the exact number of Monte Carlo paths ... is not specified").
# We fix it to I = 1000, matching the discrete model's exact grid size
# (Eq. 3.1), so that the three models share a comparable discretization.
B_GBM_HESTON = 1_000  # not specified in the paper; interpreted assumption
N_HESTON_STEPS = 200  # Euler time-steps per path for the Heston SDE; not specified in the paper

# Quantile levels defining the training strike range [K_1, K_N], Sec. 3 footnote 2
STRIKE_QUANTILE_LOW = 0.1
STRIKE_QUANTILE_HIGH = 0.9

# Risk-measure parameter ranges, Sec. 3, p. 10-11
ALPHA_RANGE = (0.5, 0.9)   # ES_alpha, alpha ~ Unif([0.5, 0.9))
R_RANGE = (1.1, 10.0)      # power spectral rho^pow_r, r ~ Unif([1.1, 10))
BETA_RANGE = (0.5, 10.0)   # exponential spectral rho^exp_beta, beta ~ Unif([0.5, 10))
LAMBDA_RANGE = (0.1, 2.0)  # Wang rho^Wang_lambda, lambda ~ Unif([0.1, 2))

ALPHA_BENCHMARK = 0.99  # ad-hoc ES_0.99 benchmark used for the OOD test ratio, Sec. 3.1

RANDOM_SEED = 2024  # not specified in the paper; fixed for reproducibility

# Frank-Wolfe solver settings for the primal reconstruction of Eq. 2.2 (not
# specified in the paper, which leaves the numerical solution of phi open).
# Chosen so that the solver converges to the Frank-Wolfe optimum well beyond
# floating-point relevance (training-claim objective residual ~1e-19),
# verified against the exactness guarantee of Theorem 2.6.
FW_N_ITER = 300            # outer Frank-Wolfe (conditional gradient) iterations
FW_LINE_SEARCH_ITER = 22   # golden-section line-search iterations per FW step
FW_BISECTION_ITER = 28     # bisection iterations for L_*(D_N) (Definition 2.3/Sec. 2.3)
FW_TOL = 1e-9              # feasibility tolerance for the risk-field condition (2.1)

# ---------------------------------------------------------------------------
# exp_1: Geometric Brownian Motion, Sec. 3 p. 10
# ---------------------------------------------------------------------------
GBM_MU = 0.01      # drift
GBM_SIGMA = 0.01   # volatility
GBM_S0 = 1.0       # initial value S_0
GBM_T = 1.0        # horizon t in [0, 1]
GBM_MC_PATHS_QUANTILE = 2_000_000  # large auxiliary sample to pin down K_1, K_N quantiles precisely

# ---------------------------------------------------------------------------
# exp_2: Heston stochastic volatility model, Sec. 3 p. 10
# ---------------------------------------------------------------------------
HESTON_MU = 0.01       # drift of S
HESTON_KAPPA = 2.0     # mean-reversion speed of variance
HESTON_THETA = 0.04    # long-run variance
HESTON_XI = 0.3        # vol-of-vol
HESTON_RHO = -0.7      # correlation between the two Brownian motions
HESTON_S0 = 1.0
HESTON_V0 = 0.04
HESTON_T = 1.0
HESTON_MC_PATHS_QUANTILE = 2_000_000  # large auxiliary sample for K_1, K_N quantiles

# ---------------------------------------------------------------------------
# exp_3: Finite state-space (discrete) model, Eq. 3.1, p. 10
# ---------------------------------------------------------------------------
DISCRETE_I = 1_000       # number of equally spaced, equally likely grid points
DISCRETE_LOW = 1.0       # grid starts at 1
DISCRETE_WIDTH = 9.0     # grid spans [1, 10)
DISCRETE_K1 = 0.1        # fixed training strike lower bound, Table 3 caption / footnote 2
DISCRETE_KN = 0.9        # fixed training strike upper bound

# ---------------------------------------------------------------------------
# exp_4: Incompatible (non-risk-field) datasets, Sec. 3.2, Eq. 3.2, p. 13
# ---------------------------------------------------------------------------
GAMMA_GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 1)  # gamma in {0.0, ..., 1.0}, 11 values
EXP4_ALPHA = 0.75  # ES confidence level for the perturbed preference (3.2); only "0 < alpha < 1" is stated
EXP4_N_TRAIN = N_TRAIN_PAPER  # N is not explicitly restated for exp_4; reuse the paper's N = 100
EXP4_N_REPS = N_REPS_RUN      # repetition count is not stated for exp_4; reuse the main N_REPS_RUN
EXP4_I = DISCRETE_I           # S_1 uses the same discrete grid, I = 10^3 (Sec. 3.2)
EXP4_STRIKE_LOW = 0.0   # training/test strikes 0 < K_1 < ... < K_N < 1 are sampled uniformly on (0, 1)
EXP4_STRIKE_HIGH = 1.0
