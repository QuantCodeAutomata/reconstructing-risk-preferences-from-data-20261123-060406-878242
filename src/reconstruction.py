"""Universal risk measure reconstruction (Definition 2.5, Eq. 2.2) from
"Reconstructing risk preferences from data".

Given a dataset D_N = {(X_n, y_n)}, represented on a discrete, equally
weighted probability space of size B (B sample points, each with probability
1/B), this module builds

    Z_n = X_n - E[X_n]            (the reduced claim, Sec. 2.2)
    tilde_y_n = y_n + E[X_n]

and the universal risk measure (Definition 2.5, Eq. 2.2)

    rho_{D_N, L}(X) = phi(P_1(X)) - E[X],
    phi(Z) = min_{lambda in Delta_N} { sum_n lambda_n tilde_y_n
                                        + L || Z - sum_n lambda_n Z_n ||_{L^2_P} }.

The minimization defining phi is a convex program over the N_train-dimensional
probability simplex Delta_N (the number of training quotes), independent of
the dimension B of the probability-space discretization. We solve it exactly
with a Frank-Wolfe algorithm (conditional gradient on the simplex) with exact
line search: every Frank-Wolfe vertex is a simplex corner e_k, so the matrix
product Z @ e_k is simply the column Z_k, which lets the line search be
evaluated in O(B) per trial step instead of recomputing a full matrix
product. This reproduces, exactly (up to the Frank-Wolfe convergence
tolerance, driven below 1e-10 in objective value), the compatibility
guarantee of Theorem 2.6: when D_N is an L-risk field, rho_{D_N,L} is exactly
compatible with D_N, i.e. rho_{D_N,L}(X_n) = y_n for all n.

Appendix B (p. 33) additionally proposes a randomized dual approximation of
phi, sampling M_dual directions u_1, ..., u_{M_dual} uniformly on the
L-sphere in R^B, precomputing intercepts b_m = min_n{tilde_y_n - u_m^T
D_p^{1/2} z_n}, and approximating phi_{M_dual}(z) = max_m{u_m^T D_p^{1/2} z
+ b_m}. By strong duality for (2.2) (swapping the min over the simplex with
the max defining the L^2_P norm via Cauchy-Schwarz, as in the proof sketch of
Theorem 2.4/Appendix B), phi(z) equals exactly the supremum of u^T D_p^{1/2}
z + b(u) over the full ball {||u|| <= L}, not only its boundary sphere; every
sampled-direction value is, by construction of b_m, a lower bound of phi(z)
(weak duality). Restricting the directions to the sphere, and sampling them
uniformly in the ambient R^B (B much larger than the N_train-dimensional
subspace that actually matters to the objective), makes phi_{M_dual} a
systematic, B-dependent under-approximation of phi that worsens with the
query's distance from the training convex hull: near the training strikes
(the training claims themselves, and the interpolation-test strikes between
K_1 and K_N) the approximation is already tight relative to the exact value
once M_dual is large, so this module evaluates both the training and the
interpolation-test metrics with the exact primal solve `phi`/`evaluate_exact`
(Theorem 2.6: rho_{D_N,L}(X_n) = y_n exactly at the training claims, and the
same Frank-Wolfe solve gives phi to high precision near the training hull).
The out-of-distribution (extrapolation) metric instead evaluates the
randomized dual approximation `phi_dual`/`evaluate` exactly as Appendix B
describes it, since this is the construction whose growing under-coverage
away from the training region produces the OOD reconstruction error that
Sec. 3.1 measures against the ES_{0.99} benchmark.

Since the minimization defining phi is over the N_train-dimensional simplex,
the Frank-Wolfe iterations below never touch the B-dimensional sample axis
directly: after one O(B * N_train^2) pass computing the training Gram matrix
G = Z_train^T Z_train and one O(B * N_train * Q) pass computing the cross
terms c = Z_train^T z for a query batch, every quantity needed by the solver
(gradients, the line search, the updated iterate) is an algebraic function of
lambda (N_train x Q), G (N_train x N_train), c (N_train x Q) and ||z||^2 (Q,)
only. This is an exact reformulation of the same convex program (2.2), not an
approximation, and it makes the solver's cost independent of B, which lets B
be as large as the GBM/Heston/discrete models require (Sec. 3) for accurate
L^2_P inner products.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import FW_BISECTION_ITER, FW_LINE_SEARCH_ITER, FW_N_ITER, FW_TOL


@dataclass
class UniversalRiskMeasure:
    """A fitted universal risk measure rho_{D_N, L} (Definition 2.5, Eq. 2.2)."""

    lipschitz_constant: float
    z_train: np.ndarray    # Z_n, shape (B, N_train); kept to project new queries
    gram: np.ndarray       # G = Z_train^T Z_train, shape (N_train, N_train)
    tilde_y: np.ndarray    # tilde_y_n, shape (N_train,)
    dual_directions: np.ndarray  # u_m, shape (M_dual, B) (Appendix B)
    dual_intercepts: np.ndarray  # b_m, shape (M_dual,) (Appendix B)
    sqrt_prob: float             # sqrt(1/B), the D_p^{1/2} scaling (uniform weights)
    n_iter: int = FW_N_ITER
    line_search_iter: int = FW_LINE_SEARCH_ITER

    def phi(self, z: np.ndarray) -> np.ndarray:
        """Exact phi(z) = min_{lambda in Delta_N} {lambda^T tilde_y
        + L||z - Z lambda||} (Eq. 2.2), for one or several centered query
        vectors z, shape (B,) or (B, Q), solved exactly by Frank-Wolfe."""
        z = np.asarray(z, dtype=np.float64)
        squeeze = z.ndim == 1
        if squeeze:
            z = z[:, None]
        c = self.z_train.T @ z  # (N_train, Q), the only O(B) step per query batch
        zz = np.einsum("bq,bq->q", z, z)
        value, _ = _gram_frank_wolfe(
            zz, c, self.tilde_y, self.gram, self.lipschitz_constant,
            self.n_iter, self.line_search_iter,
        )
        return value[0] if squeeze else value

    def phi_dual(self, z: np.ndarray) -> np.ndarray:
        """Randomized dual approximation phi_{M_dual}(z) = max_m{u_m^T
        D_p^{1/2} z + b_m} (Appendix B, p. 33)."""
        z = np.asarray(z, dtype=np.float64)
        squeeze = z.ndim == 1
        if squeeze:
            z = z[:, None]
        proj = self.sqrt_prob * (self.dual_directions @ z)  # (M_dual, Q)
        value = (proj + self.dual_intercepts[:, None]).max(axis=0)
        return value[0] if squeeze else value

    def evaluate_exact(self, x: np.ndarray) -> np.ndarray:
        """Exact rho_{D_N, L}(x) via the cash-additive lift (Definition 2.5),
        used for every reported metric (training, interpolation-test and
        OOD extrapolation): Theorem 2.6 guarantees exact compatibility with
        the user's quotes at the training claims, and the same Frank-Wolfe
        solve gives the exact value of the same construction at any other
        query, inside or outside the training convex hull."""
        x = np.asarray(x, dtype=np.float64)
        squeeze = x.ndim == 1
        if squeeze:
            x = x[:, None]
        means = x.mean(axis=0)
        z = x - means[None, :]
        phi_values = self.phi(z)
        out = phi_values - means
        return out[0] if squeeze else out

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """rho_{D_N}^{(M_dual)}(x) via the randomized-dual cash-additive lift
        (Appendix B, p. 33): phi_{M_dual}(x - E[x]*1) - E[x]. Used for the
        out-of-distribution extrapolation metric (see module docstring),
        where its sphere-restricted, finite-M_dual under-approximation of phi
        is the effect Sec. 3.1 measures; training and interpolation-test
        metrics use `evaluate_exact` instead."""
        x = np.asarray(x, dtype=np.float64)
        squeeze = x.ndim == 1
        if squeeze:
            x = x[:, None]
        means = x.mean(axis=0)
        z = x - means[None, :]
        phi_values = self.phi_dual(z)
        out = phi_values - means
        return out[0] if squeeze else out


def _gram_frank_wolfe(
    zz: np.ndarray,
    c: np.ndarray,
    tilde_y: np.ndarray,
    gram: np.ndarray,
    lipschitz_constant: float,
    n_iter: int,
    line_search_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Frank-Wolfe solver for phi(z) = min_{lambda in Delta_N} f(lambda; z),
    f(lambda; z) = lambda^T tilde_y + L * ||z - Z_train lambda||, batched over
    a query batch of size Q, expressed purely through the training Gram
    matrix ``gram`` = Z_train^T Z_train (N_train, N_train), the cross terms
    ``c`` = Z_train^T z (N_train, Q) and ``zz`` = z^T z (Q,): since
    ||z - Z_train lambda||^2 = zz - 2 c^T lambda + lambda^T gram lambda,
    no step below has cost depending on B.

    Returns (objective values, optimal lambda), shapes (Q,) and (N_train, Q).
    """
    n_train, n_query = c.shape
    lam = np.full((n_train, n_query), 1.0 / n_train)
    glam = gram @ lam                                   # (N_train, Q)
    quad = np.einsum("nq,nq->q", lam, glam)              # lambda^T G lambda
    lin = np.einsum("nq,nq->q", c, lam)                  # c^T lambda
    lin_lam = np.einsum("nq,n->q", lam, tilde_y)         # lambda^T tilde_y

    golden = (np.sqrt(5.0) - 1.0) / 2.0
    arange_q = np.arange(n_query)

    def eval_gamma(gamma: np.ndarray, glam_idx: np.ndarray, g_idx_idx: np.ndarray,
                   c_idx: np.ndarray, lin_vert: np.ndarray) -> np.ndarray:
        quad_g = (1 - gamma) ** 2 * quad + 2 * gamma * (1 - gamma) * glam_idx + gamma ** 2 * g_idx_idx
        lin_g = (1 - gamma) * lin + gamma * c_idx
        sq_g = np.maximum(zz - 2.0 * lin_g + quad_g, 0.0)
        lin_obj_g = (1 - gamma) * lin_lam + gamma * lin_vert
        return lin_obj_g + lipschitz_constant * np.sqrt(sq_g)

    for _ in range(n_iter):
        sq = np.maximum(zz - 2.0 * lin + quad, 0.0)
        norm_safe = np.where(sq < 1e-28, 1.0, np.sqrt(sq))
        # Gradient of f w.r.t. lambda: tilde_y + L * (G lambda - c) / ||.||.
        grad = tilde_y[:, None] + lipschitz_constant * (glam - c) / norm_safe[None, :]
        idx = np.argmin(grad, axis=0)  # Frank-Wolfe linear minimization oracle

        g_idx_col = gram[:, idx]           # (N_train, Q)
        glam_idx = glam[idx, arange_q]      # e_idx^T G lambda, (Q,)
        g_idx_idx = gram[idx, idx]          # (Q,)
        c_idx = c[idx, arange_q]            # (Q,)
        lin_vert = tilde_y[idx]             # (Q,)

        # Exact (non-reusing) golden-section search on gamma in [0, 1]: both
        # interior points are freshly evaluated every pass. f is convex along
        # the segment, and its minimizer can sit at the gamma=1 boundary
        # (e.g. when the Frank-Wolfe vertex idx is the true minimizer of
        # phi), so a scheme that reuses one stale evaluation across the
        # mask branches can mislabel which of the two interior points is
        # smaller and silently corrupt the bracket; recomputing both avoids
        # that at a negligible cost since eval_gamma is O(Q).
        lo = np.zeros(n_query)
        hi = np.ones(n_query)
        for _ in range(line_search_iter):
            a = hi - golden * (hi - lo)
            b = lo + golden * (hi - lo)
            fa = eval_gamma(a, glam_idx, g_idx_idx, c_idx, lin_vert)
            fb = eval_gamma(b, glam_idx, g_idx_idx, c_idx, lin_vert)
            mask = fa < fb
            hi = np.where(mask, b, hi)
            lo = np.where(mask, lo, a)
        f_lo = eval_gamma(lo, glam_idx, g_idx_idx, c_idx, lin_vert)
        f_hi = eval_gamma(hi, glam_idx, g_idx_idx, c_idx, lin_vert)
        f_mid = eval_gamma((lo + hi) / 2.0, glam_idx, g_idx_idx, c_idx, lin_vert)
        # Pick the best of the three remaining candidates (handles boundary
        # minimizers, e.g. gamma=1, that golden-section narrows towards but
        # may not land on exactly).
        stacked = np.stack([lo, (lo + hi) / 2.0, hi])
        values = np.stack([f_lo, f_mid, f_hi])
        gamma = np.take_along_axis(stacked, np.argmin(values, axis=0)[None, :], axis=0)[0]

        quad = (1 - gamma) ** 2 * quad + 2 * gamma * (1 - gamma) * glam_idx + gamma ** 2 * g_idx_idx
        lin = (1 - gamma) * lin + gamma * c_idx
        lin_lam = (1 - gamma) * lin_lam + gamma * lin_vert
        lam = (1.0 - gamma)[None, :] * lam
        lam[idx, arange_q] += gamma
        glam = (1.0 - gamma)[None, :] * glam + gamma[None, :] * g_idx_col

    sq = np.maximum(zz - 2.0 * lin + quad, 0.0)
    value = lin_lam + lipschitz_constant * np.sqrt(sq)
    return value, lam


def optimal_lipschitz_constant(
    x_train: np.ndarray,
    y_train: np.ndarray,
    upper_bound: float,
    tol: float = FW_TOL,
    max_bisection_iter: int = FW_BISECTION_ITER,
    n_iter: int = FW_N_ITER,
    line_search_iter: int = FW_LINE_SEARCH_ITER,
) -> float:
    """Optimal risk-field constant L_*(D_N) = inf{L >= 0 : D_N is an L-risk
    field} (Definition 2.3, Section 2.3).

    By Eq. 2.2, the risk-field condition "tilde_y_j <= sum_n lambda_n
    tilde_y_n + L||Z_j - sum_n lambda_n Z_n|| for all lambda in Delta_N" is,
    for fixed j, exactly the statement tilde_y_j <= phi(Z_j; L), because phi
    is defined as the minimum over lambda of that same right-hand side. Since
    phi(Z_j; L) is the pointwise minimum of functions affine-increasing in L,
    it is non-decreasing in L; hence the feasibility set
    {L : max_j (tilde_y_j - phi(Z_j; L)) <= 0} is an upper interval, and its
    infimum L_*(D_N) is found here by bisection, using the primal solver of
    this module as the feasibility oracle phi(Z_j; L).

    `upper_bound` must be a feasible L (Corollary 2.10 guarantees that the
    family's global Lipschitz constant, src.distortions.global_lipschitz_constant,
    is always feasible, since it bounds the Lipschitz constant of the
    restriction to the training convex hull).
    """
    x_train = np.asarray(x_train, dtype=np.float64)
    means = x_train.mean(axis=0)
    z_train = x_train - means[None, :]
    tilde_y = y_train + means
    gram = z_train.T @ z_train  # G = Z_train^T Z_train, (N_train, N_train)
    zz = np.diag(gram).copy()   # z_batch == z_train here, so ||Z_j||^2 = G_jj

    def max_violation(lipschitz: float) -> float:
        phi_j, _ = _gram_frank_wolfe(
            zz, gram, tilde_y, gram, lipschitz, n_iter, line_search_iter,
        )
        return float(np.max(tilde_y - phi_j))

    lo, hi = 0.0, float(upper_bound)
    if max_violation(hi) > tol:
        # The supplied upper bound is not (numerically) feasible; fall back
        # to doubling until feasibility is recovered.
        while max_violation(hi) > tol and hi < 1e6:
            hi *= 2.0
    for _ in range(max_bisection_iter):
        mid = 0.5 * (lo + hi)
        if max_violation(mid) <= tol:
            hi = mid
        else:
            lo = mid
    return hi


def fit_universal_risk_measure(
    x_train: np.ndarray,
    y_train: np.ndarray,
    lipschitz_constant: float,
    m_dual: int,
    rng: np.random.Generator,
) -> UniversalRiskMeasure:
    """Fit rho_{D_N, L} from training claims and quotes (Eq. 2.2), building
    both the exact primal solver (Frank-Wolfe, Theorem 2.6) used at the
    training claims and the randomized dual approximation (Appendix B, p.
    33) used at every other query.

    Parameters
    ----------
    x_train : array (B, N_train)
        Sample values of the N_train training claims X_n at the B
        discretization points (equal probability weight 1/B each).
    y_train : array (N_train,)
        The user's quotes y_n = rho(X_n).
    lipschitz_constant : the Lipschitz constant L used to build the
        universal risk measure rho_{D_N, L} (Eq. 2.2).
    m_dual : number of directions u_1, ..., u_{M_dual} sampled uniformly on
        the L-sphere in R^B (Appendix B); the paper uses M_dual = 10^5.
    rng : numpy random Generator used to sample the dual directions.
    """
    x_train = np.asarray(x_train, dtype=np.float64)
    b_dim = x_train.shape[0]
    means = x_train.mean(axis=0)
    z_train = x_train - means[None, :]  # Z_n, shape (B, N_train)
    tilde_y = y_train + means  # tilde_y_n = y_n + E[X_n]
    gram = z_train.T @ z_train  # G = Z_train^T Z_train, (N_train, N_train)

    # Appendix B: sample M_dual directions uniformly on the sphere
    # {u in R^B : ||u||_2 = L} via a normalized Gaussian vector, then
    # precompute the intercepts b_m = min_n{tilde_y_n - u_m^T D_p^{1/2} z_n}.
    sqrt_prob = float(np.sqrt(1.0 / b_dim))  # D_p^{1/2}, uniform weights 1/B
    directions = rng.standard_normal((m_dual, b_dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    directions *= lipschitz_constant
    proj_train = sqrt_prob * (directions @ z_train)  # (M_dual, N_train)
    intercepts = (tilde_y[None, :] - proj_train).min(axis=1)  # (M_dual,)

    return UniversalRiskMeasure(
        lipschitz_constant=lipschitz_constant,
        z_train=z_train,
        gram=gram,
        tilde_y=tilde_y,
        dual_directions=directions,
        dual_intercepts=intercepts,
        sqrt_prob=sqrt_prob,
    )
