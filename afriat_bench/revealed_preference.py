"""Axiomatic Revealed Preference (GARP, Afriat CCEI, MPI, Afriat LP & CES Recovery).

Implements exact nonparametric and structural revealed-preference tests for Pillar A:
  - Direct (`R_0(e)`) and strict (`P_0(e)`) revealed preference relations at efficiency `e in (0, 1]`.
  - Warshall transitive closure `R(e)` and exact GARP / WARP cycle detection (`check_garp`).
  - Exact Afriat (1972) / Varian (1990) Critical Cost Efficiency Index (`compute_ccei`) via
    binary search over critical expenditure ratios with `(1 - 1e-11)` lower-epsilon probing.
  - Echenique, Lee & Shum (2011) Money Pump Index (`compute_mpi`) across minimal violation cycles.
  - Afriat (1967) linear programming piecewise-linear concave utility reconstruction
    (`construct_afriat_utility`) via `scipy.optimize.linprog`.
  - Structural 2-good Constant Elasticity of Substitution (`fit_ces_preferences`) archetype recovery.
  - Paired Clean vs. Moral-Perturbed Revealed Preference comparison (`compare_clean_vs_perturbed`).
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
from scipy.optimize import linprog, minimize


class _DictCompatibleMixin:
    """Allows both attribute access and dict-like key lookup for benchmark interoperability."""

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)  # type: ignore[arg-type]
        return out

    def keys(self) -> list[str]:
        return list(self.__dataclass_fields__.keys())  # type: ignore[attr-defined]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass(frozen=True)
class GARPResult(_DictCompatibleMixin):
    passes_garp: bool
    n_violations: int
    violation_pairs: list[tuple[int, int]]
    shortest_cycle: list[int]
    passes_warp: bool
    warp_violations: int


@dataclass(frozen=True)
class AfriatLPResult:
    feasible: bool
    U: np.ndarray | None
    lambdas: np.ndarray | None
    evaluate_utility_fn: Callable[[np.ndarray], float | np.ndarray] | None


@dataclass(frozen=True)
class CESFitResult(_DictCompatibleMixin):
    alpha: float
    rho: float
    elasticity_of_substitution: float
    archetype: str
    l1_share_error: float
    predicted_bundles: np.ndarray


@dataclass(frozen=True)
class PairedBudgetComparison(_DictCompatibleMixin):
    ccei_clean: float
    ccei_perturbed: float
    moral_efficiency_loss: float
    bundle_shift_l1: float
    good1_share_clean: float
    good1_share_perturbed: float
    delta_good1_share: float
    garp_broken: bool
    passes_clean: bool
    passes_perturbed: bool
    mpi_clean: float
    mpi_perturbed: float
    archetype_clean: str
    archetype_perturbed: str


def _validate_price_quantity_matrices(P: Any, X: Any) -> tuple[np.ndarray, np.ndarray]:
    P_arr = np.asarray(P, dtype=np.float64)
    X_arr = np.asarray(X, dtype=np.float64)
    if P_arr.ndim != 2 or X_arr.ndim != 2:
        raise ValueError(f"P and X must be 2D arrays of shape [T, K]; got {P_arr.shape} and {X_arr.shape}")
    if P_arr.shape != X_arr.shape:
        raise ValueError(f"P and X must have identical shape [T, K]; got {P_arr.shape} vs {X_arr.shape}")
    if P_arr.shape[0] < 1 or P_arr.shape[1] < 1:
        raise ValueError(f"P and X must be non-empty; got shape {P_arr.shape}")
    if np.any(~np.isfinite(P_arr)) or np.any(~np.isfinite(X_arr)):
        raise ValueError("P and X must contain only finite values")
    if np.any(P_arr <= 0.0):
        raise ValueError("Prices P must be strictly positive (P > 0)")
    if np.any(X_arr < -1e-12):
        raise ValueError("Quantities X must be non-negative (X >= 0)")
    X_arr = np.maximum(X_arr, 0.0)
    return P_arr, X_arr


def direct_revealed_preference(
    P: Any,
    X: Any,
    e: float = 1.0,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute direct R_0(e) and strict P_0(e) revealed-preference boolean matrices [T, T].

    At efficiency level `e in (0, 1]`:
      - Direct `R_0(e)[t, s]`: `e * (p_t @ x_t) >= (p_t @ x_s) - tol`
      - Strict `P_0(e)[t, s]`: `e * (p_t @ x_t) >  (p_t @ x_s) + tol`
    """
    if not (0.0 < e <= 1.0 + 1e-12):
        raise ValueError(f"Efficiency e must lie in (0, 1], got {e}")
    P_arr, X_arr = _validate_price_quantity_matrices(P, X)
    E = P_arr @ X_arr.T  # E[t, s] = p_t @ x_s
    own_scaled = (e * np.diag(E))[:, None]
    R_0 = own_scaled >= (E - tol)
    P_0 = own_scaled > (E + tol)
    np.fill_diagonal(P_0, False)
    return R_0, P_0


def warshall_transitive_closure(R_0: np.ndarray) -> np.ndarray:
    """Compute the Floyd-Warshall boolean transitive closure R(e) of R_0(e)."""
    R = np.asarray(R_0, dtype=bool).copy()
    T = R.shape[0]
    for k in range(T):
        R |= R[:, k : k + 1] & R[k : k + 1, :]
    return R


def _canonicalize_cycle(cycle: list[int]) -> list[int]:
    if not cycle:
        return []
    min_pos = int(np.argmin(cycle))
    return cycle[min_pos:] + cycle[:min_pos]


def _shortest_path_r0(adj: np.ndarray, src: int, dst: int) -> list[int] | None:
    """BFS shortest directed path from `src` to `dst` in boolean adjacency `adj`."""
    if src == dst:
        return [src]
    T = adj.shape[0]
    parent = [-1] * T
    visited = [False] * T
    visited[src] = True
    queue: deque[int] = deque([src])
    while queue:
        u = queue.popleft()
        neighbors = np.flatnonzero(adj[u])
        for v_np in neighbors:
            v = int(v_np)
            if v == u or visited[v]:
                continue
            parent[v] = u
            if v == dst:
                path = [dst]
                cur = dst
                while cur != src:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()
                return path
            visited[v] = True
            queue.append(v)
    return None


def _find_shortest_garp_cycle(R_0: np.ndarray, P_0: np.ndarray, R: np.ndarray) -> list[int]:
    """Find the shortest simple cycle in R_0 containing at least one strict P_0 edge."""
    adj = R_0.copy()
    np.fill_diagonal(adj, False)
    strict_pairs = np.argwhere(P_0 & R.T)  # (s, t) where s P_0 t and t R s
    best_cycle: list[int] | None = None
    for s_np, t_np in strict_pairs:
        s, t = int(s_np), int(t_np)
        if s == t:
            continue
        path = _shortest_path_r0(adj, t, s)
        if path is None:
            continue
        canon = _canonicalize_cycle(path)
        if best_cycle is None or (len(canon), canon) < (len(best_cycle), best_cycle):
            best_cycle = canon
    return best_cycle if best_cycle is not None else []


def check_garp(
    P: Any,
    X: Any,
    e: float = 1.0,
    tol: float = 1e-9,
) -> GARPResult:
    """Exact Varian (1982) Generalized Axiom of Revealed Preference test at efficiency `e`."""
    R_0, P_0 = direct_revealed_preference(P, X, e=e, tol=tol)
    R = warshall_transitive_closure(R_0)

    viol_mask = R & P_0.T
    np.fill_diagonal(viol_mask, False)
    violation_pairs = [(int(t), int(s)) for t, s in zip(*np.nonzero(viol_mask))]
    n_violations = len(violation_pairs)
    passes_garp = n_violations == 0

    warp_mask = R_0 & P_0.T
    np.fill_diagonal(warp_mask, False)
    warp_violations = int(np.count_nonzero(warp_mask))
    passes_warp = warp_violations == 0

    shortest_cycle = [] if passes_garp else _find_shortest_garp_cycle(R_0, P_0, R)
    return GARPResult(
        passes_garp=passes_garp,
        n_violations=n_violations,
        violation_pairs=violation_pairs,
        shortest_cycle=shortest_cycle,
        passes_warp=passes_warp,
        warp_violations=warp_violations,
    )


def compute_ccei(P: Any, X: Any, tol: float = 1e-9) -> float:
    """Exact Afriat (1972) / Varian (1990) Critical Cost Efficiency Index `e* in (0, 1]`.

    Uses the finite set of critical expenditure ratios `r_{t,s} = (p_t @ x_s) / (p_t @ x_t)` in `(0, 1]`
    and binary-searches over sorted unique candidates with a `1 - 1e-11` lower-epsilon probe to recover
    the exact supremum `e*` at which `X` satisfies GARP(e).
    """
    P_arr, X_arr = _validate_price_quantity_matrices(P, X)
    if check_garp(P_arr, X_arr, e=1.0, tol=tol).passes_garp:
        return 1.0

    E = P_arr @ X_arr.T
    own_exp = np.diag(E)
    positive_own = own_exp > 0.0
    if not np.all(positive_own):
        P_arr = P_arr[positive_own]
        X_arr = X_arr[positive_own]
        if P_arr.shape[0] <= 1 or check_garp(P_arr, X_arr, e=1.0, tol=tol).passes_garp:
            return 1.0
        E = P_arr @ X_arr.T
        own_exp = np.diag(E)

    ratios = E / own_exp[:, None]
    mask = (ratios > 0.0) & (ratios <= 1.0)
    raw_cands = np.sort(np.unique(np.append(ratios[mask], 1.0)))

    # Merge machine-epsilon duplicate ratios while keeping exact analytical candidate values.
    candidates: list[float] = [float(raw_cands[0])]
    for val in raw_cands[1:]:
        fval = float(val)
        if fval - candidates[-1] > 1e-12:
            candidates.append(fval)

    probe_tol = 1e-15 * float(np.min(own_exp))
    lo = 0
    hi = len(candidates) - 1
    best_idx = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        c_mid = candidates[mid]
        c_prev = candidates[mid - 1] if mid > 0 else 0.0
        e_probe = max(0.5 * (c_prev + c_mid), c_mid * (1.0 - 1e-11))
        if check_garp(P_arr, X_arr, e=e_probe, tol=probe_tol).passes_garp:
            best_idx = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return float(candidates[best_idx])


def _enumerate_minimal_violation_cycles(
    R_0: np.ndarray,
    P_0: np.ndarray,
    max_raw_cycles: int = 5000,
) -> list[list[int]]:
    """Enumerate inclusion-minimal simple directed cycles in R_0 with >= 1 strict P_0 edge."""
    T = R_0.shape[0]
    adj = R_0.copy()
    np.fill_diagonal(adj, False)
    R = warshall_transitive_closure(adj)

    raw_cycles: list[list[int]] = []

    def dfs(start: int, u: int, path: list[int], visited: set[int], has_strict: bool) -> None:
        if len(raw_cycles) >= max_raw_cycles:
            return
        for v_np in np.flatnonzero(adj[u]):
            v = int(v_np)
            if v < start:
                continue
            if v == start:
                if len(path) >= 2 and (has_strict or bool(P_0[u, start])):
                    raw_cycles.append(list(path))
            elif v not in visited and bool(R[v, start]):
                visited.add(v)
                path.append(v)
                dfs(start, v, path, visited, has_strict or bool(P_0[u, v]))
                path.pop()
                visited.remove(v)

    for s in range(T):
        if len(raw_cycles) >= max_raw_cycles:
            break
        dfs(s, s, [s], {s}, False)

    if not raw_cycles:
        fallback = _find_shortest_garp_cycle(R_0, P_0, warshall_transitive_closure(R_0))
        return [fallback] if fallback else []

    # Sort by length and retain cycles whose vertex sets are inclusion-minimal.
    raw_cycles.sort(key=lambda c: (len(c), c))
    minimal_cycles: list[list[int]] = []
    minimal_vsets: list[frozenset[int]] = []
    for cyc in raw_cycles:
        vset = frozenset(cyc)
        if any(prev_vset < vset for prev_vset in minimal_vsets):
            continue
        if vset in minimal_vsets:
            continue
        minimal_cycles.append(cyc)
        minimal_vsets.append(vset)
    return minimal_cycles


def compute_mpi(P: Any, X: Any, tol: float = 1e-9) -> float:
    """Echenique, Lee & Shum (2011) Money Pump Index across minimal preference cycles.

    Returns `0.0` if `(P, X)` satisfies GARP, otherwise the mean fractional budget waste:
        sum_{k} p_{i_k} @ (x_{i_k} - x_{i_{k+1}}) / sum_{k} p_{i_k} @ x_{i_k}
    averaged across minimal GARP-violating preference cycles.
    """
    P_arr, X_arr = _validate_price_quantity_matrices(P, X)
    garp = check_garp(P_arr, X_arr, e=1.0, tol=tol)
    if garp.passes_garp:
        return 0.0

    R_0, P_0 = direct_revealed_preference(P_arr, X_arr, e=1.0, tol=tol)
    cycles = _enumerate_minimal_violation_cycles(R_0, P_0)
    if not cycles:
        return 0.0

    E = P_arr @ X_arr.T
    wastes: list[float] = []
    for cyc in cycles:
        L = len(cyc)
        num = 0.0
        den = 0.0
        for idx in range(L):
            u = cyc[idx]
            v = cyc[(idx + 1) % L]
            num += float(E[u, u] - E[u, v])
            den += float(E[u, u])
        if den > 0.0 and num > 0.0:
            wastes.append(num / den)
    return float(np.mean(wastes)) if wastes else 0.0


def construct_afriat_utility(
    P: Any,
    X: Any,
    e: float = 1.0,
    tol: float = 1e-9,
) -> AfriatLPResult:
    """Solve Afriat's (1967) linear program to reconstruct a piecewise-linear concave utility.

    Solves for Afriat numbers `U_t` and marginal-utility multipliers `lambda_t`:
        U_s - U_t <= lambda_t * p_t @ (x_s - e * x_t)   for all t != s
        lambda_t >= 1.0                                 for all t
        U_0 = 0.0
    Minimizing `sum_t lambda_t` via `scipy.optimize.linprog` (HiGHS).
    """
    if not (0.0 < e <= 1.0 + 1e-12):
        raise ValueError(f"Efficiency e must lie in (0, 1], got {e}")
    P_arr, X_arr = _validate_price_quantity_matrices(P, X)
    T, K = P_arr.shape

    if T == 1:
        U_single = np.zeros(1, dtype=np.float64)
        lam_single = np.ones(1, dtype=np.float64)
        own0 = float(P_arr[0] @ X_arr[0])

        def _eval_single(x: np.ndarray) -> float | np.ndarray:
            arr = np.asarray(x, dtype=np.float64)
            if arr.ndim == 1:
                return float(U_single[0] + lam_single[0] * (float(P_arr[0] @ arr) - e * own0))
            return U_single[0] + lam_single[0] * ((arr @ P_arr[0]) - e * own0)

        return AfriatLPResult(feasible=True, U=U_single, lambdas=lam_single, evaluate_utility_fn=_eval_single)

    E = P_arr @ X_arr.T
    own_exp = np.diag(E)
    # delta[t, s] = p_t @ (x_s - e * x_t)
    delta = E - (e * own_exp)[:, None]
    delta[np.abs(delta) <= tol] = 0.0

    # Decision vector z = [U_0, ..., U_{T-1}, lambda_0, ..., lambda_{T-1}] of length 2*T.
    # Inequality for each (t, s) with t != s:
    #   U_s - U_t - delta[t, s] * lambda_t <= 0
    n_rows = T * (T - 1)
    A_ub = np.zeros((n_rows, 2 * T), dtype=np.float64)
    b_ub = np.zeros(n_rows, dtype=np.float64)
    row = 0
    for t in range(T):
        for s in range(T):
            if t == s:
                continue
            A_ub[row, s] = 1.0
            A_ub[row, t] = -1.0
            A_ub[row, T + t] = -delta[t, s]
            row += 1

    c = np.zeros(2 * T, dtype=np.float64)
    c[T:] = 1.0  # Minimize sum_t lambda_t (bounded below by T)

    bounds: list[tuple[float | None, float | None]] = [(0.0, 0.0)]  # U_0 = 0
    for _ in range(1, T):
        bounds.append((None, None))
    for _ in range(T):
        bounds.append((1.0, None))  # lambda_t >= 1.0

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success or res.x is None:
        return AfriatLPResult(feasible=False, U=None, lambdas=None, evaluate_utility_fn=None)

    U = np.asarray(res.x[:T], dtype=np.float64)
    U[0] = 0.0
    lambdas = np.asarray(res.x[T:], dtype=np.float64)
    P_frozen = P_arr.copy()
    own_frozen = own_exp.copy()
    e_frozen = float(e)

    def evaluate_utility_fn(x: np.ndarray) -> float | np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            if arr.shape[0] != K:
                raise ValueError(f"Expected bundle of length {K}, got {arr.shape}")
            vals = U + lambdas * ((P_frozen @ arr) - e_frozen * own_frozen)
            return float(np.min(vals))
        if arr.ndim == 2 and arr.shape[1] == K:
            vals = U[:, None] + lambdas[:, None] * ((P_frozen @ arr.T) - e_frozen * own_frozen[:, None])
            return np.min(vals, axis=0)
        raise ValueError(f"Expected shape [{K}] or [N, {K}], got {arr.shape}")

    return AfriatLPResult(
        feasible=True,
        U=U,
        lambdas=lambdas,
        evaluate_utility_fn=evaluate_utility_fn,
    )


def _ces_good1_share(log_rel_price: np.ndarray, alpha: float, rho: float) -> np.ndarray:
    """Closed-form Marshallian expenditure share `w_1 = p_1 x_1^* / m` for 2-good CES utility."""
    alpha_c = float(np.clip(alpha, 1e-6, 1.0 - 1e-6))
    rho_c = float(np.clip(rho, -40.0, 0.995))
    inv_one_minus_rho = 1.0 / (1.0 - rho_c)
    log_alpha_odds = float(np.log(alpha_c / (1.0 - alpha_c)))
    logit = inv_one_minus_rho * log_alpha_odds - (rho_c * inv_one_minus_rho) * log_rel_price
    logit = np.clip(logit, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logit))


def fit_ces_preferences(P: Any, X: Any) -> CESFitResult:
    """Fit 2-good CES utility `u(x1, x2) = (alpha * x1^rho + (1 - alpha) * x2^rho)^(1/rho)`.

    Uses coarse grid search followed by multi-start L-BFGS-B minimizing normalized bundle L1
    (expenditure-share) error along observed budget lines, and classifies the archetype into:
      - `"selfish_good1"`
      - `"altruistic_good2"`
      - `"egalitarian_leontief"` (`rho < -1.5`)
      - `"utilitarian_substitutes"` (`rho > 0.6`)
      - `"cobb_douglas"`
    """
    P_arr, X_arr = _validate_price_quantity_matrices(P, X)
    if P_arr.shape[1] != 2:
        raise ValueError(f"fit_ces_preferences requires 2-good observations [T, 2], got {P_arr.shape}")

    m = np.sum(P_arr * X_arr, axis=1)
    if np.any(m <= 0.0):
        raise ValueError("Observed expenditures p_t @ x_t must be strictly positive")

    obs_share1 = (P_arr[:, 0] * X_arr[:, 0]) / m
    log_rel_price = np.log(P_arr[:, 0] / P_arr[:, 1])

    def smooth_obj(params: np.ndarray) -> float:
        a, r = float(params[0]), float(params[1])
        w1 = _ces_good1_share(log_rel_price, a, r)
        diff = obs_share1 - w1
        # Pseudo-Huber approximation to L1 share error + tiny rho^2 tie-breaker at corner alpha
        pseudo_l1 = float(np.mean(np.sqrt(diff * diff + 1e-10) - 1e-5))
        return pseudo_l1 + 1e-7 * (r * r)

    alpha_grid = np.array([1e-4, 0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.0 - 1e-4], dtype=np.float64)
    rho_grid = np.array([-25.0, -8.0, -3.0, -1.0, -0.25, 0.0, 0.35, 0.75, 0.98], dtype=np.float64)

    candidates: list[tuple[float, float, float]] = []
    for a in alpha_grid:
        for r in rho_grid:
            val = smooth_obj(np.array([a, r], dtype=np.float64))
            candidates.append((val, float(a), float(r)))
    candidates.sort(key=lambda item: item[0])

    bounds = [(1e-5, 1.0 - 1e-5), (-30.0, 0.992)]
    best_alpha = candidates[0][1]
    best_rho = candidates[0][2]
    best_loss = float("inf")

    for _, a0, r0 in candidates[:6]:
        opt = minimize(
            smooth_obj,
            x0=np.array([a0, r0], dtype=np.float64),
            method="L-BFGS-B",
            bounds=bounds,
        )
        a_hat, r_hat = float(opt.x[0]), float(opt.x[1])
        w1_hat = _ces_good1_share(log_rel_price, a_hat, r_hat)
        l1_err = float(np.mean(np.abs(obs_share1 - w1_hat))) + 1e-8 * (r_hat * r_hat)
        if l1_err < best_loss:
            best_loss = l1_err
            best_alpha = a_hat
            best_rho = r_hat

    w1_pred = _ces_good1_share(log_rel_price, best_alpha, best_rho)
    raw_l1_error = float(np.mean(np.abs(obs_share1 - w1_pred)))
    pred_x1 = (w1_pred * m) / P_arr[:, 0]
    pred_x2 = ((1.0 - w1_pred) * m) / P_arr[:, 1]
    pred_bundles = np.column_stack([pred_x1, pred_x2])

    mean_obs_s1 = float(np.mean(obs_share1))
    mean_pred_s1 = float(np.mean(w1_pred))

    if (best_alpha >= 0.88 and mean_pred_s1 >= 0.88) or mean_obs_s1 >= 0.95:
        archetype = "selfish_good1"
    elif (best_alpha <= 0.12 and mean_pred_s1 <= 0.12) or mean_obs_s1 <= 0.05:
        archetype = "altruistic_good2"
    elif best_rho < -1.5:
        archetype = "egalitarian_leontief"
    elif best_rho > 0.6:
        archetype = "utilitarian_substitutes"
    else:
        archetype = "cobb_douglas"

    sigma = float(1.0 / (1.0 - best_rho))
    return CESFitResult(
        alpha=best_alpha,
        rho=best_rho,
        elasticity_of_substitution=sigma,
        archetype=archetype,
        l1_share_error=raw_l1_error,
        predicted_bundles=pred_bundles,
    )


def compare_clean_vs_perturbed(
    P: Any,
    X_clean: Any,
    X_perturbed: Any,
    incomes: Any | None = None,
) -> PairedBudgetComparison:
    """Compare clean vs. moral-perturbed chosen bundle matrices on identical price schedules `P`."""
    P_arr, X_c = _validate_price_quantity_matrices(P, X_clean)
    _, X_p = _validate_price_quantity_matrices(P, X_perturbed)
    if X_c.shape != X_p.shape:
        raise ValueError(f"X_clean and X_perturbed shapes must match; got {X_c.shape} vs {X_p.shape}")

    if incomes is not None:
        m = np.asarray(incomes, dtype=np.float64)
        if m.shape != (P_arr.shape[0],) or np.any(m <= 0.0):
            raise ValueError(f"incomes must be positive of shape ({P_arr.shape[0]},)")
    else:
        m_c = np.sum(P_arr * X_c, axis=1)
        m_p = np.sum(P_arr * X_p, axis=1)
        m = 0.5 * (m_c + m_p)
        if np.any(m <= 0.0):
            raise ValueError("Budget expenditures m must be strictly positive")

    garp_clean = check_garp(P_arr, X_c)
    garp_perturbed = check_garp(P_arr, X_p)
    ccei_clean = compute_ccei(P_arr, X_c)
    ccei_perturbed = compute_ccei(P_arr, X_p)
    moral_efficiency_loss = float(ccei_clean - ccei_perturbed)

    max_quantities = m[:, None] / P_arr
    normalized_abs_diff = np.abs(X_p - X_c) / max_quantities
    bundle_shift_l1 = float(np.mean(np.mean(normalized_abs_diff, axis=1)))

    good1_share_clean = float(np.mean((P_arr[:, 0] * X_c[:, 0]) / m))
    good1_share_perturbed = float(np.mean((P_arr[:, 0] * X_p[:, 0]) / m))
    delta_good1_share = float(good1_share_perturbed - good1_share_clean)

    mpi_clean = compute_mpi(P_arr, X_c)
    mpi_perturbed = compute_mpi(P_arr, X_p)

    if P_arr.shape[1] == 2:
        arch_clean = fit_ces_preferences(P_arr, X_c).archetype
        arch_perturbed = fit_ces_preferences(P_arr, X_p).archetype
    else:
        arch_clean = "multi_good"
        arch_perturbed = "multi_good"

    return PairedBudgetComparison(
        ccei_clean=ccei_clean,
        ccei_perturbed=ccei_perturbed,
        moral_efficiency_loss=moral_efficiency_loss,
        bundle_shift_l1=bundle_shift_l1,
        good1_share_clean=good1_share_clean,
        good1_share_perturbed=good1_share_perturbed,
        delta_good1_share=delta_good1_share,
        garp_broken=bool(garp_clean.passes_garp and not garp_perturbed.passes_garp),
        passes_clean=garp_clean.passes_garp,
        passes_perturbed=garp_perturbed.passes_garp,
        mpi_clean=mpi_clean,
        mpi_perturbed=mpi_perturbed,
        archetype_clean=arch_clean,
        archetype_perturbed=arch_perturbed,
    )


def count_interior_budget_intersections(P: Any, incomes: Any, tol: float = 1e-9) -> int:
    """Count pairs `(t < s)` of 2-good linear budget lines that cross strictly in the positive interior."""
    P_arr = np.asarray(P, dtype=np.float64)
    m_arr = np.asarray(incomes, dtype=np.float64)
    if P_arr.ndim != 2 or P_arr.shape[1] != 2 or m_arr.shape != (P_arr.shape[0],):
        raise ValueError("Expected P of shape [T, 2] and incomes of shape [T]")
    x1_int = m_arr / P_arr[:, 0]
    x2_int = m_arr / P_arr[:, 1]
    count = 0
    T = P_arr.shape[0]
    for i in range(T):
        for j in range(i + 1, T):
            d1 = x1_int[i] - x1_int[j]
            d2 = x2_int[i] - x2_int[j]
            if (d1 > tol and d2 < -tol) or (d1 < -tol and d2 > tol):
                count += 1
    return count
