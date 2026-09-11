"""Probabilistic regret bound (PRB) stopping rule for the campaign.

Reference: J. T. Wilson, "Stopping Bayesian Optimization with Probabilistic Regret Bounds",
NeurIPS 2024 (resources/NeurIPS-2024-stopping-bayesian-optimization-with-probabilistic-
regret-bounds-Paper-Conference.pdf). Stop once the solution is within ε of the optimum with
probability ≥ 1 − δ under the model. δ = δ_mod + δ_est: δ_mod bounds the model risk (test
Ψ = P(regret ≤ ε) ≥ λ = 1 − δ_mod) and δ_est bounds the Monte Carlo estimation risk through
a sequential Clopper–Pearson test (Algorithm 2; d_j = j^−α (α−1)/α · δ_est, n_j = ⌈β^(j−1) N⌉,
α = 1.1, β = 1.5, N = 64). With a fixed budget of T iterations the estimation risk is spread
over the remaining checks, δ_est^t = δ_est / (T − t) (paper, Appendix A.3).

Multi-objective adaptation. The paper's regret is r_t(x) = sup_X f_t − f_t(x): the best value
any point could have under a posterior draw minus the value of our solution under the same
draw. In this campaign the solution is the set of tested formulations and its quality is the
hypervolume (HV) of their Pareto front, and one formulation is tested per iteration, so the
equivalent question is "could ONE more formulation still improve the hypervolume by more
than ε?":

    r_1(f) = max_x [ HV(tested ∪ {x}; f) − HV(tested; f) ]      (one-step hypervolume regret)

with the tested values and the candidate values taken from the SAME joint posterior draw f
(the paper samples f_t(x) jointly with f_t*). The maximum over x is exact on a grid using the
box decomposition of the non-dominated region (the q = 1 hypervolume-improvement formula
behind qEHVI). Stop when P(r_1 ≤ ε) ≥ 1 − δ_mod, confirmed by the sequential test.

Two complements are reported: the whole-front regret HV*(f) − HV(tested; f), where HV* is
the hypervolume of the Pareto front of f over the entire design space — a front-coverage
statistic that cannot reach zero with finitely many points, so it is not used to stop — and
the paper's original single-objective rule applied to each output separately.
Draws are exact joint samples of the three fixed-noise GP posteriors (train_Yvar = s²/n)
on grid ∪ tested points; no random-feature approximation is needed in two dimensions.
"""
import math
import numpy as np
import pandas as pd
import torch
import gpytorch
from scipy.stats import beta as beta_dist
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from .config import SEED, DTYPE, X_COLS, Y_COLS, Y_SHORT

PRB_ALPHA, PRB_BETA, PRB_N0 = 1.1, 1.5, 64      # Algorithm 2 schedules (Wilson 2024, §3.2)


# ────────────────────────────── statistics ───────────────────────────────────
def clopper_pearson(k, n, delta):
    """Exact (conservative) 1 − delta confidence interval for a Bernoulli mean, k successes in n."""
    lo = 0.0 if k == 0 else float(beta_dist.ppf(delta / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - delta / 2, k + 1, n - k))
    return lo, hi


def sequential_test(z, lam, delta_est, alpha=PRB_ALPHA, beta=PRB_BETA, n0=PRB_N0):
    """Algorithm 2 (Monte Carlo PRB) on a pre-drawn i.i.d. Bernoulli sequence `z`.
    Batches grow geometrically (n_j = ⌈β^(j−1) n0⌉); after each batch a Clopper–Pearson
    interval at level d_j = j^−α (α−1)/α · delta_est is built and the test stops as soon as
    λ falls outside it. Σ_j d_j ≤ delta_est, so a wrong decision has probability ≤ delta_est.
    Returns dict(estimate, ci, n_used, rounds (DataFrame, one row per round),
    decision ∈ {'stop', 'continue', 'undecided'})."""
    z = np.asarray(z, dtype=float); n_max = len(z)
    rounds, j = [], 0
    while True:
        j += 1
        n_j = min(int(math.ceil(beta ** (j - 1) * n0)), n_max)
        d_j = j ** (-alpha) * (alpha - 1) / alpha * delta_est
        k = int(z[:n_j].sum())
        ci, est = clopper_pearson(k, n_j, d_j), k / n_j
        decided = not (ci[0] <= lam <= ci[1])
        rounds.append({'round j': j, 'draws n_j': n_j, 'successes k': k, 'level d_j': d_j, 'estimate': est,
                       'CI low': ci[0], 'CI high': ci[1],
                       'outcome': ('decided: ' + ('stop' if est >= lam else 'continue')) if decided
                                  else ('draws exhausted' if n_j >= n_max else 'λ inside CI → more draws')})
        if decided or n_j >= n_max:
            return dict(estimate=est, ci=ci, n_used=n_j, level=d_j, rounds=pd.DataFrame(rounds),
                        decision=('stop' if est >= lam else 'continue') if decided else 'undecided')


# ────────────────────────────── hypervolume ──────────────────────────────────
def hypervolume_3d(Y, ref):
    """Exact hypervolume (minimization) dominated by the rows of Y within the box bounded by
    `ref`, for 3 objectives: sweep the third objective, accumulate 2-D staircase areas."""
    V = np.asarray(ref, dtype=float) - np.asarray(Y, dtype=float)       # anchored boxes [0, v]
    V = V[(V > 0).all(axis=1)]
    if len(V) == 0:
        return 0.0
    V = V[np.argsort(-V[:, 2])]
    z = V[:, 2]; z_next = np.append(z[1:], 0.0)
    hv = 0.0
    for i in range(len(V)):
        if z[i] > z_next[i]:
            hv += (z[i] - z_next[i]) * _staircase_area(V[:i + 1])
    return float(hv)


def _staircase_area(P):
    """Area of the union of anchored rectangles [0, p1] × [0, p2]."""
    P = P[np.argsort(-P[:, 0])]
    heights = np.maximum.accumulate(P[:, 1])
    widths = P[:, 0] - np.append(P[1:, 0], 0.0)
    return float((widths * heights).sum())


def pareto_hv(space, Y):
    """(non-dominated mask, hypervolume) — fast path for the Monte Carlo loop."""
    pm = is_non_dominated(-torch.as_tensor(np.asarray(Y, dtype=float), dtype=DTYPE)).numpy()
    return pm, hypervolume_3d(np.asarray(Y)[pm], -space.ref_point.numpy())


def one_step_hvi(space, Y_tested, Y_cand):
    """Hypervolume improvement of each candidate row of Y_cand added alone to Y_tested
    (minimization scale). Exact: Σ over the cells of the non-dominated region of the volume
    of the candidate's box inside the cell (the q = 1 formula of qEHVI)."""
    Yt = -torch.as_tensor(np.asarray(Y_tested, dtype=float), dtype=DTYPE)
    Yc = -torch.as_tensor(np.asarray(Y_cand, dtype=float), dtype=DTYPE)
    part = FastNondominatedPartitioning(ref_point=space.ref_point, Y=Yt)
    bounds = part.get_hypercell_bounds()                          # (2, K, 3): lower, upper
    lo, hi = bounds[0], bounds[1]
    diff = (torch.minimum(Yc[:, None, :], hi[None]) - lo[None]).clamp_min(0.0)
    return diff.prod(-1).sum(-1).numpy()                         # (n_cand,)


# ────────────────────────────── sampling ─────────────────────────────────────
def sample_posterior(space, model, X_orig, n_draws, seed=SEED):
    """Exact joint draws of the three GP posteriors (latent f, fixed-noise model) at X_orig.
    Returns an array (n_draws, N, 3) on the ORIGINAL (minimization) scale."""
    X01 = space.to01(X_orig)
    g = torch.Generator().manual_seed(seed)
    out = []
    for m in model.models:
        with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
            post = m.posterior(X01)                              # exact joint posterior (LOVE off)
            mean_neg = post.mean.reshape(-1)
            cov = post.distribution.covariance_matrix
            cov = cov.to_dense() if hasattr(cov, 'to_dense') else cov
            cov = 0.5 * (cov + cov.T)
            scale = float(cov.diagonal().max())
            L = None
            for jit in (1e-10, 1e-8, 1e-6, 1e-4):
                try:
                    L = torch.linalg.cholesky(cov + jit * scale * torch.eye(len(cov), dtype=DTYPE))
                    break
                except RuntimeError:
                    continue
            if L is None:
                raise RuntimeError('posterior covariance is not positive definite even with jitter')
            zs = torch.randn(len(cov), n_draws, generator=g, dtype=DTYPE)
            draws = -(mean_neg[:, None] + L @ zs)                # back to the minimization scale
        out.append(draws.T.numpy())
    return np.stack(out, axis=-1)


def draw_paths(space, model, stats, n_draws=2000, grid_n=40, seed=SEED):
    """STEP 1 — exact joint draws of the latent objectives on a grid ∪ the tested conditions.
    Returns dict(F_grid (n, G, 3), F_tested (n, T, 3), X_grid, G1, G2, posterior mean/sd)."""
    lb, ub = space.x_bounds
    G1, G2 = np.meshgrid(np.linspace(lb[0], ub[0], grid_n), np.linspace(lb[1], ub[1], grid_n))
    X_grid = np.column_stack([G1.ravel(), G2.ravel()])
    X_tested = stats[X_COLS].values.astype(float)
    X_all = np.vstack([X_grid, X_tested])
    F = sample_posterior(space, model, X_all, n_draws, seed=seed)
    with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
        post = model.posterior(space.to01(X_all))
        mu, sd = -post.mean.numpy(), post.variance.sqrt().numpy()
    g = len(X_grid)
    return dict(F_grid=F[:, :g], F_tested=F[:, g:], X_grid=X_grid, X_tested=X_tested, G1=G1, G2=G2,
                grid_n=grid_n, n_draws=n_draws, mu_grid=mu[:g], sd_grid=sd[:g], mu_tested=mu[g:], sd_tested=sd[g:])


def compute_regrets(space, paths, n_front=300):
    """STEP 2 — regrets under each draw f:
      hvi_regret (n,)     max_x HV(tested ∪ {x}; f) − HV(tested; f)   ← the stopping rule's regret
      x_best (n, 2)       the grid point achieving it; i_best its grid index
      hv_tested (n,)      HV(tested; f)
      hv_star (n_front,)  HV of the Pareto front of f over grid ∪ tested (whole-front coverage)
      obj_regret (n, T, 3) f_k(x_tested) − min_X f_k   (the paper's single-objective regret, per output)"""
    F_grid, F_tested, X_grid, n = paths['F_grid'], paths['F_tested'], paths['X_grid'], paths['n_draws']
    hvi_regret, x_best, i_best = np.empty(n), np.empty((n, 2)), np.empty(n, dtype=int)
    for i in range(n):
        hvi = one_step_hvi(space, F_tested[i], F_grid[i])
        j = int(np.argmax(hvi)); hvi_regret[i], x_best[i], i_best[i] = hvi[j], X_grid[j], j
    hv_tested = np.array([pareto_hv(space, F_tested[i])[1] for i in range(n)])
    n_front = min(n_front, n)
    hv_star = np.array([pareto_hv(space, np.vstack([F_grid[i], F_tested[i]]))[1] for i in range(n_front)])
    obj_regret = F_tested - np.minimum(F_grid.min(axis=1), F_tested.min(axis=1))[:, None, :]
    return dict(hvi_regret=hvi_regret, x_best=x_best, i_best=i_best, hv_tested=hv_tested, hv_star=hv_star,
                hv_regret=np.maximum(hv_star - hv_tested[:n_front], 0.0), obj_regret=obj_regret)


def example_draw(space, paths, regrets, i):
    """Everything needed to illustrate one draw i: tested values, their Pareto mask and HV,
    the best additional point, its objective values and its hypervolume improvement."""
    Yt, Yg = paths['F_tested'][i], paths['F_grid'][i]
    pm, hv_t = pareto_hv(space, Yt)
    j = regrets['i_best'][i]
    return dict(i=i, Y_tested=Yt, pm=pm, hv_tested=hv_t, Y_grid=Yg, j=j, x_best=paths['X_grid'][j],
                y_best=Yg[j], hvi=regrets['hvi_regret'][i], hv_with=hv_t + regrets['hvi_regret'][i])


# ────────────────────────────── the rule ─────────────────────────────────────
def prb_table(regret, hv_now, eps_fracs, delta_mod, delta_est_t):
    """STEP 3 — Ψ̂(ε) = P(regret ≤ ε) with the sequential test, for ε = frac × current HV."""
    lam, rows, tests = 1 - delta_mod, [], {}
    for frac in eps_fracs:
        eps = frac * hv_now
        t = sequential_test(regret <= eps, lam, delta_est_t)
        tests[frac] = t
        rows.append({'ε (% of HV)': 100 * frac, 'ε (HV units)': eps, 'P̂(regret ≤ ε)': t['estimate'],
                     'CI low': t['ci'][0], 'CI high': t['ci'][1], 'draws used': t['n_used'],
                     'decision': t['decision'].upper()})
    return pd.DataFrame(rows), tests


def delta_sensitivity(regret, hv_now, eps_fracs, deltas, remaining):
    """Decision for every (δ, ε) pair: δ_mod = δ_est = δ/2, δ_est^t = δ_est / remaining checks.
    The point estimate does not depend on δ — only λ and the width of the test interval do."""
    rows = []
    for d in deltas:
        r = {'δ': d, 'λ = 1 − δ/2': 1 - d / 2}
        for frac in eps_fracs:
            t = sequential_test(regret <= frac * hv_now, 1 - d / 2, d / 2 / remaining)
            r[f'ε = {100 * frac:g}% HV'] = t['decision'].upper()
        rows.append(r)
    return pd.DataFrame(rows).set_index('δ')


def budget_sensitivity(regret, hv_now, eps_frac, delta, budgets, t_done):
    """Effect of the planned budget T (through δ_est^t = δ_est / (T − t)) for the headline ε."""
    rows = []
    for T in budgets:
        rem = max(T - t_done, 1)
        t = sequential_test(regret <= eps_frac * hv_now, 1 - delta / 2, delta / 2 / rem)
        rows.append({'budget T': T, 'remaining checks': rem, 'δ_est^t': delta / 2 / rem, 'P̂': t['estimate'],
                     'CI low': t['ci'][0], 'CI high': t['ci'][1], 'draws used': t['n_used'], 'decision': t['decision'].upper()})
    return pd.DataFrame(rows).set_index('budget T')


def prb_objectives(paths, regrets, stats, eps_abs, delta_mod, delta_est_t):
    """The paper's single-objective rule per output: candidate = the tested condition with the
    lowest posterior mean for that output (s_t), regret = f_k(s_t) − min_X f_k under the draw."""
    lam, mu, rows = 1 - delta_mod, paths['mu_tested'], []
    for k, c in enumerate(Y_COLS):
        i_best = int(np.argmin(mu[:, k]))
        r = regrets['obj_regret'][:, i_best, k]
        t = sequential_test(r <= eps_abs[k], lam, delta_est_t)
        rows.append({'Objective': Y_SHORT[c], 'best tested (X1, X2)': f"({stats.X1.iloc[i_best]:g}, {stats.X2.iloc[i_best]:g})",
                     'posterior mean there': mu[i_best, k], 'ε_k': eps_abs[k],
                     'median regret': float(np.median(r)), 'P̂(regret ≤ ε_k)': t['estimate'],
                     'CI low': t['ci'][0], 'CI high': t['ci'][1], 'draws used': t['n_used'],
                     'decision': t['decision'].upper()})
    return pd.DataFrame(rows)


def psi_curve(regret, hv_now, eps_pct=np.linspace(0, 12, 121)):
    """Empirical P(regret ≤ ε) as a function of ε in % of the current HV."""
    return eps_pct, np.array([(regret <= e / 100 * hv_now).mean() for e in eps_pct])
