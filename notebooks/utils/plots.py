"""All figures of the campaign notebook. Every function draws one figure from plain data."""
import numpy as np
import torch
import matplotlib.pyplot as plt
from .config import X_COLS, Y_COLS, X_NAMES, Y_NAMES, Y_SHORT, IT_COLORS, DTYPE

XLAB, YLAB = f'$X_1$ ({X_NAMES["X1"]})', f'$X_2$ ({X_NAMES["X2"]})'


def scatter_conditions(ax, stats, first=True, doe_color='white', doe_size=60, bo_size=150):
    """Initial DOE as dots, tested campaign samples as coloured triangles."""
    for it, g in stats.groupby('Iteration'):
        ax.scatter(g.X1, g.X2, c=doe_color if it == 0 else IT_COLORS.get(it, 'gray'),
                   marker='o' if it == 0 else '^', s=doe_size if it == 0 else bo_size,
                   edgecolors='k', zorder=5 + it,
                   label=('Initial DOE' if it == 0 else f'Iteration {it} (tested)') if first else None)


def design_space(stats, proposed=None, title='Design space — tested conditions'):
    fig, ax = plt.subplots(figsize=(6, 4.4))
    scatter_conditions(ax, stats, doe_color='steelblue', doe_size=70, bo_size=170)
    if proposed is not None:
        ax.scatter(*proposed, c='gold', marker='*', s=320, edgecolors='k', zorder=10, label='Proposed (next)')
    ax.set_xlim(4, 21); ax.set_ylim(0.5, 5.5)
    ax.set_xlabel(XLAB); ax.set_ylabel(YLAB); ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


def acquisition(X1g, X2g, acq, stats, cand, title, cand_label='Proposed'):
    fig, ax = plt.subplots(figsize=(6.8, 5))
    cp = ax.contourf(X1g, X2g, acq, levels=30, cmap='viridis')
    plt.colorbar(cp, ax=ax, label='qLogEHVI (brighter = more expected HV gain)')
    scatter_conditions(ax, stats)
    ax.scatter(cand[:, 0], cand[:, 1], c='red', marker='*', s=320, edgecolors='k', zorder=10,
               label=f'{cand_label}\n({cand[0, 0]:.2f}, {cand[0, 1]:.2f})')
    ax.set_xlabel(XLAB); ax.set_ylabel(YLAB); ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc='lower left')
    plt.tight_layout(); plt.show()


def pareto_pairs(stats, pm, title):
    Y = stats[Y_COLS].values; utopia = Y.min(axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (a, b) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
        for it, g in stats.groupby('Iteration'):
            ax.scatter(g[Y_COLS[a]], g[Y_COLS[b]], c=IT_COLORS.get(it, 'gray'),
                       marker='o' if it == 0 else '^', s=60 if it == 0 else 200, edgecolors='k', lw=0.5,
                       zorder=4 + it, label='Initial DOE' if it == 0 else f'Iteration {it} sample')
        ax.scatter(Y[pm, a], Y[pm, b], facecolors='none', edgecolors='k', s=170, lw=1.2, zorder=8,
                   label='Non-dominated (Pareto)')
        ax.scatter(utopia[a], utopia[b], c='gold', marker='*', s=280, edgecolors='k', zorder=9, label='Utopia (ideal)')
        ax.set_xlabel(Y_SHORT[Y_COLS[a]]); ax.set_ylabel(Y_SHORT[Y_COLS[b]])
    axes[0].legend(fontsize=8)
    plt.suptitle(title, fontweight='bold'); plt.tight_layout(); plt.show()


def compare(it, obs, pred, sent=None):
    """Sent interval · v4 CI/PI · observed replicates, one panel per objective."""
    n = len(obs)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for k, (ax, c) in enumerate(zip(axes, Y_COLS)):
        o = obs[c].values; p = pred.loc[k]
        if sent is not None:
            s = sent.loc[k]
            ax.errorbar([0], [s['mean']], yerr=[[s['mean'] - s['low']], [s['high'] - s['mean']]], fmt='D', ms=8,
                        capsize=8, color='dimgray', label='Sent to UIC (95% range)', zorder=3)
        ax.errorbar([1], [p['mean']], yerr=[[p['mean'] - p['pi_low']], [p['pi_high'] - p['mean']]], fmt='none',
                    capsize=10, elinewidth=6, color='lightsteelblue', alpha=0.9, label='v4 95% PI (one replicate)', zorder=2)
        ax.errorbar([1], [p['mean']], yerr=[[p['mean'] - p['ci_low']], [p['ci_high'] - p['mean']]], fmt='s', ms=8,
                    capsize=6, color='tab:blue', label='v4 95% CI (latent mean)', zorder=3)
        ax.scatter([2] * n, o, c='red', marker='*', s=180, edgecolors='k', zorder=5, label='Observed replicates')
        ax.scatter([2], [o.mean()], facecolors='none', edgecolors='red', s=280, zorder=4, label='Observed mean')
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['Sent', 'v4 model', 'Observed'])
        ax.set_xlim(-0.6, 2.6); ax.set_title(Y_SHORT[c], fontsize=10)
        if k == 0: ax.legend(fontsize=7)
    plt.suptitle(f'Iteration {it} — predicted vs. observed at X1 = {obs.X1.iloc[0]:g}, X2 = {obs.X2.iloc[0]:g}',
                 fontweight='bold')
    plt.tight_layout(); plt.show()


def improvement(tab, full=True):
    """Best-so-far per output (top row) and, if `full`, HV growth / Pareto composition / utopia gap."""
    its = tab.Iteration.values
    if full:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8)); top = axes[0]
    else:
        fig, top = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, c in zip(top, Y_COLS):
        best = tab[f'Best {c}'].values
        ax.step(its, best, where='post', marker='o', lw=2, color='tab:blue', zorder=3)
        for t, b in zip(its, best):
            ax.annotate(f'{b:.4g}', (t, b), textcoords='offset points', xytext=(6, 6), fontsize=8)
        ax.set_title(f"{Y_SHORT[c]} — best so far ({tab[f'Δ{c} vs DOE %'].iloc[-1]:+.1f}%)", fontsize=10)
        ax.set_xticks(its); ax.set_xlabel('Iteration (0 = initial DOE)')
    if full:
        ax = axes[1, 0]
        ax.plot(its, tab['HV (means)'], 'o-', lw=2, color='tab:blue', label='Condition means')
        ax.plot(its, tab['HV (reps)'], 's--', lw=2, color='tab:red', label='Replicates as points')
        for t, a, b in zip(its, tab['HV (means)'], tab['HV (reps)']):
            ax.annotate(f'{a:.2f}', (t, a), textcoords='offset points', xytext=(6, -12), fontsize=8, color='tab:blue')
            ax.annotate(f'{b:.2f}', (t, b), textcoords='offset points', xytext=(6, 6), fontsize=8, color='tab:red')
        ax.set_title('Hypervolume growth', fontsize=10); ax.set_xticks(its); ax.set_xlabel('Iteration'); ax.legend(fontsize=8)
        ax = axes[1, 1]
        doe = tab['Pareto size'] - tab['Pareto from BO']
        ax.bar(its, doe, color='steelblue', label='From initial DOE')
        ax.bar(its, tab['Pareto from BO'], bottom=doe, color='crimson', label='From BO samples')
        for t, n_ in zip(its, tab['Pareto size']):
            ax.text(t, n_ + 0.15, f'{int(n_)}', ha='center', fontsize=9)
        ax.set_title('Pareto-front composition', fontsize=10); ax.set_xticks(its)
        ax.set_xlabel('Iteration'); ax.set_ylabel('# Pareto solutions'); ax.legend(fontsize=8)
        ax = axes[1, 2]
        ax.plot(its, tab['Utopia gap'], 'o-', lw=2, color='goldenrod')
        for t, g_ in zip(its, tab['Utopia gap']):
            ax.annotate(f'{g_:.3f}', (t, g_), textcoords='offset points', xytext=(6, 6), fontsize=8)
        ax.set_title('Closest condition to the utopia point (lower = better)', fontsize=10)
        ax.set_xticks(its); ax.set_xlabel('Iteration'); ax.set_ylabel('Normalized distance')
    plt.suptitle(f'Improvement by iteration (through iteration {its[-1]})', fontweight='bold')
    plt.tight_layout(); plt.show()


def prediction_accuracy(acc):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    acc.pivot(index='Iteration', columns='Objective', values='|Δ| / σ_pred (v4)').plot.bar(ax=axes[0], edgecolor='k', rot=0)
    axes[0].axhline(1.96, ls='--', color='gray', lw=1)
    axes[0].set_ylabel('|obs mean − v4 µ| / σ_pred'); axes[0].set_title('Prediction error in σ units (dashed = 1.96)', fontsize=10)
    axes[0].legend(fontsize=7)
    acc.pivot(index='Iteration', columns='Objective', values='reps in v4 PI').plot.bar(ax=axes[1], edgecolor='k', rot=0)
    axes[1].set_ylim(0, 3.4); axes[1].set_ylabel('# replicates inside v4 95% PI (of 3)')
    axes[1].set_title('Interval coverage', fontsize=10); axes[1].legend(fontsize=7)
    plt.suptitle("How well did the model anticipate UIC's results?", fontweight='bold')
    plt.tight_layout(); plt.show()


def response_surfaces(space, model, stats, proposed=None, title='', n=60):
    lb, ub = space.x_bounds
    G1, G2 = np.meshgrid(np.linspace(lb[0], ub[0], n), np.linspace(lb[1], ub[1], n))
    with torch.no_grad():
        post = model.posterior(space.to01(np.column_stack([G1.ravel(), G2.ravel()])))
        mu, sd = -post.mean.numpy(), post.variance.sqrt().numpy()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for k, c in enumerate(Y_COLS):
        for row, (Z, cmap, ttl) in enumerate([(mu[:, k], 'viridis', 'posterior mean'),
                                              (sd[:, k], 'Reds', 'posterior std (latent)')]):
            ax = axes[row, k]; first = (row == 0 and k == 0)
            cp = ax.contourf(G1, G2, Z.reshape(G1.shape), levels=20, cmap=cmap)
            plt.colorbar(cp, ax=ax, label=Y_NAMES[c])
            scatter_conditions(ax, stats, first=first, doe_size=40)
            if proposed is not None:
                ax.scatter(*proposed, c='gold', marker='*', s=240, edgecolors='k', zorder=9,
                           label='Proposed (next)' if first else None)
            if first: ax.legend(fontsize=7, loc='upper left', framealpha=0.9)
            ax.set_xlabel(XLAB); ax.set_ylabel(YLAB); ax.set_title(f'{c} — GP {ttl}', fontsize=10)
    plt.suptitle(title, fontweight='bold'); plt.tight_layout(); plt.show()


def regret_histogram(regret, hv_table, observed_gains, title):
    """Posterior distribution of the hypervolume regret with the ε thresholds and the HV gains
    observed so far as reference lines."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(regret, bins=40, color='steelblue', alpha=0.8, edgecolor='k', lw=0.4)
    ymax = ax.get_ylim()[1]
    for _, r in hv_table.iterrows():
        ax.axvline(r['ε (HV units)'], color='crimson', ls='--', lw=1)
        ax.text(r['ε (HV units)'], ymax * 0.97, f" ε={r['ε (% of HV)']:g}%\n P={r['P̂(regret ≤ ε)']:.2f}",
                fontsize=7, color='crimson', va='top')
    for k, g in enumerate(observed_gains):
        ax.axvline(g, color='seagreen', ls=':', lw=1.5)
        ax.text(g, ymax * 0.55, f' gain iter {k + 1}\n {g:.2f}', fontsize=7, color='seagreen', va='top')
    ax.set_xlabel('Best one-step hypervolume gain  max_x HVI(x)  (HV units)'); ax.set_ylabel('# posterior draws')
    ax.set_title(title, fontsize=10)
    plt.tight_layout(); plt.show()


def stopping_trajectory(traj, lam):
    fig, ax = plt.subplots(figsize=(7, 4))
    for eps, g in traj.groupby('ε (% of HV)'):
        ax.plot(g['Check before iteration'], g['P̂(regret ≤ ε)'], 'o-', lw=2, label=f'ε = {eps:g}% of HV')
    ax.axhline(lam, color='k', ls='--', lw=1); ax.text(traj['Check before iteration'].min(), lam, f' λ = 1 − δ_mod = {lam:g}', va='bottom', fontsize=8)
    ax.set_ylim(-0.03, 1.05); ax.set_xticks(sorted(traj['Check before iteration'].unique()))
    ax.set_xlabel('Stopping check before requesting iteration'); ax.set_ylabel('P̂(one-step HV regret ≤ ε)')
    ax.set_title('Probabilistic regret bound over the campaign (stop when a curve clears λ)', fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()


def improvement_map(x_best, stats, title):
    """Where the posterior draws locate the best one-step improvement (2-D histogram) with the
    tested conditions on top."""
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    h = ax.hist2d(x_best[:, 0], x_best[:, 1], bins=[30, 20], range=[[5, 20], [1, 5]], cmap='magma')
    plt.colorbar(h[3], ax=ax, label='# draws whose best next point falls here')
    scatter_conditions(ax, stats, doe_color='white', doe_size=40, bo_size=140)
    ax.set_xlabel(XLAB); ax.set_ylabel(YLAB); ax.set_title(title, fontsize=10); ax.legend(fontsize=7, loc='upper left')
    plt.tight_layout(); plt.show()


# ────────────────────────────── stopping rule, step by step ──────────────────
def posterior_draws(paths, stats, n_show=3, title=''):
    """STEP 1 figure: posterior mean and a few joint draws of every objective on the grid."""
    G1, G2, g = paths['G1'], paths['G2'], paths['grid_n']
    fig, axes = plt.subplots(3, n_show + 1, figsize=(4.2 * (n_show + 1), 10.5))
    for k, c in enumerate(Y_COLS):
        panels = [(paths['mu_grid'][:, k], 'posterior mean')] + [(paths['F_grid'][i, :, k], f'draw #{i + 1}') for i in range(n_show)]
        vmin = min(z.min() for z, _ in panels); vmax = max(z.max() for z, _ in panels)
        for col, (Z, ttl) in enumerate(panels):
            ax = axes[k, col]
            cp = ax.contourf(G1, G2, Z.reshape(g, g), levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
            plt.colorbar(cp, ax=ax, label=Y_NAMES[c] if col == n_show else None)
            scatter_conditions(ax, stats, first=(k == 0 and col == 0), doe_size=25, bo_size=90)
            ax.set_title(f'{c} — {ttl}', fontsize=9); ax.set_xlabel(XLAB, fontsize=8); ax.set_ylabel(YLAB, fontsize=8)
            if k == 0 and col == 0: ax.legend(fontsize=6, loc='upper left')
    plt.suptitle(title or 'Step 1 — posterior mean and joint draws of the latent objectives', fontweight='bold')
    plt.tight_layout(); plt.show()


def draws_at_tested(paths, stats, title=''):
    """STEP 1 figure: the sampled values at the tested conditions vs. the observed condition means."""
    F = paths['F_tested']; labels = [f'({a:g},{b:g})' for a, b in stats[X_COLS].values]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for k, (ax, c) in enumerate(zip(axes, Y_COLS)):
        parts = ax.violinplot(F[:, :, k], positions=range(len(labels)), widths=0.8, showextrema=False)
        for b in parts['bodies']: b.set_facecolor('lightsteelblue'); b.set_alpha(0.9)
        ax.scatter(range(len(labels)), paths['mu_tested'][:, k], marker='_', s=200, color='tab:blue', zorder=4, label='posterior mean')
        ax.scatter(range(len(labels)), stats[c].values, c=[IT_COLORS.get(it, 'gray') for it in stats.Iteration],
                   edgecolors='k', s=40, zorder=5, label='observed condition mean')
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_title(Y_SHORT[c], fontsize=10)
        if k == 0: ax.legend(fontsize=7)
    plt.suptitle(title or 'Step 1 — draws at the tested conditions (violins = posterior of the latent value, incl. replicate noise)',
                 fontweight='bold')
    plt.tight_layout(); plt.show()


def regret_examples(examples, stats, hv_now, title=''):
    """STEP 2 figure: for a few draws, the tested set under the draw (Pareto circled), the whole
    surface under the draw (cloud), and the best single additional point with its HV gain."""
    n = len(examples)
    fig, axes = plt.subplots(n, 4, figsize=(19, 4.2 * n))
    axes = np.atleast_2d(axes)
    for r, ex in enumerate(examples):
        Yt, Yg, pm = ex['Y_tested'], ex['Y_grid'], ex['pm']
        for col, (a, b) in enumerate([(0, 1), (0, 2), (1, 2)]):
            ax = axes[r, col]
            ax.scatter(Yg[:, a], Yg[:, b], s=4, c='lightgray', label='surface under this draw (grid)')
            ax.scatter(Yt[:, a], Yt[:, b], c='steelblue', edgecolors='k', s=50, zorder=4, label='tested, under this draw')
            ax.scatter(Yt[pm, a], Yt[pm, b], facecolors='none', edgecolors='k', s=140, lw=1.2, zorder=5, label='Pareto (this draw)')
            ax.scatter(ex['y_best'][a], ex['y_best'][b], c='red', marker='*', s=260, edgecolors='k', zorder=6,
                       label=f"best extra point: +{ex['hvi']:.2f} HV")
            ax.set_xlabel(Y_SHORT[Y_COLS[a]]); ax.set_ylabel(Y_SHORT[Y_COLS[b]])
            if col == 0:
                ax.legend(fontsize=7)
                ax.set_title(f"draw #{ex['i'] + 1}: HV(tested) = {ex['hv_tested']:.2f} → +{ex['hvi']:.3f} with one more point", fontsize=9)
        ax = axes[r, 3]
        scatter_conditions(ax, stats, doe_color='white', doe_size=40, bo_size=140)
        ax.scatter(*ex['x_best'], c='red', marker='*', s=300, edgecolors='k', zorder=10,
                   label=f"best extra point ({ex['x_best'][0]:.1f}, {ex['x_best'][1]:.1f})")
        ax.set_xlim(4, 21); ax.set_ylim(0.5, 5.5); ax.set_xlabel(XLAB); ax.set_ylabel(YLAB)
        ax.set_title('where that point is', fontsize=9); ax.legend(fontsize=7, loc='upper left')
    plt.suptitle(title or f'Step 2 — one-step hypervolume regret under individual draws (observed HV = {hv_now:.2f})',
                 fontweight='bold')
    plt.tight_layout(); plt.show()


def sequential_rounds(rounds, lam, title=''):
    """STEP 3 figure: the sequential Clopper–Pearson test round by round."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(rounds['draws n_j'], rounds['estimate'],
                yerr=[rounds['estimate'] - rounds['CI low'], rounds['CI high'] - rounds['estimate']],
                fmt='o-', capsize=5, color='tab:blue', label='estimate ± Clopper–Pearson interval')
    ax.axhline(lam, color='k', ls='--', lw=1, label=f'λ = 1 − δ_mod = {lam:g}')
    ax.set_xscale('log'); ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel('draws used n_j (log scale)'); ax.set_ylabel('P̂(regret ≤ ε)')
    ax.set_title(title or 'Step 3 — sequential test: stop sampling once λ leaves the interval', fontsize=10)
    ax.legend(fontsize=8); plt.tight_layout(); plt.show()


def psi_vs_eps(curves, lam_lines, eps_marks, title=''):
    """P(regret ≤ ε) as a function of ε for several stages (curves = [(label, eps_pct, psi), ...]),
    with λ lines for several δ and the ε values used in the tables marked."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, e, p in curves:
        ax.plot(e, p, lw=2, label=label)
    for d, lam in lam_lines:
        ax.axhline(lam, color='k', ls='--', lw=0.8, alpha=0.7); ax.text(e[-1], lam, f' δ = {d:g}', fontsize=7, va='bottom', ha='right')
    for m in eps_marks:
        ax.axvline(m, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('ε as % of the current hypervolume'); ax.set_ylabel('P(one-step HV regret ≤ ε)')
    ax.set_ylim(-0.02, 1.03); ax.set_title(title or 'Probability that no single further experiment is worth more than ε', fontsize=10)
    ax.legend(fontsize=8, loc='lower right'); plt.tight_layout(); plt.show()


def regret_by_stage(stages, title=''):
    """Overlaid histograms of the one-step regret for each stage (label, regret array)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for (label, r), col in zip(stages, ['tab:gray', 'limegreen', 'darkorange', 'seagreen', 'orchid']):
        ax.hist(r, bins=40, alpha=0.5, color=col, edgecolor='k', lw=0.3, label=f'{label} (mean {r.mean():.2f})')
    ax.set_xlabel('best one-step hypervolume gain  max_x HVI(x)  (HV units)'); ax.set_ylabel('# draws')
    ax.set_title(title or 'How the expected gain of one more experiment shrank over the campaign', fontsize=10)
    ax.legend(fontsize=8); plt.tight_layout(); plt.show()


# ────────────────────────────── figures for sharing ──────────────────────────
def model_performance(perf_rows, path=None):
    """Predicted (as sent to UIC, with the expected range) vs. measured, per objective and
    iteration. perf_rows: list of dicts with it, objective key c, pred mean, low, high, obs array."""
    its = sorted({r['it'] for r in perf_rows})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for k, (ax, c) in enumerate(zip(axes, Y_COLS)):
        for r in [r for r in perf_rows if r['c'] == c]:
            x = r['it']
            ax.errorbar([x - 0.15], [r['mean']], yerr=[[r['mean'] - r['low']], [r['high'] - r['mean']]], fmt='D', ms=8,
                        capsize=8, color='k', label='Predicted before the test (bar = expected range)' if r['it'] == its[0] else None, zorder=3)
            ax.scatter([x + 0.15] * len(r['obs']), r['obs'], c='red', marker='*', s=170, edgecolors='k', zorder=5,
                       label='Measured by UIC (each replicate)' if r['it'] == its[0] else None)
            ax.scatter([x + 0.15], [r['obs'].mean()], facecolors='none', edgecolors='red', s=300, lw=1.5, zorder=4,
                       label='Measured mean' if r['it'] == its[0] else None)
            ax.annotate(f"{r['fab']}", (x, ax.get_ylim()[0]), textcoords='offset points', xytext=(0, 4), ha='center', fontsize=8, color='dimgray')
        ax.set_xticks(its); ax.set_xticklabels([f'Iteration {i}' for i in its]); ax.set_xlim(its[0] - 0.6, its[-1] + 0.6)
        ax.set_ylabel(Y_NAMES[c]); ax.set_title(Y_NAMES[c], fontsize=10); ax.grid(alpha=0.3)
        if k == 0: ax.legend(fontsize=7, loc='best')
    plt.suptitle('What we predicted vs. what UIC measured, iteration by iteration', fontweight='bold')
    plt.tight_layout()
    if path: fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.show()


def table_png(df, path, title='', col_widths=None, fontsize=8):
    """Render a small DataFrame as an image (for emails / slides)."""
    n_rows, n_cols = df.shape
    fig, ax = plt.subplots(figsize=(min(2.1 * n_cols, 22), 0.34 * (n_rows + 1) + 0.9))
    ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center',
                   colWidths=col_widths or [1.0 / n_cols] * n_cols)
    tbl.auto_set_font_size(False); tbl.set_fontsize(fontsize); tbl.scale(1, 1.45)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0: cell.set_text_props(weight='bold'); cell.set_facecolor('#dfe7f1')
        elif r % 2 == 0: cell.set_facecolor('#f5f7fa')
    if title: ax.set_title(title, fontweight='bold', pad=8)
    fig.savefig(path, dpi=220, bbox_inches='tight'); plt.show()


def stopping_summary(traj, observed_gains_pct, lam, eps_show=(1.0, 2.0, 5.0), path=None):
    """The one figure to share about stopping.
    Left: how much one more experiment is expected to gain, check by check, next to what each
    experiment actually gained. Right: the probability that NO further experiment is worth more
    than a small tolerance, against the confidence threshold used to stop."""
    stages = traj.drop_duplicates('Check before iteration')
    its = stages['Check before iteration'].values
    gain_pct = 100 * stages['E[max HVI]'].values / stages['HV (means)'].values
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8))
    ax = axes[0]
    ax.bar(its, gain_pct, color='steelblue', edgecolor='k', width=0.55, label='Expected gain of the next experiment (model)')
    for x, g in zip(its, gain_pct):
        ax.text(x, g + 0.08, f'{g:.1f}%', ha='center', fontsize=9)
    obs_x = [it for it, g in observed_gains_pct]; obs_y = [g for it, g in observed_gains_pct]
    if obs_x:
        ax.scatter(obs_x, obs_y, marker='D', s=80, c='darkorange', edgecolors='k', zorder=5, label='Gain the experiment actually delivered')
    ax.set_xticks(its); ax.set_xticklabels([f'before\niteration {i}' for i in its])
    ax.set_ylabel('Improvement of the Pareto front (% of current hypervolume)')
    ax.set_title('How much is left to gain from one more experiment?', fontsize=11)
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    ax = axes[1]
    for eps in eps_show:
        g = traj[np.isclose(traj['ε (% of HV)'], eps)]
        ax.plot(g['Check before iteration'], 100 * g['P̂(regret ≤ ε)'], 'o-', lw=2.2, ms=7,
                label=f'if a gain below {eps:g}% is negligible')
    ax.axhline(100 * lam, color='k', ls='--', lw=1.2)
    ax.text(its[0] - 0.05, 100 * lam + 1.2, f'stop when a line is above this ({100 * lam:g}% confidence)', fontsize=8)
    ax.set_xticks(its); ax.set_xticklabels([f'before\niteration {i}' for i in its]); ax.set_ylim(-3, 108)
    ax.set_ylabel('Chance that no further experiment beats the tolerance (%)')
    ax.set_title('Is another experiment still worth it?', fontsize=11)
    ax.legend(fontsize=8, loc='center left'); ax.grid(alpha=0.3)
    plt.suptitle('Stopping the campaign — probabilistic regret bound (Wilson, NeurIPS 2024)', fontweight='bold')
    plt.tight_layout()
    if path: fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.show()
