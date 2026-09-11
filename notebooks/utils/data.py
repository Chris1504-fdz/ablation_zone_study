"""Readers for the files exchanged in the campaign, and per-condition statistics.

load_dataset             master spreadsheet (one row per replicate)
load_uic_results         UIC's results workbook (rows = outputs, columns = replicates)
load_sent_recommendation the one-sheet recommendation workbook NU sent to UIC
results_from_table       replicates typed in from an email table
sent_from_email          predicted mean ± σ quoted in an email
condition_stats          replicate rows → condition means, s², n, GP noise s²/n, pooled s̄²
"""
import re
import numpy as np
import pandas as pd
from .config import X_COLS, Y_COLS, NOISE_FLOOR


def find_col(columns, keyword):
    """The unique column whose name contains `keyword` (case-insensitive)."""
    hits = [c for c in columns if keyword.lower() in str(c).lower()]
    assert len(hits) == 1, f"expected one column matching '{keyword}', found {hits}"
    return hits[0]


def load_dataset(path):
    """Master spreadsheet → replicate-level DataFrame with columns X1, X2, Y1, Y2, Y3."""
    raw = pd.read_excel(path)
    col_map = {find_col(raw.columns, 'carbon black'): 'X1', find_col(raw.columns, 'regolith'): 'X2',
               find_col(raw.columns, 'ablative'): 'Y1', find_col(raw.columns, 'backside'): 'Y2',
               find_col(raw.columns, 'density'): 'Y3'}
    return raw.rename(columns=col_map)[X_COLS + Y_COLS].astype(float)


def condition_stats(df_rep, tagger=lambda x1, x2: 0):
    """Replicate rows → (stats, Yvar, s2_pooled).

    stats     : one row per condition — means, replicate variances `Y*_s2`, n, Iteration
    Yvar      : GP observation noise per condition and objective = s²/n (floored)
    s2_pooled : pooled replicate variance s̄² per objective (for prediction intervals)
    """
    g = df_rep.groupby(X_COLS)
    means, var, n = g[Y_COLS].mean(), g[Y_COLS].var(), g.size()
    stats = means.reset_index()
    for c in Y_COLS:
        stats[f'{c}_s2'] = var[c].values
    stats['n'] = n.values
    stats['Iteration'] = [tagger(a, b) for a, b in stats[X_COLS].values]
    Yvar = np.maximum(var.values / n.values[:, None], NOISE_FLOOR)
    s2_pooled = np.maximum(var.values.mean(axis=0), NOISE_FLOOR)
    return stats, Yvar, s2_pooled


def results_from_table(x1, x2, y1, y2, y3, source):
    """Replicates typed in from an email table → replicate-level DataFrame (attrs['source'] kept)."""
    df = pd.DataFrame({'X1': [x1] * len(y1), 'X2': [x2] * len(y1), 'Y1': y1, 'Y2': y2, 'Y3': y3}).astype(float)
    df.attrs['source'] = source
    return df


def load_uic_results(path, x1, x2):
    """UIC's results workbook (one row per output, one column per replicate, optional
    'average' column) → replicate-level DataFrame for the fabricated condition (x1, x2)."""
    raw = pd.read_excel(path, header=None)
    labels = raw.iloc[:, 0].astype(str).str.lower()

    def row(keyword):
        idx = labels[labels.str.contains(keyword)].index
        assert len(idx) == 1, f"expected one row matching '{keyword}' in {path}"
        return idx[0]

    header = raw.iloc[0]
    rep_cols = [c for c in raw.columns[1:] if isinstance(header[c], (int, float, np.integer, np.floating))]
    vals = {c: raw.loc[row(kw), rep_cols].astype(float).tolist()
            for c, kw in [('Y1', 'ablat'), ('Y2', 'backside'), ('Y3', 'density')]}
    return results_from_table(x1, x2, vals['Y1'], vals['Y2'], vals['Y3'], source=path)


def load_sent_recommendation(path):
    """The one-sheet recommendation workbook NU sent to UIC →
    (composition dict in wt%, DataFrame of predicted mean / low / high per objective)."""
    raw = pd.read_excel(path, header=None).fillna('')
    first = raw.iloc[:, 0].astype(str)
    i_comp = first[first.str.startswith('Component')].index[0]
    i_obj = first[first.str.startswith('Objective')].index[0]
    comp = {first[i]: float(raw.iloc[i, 1]) for i in range(i_comp + 1, i_obj)
            if first[i] and raw.iloc[i, 1] != '' and first[i] != 'Predicted values'}
    rows = []
    for i in range(i_obj + 1, i_obj + 1 + len(Y_COLS)):
        lo, hi = [float(v) for v in re.findall(r'[-+]?\d*\.?\d+', str(raw.iloc[i, 2]))][:2]
        rows.append({'Objective': first[i], 'mean': float(raw.iloc[i, 1]), 'low': lo, 'high': hi})
    sent = pd.DataFrame(rows)
    sent.attrs['source'] = path
    sent.attrs['kind'] = '95% expected range per replicate (replicate-noise model)'
    return comp, sent


def sent_from_email(means, sds, source):
    """Predicted mean ± σ quoted in an email (latent 95% CI) → the `sent` table format."""
    from .config import Y_NAMES
    sent = pd.DataFrame({'Objective': [Y_NAMES[c] for c in Y_COLS], 'mean': means, 'sd': sds})
    sent['low'], sent['high'] = sent['mean'] - 1.96 * sent['sd'], sent['mean'] + 1.96 * sent['sd']
    sent.attrs['source'] = source
    sent.attrs['kind'] = '95% interval of the mean (homoscedastic model, no replicate scatter)'
    return sent
