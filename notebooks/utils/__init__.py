"""Reusable pieces of the MOBO campaign notebook (notebooks/mobo_analysis_v4.ipynb).

config     constants: columns, names, design bounds, seed, colours
data       readers for the files exchanged with UIC + per-condition statistics
surrogate  design space, replicate-noise GP, prediction, qLogEHVI proposal
plots      every figure
campaign   the Campaign class: the five iteration steps, summaries, export
stopping   probabilistic regret bound stopping rule (Wilson, NeurIPS 2024), hypervolume form
"""
from .config import SEED, DTYPE, X_COLS, Y_COLS, X_NAMES, Y_NAMES, Y_SHORT, X_BOUNDS, NOISE_FLOOR, I369_WT, IT_COLORS
from .data import load_dataset, condition_stats, results_from_table, load_uic_results, load_sent_recommendation, sent_from_email
from .surrogate import DesignSpace, fit_surrogate, predict, propose, acquisition_map, gp_hyperparameters, fit_quality
from .campaign import Campaign
from . import plots
from . import stopping

__all__ = ['SEED', 'DTYPE', 'X_COLS', 'Y_COLS', 'X_NAMES', 'Y_NAMES', 'Y_SHORT', 'X_BOUNDS', 'NOISE_FLOOR', 'I369_WT',
           'IT_COLORS', 'load_dataset', 'condition_stats', 'results_from_table', 'load_uic_results',
           'load_sent_recommendation', 'sent_from_email', 'DesignSpace', 'fit_surrogate', 'predict', 'propose',
           'acquisition_map', 'gp_hyperparameters', 'fit_quality', 'Campaign', 'plots', 'stopping']
