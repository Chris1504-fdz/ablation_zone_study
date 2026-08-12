# Ablation Zone Study — Composite Formulation Analysis & Optimization

Statistical analysis and multi-objective Bayesian optimization (MOBO) of an ablative
composite material (carbon black + Fe-rich regolith filler), based on ablation
experiments from the UIC–NU collaboration.

## Problem Setup

| | Variable | Range / Unit |
|---|---|---|
| **Input X1** | Carbon black content | 5 – 20 wt% |
| **Input X2** | Fe-rich regolith content | 1 – 5 wt% |
| **Output Y1** | Linear ablative rate (minimize) | µm/s |
| **Output Y2** | Backside temperature (minimize) | °C |
| **Output Y3** | Density (minimize) | g/cm³ |

All three objectives are conflicting: increasing carbon black lowers ablation rate and
backside temperature but raises density, so there is no single best formulation — the
goal is to map the Pareto front.

## Data

`ablation_data.xlsx` — **39 measurements: 13 unique formulations × 3 replicates.**
The design covers X1 ∈ {5, 7.5, 10, 12.5, 15, 20} and X2 ∈ {1, 2, 3, 4, 5} (13
combinations, unbalanced/D-optimal style). Both notebooks also carry this data inline,
so they run without the spreadsheet. The MOBO notebook works with the 13 per-condition
means; the analysis notebook uses all 39 replicates.

## Notebooks

The two notebooks answer different questions on the same data:

- **`ablation_analysis.ipynb` — "What does the data tell us?"** (inference/modeling)
  - Design-matrix orthogonality check (condition number, correlation of X'X)
  - Two-way ANOVA per output → X1 (carbon black) is the dominant factor for ablation
    rate; both factors significantly affect all three responses
  - Quadratic linear regression per output, with parity plots and response surfaces
    (Y1 strong fit, Y2 moderate, Y3 nearly deterministic)
  - Heteroscedastic multi-output GP (GPyTorch, ICM kernel) — learns inter-task
    correlations and per-output noise; compared against the linear models

- **`mobo_analysis.ipynb` — "Which formulation should we test next?"** (optimization)
  - BoTorch surrogates: independent `SingleTaskGP` per objective (`ModelListGP`)
  - Acquisition: `qLogExpectedHypervolumeImprovement`, batch size q = 2
  - Pareto front / hypervolume tracking, candidate recommendation, preference-weighted
    ranking of Pareto solutions

## Optimization Status (as of Aug 2026)

**The MOBO loop is an in-silico demonstration — no new physical experiments have been
run yet.** New candidate evaluations are answered by a quadratic response-surface
"oracle" fit to the existing 39 points (R²: Y1 = 0.92, Y2 = 0.74, Y3 = 1.00), standing
in for real fabrication + testing.

Simulated campaign: 5 iterations × 2 candidates = 10 evaluations added to the 13
initial points (23 total). Hypervolume improved **51.17 → 56.05 (+9.5%)**; Pareto
front grew from 12 to 20 solutions.

### Proposed candidates (oracle-evaluated, original scale)

| Iter | Candidate 1 (X1, X2) | Candidate 2 (X1, X2) |
|---|---|---|
| 1 | (17.08, 1.24) | (20.00, 1.95) |
| 2 | (20.00, 3.87) | (19.45, 1.73) |
| 3 | (14.24, 1.77) | (20.00, 2.42) |
| 4 | (16.69, 4.83) | (18.07, 1.51) |
| 5 | (5.00, 2.49) | (7.33, 3.04) |

For a **real** next experiment, only the first batch is actionable (later iterations
depend on oracle feedback): the acquisition function points to **high carbon black
(≈17–20 wt%) with low regolith (≈1–2 wt%)** — e.g. the single-point recommendation
X1 = 19.58, X2 = 1.99 wt%. These trade slightly higher density for the lowest ablation
rates (~5.5 µm/s) and backside temperatures (~175–190 °C) on the predicted front.

### Next steps (from the notebook)

- Fabricate and test BO-recommended formulations (replace oracle with real data)
- Add constraints (e.g. density < 1.25 g/cm³)
- Consider LVGP for mixed/qualitative variables, heteroscedastic per-replicate noise

## Files

| File | Description |
|---|---|
| `ablation_data.xlsx` | Raw experimental data (39 rows) |
| `ablation_analysis.ipynb` | ANOVA, regression, heteroscedastic MOGP analysis |
| `mobo_analysis.ipynb` | Multi-objective BO (qLogEHVI) with simulated loop |
| `ablation_study.pptx` | Slides for the statistical analysis |
| `mobo_results.pptx` | Slides for the MOBO results |
| `Ablation Preliminary Experiment Study 06252026.pptx` | Preliminary experiment study (June 2026) |
| `data for nw yinong.docx` | Experimental data document from collaborator |

**Dependencies:** numpy, pandas, matplotlib, seaborn, statsmodels, scikit-learn,
torch, gpytorch, botorch.
