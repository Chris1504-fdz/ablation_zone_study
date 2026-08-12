# Ablation Zone Study — Composite Formulation Analysis & Optimization

Statistical analysis and multi-objective Bayesian optimization (MOBO) of an ablative
composite material (carbon black + Fe-rich regolith filler), a UIC–Northwestern
collaboration (MIRO project). NU builds the models and recommends formulations;
UIC fabricates and tests them.

## Repository Structure

```
ablation_zone_study/
├── README.md
├── data/
│   └── ablation_data.xlsx            # Raw replicate data (1 row = 1 measurement)
├── notebooks/
│   ├── ablation_analysis.ipynb       # ANOVA, regression, heteroscedastic MOGP
│   ├── mobo_analysis.ipynb           # MOBO v1 (data hard-coded inline) — frozen reference
│   └── mobo_analysis_v2.ipynb        # MOBO v2 (reads data/ablation_data.xlsx) — active version
├── slides/
│   ├── Ablation Preliminary Experiment Study 06252026.pptx
│   ├── ablation_study.pptx           # Slides for the statistical analysis
│   └── mobo_results.pptx             # Slides for the MOBO results
└── resources/
    ├── email.pdf                     # NU–UIC email thread (Jul 2026): recommendation + test results
    └── data for nw yinong.docx       # Experimental data document from UIC
```

## Problem Setup

| | Variable | Range / Unit |
|---|---|---|
| **Input X1** | Carbon black content | 5 – 20 wt% |
| **Input X2** | Fe-rich regolith content | 1 – 5 wt% |
| **Output Y1** | Linear ablative rate (minimize) | µm/s |
| **Output Y2** | Backside temperature (minimize) | °C |
| **Output Y3** | Density (minimize) | g/cm³ |

The three objectives conflict (more carbon black → less ablation and lower backside
temperature, but higher density), so the goal is to map the Pareto front with as few
experiments as possible.

## Data

`data/ablation_data.xlsx` — **39 measurements: 13 unique formulations × 3 replicates**
(initial DOE). The MOBO notebooks train on per-condition means; the analysis notebook
uses all replicates.

**To add new experiments:** append one row per replicate to the spreadsheet and re-run
`notebooks/mobo_analysis_v2.ipynb` top to bottom. Everything (GP fit, Pareto front,
hypervolume, next-sample recommendation) updates automatically.

## Notebooks

- **`ablation_analysis.ipynb` — "What does the data tell us?"** Design orthogonality
  checks, two-way ANOVA (carbon black dominates ablation rate), quadratic regression
  per output, and a heteroscedastic multi-output GP (GPyTorch, ICM kernel) capturing
  inter-task correlations. Data is inline; unaffected by spreadsheet updates.

- **`mobo_analysis.ipynb` (v1) — frozen reference.** The original MOBO notebook with
  data hard-coded inline. This is the notebook that produced the recommendation sent
  to UIC on Jul 10, 2026 (Section 4: X1 = 19.579, X2 = 1.986, qLogEHVI = 0.785). Kept
  unchanged for provenance.

- **`mobo_analysis_v2.ipynb` (v2) — active version.** Identical pipeline
  (BoTorch `ModelListGP` of independent `SingleTaskGP`s, qLogEHVI acquisition), but
  loads `../data/ablation_data.xlsx` instead of inline data, and adds Section 10
  reproducing the GP prediction table emailed to UIC on Jul 18, 2026. Runs on the
  `ml_gp_env` Jupyter kernel (the base anaconda env has a broken statsmodels/scipy
  pairing).

### v1 ↔ v2 consistency (verified Aug 12, 2026, on the original 39-point dataset)

| Quantity | v1 | v2 | Rick's email |
|---|---|---|---|
| Initial hypervolume | 51.1684 | 51.1684 | 51.168 |
| Recommended sample (Sec. 4) | X1=19.579, X2=1.986 | X1=19.579, X2=1.986 | 19.58 / 1.99 → tested as 19.6 / 2.0 |
| qLogEHVI at recommendation | 0.7850 | 0.7850 | 0.785 |
| GP prediction at (19.58, 1.99) | — | Y1 5.964±1.028, Y2 177.85±1.43, Y3 1.2157±0.0008 | Y1 5.968±1.026, Y2 177.67±1.43, Y3 1.2160±0.0008 |

The **simulated** 5-iteration BO loop (Sections 6–7, which uses a regression oracle in
place of real experiments) matches v1 at iteration 1 and then drifts slightly
(final HV 56.03 vs 56.05) due to library-version/RNG differences in the multi-restart
acquisition optimizer. This loop is a demonstration only and is superseded by the real
campaign below.

## Campaign Status (as of Aug 2026)

**One real BO iteration is complete** (see `resources/email.pdf`):

1. **Jul 10** — NU (Rick Tsai) recommended X1 = 19.58, X2 = 1.99 (EHVI = 0.785),
   rounded to **carbon black 19.6 wt%, regolith 2.0 wt%** (full formulation also:
   I-369 2 wt%, resin 76.4 wt%).
2. **Jul 18** — GP predictions sent: Y1 = 5.968 ± 1.026 µm/s, Y2 = 177.67 ± 1.434 °C,
   Y3 = 1.2160 ± 0.0008 g/cm³.
3. **Jul 20** — UIC (Yinong Chen) tested 3 replicates: **Y1 = 5.65 / 5.68 / 6.27 µm/s,
   Y2 = 187 / 178 / 182 °C, Y3 = 1.216 g/cm³** — the lowest ablation rates observed
   in the campaign. Y1 and Y3 landed inside the predicted intervals; Y2 was
   under-predicted (2 of 3 replicates above the 95% CI).
4. **Jul 21** — Updated front: **hypervolume 51.168 → 53.240, 14 Pareto solutions.**
   Project handed over from Rick Tsai to Christian Fernandez.

**⚠ The July sample has NOT yet been added to `data/ablation_data.xlsx`** — the
spreadsheet still holds only the initial 13 conditions. Next steps:

- Append the 3 new replicate rows (19.6, 2.0, …) to the spreadsheet.
- Re-run `mobo_analysis_v2.ipynb` to refit the GPs and generate the next
  recommended sample for UIC.
- Consider a better noise model for Y2 (backside temp), which the GP under-covered.

## Dependencies

numpy, pandas, openpyxl, matplotlib, seaborn, statsmodels, scikit-learn, torch,
gpytorch, botorch. Use the `ml_gp_env` kernel (Python 3.13, botorch 0.14) for the
notebooks.
