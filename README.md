# The Rise of Negative Earnings and Demand Shifting Investment

Replication package for "The Rise of Negative Earnings and Demand Shifting Investment." This package contains all code to reproduce the empirical analysis and quantitative results in the paper.

---

## Data

**Data access**: A [WRDS](https://wrds-www.wharton.upenn.edu/) subscription is required to run Stage 0 (Compustat pull). All other stages use intermediate files produced by the pipeline.

Raw and intermediate data are **not** included in the repository. All R scripts use a shared working directory:

```
~/Library/CloudStorage/GoogleDrive-<email>/My Drive/research_ideas/negative_earnings/
```

The directory tree expected there:

```
data/
  raw/           ← output of Stage 0
  intermediate/  ← output of Stages 1–2
  clean/         ← output of Stages 3–5
figures/
tables/
```

IRS data (used for validation) is downloaded automatically by `0_pull_IRS/pull_IRS.R` from the IRS Statistics of Income website (pre-1994 statistics are recorded by hand from the public PDFs).

---

## Pipeline

Run stages in order. Each stage consumes the outputs of the previous one.

| Stage | File(s) | Output |
|-------|---------|--------|
| 0 | `0_apiPull_Compustat.R` | `data/raw/compustat_raw.RData` |
| 0b | `0_pull_IRS/pull_IRS.R` → `0_pull_IRS/proc_IRS.py` | Processed IRS aggregate data |
| 1 | `1_cleanCompustat.R` | `data/intermediate/compustat_preMU.csv` |
| 2 | `2_EstProd_DGM.py` | `data/intermediate/compustat_postMU.csv`, `data/clean/ACF_bysector.csv` |
| 3 | `3_build_analysis_data.R` | `data/clean/analysis_data.RData` |
| 4a–4c | `4a_Empirical_Trends.R`, `4b_Empirical_Trends.R`, `4c_Empirical_Trends.R` | Figures and tables |
| 5a | `5a_mstock_coef.R` | `data/clean/sales_elasticity_m_by_year.csv` |
| 5b | `5b_exog_params.R` | `6_ComputationalEx/data/structural_parameters.csv` |
| 6 | `6_ComputationalEx/` (see below) | Quantitative model results and figures |

### Stage-by-stage notes

**Stage 0** — pulls Compustat Fundamentals Annual data (also merges with the CCM linking table for CRSP PERMNOs)

**Stage 1** — deflates to real terms (GDP deflator, base year 2017), constructs key variables (EBITDA, earnings, cost ratios), and filters to the analysis sample (1980–2019, non-utility/finance/unclassified NAICS).

**Stage 2** — estimates sector-level production functions using the ACF method (Ackerberg, Caves & Frazer 2015) via `ProdFun_VPublic.py`; code is courtesy of De Ridder, Grassi & Morzenti (2022).

**Stage 3** — merges in IPO/founding dates (Ritter dataset), constructs customer capital stock via perpetual inventory, builds negative-earnings spell variables, and saves `analysis_data.RData`.

**Stages 4a–4c** — empirical trend figures: share of firms with negative EBITDA, cost structure decomposition, markup and earnings dispersion by sector-year, and cross-sectional regressions.

**Stage 5a** — within-firm regressions of log sales on log customer capital stock (with firm, year, and sector-year FEs) to estimate the demand-shifting elasticity φ by year.

**Stage 5b** — computes exogenous structural parameters (AR(1) productivity moments, production function betas, exit rates) and writes them to `6_ComputationalEx/data/structural_parameters.csv` for use in the quantitative model.

---

## Computational Model (`6_ComputationalEx/`)

The structural model is a heterogeneous-firm general equilibrium with two endogenous firm-level state variables: physical capital and customer capital. Firms choose investment in each, subject to adjustment costs, and face idiosyncratic productivity shocks.

### Solver modules (`ss_solver/`)
- `solve_eqm.py` — steady-state equilibrium solver (`EqmParams` class, least-squares market clearing)
- `solve_vf.py` — EGM value function iteration (I actually iterate on the marginal value functions, using the envelope condition)
- `integrate_dist.py` — stationary distribution computation and aggregate moment extraction
- `prod_fncts.py` — production function, labor demand, and CES aggregator

### Calibration
- `calibrate_investment_params.py` — calibrates adjustment cost parameters `(alpha_a, alpha_k)` via differential evolution + least-squares refinement, targeting the median CapEx/Revenue ratio and share of firms with negative EBITDA
- `calibrate_phi_path.py` — for each year 1980–2019, uses Brent's method to find the φ (demand-shifting elasticity) consistent with the empirical share of negative-EBITDA firms

### Solving the transition path
- `run_phi_path.py` — solves equilibrium at each point along the empirical φ path (parallelized); saves pickled equilibrium objects
- `run_phi_alt_path.py`, `run_sigma_path.py`, `run_sigma_eps_path.py`, `run_beta_path.py` — sensitivity analyses over alternative parameter paths

### Plotting
- `plot_trends.py` — main quantitative figures (share negative, investment ratios, labor allocation, productivity aggregates)
- `plot_aggregates.py` — aggregate capital stock, labor share, and sales-weighted productivity
- `plot_trends_alt.py` — sensitivity and counterfactual visualizations

### HPC
Calibration is computationally intensive. I run most jobs on Northwestern's Quest HPC cluster using the provided SLURM scripts in `6_ComputationalEx/quest_slurm_cmds/`.

Grid resolution is controlled by `grid_config.txt`.

---

## Miscellaneous

- `7a_misc_empirical.R` — supplementary empirical analysis
- `7b_stylized_ex.ipynb` — stylized model examples and illustrative figures
- `wrds_api.ipynb` — one-time setup notebook for WRDS database credentials
