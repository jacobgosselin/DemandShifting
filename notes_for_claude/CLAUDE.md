# DemandShifting — Context for Claude

## What This Project Is

This is the codebase for **"The Rise of Negative Earnings and Demand Shifting Investment"** by Jacob Toner Gosselin and Dalton Rongxuan Zhang (Northwestern University). The paper documents a secular rise in the share of US firms reporting losses (1980–2019), and provides both empirical evidence and a structural model to rationalize it.

**Core mechanism:** A model of heterogeneous firms investing in both *supply-shifting* capital (physical/CapEx) and *demand-shifting* customer capital (SG&A/advertising). The key parameter is **phi (φ)**, the *scale elasticity of demand* — the elasticity of demand per customer with respect to customer base size. Rising φ alone can quantitatively match the rise in negative earnings and qualitatively match persistence of losses, distribution spread, and SG&A growth.

**Aggregate implication:** The model-implied rise in φ lowers GDP by reallocating labor away from production/capital and reallocating demand away from productive firms.

---

## Working Directory

All R scripts set:
```r
setwd("/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings")
```

The GitHub repo (`DemandShifting/`) holds the code; data lives in the Google Drive working directory.

---

## Pipeline Order

| Step | File | Output |
|------|------|--------|
| 0 | `0_apiPull_Compustat.R` | raw Compustat pull via WRDS API |
| 1 | `1_cleanCompustat.R` | `data/intermediate/compustat_preMU.RData` |
| 2 | `2_EstProd_DGM.py` | `data/intermediate/compustat_postMU.csv` (adds `mu_v`), `data/clean/ACF_bysector.csv` |
| 3 | `3_build_analysis_data.R` | `data/clean/analysis_data.RData` (canonical panel + `med_preIPO_growth`) |
| 4a | `4a_Empirical_Trends.R` | figures: neg earnings, distribution spread |
| 4b | `4b_Empirical_Trends_Robustness.R` | robustness figures (alt earnings defs, by sector) |
| 4c | `4c_Spending_Ratios.R` | figures: COGS/SGA/CapEx spending ratios |
| 4d | `4d_Regressions.R` | sector regression tables (He et al. sales & marketing data) |
| 5a | `5a_mstock_coef.R` | `data/clean/sales_elasticity_m_by_year.csv` (year-specific φ estimates) |
| 5b | `5b_exog_params.R` | `6_ComputationalEx/data/structural_parameters.csv` |
| 6 | `6_ComputationalEx/` | model solution, calibration, figures |
| 7a | `7a_misc_empirical.R` | miscellaneous empirical checks |
| 7b | `7b_stylized_ex.ipynb` | stylized example notebook |
| IRS | `IRS_SOI_Code/` | IRS SOI distributional decompositions |

---

## Key Data Objects

- **`analysis_data.RData`**: Main panel. Compustat firms, 1980–2019, filtered to non-finance/non-utility (NAICS ≠ 52, 22; NAICS < 90), non-missing sale/ebitda/cogs, positive sales. Contains: `neg_ebitda`, `neg_spell`, `m_stock`, `sga_sale`, `capx_sale`, `cogs_sale`, `mu`, `cust_capital` (He et al. proxy), founding/age imputation, etc.
- **`compustat_postMU.csv`**: Compustat + markup column `mu_v` from DGM estimator.
- **`ACF_bysector.csv`**: Production function estimates by 2-digit NAICS: `beta_l`, `beta_k`, `rho`, `sigma_xi`.
- **`sales_elasticity_m_by_year.csv`**: Year-specific β_t from regression of log(sales) on log(m_stock) × year.
- **`structural_parameters.csv`**: `gamma_l`, `gamma_k`, `rho`, `sigma_xi`, `sigma`, `exit_rate`, `med_capx_sale`, `neg_ebitda_base`.

---

## Key Variables and Definitions

- **Earnings = EBITDA** (primary); robustness: NI, pre-tax income, profits (sales − COGS − SGA − CapEx)
- **neg_ebitda**: indicator for EBITDA < 0
- **neg_spell**: consecutive years of negative EBITDA (capped at 19 years; data starts 1961)
- **m_inv**: firm SGA / total sector-year SGA (normalized advertising investment)
- **m_stock**: perpetual inventory of `m_inv` with δ_m = 0.15 (FIXED, not calibrated)
  - Initial value imputed from pre-IPO growth rate (`med_preIPO_growth`) and firm age (`founding_imputed`)
- **sga**: SGA excluding R&D (follows Peters & Taylor; missing → 0)
- **rd**: `xrd + rdip`
- **mu**: markup estimated via ratio estimator (= `mu_v` from `2_EstProd_DGM.py`)

---

## Model Summary

**Household:** CES aggregator with scale elasticity φ. Demand for good i: `C_i = M_i^(1+φ) P_i^(-σ) C`.

**Firms:** Heterogeneous productivity (log-normal entry, AR(1) incumbents with params ρ, σ_z). Each firm has states (M_i, K_i, Z_i). Static monopolist pricing; dynamic investment in:
- Physical capital K (supply shifting): `K' = (1-δ_k)K + i`, costs `W·i^(1/α_k)`
- Customer base M (demand shifting): `M' = (1-δ_m)M + a/P_m`, costs `W·a^(1/α_a)`

**Equilibrium:** Stationary distribution with labor/goods/customer markets clearing.

**Key insight:** Customer investment is pure business stealing (P_m adjusts to clear customer market). Physical investment is not. The ratio of marginal profits: `Π_M / Π_K = (K/M) × (1+φ) / (γ_k(σ−1))`.

**Calibration targets:**
- φ = 0 in 1980; φ calibrated to match % neg EBITDA each year 1980–2019
- α_a, α_k calibrated to match (1) % neg EBITDA in 1980 and (2) median CapEx/sales in 1980
- Other params: β=0.96, δ_k=0.10, δ_m=0.15; σ from median markup; γ_k, γ_l, ρ, σ_z from ACF

---

## Computational Model (`6_ComputationalEx/`)

**Solver structure (`ss_solver/`):**
- `solve_vf.py`: value function iteration (endogenous grid method)
- `integrate_dist.py`: stationary distribution integration; computes moments (`pct_negative`, `median_adv_ratio`, `median_inv_ratio`)
- `solve_eqm.py`: `EqmParams` class + `solve_ss_equilibrium_least_squares` — finds fixed point (C*, W*, P_m*)
- `prod_fncts.py`: production function helpers (Numba JIT)

**Key scripts:**
- `calibrate_investment_params.py`: solves 3×3 system for (α_a, α_k, σ) matching base-period moments; writes CSV with 3 columns
- `calibrate_phi_path.py`: calibrates φ path across years to match % neg EBITDA time series
- `plot_trends.py` / `plot_trends_alt.py`: plot model-implied trends vs. data
- `plot_aggregates.py`: GDP/C/I/A aggregates across equilibria
- `run_paths.py`: runs full equilibrium path
- `grid_config.txt`: grid resolution settings (choice_low/high, n_choice_grid, n_prod_grid)

---

## IRS SOI Code (`IRS_SOI_Code/`)

Uses public IRS Statistics of Income (Corporation Complete Report) data to document that the rise in negative earnings appears in the universe of US corporations (not just Compustat).

- `pull_IRS.R`: pulls/processes IRS SOI data
- `distributional_decomp.R`: distributional decomposition of returns across size brackets (8 core brackets: $0–$25K to $50M+) using `agg_brackets_receipts_R5.dta`
- `distributional_decomp_assets.R`: same but by asset size
- `distributional_decomp_sector.R`: by sector
- `proc_IRS.py` / `digitize_irs_pdfs.py`: processing IRS PDFs

---

## Environment

- **R**: 4.4.3 arm64 via `/Library/Frameworks/R.framework/Versions/Current/Resources/bin/Rscript`
- **Python**: scripts in `6_ComputationalEx/` use Numba JIT compilation; run from that directory
- **Key R packages**: `fixest`, `tidyverse`, `dplyr`, `data.table`, `readxl`, `lubridate`, `haven`
- **Key Python packages**: `numpy`, `scipy`, `numba`, `pandas`
