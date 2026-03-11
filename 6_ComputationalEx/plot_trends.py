"""
plot_trends.py
--------------
Load the solved equilibria from solve_phi_path.py and generate all trend figures.

Reads:  data/clean/solved_eqm/eqm_phi_path_all.pkl  (produced by solve_phi_path.py)
Writes: 6 PDF figures to FIGURES_DIR

Set DEMANDSHIFT_FIGURES_DIR env var to override the default figures path.

Usage:
  python plot_trends.py
  DEMANDSHIFT_FIGURES_DIR=/path/to/figures python plot_trends.py
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from integrate_dist import (
    pct_negative, pct_negative_income, est_dist, est_sd,
    median_adv_ratio, median_inv_ratio, median_cogs_ratio,
    median_earnings, mean_earnings,
    agg_capital_stock, sales_wtd_productivity, agg_labor_shares,
    avg_firm_earnings_path, cohort_neg_path, avg_age_neg_earners,
)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = "/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings"
SOLVED_EQM_DIR = os.path.join(MAIN_DIR, "data", "clean")

_default_figures = os.path.join(
    os.path.expanduser("~"),
    "Library/CloudStorage/"
    "GoogleDrive-jacob.gosselin@u.northwestern.edu/"
    "My Drive/research_ideas/negative_earnings/figures/quantitative",
)
FIGURES_DIR = os.environ.get("DEMANDSHIFT_FIGURES_DIR", _default_figures)
os.makedirs(FIGURES_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------

pkl_path = os.path.join(SOLVED_EQM_DIR, "eqm_phi_path.pkl")
print(f"Loading equilibria from {pkl_path} ...")
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

eqms       = data["eqms"]        # {year: eqm_dict}
phi_by_year = data["phi_values"]  # {year: phi}
years      = data["years"]        # sorted list of int years
m_grid     = data["grids"]["m_grid"]
k_grid     = data["grids"]["k_grid"]
z_grid     = data["grids"]["z_grid"]
Pi         = data["grids"]["Pi"]

print(f"Loaded {len(eqms)} equilibria for years {years[0]}–{years[-1]}.\n")

# Load empirical time-series moments (produced by 5b_exog_params.R)
_emp_csv = os.path.join(_DIR, "empirical_trends_byyear.csv")
emp_df = pd.read_csv(_emp_csv).set_index("year")
_base_log_sd_earn = emp_df.loc[years[0], "log_sd_earnings"]
_base_log_sd_sale = emp_df.loc[years[0], "log_sd_sales"]
emp_df["log_sd_earnings_norm"] = emp_df["log_sd_earnings"] - _base_log_sd_earn
emp_df["log_sd_sales_norm"]    = emp_df["log_sd_sales"]    - _base_log_sd_sale

# phi by year (from solved equilibrium path)
phi_vals = [phi_by_year[yr] for yr in years]

# -----------------------------------------------------------------------------
# Compute moments
# -----------------------------------------------------------------------------

pct_neg_vals  = []
pct_neg_income_vals = []
sd_earnings_raw, sd_sales_raw = [], []
med_adv_all, med_inv_all, med_cogs_all = [], [], []
med_adv_neg, med_inv_neg, med_cogs_neg = [], [], []
med_earn_vals, mean_earn_vals = [], []
c_vals        = []
m_bnd_vals    = []
k_bnd_vals    = []
agg_k_vals    = []
sales_wtd_z_vals = []
La_vals, Lk_vals, Ls_vals = [], [], []

for yr in years:
    eqm  = eqms[yr]
    dist = eqm["Dist"]

    pct_neg_vals.append(pct_negative(m_grid, k_grid, z_grid, eqm))
    pct_neg_income_vals.append(pct_negative_income(m_grid, k_grid, z_grid, eqm))

    m_bnd_vals.append(np.sum(dist[-10:, :, :]) / np.sum(dist))
    k_bnd_vals.append(np.sum(dist[:, -10:, :]) / np.sum(dist))

    _, earnings_cdf = est_dist(m_grid, k_grid, z_grid, eqm, "earnings")
    _, sales_cdf    = est_dist(m_grid, k_grid, z_grid, eqm, "revenue")
    sd_earnings_raw.append(est_sd(earnings_cdf))
    sd_sales_raw.append(est_sd(sales_cdf))

    med_adv_all.append(median_adv_ratio(m_grid, k_grid, z_grid, eqm, negative_earnings_only=False))
    med_inv_all.append(median_inv_ratio(m_grid, k_grid, z_grid, eqm, negative_earnings_only=False))
    med_cogs_all.append(median_cogs_ratio(m_grid, k_grid, z_grid, eqm, negative_earnings_only=False))
    med_adv_neg.append(median_adv_ratio(m_grid, k_grid, z_grid, eqm, negative_earnings_only=True))
    med_inv_neg.append(median_inv_ratio(m_grid, k_grid, z_grid, eqm, negative_earnings_only=True))
    med_cogs_neg.append(median_cogs_ratio(m_grid, k_grid, z_grid, eqm, negative_earnings_only=True))

    med_earn_vals.append(median_earnings(m_grid, k_grid, z_grid, eqm))
    mean_earn_vals.append(mean_earnings(m_grid, k_grid, z_grid, eqm))

    c_vals.append(eqm["c_agg"])
    agg_k_vals.append(agg_capital_stock(m_grid, k_grid, z_grid, eqm))
    sales_wtd_z_vals.append(sales_wtd_productivity(m_grid, k_grid, z_grid, eqm))
    La, Lk, Ls = agg_labor_shares(m_grid, k_grid, z_grid, eqm)
    La_vals.append(La); Lk_vals.append(Lk); Ls_vals.append(Ls)
    print(
        f"  {yr}: pct_neg={pct_neg_vals[-1]:.2f}%  "
        f"m_bnd={m_bnd_vals[-1]:.4f}  k_bnd={k_bnd_vals[-1]:.4f}"
    )

# Log-normalize standard deviations (log change from base year)
sd_earnings_arr = np.array(sd_earnings_raw)
sd_sales_arr    = np.array(sd_sales_raw)
sd_earnings_vals = np.log(sd_earnings_arr / sd_earnings_arr[0])
sd_sales_vals    = np.log(sd_sales_arr    / sd_sales_arr[0])

# Average age of negative-earning firms in the stationary distribution
print("\nComputing avg age of neg-earning firms by year (slow) ...")
avg_neg_vals = [avg_age_neg_earners(eqms[yr], z_grid, Pi, T=10) for yr in years]

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def _save(fname):
    path = os.path.join(FIGURES_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


print(f"\nGenerating figures → {FIGURES_DIR}/")

# Figure 1: Percent negative earnings vs year
plt.figure(figsize=(10, 10))
plt.plot(years, pct_neg_vals, "o-", linewidth=3, markersize=10)
plt.xlabel("Year", fontsize=18)
plt.ylabel("Percent of Firms with Negative Earnings", fontsize=18)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.grid(True, alpha=0.3)
_save("pct_negative_vs_phi.pdf")

# Figure 2: Median ratios — all firms
plt.figure(figsize=(10, 10))
plt.plot(years, med_adv_all,  "o-", linewidth=3, markersize=10, label="Adv/Revenue")
plt.plot(years, med_inv_all,  "s-", linewidth=3, markersize=10, label="Inv/Revenue")
plt.plot(years, med_cogs_all, "^-", linewidth=3, markersize=10, label="COGS/Revenue")
plt.xlabel("Year", fontsize=18)
plt.ylabel("Median Ratio", fontsize=18)
plt.legend(fontsize=16)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.grid(True, alpha=0.3)
_save("median_ratios_all_vs_phi.pdf")

# Figure 3: Median ratios — negative-earnings firms
plt.figure(figsize=(10, 10))
plt.plot(years, med_adv_neg,  "o-", linewidth=3, markersize=10, label="Adv/Revenue")
plt.plot(years, med_inv_neg,  "s-", linewidth=3, markersize=10, label="Inv/Revenue")
plt.plot(years, med_cogs_neg, "^-", linewidth=3, markersize=10, label="COGS/Revenue")
plt.xlabel("Year", fontsize=18)
plt.ylabel("Median Ratio (Negative Earnings Firms)", fontsize=18)
plt.legend(fontsize=16)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.grid(True, alpha=0.3)
_save("median_ratios_neg_vs_phi.pdf")

# Figure 4: Std. dev. of earnings (log change from base year)
plt.figure(figsize=(10, 10))
plt.plot(years, sd_earnings_vals, "o-", linewidth=3, markersize=10)
plt.xlabel("Year", fontsize=18)
plt.ylabel("Std. Deviation of Earnings (Log Change from 1980)", fontsize=18)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.grid(True, alpha=0.3)
_save("sd_earnings_vs_phi.pdf")

# Figure 5: Std. dev. of sales (log change from base year)
plt.figure(figsize=(10, 10))
plt.plot(years, sd_sales_vals, "o-", linewidth=3, markersize=10)
plt.xlabel("Year", fontsize=18)
plt.ylabel("Std. Deviation of Sales (Log Change from 1980)", fontsize=18)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.grid(True, alpha=0.3)
_save("sd_sales_vs_phi.pdf")

# Figure 6: Aggregate consumption
plt.figure(figsize=(10, 10))
plt.plot(years, c_vals, "o-", linewidth=3, markersize=10)
plt.xlabel("Year", fontsize=18)
plt.ylabel("Aggregate Consumption C", fontsize=18)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.grid(True, alpha=0.3)
_save("aggregate_consumption_vs_phi.pdf")

# Figure 7: Median and mean earnings (scaled by aggregate revenue)
plt.figure(figsize=(10, 10))
plt.plot(years, med_earn_vals, "o-", linewidth=3, markersize=10, label="Median Earnings")
plt.plot(years, mean_earn_vals, "s-", linewidth=3, markersize=10, label="Mean Earnings")
plt.xlabel("Year", fontsize=18)
plt.ylabel("Earnings", fontsize=18)
plt.legend(fontsize=16)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.grid(True, alpha=0.3)
_save("earnings_median_mean_vs_phi.pdf")

# Figure 8: Aggregate capital stock
plt.figure(figsize=(10, 10))
plt.plot(years, agg_k_vals, "o-", linewidth=3, markersize=10)
plt.xlabel("Year", fontsize=18)
plt.ylabel("Aggregate Capital Stock", fontsize=18)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.grid(True, alpha=0.3)
_save("agg_capital_stock_vs_phi.pdf")

# Figure 9: Sales-weighted productivity
plt.figure(figsize=(10, 10))
plt.plot(years, sales_wtd_z_vals, "o-", linewidth=3, markersize=10)
plt.xlabel("Year", fontsize=18)
plt.ylabel("Sales-Weighted Productivity", fontsize=18)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.grid(True, alpha=0.3)
_save("sales_wtd_productivity_vs_phi.pdf")

# Figure 10: Aggregate labor allocations
plt.figure(figsize=(10, 10))
plt.plot(years, La_vals, "o-", linewidth=3, markersize=10, label=r"$L_a$ (advertising)")
plt.plot(years, Lk_vals, "s-", linewidth=3, markersize=10, label=r"$L_k$ (capital inv.)")
plt.plot(years, Ls_vals, "^-", linewidth=3, markersize=10, label=r"$L_s$ (goods prod.)")
plt.xlabel("Year", fontsize=18)
plt.ylabel("Aggregate Labor Allocation", fontsize=18)
plt.legend(fontsize=16)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.grid(True, alpha=0.3)
_save("agg_labor_shares_vs_phi.pdf")
# Print La, Lk, Ls values for reference
print("\nAggregate labor allocations by year:")
for yr, La, Lk, Ls in zip(years, La_vals, Lk_vals, Ls_vals):
    print(f"  {yr}: La={La:.4f}, Lk={Lk:.4f}, Ls={Ls:.4f}")

# Figure 11: Average firm earnings path — phi_0 vs phi_T
eqm_phi0 = eqms[years[0]]
eqm_phiT = eqms[years[-1]]
path_phi0 = avg_firm_earnings_path(eqm_phi0, z_grid, T=50)
path_phiT = avg_firm_earnings_path(eqm_phiT, z_grid, T=50)
periods = np.arange(1, 51)
plt.figure(figsize=(10, 10))
plt.plot(periods, path_phi0, "o-", linewidth=3, markersize=10, label=f"{years[0]} ($\\phi_0$)")
plt.plot(periods, path_phiT, "s-", linewidth=3, markersize=10, label=f"{years[-1]} ($\\phi_T$)")
plt.axhline(0, color="black", linewidth=1, linestyle="--")
plt.xlabel("Period", fontsize=18)
plt.ylabel("Earnings (Median Productivity Entrant)", fontsize=18)
plt.legend(fontsize=16)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.grid(True, alpha=0.3)
_save("avg_firm_earnings_path_phi0_vs_phiT.pdf")

# Figure 12: Cohort pct negative earnings by period — phi_0 vs phi_T
print("\nComputing cohort earnings paths (this may take ~15 sec)...")
cneg_phi0 = cohort_neg_path(eqm_phi0, z_grid, Pi, T=10)
cneg_phiT = cohort_neg_path(eqm_phiT, z_grid, Pi, T=10)
periods = np.arange(1, 11)
print(f"  Expected neg-earnings periods (phi_0): {np.sum(cneg_phi0)/100:.2f}")
print(f"  Expected neg-earnings periods (phi_T): {np.sum(cneg_phiT)/100:.2f}")
plt.figure(figsize=(10, 10))
plt.plot(periods, cneg_phi0, "o-", linewidth=3, markersize=10, label=f"{years[0]} ($\\phi_0$)")
plt.plot(periods, cneg_phiT, "s-", linewidth=3, markersize=10, label=f"{years[-1]} ($\\phi_T$)")
plt.xlabel("Period", fontsize=18)
plt.ylabel("% of Entrant Cohort with Negative Earnings", fontsize=18)
plt.legend(fontsize=16)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.grid(True, alpha=0.3)
# x-axis ticks only at integers
plt.xticks(periods)
_save("cohort_pct_neg_by_period_phi0_vs_phiT.pdf")
print(cneg_phiT)

# Figure A: Two-panel — phi by year + pct_neg by year
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
ax1.plot(years, phi_vals, "o-", linewidth=3, markersize=10)
ax1.set_xlabel("Year", fontsize=18)
ax1.set_ylabel(r"$\phi$ (Sales Elasticity of Customer Capital)", fontsize=18)
ax1.tick_params(axis="both", which="major", labelsize=16)
ax1.grid(True, alpha=0.3)
ax2.plot(years, pct_neg_vals, "o-", linewidth=3, markersize=10)
ax2.set_xlabel("Year", fontsize=18)
ax2.set_ylabel("Percent of Firms with EBITDA < 0 (%)", fontsize=18)
ax2.tick_params(axis="both", which="major", labelsize=16)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
_save("phi_and_pct_neg_by_year.pdf")

# Figure B: Two-panel — empirical vs. model avg neg spell/periods by year
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
ax1.plot(emp_df.index, emp_df["avg_neg_spell"], "o-", linewidth=3, markersize=10)
ax1.set_xlabel("Year", fontsize=18)
ax1.set_ylabel("Avg. Neg. Earnings Spell Length (Data)", fontsize=18)
ax1.tick_params(axis="both", which="major", labelsize=16)
ax1.grid(True, alpha=0.3)
ax2.plot(years, avg_neg_vals, "o-", linewidth=3, markersize=10)
ax2.set_xlabel("Year", fontsize=18)
ax2.set_ylabel("Avg. Age of Neg.-Earning Firms (Model, periods)", fontsize=18)
ax2.tick_params(axis="both", which="major", labelsize=16)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
_save("avg_neg_periods_by_year.pdf")

# Figure C: Two-panel — empirical vs model cost ratios
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
ax1.plot(emp_df.index, emp_df["med_sga_sale"],  "o-", linewidth=3, markersize=8, label="SG&A/Rev")
ax1.plot(emp_df.index, emp_df["med_cogs_sale"], "s-", linewidth=3, markersize=8, label="COGS/Rev")
ax1.plot(emp_df.index, emp_df["med_capx_sale"], "^-", linewidth=3, markersize=8, label="CapEx/Rev")
ax1.set_xlabel("Year", fontsize=18)
ax1.set_ylabel("Median Ratio (Data)", fontsize=18)
ax1.legend(fontsize=14)
ax1.tick_params(axis="both", which="major", labelsize=16)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)
ax2.plot(years, med_adv_all,  "o-", linewidth=3, markersize=8, label="Adv/Rev")
ax2.plot(years, med_cogs_all, "s-", linewidth=3, markersize=8, label="COGS/Rev")
ax2.plot(years, med_inv_all,  "^-", linewidth=3, markersize=8, label="Inv/Rev")
ax2.set_xlabel("Year", fontsize=18)
ax2.set_ylabel("Median Ratio (Model)", fontsize=18)
ax2.legend(fontsize=14)
ax2.tick_params(axis="both", which="major", labelsize=16)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1)
plt.tight_layout()
_save("cost_ratios_data_vs_model.pdf")

# Figure D: Two-panel — empirical vs model log SD earnings
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
ax1.plot(emp_df.index, emp_df["log_sd_earnings_norm"], "o-", linewidth=3, markersize=10)
ax1.set_xlabel("Year", fontsize=18)
ax1.set_ylabel("Log SD Earnings (Data, Change from 1980)", fontsize=18)
ax1.tick_params(axis="both", which="major", labelsize=16)
ax1.grid(True, alpha=0.3)
ax2.plot(years, sd_earnings_vals, "o-", linewidth=3, markersize=10)
ax2.set_xlabel("Year", fontsize=18)
ax2.set_ylabel("Log SD Earnings (Model, Change from 1980)", fontsize=18)
ax2.tick_params(axis="both", which="major", labelsize=16)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
_save("sd_earnings_data_vs_model.pdf")

# Figure E: Two-panel — empirical vs model log SD sales
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
ax1.plot(emp_df.index, emp_df["log_sd_sales_norm"], "o-", linewidth=3, markersize=10)
ax1.set_xlabel("Year", fontsize=18)
ax1.set_ylabel("Log SD Sales (Data, Change from 1980)", fontsize=18)
ax1.tick_params(axis="both", which="major", labelsize=16)
ax1.grid(True, alpha=0.3)
ax2.plot(years, sd_sales_vals, "o-", linewidth=3, markersize=10)
ax2.set_xlabel("Year", fontsize=18)
ax2.set_ylabel("Log SD Sales (Model, Change from 1980)", fontsize=18)
ax2.tick_params(axis="both", which="major", labelsize=16)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
_save("sd_sales_data_vs_model.pdf")

# Figure F: Overlay — % negative EBITDA vs % negative income (model)
plt.figure(figsize=(10, 10))
plt.plot(years, pct_neg_vals,        "o-", linewidth=3, markersize=10, label="% Negative EBITDA")
plt.plot(years, pct_neg_income_vals, "s-", linewidth=3, markersize=10, label="% Negative Income (net of dep. cost)")
plt.xlabel("Year", fontsize=18)
plt.ylabel("Percent of Firms (%)", fontsize=18)
plt.legend(fontsize=16)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.grid(True, alpha=0.3)
_save("pct_neg_ebitda_vs_income_by_year.pdf")

print("\nDone.")
