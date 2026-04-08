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
    avg_firm_earnings_path, cohort_neg_path,
    avg_neg_spell_cohort,
)
from prod_fncts import *

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
c_vals, i_vals, a_vals        = [], [], []
vk_vals = []
W_vals = []
m_bnd_vals    = []
k_bnd_vals    = []
agg_k_vals    = []
sales_wtd_z_vals = []
La_vals, Lk_vals, Ls_vals = [], [], []

for yr in years:
    eqm  = eqms[yr]
    dist = eqm["Dist"]
    p = eqm["params"]

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
    W = eqm["W"]
    W_vals.append(W)
    vk = W * L_k(agg_capital_stock(m_grid, k_grid, z_grid, eqm), p['alpha_k'])
    vk_vals.append(vk)
    i_vals.append(W * Lk); a_vals.append(W * La)
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
avg_neg_vals = [avg_neg_spell_cohort(eqms[yr], z_grid, Pi, T=20) for yr in years]

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def _save(fname, fig=None, close=True):
    path = os.path.join(FIGURES_DIR, fname)
    if fig is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        if close:
            plt.close(fig)
    else:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        if close:
            plt.close()
    print(f"  Saved {path}")


print(f"\nGenerating figures → {FIGURES_DIR}/")


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
ax1.plot(years, emp_df["avg_neg_spell"], "o-", linewidth=3, markersize=10)
ax1.set_xlabel("Year", fontsize=18)
ax1.set_ylabel("Avg. Neg. Earnings Spell Length (Data)", fontsize=18)
ax1.tick_params(axis="both", which="major", labelsize=16)
ax1.set_ylim(0, 5)
ax1.grid(True, alpha=0.3)
ax2.plot(years, avg_neg_vals, "o-", linewidth=3, markersize=10)
ax2.set_xlabel("Year", fontsize=18)
ax2.set_ylabel("Avg. Neg. Earnings Spell Length (Model)", fontsize=18)
ax2.set_ylim(0, 5)
ax2.tick_params(axis="both", which="major", labelsize=16)
ax2.grid(True, alpha=0.3)
gs = fig.add_gridspec(2, 1)
ax1.set_subplotspec(gs[0])
ax2.set_subplotspec(gs[1])
fig.set_size_inches(8, 12)
fig.tight_layout()
_save("avg_neg_periods_by_year_slide.pdf", fig, close=False)
gs = fig.add_gridspec(1, 2)
ax1.set_subplotspec(gs[0])
ax2.set_subplotspec(gs[1])
fig.set_size_inches(16, 8)
fig.tight_layout()
_save("avg_neg_periods_by_year.pdf", fig, close=True)

# Figure C: Two-panel — empirical vs model cost ratios
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
ax1.plot(years, emp_df["med_sga_sale"],  "o-", linewidth=3, markersize=8, label="SG&A/Rev")
ax1.plot(years, emp_df["med_cogs_sale"], "s-", linewidth=3, markersize=8, label="COGS/Rev")
ax1.plot(years, emp_df["med_capx_sale"], "^-", linewidth=3, markersize=8, label="CapEx/Rev")
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
gs = fig.add_gridspec(2, 1)
ax1.set_subplotspec(gs[0])
ax2.set_subplotspec(gs[1])
fig.set_size_inches(8, 12)
fig.tight_layout()
_save("cost_ratios_data_vs_model_slide.pdf", fig, close=False)
gs = fig.add_gridspec(1, 2)
ax1.set_subplotspec(gs[0])
ax2.set_subplotspec(gs[1])
fig.set_size_inches(16, 8)
fig.tight_layout()
_save("cost_ratios_data_vs_model.pdf", fig, close=True)

# Figure D: Two-panel — empirical vs model log SD earnings
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
ax1.plot(years, emp_df["log_sd_earnings_norm"], "o-", linewidth=3, markersize=10)
ax1.set_xlabel("Year", fontsize=18)
ax1.set_ylabel("Log SD Earnings (Data, Change from 1980)", fontsize=18)
ax1.tick_params(axis="both", which="major", labelsize=16)
ax1.grid(True, alpha=0.3)
ax2.plot(years, sd_earnings_vals, "o-", linewidth=3, markersize=10)
ax2.set_xlabel("Year", fontsize=18)
ax2.set_ylabel("Log SD Earnings (Model, Change from 1980)", fontsize=18)
ax2.tick_params(axis="both", which="major", labelsize=16)
ax2.grid(True, alpha=0.3)
gs = fig.add_gridspec(2, 1)
ax1.set_subplotspec(gs[0])
ax2.set_subplotspec(gs[1])
fig.set_size_inches(8, 12)
fig.tight_layout()
_save("sd_earnings_data_vs_model_slide.pdf", fig, close=False)
gs = fig.add_gridspec(1, 2)
ax1.set_subplotspec(gs[0])
ax2.set_subplotspec(gs[1])
fig.set_size_inches(16, 9)
fig.tight_layout()
_save("sd_earnings_data_vs_model.pdf", fig, close=True)

# Figure E: Two-panel — empirical vs model log SD sales
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
ax1.plot(years, emp_df["log_sd_sales_norm"], "o-", linewidth=3, markersize=10)
ax1.set_xlabel("Year", fontsize=18)
ax1.set_ylabel("Log SD Sales (Data, Change from 1980)", fontsize=18)
ax1.tick_params(axis="both", which="major", labelsize=16)
ax1.grid(True, alpha=0.3)
ax2.plot(years, sd_sales_vals, "o-", linewidth=3, markersize=10)
ax2.set_xlabel("Year", fontsize=18)
ax2.set_ylabel("Log SD Sales (Model, Change from 1980)", fontsize=18)
ax2.tick_params(axis="both", which="major", labelsize=16)
ax2.grid(True, alpha=0.3)
gs = fig.add_gridspec(2, 1)
ax1.set_subplotspec(gs[0])
ax2.set_subplotspec(gs[1])
fig.set_size_inches(8, 12)
fig.tight_layout()
_save("sd_sales_data_vs_model_slide.pdf", fig, close=False)
gs = fig.add_gridspec(1, 2)
ax1.set_subplotspec(gs[0])
ax2.set_subplotspec(gs[1])
fig.set_size_inches(16, 8)
fig.tight_layout()
_save("sd_sales_data_vs_model.pdf", fig, close=True)

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

####
# Aggregate Variable Paths 
####

# Figure 1a: GDP proxy (C + I + A)
gdp = np.array(c_vals) + np.array(i_vals) + np.array(a_vals)
plt.figure(figsize=(10, 10))
plt.plot(years, gdp/gdp[0] - 1, "o-", linewidth=3, markersize=10)
plt.xlabel("Year", fontsize=18)
plt.ylabel("GDP Proxy (C + I + A)", fontsize=18)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.grid(True, alpha=0.3)
_save("gdp_proxy_vs_phi.pdf")

# Figure 1b: C vs. I vs. A
plt.figure(figsize=(10, 10))
plt.plot(years, c_vals/c_vals[0] - 1, "o-", linewidth=3, markersize=10, label=r"$\int_0^1 P_i C_i di$ (C)")
plt.plot(years, i_vals/i_vals[0] - 1, "s-", linewidth=3, markersize=10, label=r"$\int_0^1 W L_i^k di$ (I)")
plt.plot(years, a_vals/a_vals[0] - 1, "^-", linewidth=3, markersize=10, label=r"$\int_0^1 W L_i^a di$ (A)")
plt.xlabel("Year", fontsize=18)
plt.ylabel("", fontsize=18)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=16)
_save("C_I_A_vs_phi.pdf")

# Figure 1c: Recover TFP
cost_shares_k = np.array(vk_vals) / np.array(c_vals)
cost_shares_k_avg = (cost_shares_k[:-1] + cost_shares_k[1:]) / 2
cost_shares_l = np.array(Ls_vals) * np.array(W_vals) / np.array(c_vals)
cost_shares_l_avg = (cost_shares_l[:-1] + cost_shares_l[1:]) / 2
agg_k_arr = np.array(agg_k_vals)
agg_k_pct_change = np.diff(agg_k_arr) / agg_k_arr[:-1]
Ls_arr = np.array(Ls_vals)
Ls_pct_change = np.diff(Ls_arr) / Ls_arr[:-1]
c_arr = np.array(c_vals)
c_pct_change = np.diff(c_arr) / c_arr[:-1]
tfp_annual_growth = c_pct_change - cost_shares_k_avg * agg_k_pct_change - cost_shares_l_avg * Ls_pct_change
plt.figure(figsize=(10, 10))
# do a bar plot with bars colored by positive vs negative TFP growth
bar_colors = ["green" if x >= 0 else "red" for x in tfp_annual_growth]
plt.bar(years[1:], tfp_annual_growth, color=bar_colors, alpha=0.7)
plt.xlabel("Year", fontsize=18)
plt.ylabel("TFP Effect of Scale Elasticity", fontsize=18)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.grid(True, alpha=0.3)
_save("tfp_vs_phi.pdf")

# Figure 2: Aggregate capital stock
plt.figure(figsize=(10, 10))
plt.plot(years, agg_k_vals, "o-", linewidth=3, markersize=10)
plt.xlabel("Year", fontsize=18)
plt.ylabel("Aggregate Capital Stock", fontsize=18)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.grid(True, alpha=0.3)
_save("agg_capital_stock_vs_phi.pdf")

# Figure 3: Sales-weighted productivity
plt.figure(figsize=(10, 10))
plt.plot(years, sales_wtd_z_vals, "o-", linewidth=3, markersize=10)
plt.xlabel("Year", fontsize=18)
plt.ylabel("Sales-Weighted Productivity", fontsize=18)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.grid(True, alpha=0.3)
_save("sales_wtd_productivity_vs_phi.pdf")

# Figure 4: Aggregate labor allocations
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

# Figure 5: Average firm earnings path — phi_0 vs phi_T
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
