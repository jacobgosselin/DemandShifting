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
import matplotlib.cm as cm
import pandas as pd
import os
from ss_solver.integrate_dist import agg_consumption

palette_2 = [cm.inferno(x) for x in np.linspace(0.0, 0.9, 2)]
palette_3 = [cm.inferno(x) for x in np.linspace(0.0, 0.9, 3)]
palette_5 = [cm.inferno(x) for x in np.linspace(0.0, 0.9, 5)]

from ss_solver.integrate_dist import (
    pct_negative, pct_negative_income, est_dist, est_sd,
    median_adv_ratio, median_inv_ratio, median_cogs_ratio,
    median_earnings, mean_earnings,
    agg_capital_stock, sales_wtd_productivity, agg_labor_shares,
    avg_firm_earnings_path, cohort_neg_path,
    avg_neg_spell_cohort,
)
from ss_solver.prod_fncts import *

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
_emp_csv = os.path.join(_DIR, "data", "empirical_trends_byyear.csv")
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

c_vals, i_vals, a_vals = [], [], []
c_phi_fixed, c_only_phi = [], []
vk_vals = []
W_vals = []
m_bnd_vals    = []
k_bnd_vals    = []
agg_k_vals    = []
sales_wtd_z_vals = []
La_vals, Lk_vals, Ls_vals = [], [], []

eqm_0 = eqms[years[0]]
dist_0 = eqm_0["Dist"]
p_0 = eqm_0["params"]
m_pol_0 = eqm_0['policies']["m_policy"]
k_pol_0 = eqm_0['policies']["k_policy"]

for yr in years:
    eqm  = eqms[yr]
    dist = eqm["Dist"]
    p = eqm["params"]

    m_bnd_vals.append(np.sum(dist[-10:, :, :]) / np.sum(dist))
    k_bnd_vals.append(np.sum(dist[:, -10:, :]) / np.sum(dist))

    # compute different c aggregates 
    c_main = eqm["c_agg"]
    c_only_phi_val = agg_consumption(m_grid, k_grid, z_grid, dist_0, p_0['sigma'], eqm_0['c_agg'], eqm_0['W'], p_0['gamma_k'], p_0['gamma_l'], p['phi'])
    c_phi_fixed_val = agg_consumption(m_grid, k_grid, z_grid, dist, p['sigma'], eqm['c_agg'], eqm['W'], p['gamma_k'], p['gamma_l'], p_0['phi'])

    c_vals.append(c_main)
    c_only_phi.append(c_only_phi_val)
    c_phi_fixed.append(c_phi_fixed_val)

    # c_vals.append(eqm["c_agg"])
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
        f"m_bnd={m_bnd_vals[-1]:.4f}  k_bnd={k_bnd_vals[-1]:.4f}"
    )

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

plt.rcParams.update({
    # 'font.family': 'serif',
    'font.size': 24,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
})

# Figure 1a: GDP proxy (C + I + A)
gdp = np.array(c_vals) + np.array(i_vals) + np.array(a_vals)
plt.figure(figsize=(10, 10))
plt.plot(years, gdp/gdp[0] - 1, "o-", linewidth=3, markersize=10, color="black")
plt.xlabel("Year", fontsize=18)
plt.ylabel("GDP Proxy (C + I + A)", fontsize=18)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.grid(True, alpha=0.3)
_save("gdp_proxy_vs_phi.pdf")

# Figure 1b: C vs. I vs. A
plt.figure(figsize=(10, 10))
plt.plot(years, c_vals/c_vals[0] - 1, "o-", linewidth=3, markersize=10, label=r"C", color=palette_3[0])
plt.plot(years, i_vals/i_vals[0] - 1, "s-", linewidth=3, markersize=10, label=r"I", color=palette_3[1])
plt.plot(years, a_vals/a_vals[0] - 1, "^-", linewidth=3, markersize=10, label=r"A", color=palette_3[2])
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
plt.plot(years, agg_k_vals, "o-", linewidth=3, markersize=10, color="black")
plt.xlabel("Year", fontsize=18)
plt.ylabel("Aggregate Capital Stock", fontsize=18)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.grid(True, alpha=0.3)
_save("agg_capital_stock_vs_phi.pdf")

# Figure 3: Sales-weighted productivity
plt.figure(figsize=(10, 10))
plt.plot(years, sales_wtd_z_vals, "o-", linewidth=3, markersize=10, color="black")
plt.xlabel("Year", fontsize=18)
plt.ylabel("Sales-Weighted Productivity", fontsize=18)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.grid(True, alpha=0.3)
_save("sales_wtd_productivity_vs_phi.pdf")

# Figure 4: Aggregate labor allocations
plt.figure(figsize=(10, 10))
plt.plot(years, La_vals, "o-", linewidth=3, markersize=10, label=r"$L_a$ (advertising)", color=palette_3[0])
plt.plot(years, Lk_vals, "s-", linewidth=3, markersize=10, label=r"$L_k$ (capital inv.)", color=palette_3[1])
plt.plot(years, Ls_vals, "^-", linewidth=3, markersize=10, label=r"$L_s$ (goods prod.)", color=palette_3[2])
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

# Figure 5: Different C aggregates (fixed)
plt.figure(figsize=(10, 10))
plt.plot(years, c_vals/c_vals[0] - 1, "o-", linewidth=3, markersize=10, label=r"Overall", color=palette_3[0])
plt.plot(years, c_only_phi/c_vals[0] - 1, "s-", linewidth=3, markersize=10, label=r"Changing $\phi$, fixing dist.", color=palette_3[1])
plt.plot(years, c_phi_fixed/c_vals[0] - 1, "^-", linewidth=3, markersize=10, label=r"Fixing $\phi$, changing dist.", color=palette_3[2])
plt.xlabel("Year", fontsize=18)
plt.ylabel("Aggregate Consumption", fontsize=18)
plt.legend(fontsize=16)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.grid(True, alpha=0.3)
_save("C_aggregates_vs_phi.pdf")
