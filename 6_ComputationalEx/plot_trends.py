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
import os

from integrate_dist import (
    pct_negative, est_dist, est_sd,
    median_adv_ratio, median_inv_ratio, median_cogs_ratio,
    median_earnings, mean_earnings,
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
    "My Drive/research_ideas/negative_earnings/figures",
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

print(f"Loaded {len(eqms)} equilibria for years {years[0]}–{years[-1]}.\n")

# -----------------------------------------------------------------------------
# Compute moments
# -----------------------------------------------------------------------------

pct_neg_vals  = []
sd_earnings_raw, sd_sales_raw = [], []
med_adv_all, med_inv_all, med_cogs_all = [], [], []
med_adv_neg, med_inv_neg, med_cogs_neg = [], [], []
med_earn_vals, mean_earn_vals = [], []
c_vals        = []
m_bnd_vals    = []
k_bnd_vals    = []

for yr in years:
    eqm  = eqms[yr]
    dist = eqm["Dist"]

    pct_neg_vals.append(pct_negative(m_grid, k_grid, z_grid, eqm))

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
    print(
        f"  {yr}: pct_neg={pct_neg_vals[-1]:.2f}%  "
        f"m_bnd={m_bnd_vals[-1]:.4f}  k_bnd={k_bnd_vals[-1]:.4f}"
    )

# Log-normalize standard deviations (log change from base year)
sd_earnings_arr = np.array(sd_earnings_raw)
sd_sales_arr    = np.array(sd_sales_raw)
sd_earnings_vals = np.log(sd_earnings_arr / sd_earnings_arr[0])
sd_sales_vals    = np.log(sd_sales_arr    / sd_sales_arr[0])

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
_save("pct_negative_vs_phi_arb_scale.pdf")

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
_save("median_ratios_all_vs_phi_arb_scale.pdf")

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
_save("median_ratios_neg_vs_phi_arb_scale.pdf")

# Figure 4: Std. dev. of earnings (log change from base year)
plt.figure(figsize=(10, 10))
plt.plot(years, sd_earnings_vals, "o-", linewidth=3, markersize=10)
plt.xlabel("Year", fontsize=18)
plt.ylabel("Std. Deviation of Earnings (Log Change from 1980)", fontsize=18)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.grid(True, alpha=0.3)
_save("sd_earnings_vs_phi_arb_scale.pdf")

# Figure 5: Std. dev. of sales (log change from base year)
plt.figure(figsize=(10, 10))
plt.plot(years, sd_sales_vals, "o-", linewidth=3, markersize=10)
plt.xlabel("Year", fontsize=18)
plt.ylabel("Std. Deviation of Sales (Log Change from 1980)", fontsize=18)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.grid(True, alpha=0.3)
_save("sd_sales_vs_phi_arb_scale.pdf")

# Figure 6: Aggregate consumption
plt.figure(figsize=(10, 10))
plt.plot(years, c_vals, "o-", linewidth=3, markersize=10)
plt.xlabel("Year", fontsize=18)
plt.ylabel("Aggregate Consumption C", fontsize=18)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.grid(True, alpha=0.3)
_save("aggregate_consumption_vs_phi_arb_scale.pdf")

# Figure 7: Median and mean earnings (scaled by aggregate revenue)
plt.figure(figsize=(10, 10))
plt.plot(years, med_earn_vals, "o-", linewidth=3, markersize=10, label="Median Earnings")
plt.plot(years, mean_earn_vals, "s-", linewidth=3, markersize=10, label="Mean Earnings")
plt.xlabel("Year", fontsize=18)
plt.ylabel("Earnings", fontsize=18)
plt.legend(fontsize=16)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.grid(True, alpha=0.3)
_save("earnings_median_mean_vs_phi_arb_scale.pdf")

print("\nDone.")
