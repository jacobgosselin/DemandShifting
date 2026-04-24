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

palette_2 = [cm.inferno(x) for x in np.linspace(0.0, 0.9, 2)]
palette_3 = [cm.inferno(x) for x in np.linspace(0.0, 0.9, 3)]

from ss_solver.integrate_dist import (
    pct_negative, pct_negative_income, est_dist, est_sd,
    median_adv_ratio, median_inv_ratio, median_cogs_ratio,
    median_earnings, mean_earnings, avg_firm_earnings_path, avg_neg_spell_cohort, percentile_from_cdf
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
    eqms = pickle.load(f)  # plain {year: eqm_dict}

years = sorted(eqms.keys())

print(f"Loaded {len(eqms)} equilibria for years {years[0]}–{years[-1]}.\n")

# Load empirical time-series moments (produced by 5b_exog_params.R)
_emp_csv = os.path.join(_DIR, "data", "empirical_trends_byyear.csv")
emp_df = pd.read_csv(_emp_csv).set_index("year")

# Normalize log sd/iqr for sales and earnings, and cv and iqr/median for earnings
_base_log_sd_earn = emp_df.loc[years[0], "log_sd_earnings"]
_base_log_sd_sale = emp_df.loc[years[0], "log_sd_sales"]
emp_df["log_sd_earnings_norm"] = emp_df["log_sd_earnings"] - _base_log_sd_earn
emp_df["log_sd_sales_norm"]    = emp_df["log_sd_sales"]    - _base_log_sd_sale
_base_log_iqr_earn = emp_df.loc[years[0], "log_iqr_earnings"]
_base_log_iqr_sale = emp_df.loc[years[0], "log_iqr_sales"]
emp_df["log_iqr_earnings_norm"] = emp_df["log_iqr_earnings"] - _base_log_iqr_earn
emp_df["log_iqr_sales_norm"]    = emp_df["log_iqr_sales"]    - _base_log_iqr_sale
_base_log_iqr_earn_med = emp_df.loc[years[0], "log_iqr_earnings_med"]
emp_df["log_iqr_earnings_med"] = emp_df["log_iqr_earnings_med"] - _base_log_iqr_earn_med
_base_log_cv = emp_df.loc[years[0], "log_cv_earnings"]
emp_df["log_cv_earnings_norm"] = emp_df["log_cv_earnings"] - _base_log_cv

# phi by year (from solved equilibrium params)
phi_vals = [eqms[yr]["params"]["phi"] for yr in years]

# -----------------------------------------------------------------------------
# Compute moments
# -----------------------------------------------------------------------------

pct_neg_vals  = []
pct_neg_income_vals = []
sd_earnings_raw, sd_sales_raw = [], []
iqr_earnings_raw, iqr_sales_raw = [], []
iqr_earnings_med_raw, cv_earnings = [], []
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
    eqm    = eqms[yr]
    dist   = eqm["Dist"]
    p      = eqm["params"]
    m_grid = eqm["m_grid"]
    k_grid = eqm["k_grid"]
    z_grid = eqm["z_grid"]

    pct_neg_vals.append(pct_negative(m_grid, k_grid, z_grid, eqm))
    pct_neg_income_vals.append(pct_negative_income(m_grid, k_grid, z_grid, eqm))

    m_bnd_vals.append(np.sum(dist[-10:, :, :]) / np.sum(dist))
    k_bnd_vals.append(np.sum(dist[:, -10:, :]) / np.sum(dist))

    _, earnings_cdf = est_dist(m_grid, k_grid, z_grid, eqm, "earnings")
    _, sales_cdf    = est_dist(m_grid, k_grid, z_grid, eqm, "revenue")

    # find p75 and p25 of earnings and sales, compute QCD, and log-normalize by base year
    earnings_p75 = percentile_from_cdf(earnings_cdf, .75)
    earnings_p25 = percentile_from_cdf(earnings_cdf, .25)
    sales_p25 = percentile_from_cdf(sales_cdf, .25)
    sales_p75 = percentile_from_cdf(sales_cdf, .75)
    iqr_earnings_raw.append(earnings_p75 - earnings_p25)
    iqr_sales_raw.append(sales_p75 - sales_p25)

    # Compute normalized iqr for earnings and sales, 
    iqr_earnings_med = (earnings_p75 - earnings_p25) / percentile_from_cdf(earnings_cdf, .5)
    iqr_earnings_med_raw.append(iqr_earnings_med)
    cv_earnings_val = est_sd(earnings_cdf) / np.sum(earnings_cdf[1] * earnings_cdf[0])  # sd / mean
    cv_earnings.append(cv_earnings_val)

    # append std. dev. of earnings and sales to raw lists (to be log-normalized later)
    sd_earnings_raw.append(est_sd(earnings_cdf))
    sd_sales_raw.append(est_sd(sales_cdf))

    # append median cost ratios for all firms and for negative-earning firms
    med_adv_all.append(median_adv_ratio(m_grid, k_grid, z_grid, eqm, negative_earnings_only=False))
    med_inv_all.append(median_inv_ratio(m_grid, k_grid, z_grid, eqm, negative_earnings_only=False))
    med_cogs_all.append(median_cogs_ratio(m_grid, k_grid, z_grid, eqm, negative_earnings_only=False))
    med_adv_neg.append(median_adv_ratio(m_grid, k_grid, z_grid, eqm, negative_earnings_only=True))
    med_inv_neg.append(median_inv_ratio(m_grid, k_grid, z_grid, eqm, negative_earnings_only=True))
    med_cogs_neg.append(median_cogs_ratio(m_grid, k_grid, z_grid, eqm, negative_earnings_only=True))

    # append mean and median earnings
    med_earn_vals.append(median_earnings(m_grid, k_grid, z_grid, eqm))
    mean_earn_vals.append(mean_earnings(m_grid, k_grid, z_grid, eqm))

    print(
        f"  {yr}: pct_neg={pct_neg_vals[-1]:.2f}%  "
        f"m_bnd={m_bnd_vals[-1]:.4f}  k_bnd={k_bnd_vals[-1]:.4f}"
    )

# Log-normalize standard deviations (log change from base year)
sd_earnings_arr = np.array(sd_earnings_raw)
sd_sales_arr    = np.array(sd_sales_raw)
sd_earnings_vals = np.log(sd_earnings_arr / sd_earnings_arr[0])
sd_sales_vals    = np.log(sd_sales_arr    / sd_sales_arr[0])

# Log-normalize iqr (log change from base year)
iqr_earnings_arr = np.array(iqr_earnings_raw)
iqr_sales_arr    = np.array(iqr_sales_raw)
iqr_earnings_norm_arr = np.array(iqr_earnings_med_raw)
iqr_earnings_vals = np.log(iqr_earnings_arr / iqr_earnings_arr[0])
iqr_sales_vals    = np.log(iqr_sales_arr    / iqr_sales_arr[0])
iqr_earnings_norm_vals = np.log(iqr_earnings_norm_arr / iqr_earnings_norm_arr[0])

# Log normalize iqr_med and cv for earnings
iqr_earnings_med_arr = np.array(iqr_earnings_med_raw)
iqr_earnings_med_vals = np.log(iqr_earnings_med_arr / iqr_earnings_med_arr[0])
cv_earnings_arr = np.array(cv_earnings)
cv_earnings_vals = np.log(cv_earnings_arr / cv_earnings_arr[0])

# Average age of negative-earning firms in the stationary distribution
print("\nComputing avg age of neg-earning firms by year (slow) ...")
avg_neg_vals = [avg_neg_spell_cohort(eqms[yr], eqms[yr]["z_grid"], eqms[yr]["Pi"], T=20) for yr in years]

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

# Figure A: Two-panel — phi by year + pct_neg by year
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
ax1.plot(years, phi_vals, "o-", linewidth=3, markersize=10, color="black")
ax1.set_xlabel("Year")
ax1.set_ylabel("")
ax1.set_title(r"Scale Elasticity of Demand")
ax1.grid(True, alpha=0.3)
ax2.plot(years, pct_neg_vals, "o-", linewidth=3, markersize=10, color="black")
ax2.set_xlabel("Year")
ax2.set_ylabel("")
ax2.set_title("Percent of Firms with EBITDA < 0")
ax2.grid(True, alpha=0.3)
plt.tight_layout()
_save("phi_and_pct_neg_by_year.pdf")

# Figure B: Two-panel — empirical vs. model avg neg spell/periods by year
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
ax1.plot(years, emp_df["avg_neg_spell"], "o-", linewidth=3, markersize=10, color="black")
ax1.set_xlabel("Year")
ax1.set_ylabel("")
ax1.set_title("Avg. Neg. Spell Length (Data)")
ax1.set_ylim(0, 5)
ax1.grid(True, alpha=0.3)
ax2.plot(years, avg_neg_vals, "o-", linewidth=3, markersize=10, color="black")
ax2.set_xlabel("Year")
ax2.set_ylabel("")
ax2.set_title("Avg. Neg. Spell Length (Model)")
ax2.set_ylim(0, 5)
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
# _save("avg_neg_periods_by_year.pdf", fig, close=True)

# Figure C: Two-panel — empirical vs model cost ratios
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
ax1.plot(years, emp_df["med_sga_sale"],  "o-", linewidth=3, markersize=8, label="SG&A/Rev",  color=palette_3[0])
ax1.plot(years, emp_df["med_cogs_sale"], "s-", linewidth=3, markersize=8, label="COGS/Rev",  color=palette_3[1])
ax1.plot(years, emp_df["med_capx_sale"], "^-", linewidth=3, markersize=8, label="CapEx/Rev", color=palette_3[2])
ax1.set_xlabel("Year")
ax1.set_ylabel("")
ax1.set_title("Median Spending Ratios (Data)")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)
ax2.plot(years, med_adv_all,  "o-", linewidth=3, markersize=8, label="Adv/Rev",  color=palette_3[0])
ax2.plot(years, med_cogs_all, "s-", linewidth=3, markersize=8, label="COGS/Rev", color=palette_3[1])
ax2.plot(years, med_inv_all,  "^-", linewidth=3, markersize=8, label="Inv/Rev",  color=palette_3[2])
ax2.set_xlabel("Year")
ax2.set_ylabel("")
ax2.set_title("Median Spending Ratios (Model)")
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1)
handles_c, labels_c = ax1.get_legend_handles_labels()
gs = fig.add_gridspec(2, 1)
ax1.set_subplotspec(gs[0])
ax2.set_subplotspec(gs[1])
fig.set_size_inches(8, 12)
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.legend(handles_c, labels_c, fontsize=14, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0))
_save("cost_ratios_data_vs_model_slide.pdf", fig, close=False)
fig.legends.clear()
gs = fig.add_gridspec(1, 2)
ax1.set_subplotspec(gs[0])
ax2.set_subplotspec(gs[1])
fig.set_size_inches(16, 8)
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.legend(handles_c, labels_c, fontsize=14, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0))
# _save("cost_ratios_data_vs_model.pdf", fig, close=True)

# Figure D: Two-panel overlay — Log QCD Sales + Log QCD Earnings (Change from 1980)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
ax1.plot(years, emp_df["log_sd_sales_norm"],    "o-", linewidth=3, markersize=10, label="Sales",    color=palette_2[0])
ax1.plot(years, emp_df["log_sd_earnings_norm"], "s-", linewidth=3, markersize=10, label="Earnings", color=palette_2[1])
ax1.set_ylim(-0.4, 1.2)
ax1.set_xlabel("Year")
ax1.set_ylabel("")
ax1.set_title("Std. Dev. (Data, Log Change)")
ax1.grid(True, alpha=0.3)
ax2.plot(years, sd_sales_vals,    "o-", linewidth=3, markersize=10, label="Sales",    color=palette_2[0])
ax2.plot(years, sd_earnings_vals, "s-", linewidth=3, markersize=10, label="Earnings", color=palette_2[1])
ax2.set_ylim(-0.4, 1.2)
ax2.set_xlabel("Year")
ax2.set_ylabel("")
ax2.set_title("Std. Dev. (Model, Log Change)")
ax2.grid(True, alpha=0.3)
handles_e1, labels_e1 = ax1.get_legend_handles_labels()
gs = fig.add_gridspec(2, 1)
ax1.set_subplotspec(gs[0])
ax2.set_subplotspec(gs[1])
fig.set_size_inches(8, 12)
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.legend(handles_e1, labels_e1, fontsize=14, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0))
_save("sd_sales_earnings_overlay_slide.pdf", fig, close=False)
fig.legends.clear()
gs = fig.add_gridspec(1, 2)
ax1.set_subplotspec(gs[0])
ax2.set_subplotspec(gs[1])
fig.set_size_inches(16, 8)
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.legend(handles_e1, labels_e1, fontsize=14, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0))
# _save("sd_sales_earnings_overlay.pdf", fig, close=True)

# Figure F: Overlay — % negative EBITDA vs % negative income (model)
plt.figure(figsize=(10, 10))
plt.plot(years, pct_neg_vals,        "o-", linewidth=3, markersize=10, label="% Negative EBITDA",                  color=palette_2[0])
plt.plot(years, pct_neg_income_vals, "s-", linewidth=3, markersize=10, label="% Negative Income (net of dep. cost)", color=palette_2[1])
plt.xlabel("Year", fontsize=18)
plt.ylabel("Percent of Firms (%)", fontsize=18)
plt.legend(fontsize=16)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.grid(True, alpha=0.3)
_save("pct_neg_ebitda_vs_income_by_year.pdf")

# combine Figure B, E1, and C into a 6X2 grid of subplots (in that order)
fig, axes = plt.subplots(3, 2, figsize=(16, 24))
fig.subplots_adjust(hspace=0.25, wspace=0.25)
axes[0, 0].plot(years, emp_df["avg_neg_spell"], "o-", linewidth=3, markersize=10, color="black")
axes[0, 0].set_xlabel("Year")
axes[0, 0].set_ylabel("")
axes[0, 0].set_title("Avg. Neg. Spell Length (Data)")
axes[0, 0].set_ylim(0, 5)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 1].plot(years, avg_neg_vals, "o-", linewidth=3, markersize=10, color="black")
axes[0, 1].set_xlabel("Year")
axes[0, 1].set_ylabel("")
axes[0, 1].set_title("Avg. Neg. Spell Length (Model)")
axes[0, 1].set_ylim(0, 5)
axes[0, 1].grid(True, alpha=0.3)

# Plot un-normalized IQR for Sales and Earnings
axes[1, 0].plot(years, emp_df["log_sd_sales_norm"],    "o-", linewidth=3, markersize=10, label="Sales",    color=palette_2[0])
axes[1, 0].plot(years, emp_df["log_sd_earnings_norm"], "s-", linewidth=3, markersize=10, label="Earnings", color=palette_2[1])
axes[1, 0].set_ylim(-0.4, 1.2)
axes[1, 0].set_xlabel("Year")
axes[1, 0].set_ylabel("")
axes[1, 0].set_title("Std. Dev. (Data, Log Change)")
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend(fontsize=16)
axes[1, 1].plot(years, sd_sales_vals,    "o-", linewidth=3, markersize=10, label="Sales",    color=palette_2[0])
axes[1, 1].plot(years, sd_earnings_vals, "s-", linewidth=3, markersize=10, label="Earnings", color=palette_2[1])
axes[1, 1].set_ylim(-0.4, 1.2)
axes[1, 1].set_xlabel("Year")
axes[1, 1].set_ylabel("")
axes[1, 1].set_title("Std. Dev. (Model, Log Change)")
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend(fontsize=16)

# Plot CV and IQR/Median for earnings
# axes[1, 0].plot(years, emp_df["log_iqr_earnings_med_norm"], "o-", linewidth=3, markersize=10, label="IQR/Median (Data)", color=palette_2[1])
# axes[1, 0].plot(years, emp_df["log_cv_earnings_norm"],    "s-", linewidth=3, markersize=10, label="CV (Data)",        color=palette_2[0])
# axes[1, 0].set_ylim(-0.1, 0.5)
# axes[1, 0].set_xlabel("Year")
# axes[1, 0].set_ylabel("")
# axes[1, 0].set_title("Earnings Dispersion (Data, Log Change)")
# axes[1, 0].grid(True, alpha=0.3)
# axes[1, 0].legend(fontsize=16)
# axes[1, 1].plot(years, iqr_earnings_med_vals, "o-", linewidth=3, markersize=10, label="IQR/Median (Model)", color=palette_2[1])
# axes[1, 1].plot(years, cv_earnings_vals,    "s-", linewidth=3, markersize=10, label="CV (Model)",        color=palette_2[0])
# axes[1, 1].set_ylim(-0.1, 0.5)
# axes[1, 1].set_xlabel("Year")
# axes[1, 1].set_ylabel("")
# axes[1, 1].set_title("Earnings Dispersion (Model, Log Change)")
# axes[1, 1].grid(True, alpha=0.3)
# axes[1, 1].legend(fontsize=16)


axes[2, 0].plot(years, emp_df["med_sga_sale"],  "o-", linewidth=3, markersize=8, label="SG&A/Rev",  color=palette_3[0])
axes[2, 0].plot(years, emp_df["med_cogs_sale"], "s-", linewidth=3, markersize=8, label="COGS/Rev",  color=palette_3[1])
axes[2, 0].plot(years, emp_df["med_capx_sale"], "^-", linewidth=3, markersize=8, label="CapEx/Rev", color=palette_3[2])
axes[2, 0].set_xlabel("Year")
axes[2, 0].set_ylabel("")
axes[2, 0].set_title("Median Spending Ratios (Data)")
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].set_ylim(0, 1)
axes[2, 0].legend(fontsize=16)
axes[2, 1].plot(years, med_adv_all,  "o-", linewidth=3, markersize=8, label="Adv/Rev",  color=palette_3[0])
axes[2, 1].plot(years, med_cogs_all, "s-", linewidth=3, markersize=8, label="COGS/Rev", color=palette_3[1])
axes[2, 1].plot(years, med_inv_all,  "^-", linewidth=3, markersize=8, label="Inv/Rev",  color=palette_3[2])
axes[2, 1].set_xlabel("Year")
axes[2, 1].set_ylabel("")   
axes[2, 1].set_title("Median Spending Ratios (Model)")
axes[2, 1].grid(True, alpha=0.3)
axes[2, 1].set_ylim(0, 1)
axes[2, 1].legend(fontsize=16)
_save("combined_trends_paper.pdf", fig, close=True)

# Figure G: Avg firm earning path for 1980 vs 2019 cohort (model)
earning_path_1980 = avg_firm_earnings_path(eqms[1980], eqms[1980]["z_grid"], T=50)
earning_path_2019 = avg_firm_earnings_path(eqms[2019], eqms[2019]["z_grid"], T=50)
plt.figure(figsize=(8, 6))
plt.plot(range(50), earning_path_1980, "o-", linewidth=3, markersize=5, label="1980 Cohort", color=palette_2[0])
plt.plot(range(50), earning_path_2019, "s-", linewidth=3, markersize=5, label="2019 Cohort", color=palette_2[1])
plt.xlabel("Years Since Entry")
plt.ylabel("Earnings")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
_save("avg_earnings_path_1980_vs_2019.pdf")


# -----------------------------------------------------------------------------
# Paper Stats
# -----------------------------------------------------------------------------

PAPER_STATS_PATH = os.path.join(MAIN_DIR, "paper", "paper_stats.csv")
paper_stats = pd.read_csv(PAPER_STATS_PATH)

stats_model = {
    "neg_spell_1980_model":         round(avg_neg_vals[years.index(1980)], 2),
    "neg_spell_2019_model":         round(avg_neg_vals[years.index(2019)], 2),
    "sd_earnings_percchange_model": round((sd_earnings_arr[-1] - sd_earnings_arr[0]) / abs(sd_earnings_arr[0]) * 100, 2),
    "sd_sales_percchange_model":    round((sd_sales_arr[-1] - sd_sales_arr[0]) / abs(sd_sales_arr[0]) * 100, 2),
    "adv_ratio_1980_model":         round(med_adv_all[years.index(1980)], 2),
    "adv_ratio_2019_model":         round(med_adv_all[years.index(2019)], 2),
    "inv_ratio_change_model":       round((med_inv_all[-1] - med_inv_all[0]) / abs(med_inv_all[0]) * 100, 2),
}

for key, val in stats_model.items():
    paper_stats.loc[paper_stats["key"] == key, "value"] = val

paper_stats.to_csv(PAPER_STATS_PATH, index=False)
print(f"\nPaper stats written to {PAPER_STATS_PATH}")
