"""
plot_trends_alt.py
------------------
Reproduce all trend figures from plot_trends.py for three alternative
parameter scans: sigma, sigma_eps, and beta.

For each scan, every figure is saved under FIGURES_DIR/alt_eqm_figs/ with
the parameter name appended to the filename, e.g.:
  phi_and_pct_neg_by_year_sigma.pdf
  avg_neg_periods_by_year_slide_beta.pdf

Each figure shows all equilibria in the scan along the x-axis (parameter
values), so there is one figure per scan per figure type.

Data panels (empirical time-series) are kept as-is on the year axis so the
reader can compare whether the parameter sweep spans the empirical range.
Model panels use the scanned parameter values on the x-axis.

Reads:
  SOLVED_EQM_DIR/eqm_sigma_path.pkl
  SOLVED_EQM_DIR/eqm_sigma_eps_path.pkl
  SOLVED_EQM_DIR/eqm_beta_path.pkl
  6_ComputationalEx/data/empirical_trends_byyear.csv

Writes:
  FIGURES_DIR/alt_eqm_figs/*.pdf

Set DEMANDSHIFT_FIGURES_DIR env var to override the default figures path.

Usage:
  python plot_trends_alt.py
"""

import pickle
import configparser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import os

palette_2 = [cm.inferno(x) for x in np.linspace(0.0, 0.9, 2)]
palette_3 = [cm.inferno(x) for x in np.linspace(0.0, 0.9, 3)]

from ss_solver.integrate_dist import (
    pct_negative, est_dist, est_sd, median_adv_ratio,
    median_inv_ratio, median_cogs_ratio, avg_neg_spell_cohort
)
from ss_solver.prod_fncts import *
from ss_solver.solve_eqm import EqmParams, solve_ss_equilibrium_least_squares
from ss_solver.solve_vf import discretize_productivity, discretize_choices

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

MAIN_DIR = "/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings"
SOLVED_EQM_DIR = os.path.join(MAIN_DIR, "data", "clean")

_default_figures = os.path.join(
    os.path.expanduser("~"),
    "Library/CloudStorage/"
    "GoogleDrive-jacob.gosselin@u.northwestern.edu/"
    "My Drive/research_ideas/negative_earnings/figures/quantitative",
)
FIGURES_DIR = os.environ.get("DEMANDSHIFT_FIGURES_DIR", _default_figures)

# -----------------------------------------------------------------------------
# Scan configurations
# -----------------------------------------------------------------------------
# "attr" is the EqmParams attribute name for the varied parameter, or None if
# the param value is only accessible as the pkl dict key (sigma_eps).
# "label" is the matplotlib x-axis label for model panels.

SCANS = [
    {
        "name":  "phi",
        "pkl":   "eqm_phi_alt_path.pkl",
        "attr":  "phi",
        "label": r"$\phi$",
    },
    # {
    #     "name":  "phi_kunal",
    #     "pkl":   "eqm_phi_Kunal_path.pkl",
    #     "attr":  "phi",
    #     "label": r"$\phi$ (Kunal)",
    # },
    {
        "name":  "sigma",
        "pkl":   "eqm_sigma_path.pkl",
        "attr":  "sigma",
        "label": r"$\sigma$",
    },
    {
        "name":  "beta",
        "pkl":   "eqm_beta_path.pkl",
        "attr":  "beta",
        "label": r"$\beta$",
    },
    # {
    #     "name":  "sigma_eps",
    #     "pkl":   "eqm_sigma_eps_path.pkl",
    #     "attr":  None,   # not stored in EqmParams; use dict key
    #     "label": r"$\sigma_\varepsilon$",
    # },
]

# -----------------------------------------------------------------------------
# matplotlib style
# -----------------------------------------------------------------------------

plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
})

# -----------------------------------------------------------------------------
# Save helper
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Main loop over scans
# -----------------------------------------------------------------------------

n_scans = len(SCANS)
fig, axes = plt.subplots(n_scans, 4, figsize=(16, 12))
if n_scans == 1:
    axes = axes[np.newaxis, :]

for i, scan in enumerate(SCANS):
    pname  = scan["name"]
    plabel = scan["label"]
    pkl_path = os.path.join(SOLVED_EQM_DIR, scan["pkl"])

    print(f"\n{'='*60}")
    print(f"Scan: {pname}  ({pkl_path})")
    print(f"{'='*60}")

    if not os.path.isfile(pkl_path):
        print(f"  [SKIP] pkl not found: {pkl_path}")
        continue

    with open(pkl_path, "rb") as f:
        eqms = pickle.load(f)

    param_vals = sorted(eqms.keys())
    print(f"  Loaded {len(param_vals)} equilibria, "
          f"range [{param_vals[0]:.4f}, {param_vals[-1]:.4f}]")

    # ------------------------------------------------------------------
    # Compute moments for each param value
    # ------------------------------------------------------------------

    p_vals = []
    pct_neg_vals      = []
    sd_earnings_raw, sd_sales_raw = [], []
    med_adv_all, med_inv_all, med_cogs_all = [], [], []
    avg_neg_vals = []

    for pv in param_vals:
        eqm    = eqms[pv]
        m_grid = eqm["m_grid"]
        k_grid = eqm["k_grid"]
        z_grid = eqm["z_grid"]

        p_vals.append(pv)
        pct_neg_vals.append(pct_negative(m_grid, k_grid, z_grid, eqm))
        _, earnings_cdf = est_dist(m_grid, k_grid, z_grid, eqm, "earnings")
        _, sales_cdf    = est_dist(m_grid, k_grid, z_grid, eqm, "revenue")
        sd_earnings_raw.append(est_sd(earnings_cdf))
        sd_sales_raw.append(est_sd(sales_cdf))

        med_adv_all.append(median_adv_ratio(m_grid, k_grid, z_grid, eqm, negative_earnings_only=False))
        med_inv_all.append(median_inv_ratio(m_grid, k_grid, z_grid, eqm, negative_earnings_only=False))
        med_cogs_all.append(median_cogs_ratio(m_grid, k_grid, z_grid, eqm, negative_earnings_only=False))

        print(f"  {pname}={pv:.4f}: pct_neg={pct_neg_vals[-1]:.2f}%")

    # Log-normalize model SDs from first param value
    sd_earnings_arr  = np.array(sd_earnings_raw)
    sd_sales_arr     = np.array(sd_sales_raw)
    sd_earnings_vals = np.log(sd_earnings_arr / sd_earnings_arr[0])
    sd_sales_vals    = np.log(sd_sales_arr    / sd_sales_arr[0])

    # Avg neg spell (slow)
    print(f"\n  Computing avg neg spell for {pname} scan ...")
    for pv in param_vals:
        eqm = eqms[pv]
        avg_neg_vals.append(
            avg_neg_spell_cohort(eqm, eqm["z_grid"], eqm["Pi"], T=20)
        )

    # ------------------------------------------------------------------
    # Figures — single-axis model-only plots
    # ------------------------------------------------------------------
    
    ax_a = axes[i, 0]
    ax_b = axes[i, 1]
    ax_c = axes[i, 2]
    ax_d = axes[i, 3]

    # Panel A: pct_neg vs param
    ax_a.plot(param_vals, pct_neg_vals, "o-", linewidth=3, markersize=10, color="black")
    ax_a.set_xlabel(plabel)
    ax_a.set_ylabel("")
    ax_a.set_title("Percent of Firms with Earnings < 0")
    ax_a.grid(True, alpha=0.3)

    # Panel B: avg neg spell vs param
    ax_b.plot(param_vals, avg_neg_vals, "o-", linewidth=3, markersize=10, color="black")
    ax_b.set_xlabel(plabel)
    ax_b.set_ylabel("")
    ax_b.set_title("Avg. Neg. Spell Length")
    # ax_b.set_ylim(0, 5)
    ax_b.grid(True, alpha=0.3)

    # Panel C: log SD sales + earnings vs param
    ax_c.plot(param_vals, sd_sales_vals,    "o-", linewidth=3, markersize=10, label="Log SD Sales",    color=palette_2[1])
    ax_c.plot(param_vals, sd_earnings_vals, "s-", linewidth=3, markersize=10, label="Log SD Earnings", color=palette_2[0])
    # ax_c.set_ylim(-0.4, 1.2)
    ax_c.set_xlabel(plabel)
    ax_c.set_ylabel("")
    ax_c.set_title("Std. Dev. (Log Change)")
    ax_c.legend(fontsize=16)
    ax_c.grid(True, alpha=0.3)

    # Panel D: cost ratios vs param
    ax_d.plot(param_vals, med_cogs_all, "o-", linewidth=3, markersize=8, label="$W L_i/P_i Y_i$", color=palette_3[0])
    ax_d.plot(param_vals, med_adv_all,  "s-", linewidth=3, markersize=8, label="$W L_{a,i}/P_i Y_i$",  color=palette_3[1])
    ax_d.plot(param_vals, med_inv_all,  "^-", linewidth=3, markersize=8, label="$W L_{k,i}/P_i Y_i$",  color=palette_3[2])
    ax_d.set_xlabel(plabel)
    ax_d.set_ylabel("")
    ax_d.set_title("Median Spending Ratios")
    ax_d.grid(True, alpha=0.3)
    ax_d.legend(fontsize=16)
    ax_d.set_ylim(0, 1)

    # ------------------------------------------------------------------
    # Per-scan 2x2 figure (10x10 inches)
    # ------------------------------------------------------------------

    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))
    ax_a2, ax_b2, ax_c2, ax_d2 = axes2[0, 0], axes2[0, 1], axes2[1, 0], axes2[1, 1]

    ax_a2.plot(param_vals, pct_neg_vals, "o-", linewidth=3, markersize=10, color="black")
    ax_a2.set_xlabel(plabel)
    ax_a2.set_ylabel("")
    ax_a2.set_title("Percent of Firms with Earnings < 0")
    ax_a2.grid(True, alpha=0.3)

    ax_b2.plot(param_vals, avg_neg_vals, "o-", linewidth=3, markersize=10, color="black")
    ax_b2.set_xlabel(plabel)
    ax_b2.set_ylabel("")
    ax_b2.set_title("Avg. Neg. Spell Length")
    ax_b2.grid(True, alpha=0.3)

    ax_c2.plot(param_vals, sd_sales_vals,    "o-", linewidth=3, markersize=10, label="Log SD Sales",    color=palette_2[1])
    ax_c2.plot(param_vals, sd_earnings_vals, "s-", linewidth=3, markersize=10, label="Log SD Earnings", color=palette_2[0])
    ax_c2.set_xlabel(plabel)
    ax_c2.set_ylabel("")
    ax_c2.set_title("Std. Dev. (Log Change)")
    ax_c2.legend(fontsize=16)
    ax_c2.grid(True, alpha=0.3)

    ax_d2.plot(param_vals, med_cogs_all, "o-", linewidth=3, markersize=8, label="$W L_i/P_i Y_i$",     color=palette_3[0])
    ax_d2.plot(param_vals, med_adv_all,  "s-", linewidth=3, markersize=8, label="$W L_{a,i}/P_i Y_i$", color=palette_3[1])
    ax_d2.plot(param_vals, med_inv_all,  "^-", linewidth=3, markersize=8, label="$W L_{k,i}/P_i Y_i$", color=palette_3[2])
    ax_d2.set_xlabel(plabel)
    ax_d2.set_ylabel("")
    ax_d2.set_title("Median Spending Ratios")
    ax_d2.grid(True, alpha=0.3)
    ax_d2.legend(fontsize=16)
    ax_d2.set_ylim(0, 1)

    fig2.tight_layout()
    out_path_2x2 = os.path.join(FIGURES_DIR, f"alt_paths_{pname}.pdf")
    fig2.savefig(out_path_2x2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved 2x2 figure → {out_path_2x2}")


fig.tight_layout()
out_path = os.path.join(FIGURES_DIR, "alt_paths_combined.pdf")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved combined figure → {out_path}")