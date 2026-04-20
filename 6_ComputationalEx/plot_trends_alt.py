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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

palette_2 = [cm.inferno(x) for x in np.linspace(0.0, 0.9, 2)]
palette_3 = [cm.inferno(x) for x in np.linspace(0.0, 0.9, 3)]

from ss_solver.integrate_dist import (
    pct_negative, pct_negative_income, est_dist, est_sd,
    median_adv_ratio, median_inv_ratio, median_cogs_ratio,
    median_earnings, mean_earnings, avg_neg_spell_cohort
)
from ss_solver.prod_fncts import *

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
ALT_FIGURES_DIR = os.path.join(FIGURES_DIR, "alt_eqm_figs")
os.makedirs(ALT_FIGURES_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Scan configurations
# -----------------------------------------------------------------------------
# "attr" is the EqmParams attribute name for the varied parameter, or None if
# the param value is only accessible as the pkl dict key (sigma_eps).
# "label" is the matplotlib x-axis label for model panels.

SCANS = [
    {
        "name":  "sigma",
        "pkl":   "eqm_sigma_path.pkl",
        "attr":  "sigma",
        "label": r"$\sigma$",
    },
    # {
    #     "name":  "sigma_eps",
    #     "pkl":   "eqm_sigma_eps_path.pkl",
    #     "attr":  None,   # not stored in EqmParams; use dict key
    #     "label": r"$\sigma_\varepsilon$",
    # },
    {
        "name":  "beta",
        "pkl":   "eqm_beta_path.pkl",
        "attr":  "beta",
        "label": r"$\beta$",
    },
]

# -----------------------------------------------------------------------------
# matplotlib style
# -----------------------------------------------------------------------------

plt.rcParams.update({
    'font.size': 24,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
})

# -----------------------------------------------------------------------------
# Save helper
# -----------------------------------------------------------------------------

def _save_alt(base_fname, param_name, fig=None, close=True):
    stem, ext = os.path.splitext(base_fname)
    fname = f"{stem}_{param_name}{ext}"
    path = os.path.join(ALT_FIGURES_DIR, fname)
    if fig is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        if close:
            plt.close(fig)
    else:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        if close:
            plt.close()
    print(f"  Saved {path}")

# -----------------------------------------------------------------------------
# Main loop over scans
# -----------------------------------------------------------------------------

for scan in SCANS:
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
    pct_neg_income_vals = []
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
        pct_neg_income_vals.append(pct_negative_income(m_grid, k_grid, z_grid, eqm))

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

    print(f"\n  Generating figures → {ALT_FIGURES_DIR}/")

    
    fig, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(2, 2, figsize=(20, 16))

    # Figure A: pct_neg vs param
    ax_a.plot(param_vals, pct_neg_vals, "o-", linewidth=3, markersize=10, color="black")
    ax_a.set_xlabel(plabel)
    ax_a.set_ylabel("")
    ax_a.set_title("Percent of Firms with EBITDA < 0")
    ax_a.grid(True, alpha=0.3)
   
    # Figure B: model avg neg spell vs param
    ax_b.plot(param_vals, avg_neg_vals, "o-", linewidth=3, markersize=10, color="black")
    ax_b.set_xlabel(plabel)
    ax_b.set_ylabel("")
    ax_b.set_title("Avg. Neg. Spell Length")
    ax_b.set_ylim(0, 5)
    ax_b.grid(True, alpha=0.3)

    # Figure C: model cost ratios vs param
    ax_c.plot(param_vals, med_adv_all,  "o-", linewidth=3, markersize=8, label="Adv/Rev",  color=palette_3[0])
    ax_c.plot(param_vals, med_cogs_all, "s-", linewidth=3, markersize=8, label="COGS/Rev", color=palette_3[1])
    ax_c.plot(param_vals, med_inv_all,  "^-", linewidth=3, markersize=8, label="Inv/Rev",  color=palette_3[2])
    ax_c.set_xlabel(plabel)
    ax_c.set_ylabel("")
    ax_c.set_title("Median Spending Ratios")
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(fontsize=16)
    ax_c.set_ylim(0, 1)

    # Figure D: overlay log SD sales + earnings — model only
    ax_d.plot(param_vals, sd_sales_vals,    "o-", linewidth=3, markersize=10, label="Log SD Sales",    color=palette_2[0])
    ax_d.plot(param_vals, sd_earnings_vals, "s-", linewidth=3, markersize=10, label="Log SD Earnings", color=palette_2[1])
    ax_d.set_ylim(-0.4, 1.2)
    ax_d.set_xlabel(plabel)
    ax_d.set_ylabel("")
    ax_d.set_title("Std. Dev. (Log Change)")
    ax_d.legend(fontsize=16)
    ax_d.grid(True, alpha=0.3)

    # 2X2 grid of A B C D
    fig.tight_layout()
    _save_alt("all_trends_by_param.pdf", pname, fig)

    # Figure F: pct_neg EBITDA vs pct_neg income — model, vs param
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(param_vals, pct_neg_vals,
            "o-", linewidth=3, markersize=10, label="% Negative EBITDA",                   color=palette_2[0])
    ax.plot(param_vals, pct_neg_income_vals,
            "s-", linewidth=3, markersize=10, label="% Negative Income (net of dep. cost)", color=palette_2[1])
    ax.set_xlabel(plabel)
    ax.set_ylabel("Percent of Firms (%)")
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_alt("pct_neg_ebitda_vs_income_by_param.pdf", pname, fig)

    print(f"\n  Done with {pname} scan.")

print(f"\nAll done. Figures saved to {ALT_FIGURES_DIR}/")
