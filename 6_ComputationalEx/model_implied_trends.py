# DEPRECATED: Use solve_phi_path.py (parallelized solve) + plot_trends.py (figures) instead.
# This script is kept as a sequential fallback / reference implementation.
# See run_phi_path.slurm for Quest submission.

from solve_eqm import *
from integrate_dist import *
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd

# Helper Functions ----------------------------------------------------------------

def get_eqm_filename(param_key, val, entry_perc, base_phi):
    """Generate filename for saved equilibrium."""
    val_str = str(val).replace('.', 'p').replace('-', 'neg')
    if param_key == 'phi':
        return f"eqm_{param_key}_{val_str}_entry{str(entry_perc).replace('.', 'p')}.pkl"
    else:
        phi_str = str(base_phi).replace('.', 'p').replace('-', 'neg')
        return f"eqm_{param_key}_{val_str}_phi{phi_str}_entry{str(entry_perc).replace('.', 'p')}.pkl"

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

MAIN_DIR = '/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings'
FIGURES_DIR = '/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings/figures'
SOLVED_EQM_DIR = '/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings/code/3_ComputationalEx_NEW/solved_eqm'

# load structural parameters
struct_params = pd.read_csv(os.path.join(os.path.dirname(__file__), "structural_parameters.csv"))
rho = struct_params['rho'].iloc[0]
sigma_eps = struct_params['sigma_xi'].iloc[0]
exit_rate = struct_params['exit_rate'].iloc[0]

# -----------------------------------------------------------------------------
# Setup grids
# -----------------------------------------------------------------------------

print(f"AR(1) parameters for productivity: rho = {rho:.4f}, sigma_eps = {sigma_eps:.4f}")
z_grid, pi, Pi = discretize_productivity(rho, sigma_eps, 10)
m_grid = discretize_choices(1e-3, 10, 100, type = "exp")
k_grid = discretize_choices(1e-3, 10, 100, type = "exp")
# Create solved_eqm directory if it doesn't exist
os.makedirs(SOLVED_EQM_DIR, exist_ok=True)

# sigma is fixed = mu/(mu-1) from ACF markup (not calibrated)
sigma_cal = struct_params["sigma"].iloc[0]

# Load calibrated params (alpha_a, alpha_k, phi) produced by calibrate_investment_params.py
calib_path = os.path.join(os.path.dirname(__file__), "calibrated_investment_params.csv")
use_calib_values = True
if os.path.exists(calib_path) & use_calib_values:
    calib_vals = np.loadtxt(calib_path, delimiter=",", skiprows=1)
    calib_vals = np.atleast_1d(calib_vals)
    alpha_a_cal, alpha_k_cal, phi_cal = (
        calib_vals[0], calib_vals[1], calib_vals[2]
    )
    z_k_cal        = float(calib_vals[3]) if len(calib_vals) > 3 else 1.0
    fixed_cost_cal = float(calib_vals[4]) if len(calib_vals) > 4 else 0.0
    print(
        f"Loaded calibrated params: alpha_a={alpha_a_cal:.4f}, "
        f"alpha_k={alpha_k_cal:.4f}, phi={phi_cal:.4f}, "
        f"sigma={sigma_cal:.4f} (fixed), z_k={z_k_cal:.4f}, fixed_cost={fixed_cost_cal:.4f}"
    )
else:
    print(
        "Warning: calibrated_investment_params.csv not found; "
        "falling back to defaults: alpha_a=0.5, alpha_k=0.5, sigma=4, z_k=1.0, fixed_cost=0.0."
    )
    alpha_a_cal, alpha_k_cal, sigma_cal, z_k_cal, fixed_cost_cal = 0.5, 0.5, 4.0, 1.0, 0.0

# gamma_k and gamma_l are fixed structural parameters (not calibrated).
# Loaded from structural_parameters.csv (computed in 3c_exog_params.R).
gamma_k_cal = struct_params["gamma_k"].iloc[0]
gamma_l_cal = struct_params["gamma_l"].iloc[0]
print(
    f"Production function parameters (fixed): "
    f"gamma_k={gamma_k_cal:.4f}, gamma_l={gamma_l_cal:.4f}, "
    f"sum={gamma_k_cal + gamma_l_cal:.4f}"
)

# -------------------------------------------------------------------------
# Tracking moments
# -------------------------------------------------------------------------
# Load phi trajectory from R output (4b_mstock_coef.R)
_coefs_df    = pd.read_csv(f'{MAIN_DIR}/data/clean/sales_elasticity_m_by_year.csv')
coefs_byyear = _coefs_df[['year', 'coef']].to_numpy()  # shape (T, 2)

# phi back-out uses the arbitrary-scale Cobb-Douglas structural mapping:
#   sales_elasticity = (1+phi) / ((1-gamma_l)*sigma + gamma_l)
#   => phi = beta * ((1-gamma_l)*sigma + gamma_l) - 1
phi_track_values = coefs_byyear[:, 1] * ((1 - gamma_l_cal) * sigma_cal + gamma_l_cal) - 1
print("\nTracking moments across phi values implied by sales elasticity estimates:")
for i, phi in enumerate(phi_track_values):
    print(f"  Year {int(coefs_byyear[i, 0])}: phi = {phi:.4f}")

load_eqm = False
# Solve/load equilibria for tracking
eqms_track = {}
last_start = None
for phi in phi_track_values:
    eqm_filename = get_eqm_filename('phi', phi, exit_rate, 0.0)
    eqm_path = os.path.join(SOLVED_EQM_DIR, eqm_filename)
    if os.path.exists(eqm_path) & load_eqm:
        with open(eqm_path, 'rb') as f:
            eqms_track[phi] = pickle.load(f)
    else:
        print(f"  Solving equilibrium for phi = {phi}")
        params = EqmParams(
            phi=phi,
            entry_perc=exit_rate,
            sigma=sigma_cal,
            alpha_a=alpha_a_cal,
            alpha_k=alpha_k_cal,
            z_k=z_k_cal,
            fixed_cost=fixed_cost_cal,
            gamma_k=gamma_k_cal,
            gamma_l=gamma_l_cal,
        )
        eqms_track[phi] = solve_ss_equilibrium_least_squares(m_grid, k_grid, z_grid, Pi, params, start=last_start, verbose=False)
        pct_neg = pct_negative(m_grid, k_grid, z_grid, eqms_track[phi])
        print(f"    Percent negative earnings: {pct_neg:.6f}%")
        with open(eqm_path, 'wb') as f:
            pickle.dump(eqms_track[phi], f)
    eqm = eqms_track[phi]
    last_start = np.array([eqm['c_agg'], eqm['W'], eqm['P_M']])

# Compute moments
pct_neg_vals = []
sd_earnings_vals = []
sd_sales_vals = []
med_adv_all, med_inv_all, med_cogs_all = [], [], []
med_adv_neg, med_inv_neg, med_cogs_neg = [], [], []
c_vals = []

m_bnd_vals = []
k_bnd_vals = []

for phi in phi_track_values:
    eqm = eqms_track[phi]
    pct_neg_vals.append(pct_negative(m_grid, k_grid, z_grid, eqm))
    dist = eqm['Dist']
    m_bnd = np.sum(dist[-10:, :, :]) / np.sum(dist)
    k_bnd = np.sum(dist[:, -10:, :]) / np.sum(dist)
    m_bnd_vals.append(m_bnd)
    k_bnd_vals.append(k_bnd)
    print(f"  phi={phi:.4f}: pct_neg={pct_neg_vals[-1]:.2f}%  m_bnd={m_bnd:.4f}  k_bnd={k_bnd:.4f}")
    _, earnings_cdf = est_dist(m_grid, k_grid, z_grid, eqm, 'earnings')
    _, sales_cdf = est_dist(m_grid, k_grid, z_grid, eqm, 'revenue')
    sd_earnings_vals.append(est_sd(earnings_cdf))
    sd_sales_vals.append(est_sd(sales_cdf))
    med_adv_all.append(median_adv_ratio(m_grid, k_grid, z_grid, eqm, negative_earnings_only=False))
    med_inv_all.append(median_inv_ratio(m_grid, k_grid, z_grid, eqm, negative_earnings_only=False))
    med_cogs_all.append(median_cogs_ratio(m_grid, k_grid, z_grid, eqm, negative_earnings_only=False))
    med_adv_neg.append(median_adv_ratio(m_grid, k_grid, z_grid, eqm, negative_earnings_only=True))
    med_inv_neg.append(median_inv_ratio(m_grid, k_grid, z_grid, eqm, negative_earnings_only=True))
    med_cogs_neg.append(median_cogs_ratio(m_grid, k_grid, z_grid, eqm, negative_earnings_only=True))
    c_vals.append(eqm['c_agg'])

# Plot (1): Percent negative earnings vs year
years = coefs_byyear[:, 0]
plt.figure(figsize=(10, 10))
plt.plot(years, pct_neg_vals, 'o-', linewidth=3, markersize=10)
plt.xlabel('Year', fontsize=18)
plt.ylabel('Percent of Firms with Negative Earnings', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.grid(True, alpha=0.3)
plt.savefig(f'{FIGURES_DIR}/pct_negative_vs_phi_arb_scale.pdf', dpi=150, bbox_inches='tight')
plt.close()

# Plot (2): Median ratios (all firms) vs year
plt.figure(figsize=(10, 10))
plt.plot(years, med_adv_all, 'o-', linewidth=3, markersize=10, label='Adv/Revenue')
plt.plot(years, med_inv_all, 's-', linewidth=3, markersize=10, label='Inv/Revenue')
plt.plot(years, med_cogs_all, '^-', linewidth=3, markersize=10, label='COGS/Revenue')
plt.xlabel('Year', fontsize=18)
plt.ylabel('Median Ratio', fontsize=18)
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.grid(True, alpha=0.3)
plt.savefig(f'{FIGURES_DIR}/median_ratios_all_vs_phi_arb_scale.pdf', dpi=150, bbox_inches='tight')
plt.close()

# Plot (3): Median ratios (negative earnings firms) vs year
plt.figure(figsize=(10, 10))
plt.plot(years, med_adv_neg, 'o-', linewidth=3, markersize=10, label='Adv/Revenue')
plt.plot(years, med_inv_neg, 's-', linewidth=3, markersize=10, label='Inv/Revenue')
plt.plot(years, med_cogs_neg, '^-', linewidth=3, markersize=10, label='COGS/Revenue')
plt.xlabel('Year', fontsize=18)
plt.ylabel('Median Ratio (Negative Earnings Firms)', fontsize=18)
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.grid(True, alpha=0.3)
plt.savefig(f'{FIGURES_DIR}/median_ratios_neg_vs_phi_arb_scale.pdf', dpi=150, bbox_inches='tight')
plt.close()

# Plot (4): Standard deviation of earnings vs year
sd_earning_vals = np.array(sd_earnings_vals)
sd_earnings_vals = np.log(sd_earning_vals / sd_earning_vals[0])
plt.figure(figsize=(10, 10))
plt.plot(years, sd_earnings_vals, 'o-', linewidth=3, markersize=10)
plt.xlabel('Year', fontsize=18)
plt.ylabel('Std. Deviation of Earnings (Log Change from 1980)', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.grid(True, alpha=0.3)
plt.savefig(f'{FIGURES_DIR}/sd_earnings_vs_phi_arb_scale.pdf', dpi=150, bbox_inches='tight')
plt.close()

# Plot (5): Standard deviation of sales vs year
sd_sales_vals = np.array(sd_sales_vals)
sd_sales_vals = np.log(sd_sales_vals / sd_sales_vals[0])
plt.figure(figsize=(10, 10))
plt.plot(years, sd_sales_vals, 'o-', linewidth=3, markersize=10)
plt.xlabel('Year', fontsize=18)
plt.ylabel('Std. Deviation of Sales (Log Change from 1980)', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.grid(True, alpha=0.3)
plt.savefig(f'{FIGURES_DIR}/sd_sales_vs_phi_arb_scale.pdf', dpi=150, bbox_inches='tight')
plt.close()

# Plot (6): Aggregate consumption vs year
plt.figure(figsize=(10, 10))
plt.plot(years, c_vals, 'o-', linewidth=3, markersize=10)
plt.xlabel('Year', fontsize=18)
plt.ylabel('Aggregate Consumption C', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.grid(True, alpha=0.3)
plt.savefig(f'{FIGURES_DIR}/aggregate_consumption_vs_phi_arb_scale.pdf', dpi=150, bbox_inches='tight')
plt.close()

# save tracking moments to csv
tracking_df = pd.DataFrame({
    'year': years,
    'phi': phi_track_values,
    'pct_negative': pct_neg_vals,
    'sd_earnings': sd_earnings_vals,
    'sd_sales': sd_sales_vals,
    'med_adv_all': med_adv_all,
    'med_inv_all': med_inv_all,
    'med_cogs_all': med_cogs_all,
    'med_adv_neg': med_adv_neg,
    'med_inv_neg': med_inv_neg,
    'med_cogs_neg': med_cogs_neg,
    'c_agg': c_vals,
    'm_bnd_mass': m_bnd_vals,
    'k_bnd_mass': k_bnd_vals,
})
tracking_df.to_csv(f'{MAIN_DIR}/data/clean/model_implied_trends_arb_scale.csv', index=False)
print(f"\nTracking moments figures saved to {FIGURES_DIR}/ (suffix: _arb_scale)")
