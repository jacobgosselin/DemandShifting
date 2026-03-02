from solve_eqm import EqmParams, solve_ss_equilibrium_least_squares
from integrate_dist import (
    median_adv_ratio,
    median_inv_ratio,
    pct_negative,
)
from compute_coefs_byyear_np import compute_coefs_byyear_numpy
import numpy as np
import pandas as pd
import os
from scipy.optimize import least_squares, differential_evolution, minimize


# -----------------------------------------------------------------------------
# Paths and data loading (mirror model_implied_trends.py)
# -----------------------------------------------------------------------------

MAIN_DIR = "/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings"

struct_params = pd.read_csv(f"{MAIN_DIR}/data/clean/structural_parameters.csv")
rho = struct_params["rho"].iloc[0]
sigma_eps = struct_params["sigma_xi"].iloc[0]
exit_rate = struct_params["exit_rate"].iloc[0]
med_sales_init = struct_params["med_sales"].iloc[0]
med_capx_init = struct_params["med_capx_sale"].iloc[0]
neg_ebitda_base_pct  = struct_params["neg_ebitda_base"].iloc[0] * 100
neg_ebitda_final_pct = struct_params["neg_ebitda_final"].iloc[0] * 100

# Pre-load prepped panel data once so compute_coefs_byyear_numpy
# doesn't reload from disk on every calibration iteration.
_prepped_df = pd.read_csv(f"{MAIN_DIR}/data/intermediate/analysis_data_mstock_input.csv")
_scalars    = pd.read_csv(f"{MAIN_DIR}/data/intermediate/prepped_scalars.csv")
_med_g      = float(_scalars["med_preIPO_growth"].iloc[0])

# -----------------------------------------------------------------------------
# Grids and productivity discretization
# -----------------------------------------------------------------------------

from solve_vf import discretize_productivity, discretize_choices

print(f"AR(1) parameters for productivity: rho = {rho:.4f}, sigma_eps = {sigma_eps:.4f}")
z_grid, pi, Pi = discretize_productivity(rho, sigma_eps, 10)
m_grid = discretize_choices(1e-3, 100, 100, type="exp")
k_grid = discretize_choices(1e-3, 100, 100, type="exp")
print(f"m_grid: {m_grid[:10]} ... {m_grid[-10:]}")
print(f"k_grid: {k_grid[:10]} ... {k_grid[-10:]}")

# gamma_k and gamma_l are fixed structural parameters (not calibrated).
# Loaded from structural_parameters.csv (computed in 3c_exog_params.R).
gamma_l = struct_params["gamma_l"].iloc[0]  # labor exponent, used in phi back-out
gamma_k = struct_params["gamma_k"].iloc[0]  # capital exponent, used in equilibrium only

print(f"\nProduction function parameters (fixed, not calibrated):")
print(f"  gamma_k = {gamma_k:.4f}  (capital exponent in Y = Z*K^gamma_k*L^gamma_l)")
print(f"  gamma_l = {gamma_l:.4f}  (labor exponent)")
print(f"  gamma_k + gamma_l = {gamma_k + gamma_l:.4f}  (< 1 => DRS, = 1 => CRS, > 1 => IRS)")

print("\nCalibration setup:")
print(f"  Target med_sales           = {med_sales_init:.4f}")
print(f"  Target med_capx            = {med_capx_init:.4f}")
print(f"  Target neg_ebitda_base_pct = {neg_ebitda_base_pct:.4f}")
print(f"  Target neg_ebitda_final_pct= {neg_ebitda_final_pct:.4f}")
print(f"  (phi_track_values will be computed from calibrated sigma and fixed gamma_l inside the objective)")

# First, check compute_coefs_byyear_numpy with the initial guess for delta_m (0.15) to verify it matches the R version used in model_implied_trends.py
coefs_check_py = compute_coefs_byyear_numpy(0.15, prepped_df=_prepped_df, med_preIPO_growth=_med_g)
coefs_check_r = pd.read_csv(f"{MAIN_DIR}/data/clean/coefs_byyear_delta_m_15.csv")
coefs_check = pd.DataFrame({
    "year": coefs_check_py[:, 0],
    "coef_py": coefs_check_py[:, 1],
    "coef_r": coefs_check_r["coef"],
})
print("\nChecking compute_coefs_byyear_numpy with delta_m=0.15:")
print(coefs_check)

# -----------------------------------------------------------------------------
# Calibration objective
# -----------------------------------------------------------------------------

_warm_start_phi0 = None   # np.array([c_agg, W, P_M]) from last successful phi0 solve
_warm_start_phiT = None   # same for phiT

def calibration_residuals(x):
    """
    Map (alpha_a, alpha_k, z_k, sigma, delta_m) -> relative moment errors.

    Parameters
    ----------
    alpha_a : DRS exponent for advertising labor: L^a(a) = (a/z_a)^(1/alpha_a), z_a=1 (normalized)
    alpha_k : DRS exponent for capital investment: L^k(i) = (i/z_k)^(1/alpha_k)
    z_k     : scale for capital investment cost function
    sigma   : demand elasticity (also governs phi_track_values via structural mapping)
    delta_m : depreciation rate of customer capital (m_stock)

    gamma_k, gamma_l are fixed from EqmParams defaults (not calibrated).

    phi back-out uses the arbitrary-scale Cobb-Douglas mapping:
      phi_t = beta_t * ((1 - gamma_l) * sigma + gamma_l) - 1

    Returns a 1-vector:
      [ ((pct_neg_phiT - pct_neg_phi0) - (neg_ebitda_final_pct - neg_ebitda_base_pct)) / 100 ]

    Targets the RISE in negative earnings (phiT minus phi0) to match the data trend.
    med_adv_ratio and med_inv_ratio are printed for information but not targeted.
    Both phi_track_values and coefs_byyear depend on delta_m and sigma,
    so they are recomputed at each evaluation.
    """
    global _warm_start_phi0, _warm_start_phiT
    alpha_a_try, alpha_k_try, z_k_try, sigma_try, delta_m_try = (
        float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])
    )
    print(
        "Trying: alpha_a={:.4f}, alpha_k={:.4f}, z_k={:.4f}, "
        "sigma={:.4f}, delta_m={:.4f}".format(
            alpha_a_try, alpha_k_try, z_k_try, sigma_try, delta_m_try
        )
    )

    # Enforce parameter constraints (Nelder-Mead has no native bounds; violations penalized here)
    if not (lower_bounds[0] <= alpha_a_try <= upper_bounds[0]):
        return 1e6
    if not (lower_bounds[1] <= alpha_k_try <= upper_bounds[1]):
        return 1e6
    if not (lower_bounds[2] <= z_k_try <= upper_bounds[2]):
        return 1e6
    if not (lower_bounds[3] <= sigma_try <= upper_bounds[3]):
        return 1e6
    if not (lower_bounds[4] <= delta_m_try <= upper_bounds[4]):
        return 1e6

    # Recompute coefs_byyear for the trial delta_m (fast JIT call)
    coefs_byyear_try = compute_coefs_byyear_numpy(
        delta_m_try, prepped_df=_prepped_df, med_preIPO_growth=_med_g
    )

    # Back out phi path using arbitrary-scale Cobb-Douglas structural mapping:
    # phi_t = beta_t * ((1 - gamma_l) * sigma + gamma_l) - 1
    phi_track_values_try = coefs_byyear_try[:, 1] * ((1 - gamma_l) * sigma_try + gamma_l) - 1
    phi0_try = phi_track_values_try[0]
    phiT_try = phi_track_values_try[-1]

    def _solve(phi_val, warm_start=None):
        params = EqmParams(
            phi=phi_val,
            entry_perc=exit_rate,
            sigma=sigma_try,
            alpha_a=alpha_a_try,
            alpha_k=alpha_k_try,
            z_k=z_k_try,
            z_a=1.0,
            delta_m=delta_m_try,
            gamma_k=gamma_k,
            gamma_l=gamma_l,
        )
        try:
            eqm = solve_ss_equilibrium_least_squares(
                m_grid, k_grid, z_grid, Pi, params, verbose=False,
                start=warm_start,
            )
        except Exception as e:
            print(f"  [FAIL] phi={phi_val:.4f} -> exception in equilibrium solver: {e}")
            return None
        ls_success = eqm.get("ls_success", False)
        eqm_res    = eqm.get("residuals", np.array([np.inf, np.inf, np.inf]))
        res_norm   = float(np.linalg.norm(eqm_res))
        if (not ls_success) or not np.isfinite(res_norm) or res_norm > 1e-4:
            print(
                f"  [BAD EQM] phi={phi_val:.4f} -> eqm_res_norm={res_norm:.3e}, "
                f"ls_success={ls_success}"
            )
            return None
        return eqm

    # Solve at phi0 (moments 1-3)
    eqm_phi0 = _solve(phi0_try, warm_start=_warm_start_phi0)
    if eqm_phi0 is None:
        return 1e4
    _warm_start_phi0 = np.array([eqm_phi0["c_agg"], eqm_phi0["W"], eqm_phi0["P_M"]])

    # Solve at phiT (moment 2)
    eqm_phiT = _solve(phiT_try, warm_start=_warm_start_phiT)
    if eqm_phiT is None:
        return 1e4
    _warm_start_phiT = np.array([eqm_phiT["c_agg"], eqm_phiT["W"], eqm_phiT["P_M"]])

    # Compute model-implied moments
    med_sales_model = median_adv_ratio(m_grid, k_grid, z_grid, eqm_phi0, negative_earnings_only=False)
    med_capx_model  = median_inv_ratio(m_grid, k_grid, z_grid, eqm_phi0, negative_earnings_only=False)
    pct_neg_phi0    = pct_negative(m_grid, k_grid, z_grid, eqm_phi0)
    pct_neg_phiT    = pct_negative(m_grid, k_grid, z_grid, eqm_phiT)

    data_rise  = neg_ebitda_final_pct - neg_ebitda_base_pct
    model_rise = pct_neg_phiT - pct_neg_phi0
    obj = ((model_rise - data_rise) / 100) ** 2

    print(
        f"  alpha_a={alpha_a_try:.4f}, alpha_k={alpha_k_try:.4f}, z_k={z_k_try:.4f}, "
        f"sigma={sigma_try:.4f}, delta_m={delta_m_try:.4f} | "
        f"phi0={phi0_try:.4f}, phiT={phiT_try:.4f} -> "
        f"sales={med_sales_model:.4f} (info), capx={med_capx_model:.4f} (info), "
        f"pct_neg_phi0={pct_neg_phi0:.2f}% (info), rise_model={model_rise:.2f}% (data={data_rise:.2f}%) | "
        f"obj={obj:.6f}"
    )

    return obj


# -----------------------------------------------------------------------------
# Initial guess and bounds
# Parameters: [alpha_a, alpha_k, z_k, sigma, delta_m]
# alpha_a, alpha_k in (0,1) for DRS; z_k > 0; sigma > 1; delta_m in [0.10, 0.50]
# -----------------------------------------------------------------------------

x0           = np.array([0.5,  0.5,  1.0, 4.0, 0.15])
lower_bounds = np.array([1e-3, 1e-3, 1e-3, 1.01, 0.10])
upper_bounds = np.array([0.75, 0.75, 2.0, 10.0, 0.30])  # enforced as penalties inside objective (Nelder-Mead has no native bounds)

print("\n" + "=" * 60)
print("Starting Nelder-Mead calibration of alpha_a, alpha_k, z_k, sigma, delta_m")
print("z_a normalized to 1.0 (advertising scale absorbed by P_M)")
print(f"gamma_k = {gamma_k:.4f}, gamma_l = {gamma_l:.4f} (fixed, not calibrated)")
print("Moment: rise in pct_neg (phiT - phi0)  (levels, med_sales, med_capx printed for info only)")
print("delta_m bounds: [0.10, 0.30], initial guess: 0.15")
print("phi back-out: beta * ((1 - gamma_l) * sigma + gamma_l) - 1")
print("=" * 60)

# cal_result = least_squares(
#     calibration_residuals,
#     x0,
#     bounds=(lower_bounds, upper_bounds),
#     method="trf",
#     xtol=1e-6,
#     ftol=1e-6,
#     gtol=1e-6,
#     max_nfev=10000,
#     verbose=2,
# )

# cal_result = differential_evolution(
#     calibration_residuals,
#     bounds=list(zip(lower_bounds, upper_bounds)),
#     seed=42,
#     maxiter=1000,
#     tol=1e-8,
#     mutation=(0.5, 1.5),
#     recombination=0.9,
#     popsize=15,
#     init="latinhypercube",
#     disp=True,
# )

# cal_result = minimize(
#     calibration_residuals,
#     x0,
#     method="Powell",
#     bounds=list(zip(lower_bounds, upper_bounds)),
#     options={"maxiter": 10000, "ftol": 1e-8, "xtol": 1e-8, "disp": True},
# )

cal_result = minimize(
    calibration_residuals,
    x0,
    method="L-BFGS-B",
    bounds=list(zip(lower_bounds, upper_bounds))
)

alpha_a_cal, alpha_k_cal, z_k_cal, sigma_cal, delta_m_cal = cal_result.x
z_a_cal = 1.0

# Recompute coefs and phi path at calibrated parameters
coefs_cal = compute_coefs_byyear_numpy(delta_m_cal, prepped_df=_prepped_df, med_preIPO_growth=_med_g)
phi_track_values_cal = coefs_cal[:, 1] * ((1 - gamma_l) * sigma_cal + gamma_l) - 1
phi0_cal = phi_track_values_cal[0]
phiT_cal = phi_track_values_cal[-1]

print("\nCalibration result:")
print(f"  success      = {cal_result.success}")
print(f"  message      = {cal_result.message}")
print(f"  alpha_a_cal  = {alpha_a_cal:.6f}")
print(f"  alpha_k_cal  = {alpha_k_cal:.6f}")
print(f"  z_k_cal      = {z_k_cal:.6f}")
print(f"  sigma_cal    = {sigma_cal:.6f}")
print(f"  delta_m_cal  = {delta_m_cal:.6f}")
print(f"  z_a          = {z_a_cal:.6f} (normalized)")
print(f"  gamma_k      = {gamma_k:.6f} (fixed)")
print(f"  gamma_l      = {gamma_l:.6f} (fixed)")
print(f"  phi0_cal     = {phi0_cal:.6f}")
print(f"  phiT_cal     = {phiT_cal:.6f}")
print(f"  final_resids = {cal_result.fun}")

# Post-calibration diagnostics: recompute equilibria at phi0 and phiT
print("\nPost-calibration diagnostics at calibrated parameters:")

def _make_params(phi_val):
    return EqmParams(
        phi=phi_val,
        entry_perc=exit_rate,
        sigma=sigma_cal,
        alpha_a=alpha_a_cal,
        alpha_k=alpha_k_cal,
        z_k=z_k_cal,
        z_a=z_a_cal,
        delta_m=delta_m_cal,
        gamma_k=gamma_k,
        gamma_l=gamma_l,
    )

eqm_phi0_cal = solve_ss_equilibrium_least_squares(
    m_grid, k_grid, z_grid, Pi, _make_params(phi0_cal), verbose=False,
)
eqm_phiT_cal = solve_ss_equilibrium_least_squares(
    m_grid, k_grid, z_grid, Pi, _make_params(phiT_cal), verbose=False,
)

med_sales_model  = median_adv_ratio(m_grid, k_grid, z_grid, eqm_phi0_cal, negative_earnings_only=False)
med_capx_model   = median_inv_ratio(m_grid, k_grid, z_grid, eqm_phi0_cal, negative_earnings_only=False)
pct_neg_phi0_cal = pct_negative(m_grid, k_grid, z_grid, eqm_phi0_cal)
pct_neg_phiT_cal = pct_negative(m_grid, k_grid, z_grid, eqm_phiT_cal)

print(
    f"  med_sales  @ phi0: data={med_sales_init:.4f}, model={med_sales_model:.4f}, "
    f"rel_err={(med_sales_model - med_sales_init) / med_sales_init:.4f}"
)
print(
    f"  med_capx   @ phi0: data={med_capx_init:.4f}, model={med_capx_model:.4f}, "
    f"rel_err={(med_capx_model - med_capx_init) / med_capx_init:.4f}"
)
print(
    f"  pct_neg    @ phi0: data={neg_ebitda_base_pct:.2f}%, model={pct_neg_phi0_cal:.2f}%, "
    f"rel_err={(pct_neg_phi0_cal - neg_ebitda_base_pct) / neg_ebitda_base_pct:.4f}"
)
print(
    f"  pct_neg    @ phiT: data={neg_ebitda_final_pct:.2f}%, model={pct_neg_phiT_cal:.2f}%, "
    f"rel_err={(pct_neg_phiT_cal - neg_ebitda_final_pct) / neg_ebitda_final_pct:.4f}"
)

# Save calibrated parameters
out_path = os.path.join(
    MAIN_DIR, "code", "3_ComputationalEx_NEW", "calibrated_investment_params.csv"
)
out_arr = np.array([[alpha_a_cal, alpha_k_cal, z_k_cal, sigma_cal, delta_m_cal]])
np.savetxt(
    out_path,
    out_arr,
    delimiter=",",
    header="alpha_a,alpha_k,z_k,sigma,delta_m",
    comments="",
)
print(f"\nCalibrated parameters saved to {out_path}")
