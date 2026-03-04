from solve_eqm import EqmParams, solve_ss_equilibrium_least_squares
from integrate_dist import median_adv_ratio, median_inv_ratio, pct_negative
import numpy as np
import pandas as pd
import os
from scipy.optimize import least_squares, differential_evolution


# -----------------------------------------------------------------------------
# Paths and data loading
# -----------------------------------------------------------------------------

MAIN_DIR = "/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings"

struct_params = pd.read_csv(os.path.join(os.path.dirname(__file__), "structural_parameters.csv"))
rho              = struct_params["rho"].iloc[0]
sigma_eps        = struct_params["sigma_xi"].iloc[0]
exit_rate        = struct_params["exit_rate"].iloc[0]
med_capx_init    = struct_params["med_capx_sale"].iloc[0]
neg_ebitda_base_pct  = struct_params["neg_ebitda_base"].iloc[0] * 100
neg_ebitda_final_pct = struct_params["neg_ebitda_final"].iloc[0] * 100


# -----------------------------------------------------------------------------
# Grids and productivity discretization
# -----------------------------------------------------------------------------

from solve_vf import discretize_productivity, discretize_choices

print(f"AR(1) parameters for productivity: rho = {rho:.4f}, sigma_eps = {sigma_eps:.4f}")
z_grid, pi, Pi = discretize_productivity(rho, sigma_eps, 10)
m_grid = discretize_choices(1e-3, 10, 100, type="exp")
k_grid = discretize_choices(1e-3, 10, 100, type="exp")

gamma_l = struct_params["gamma_l"].iloc[0]
gamma_k = struct_params["gamma_k"].iloc[0]

print(f"\nProduction function parameters (fixed):")
print(f"  gamma_k = {gamma_k:.4f},  gamma_l = {gamma_l:.4f}")

# Load phi trajectory from R output (4b_mstock_coef.R)
_coefs_df   = pd.read_csv(os.path.join(os.path.dirname(__file__), "sales_elasticity_m_by_year.csv"))
coefs_fixed = _coefs_df[["year", "coef"]].to_numpy()  # shape (T, 2)

# -----------------------------------------------------------------------------
# Calibration targets
# -----------------------------------------------------------------------------

print("\nCalibration targets:")
print(f"  med_capx  @ phi0       = {med_capx_init:.4f}")
print(f"  pct_neg   @ phi0       = {neg_ebitda_base_pct:.4f}%")
print(f"  pct_neg   @ phiT       = {neg_ebitda_final_pct:.4f}%")

# -----------------------------------------------------------------------------
# Load initial guesses from previous calibration (or use defaults)
# -----------------------------------------------------------------------------

out_path = os.path.join(
    os.path.dirname(__file__), "calibrated_investment_params.csv"
)

sigma_init   = 4.0
alpha_k_init = 0.5
alpha_a_init = 0.5

if os.path.isfile(out_path):
    _cal_prev    = pd.read_csv(out_path)
    sigma_init   = float(_cal_prev["sigma"].iloc[0])
    alpha_k_init = float(_cal_prev["alpha_k"].iloc[0])
    alpha_a_init = float(_cal_prev["alpha_a"].iloc[0])
    print(f"\nLoaded initial guesses from {out_path}:")
    print(f"  sigma_init   = {sigma_init:.6f}")
    print(f"  alpha_k_init = {alpha_k_init:.6f}")
    print(f"  alpha_a_init = {alpha_a_init:.6f}")
else:
    print(f"\nNo existing calibrated parameters found; using defaults.")

# -----------------------------------------------------------------------------
# Helper: phi endpoints as a function of sigma
# -----------------------------------------------------------------------------

def _phi_endpoints(sigma):
    """Compute phi at t=0 and t=T from sigma and the pre-computed coefs array."""
    phi_track = coefs_fixed[:, 1] * ((1 - gamma_l) * sigma + gamma_l) - 1
    return phi_track[0], phi_track[-1]

# -----------------------------------------------------------------------------
# Helper: solve equilibrium with convergence check
# -----------------------------------------------------------------------------

def _solve_eqm(phi_val, sigma, alpha_a, alpha_k, warm_start=None):
    params = EqmParams(
        phi=phi_val,
        entry_perc=exit_rate,
        sigma=sigma,
        alpha_a=alpha_a,
        alpha_k=alpha_k,
        gamma_k=gamma_k,
        gamma_l=gamma_l,
    )
    try:
        eqm = solve_ss_equilibrium_least_squares(
            m_grid, k_grid, z_grid, Pi, params, verbose=False,
            start=warm_start,
        )
    except Exception as e:
        print(f"  [FAIL] phi={phi_val:.4f} sigma={sigma:.4f} -> {e}")
        return None
    ls_success = eqm.get("ls_success", False)
    res_norm = float(np.linalg.norm(eqm.get("residuals", np.array([np.inf, np.inf, np.inf]))))
    if (not ls_success) or not np.isfinite(res_norm) or res_norm > 1e-4:
        print(f"  [BAD EQM] phi={phi_val:.4f} sigma={sigma:.4f} res_norm={res_norm:.3e}")
        return None
    return eqm

# -----------------------------------------------------------------------------
# Calibration: (sigma, alpha_a, alpha_k)
#   Phase 1 — Differential Evolution (global, gradient-free)
#   Phase 2 — least_squares refinement (local, from DE best point)
#   Residuals:
#     pct_neg  @ phi0  =  neg_ebitda_base_pct
#     pct_neg  @ phiT  =  neg_ebitda_final_pct
#     med_capx @ phi0  =  med_capx_init
# -----------------------------------------------------------------------------

# Warm-start state dict (updated in-place by obj_joint)
_ws = {'phi0': None, 'phiT': None}

# Initialize warm starts from initial guesses
_phi0_init, _phiT_init = _phi_endpoints(sigma_init)
print(f"\nInitial phi endpoints: phi0={_phi0_init:.4f}, phiT={_phiT_init:.4f}")

_pre_phi0 = _solve_eqm(_phi0_init, sigma_init, alpha_a_init, alpha_k_init)
_pre_phiT = _solve_eqm(_phiT_init, sigma_init, alpha_a_init, alpha_k_init)
if _pre_phi0 is not None:
    _ws['phi0'] = np.array([_pre_phi0["c_agg"], _pre_phi0["W"], _pre_phi0["P_M"]])
if _pre_phiT is not None:
    _ws['phiT'] = np.array([_pre_phiT["c_agg"], _pre_phiT["W"], _pre_phiT["P_M"]])


def obj_joint(x):
    sigma_try, alpha_a_try, alpha_k_try = float(x[0]), float(x[1]), float(x[2])
    phi0, phiT = _phi_endpoints(sigma_try)

    eqm_phi0 = _solve_eqm(phi0, sigma_try, alpha_a_try, alpha_k_try, warm_start=_ws['phi0'])
    if eqm_phi0 is None:
        return np.full(3, 1e3)
    _ws['phi0'] = np.array([eqm_phi0["c_agg"], eqm_phi0["W"], eqm_phi0["P_M"]])

    eqm_phiT = _solve_eqm(phiT, sigma_try, alpha_a_try, alpha_k_try, warm_start=_ws['phiT'])
    if eqm_phiT is None:
        return np.full(3, 1e3)
    _ws['phiT'] = np.array([eqm_phiT["c_agg"], eqm_phiT["W"], eqm_phiT["P_M"]])

    pct0     = pct_negative(m_grid, k_grid, z_grid, eqm_phi0)
    pctT     = pct_negative(m_grid, k_grid, z_grid, eqm_phiT)
    med_capx = median_inv_ratio(m_grid, k_grid, z_grid, eqm_phi0)

    res = np.array([
        pct0     - neg_ebitda_base_pct,
        pctT     - neg_ebitda_final_pct,
        med_capx - med_capx_init,
    ])
    print(
        f"  sig={sigma_try:.4f} ak={alpha_k_try:.4f} aa={alpha_a_try:.4f} | "
        f"pct0={pct0:.2f}% pctT={pctT:.2f}% capx={med_capx:.4f} | "
        f"||res||={np.linalg.norm(res):.6f}"
    )
    return res


_BOUNDS_LOW  = [1.0,  1e-3, 1e-3]
_BOUNDS_HIGH = [10.0, 0.99, 0.99]

def _obj_scalar(x):
    """Scalar sum-of-squares objective for DE (each worker gets its own process copy of _ws)."""
    return float(np.sum(obj_joint(x) ** 2))

# -----------------------------------------------------------------------------
# Phase 1: Differential Evolution — global search
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Phase 1: Differential Evolution (global search)")
print(f"  Targets: med_capx={med_capx_init:.4f}, "
      f"pct_neg@phi0={neg_ebitda_base_pct:.2f}%, pct_neg@phiT={neg_ebitda_final_pct:.2f}%")
print("=" * 60)

de_result = differential_evolution(
    _obj_scalar,
    bounds=list(zip(_BOUNDS_LOW, _BOUNDS_HIGH)),
    seed=42,
    strategy="best1bin",
    maxiter=2,
    popsize=5,          # 15 × 3 = 45 candidates per generation
    tol=1e-4,
    mutation=(0.5, 1.0),
    recombination=0.7,
    workers=-1,          # parallel across all available CPUs
    polish=False,        # we polish with least_squares below
    disp=True,
)

print(f"\nDE result:  success={de_result.success},  cost={de_result.fun:.6f}")
print(f"  sigma={de_result.x[0]:.4f}  alpha_a={de_result.x[1]:.4f}  alpha_k={de_result.x[2]:.4f}")

# -----------------------------------------------------------------------------
# Re-seed warm starts from DE best point before LS refinement
# -----------------------------------------------------------------------------

_sigma_de, _aa_de, _ak_de = de_result.x
_phi0_de, _phiT_de = _phi_endpoints(_sigma_de)
print(f"\nRe-seeding warm starts from DE solution ...")
_pre_phi0 = _solve_eqm(_phi0_de, _sigma_de, _aa_de, _ak_de)
_pre_phiT = _solve_eqm(_phiT_de, _sigma_de, _aa_de, _ak_de)
if _pre_phi0 is not None:
    _ws['phi0'] = np.array([_pre_phi0["c_agg"], _pre_phi0["W"], _pre_phi0["P_M"]])
if _pre_phiT is not None:
    _ws['phiT'] = np.array([_pre_phiT["c_agg"], _pre_phiT["W"], _pre_phiT["P_M"]])

# -----------------------------------------------------------------------------
# Phase 2: Least Squares — local refinement from DE best point
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Phase 2: Least Squares refinement from DE solution")
print("=" * 60)

result = least_squares(
    obj_joint,
    x0=de_result.x,
    method="trf",
    bounds=(_BOUNDS_LOW, _BOUNDS_HIGH),
    xtol=1e-5,
    ftol=1e-8,
    gtol=1e-8,
    verbose=2,
)

sigma_cal   = float(result.x[0])
alpha_a_cal = float(result.x[1])
alpha_k_cal = float(result.x[2])
phi0_cal, phiT_cal = _phi_endpoints(sigma_cal)

print(f"\nCalibration result:")
print(f"  success      = {result.success}")
print(f"  message      = {result.message}")
print(f"  sigma_cal    = {sigma_cal:.6f}")
print(f"  alpha_a_cal  = {alpha_a_cal:.6f}")
print(f"  alpha_k_cal  = {alpha_k_cal:.6f}")
print(f"  phi0         = {phi0_cal:.6f}")
print(f"  phiT         = {phiT_cal:.6f}")
print(f"  residuals    = {result.fun}")
print(f"  cost         = {result.cost:.8f}")

# -----------------------------------------------------------------------------
# Post-calibration diagnostics
# -----------------------------------------------------------------------------

print("\nPost-calibration diagnostics:")

eqm_phi0_cal = _solve_eqm(phi0_cal, sigma_cal, alpha_a_cal, alpha_k_cal)
eqm_phiT_cal = _solve_eqm(phiT_cal, sigma_cal, alpha_a_cal, alpha_k_cal)

med_capx_model   = median_inv_ratio(m_grid, k_grid, z_grid, eqm_phi0_cal)
pct_neg_phi0_cal = pct_negative(m_grid, k_grid, z_grid, eqm_phi0_cal)
pct_neg_phiT_cal = pct_negative(m_grid, k_grid, z_grid, eqm_phiT_cal) if eqm_phiT_cal is not None else float('nan')

print(f"  med_capx @ phi0: data={med_capx_init:.4f},           model={med_capx_model:.4f}")
print(f"  pct_neg  @ phi0: data={neg_ebitda_base_pct:.2f}%,  model={pct_neg_phi0_cal:.2f}%")
print(f"  pct_neg  @ phiT: data={neg_ebitda_final_pct:.2f}%, model={pct_neg_phiT_cal:.2f}%")

# -----------------------------------------------------------------------------
# Save calibrated parameters
# -----------------------------------------------------------------------------

out_arr = np.array([[alpha_a_cal, alpha_k_cal, sigma_cal]])
np.savetxt(
    out_path,
    out_arr,
    delimiter=",",
    header="alpha_a,alpha_k,sigma",
    comments="",
)
print(f"\nCalibrated parameters saved to {out_path}")
