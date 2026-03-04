from solve_eqm import EqmParams, solve_ss_equilibrium_least_squares
from integrate_dist import median_adv_ratio, median_inv_ratio, pct_negative
import numpy as np
import pandas as pd
import os
from scipy.optimize import least_squares, differential_evolution


# -----------------------------------------------------------------------------
# Paths and data loading
#   All paths are relative to this script's directory so it runs anywhere
#   (local machine or Quest cluster).
# -----------------------------------------------------------------------------

_DIR = os.path.dirname(__file__)

struct_params = pd.read_csv(os.path.join(_DIR, "structural_parameters.csv"))
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

print("AR(1) parameters for productivity: rho = {:.4f}, sigma_eps = {:.4f}".format(rho, sigma_eps))
z_grid, pi, Pi = discretize_productivity(rho, sigma_eps, 10)
m_grid = discretize_choices(1e-3, 10, 100, type="exp")
k_grid = discretize_choices(1e-3, 10, 100, type="exp")

gamma_l = struct_params["gamma_l"].iloc[0]
gamma_k = struct_params["gamma_k"].iloc[0]

print("\nProduction function parameters (fixed):")
print("  gamma_k = {:.4f},  gamma_l = {:.4f}".format(gamma_k, gamma_l))

# Load phi trajectory from R output (4b_mstock_coef.R)
_coefs_df   = pd.read_csv(os.path.join(_DIR, "sales_elasticity_m_by_year.csv"))
coefs_fixed = _coefs_df[["year", "coef"]].values  # shape (T, 2)

# -----------------------------------------------------------------------------
# Calibration targets
# -----------------------------------------------------------------------------

print("\nCalibration targets:")
print("  med_capx  @ phi0       = {:.4f}".format(med_capx_init))
print("  pct_neg   @ phi0       = {:.4f}%".format(neg_ebitda_base_pct))
print("  pct_neg   @ phiT       = {:.4f}%".format(neg_ebitda_final_pct))

# -----------------------------------------------------------------------------
# Load initial guesses from previous calibration (or use defaults)
# -----------------------------------------------------------------------------

out_path = os.path.join(_DIR, "calibrated_investment_params.csv")

sigma_init   = 4.0
alpha_k_init = 0.5
alpha_a_init = 0.5

if os.path.isfile(out_path):
    _cal_prev    = pd.read_csv(out_path)
    sigma_init   = float(_cal_prev["sigma"].iloc[0])
    alpha_k_init = float(_cal_prev["alpha_k"].iloc[0])
    alpha_a_init = float(_cal_prev["alpha_a"].iloc[0])
    print("\nLoaded initial guesses from {}:".format(out_path))
    print("  sigma_init   = {:.6f}".format(sigma_init))
    print("  alpha_k_init = {:.6f}".format(alpha_k_init))
    print("  alpha_a_init = {:.6f}".format(alpha_a_init))
else:
    print("\nNo existing calibrated parameters found; using defaults.")

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

def _solve_eqm(phi_val, sigma, alpha_a, alpha_k, warm_start=None, max_nfev=1000, vf_maxit=250):
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
            start=warm_start, max_nfev=max_nfev, vf_maxit=vf_maxit,
        )
    except Exception as e:
        print("  [FAIL] phi={:.4f} sigma={:.4f} -> {}".format(phi_val, sigma, e))
        return None
    ls_success = eqm.get("ls_success", False)
    res_norm = float(np.linalg.norm(eqm.get("residuals", np.array([np.inf, np.inf, np.inf]))))
    if (not ls_success) or not np.isfinite(res_norm) or res_norm > 1e-4:
        print("  [BAD EQM] phi={:.4f} sigma={:.4f} res_norm={:.3e}".format(phi_val, sigma, res_norm))
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

# Warm-start state dict (updated in-place by obj_joint during LS phase)
# Defined at module level so each spawned worker process gets its own copy.
_ws = {'phi0': None, 'phiT': None}


def obj_joint(x, max_nfev=20, vf_maxit=100):
    sigma_try, alpha_a_try, alpha_k_try = float(x[0]), float(x[1]), float(x[2])
    phi0, phiT = _phi_endpoints(sigma_try)

    eqm_phi0 = _solve_eqm(phi0, sigma_try, alpha_a_try, alpha_k_try, warm_start=_ws['phi0'], max_nfev=max_nfev, vf_maxit=vf_maxit)
    if eqm_phi0 is None:
        return np.full(3, 1e3)
    _ws['phi0'] = np.array([eqm_phi0["c_agg"], eqm_phi0["W"], eqm_phi0["P_M"]])

    eqm_phiT = _solve_eqm(phiT, sigma_try, alpha_a_try, alpha_k_try, warm_start=_ws['phiT'], max_nfev=max_nfev, vf_maxit=vf_maxit)
    if eqm_phiT is None:
        return np.full(3, 1e3)
    _ws['phiT'] = np.array([eqm_phiT["c_agg"], eqm_phiT["W"], eqm_phiT["P_M"]])

    pct0     = pct_negative(m_grid, k_grid, z_grid, eqm_phi0)
    pctT     = pct_negative(m_grid, k_grid, z_grid, eqm_phiT)
    med_capx = median_inv_ratio(m_grid, k_grid, z_grid, eqm_phi0)

    res = np.array([
        (pct0     - neg_ebitda_base_pct)/100,
        (pctT     - neg_ebitda_final_pct)/100,
        (med_capx - med_capx_init),
    ])
    print(
        "sig={:.4f} ak={:.4f} aa={:.4f} | "
        "pct0={:.2f}% pctT={:.2f}% capx={:.4f} | "
        "||res||={:.6f}".format(sigma_try, alpha_k_try, alpha_a_try,
                                pct0, pctT, med_capx,
                                np.linalg.norm(res))
    )
    return res


_BOUNDS_LOW  = [1.5, 0.1, 0.1]
_BOUNDS_HIGH = [10.0, 0.9, 0.9]

def _obj_scalar(x):
    """Scalar sum-of-squares objective for DE (each worker gets its own process copy of _ws)."""
    return float(np.sum(obj_joint(x) ** 2))

if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # Initialize warm starts from initial guesses
    # -------------------------------------------------------------------------

    _phi0_init, _phiT_init = _phi_endpoints(sigma_init)
    print("\nInitial phi endpoints: phi0={:.4f}, phiT={:.4f}".format(_phi0_init, _phiT_init))

    _pre_phi0 = _solve_eqm(_phi0_init, sigma_init, alpha_a_init, alpha_k_init, max_nfev=1000)
    _pre_phiT = _solve_eqm(_phiT_init, sigma_init, alpha_a_init, alpha_k_init, max_nfev=1000)
    if _pre_phi0 is not None:
        _ws['phi0'] = np.array([_pre_phi0["c_agg"], _pre_phi0["W"], _pre_phi0["P_M"]])
    if _pre_phiT is not None:
        _ws['phiT'] = np.array([_pre_phiT["c_agg"], _pre_phiT["W"], _pre_phiT["P_M"]])

    # -------------------------------------------------------------------------
    # Phase 1: Differential Evolution — global search
    # -------------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("Phase 1: Differential Evolution (global search)")
    print("  Targets: med_capx={:.4f}, pct_neg@phi0={:.2f}%, pct_neg@phiT={:.2f}%".format(
        med_capx_init, neg_ebitda_base_pct, neg_ebitda_final_pct))
    print("=" * 60)

    de_result = differential_evolution(
        _obj_scalar,
        bounds=list(zip(_BOUNDS_LOW, _BOUNDS_HIGH)),
        seed=42,
        strategy="best1bin",
        maxiter=300,
        popsize=5,          # 5 × 3 = 15 candidates per generation
        tol=1e-4,
        mutation=(0.5, 1.0),
        recombination=0.7,
        # workers=-1,          # parallel across all available CPUs (set by --cpus-per-task in SLURM)
        updating='deferred', # required for workers > 1; set explicitly to suppress scipy warning
        polish=False,        # we polish with least_squares below
        disp=True,
    )

    print("\nDE result:  success={},  cost={:.6f}".format(de_result.success, de_result.fun))
    print("  sigma={:.4f}  alpha_a={:.4f}  alpha_k={:.4f}".format(de_result.x[0], de_result.x[1], de_result.x[2]))

    # -------------------------------------------------------------------------
    # Re-seed warm starts from DE best point before LS refinement
    # -------------------------------------------------------------------------

    _sigma_de, _aa_de, _ak_de = de_result.x
    _phi0_de, _phiT_de = _phi_endpoints(_sigma_de)
    print("\nRe-seeding warm starts from DE solution ...")
    _pre_phi0 = _solve_eqm(_phi0_de, _sigma_de, _aa_de, _ak_de, max_nfev=1000)
    _pre_phiT = _solve_eqm(_phiT_de, _sigma_de, _aa_de, _ak_de, max_nfev=1000)
    if _pre_phi0 is not None:
        _ws['phi0'] = np.array([_pre_phi0["c_agg"], _pre_phi0["W"], _pre_phi0["P_M"]])
    if _pre_phiT is not None:
        _ws['phiT'] = np.array([_pre_phiT["c_agg"], _pre_phiT["W"], _pre_phiT["P_M"]])

    # -------------------------------------------------------------------------
    # Phase 2: Least Squares — local refinement from DE best point
    # -------------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("Phase 2: Least Squares refinement from DE solution")
    print("=" * 60)

    result = least_squares(
        lambda x: obj_joint(x, max_nfev=500),
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

    print("\nCalibration result:")
    print("  success      = {}".format(result.success))
    print("  message      = {}".format(result.message))
    print("  sigma_cal    = {:.6f}".format(sigma_cal))
    print("  alpha_a_cal  = {:.6f}".format(alpha_a_cal))
    print("  alpha_k_cal  = {:.6f}".format(alpha_k_cal))
    print("  phi0         = {:.6f}".format(phi0_cal))
    print("  phiT         = {:.6f}".format(phiT_cal))
    print("  residuals    = {}".format(result.fun))
    print("  cost         = {:.8f}".format(result.cost))

    # -------------------------------------------------------------------------
    # Post-calibration diagnostics
    # -------------------------------------------------------------------------

    print("\nPost-calibration diagnostics:")

    eqm_phi0_cal = _solve_eqm(phi0_cal, sigma_cal, alpha_a_cal, alpha_k_cal)
    eqm_phiT_cal = _solve_eqm(phiT_cal, sigma_cal, alpha_a_cal, alpha_k_cal)

    med_capx_model   = median_inv_ratio(m_grid, k_grid, z_grid, eqm_phi0_cal)
    pct_neg_phi0_cal = pct_negative(m_grid, k_grid, z_grid, eqm_phi0_cal)
    pct_neg_phiT_cal = pct_negative(m_grid, k_grid, z_grid, eqm_phiT_cal) if eqm_phiT_cal is not None else float('nan')

    print("  med_capx @ phi0: data={:.4f},           model={:.4f}".format(med_capx_init, med_capx_model))
    print("  pct_neg  @ phi0: data={:.2f}%,  model={:.2f}%".format(neg_ebitda_base_pct, pct_neg_phi0_cal))
    print("  pct_neg  @ phiT: data={:.2f}%, model={:.2f}%".format(neg_ebitda_final_pct, pct_neg_phiT_cal))

    # -------------------------------------------------------------------------
    # Save calibrated parameters
    # -------------------------------------------------------------------------

    out_arr = np.array([[alpha_a_cal, alpha_k_cal, sigma_cal]])
    np.savetxt(
        out_path,
        out_arr,
        delimiter=",",
        header="alpha_a,alpha_k,sigma",
        comments="",
    )
    print("\nCalibrated parameters saved to {}".format(out_path))
