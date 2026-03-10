from solve_eqm import EqmParams, solve_ss_equilibrium_least_squares
from integrate_dist import pct_negative, median_adv_ratio, median_inv_ratio
import multiprocessing
import numpy as np
import pandas as pd
import os
from scipy.optimize import differential_evolution, least_squares, brentq, minimize


# -----------------------------------------------------------------------------
# Paths and data loading
#   All paths are relative to this script's directory so it runs anywhere
#   (local machine or Quest cluster).
# -----------------------------------------------------------------------------

_DIR = os.path.dirname(__file__)

struct_params = pd.read_csv(os.path.join(_DIR, "structural_parameters.csv"))

# Fixed structural parameters (from ACF estimation and markup inversion)
gamma_l   = struct_params["gamma_l"].iloc[0]
gamma_k   = struct_params["gamma_k"].iloc[0]
rho       = struct_params["rho"].iloc[0]
sigma_eps = struct_params["sigma_xi"].iloc[0]
sigma_fixed = struct_params["sigma"].iloc[0]  # fixed = mu/(mu-1) from ACF markup
exit_rate = struct_params["exit_rate"].iloc[0]

print("\nFixed structural parameters:")
print("  gamma_k     = {:.6f}".format(gamma_k))
print("  gamma_l     = {:.6f}".format(gamma_l))
print("  rho         = {:.6f}".format(rho))
print("  sigma_eps   = {:.6f}".format(sigma_eps))
print("  sigma_fixed = {:.6f}  (fixed = mu/(mu-1) from ACF markup)".format(sigma_fixed))
print("  exit_rate   = {:.6f}".format(exit_rate))

# Calibration moment targets
med_capx_sale       = struct_params["med_capx_sale"].iloc[0]
neg_ebitda_base_pct = struct_params["neg_ebitda_base"].iloc[0] * 100

print("\nCalibration targets (base period):")
print("  med_capx/rev = {:.4f}".format(med_capx_sale))
print("  pct_neg      = {:.4f}%".format(neg_ebitda_base_pct))

# Year-by-year pct_neg for phi path inversion
pct_neg_df   = pd.read_csv(os.path.join(_DIR, "pct_neg_byyear.csv"))
years        = pct_neg_df["year"].values.astype(int)
pct_neg_data = pct_neg_df["pct_neg"].values   # already in percent (0-100)

print("\nLoaded pct_neg by year: {} years ({}-{})".format(
    len(years), years[0], years[-1]))

# -----------------------------------------------------------------------------
# Normalization: phi_0 = 0 in base period; sigma fixed from structural_parameters
# -----------------------------------------------------------------------------

phi_0 = 0.0
print("\nNormalization: phi_0 = {:.4f}  (base period)".format(phi_0))
print("Fixed:         sigma = {:.6f}  (from structural_parameters)".format(sigma_fixed))

# -----------------------------------------------------------------------------
# Grids (m and k only; z_grid is pre-computed from fixed rho/sigma_eps)
# -----------------------------------------------------------------------------

from solve_vf import discretize_productivity, discretize_choices

m_grid = discretize_choices(1e-3, 5, 100, type="exp")
k_grid = discretize_choices(1e-3, 5, 100, type="exp")

# z_grid is fixed (rho, sigma_eps not calibrated)
z_grid, _, Pi = discretize_productivity(rho, sigma_eps, 15)

# -----------------------------------------------------------------------------
# Load initial guesses from previous calibration (or use defaults)
# -----------------------------------------------------------------------------

out_path = os.path.join(_DIR, "calibrated_investment_params.csv")

alpha_a_init = 0.5
alpha_k_init = 0.5

if os.path.isfile(out_path):
    _cal_prev = pd.read_csv(out_path)
    if "alpha_a" in _cal_prev.columns:
        alpha_a_init = float(_cal_prev["alpha_a"].iloc[0])
    if "alpha_k" in _cal_prev.columns:
        alpha_k_init = float(_cal_prev["alpha_k"].iloc[0])
    print("\nLoaded initial guesses from {}:".format(out_path))
    print("  alpha_a_init = {:.6f}".format(alpha_a_init))
    print("  alpha_k_init = {:.6f}".format(alpha_k_init))
else:
    print("\nNo existing calibrated parameters found; using defaults.")

# -----------------------------------------------------------------------------
# Helper: solve equilibrium with convergence check
# -----------------------------------------------------------------------------

def _solve_eqm(phi_val, alpha_a, alpha_k, sigma_val,
               fixed_cost=0.0, max_nfev=1000, vf_maxit=250, check_upperbound = False):
    params = EqmParams(
        phi=phi_val,
        entry_perc=exit_rate,
        sigma=sigma_val,
        alpha_a=alpha_a,
        alpha_k=alpha_k,
        z_k=1.0,
        fixed_cost=fixed_cost,
        gamma_k=gamma_k,
        gamma_l=gamma_l,
    )
    try:
        eqm = solve_ss_equilibrium_least_squares(
            m_grid, k_grid, z_grid, Pi, params, verbose=False,
            max_nfev=max_nfev, vf_maxit=vf_maxit,
        )
    except Exception as e:
        print("  [FAIL] phi={:.4f} -> {}".format(phi_val, e))
        return None
    ls_success = eqm.get("ls_success", False)
    res_norm = float(np.linalg.norm(eqm.get("residuals", np.array([np.inf, np.inf, np.inf]))))
    if (not ls_success) or not np.isfinite(res_norm) or res_norm > 1e-4:
        dist = eqm.get("Dist", None)
        print(
            "BAD EQM! Bounds: m={:.3f} k={:.3f}".format(
                np.sum(dist[-10:, :, :]) / np.sum(dist) if dist is not None else float('nan'),
                np.sum(dist[:, -10:, :]) / np.sum(dist) if dist is not None else float('nan')
            )
        )
        return None
    if check_upperbound and eqm is not None:
        # check if >5% of firms are at the m or k grid boundaries (indicating potential non-convergence)
        dist = eqm.get("Dist", None)
        pct_m_hi = np.sum(dist[-10:, :, :]) / np.sum(dist)
        pct_k_hi = np.sum(dist[:, -10:, :]) / np.sum(dist)
        if any(pct > 0.05 for pct in [pct_m_hi, pct_k_hi]):
            print(
                "  [WARN] Potential non-convergence: {:.2f}% of firms at m upper bound, {:.2f}% at k upper bound".format(
                    pct_m_hi * 100, pct_k_hi * 100
                )
            )
    return eqm


def _solve_year_brentq(yr, pct_target, alpha_a, alpha_k, sigma,
                       phi_lo, phi_hi, phi_0_val):
    """Invert pct_neg -> phi for a single year via brentq. Top-level for pickling."""
    if yr == 1980:
        return (yr, phi_0_val)

    def _pct_resid(phi_try):
        eqm = _solve_eqm(phi_try, alpha_a, alpha_k, sigma,
                         fixed_cost=0.0, max_nfev=500, vf_maxit=200)
        if eqm is None:
            return 1e3
        pct_neg = pct_negative(m_grid, k_grid, z_grid, eqm)
        print("  [{}] phi={:.4f} -> pct_neg={:.4f}%, target={:.4f}".format(
            yr, phi_try, pct_neg, pct_target), flush=True)
        return pct_neg - pct_target

    try:
        phi_t = brentq(_pct_resid, phi_lo, phi_hi, xtol=1e-6, maxiter=50)
        print("  phi_{} = {:.6f}".format(yr, phi_t), flush=True)
    except ValueError as e:
        print("  [WARN] brentq failed for year {}: {}  — using phi_0 = {:.4f}".format(
            yr, e, phi_0_val), flush=True)
        phi_t = phi_0_val

    return (yr, phi_t)


# -----------------------------------------------------------------------------
# Calibration: (alpha_a, alpha_k)
#   phi_0 = 0  (normalization)
#   sigma = sigma_fixed  (from structural_parameters, not calibrated)
#   Targets:
#     median inv_ratio  =  med_capx_sale
#     pct_neg           =  neg_ebitda_base_pct
# -----------------------------------------------------------------------------

_BOUNDS_LOW  = [0.1, 0.1]   # alpha_a, alpha_k
_BOUNDS_HIGH = [0.9, 0.9]

def obj_base(x, max_nfev=30, vf_maxit=200):
    alpha_a_try, alpha_k_try = float(x[0]), float(x[1])

    # bounds check (needed for unconstrained solvers like Nelder-Mead)
    if not (_BOUNDS_LOW[0] <= alpha_a_try <= _BOUNDS_HIGH[0] and
            _BOUNDS_LOW[1] <= alpha_k_try <= _BOUNDS_HIGH[1]):
        return np.full(2, 1e3)

    eqm = _solve_eqm(phi_0, alpha_a_try, alpha_k_try, sigma_fixed,
                     fixed_cost=0.0, max_nfev=max_nfev, vf_maxit=vf_maxit)
    if eqm is None:
        return np.full(2, 1e3)

    inv_med = median_inv_ratio(m_grid, k_grid, z_grid, eqm)
    pct     = pct_negative(m_grid, k_grid, z_grid, eqm)

    res = np.array([
        (inv_med - med_capx_sale) / med_capx_sale,  # relative residual for median inv_ratio
        (pct     - neg_ebitda_base_pct) / neg_ebitda_base_pct,  # relative residual for pct_neg
    ])
    if not np.all(np.isfinite(res)):
        return np.full(2, 1e3)

    print(
        "aa={:.4f} ak={:.4f} | "
        "inv={:.4f} pct={:.2f}% | "
        "||res||={:.6f}".format(
            alpha_a_try, alpha_k_try,
            inv_med, pct,
            np.linalg.norm(res))
    )
    import sys; sys.stdout.flush()
    return res


if __name__ == '__main__':
    multiprocessing.set_start_method('fork', force=True)

    # -------------------------------------------------------------------------
    # Phase 1: Solve for alpha_a, alpha_k  (2 params, 2 moments)
    # -------------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("Phase 1: Solve for alpha_a, alpha_k  (2 params, 2 moments)")
    print("  phi_0  = {:.4f}  (normalized)".format(phi_0))
    print("  sigma  = {:.6f}  (fixed from structural_parameters)".format(sigma_fixed))
    print("  Targets: capx/rev={:.4f}  pct_neg@1980={:.2f}%".format(
        med_capx_sale, neg_ebitda_base_pct))
    print("=" * 60)

    def obj_base_scalar(x):
        res = obj_base(x)
        return float(np.sum(res**2))

    # Differential evolution (global optimization)
    result = differential_evolution(
        obj_base_scalar,
        bounds=list(zip(_BOUNDS_LOW, _BOUNDS_HIGH)),
        workers=-1,
        seed=42,
        tol=1e-6,
        atol=1e-6,
        maxiter=200,
        popsize=10,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=False,
        updating='deferred',
        disp=True
    )

    alpha_a_cal = float(result.x[0])
    alpha_k_cal = float(result.x[1])

    # -------------------------------------------------------------------------
    # Post-calibration diagnostics (base period)
    # -------------------------------------------------------------------------

    eqm_base = _solve_eqm(phi_0, alpha_a_cal, alpha_k_cal, sigma_fixed, fixed_cost=0.0)
    if eqm_base is not None:
        inv_med_cal = median_inv_ratio(m_grid, k_grid, z_grid, eqm_base)
        pct_cal_val = pct_negative(m_grid, k_grid, z_grid, eqm_base)
        print("\nPost-calibration diagnostics (base period):")
        print("  capx/rev: data={:.4f},  model={:.4f}".format(med_capx_sale,       inv_med_cal))
        print("  pct_neg:  data={:.2f}%, model={:.2f}%".format(neg_ebitda_base_pct, pct_cal_val))

    # -------------------------------------------------------------------------
    # Save base calibration
    # -------------------------------------------------------------------------

    out_arr = np.array([[alpha_a_cal, alpha_k_cal]])
    np.savetxt(
        out_path,
        out_arr,
        delimiter=",",
        header="alpha_a,alpha_k",
        comments="",
    )
    print("\nBase calibration saved to {}".format(out_path))

    # -------------------------------------------------------------------------
    # Phi path inversion: find phi_t for each year to match pct_neg_t
    # -------------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("Phi path inversion: year-by-year brentq on pct_neg")
    print("=" * 60)

    # Wide bounds for phi brentq search
    phi_lo_global = 0
    phi_hi_global = 0.75

    n_workers = multiprocessing.cpu_count()
    args_list = [
        (yr, pct, alpha_a_cal, alpha_k_cal, sigma_fixed,
         phi_lo_global, phi_hi_global, phi_0)
        for yr, pct in zip(years, pct_neg_data)
    ]
    print("Running phi-path inversion on {} cores ({} years)...".format(
        n_workers, len(args_list)))
    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.starmap(_solve_year_brentq, args_list)
    phi_path = sorted(results, key=lambda x: x[0])

    # -------------------------------------------------------------------------
    # Save phi path
    # -------------------------------------------------------------------------

    phi_df = pd.DataFrame(phi_path, columns=["year", "phi"])
    phi_path_out = os.path.join(_DIR, "phi_path.csv")
    phi_df.to_csv(phi_path_out, index=False)
    print("\nPhi path saved to {}".format(phi_path_out))
    print(phi_df.to_string(index=False))
