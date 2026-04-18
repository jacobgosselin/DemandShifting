from ss_solver.solve_eqm import EqmParams, solve_ss_equilibrium_least_squares
from ss_solver.integrate_dist import pct_negative
import multiprocessing
import numpy as np
import pandas as pd
import os
from scipy.optimize import brentq


# -----------------------------------------------------------------------------
# Paths and data loading
# -----------------------------------------------------------------------------

_DIR = os.path.dirname(__file__)

struct_params = pd.read_csv(os.path.join(_DIR, "data", "structural_parameters.csv"))

# Fixed structural parameters (from ACF estimation and markup inversion)
gamma_l     = struct_params["gamma_l"].iloc[0]
gamma_k     = struct_params["gamma_k"].iloc[0]
rho         = struct_params["rho"].iloc[0]
sigma_eps   = struct_params["sigma_xi"].iloc[0]
sigma_fixed = struct_params["sigma"].iloc[0]
exit_rate   = struct_params["exit_rate"].iloc[0]

print("\nFixed structural parameters:")
print("  gamma_k     = {:.6f}".format(gamma_k))
print("  gamma_l     = {:.6f}".format(gamma_l))
print("  rho         = {:.6f}".format(rho))
print("  sigma_eps   = {:.6f}".format(sigma_eps))
print("  sigma_fixed = {:.6f}  (fixed = mu/(mu-1) from ACF markup)".format(sigma_fixed))
print("  exit_rate   = {:.6f}".format(exit_rate))

# Load calibrated investment params
cal_path = os.path.join(_DIR, "data", "calibrated_investment_params.csv")
if not os.path.isfile(cal_path):
    raise FileNotFoundError(
        "calibrated_investment_params.csv not found at {}. "
        "Run calibrate_investment_params.py first.".format(cal_path)
    )
_cal = pd.read_csv(cal_path)
alpha_a_cal = float(_cal["alpha_a"].iloc[0])
alpha_k_cal = float(_cal["alpha_k"].iloc[0])

print("\nLoaded calibrated params from {}:".format(cal_path))
print("  alpha_a = {:.6f}".format(alpha_a_cal))
print("  alpha_k = {:.6f}".format(alpha_k_cal))

# Year-by-year pct_neg targets
pct_neg_df   = pd.read_csv(os.path.join(_DIR, "data", "pct_neg_byyear.csv"))
years        = pct_neg_df["year"].values.astype(int)
pct_neg_data = pct_neg_df["pct_neg"].values   # already in percent (0-100)

print("\nLoaded pct_neg by year: {} years ({}-{})".format(
    len(years), years[0], years[-1]))

# Normalization
phi_0 = 0.0
print("\nNormalization: phi_0 = {:.4f}  (base period)".format(phi_0))

# -----------------------------------------------------------------------------
# Grids
# -----------------------------------------------------------------------------

from ss_solver.solve_vf import discretize_productivity, discretize_choices

m_grid = discretize_choices(1e-3, 5, 100, type="exp")
k_grid = discretize_choices(1e-3, 5, 100, type="exp")
z_grid, _, Pi = discretize_productivity(rho, sigma_eps, 15)

# -----------------------------------------------------------------------------
# Helper: solve equilibrium with convergence check
# -----------------------------------------------------------------------------

def _solve_eqm(phi_val, alpha_a, alpha_k, sigma_val,
               fixed_cost=0.0, max_nfev=1000, vf_maxit=250):
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
    return eqm


# -----------------------------------------------------------------------------
# Top-level worker function for multiprocessing
# -----------------------------------------------------------------------------

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
# Main
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    multiprocessing.set_start_method('fork', force=True)

    print("\n" + "=" * 60)
    print("Phi path inversion: year-by-year brentq on pct_neg")
    print("  alpha_a = {:.6f}  alpha_k = {:.6f}".format(alpha_a_cal, alpha_k_cal))
    print("  sigma   = {:.6f}  (fixed from structural_parameters)".format(sigma_fixed))
    print("=" * 60)

    phi_lo_global = 0
    phi_hi_global = 0.75

    n_workers = multiprocessing.cpu_count()
    # n_workers = 2

    args_list = [
        (yr, pct, alpha_a_cal, alpha_k_cal, sigma_fixed,
         phi_lo_global, phi_hi_global, phi_0)
        for yr, pct in zip(years, pct_neg_data)
    ]
    
    # ---- Numba JIT warmup (BEFORE Pool creation) ----
    print("Numba JIT warmup — compiling @njit functions on tiny 3x3x3 grid...")
    _tiny_m  = m_grid[:3].copy()
    _tiny_k  = k_grid[:3].copy()
    _tiny_z  = z_grid[:3].copy()
    _tiny_Pi = Pi[:3, :3].copy()
    _tiny_Pi = _tiny_Pi / _tiny_Pi.sum(axis=1, keepdims=True)
    _warmup_params = EqmParams(
        phi=phi_0, entry_perc=exit_rate, sigma=sigma_fixed,
        alpha_a=alpha_a_cal, alpha_k=alpha_k_cal,
        z_k=1.0, fixed_cost=0.0, gamma_k=gamma_k, gamma_l=gamma_l,
    )
    try:
        solve_ss_equilibrium_least_squares(
            _tiny_m, _tiny_k, _tiny_z, _tiny_Pi, _warmup_params,
            verbose=False, max_nfev=3, vf_maxit=2,
        )
    except Exception:
        pass
    print("Numba JIT warmup complete.\n")

    print("Running phi-path inversion on {} cores ({} years)...".format(
        n_workers, len(args_list)))
    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.starmap(_solve_year_brentq, args_list)
    phi_path = sorted(results, key=lambda x: x[0])

    phi_df = pd.DataFrame(phi_path, columns=["year", "phi"])
    phi_path_out = os.path.join(_DIR, "data", "phi_path.csv")
    phi_df.to_csv(phi_path_out, index=False)
    print("\nPhi path saved to {}".format(phi_path_out))
    print(phi_df.to_string(index=False))
