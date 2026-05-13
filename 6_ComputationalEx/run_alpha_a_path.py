"""
run_alpha_a_path.py
-------------------
Scan steady-state equilibria over 20 evenly-spaced alpha_a values in [0.2, 0.6],
with phi=0 and all other parameters fixed.  For each grid point, record
pct_neg, W, c_agg, P_M, and convergence info.

The resulting pct_neg vs. alpha_a curve reveals the shape of the mapping so that
a calibration strategy can be designed.

Usage:
  python run_alpha_a_path.py
"""

import multiprocessing
import numpy as np
import pandas as pd
import pickle
import os

from ss_solver.solve_eqm import EqmParams, solve_ss_equilibrium_least_squares
from ss_solver.solve_vf import discretize_productivity, discretize_choices
from ss_solver.integrate_dist import pct_negative

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

N_WORKERS  = 1        # None = use all CPUs; set to 1 for local single-threaded runs
N_ALPHA_A  = 20          # number of alpha_a grid points to scan
ALPHA_A_LO = 0.2
ALPHA_A_HI = 0.6

_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Grid configuration ---
import configparser as _cp
_gcfg = _cp.ConfigParser()
_gcfg.read_string("[grid]\n" + open(os.path.join(_DIR, "grid_config.txt")).read())
choice_low    = float(_gcfg["grid"]["choice_low"])
choice_high   = float(_gcfg["grid"]["choice_high"])
n_choice_grid = int(_gcfg["grid"]["n_choice_grid"])
n_prod_grid   = int(_gcfg["grid"]["n_prod_grid"])

MAIN_DIR = "/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings"
QUEST_DIR = "."
SOLVED_EQM_DIR = os.path.join(MAIN_DIR, "data", "clean")
# SOLVED_EQM_DIR = os.path.join(QUEST_DIR, "data")

# -----------------------------------------------------------------------------
# Module-level setup (inherited by worker processes via fork)
# -----------------------------------------------------------------------------

struct_params = pd.read_csv(os.path.join(_DIR, "data", "structural_parameters.csv"))

gamma_l     = struct_params["gamma_l"].iloc[0]
gamma_k     = struct_params["gamma_k"].iloc[0]
rho         = struct_params["rho"].iloc[0]
sigma_eps   = struct_params["sigma_xi"].iloc[0]
exit_rate   = struct_params["exit_rate"].iloc[0]

print("\nFixed structural parameters:", flush=True)
print("  gamma_k   = {:.6f}".format(gamma_k), flush=True)
print("  gamma_l   = {:.6f}".format(gamma_l), flush=True)
print("  rho       = {:.6f}".format(rho), flush=True)
print("  sigma_eps = {:.6f}".format(sigma_eps), flush=True)
print("  exit_rate = {:.6f}".format(exit_rate), flush=True)

m_grid = discretize_choices(choice_low, choice_high, n_choice_grid, type="exp")
k_grid = discretize_choices(choice_low, choice_high, n_choice_grid, type="exp")
z_grid, _, Pi = discretize_productivity(rho, sigma_eps, n_prod_grid)

calib_path = os.path.join(_DIR, "data", "calibrated_investment_params.csv")
if not os.path.isfile(calib_path):
    raise FileNotFoundError(
        "calibrated_investment_params.csv not found at {}. "
        "Run calibrate_investment_params.py first.".format(calib_path)
    )
_cal = pd.read_csv(calib_path)
alpha_k_cal = float(_cal["alpha_k"].iloc[0])
sigma_cal   = float(struct_params["sigma"].iloc[0])

print("\nCalibrated params: alpha_k={:.6f}  sigma={:.6f}".format(alpha_k_cal, sigma_cal), flush=True)

_params_dict = dict(
    exit_rate=exit_rate, alpha_k=alpha_k_cal, sigma=sigma_cal,
    z_k=1.0, fixed_cost=0.0, gamma_k=gamma_k, gamma_l=gamma_l,
)

# alpha_a grid (varied parameter)
alpha_a_grid = np.linspace(ALPHA_A_LO, ALPHA_A_HI, N_ALPHA_A)

# -----------------------------------------------------------------------------
# Helper: make EqmParams for a given alpha_a value
# -----------------------------------------------------------------------------

def _make_params(alpha_a_val, params_dict):
    return EqmParams(
        phi=0.0,
        entry_perc=params_dict["exit_rate"],
        sigma=params_dict["sigma"],
        alpha_a=alpha_a_val,
        alpha_k=params_dict["alpha_k"],
        z_k=params_dict["z_k"],
        fixed_cost=params_dict["fixed_cost"],
        gamma_k=params_dict["gamma_k"],
        gamma_l=params_dict["gamma_l"],
    )

# -----------------------------------------------------------------------------
# Warm-start interpolation
# -----------------------------------------------------------------------------

def interpolate_warm_starts(param_values, lo_sol, hi_sol):
    """
    Linearly interpolate (c_agg, W, P_M) between the low and high anchor
    solutions.  Returns dict mapping param_value -> np.array([c_agg, W, P_M]).
    """
    p_lo, p_hi = param_values[0], param_values[-1]
    x_lo = np.array([lo_sol["c_agg"], lo_sol["W"], lo_sol["P_M"]])
    x_hi = np.array([hi_sol["c_agg"], hi_sol["W"], hi_sol["P_M"]])
    warm_starts = {}
    for p in param_values:
        t = (p - p_lo) / (p_hi - p_lo)
        warm_starts[p] = (1.0 - t) * x_lo + t * x_hi
    return warm_starts

# -----------------------------------------------------------------------------
# Top-level worker (must be module-level for multiprocessing pickling)
# -----------------------------------------------------------------------------

def solve_single_alpha_a(args):
    """
    Solve the steady-state equilibrium for one alpha_a value.
    Returns (alpha_a_val, eqm) or (alpha_a_val, None) on failure.
    pct_neg, ls_success, res_norm are stored directly in the eqm dict.
    """
    alpha_a_val, warm_start, params_dict = args
    params = _make_params(alpha_a_val, params_dict)
    try:
        eqm = solve_ss_equilibrium_least_squares(
            m_grid, k_grid, z_grid, Pi, params,
            start=warm_start, verbose=False,
        )
    except Exception as e:
        print("  [FAIL] alpha_a={:.4f}: {}".format(alpha_a_val, e), flush=True)
        return alpha_a_val, None

    eqm["res_norm"] = float(np.linalg.norm(eqm.get("residuals", np.full(3, np.inf))))
    pct_neg = pct_negative(m_grid, k_grid, z_grid, eqm)

    print(
        "  alpha_a={:.4f}: W={:.4f}  c={:.4f}  P_M={:.4f}  "
        "pct_neg={:.2f}%  res={:.2e}  ok={}".format(
            alpha_a_val, eqm["W"], eqm["c_agg"], eqm["P_M"],
            pct_neg, eqm["res_norm"], eqm["ls_success"],
        ),
        flush=True,
    )
    return alpha_a_val, eqm

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

    os.makedirs(SOLVED_EQM_DIR, exist_ok=True)
    out_path = os.path.join(SOLVED_EQM_DIR, "eqm_alpha_a_path.pkl")

    print("\n" + "=" * 60)
    print("Alpha_a scan: {} points in [{}, {}]".format(N_ALPHA_A, ALPHA_A_LO, ALPHA_A_HI))
    print("  phi=0, alpha_k={:.4f}, sigma={:.4f}".format(alpha_k_cal, sigma_cal))
    print("=" * 60)

    # ---- Numba JIT warmup (BEFORE Pool creation) ----
    print("\nNumba JIT warmup — compiling @njit functions on tiny 3x3x3 grid...")
    _tiny_m  = m_grid[:3].copy()
    _tiny_k  = k_grid[:3].copy()
    _tiny_z  = z_grid[:3].copy()
    _tiny_Pi = Pi[:3, :3].copy()
    _tiny_Pi = _tiny_Pi / _tiny_Pi.sum(axis=1, keepdims=True)
    _warmup_params = _make_params(alpha_a_grid[0], _params_dict)
    try:
        solve_ss_equilibrium_least_squares(
            _tiny_m, _tiny_k, _tiny_z, _tiny_Pi, _warmup_params,
            start=np.array([1.0, 1.0, 1.0]),
            verbose=False, max_nfev=3, vf_maxit=2,
        )
    except Exception:
        pass
    print("Numba JIT warmup complete.\n")

    # ---- Solve anchor equilibria serially ----
    print("Solving anchor: alpha_a_lo = {:.4f}...".format(alpha_a_grid[0]))
    lo_sol = solve_ss_equilibrium_least_squares(
        m_grid, k_grid, z_grid, Pi, _make_params(alpha_a_grid[0], _params_dict),
        start=np.array([1.0, 1.0, 1.0]), verbose=False,
    )
    print("  lo anchor: W={:.4f}  c={:.4f}  P_M={:.4f}".format(
        lo_sol["W"], lo_sol["c_agg"], lo_sol["P_M"]))

    print("Solving anchor: alpha_a_hi = {:.4f}...".format(alpha_a_grid[-1]))
    hi_sol = solve_ss_equilibrium_least_squares(
        m_grid, k_grid, z_grid, Pi, _make_params(alpha_a_grid[-1], _params_dict),
        start=np.array([lo_sol["c_agg"], lo_sol["W"], lo_sol["P_M"]]), verbose=False,
    )
    print("  hi anchor: W={:.4f}  c={:.4f}  P_M={:.4f}\n".format(
        hi_sol["W"], hi_sol["c_agg"], hi_sol["P_M"]))

    # ---- Interpolate warm starts ----
    warm_starts = interpolate_warm_starts(alpha_a_grid, lo_sol, hi_sol)

    # ---- Build task args ----
    task_args = [
        (alpha_a_grid[i], warm_starts[alpha_a_grid[i]], _params_dict)
        for i in range(N_ALPHA_A)
    ]

    # ---- Parallel solve ----
    n_workers = min(N_ALPHA_A, N_WORKERS or os.cpu_count())
    print("Launching Pool with {} workers for {} alpha_a values...\n".format(
        n_workers, N_ALPHA_A))
    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.map(solve_single_alpha_a, task_args)

    # ---- Save output ----
    eqms_all = {val: eqm for val, eqm in results if eqm is not None}
    failed = [val for val, eqm in results if eqm is None]
    if failed:
        print("\nWARNING: {} points failed to converge: {}".format(
            len(failed), ["{:.4f}".format(s) for s in failed]))

    with open(out_path, "wb") as f:
        pickle.dump(eqms_all, f)

    print("\nSaved {} equilibria → {}".format(len(eqms_all), out_path))
    print("\n{:>10}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}".format(
        "alpha_a", "W", "c_agg", "P_M", "pct_neg", "ok"))
    for val, eqm in sorted(eqms_all.items()):
        print("  {:8.4f}  {:8.4f}  {:8.4f}  {:8.4f}  {:7.2f}%  {}".format(
            val,
            eqm["W"], eqm["c_agg"], eqm["P_M"],
            eqm.get("pct_neg", float("nan")),
            eqm["ls_success"],
        ))
