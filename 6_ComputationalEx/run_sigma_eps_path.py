"""
run_sigma_eps_path.py
---------------------
Scan steady-state equilibria over 20 evenly-spaced sigma_eps values in [0.01, 2.0],
with phi=0 and all other parameters fixed.  For each grid point, the z_grid
and Pi are rebuilt via discretize_productivity (sigma_eps changes the
Rouwenhorst grid).  For each point, record pct_neg, W, c_agg, P_M, and
convergence info.

The resulting pct_neg vs. sigma_eps curve reveals the shape of the mapping so
that a calibration strategy can be designed.

Usage:
  python run_sigma_eps_path.py
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

N_WORKERS     = None     # None = use all CPUs; set to 1 for local single-threaded runs
N_GRID        = 100     # grid size for m, k (reduce to e.g. 50 for fast test runs)
N_SIGMA_EPS   = 20      # number of sigma_eps grid points to scan
SIGMA_EPS_LO  = 0.01
SIGMA_EPS_HI  = 1.0

_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = "/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings"
QUEST_DIR = "./data"
# SOLVED_EQM_DIR = os.path.join(MAIN_DIR, "data", "clean")
SOLVED_EQM_DIR = os.path.join(QUEST_DIR, "data", "clean")

# -----------------------------------------------------------------------------
# Module-level setup (inherited by worker processes via fork)
# -----------------------------------------------------------------------------

struct_params = pd.read_csv(os.path.join(_DIR, "data", "structural_parameters.csv"))

gamma_l       = struct_params["gamma_l"].iloc[0]
gamma_k       = struct_params["gamma_k"].iloc[0]
rho           = struct_params["rho"].iloc[0]
sigma_fixed   = struct_params["sigma"].iloc[0]    # fixed markup parameter
sigma_eps_0   = struct_params["sigma_xi"].iloc[0] # base-period value (for warmup)
exit_rate     = struct_params["exit_rate"].iloc[0]

print("\nFixed structural parameters:")
print("  gamma_k     = {:.6f}".format(gamma_k))
print("  gamma_l     = {:.6f}".format(gamma_l))
print("  rho         = {:.6f}".format(rho))
print("  sigma       = {:.6f}  (fixed markup, from ACF)".format(sigma_fixed))
print("  sigma_eps_0 = {:.6f}  (base period, for JIT warmup)".format(sigma_eps_0))
print("  exit_rate   = {:.6f}".format(exit_rate))

m_grid = discretize_choices(1e-3, 5, N_GRID, type="exp")
k_grid = discretize_choices(1e-3, 5, N_GRID, type="exp")
# z_grid is NOT built module-level here — it is rebuilt per sigma_eps trial.

calib_path = os.path.join(_DIR, "data", "calibrated_investment_params.csv")
if not os.path.isfile(calib_path):
    raise FileNotFoundError(
        "calibrated_investment_params.csv not found at {}. "
        "Run calibrate_investment_params.py first.".format(calib_path)
    )
_cal = pd.read_csv(calib_path)
alpha_a_cal = float(_cal["alpha_a"].iloc[0])
alpha_k_cal = float(_cal["alpha_k"].iloc[0])

print("\nCalibrated params: alpha_a={:.6f}  alpha_k={:.6f}".format(alpha_a_cal, alpha_k_cal))

_params_dict = dict(
    exit_rate=exit_rate, sigma=sigma_fixed, alpha_a=alpha_a_cal, alpha_k=alpha_k_cal,
    z_k=1.0, fixed_cost=0.0, gamma_k=gamma_k, gamma_l=gamma_l, rho=rho,
)

# sigma_eps grid (varied parameter)
sigma_eps_grid = np.linspace(SIGMA_EPS_LO, SIGMA_EPS_HI, N_SIGMA_EPS)

# -----------------------------------------------------------------------------
# Helper: make EqmParams for a given sigma_eps value
# -----------------------------------------------------------------------------

def _make_params(sigma_eps_val, params_dict):
    return EqmParams(
        phi=0.0,
        entry_perc=params_dict["exit_rate"],
        sigma=params_dict["sigma"],
        alpha_a=params_dict["alpha_a"],
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

def solve_single_sigma_eps(args):
    """
    Solve the steady-state equilibrium for one sigma_eps value.
    Rebuilds z_grid and Pi per trial (sigma_eps changes the Rouwenhorst grid).
    Returns (sigma_eps_val, eqm) or (sigma_eps_val, None) on failure.
    pct_neg, ls_success, res_norm are stored directly in the eqm dict.
    """
    sigma_eps_val, warm_start, params_dict = args

    # Rebuild z_grid and Pi for this sigma_eps value
    z_grid_t, _, Pi_t = discretize_productivity(params_dict["rho"], sigma_eps_val, 15)

    params = _make_params(sigma_eps_val, params_dict)
    try:
        eqm = solve_ss_equilibrium_least_squares(
            m_grid, k_grid, z_grid_t, Pi_t, params,
            start=warm_start, verbose=False,
        )
    except Exception as e:
        print("  [FAIL] sigma_eps={:.4f}: {}".format(sigma_eps_val, e), flush=True)
        return sigma_eps_val, None

    eqm["res_norm"] = float(np.linalg.norm(eqm.get("residuals", np.full(3, np.inf))))
    pct_neg = pct_negative(m_grid, k_grid, z_grid_t, eqm)

    print(
        "  sigma_eps={:.4f}: W={:.4f}  c={:.4f}  P_M={:.4f}  "
        "pct_neg={:.2f}%  res={:.2e}  ok={}".format(
            sigma_eps_val, eqm["W"], eqm["c_agg"], eqm["P_M"],
            pct_neg, eqm["res_norm"], eqm["ls_success"],
        ),
        flush=True,
    )
    return sigma_eps_val, eqm

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

    os.makedirs(SOLVED_EQM_DIR, exist_ok=True)
    out_path = os.path.join(SOLVED_EQM_DIR, "eqm_sigma_eps_path.pkl")

    print("\n" + "=" * 60)
    print("Sigma_eps scan: {} points in [{}, {}]".format(
        N_SIGMA_EPS, SIGMA_EPS_LO, SIGMA_EPS_HI))
    print("  phi=0, sigma={:.4f}, alpha_a={:.4f}, alpha_k={:.4f}".format(
        sigma_fixed, alpha_a_cal, alpha_k_cal))
    print("=" * 60)

    # ---- Numba JIT warmup (BEFORE Pool creation) ----
    # Use base-period sigma_eps_0 to build a valid z_grid for compilation.
    print("\nNumba JIT warmup — compiling @njit functions on tiny 3x3x3 grid...")
    _tiny_m  = m_grid[:3].copy()
    _tiny_k  = k_grid[:3].copy()
    _tiny_z_full, _, _tiny_Pi_full = discretize_productivity(rho, sigma_eps_0, 15)
    _tiny_z  = _tiny_z_full[:3].copy()
    _tiny_Pi = _tiny_Pi_full[:3, :3].copy()
    _tiny_Pi = _tiny_Pi / _tiny_Pi.sum(axis=1, keepdims=True)
    _warmup_params = _make_params(sigma_eps_grid[0], _params_dict)
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
    _z_lo, _, _Pi_lo = discretize_productivity(rho, sigma_eps_grid[0], 15)
    print("Solving anchor: sigma_eps_lo = {:.4f}...".format(sigma_eps_grid[0]))
    lo_sol = solve_ss_equilibrium_least_squares(
        m_grid, k_grid, _z_lo, _Pi_lo,
        _make_params(sigma_eps_grid[0], _params_dict),
        start=np.array([1.0, 1.0, 1.0]), verbose=False,
    )
    print("  lo anchor: W={:.4f}  c={:.4f}  P_M={:.4f}".format(
        lo_sol["W"], lo_sol["c_agg"], lo_sol["P_M"]))

    _z_hi, _, _Pi_hi = discretize_productivity(rho, sigma_eps_grid[-1], 15)
    print("Solving anchor: sigma_eps_hi = {:.4f}...".format(sigma_eps_grid[-1]))
    hi_sol = solve_ss_equilibrium_least_squares(
        m_grid, k_grid, _z_hi, _Pi_hi,
        _make_params(sigma_eps_grid[-1], _params_dict),
        start=np.array([lo_sol["c_agg"], lo_sol["W"], lo_sol["P_M"]]), verbose=False,
    )
    print("  hi anchor: W={:.4f}  c={:.4f}  P_M={:.4f}\n".format(
        hi_sol["W"], hi_sol["c_agg"], hi_sol["P_M"]))

    # ---- Interpolate warm starts ----
    warm_starts = interpolate_warm_starts(sigma_eps_grid, lo_sol, hi_sol)

    # ---- Build task args ----
    task_args = [
        (sigma_eps_grid[i], warm_starts[sigma_eps_grid[i]], _params_dict)
        for i in range(N_SIGMA_EPS)
    ]

    # ---- Parallel solve ----
    n_workers = min(N_SIGMA_EPS, N_WORKERS or os.cpu_count() or 20)
    print("Launching Pool with {} workers for {} sigma_eps values...\n".format(
        n_workers, N_SIGMA_EPS))
    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.map(solve_single_sigma_eps, task_args)

    # ---- Save output ----
    eqms_all = {val: eqm for val, eqm in results if eqm is not None}
    failed = [val for val, eqm in results if eqm is None]
    if failed:
        print("\nWARNING: {} points failed to converge: {}".format(
            len(failed), ["{:.4f}".format(s) for s in failed]))

    with open(out_path, "wb") as f:
        pickle.dump(eqms_all, f)

    print("\nSaved {} equilibria → {}".format(len(eqms_all), out_path))
    print("\n{:>12}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}".format(
        "sigma_eps", "W", "c_agg", "P_M", "pct_neg", "ok"))
    for val, eqm in sorted(eqms_all.items()):
        print("  {:10.4f}  {:8.4f}  {:8.4f}  {:8.4f}  {:7.2f}%  {}".format(
            val,
            eqm["W"], eqm["c_agg"], eqm["P_M"],
            eqm.get("pct_neg", float("nan")),
            eqm["ls_success"],
        ))
