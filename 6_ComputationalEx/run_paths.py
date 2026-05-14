"""
run_paths.py
------------
Scan steady-state equilibria along parameter paths.

To control which paths are run, comment/uncomment entries in ACTIVE_PATHS.
Configure ranges and grid sizes in PATH_CONFIG.

  "phi_path"  — empirical year-by-year phi series (phi_path.csv); output is year-keyed
  "phi_alt"   — uniform phi grid
  "alpha_a"   — uniform alpha_a grid  (sigma held fixed)
  "beta"      — uniform beta grid
  "sigma"     — uniform sigma grid
  "sigma_eps" — uniform sigma_eps grid (z_grid / Pi rebuilt per trial)

Usage:
  python run_paths.py
"""

import multiprocessing
import numpy as np
import pandas as pd
import pickle
import os
import configparser as _cp

from ss_solver.solve_eqm import EqmParams, solve_ss_equilibrium_least_squares
from ss_solver.solve_vf import discretize_productivity, discretize_choices
from ss_solver.integrate_dist import pct_negative

# =============================================================================
# ACTIVE PATHS — comment out any you don't want to run
# =============================================================================

ACTIVE_PATHS = [
    "phi_path",    # empirical phi time-series (phi_path.csv, year-keyed output)
    "phi_alt",     # phi grid scan [-0.5, 0.5]
    "alpha_a",     # alpha_a grid scan [0.2, 0.6]
    "beta",        # beta grid scan [0.95, 0.99]
    "sigma",       # sigma grid scan [1.5, 15.0]
    # "sigma_eps",   # sigma_eps grid scan [0.01, 1.0]
]

# =============================================================================
# GRID SCAN CONFIGURATION  (lo, hi, n_points)
# phi_path reads from phi_path.csv — no range needed
# =============================================================================

PATH_CONFIG = {
    "phi_alt"   : (-0.5,  0.5,  20),
    "alpha_a"   : ( 0.2,  0.6,  20),
    "beta"      : (0.95, 0.99,  20),
    "sigma"     : ( 1.5, 15.0,  20),
    "sigma_eps" : (0.01,  1.0,  20),
}

# =============================================================================
# PARALLELISM / OUTPUT
# =============================================================================

N_WORKERS = 1   # None = use all CPUs; 1 = single-threaded

MAIN_DIR  = "/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings"
QUEST_DIR = "."
# SOLVED_EQM_DIR = os.path.join(MAIN_DIR, "data", "clean")
SOLVED_EQM_DIR = os.path.join(QUEST_DIR, "data")

# =============================================================================
# Shared setup — runs once; inherited by worker processes via fork
# =============================================================================

_DIR = os.path.dirname(os.path.abspath(__file__))

_gcfg = _cp.ConfigParser()
_gcfg.read_string("[grid]\n" + open(os.path.join(_DIR, "grid_config.txt")).read())
choice_low    = float(_gcfg["grid"]["choice_low"])
choice_high   = float(_gcfg["grid"]["choice_high"])
n_choice_grid = int(_gcfg["grid"]["n_choice_grid"])
n_prod_grid   = int(_gcfg["grid"]["n_prod_grid"])

struct_params = pd.read_csv(os.path.join(_DIR, "data", "structural_parameters.csv"))
gamma_l      = struct_params["gamma_l"].iloc[0]
gamma_k      = struct_params["gamma_k"].iloc[0]
rho          = struct_params["rho"].iloc[0]
sigma_eps_0  = struct_params["sigma_xi"].iloc[0]
sigma_fixed  = struct_params["sigma"].iloc[0]
exit_rate    = struct_params["exit_rate"].iloc[0]

print("\nFixed structural parameters:", flush=True)
for k, v in [("gamma_k", gamma_k), ("gamma_l", gamma_l), ("rho", rho),
             ("sigma_eps_0", sigma_eps_0), ("sigma", sigma_fixed), ("exit_rate", exit_rate)]:
    print("  {:12s} = {:.6f}".format(k, v), flush=True)

m_grid = discretize_choices(choice_low, choice_high, n_choice_grid, type="exp")
k_grid = discretize_choices(choice_low, choice_high, n_choice_grid, type="exp")
z_grid, _, Pi = discretize_productivity(rho, sigma_eps_0, n_prod_grid)

calib_path = os.path.join(_DIR, "data", "calibrated_investment_params.csv")
if not os.path.isfile(calib_path):
    raise FileNotFoundError(
        "calibrated_investment_params.csv not found at {}. "
        "Run calibrate_investment_params.py first.".format(calib_path)
    )
_cal        = pd.read_csv(calib_path)
alpha_a_cal = float(_cal["alpha_a"].iloc[0])
alpha_k_cal = float(_cal["alpha_k"].iloc[0])
print("\nCalibrated params: alpha_a={:.6f}  alpha_k={:.6f}".format(alpha_a_cal, alpha_k_cal), flush=True)

_BASE = dict(
    exit_rate=exit_rate, sigma=sigma_fixed,
    alpha_a=alpha_a_cal, alpha_k=alpha_k_cal,
    z_k=1.0, fixed_cost=0.0,
    gamma_k=gamma_k, gamma_l=gamma_l, rho=rho,
)

# =============================================================================
# Shared helpers
# =============================================================================

def _make_eqm_params(phi, p):
    kw = dict(phi=phi, entry_perc=p["exit_rate"], sigma=p["sigma"],
              alpha_a=p["alpha_a"], alpha_k=p["alpha_k"],
              z_k=p["z_k"], fixed_cost=p["fixed_cost"],
              gamma_k=p["gamma_k"], gamma_l=p["gamma_l"])
    if "beta" in p:
        kw["beta"] = p["beta"]
    return EqmParams(**kw)


def _interp_warm_starts(vals, lo_sol, hi_sol):
    p0, pT = vals[0], vals[-1]
    x0 = np.array([lo_sol["c_agg"], lo_sol["W"], lo_sol["P_M"]])
    xT = np.array([hi_sol["c_agg"], hi_sol["W"], hi_sol["P_M"]])
    return {p: (1 - (p - p0) / (pT - p0)) * x0 + ((p - p0) / (pT - p0)) * xT for p in vals}


def _numba_warmup(params):
    tm, tk, tz = m_grid[:3].copy(), k_grid[:3].copy(), z_grid[:3].copy()
    tPi = Pi[:3, :3].copy(); tPi /= tPi.sum(axis=1, keepdims=True)
    print("Numba JIT warmup...", flush=True)
    try:
        solve_ss_equilibrium_least_squares(tm, tk, tz, tPi, params,
            start=np.array([1.0, 1.0, 1.0]), verbose=False, max_nfev=3, vf_maxit=2)
    except Exception:
        pass
    print("Warmup complete.\n", flush=True)

# =============================================================================
# Worker (module-level for multiprocessing pickling)
# =============================================================================

def _worker(args):
    """
    Generic worker. For sigma_eps, rebuilds z_grid/Pi from params_dict["rho"]
    and the varied value passed as param_val; param_label=="sigma_eps" triggers this.
    For phi_path, key is an integer year; otherwise key==param_val.
    """
    param_val, phi, warm_start, params_dict, param_label, key = args

    if param_label == "sigma_eps":
        z_t, _, Pi_t = discretize_productivity(params_dict["rho"], param_val, n_prod_grid)
    else:
        z_t, Pi_t = z_grid, Pi

    params = _make_eqm_params(phi, params_dict)
    try:
        eqm = solve_ss_equilibrium_least_squares(
            m_grid, k_grid, z_t, Pi_t, params, start=warm_start, verbose=False)
    except Exception as e:
        print("  [FAIL] {}={:.4f}: {}".format(param_label, param_val, e), flush=True)
        return key, None

    eqm["res_norm"] = float(np.linalg.norm(eqm.get("residuals", np.full(3, np.inf))))
    pct_neg = pct_negative(m_grid, k_grid, z_t, eqm)
    print("  {}={:.4f}: W={:.4f}  c={:.4f}  P_M={:.4f}  pct_neg={:.2f}%  res={:.2e}  ok={}".format(
        param_label, param_val,
        eqm["W"], eqm["c_agg"], eqm["P_M"],
        pct_neg, eqm["res_norm"], eqm["ls_success"]), flush=True)
    return key, eqm

# =============================================================================
# run_path — single entry point for all path types
# =============================================================================

def run_path(path_name, n_workers):
    """
    Scan equilibria for path_name. Reads range from PATH_CONFIG (except phi_path).
    Saves output to SOLVED_EQM_DIR/eqm_{path_name}.pkl.
    """
    print("\n" + "=" * 60)
    print("Running: {}".format(path_name))
    print("=" * 60)

    params_dict = dict(_BASE)

    # --- Build parameter grid and task args ---
    if path_name == "phi_path":
        phi_df = pd.read_csv(os.path.join(_DIR, "data", "phi_path.csv"))
        years  = phi_df["year"].to_numpy(dtype=int)
        vals   = phi_df["phi"].to_numpy()
        print("\nPhi path:")
        for yr, phi in zip(years, vals):
            print("  Year {}: phi = {:.4f}".format(yr, phi))
        print()
        param_label = "phi"
        keys = years
    else:
        lo, hi, n = PATH_CONFIG[path_name]
        vals = np.linspace(lo, hi, n)
        param_label = {"phi_alt": "phi", "alpha_a": "alpha_a",
                       "beta": "beta", "sigma": "sigma",
                       "sigma_eps": "sigma_eps"}[path_name]
        keys = vals
        print("{} points in [{}, {}]\n".format(n, lo, hi))

    _numba_warmup(_make_eqm_params(vals[0] if param_label == "phi" else 0.0, params_dict))

    # --- Build per-trial params dicts and anchor params ---
    def _trial_params(val):
        d = dict(params_dict)
        if param_label not in ("phi", "sigma_eps"):
            d[param_label] = val
        return d

    def _anchor_params(val):
        phi = val if param_label == "phi" else 0.0
        return _make_eqm_params(phi, _trial_params(val))

    def _anchor_grids(val):
        if param_label == "sigma_eps":
            return discretize_productivity(rho, val, n_prod_grid)[::2]  # z, Pi
        return z_grid, Pi

    # --- Solve anchor equilibria ---
    z_lo, Pi_lo = _anchor_grids(vals[0])
    print("Solving lo anchor ({})={:.4f}...".format(param_label, vals[0]))
    lo_sol = solve_ss_equilibrium_least_squares(
        m_grid, k_grid, z_lo, Pi_lo, _anchor_params(vals[0]),
        start=np.array([1.0, 1.0, 1.0]), verbose=False)
    print("  W={:.4f}  c={:.4f}  P_M={:.4f}".format(lo_sol["W"], lo_sol["c_agg"], lo_sol["P_M"]))

    z_hi, Pi_hi = _anchor_grids(vals[-1])
    print("Solving hi anchor ({})={:.4f}...".format(param_label, vals[-1]))
    hi_sol = solve_ss_equilibrium_least_squares(
        m_grid, k_grid, z_hi, Pi_hi, _anchor_params(vals[-1]),
        start=np.array([lo_sol["c_agg"], lo_sol["W"], lo_sol["P_M"]]), verbose=False)
    print("  W={:.4f}  c={:.4f}  P_M={:.4f}\n".format(hi_sol["W"], hi_sol["c_agg"], hi_sol["P_M"]))

    warm_starts = _interp_warm_starts(vals, lo_sol, hi_sol)

    # --- Build task args ---
    task_args = []
    for val, key in zip(vals, keys):
        phi = val if param_label == "phi" else 0.0
        task_args.append((val, phi, warm_starts[val], _trial_params(val), param_label, key))

    # --- Parallel solve ---
    n = min(len(vals), n_workers or os.cpu_count())
    print("Launching Pool with {} workers for {} {} values...\n".format(n, len(vals), param_label))
    with multiprocessing.Pool(processes=n) as pool:
        results = pool.map(_worker, task_args)

    eqms_all = {k: eqm for k, eqm in results if eqm is not None}
    failed   = [k for k, eqm in results if eqm is None]
    if failed:
        print("\nWARNING: {} points failed: {}".format(len(failed), failed))

    out_path = os.path.join(SOLVED_EQM_DIR, "eqm_{}.pkl".format(path_name))
    with open(out_path, "wb") as f:
        pickle.dump(eqms_all, f)
    print("\nSaved {} equilibria → {}".format(len(eqms_all), out_path))

    col = max(len(param_label), 10)
    print("\n{:{w}}  {:>8}  {:>8}  {:>8}  {:>8}  {:>6}".format(
        param_label, "W", "c_agg", "P_M", "pct_neg", "ok", w=col))
    for k, eqm in sorted(eqms_all.items()):
        print("  {:{w}}  {:8.4f}  {:8.4f}  {:8.4f}  {:7.2f}%  {}".format(
            k if isinstance(k, int) else "{:.4f}".format(k),
            eqm["W"], eqm["c_agg"], eqm["P_M"],
            eqm.get("pct_neg", float("nan")), eqm["ls_success"], w=col))

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    os.makedirs(SOLVED_EQM_DIR, exist_ok=True)

    for path_name in ACTIVE_PATHS:
        run_path(path_name, N_WORKERS)
