"""
solve_phi_path.py
-----------------
Solve steady-state equilibria along the empirical phi path (1980–2019).

Parallelized via multiprocessing.Pool for Quest execution.
Output: a single pickle file containing all 40 equilibrium objects.

Usage:
  python solve_phi_path.py              # solve all phi values in parallel
"""

import multiprocessing
import numpy as np
import pandas as pd
import pickle
import os

from solve_eqm import EqmParams, solve_ss_equilibrium_least_squares
from solve_vf import discretize_productivity, discretize_choices

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

N_WORKERS = 1          # None = use all CPUs; set to 1 for local single-threaded runs

_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = "/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings"
QUEST_DIR = "./"
SOLVED_EQM_DIR = os.path.join(MAIN_DIR, "data", "clean")
N_GRID = 100 # Grid size constant (reduce to e.g. 50 for fast test runs)

# -----------------------------------------------------------------------------
# Module-level setup: parameters and grids
# (Runs on import; inherited by worker processes via fork.)
# -----------------------------------------------------------------------------

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

m_grid = discretize_choices(1e-3, 5, N_GRID, type="exp")
k_grid = discretize_choices(1e-3, 5, N_GRID, type="exp")

# Load calibrated params (alpha_a, alpha_k)
fixed_cost_cal = 0.0
calib_path = os.path.join(_DIR, "calibrated_investment_params.csv")
if os.path.exists(calib_path):
    calib_df    = pd.read_csv(calib_path)
    alpha_a_cal = float(calib_df["alpha_a"].iloc[0])
    alpha_k_cal = float(calib_df["alpha_k"].iloc[0])
else:
    print("Warning: calibrated_investment_params.csv not found; using defaults.")
    alpha_a_cal, alpha_k_cal = 0.5, 0.5

print(
    f"Calibrated params: alpha_a={alpha_a_cal:.4f}, alpha_k={alpha_k_cal:.4f}, "
)
print(f"Production function (fixed): gamma_k={gamma_k:.4f}, gamma_l={gamma_l:.4f}")
z_grid, pi, Pi = discretize_productivity(rho, sigma_eps, 15)

# Load phi path from calibration (phi_path.csv)
_phi_df    = pd.read_csv(os.path.join(_DIR, "phi_path.csv"))
phi_byyear = _phi_df[["year", "phi"]].to_numpy()  # shape (T, 2)

# Params dict for passing to workers (avoids passing EqmParams across fork boundary)
_params_dict = dict(
    exit_rate=exit_rate, sigma=sigma_fixed, alpha_a=alpha_a_cal, alpha_k=alpha_k_cal,
    z_k=1.0, fixed_cost=fixed_cost_cal, gamma_k=gamma_k, gamma_l=gamma_l,
)

# -----------------------------------------------------------------------------
# Top-level functions (must be module-level for multiprocessing pickling)
# -----------------------------------------------------------------------------

def _make_eqm_params(phi, params_dict):
    """Construct EqmParams from a flat dict + phi."""
    return EqmParams(
        phi=phi,
        entry_perc=params_dict["exit_rate"],
        sigma=params_dict["sigma"],
        alpha_a=params_dict["alpha_a"],
        alpha_k=params_dict["alpha_k"],
        z_k=params_dict["z_k"],
        fixed_cost=params_dict["fixed_cost"],
        gamma_k=params_dict["gamma_k"],
        gamma_l=params_dict["gamma_l"],
    )


def interpolate_warm_starts(phi_values, phi0_sol, phiT_sol):
    """
    Linearly interpolate (c_agg, W, P_M) between the phi_0 and phi_T
    anchor solutions to provide starting points for all intermediate phi values.

    Returns dict mapping phi_val -> np.array([c_agg, W, P_M]).
    """
    phi0, phiT = phi_values[0], phi_values[-1]
    x0 = np.array([phi0_sol["c_agg"], phi0_sol["W"], phi0_sol["P_M"]])
    xT = np.array([phiT_sol["c_agg"], phiT_sol["W"], phiT_sol["P_M"]])
    warm_starts = {}
    for phi in phi_values:
        t = (phi - phi0) / (phiT - phi0)   # interpolation weight in [0, 1]
        warm_starts[phi] = (1.0 - t) * x0 + t * xT
    return warm_starts


def solve_single_phi(args):
    """
    Worker function: solve the steady-state equilibrium for one phi value.

    Accesses m_grid, k_grid, z_grid, Pi from module-level globals
    (inherited copy-on-write via fork — no redundant data copying).

    Returns (year, eqm_dict) or (year, None) on failure.
    """
    phi, year, warm_start, params_dict = args
    params = _make_eqm_params(phi, params_dict)
    try:
        eqm = solve_ss_equilibrium_least_squares(
            m_grid, k_grid, z_grid, Pi, params,
            start=warm_start, verbose=False
        )
    except Exception as e:
        print(f"  [FAIL] year={int(year)} phi={phi:.4f}: {e}", flush=True)
        return int(year), None

    pct_neg = float(np.nan)
    try:
        from integrate_dist import pct_negative as _pct_neg
        pct_neg = _pct_neg(m_grid, k_grid, z_grid, eqm)
    except Exception:
        pass

    print(
        f"  year={int(year)} phi={phi:.4f}: "
        f"W={eqm['W']:.4f}  c={eqm['c_agg']:.4f}  P_M={eqm['P_M']:.4f}  "
        f"pct_neg={pct_neg:.2f}%",
        flush=True,
    )
    return int(year), eqm


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

    os.makedirs(SOLVED_EQM_DIR, exist_ok=True)
    all_pkl_path = os.path.join(SOLVED_EQM_DIR, "eqm_phi_path.pkl")

    # ---- phi path ----
    # phi = beta * ((1-gamma_l)*sigma + gamma_l) - 1   (arbitrary-scale mapping)
    years_arr       = phi_byyear[:, 0].astype(int)
    phi_track_values = phi_byyear[:, 1] 

    print("\nPhi path:")
    for yr, phi in zip(years_arr, phi_track_values):
        print(f"  Year {yr}: phi = {phi:.4f}")
    print()

    # ---- Numba JIT warmup (BEFORE Pool creation) ----
    # Fork-based workers inherit already-compiled @njit functions from the parent.
    # Compiling here (once) prevents 20 workers from simultaneously writing to
    # the Numba cache, which causes race conditions and 20x compile overhead.
    print("Numba JIT warmup — compiling @njit functions on tiny 3x3x3 grid...")
    _tiny_m  = m_grid[:3].copy()
    _tiny_k  = k_grid[:3].copy()
    _tiny_z  = z_grid[:3].copy()
    _tiny_Pi = Pi[:3, :3].copy()
    _tiny_Pi = _tiny_Pi / _tiny_Pi.sum(axis=1, keepdims=True)  # renormalize rows
    _warmup_params = _make_eqm_params(phi_track_values[0], _params_dict)
    try:
        solve_ss_equilibrium_least_squares(
            _tiny_m, _tiny_k, _tiny_z, _tiny_Pi, _warmup_params,
            start=np.array([1.0, 1.0, 1.0]),
            verbose=False, max_nfev=3, vf_maxit=2,
        )
    except Exception:
        pass   # convergence is not expected; compilation is what matters
    print("Numba JIT warmup complete.\n")

    # ---- Solve anchor equilibria serially ----
    # phi_0 (1980) and phi_T (2019) provide endpoints for linear interpolation.
    print("Solving anchor equilibrium: phi_0 (1980)...")
    phi0_params = _make_eqm_params(phi_track_values[0], _params_dict)
    phi0_sol = solve_ss_equilibrium_least_squares(
        m_grid, k_grid, z_grid, Pi, phi0_params,
        start=np.array([1.0, 1.0, 1.0]), verbose=False,
    )
    print(f"  phi_0 anchor: W={phi0_sol['W']:.4f}  c={phi0_sol['c_agg']:.4f}  P_M={phi0_sol['P_M']:.4f}\n")

    print("Solving anchor equilibrium: phi_T (2019)...")
    phiT_params = _make_eqm_params(phi_track_values[-1], _params_dict)
    phiT_sol = solve_ss_equilibrium_least_squares(
        m_grid, k_grid, z_grid, Pi, phiT_params,
        start=np.array([phi0_sol["c_agg"], phi0_sol["W"], phi0_sol["P_M"]]),
        verbose=False,
    )
    print(f"  phi_T anchor: W={phiT_sol['W']:.4f}  c={phiT_sol['c_agg']:.4f}  P_M={phiT_sol['P_M']:.4f}\n")

    # ---- Interpolate warm starts for all 40 phi values ----
    warm_starts = interpolate_warm_starts(phi_track_values, phi0_sol, phiT_sol)

    # ---- Build task args ----
    task_args = [
        (phi_track_values[i], int(years_arr[i]), warm_starts[phi_track_values[i]], _params_dict)
        for i in range(len(phi_track_values))
    ]

    # ---- Parallel solve ----
    n_workers = min(len(phi_track_values), N_WORKERS or os.cpu_count() or 20)
    print(f"Launching Pool with {n_workers} workers for {len(phi_track_values)} phi values...\n")
    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.map(solve_single_phi, task_args)

    # ---- Assemble and save single output pickle ----
    failed = [(yr, phi) for (yr, _), phi in zip(results, phi_track_values) if results[0][1] is None]
    # correct: check the eqm part of each result
    failed = [(yr, phi_track_values[i]) for i, (yr, eqm) in enumerate(results) if eqm is None]
    if failed:
        print(f"\nWARNING: {len(failed)} equilibria failed to converge:")
        for yr, phi in failed:
            print(f"  year={yr}, phi={phi:.6f}")

    eqms_all = {yr: eqm for (yr, eqm) in results if eqm is not None}

    out = {
        "eqms":       eqms_all,                   # {year: eqm_dict}
        "phi_values": {                             # {year: phi}
            int(years_arr[i]): phi_track_values[i]
            for i in range(len(years_arr))
        },
        "years":      sorted(eqms_all.keys()),     # sorted list of int years
        "grids":      {
            "m_grid": m_grid,
            "k_grid": k_grid,
            "z_grid": z_grid,
            "Pi":     Pi,
        },
        "params": _params_dict,
    }

    with open(all_pkl_path, "wb") as f:
        pickle.dump(out, f)

    print(f"\nSaved {len(eqms_all)} equilibria → {all_pkl_path}")
