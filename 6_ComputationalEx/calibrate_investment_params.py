from solve_eqm import EqmParams, solve_ss_equilibrium_least_squares
from integrate_dist import pct_negative, median_adv_ratio, median_inv_ratio
import multiprocessing
import numpy as np
import pandas as pd
import os
from scipy.optimize import least_squares, differential_evolution, brentq


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
sigma     = struct_params["sigma"].iloc[0]
exit_rate = struct_params["exit_rate"].iloc[0]

print("\nFixed structural parameters:")
print("  gamma_k    = {:.6f}".format(gamma_k))
print("  gamma_l    = {:.6f}".format(gamma_l))
print("  rho        = {:.6f}".format(rho))
print("  sigma_eps  = {:.6f}".format(sigma_eps))
print("  sigma      = {:.6f}".format(sigma))
print("  exit_rate  = {:.6f}".format(exit_rate))

# Calibration moment targets
med_sga_sale        = struct_params["med_sga_sale"].iloc[0]
med_capx_sale       = struct_params["med_capx_sale"].iloc[0]
neg_ebitda_base_pct = struct_params["neg_ebitda_base"].iloc[0] * 100

print("\nCalibration targets (base period):")
print("  med_sga/rev  = {:.4f}".format(med_sga_sale))
print("  med_capx/rev = {:.4f}".format(med_capx_sale))
print("  pct_neg      = {:.4f}%".format(neg_ebitda_base_pct))

# Year-by-year pct_neg for phi path inversion
pct_neg_df   = pd.read_csv(os.path.join(_DIR, "pct_neg_byyear.csv"))
years        = pct_neg_df["year"].values.astype(int)
pct_neg_data = pct_neg_df["pct_neg"].values   # already in percent (0-100)

print("\nLoaded pct_neg by year: {} years ({}-{})".format(
    len(years), years[0], years[-1]))


# -----------------------------------------------------------------------------
# Grids (m and k only; z_grid is pre-computed from fixed rho/sigma_eps)
# -----------------------------------------------------------------------------

from solve_vf import discretize_productivity, discretize_choices

m_grid = discretize_choices(1e-3, 100, 100, type="exp")
k_grid = discretize_choices(1e-3, 100, 100, type="exp")

# z_grid is fixed (rho, sigma_eps not calibrated)
z_grid, _, Pi = discretize_productivity(rho, sigma_eps, 10)

# -----------------------------------------------------------------------------
# Load initial guesses from previous calibration (or use defaults)
# -----------------------------------------------------------------------------

out_path = os.path.join(_DIR, "calibrated_investment_params.csv")

alpha_a_init = 0.5
alpha_k_init = 0.5
phi_0_init   = 0.1

if os.path.isfile(out_path):
    _cal_prev = pd.read_csv(out_path)
    if "alpha_a" in _cal_prev.columns:
        alpha_a_init = float(_cal_prev["alpha_a"].iloc[0])
    if "alpha_k" in _cal_prev.columns:
        alpha_k_init = float(_cal_prev["alpha_k"].iloc[0])
    if "phi_1980" in _cal_prev.columns:
        phi_0_init = float(_cal_prev["phi_1980"].iloc[0])
    print("\nLoaded initial guesses from {}:".format(out_path))
    print("  alpha_a_init = {:.6f}".format(alpha_a_init))
    print("  alpha_k_init = {:.6f}".format(alpha_k_init))
    print("  phi_0_init   = {:.6f}".format(phi_0_init))
else:
    print("\nNo existing calibrated parameters found; using defaults.")

# -----------------------------------------------------------------------------
# Helper: solve equilibrium with convergence check
# -----------------------------------------------------------------------------

def _solve_eqm(phi_val, alpha_a, alpha_k,
               fixed_cost=0.0, warm_start=None, max_nfev=1000, vf_maxit=250):
    params = EqmParams(
        phi=phi_val,
        entry_perc=exit_rate,
        sigma=sigma,
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
            start=warm_start, max_nfev=max_nfev, vf_maxit=vf_maxit,
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
# Calibration: (alpha_a, alpha_k, phi_1980)
#   Targets:
#     median adv_ratio  =  med_sga_sale
#     median inv_ratio  =  med_capx_sale
#     pct_neg           =  neg_ebitda_base_pct
#   Phase 1 — Differential Evolution (global, gradient-free)
#   Phase 2 — least_squares refinement (local, from DE best point)
# -----------------------------------------------------------------------------

_ws_base = {'eqm': None}


def obj_base(x, max_nfev=20, vf_maxit=100):
    alpha_a_try, alpha_k_try, phi_try = (
        float(x[0]), float(x[1]), float(x[2])
    )

    warm = (np.array([_ws_base['eqm']['c_agg'], _ws_base['eqm']['W'], _ws_base['eqm']['P_M']])
            if _ws_base['eqm'] is not None else None)

    eqm = _solve_eqm(phi_try, alpha_a_try, alpha_k_try,
                     fixed_cost=0.0, warm_start=warm,
                     max_nfev=max_nfev, vf_maxit=vf_maxit)
    if eqm is None:
        return np.full(3, 1e3)
    _ws_base['eqm'] = eqm

    adv_med = median_adv_ratio(m_grid, k_grid, z_grid, eqm)
    inv_med = median_inv_ratio(m_grid, k_grid, z_grid, eqm)
    pct     = pct_negative(m_grid, k_grid, z_grid, eqm)

    res = np.array([
        (adv_med - med_sga_sale)        / med_sga_sale,
        (inv_med - med_capx_sale)       / med_capx_sale,
        (pct     - neg_ebitda_base_pct) / 100,
    ])
    if not np.all(np.isfinite(res)):
        return np.full(3, 1e3)

    print(
        "aa={:.4f} ak={:.4f} phi0={:.4f} | "
        "adv={:.4f} inv={:.4f} pct={:.2f}% | "
        "||res||={:.6f}".format(
            alpha_a_try, alpha_k_try, phi_try,
            adv_med, inv_med, pct,
            np.linalg.norm(res))
    )
    import sys; sys.stdout.flush()
    return res


_BOUNDS_LOW  = [0.1, 0.1, -0.5]   # alpha_a, alpha_k, phi_1980
_BOUNDS_HIGH = [0.9, 0.9,  1.0]

def _obj_scalar(x):
    return float(np.sum(obj_base(x) ** 2))


if __name__ == '__main__':
    multiprocessing.set_start_method('fork', force=True)

    # -------------------------------------------------------------------------
    # Initialize warm start from initial guesses
    # -------------------------------------------------------------------------

    print("\nInitializing warm start ...")
    _pre = _solve_eqm(phi_0_init, alpha_a_init, alpha_k_init,
                      fixed_cost=0.0, max_nfev=1000)
    if _pre is not None:
        _ws_base['eqm'] = _pre

    # -------------------------------------------------------------------------
    # Phase 1: Differential Evolution — global search
    # -------------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("Phase 1: Differential Evolution (global search)")
    print("  Targets: adv/rev={:.4f}  capx/rev={:.4f}  pct_neg@1980={:.2f}%".format(
        med_sga_sale, med_capx_sale, neg_ebitda_base_pct))
    print("=" * 60)

    de_result = differential_evolution(
        _obj_scalar,
        bounds=list(zip(_BOUNDS_LOW, _BOUNDS_HIGH)),
        seed=42,
        strategy="best1bin",
        maxiter=300,
        popsize=5,
        tol=1e-4,
        mutation=(0.5, 1.0),
        recombination=0.7,
        workers=-1,
        updating='deferred',
        polish=False,
        disp=True,
    )

    print("\nDE result:  success={},  cost={:.6f}".format(de_result.success, de_result.fun))
    print("  alpha_a={:.4f}  alpha_k={:.4f}  phi_1980={:.4f}".format(
        de_result.x[0], de_result.x[1], de_result.x[2]))

    # -------------------------------------------------------------------------
    # Re-seed warm start from DE best point before LS refinement
    # -------------------------------------------------------------------------

    _aa_de, _ak_de, _phi0_de = de_result.x
    print("\nRe-seeding warm start from DE solution ...")
    _pre = _solve_eqm(_phi0_de, _aa_de, _ak_de, fixed_cost=0.0, max_nfev=1000)
    if _pre is not None:
        _ws_base['eqm'] = _pre

    # -------------------------------------------------------------------------
    # Phase 2: Least Squares — local refinement from DE best point
    # -------------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("Phase 2: Least Squares refinement from DE solution")
    print("=" * 60)

    result = least_squares(
        lambda x: obj_base(x, max_nfev=500),
        x0=de_result.x,
        method="trf",
        bounds=(_BOUNDS_LOW, _BOUNDS_HIGH),
        xtol=1e-5,
        ftol=1e-8,
        gtol=1e-8,
        verbose=2,
    )

    alpha_a_cal = float(result.x[0])
    alpha_k_cal = float(result.x[1])
    phi_1980    = float(result.x[2])

    print("\nBase-period calibration result:")
    print("  success      = {}".format(result.success))
    print("  message      = {}".format(result.message))
    print("  alpha_a_cal  = {:.6f}".format(alpha_a_cal))
    print("  alpha_k_cal  = {:.6f}".format(alpha_k_cal))
    print("  phi_1980     = {:.6f}".format(phi_1980))
    print("  residuals    = {}".format(result.fun))
    print("  cost         = {:.8f}".format(result.cost))

    # -------------------------------------------------------------------------
    # Post-calibration diagnostics (base period)
    # -------------------------------------------------------------------------

    eqm_base = _solve_eqm(phi_1980, alpha_a_cal, alpha_k_cal, fixed_cost=0.0)
    if eqm_base is not None:
        adv_med_cal = median_adv_ratio(m_grid, k_grid, z_grid, eqm_base)
        inv_med_cal = median_inv_ratio(m_grid, k_grid, z_grid, eqm_base)
        pct_cal     = pct_negative(m_grid, k_grid, z_grid, eqm_base)
        print("\nPost-calibration diagnostics (base period):")
        print("  adv/rev:  data={:.4f},  model={:.4f}".format(med_sga_sale,        adv_med_cal))
        print("  capx/rev: data={:.4f},  model={:.4f}".format(med_capx_sale,       inv_med_cal))
        print("  pct_neg:  data={:.2f}%, model={:.2f}%".format(neg_ebitda_base_pct, pct_cal))

    # -------------------------------------------------------------------------
    # Save base calibration
    # -------------------------------------------------------------------------

    out_arr = np.array([[alpha_a_cal, alpha_k_cal, phi_1980]])
    np.savetxt(
        out_path,
        out_arr,
        delimiter=",",
        header="alpha_a,alpha_k,phi_1980",
        comments="",
    )
    print("\nBase calibration saved to {}".format(out_path))

    # -------------------------------------------------------------------------
    # Phi path inversion: find phi_t for each year to match pct_neg_t
    # -------------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("Phi path inversion: year-by-year brentq on pct_neg")
    print("=" * 60)

    # Bounds for phi search: allow wide range around phi_1980
    phi_lo_global = _BOUNDS_LOW[2]
    phi_hi_global = _BOUNDS_HIGH[2]

    phi_path = []
    last_eqm = eqm_base   # warm start carried across years
    last_phi = phi_1980

    for yr, pct_target in zip(years, pct_neg_data):
        print("\nYear {}: target pct_neg = {:.4f}%".format(yr, pct_target))

        _ws_inv = {'eqm': last_eqm}

        def _pct_resid(phi_try):
            warm = (np.array([_ws_inv['eqm']['c_agg'],
                               _ws_inv['eqm']['W'],
                               _ws_inv['eqm']['P_M']])
                    if _ws_inv['eqm'] is not None else None)
            eqm = _solve_eqm(phi_try, alpha_a_cal, alpha_k_cal,
                             fixed_cost=0.0, warm_start=warm,
                             max_nfev=500, vf_maxit=200)
            if eqm is None:
                return 1e3
            _ws_inv['eqm'] = eqm
            return pct_negative(m_grid, k_grid, z_grid, eqm) - pct_target

        try:
            phi_t = brentq(_pct_resid, phi_lo_global, phi_hi_global, xtol=1e-4, maxiter=50)
            print("  phi_{} = {:.6f}".format(yr, phi_t))
            # update warm start for next year
            if _ws_inv['eqm'] is not None:
                last_eqm = _ws_inv['eqm']
            last_phi = phi_t
        except ValueError as e:
            print("  [WARN] brentq failed for year {}: {}  — using last phi = {:.4f}".format(
                yr, e, last_phi))
            phi_t = last_phi

        phi_path.append((yr, phi_t))

    # -------------------------------------------------------------------------
    # Save phi path
    # -------------------------------------------------------------------------

    phi_df = pd.DataFrame(phi_path, columns=["year", "phi"])
    phi_path_out = os.path.join(_DIR, "phi_path.csv")
    phi_df.to_csv(phi_path_out, index=False)
    print("\nPhi path saved to {}".format(phi_path_out))
    print(phi_df.to_string(index=False))
