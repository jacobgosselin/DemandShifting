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

# Empirical phi coefficients by year (used to back out phi from sigma)
elast_df  = pd.read_csv(os.path.join(_DIR, "sales_elasticity_m_by_year.csv"))
coef_1980 = float(elast_df.loc[elast_df["year"] == years[0], "coef"].iloc[0])
print("Loaded coef_1980 = {:.6f} (from sales_elasticity_m_by_year.csv)".format(coef_1980))


def phi_from_coef(coef, sigma_val):
    """Back out model phi from regression coefficient and sigma.

    Derivation: sales elasticity of m = coef = (1+phi) / ((1-gamma_l)*sigma + gamma_l)
    => phi = coef * ((1-gamma_l)*sigma + gamma_l) - 1
    """
    return coef * ((1 - gamma_l) * sigma_val + gamma_l) - 1


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
phi_init     = phi_from_coef(coef_1980, sigma_fixed)  # default: back out from coef
sigma_init   = sigma_fixed                             # default: from structural_parameters

if os.path.isfile(out_path):
    _cal_prev = pd.read_csv(out_path)
    if "alpha_a" in _cal_prev.columns:
        alpha_a_init = float(_cal_prev["alpha_a"].iloc[0])
    if "alpha_k" in _cal_prev.columns:
        alpha_k_init = float(_cal_prev["alpha_k"].iloc[0])
    if "phi" in _cal_prev.columns:
        phi_init = float(_cal_prev["phi"].iloc[0])
    if "sigma" in _cal_prev.columns:
        sigma_init = float(_cal_prev["sigma"].iloc[0])
    print("\nLoaded initial guesses from {}:".format(out_path))
    print("  alpha_a_init = {:.6f}".format(alpha_a_init))
    print("  alpha_k_init = {:.6f}".format(alpha_k_init))
    print("  phi_init     = {:.6f}".format(phi_init))
    print("  sigma_init   = {:.6f}".format(sigma_init))
else:
    print("\nNo existing calibrated parameters found; using defaults.")
    print("  phi_init   (from coef_1980)  = {:.6f}".format(phi_init))
    print("  sigma_init (from struct_params) = {:.6f}".format(sigma_init))

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
# Calibration: (alpha_a, alpha_k, sigma)
#   phi_0 is backed out analytically from coef_1980 and sigma_try
#   Targets:
#     median adv_ratio  =  med_sga_sale
#     median inv_ratio  =  med_capx_sale
#     pct_neg           =  neg_ebitda_base_pct
# -----------------------------------------------------------------------------

_BOUNDS_LOW  = [0.1, 0.1, -0.5, 1.5]    # alpha_a, alpha_k, phi, sigma
_BOUNDS_HIGH = [0.9, 0.9,  0.5, 10.0]   # alpha_a, alpha_k, phi, sigma

def obj_base(x, max_nfev=30, vf_maxit=200):
    alpha_a_try, alpha_k_try, phi_try, sigma_try = (
        float(x[0]), float(x[1]), float(x[2]), float(x[3])
    )

    # bounds check (needed for unconstrained solvers like Nelder-Mead)
    if not (_BOUNDS_LOW[0] <= alpha_a_try <= _BOUNDS_HIGH[0] and
            _BOUNDS_LOW[1] <= alpha_k_try <= _BOUNDS_HIGH[1] and
            _BOUNDS_LOW[2] <= phi_try     <= _BOUNDS_HIGH[2] and
            _BOUNDS_LOW[3] <= sigma_try   <= _BOUNDS_HIGH[3]):
        return np.full(3, 1e3)

    eqm = _solve_eqm(phi_try, alpha_a_try, alpha_k_try, sigma_try,
                     fixed_cost=0.0, max_nfev=max_nfev, vf_maxit=vf_maxit)
    if eqm is None:
        return np.full(3, 1e3)

    adv_med = median_adv_ratio(m_grid, k_grid, z_grid, eqm)
    inv_med = median_inv_ratio(m_grid, k_grid, z_grid, eqm)
    pct     = pct_negative(m_grid, k_grid, z_grid, eqm)
 
    res = np.array([
        (adv_med - med_sga_sale),
        (inv_med - med_capx_sale),
        (pct     - neg_ebitda_base_pct),
    ])
    if not np.all(np.isfinite(res)):
        return np.full(3, 1e3)

    print(
        "aa={:.4f} ak={:.4f} phi={:.4f} sigma={:.4f} | "
        "adv={:.4f} inv={:.4f} pct={:.2f}% | "
        "||res||={:.6f}".format(
            alpha_a_try, alpha_k_try, phi_try, sigma_try,
            adv_med, inv_med, pct,
            np.linalg.norm(res))
    )
    import sys; sys.stdout.flush()
    return res


if __name__ == '__main__':
    multiprocessing.set_start_method('fork', force=True)

    # -------------------------------------------------------------------------
    # Phase 1: Powell — solve for alpha_a, alpha_k, sigma
    # -------------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("Phase 1: Solve for alpha_a, alpha_k, phi, sigma  (4 params, 3 moments)")
    print("  Targets: adv/rev={:.4f}  capx/rev={:.4f}  pct_neg@1980={:.2f}%".format(
        med_sga_sale, med_capx_sale, neg_ebitda_base_pct))
    print("  sigma_fixed (reference, not imposed) = {:.6f}".format(sigma_fixed))
    print("=" * 60)

    def obj_base_scalar(x):
        res = obj_base(x)
        return float(np.sum(res**2))

    result = differential_evolution(
        obj_base_scalar,
        bounds=list(zip(_BOUNDS_LOW, _BOUNDS_HIGH)),
        workers=-1,
        seed=42,
        tol=1e-4,
        atol=1e-3,
        maxiter=200,
        popsize=10,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=False,
        updating='deferred',
    )

    # Powell solver for the 3×3 system
    # result = minimize(
    #     obj_base_scalar,
    #     x0=[alpha_a_init, alpha_k_init, phi_init],
    #     method='Powell',
    #     bounds=list(zip(_BOUNDS_LOW, _BOUNDS_HIGH)),
    #     options=dict(xtol=1e-4, ftol=1e-3, maxiter=500, maxfev=5000),
    # )

    # Nelder-Mead solver for the 3×3 system
    # result = minimize(
    #     obj_base_scalar,
    #     x0=[alpha_a_init, alpha_k_init, phi_init],
    #     method='Nelder-Mead',
    #     options=dict(xatol=1e-4, fatol=1e-3, maxiter=500, maxfev=5000),
    # )

    # Least squares solver for the 3×3 system
    # result = least_squares(
    #     obj_base,
    #     x0=[alpha_a_init, alpha_k_init, phi_init],
    #     bounds=(_BOUNDS_LOW, _BOUNDS_HIGH),
    #     kwargs=dict(max_nfev=30, vf_maxit=200),
    #     verbose=2,
    # )

    alpha_a_cal = float(result.x[0])
    alpha_k_cal = float(result.x[1])
    phi_cal     = float(result.x[2])
    sigma_cal   = float(result.x[3])

    print("\nBase-period calibration result:")
    print("  success      = {}".format(result.success))
    print("  message      = {}".format(result.message))
    print("  alpha_a_cal  = {:.6f}".format(alpha_a_cal))
    print("  alpha_k_cal  = {:.6f}".format(alpha_k_cal))
    print("  phi_cal      = {:.6f}".format(phi_cal))
    print("  sigma_cal    = {:.6f}  (calibrated)".format(sigma_cal))
    print("  sigma_fixed  = {:.6f}  (reference from ACF)".format(sigma_fixed))
    print("  residuals    = {}".format(result.fun))
    print("  cost         = {:.8f}".format(result.cost if hasattr(result, 'cost') else result.fun))

    # -------------------------------------------------------------------------
    # Post-calibration diagnostics (base period)
    # -------------------------------------------------------------------------

    eqm_base = _solve_eqm(phi_cal, alpha_a_cal, alpha_k_cal, sigma_cal, fixed_cost=0.0)
    if eqm_base is not None:
        adv_med_cal = median_adv_ratio(m_grid, k_grid, z_grid, eqm_base)
        inv_med_cal = median_inv_ratio(m_grid, k_grid, z_grid, eqm_base)
        pct_cal_val = pct_negative(m_grid, k_grid, z_grid, eqm_base)
        print("\nPost-calibration diagnostics (base period):")
        print("  adv/rev:  data={:.4f},  model={:.4f}".format(med_sga_sale,        adv_med_cal))
        print("  capx/rev: data={:.4f},  model={:.4f}".format(med_capx_sale,       inv_med_cal))
        print("  pct_neg:  data={:.2f}%, model={:.2f}%".format(neg_ebitda_base_pct, pct_cal_val))

    # -------------------------------------------------------------------------
    # Partial-effect diagnostics: verify identification of the 3×3 system
    # -------------------------------------------------------------------------

    print("\n--- Partial effect diagnostics at calibrated optimum ---")
    x_opt = np.array([alpha_a_cal, alpha_k_cal, phi_cal, sigma_cal])
    param_names = ["alpha_a", "alpha_k", "phi", "sigma"]
    for idx, pname in enumerate(param_names):
        lo, hi = _BOUNDS_LOW[idx], _BOUNDS_HIGH[idx]
        print("\n  Sweep {} (others fixed at optimum):".format(pname))
        for v in np.linspace(lo, hi, 5):
            x_try = x_opt.copy(); x_try[idx] = v
            res_diag = obj_base(x_try, max_nfev=50, vf_maxit=150)
            print("    {}={:.3f}: ||res||={:.4f}".format(pname, v, np.linalg.norm(res_diag)))

    # -------------------------------------------------------------------------
    # Save base calibration
    # -------------------------------------------------------------------------

    out_arr = np.array([[alpha_a_cal, alpha_k_cal, phi_cal, sigma_cal]])
    np.savetxt(
        out_path,
        out_arr,
        delimiter=",",
        header="alpha_a,alpha_k,phi,sigma",
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
    phi_lo_global = -0.5
    phi_hi_global =  0.75

    phi_path = []
    last_phi = phi_cal

    for yr, pct_target in zip(years, pct_neg_data):
        print("\nYear {}: target pct_neg = {:.4f}%".format(yr, pct_target))

        def _pct_resid(phi_try):
            eqm = _solve_eqm(phi_try, alpha_a_cal, alpha_k_cal, sigma_cal,
                             fixed_cost=0.0, max_nfev=500, vf_maxit=200)
            if eqm is None:
                return 1e3
            return pct_negative(m_grid, k_grid, z_grid, eqm) - pct_target

        try:
            phi_t = brentq(_pct_resid, phi_lo_global, phi_hi_global, xtol=1e-4, maxiter=50)
            print("  phi_{} = {:.6f}".format(yr, phi_t))
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
