import numpy as np
from numba import njit
from scipy.optimize import least_squares
from solve_vf import *
from integrate_dist import stationary_distribution
from prod_fncts import *

@njit
def aggregate_vals(Dist, m_pol, k_pol, m_grid, k_grid, z_grid,
                   delta_m, delta_k, sigma, c_agg, P_M, W,
                   gamma_k, gamma_l, alpha_k, z_k, alpha_a, z_a, phi,
                   invest_m=True, invest_k=True):
    Nm, Nk, Nz = len(m_grid), len(k_grid), len(z_grid)
    agg_labor = 0.0
    agg_cust = 0.0
    agg_adv = 0.0
    consumption = 0.0
    for im in range(Nm):
        for ik in range(Nk):
            for iz in range(Nz):
                mass = Dist[im, ik, iz]
                if mass <= 0.0:
                    continue
                m_val, k_val, z_val = m_grid[im], k_grid[ik], z_grid[iz]
                c_val = c_i_star(m_val, k_val, z_val, c_agg, sigma, W, gamma_k, gamma_l, phi)
                m_prime = m_pol[im, ik, iz]
                k_prime = k_pol[im, ik, iz]

                # Compute labor for each investment type (only if active)
                L_s_i = L_s(c_val, z_val, k_val, gamma_k, gamma_l)
                if invest_m:
                    adv = (m_prime - (1.0 - delta_m) * m_val) * P_M
                    L_a_i = L_a(adv, alpha_a, z_a)
                else:
                    adv = 0.0
                    L_a_i = 0.0
                if invest_k:
                    inv = k_prime - (1.0 - delta_k) * k_val
                    L_k_i = L_k(inv, alpha_k, z_k)
                else:
                    inv = 0.0
                    L_k_i = 0.0

                agg_labor += mass * (L_s_i + L_k_i + L_a_i)
                agg_adv += mass * adv
                agg_cust += mass * m_val
                consumption += mass * c_val**((sigma-1)/sigma)

    consumption = consumption**(sigma/(sigma-1))
    return consumption, agg_labor, agg_cust, agg_adv

class EqmParams(object):
    # DRS labor cost parameters:
    #   alpha_k, z_k for capital investment: L^k(i) = (i/z_k)^(1/alpha_k)
    #   alpha_a, z_a for advertising:        L^a(a) = (a/z_a)^(1/alpha_a)
    # z_a is normalized to 1.0 (scale of advertising is absorbed by P_M).
    def __init__(self, beta=0.96, delta_m=0.15, delta_k=0.10, entry_perc=0,
                 sigma=5.0, gamma_k=1.0/3, gamma_l=2.0/3,
                 alpha_k=0.5, z_k=1.0, alpha_a=0.5, z_a=1.0,
                 phi=0.0, invest_m=True, invest_k=True, fixed_cost=0.0):
        self.beta = beta
        self.delta_m = delta_m
        self.delta_k = delta_k
        self.entry_perc = entry_perc
        self.sigma = sigma
        self.gamma_k = gamma_k  # capital exponent in Y = Z * K^gamma_k * L^gamma_l
        self.gamma_l = gamma_l  # labor exponent (not constrained to sum to 1 with gamma_k)
        self.alpha_k = alpha_k
        self.z_k = z_k
        self.alpha_a = alpha_a
        self.z_a = z_a
        self.phi = phi
        self.invest_m = invest_m  # whether to allow customer investment
        self.invest_k = invest_k  # whether to allow capital investment
        self.fixed_cost = fixed_cost  # fixed operating cost deducted from earnings

def eqm_residuals(x, m_grid, k_grid, z_grid, Pi, p: EqmParams, verbose=True):
    c_agg, W, P_M = x
    pol = solve_vf_egm(m_grid, k_grid, z_grid, Pi,
                       c_agg, W, P_M,
                       p.beta, p.entry_perc, p.sigma, p.delta_m, p.delta_k,
                       p.gamma_k, p.gamma_l, p.alpha_k, p.z_k, p.alpha_a, p.z_a, p.phi,
                       maxit=250, tol=1e-6, verbose=verbose,
                       invest_m=p.invest_m, invest_k=p.invest_k)
    Dist = stationary_distribution(pol["m_policy"], pol["k_policy"], Pi, m_grid, k_grid, p.entry_perc,
                                   invest_m = p.invest_m, invest_k = p.invest_k)
    consumption, agg_labor, agg_cust, agg_adv = aggregate_vals(
        Dist, pol["m_policy"], pol["k_policy"], m_grid, k_grid, z_grid,
        p.delta_m, p.delta_k, p.sigma, c_agg, P_M, W,
        p.gamma_k, p.gamma_l, p.alpha_k, p.z_k, p.alpha_a, p.z_a, p.phi,
        p.invest_m, p.invest_k
    )
    rC = consumption - c_agg
    rM = agg_cust - 1.0
    rL = agg_labor - 1.0
    return np.array([rC, rM, rL])

def solve_ss_equilibrium_least_squares(m_grid, k_grid, z_grid, Pi,
                                       p: EqmParams,
                                       start=None, bounds=None, verbose=True,
                                       max_nfev=1000):
    """
    Solve for steady-state equilibrium.

    If m_grid is None, use m=1 for all firms (no customer investment).
    If k_grid is None, use k=0 for all firms (no capital investment).
    """
    # Handle None grids: use single-element grid at 1.0
    if m_grid is None:
        m_grid = np.array([1.0])
        p.invest_m = False
        p.delta_m = 0.0  # No depreciation if no customer investment
        if verbose:
            print("m_grid=None: setting m=1 for all firms (no customer investment)")
    if k_grid is None:
        k_grid = np.array([1.0])
        p.invest_k = False
        p.delta_k = 0.0  # No depreciation if no capital investment
        if verbose:
            print("k_grid=None: setting k=1 for all firms (no capital investment)")

    if start is None:
        start = np.array([1.0, 1.0, 1.0])
    if bounds is None:
        lb = np.array([0, 0, 0]); ub = np.array([np.inf, np.inf, np.inf])
        bounds = (lb, ub)
    ls = least_squares(lambda x: eqm_residuals(x, m_grid, k_grid, z_grid, Pi, p, verbose), start, bounds=bounds, xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=max_nfev, verbose=2 if verbose else 0)
    x = ls.x
    c_agg, W, P_M = x
    pol = solve_vf_egm(m_grid, k_grid, z_grid, Pi,
                       c_agg, W, P_M,
                       p.beta, p.entry_perc, p.sigma, p.delta_m, p.delta_k,
                       p.gamma_k, p.gamma_l, p.alpha_k, p.z_k, p.alpha_a, p.z_a, p.phi,
                       maxit=250, tol=1e-6, verbose=False,
                       invest_m=p.invest_m, invest_k=p.invest_k)
    Dist = stationary_distribution(pol["m_policy"], pol["k_policy"], Pi, m_grid, k_grid, p.entry_perc,
                                   invest_m = p.invest_m, invest_k = p.invest_k)
    consumption, agg_labor, agg_cust, agg_adv = aggregate_vals(
        Dist, pol["m_policy"], pol["k_policy"], m_grid, k_grid, z_grid,
        p.delta_m, p.delta_k, p.sigma, c_agg, P_M, W,
        p.gamma_k, p.gamma_l, p.alpha_k, p.z_k, p.alpha_a, p.z_a, p.phi,
        p.invest_m, p.invest_k
    )
    print("Solution:", {"C_agg": c_agg}, {"C_impl": consumption}, {"Customers": agg_cust}, {"Labor": agg_labor}, {"Adv": agg_adv})
    out = {
        "W": W, "c_agg": c_agg, "P_M": P_M,
        "residuals": ls.fun,  # reuse residuals from last LS eval (avoids extra VF solve)
        "ls_success": ls.success, "ls_message": ls.message,
        "policies": pol, "Dist": Dist,
        "Agg": {"C_impl": consumption, "Adv_Impl": agg_adv},
        "params": p.__dict__,
        "m_grid": m_grid, "k_grid": k_grid  # Store the grids used (may have been converted from None)
    }
    if verbose:
        print("Solution:", {"C": c_agg, "W": W, "P_M": P_M})
        print("||res|| =", float(np.linalg.norm(out["residuals"])))
        print("LS message:", ls.message)
    return out
