
import numpy as np
from numba import njit
from prod_fncts import *
from solve_vf import *

""" Functions to compute the stationary distribution of firms over (d,K,z)"""

@njit
def get_lottery(x, grid):
    # Handle single-element grid: all mass goes to index 0
    if len(grid) == 1:
        return 0, 1.0, 0, 0.0
    x = min(max(x, grid[0]), grid[-1])
    i = np.searchsorted(grid, x) - 1
    if i < 0:
        i = 0
    if i >= len(grid) - 1:
        i = len(grid) - 2
    w_right = (x - grid[i]) / (grid[i+1] - grid[i])
    return i, 1.0 - w_right, i+1, w_right

@njit
def forward_step(Dist, m_pol, k_pol, Pi, m_grid, k_grid):
    Nm, Nk, Nz = len(m_grid), len(k_grid), Pi.shape[0]
    newDist = np.zeros_like(Dist)
    for im in range(Nm):
        for ik in range(Nk):
            for iz in range(Nz):
                mass = Dist[im, ik, iz]
                if mass <= 0.0:
                    continue
                m_next = m_pol[im, ik, iz]
                k_next = k_pol[im, ik, iz]
                mi_l, wm_l, mi_r, wm_r = get_lottery(m_next, m_grid)
                ki_l, wk_l, ki_r, wk_r = get_lottery(k_next, k_grid)
                for izp in range(Nz):
                    P = Pi[iz, izp]
                    m = mass * P
                    newDist[mi_l, ki_l, izp] += m * wm_l  * wk_l
                    newDist[mi_l, ki_r, izp] += m * wm_l  * wk_r
                    newDist[mi_r, ki_l, izp] += m * wm_r  * wk_l
                    newDist[mi_r, ki_r, izp] += m * wm_r  * wk_r
    return newDist

def stationary_distribution(m_pol, k_pol, Pi, m_grid, k_grid, entry_perc, tol=1e-10, maxit=10000,
                            invest_m=True, invest_k=True):
    """
    Compute stationary distribution over (m, k, z).

    If invest_m=False (m_grid has 1 element), all firms have m=1.
    If invest_k=False (k_grid has 1 element), all firms have k=1.
    """
    Nm, Nk, Nz = len(m_grid), len(k_grid), Pi.shape[0]
    Dist = np.zeros((Nm, Nk, Nz))
    pi = stationary_markov(Pi)

    # Initial distribution: entrants start at lowest m and k grid points
    Dist[0, 0, :] = pi
    new_entrants = np.zeros((Nm, Nk, Nz))
    new_entrants[0, 0, :] = pi

    for _ in range(maxit):
        D_new = forward_step(Dist, m_pol, k_pol, Pi, m_grid, k_grid)
        D_new = (1-entry_perc) * D_new + entry_perc * new_entrants
        if np.max(np.abs(D_new - Dist)) < tol:
            return D_new
        Dist = D_new
    return Dist

def cdf(pdf):
    # sort pdf by first column
    pdf = pdf[np.argsort(pdf[:, 0])]
    # calculate cdf
    cdf = np.cumsum(pdf[:, 1])
    # return cdf
    return np.column_stack((pdf[:, 0], cdf))

# define a function that calculates all relevant values and returns the specified one
def calc_value(m, k, z, m_prime, k_prime, eqm, value_type):
    # unpack eqm agg values
    c_agg = eqm['c_agg']
    W = eqm['W']
    P_M = eqm['P_M']
    sigma = eqm['params']['sigma']
    gamma_k = eqm['params']['gamma_k']
    gamma_l = eqm['params']['gamma_l']
    alpha_k = eqm['params']['alpha_k']
    z_k = eqm['params']['z_k']
    alpha_a = eqm['params']['alpha_a']
    z_a = eqm['params']['z_a']
    delta_m = eqm['params']['delta_m']
    delta_k = eqm['params']['delta_k']
    phi = eqm['params']['phi']
    fixed_cost = eqm['params'].get('fixed_cost', 0.0)
    # compute earnings
    c_val = c_i_star(m, k, z, c_agg, sigma, W, gamma_k, gamma_l, phi)
    # With phi: M^(1+phi) in demand, so price formula uses m^((1+phi)/sigma)
    p_val = (m**(1+phi) * c_agg)**(1/sigma) * c_val**(-1/sigma)
    adv = (m_prime - (1.0 - delta_m) * m) * P_M
    inv = k_prime - (1.0 - delta_k) * k
    L_a_i = L_a(adv, alpha_a, z_a)
    L_k_i = L_k(inv, alpha_k, z_k)
    L_s_i = L_s(c_val, z, k, gamma_k, gamma_l)
    earnings = p_val*c_val - W * (L_s_i + L_a_i) - fixed_cost
    income = earnings - W * L_k(delta_k * k, alpha_k, z_k)
    revenue = p_val * c_val
    adv_ratio = (W*L_a_i) / revenue
    inv_ratio = (W*L_k_i) / revenue
    cogs_ratio = (W*L_s_i) / revenue
    if value_type == 'earnings':
        return earnings
    elif value_type == 'income':
        return income
    elif value_type == 'revenue':
        return revenue
    elif value_type == 'adv_ratio':
        return adv_ratio
    elif value_type == 'inv_ratio':
        return inv_ratio
    elif value_type == 'cogs_ratio':
        return cogs_ratio

def est_dist(m_grid, k_grid, z_grid, eqm, value_type):
    Dist = eqm['Dist']
    policies = eqm['policies']

    # Use grids from equilibrium if stored (solver converts None to [1.0])
    m_grid = eqm.get('m_grid', m_grid)
    k_grid = eqm.get('k_grid', k_grid)

    Nm, Nk, Nz = len(m_grid), len(k_grid), len(z_grid)

    m_pol = policies.get('m_policy', None)
    k_pol = policies.get('k_policy', None)

    N = Nm * Nk * Nz
    val_pdf = np.zeros((N, 2))
    index = 0

    for im in range(Nm):
        for ik in range(Nk):
            for iz in range(Nz):
                mass = Dist[im, ik, iz]
                if mass <= 0.0:
                    continue

                m_val, k_val, z_val = m_grid[im], k_grid[ik], z_grid[iz]
                m_prime = m_pol[im, ik, iz] if m_pol is not None else m_val
                k_prime = k_pol[im, ik, iz] if k_pol is not None else k_val

                val = calc_value(m_val, k_val, z_val, m_prime, k_prime, eqm, value_type)
                val_pdf[index, :] = [val, mass]
                index += 1

    val_cdf = cdf(val_pdf)
    return val_pdf, val_cdf

def m_dist(m_grid, eqm):
    """Return PDF and CDF of the marginal distribution of m (customers)."""
    Dist = eqm['Dist']
    # Use grid from equilibrium if stored
    m_grid = eqm.get('m_grid', m_grid)
    m_marginal = Dist.sum(axis=(1, 2))  # sum over k and z
    m_pdf = np.column_stack((m_grid, m_marginal))
    m_cdf = cdf(m_pdf)
    return m_pdf, m_cdf

def median_from_cdf(val_cdf):
    """Find the value at cdf = 0.5 (the median)."""
    idx = np.searchsorted(val_cdf[:, 1], 0.5)
    return val_cdf[idx, 0]

def median_adv_ratio(m_grid, k_grid, z_grid, eqm, negative_earnings_only=False):
    """Compute median adv_ratio. If negative_earnings_only=True, restrict to firms with earnings < 0."""
    _, adv_cdf = est_dist(m_grid, k_grid, z_grid, eqm, 'adv_ratio')
    if negative_earnings_only:
        earn_pdf, _ = est_dist(m_grid, k_grid, z_grid, eqm, 'earnings')
        mask = earn_pdf[:, 0] < 0
        if not np.any(mask):
            return np.nan
        adv_pdf, _ = est_dist(m_grid, k_grid, z_grid, eqm, 'adv_ratio')
        filtered_pdf = adv_pdf[mask].copy()
        filtered_pdf[:, 1] = filtered_pdf[:, 1] / np.sum(filtered_pdf[:, 1])
        adv_cdf = cdf(filtered_pdf)
    return median_from_cdf(adv_cdf)

def median_inv_ratio(m_grid, k_grid, z_grid, eqm, negative_earnings_only=False):
    """Compute median inv_ratio. If negative_earnings_only=True, restrict to firms with earnings < 0."""
    _, inv_cdf = est_dist(m_grid, k_grid, z_grid, eqm, 'inv_ratio')
    if negative_earnings_only:
        earn_pdf, _ = est_dist(m_grid, k_grid, z_grid, eqm, 'earnings')
        mask = earn_pdf[:, 0] < 0
        if not np.any(mask):
            return np.nan
        inv_pdf, _ = est_dist(m_grid, k_grid, z_grid, eqm, 'inv_ratio')
        filtered_pdf = inv_pdf[mask].copy()
        filtered_pdf[:, 1] = filtered_pdf[:, 1] / np.sum(filtered_pdf[:, 1])
        inv_cdf = cdf(filtered_pdf)
    return median_from_cdf(inv_cdf)

def median_cogs_ratio(m_grid, k_grid, z_grid, eqm, negative_earnings_only=False):
    """Compute median cogs_ratio. If negative_earnings_only=True, restrict to firms with earnings < 0."""
    _, cogs_cdf = est_dist(m_grid, k_grid, z_grid, eqm, 'cogs_ratio')
    if negative_earnings_only:
        earn_pdf, _ = est_dist(m_grid, k_grid, z_grid, eqm, 'earnings')
        mask = earn_pdf[:, 0] < 0
        if not np.any(mask):
            return np.nan
        cogs_pdf, _ = est_dist(m_grid, k_grid, z_grid, eqm, 'cogs_ratio')
        filtered_pdf = cogs_pdf[mask].copy()
        filtered_pdf[:, 1] = filtered_pdf[:, 1] / np.sum(filtered_pdf[:, 1])
        cogs_cdf = cdf(filtered_pdf)
    return median_from_cdf(cogs_cdf)

def pct_negative(m_grid, k_grid, z_grid, eqm):
    """Compute percent of firms with earnings < 0 (returns 0-100)."""
    earn_pdf, _ = est_dist(m_grid, k_grid, z_grid, eqm, 'earnings')
    mask = earn_pdf[:, 0] < 0
    total_mass = np.sum(earn_pdf[:, 1])
    negative_mass = np.sum(earn_pdf[mask, 1])
    return 100.0 * negative_mass / total_mass

def pct_negative_income(m_grid, k_grid, z_grid, eqm):
    """Compute percent of firms with income (earnings net of depreciation cost) < 0."""
    income_pdf, _ = est_dist(m_grid, k_grid, z_grid, eqm, 'income')
    mask = income_pdf[:, 0] < 0
    total_mass = np.sum(income_pdf[:, 1])
    negative_mass = np.sum(income_pdf[mask, 1])
    return 100.0 * negative_mass / total_mass

def percentile_from_cdf(cdf, p):
    return np.interp(p, cdf[:,1], cdf[:,0])

def p99_minus_median(cdf):
    return percentile_from_cdf(cdf, 0.99) - percentile_from_cdf(cdf, 0.50)

def est_sd(cdf):
    mean = np.sum(cdf[:,0] * np.diff(np.hstack(([0], cdf[:,1]))))
    var = np.sum((cdf[:,0] - mean)**2 * np.diff(np.hstack(([0], cdf[:,1]))))
    return var**0.5

def median_earnings(m_grid, k_grid, z_grid, eqm):
    """Compute median earnings."""
    _, earn_cdf = est_dist(m_grid, k_grid, z_grid, eqm, 'earnings')
    return median_from_cdf(earn_cdf)

def mean_earnings(m_grid, k_grid, z_grid, eqm):
    """Compute mean earnings."""
    earn_pdf, _ = est_dist(m_grid, k_grid, z_grid, eqm, 'earnings')
    total_mass = np.sum(earn_pdf[:, 1])
    return np.sum(earn_pdf[:, 0] * earn_pdf[:, 1]) / total_mass

def agg_capital_stock(m_grid, k_grid, z_grid, eqm):
    """Aggregate physical capital stock = distribution-weighted sum of k."""
    Dist = eqm['Dist']
    k_grid = eqm.get('k_grid', k_grid)
    return float(np.einsum('ijk,j->', Dist, k_grid))

def sales_wtd_productivity(m_grid, k_grid, z_grid, eqm):
    """Sales-weighted average TFP: sum(z * revenue * mass) / sum(revenue * mass)."""
    Dist = eqm['Dist']
    m_grid = eqm.get('m_grid', m_grid)
    k_grid = eqm.get('k_grid', k_grid)
    policies = eqm['policies']
    m_pol = policies.get('m_policy', None)
    k_pol = policies.get('k_policy', None)
    Nm, Nk, Nz = len(m_grid), len(k_grid), len(z_grid)
    num, den = 0.0, 0.0
    for im in range(Nm):
        for ik in range(Nk):
            for iz in range(Nz):
                mass = Dist[im, ik, iz]
                if mass <= 0.0:
                    continue
                m_val, k_val, z_val = m_grid[im], k_grid[ik], z_grid[iz]
                m_prime = m_pol[im, ik, iz] if m_pol is not None else m_val
                k_prime = k_pol[im, ik, iz] if k_pol is not None else k_val
                rev = calc_value(m_val, k_val, z_val, m_prime, k_prime, eqm, 'revenue')
                num += z_val * rev * mass
                den += rev * mass
    return num / den if den > 0 else np.nan

def avg_firm_earnings_path(eqm, z_grid, T=10):
    """
    Simulate earnings path for a firm fixed at median productivity.

    The firm enters at (m_grid[0], k_grid[0]) and follows equilibrium policy
    functions for T periods with z held fixed at the median z_grid state.
    Policy values at off-grid (m, k) are bilinearly interpolated via get_lottery.

    Parameters
    ----------
    eqm    : dict   Equilibrium object (same structure used by est_dist).
    z_grid : array  Productivity grid.
    T      : int    Number of periods to simulate (default 10).

    Returns
    -------
    earnings : np.ndarray, shape (T,)
    """
    m_grid = eqm.get('m_grid')
    k_grid = eqm.get('k_grid')
    m_pol  = eqm['policies']['m_policy']
    k_pol  = eqm['policies']['k_policy']

    iz_med = len(z_grid) // 2
    z_med  = z_grid[iz_med]

    m = m_grid[0]
    k = k_grid[0]

    earnings = np.zeros(T)
    for t in range(T):
        mi_l, wm_l, mi_r, wm_r = get_lottery(m, m_grid)
        ki_l, wk_l, ki_r, wk_r = get_lottery(k, k_grid)

        m_prime = (m_pol[mi_l, ki_l, iz_med] * wm_l * wk_l +
                   m_pol[mi_l, ki_r, iz_med] * wm_l * wk_r +
                   m_pol[mi_r, ki_l, iz_med] * wm_r * wk_l +
                   m_pol[mi_r, ki_r, iz_med] * wm_r * wk_r)

        k_prime = (k_pol[mi_l, ki_l, iz_med] * wm_l * wk_l +
                   k_pol[mi_l, ki_r, iz_med] * wm_l * wk_r +
                   k_pol[mi_r, ki_l, iz_med] * wm_r * wk_l +
                   k_pol[mi_r, ki_r, iz_med] * wm_r * wk_r)

        earnings[t] = calc_value(m, k, z_med, m_prime, k_prime, eqm, 'earnings')
        m = m_prime
        k = k_prime

    return earnings


def cohort_neg_path(eqm, z_grid, Pi, T=50):
    """
    Evolve a cohort of entrants forward T periods and return the fraction
    earning < 0 at each period.

    Entrants start at (m_grid[0], k_grid[0]) with z drawn from the stationary
    z distribution. Productivity follows Pi each period; no exit/re-entry.

    Parameters
    ----------
    eqm    : dict            Equilibrium object.
    z_grid : array (Nz,)    Productivity grid.
    Pi     : array (Nz, Nz) Markov transition matrix for z.
    T      : int             Number of periods to simulate (default 50).

    Returns
    -------
    pct_neg : np.ndarray, shape (T,)
        Percent of cohort with negative earnings in each period (0-100).
    """
    m_grid = eqm.get('m_grid')
    k_grid = eqm.get('k_grid')
    m_pol  = eqm['policies']['m_policy']
    k_pol  = eqm['policies']['k_policy']

    pi_z = stationary_markov(Pi)
    Nm, Nk, Nz = len(m_grid), len(k_grid), len(z_grid)
    Dist = np.zeros((Nm, Nk, Nz))
    Dist[0, 0, :] = pi_z

    pct_neg = np.zeros(T)
    for t in range(T):
        neg_mass = 0.0
        total_mass = 0.0
        for im in range(Nm):
            for ik in range(Nk):
                for iz in range(Nz):
                    mass = Dist[im, ik, iz]
                    if mass <= 0.0:
                        continue
                    m_prime = m_pol[im, ik, iz]
                    k_prime = k_pol[im, ik, iz]
                    earn = calc_value(m_grid[im], k_grid[ik], z_grid[iz],
                                      m_prime, k_prime, eqm, 'earnings')
                    total_mass += mass
                    if earn < 0:
                        neg_mass += mass
        pct_neg[t] = 100.0 * neg_mass / total_mass if total_mass > 0 else 0.0
        Dist = forward_step(Dist, m_pol, k_pol, Pi, m_grid, k_grid)

    return pct_neg


def avg_neg_periods(eqm, z_grid, Pi, T=50):
    """Expected number of periods an entrant earns < 0 in the first T periods."""
    return np.sum(cohort_neg_path(eqm, z_grid, Pi, T)) / 100.0


def agg_labor_shares(m_grid, k_grid, z_grid, eqm):
    """
    Aggregate labor allocations: (L_a, L_k, L_s).
    By labor market clearing these must sum to 1 — asserted as a bug check.
    """
    Dist = eqm['Dist']
    policies = eqm['policies']
    m_grid = eqm.get('m_grid', m_grid)
    k_grid = eqm.get('k_grid', k_grid)

    c_agg   = eqm['c_agg']
    W       = eqm['W']
    P_M     = eqm['P_M']
    sigma   = eqm['params']['sigma']
    gamma_k = eqm['params']['gamma_k']
    gamma_l = eqm['params']['gamma_l']
    alpha_k = eqm['params']['alpha_k']
    z_k     = eqm['params']['z_k']
    alpha_a = eqm['params']['alpha_a']
    z_a     = eqm['params']['z_a']
    delta_m = eqm['params']['delta_m']
    delta_k = eqm['params']['delta_k']
    phi     = eqm['params']['phi']

    m_pol = policies.get('m_policy', None)
    k_pol = policies.get('k_policy', None)
    Nm, Nk, Nz = len(m_grid), len(k_grid), len(z_grid)

    La_total = Lk_total = Ls_total = 0.0
    for im in range(Nm):
        for ik in range(Nk):
            for iz in range(Nz):
                mass = Dist[im, ik, iz]
                if mass <= 0.0:
                    continue
                m_val = m_grid[im]; k_val = k_grid[ik]; z_val = z_grid[iz]
                m_prime = m_pol[im, ik, iz] if m_pol is not None else m_val
                k_prime = k_pol[im, ik, iz] if k_pol is not None else k_val

                c_val = c_i_star(m_val, k_val, z_val, c_agg, sigma, W, gamma_k, gamma_l, phi)
                adv   = (m_prime - (1.0 - delta_m) * m_val) * P_M
                inv   = k_prime - (1.0 - delta_k) * k_val

                La_total += L_a(adv,   alpha_a, z_a) * mass
                Lk_total += L_k(inv,   alpha_k, z_k) * mass
                Ls_total += L_s(c_val, z_val, k_val, gamma_k, gamma_l) * mass

    total = La_total + Lk_total + Ls_total
    assert abs(total - 1.0) < 1e-4, f"Labor market clearing violated: L_a+L_k+L_s = {total:.6f}"
    return La_total, Lk_total, Ls_total
