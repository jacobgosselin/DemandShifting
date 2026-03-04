
import numpy as np
from numba import njit
from prod_fncts import *
import numpy as np

# -------- helper fncts 1: create grids --------

def rouwenhorst_Pi(N, p):
    # base case Pi_2
    Pi = np.array([[p, 1 - p],
                   [1 - p, p]])
    # recursion to build up from Pi_2 to Pi_N
    for n in range(3, N + 1):
        Pi_old = Pi
        Pi = np.zeros((n, n))
        Pi[:-1, :-1] += p * Pi_old
        Pi[:-1, 1:] += (1 - p) * Pi_old
        Pi[1:, :-1] += (1 - p) * Pi_old
        Pi[1:, 1:] += p * Pi_old
        Pi[1:-1, :] /= 2

    return Pi

def stationary_markov(Pi, tol=1E-14):
    # start with uniform distribution over all states
    n = Pi.shape[0]
    pi = np.full(n, 1/n)
    # update distribution using Pi until successive iterations differ by less than tol
    for _ in range(10_000):
        pi_new = Pi.T @ pi
        if np.max(np.abs(pi_new - pi)) < tol:
            return pi_new
        pi = pi_new

def discretize_productivity(rho, sigma, n_e):
    # choose inner-switching probability p to match persistence rho
    p = (1+rho)/2
    # start with states from 0 to n_e-1, scale by alpha to match standard deviation sigma
    e = np.arange(n_e)
    alpha = 2*sigma/np.sqrt(n_e-1)
    e = alpha*e
    # obtain Markov transition matrix Pi and its stationary distribution
    Pi = rouwenhorst_Pi(n_e, p)
    pi = stationary_markov(Pi)
    # e is log income, get income y and scale so that mean is 1
    y = np.exp(e)
    y /= np.vdot(pi, y)

    return y, pi, Pi

def discretize_choices(amin, amax, n_a, type="uniform"):
    ubar_exp = np.log(1 + amax - amin)
    ubar_dexp = np.log(1 + np.log(1 + amax - amin))
    exp_grid = amin + np.exp(np.linspace(0, ubar_exp, n_a)) - 1
    doubleexp_grid = amin + np.exp(np.exp(np.linspace(0, ubar_dexp, n_a)) - 1) - 1
    uniform_grid = np.linspace(amin, amax, n_a)
    if type == "uniform": return uniform_grid
    elif type == "exp": return exp_grid
    elif type == "doubleexp": return doubleexp_grid

# -------- helper fncts 2: generate EGM and interpolate --------

@njit
def expected_marginals(V, Pi):
    Nm, Nk, Nz = V.shape
    EV = np.zeros_like(V)
    for jd in range(Nm):
        for jk in range(Nk):
            V_temp = V[jd, jk, :]
            EV[jd, jk, :] = Pi @ V_temp
    return EV

# ===== Alternating 1-D EGM (no 2-D interpolation) =====

@njit
def _invert_monotone_map(x_src, y_map, x_tgt):
    """
    Given a monotone mapping x_src -> y_map (arrays length N over the same index),
    return x_src(y) evaluated at y = x_tgt via 1-D inverse interpolation.

    We do this by swapping axes: since y = f(x), we want f^{-1}(x_tgt).
    We clamp and enforce monotonicity of y_map for safety.
    """
    order = np.argsort(y_map)
    # y_sorted = np.maximum.accumulate(y_map[order])
    y_sorted = y_map[order].copy()
    for ii in range(1, y_sorted.shape[0]):
        if y_sorted[ii] < y_sorted[ii-1]:
            y_sorted[ii] = y_sorted[ii-1]
    x_sorted = x_src[order]
    yq = np.clip(x_tgt, y_sorted[0], y_sorted[-1])
    return np.interp(yq, y_sorted, x_sorted)

@njit
def backward_step_alt1d(Vk, Vm, m_grid, k_grid, z_grid, Pi,
                        c_agg, W, P_M, beta, entry_perc,
                        sigma, delta_m, delta_k,
                        gamma_k, gamma_l, alpha_k, z_k, alpha_a, z_a, phi,
                        invest_m=True, invest_k=True):
    """
    Alternating 1-D EGM:
      Pass A (M-lines): for each (K' on grid, z), compute L_a_deriv(m'), invert L_a', map back to current M,
                        and write L_a_deriv on the structured (M,K,z) grid (column jK).
                        Skipped if invest_m=False.
      Pass B (K-lines): for each (M' on grid, z), compute L_k_deriv(K'), invert L_k', map back to current K,
                        and write L_k_deriv on the structured grid (row mi).
                        Skipped if invest_k=False.

    Output:
      m_policy, k_policy (next-period policies on current grid),
      Vm_new, Vk_new (updated marginal value functions on current grid).
    """
    Nm, Nk, Nz = Vk.shape
    EVm = expected_marginals(Vm, Pi)   # (Nd,NK,Nz) : E_z'[V_D' | z]
    EVk = expected_marginals(Vk, Pi)

    # implied L_a_deriv, L_k_deriv functions
    L_a_deriv_hat = np.zeros_like(Vm)
    L_k_deriv_hat = np.zeros_like(Vk)

    # ----- PASS A: along M' for each (K' on grid, Z) - skip if no customer investment
    if invest_m:
        for iz in range(Nz):
            for jk, kp in enumerate(k_grid):
                # From FOC: L_a'(a,m) = EVm * beta / (W*P_M)
                # adjusted for exogenous exit
                L_a_deriv = EVm[:, jk, iz] * (1-entry_perc) * beta / (W * P_M)
                a_line = L_a_prime_inv(L_a_deriv, alpha_a, z_a)  # invert L_a'(a) to get a
                a_line = np.maximum(a_line, 0.0) # no negative advertising
                m_line = a_line/P_M # advertising to customers conversion rate
                M_now_of_Mp = (m_grid - m_line) / (1.0 - delta_m) # current M implied by each M'
                Mp_of_M = _invert_monotone_map(m_grid, M_now_of_Mp, m_grid) # inverse map: D' as a function of current D (on structured grid)
                L_a_deriv_hat[:, jk, iz] = np.interp(Mp_of_M, m_grid, L_a_deriv) # interpolate Wm(M') onto current D-grid at column jK

    # ----- PASS B: along K' for each (M' on grid, Z) - skip if no capital investment
    if invest_k:
        for iz in range(Nz):
            for im, mp in enumerate(m_grid):
                # From FOC: L_k'(i) = EVk * beta / W
                # adjusted for exogenous exit (* 1 - entry_perc)
                L_k_deriv = EVk[im, :, iz] * (1-entry_perc) * beta / W
                i_line = L_k_prime_inv(L_k_deriv, alpha_k, z_k)  # invert L_k'(i) to get i
                i_line = np.maximum(i_line, 0.0) # no negative investment
                K_now_of_Kp = (k_grid - i_line) / (1.0 - delta_k)
                Kp_of_K = _invert_monotone_map(k_grid, K_now_of_Kp, k_grid)
                L_k_deriv_hat[im, :, iz] = np.interp(Kp_of_K, k_grid, L_k_deriv)

    # Recover a, i on CURRENT grid via FOC inverses on L_a_deriv_hat, L_k_deriv_hat
    MM, KK = np.meshgrid(m_grid, k_grid, indexing="ij")

    if invest_m:
        a_grid = L_a_prime_inv(L_a_deriv_hat, alpha_a, z_a)
        a_grid = np.maximum(a_grid, 0.0) # no negative advertising
        m_policy = (1.0 - delta_m) * MM[:, :, None] + a_grid/P_M
    else:
        # No customer investment: m stays constant at grid value
        m_policy = MM[:, :, None] * np.ones((1, 1, Nz))

    if invest_k:
        i_grid = L_k_prime_inv(L_k_deriv_hat, alpha_k, z_k)
        i_grid = np.maximum(i_grid, 0.0) # no negative investment
        k_policy = (1.0 - delta_k) * KK[:, :, None] + i_grid
    else:
        # No capital investment: k stays constant at grid value
        k_policy = KK[:, :, None] * np.ones((1, 1, Nz))

    # Envelope updates (mirror your existing formula)
    Vm_new = np.empty_like(Vm)
    Vk_new = np.empty_like(Vk)
    for iz in range(Nz):
        zval = z_grid[iz]
        Vm0 = pi_M(MM, KK, zval, c_agg, sigma, W, gamma_k, gamma_l, phi)
        Vk0 = pi_K(MM, KK, zval, c_agg, sigma, W, gamma_k, gamma_l, phi)
        if invest_m:
            Vm_new[:, :, iz] = Vm0 + (1.0 - delta_m) * L_a_deriv_hat[:, :, iz] * W * P_M
        else:
            Vm_new[:, :, iz] = Vm0  # No continuation value from customer investment
        if invest_k:
            Vk_new[:, :, iz] = Vk0 + (1.0 - delta_k) * L_k_deriv_hat[:, :, iz] * W
        else:
            Vk_new[:, :, iz] = Vk0  # No continuation value from capital investment

    return m_policy, k_policy, Vm_new, Vk_new

def solve_vf_egm(m_grid, k_grid, z_grid, Pi, c_agg, W, P_M, beta, entry_perc,
                 sigma, delta_m, delta_k,
                 gamma_k, gamma_l, alpha_k, z_k, alpha_a, z_a, phi,
                 maxit=100, tol=1e-9, verbose=False,
                 invest_m=True, invest_k=True):
    """
    Solve value function using EGM.

    If invest_m=False, firms do not invest in customers (m fixed).
    If invest_k=False, firms do not invest in capital (k fixed).
    """
    # initial guess: pi_K, and pi_D
    m, k, z = np.meshgrid(m_grid, k_grid, z_grid, indexing='ij')
    Vk = pi_K(m, k, z, c_agg, sigma, W, gamma_k, gamma_l, phi)
    Vm = pi_M(m, k, z, c_agg, sigma, W, gamma_k, gamma_l, phi)
    # iterate
    for it in range(maxit):
        m_pol, k_pol, Vm_new, Vk_new = backward_step_alt1d(
            Vk, Vm, m_grid, k_grid, z_grid, Pi,
            c_agg, W, P_M, beta, entry_perc,
            sigma, delta_m, delta_k,
            gamma_k, gamma_l, alpha_k, z_k, alpha_a, z_a, phi,
            invest_m, invest_k)
        # Only check convergence on the active investment dimensions
        if invest_m and invest_k:
            diff = max(np.max(np.abs(Vm_new - Vm)), np.max(np.abs(Vk_new - Vk)))
        elif invest_m:
            diff = np.max(np.abs(Vm_new - Vm))
        elif invest_k:
            diff = np.max(np.abs(Vk_new - Vk))
        else:
            diff = 0.0  # No investment, nothing to converge
        if verbose & (it>0) & (it % 10 == 0):
            print(f"VF EGM iter {it}: sup-norm diff = {diff:.3e}")
        Vm, Vk = Vm_new, Vk_new
        if diff < tol:
            if verbose:
                print(f"Converged after {it} iterations with sup-norm diff = {diff:.3e}")
            break

    if it == maxit - 1:
        print("Warning: VF EGM did not converge after {} iterations".format(maxit))
        # print diff
        print(f"Final sup-norm diff = {diff:.3e}")
        # raise RuntimeError("VF EGM did not converge after {} iterations".format(maxit))

    return {"Vm": Vm, "Vk": Vk, "m_policy": m_pol, "k_policy": k_pol}
