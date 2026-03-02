
import numpy as np
from numba import njit

@njit
def c_i_star(mi, ki, zi, c_agg, sigma, W, gamma_k, gamma_l, phi):
    """
    Optimal output for firm with states (mi, ki, zi).
    Production: Y = Z * K^gamma_k * L^gamma_l  (arbitrary-scale Cobb-Douglas)
    Cost:       C(Y) = W * (Y / (Z * K^gamma_k))^(1/gamma_l)
    Demand:     C_it = M_it^(1+phi) * P_it^(-sigma) * C

    Lambda = sigma / ((1 - gamma_l) * sigma + gamma_l)
    Y_i = (gamma_l / W * (M^(1+phi)*C)^(1/sigma) * (Z*K^gamma_k)^(1/gamma_l) * (1-1/sigma))^(gamma_l * Lambda)
    """
    Lambda = sigma / ((1.0 - gamma_l) * sigma + gamma_l)
    base = (
        (1.0 - 1.0 / sigma)
        * c_agg ** (1.0 / sigma)
        * mi ** ((1.0 + phi) / sigma)
        * (zi * ki ** gamma_k) ** (1.0 / gamma_l)
        * gamma_l / W
    )
    expo = gamma_l * Lambda
    return base ** expo

@njit
def profit_analytic(mi, ki, zi, c_agg, sigma, W, gamma_k, gamma_l, phi):
    """
    Analytical profit: pi = P*Y - W*L_s(Y)
    """
    output = c_i_star(mi, ki, zi, c_agg, sigma, W, gamma_k, gamma_l, phi)
    cost = W * L_s(output, zi, ki, gamma_k, gamma_l)
    price = (mi ** (1.0 + phi) * c_agg) ** (1.0 / sigma) * output ** (-1.0 / sigma)
    return price * output - cost

@njit
def e_k(sigma, gamma_k, gamma_l, phi):
    """
    Profit elasticities wrt capital (ek) and customers (em).

    Lambda = sigma / ((1 - gamma_l) * sigma + gamma_l)
    ek = gamma_k * (Lambda - 1) / gamma_l  =  gamma_k * (sigma - 1) / ((1-gamma_l)*sigma + gamma_l)
    em = (1 + phi) * Lambda / sigma         =  (1 + phi) / ((1-gamma_l)*sigma + gamma_l)
    """
    Lambda = sigma / ((1.0 - gamma_l) * sigma + gamma_l)
    ek = gamma_k * (Lambda - 1.0) / gamma_l
    em = (1.0 + phi) * Lambda / sigma
    return ek, em

@njit
def pi_K(mi, ki, zi, c_agg, sigma, W, gamma_k, gamma_l, phi):
    """Partial derivative of profit wrt capital K."""
    Lambda = sigma / ((1.0 - gamma_l) * sigma + gamma_l)
    ek = gamma_k * (Lambda - 1.0) / gamma_l
    pi = profit_analytic(mi, ki, zi, c_agg, sigma, W, gamma_k, gamma_l, phi)
    return ek * pi / ki

@njit
def pi_M(mi, ki, zi, c_agg, sigma, W, gamma_k, gamma_l, phi):
    """Partial derivative of profit wrt customers M."""
    Lambda = sigma / ((1.0 - gamma_l) * sigma + gamma_l)
    em = (1.0 + phi) * Lambda / sigma
    pi = profit_analytic(mi, ki, zi, c_agg, sigma, W, gamma_k, gamma_l, phi)
    return em * pi / mi

@njit
def L_s(y, z, k, gamma_k, gamma_l):
    """
    Labor required for production (supply side).
    From Y = Z * K^gamma_k * L^gamma_l  =>  L = (Y / (Z * K^gamma_k))^(1/gamma_l)
    """
    return (y / (z * k ** gamma_k)) ** (1.0 / gamma_l)

@njit
def L_k(i, alpha_k, z_k=1.0):
    """
    Labor required for capital investment (DRS form).

    From i = z_k * (L^k)^alpha_k  =>  L^k(i) = (i / z_k)^(1/alpha_k)
    """
    return (i / z_k) ** (1.0 / alpha_k)

@njit
def L_k_prime(i, alpha_k, z_k=1.0):
    """Derivative of L_k wrt investment i for the DRS form."""
    return (1.0 / alpha_k) * z_k ** (-1.0 / alpha_k) * i ** (1.0 / alpha_k - 1.0)

@njit
def L_k_prime_inv(x, alpha_k, z_k=1.0):
    """
    Inverse of L_k_prime for the DRS form.

    Solves x = (1/alpha_k) * z_k^(-1/alpha_k) * i^(1/alpha_k - 1) for i:
      i = (x * alpha_k * z_k^(1/alpha_k))^(alpha_k / (1 - alpha_k))
    """
    i = (x * alpha_k * z_k ** (1.0 / alpha_k)) ** (alpha_k / (1.0 - alpha_k))
    return np.maximum(i, 0.0)

@njit
def L_a(a, alpha_a, z_a=1.0):
    """
    Labor required for advertising (DRS form).

    From a = z_a * (L^a)^alpha_a  =>  L^a(a) = (a / z_a)^(1/alpha_a)
    """
    return (a / z_a) ** (1.0 / alpha_a)

@njit
def L_a_prime(a, alpha_a, z_a=1.0):
    """Derivative of L_a wrt advertising a for the DRS form."""
    return (1.0 / alpha_a) * z_a ** (-1.0 / alpha_a) * a ** (1.0 / alpha_a - 1.0)

@njit
def L_a_prime_inv(x, alpha_a, z_a=1.0):
    """
    Inverse of L_a_prime for the DRS form.

    Solves x = (1/alpha_a) * z_a^(-1/alpha_a) * a^(1/alpha_a - 1) for a:
      a = (x * alpha_a * z_a^(1/alpha_a))^(alpha_a / (1 - alpha_a))
    """
    a = (x * alpha_a * z_a ** (1.0 / alpha_a)) ** (alpha_a / (1.0 - alpha_a))
    return np.maximum(a, 0.0)
