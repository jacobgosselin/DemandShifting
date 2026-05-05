import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import configparser
import pandas as pd
from ss_solver.prod_fncts import *
from ss_solver.solve_eqm import EqmParams, solve_ss_equilibrium_least_squares
from ss_solver.solve_vf import discretize_productivity, discretize_choices


_DIR = os.path.dirname(os.path.abspath(__file__))
_default_figures = os.path.join(
    os.path.expanduser("~"),
    "Library/CloudStorage/"
    "GoogleDrive-jacob.gosselin@u.northwestern.edu/"
    "My Drive/research_ideas/negative_earnings/figures/misc",
)
FIGURES_DIR = os.environ.get("DEMANDSHIFT_FIGURES_DIR", _default_figures)

_gcfg = configparser.ConfigParser()
_gcfg.read_string("[grid]\n" + open(os.path.join(_DIR, "grid_config.txt")).read())
_choice_low    = float(_gcfg["grid"]["choice_low"])
_choice_high   = float(_gcfg["grid"]["choice_high"])
_n_choice_grid = int(_gcfg["grid"]["n_choice_grid"])
_n_prod_grid   = int(_gcfg["grid"]["n_prod_grid"])

_struct = pd.read_csv(os.path.join(_DIR, "data", "structural_parameters.csv"))
_gamma_l   = float(_struct["gamma_l"].iloc[0])
_gamma_k   = float(_struct["gamma_k"].iloc[0])
_rho       = float(_struct["rho"].iloc[0])
_sigma_eps = float(_struct["sigma_xi"].iloc[0])
_exit_rate = float(_struct["exit_rate"].iloc[0])

_cal = pd.read_csv(os.path.join(_DIR, "data", "calibrated_investment_params.csv"))
_alpha_a = float(_cal["alpha_a"].iloc[0])
_alpha_k = float(_cal["alpha_k"].iloc[0])

_m_grid = discretize_choices(_choice_low, _choice_high, _n_choice_grid, type="exp")
_k_grid = discretize_choices(_choice_low, _choice_high, _n_choice_grid, type="exp")
_z_grid, _, _Pi = discretize_productivity(_rho, _sigma_eps, _n_prod_grid)

z_vals = np.linspace(0.6, 1.4, 5)

palette_2 = [cm.inferno(x) for x in np.linspace(0.0, 0.9, 2)]

def _make_hom_params(z_k_val=1.0, z_a_val=1.0):
    return EqmParams(
        phi=0.0,
        entry_perc=_exit_rate,
        alpha_a=_alpha_a,
        alpha_k=_alpha_k,
        z_k=z_k_val,
        z_a=z_a_val,
        fixed_cost=0.0,
        gamma_k=_gamma_k,
        gamma_l=_gamma_l,
    )

def _sweep_consumption(z_vals, vary="z_k"):
    c_agg_vals = []
    start = np.array([1.0, 1.0, 1.0])
    for zv in z_vals:
        params = _make_hom_params(z_k_val=zv) if vary == "z_k" else _make_hom_params(z_a_val=zv)
        eqm = solve_ss_equilibrium_least_squares(
            _m_grid, _k_grid, _z_grid, _Pi, params,
            start=start, verbose=False,
        )
        start = np.array([eqm["c_agg"], eqm["W"], eqm["P_M"]])
        c_agg_vals.append(eqm["c_agg"])
        print(f"  {vary}={zv:.3f}: c_agg={eqm['c_agg']:.4f}  W={eqm['W']:.4f}  "
              f"P_M={eqm['P_M']:.4f}  ok={eqm['ls_success']}")
    return c_agg_vals

print("\n" + "="*60)
print("Homogeneity exercise: sweeping z_k")
print("="*60)
c_zk = _sweep_consumption(z_vals, vary="z_k")

print("\n" + "="*60)
print("Homogeneity exercise: sweeping z_a")
print("="*60)
c_za = _sweep_consumption(z_vals, vary="z_a")

fig_hom, ax_hom = plt.subplots(figsize=(10, 10))
ax_hom.plot(z_vals, c_zk, "o-", linewidth=5, markersize=10,
            color=palette_2[1], label=r"Scaling Inv.")
ax_hom.plot(z_vals, c_za, "s-", linewidth=5, markersize=10,
            color=palette_2[0], label=r"Scaling Adv.")
ax_hom.set_xlabel("Scale Parameter", fontsize=32)
ax_hom.set_ylabel(r"Consumption", fontsize=32)
ax_hom.set_title("")
ax_hom.legend(fontsize=32)
ax_hom.grid(True, alpha=0.3)
# set ticksize to 18
ax_hom.tick_params(axis='both', which='major', labelsize=18)
fig_hom.tight_layout()
_hom_path = os.path.join(FIGURES_DIR, "homogeneity_ex.pdf")
fig_hom.savefig(_hom_path, dpi=150, bbox_inches="tight")
plt.close(fig_hom)
print(f"\n  Saved homogeneity figure → {_hom_path}")
