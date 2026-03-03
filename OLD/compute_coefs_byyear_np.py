"""
compute_coefs_byyear_np.py

JIT-compiled numpy implementation of the two-way FE regression:
    feols(log(sale) ~ log(m_stock):i(date) | date:naics_2digit + gvkey)

Replicates the R/fixest result for the point estimates (coefficients) without
standard errors.  Used in the calibration inner loop and model_implied_trends.py
so that delta_m can be varied cheaply.

Verification: run both with delta_m=0.15 and compare outputs.
"""

import numpy as np
import pandas as pd
import os
from numba import njit

MAIN_DIR = (
    "/Users/jacobgosselin/Library/CloudStorage/"
    "GoogleDrive-jacob.gosselin@u.northwestern.edu/"
    "My Drive/research_ideas/negative_earnings"
)

# ---------------------------------------------------------------------------
# JIT-compiled helpers
# ---------------------------------------------------------------------------

@njit(cache=True)
def _demean_two_way(v, firm_id, indyear_id, n_firms, n_indyear,
                    tol=1e-10, max_iter=5000):
    """
    Partial out two fixed effects (firm + industry-year) via alternating
    projections (Gauss-Seidel).  Converges to the within-group residuals.

    Parameters
    ----------
    v         : 1-D float64 array, N observations
    firm_id   : 1-D int64 array, firm group index (0-based)
    indyear_id: 1-D int64 array, industry-year group index (0-based)
    n_firms   : int, number of distinct firms
    n_indyear : int, number of distinct industry-year cells

    Returns
    -------
    out : 1-D float64 array, demeaned residuals
    """
    N = len(v)
    out = v.copy()
    firm_sum  = np.empty(n_firms)
    firm_cnt  = np.empty(n_firms)
    iy_sum    = np.empty(n_indyear)
    iy_cnt    = np.empty(n_indyear)

    for _ in range(max_iter):
        prev_max = 0.0

        # --- demean by firm ---
        for k in range(n_firms):
            firm_sum[k] = 0.0
            firm_cnt[k] = 0.0
        for i in range(N):
            f = firm_id[i]
            firm_sum[f] += out[i]
            firm_cnt[f] += 1.0
        for i in range(N):
            out[i] -= firm_sum[firm_id[i]] / firm_cnt[firm_id[i]]

        # --- demean by industry-year ---
        for k in range(n_indyear):
            iy_sum[k] = 0.0
            iy_cnt[k] = 0.0
        for i in range(N):
            iy = indyear_id[i]
            iy_sum[iy] += out[i]
            iy_cnt[iy] += 1.0
        for i in range(N):
            out[i] -= iy_sum[indyear_id[i]] / iy_cnt[indyear_id[i]]

        # --- convergence check: max absolute change from previous demeaning step
        # (We track the iy demeaning step's contribution as a proxy.)
        for k in range(n_indyear):
            contrib = abs(iy_sum[k] / iy_cnt[k]) if iy_cnt[k] > 0.0 else 0.0
            if contrib > prev_max:
                prev_max = contrib
        if prev_max < tol:
            break

    return out


@njit(cache=True)
def _construct_m_stock(m_inv, ages, firm_start, firm_len, n_firms,
                       delta_m, g):
    """
    Perpetual-inventory construction of customer capital for all firms.

    M_{i,0}  = m_inv_{i,0} * (1 - r^{age_{i,0}}) / (1 - r),  r = (1-delta)/(1+g)
    M_{i,t}  = (1 - delta_m) * M_{i,t-1} + m_inv_{i,t}      for t >= 1

    Parameters
    ----------
    m_inv      : 1-D float64, investment values sorted by (gvkey, date)
    ages       : 1-D float64, firm age at each observation
    firm_start : 1-D int64, index of first row for each firm
    firm_len   : 1-D int64, number of rows for each firm
    n_firms    : int
    delta_m    : float, depreciation rate
    g          : float, median pre-IPO growth rate of m_inv

    Returns
    -------
    m_stock : 1-D float64, same length as m_inv
    """
    r     = (1.0 - delta_m) / (1.0 + g)
    denom = 1.0 - r
    n     = len(m_inv)
    result = np.empty(n)

    for f in range(n_firms):
        s    = firm_start[f]
        L    = firm_len[f]
        age0 = ages[s]
        mi0  = m_inv[s]

        if abs(denom) < 1e-12:
            # r ≈ 1: geometric series degenerates to arithmetic sum
            init_val = mi0 * age0
        else:
            init_val = mi0 * (1.0 - r ** age0) / denom

        result[s] = init_val
        for t in range(1, L):
            result[s + t] = (1.0 - delta_m) * result[s + t - 1] + m_inv[s + t]

    return result


# ---------------------------------------------------------------------------
# Module-level data cache (loaded once, reused across calibration iterations)
# ---------------------------------------------------------------------------

_prepped_df       = None
_med_preIPO_growth = None


def _load_data():
    global _prepped_df, _med_preIPO_growth
    if _prepped_df is None:
        # Load the full firm histories (pre-analysis-filter) so that m_stock
        # is built on complete histories before filtering, matching R behavior.
        _prepped_df = pd.read_csv(
            os.path.join(MAIN_DIR, "data", "intermediate", "analysis_data_mstock_input.csv")
        )
    if _med_preIPO_growth is None:
        scalars = pd.read_csv(
            os.path.join(MAIN_DIR, "data", "intermediate", "prepped_scalars.csv")
        )
        _med_preIPO_growth = float(scalars["med_preIPO_growth"].iloc[0])


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def compute_coefs_byyear_numpy(delta_m, prepped_df=None, med_preIPO_growth=None):
    """
    Compute year-specific sales-elasticity coefficients of customer capital.

    Replicates:
        feols(log(sale) ~ log(m_stock):i(date) | date:naics_2digit + gvkey)

    Uses FWL theorem: partial out firm + industry-year FEs via alternating
    projections, then OLS on demeaned data.

    Parameters
    ----------
    delta_m           : float, depreciation rate for m_stock construction
    prepped_df        : pd.DataFrame or None; if None, loaded from disk
    med_preIPO_growth : float or None; if None, loaded from disk

    Returns
    -------
    coefs : np.ndarray, shape (T, 2), columns = [year, coef]
    """
    # --- Load data if not supplied ---
    if prepped_df is None or med_preIPO_growth is None:
        _load_data()
    df  = _prepped_df.copy() if prepped_df is None else prepped_df.copy()
    g   = _med_preIPO_growth  if med_preIPO_growth is None else float(med_preIPO_growth)

    # --- Sort by (gvkey, date) to prepare for perpetual-inventory ---
    df = df.sort_values(["gvkey", "date"]).reset_index(drop=True)

    # Firm boundary arrays for JIT perpetual inventory
    gvkey_arr   = df["gvkey"].values
    is_new_firm = np.concatenate([[True], gvkey_arr[1:] != gvkey_arr[:-1]])
    firm_start  = np.where(is_new_firm)[0].astype(np.int64)
    firm_len    = np.diff(np.concatenate([firm_start, [len(df)]])).astype(np.int64)
    n_firms_raw = len(firm_start)

    # --- Construct m_stock ---
    m_stock = _construct_m_stock(
        df["m_inv"].values.astype(np.float64),
        df["age"].values.astype(np.float64),
        firm_start, firm_len, n_firms_raw,
        float(delta_m), float(g),
    )
    df["m_stock"] = m_stock

    # --- Analysis filters: applied AFTER m_stock construction, matching
    # original 3b_Est_Structural.R where perpetual inventory precedes filters ---
    df = df[
        df["sale"].notna() & df["ebitda"].notna() & df["cogs"].notna()
    ].copy()
    df = df[(df["date"] >= 1980) & (df["date"] < 2020)].copy()
    df = df[~df["naics_2digit"].isin([22, 52, 99])].reset_index(drop=True)

    # --- Drop obs where sale or m_stock are missing or <= 0 (avoids log warnings) ---
    df = df[(df["sale"] > 0) & (df["m_stock"] > 0)].reset_index(drop=True)

    df["log_sale"]    = np.log(df["sale"].values)
    df["log_m_stock"] = np.log(df["m_stock"].values)

    # --- Singleton filters (same order as R/fixest) ---
    # 1. Firms must have > 1 observation
    firm_cnts = df.groupby("gvkey")["gvkey"].transform("count").values
    df = df[firm_cnts > 1].reset_index(drop=True)
    # 2. Industry-year cells must have > 1 observation
    iy_key   = df["date"].astype(str) + "_" + df["naics_2digit"].astype(str)
    iy_cnts  = iy_key.groupby(iy_key).transform("count").values
    df = df[iy_cnts > 1].reset_index(drop=True)

    N = len(df)

    # --- Integer group IDs for JIT ---
    firm_id_arr, _ = pd.factorize(df["gvkey"])
    firm_id_arr    = firm_id_arr.astype(np.int64)
    iy_key2        = df["date"].astype(str) + "_" + df["naics_2digit"].astype(str)
    iy_id_arr, _   = pd.factorize(iy_key2)
    iy_id_arr      = iy_id_arr.astype(np.int64)
    n_firms    = int(firm_id_arr.max()) + 1
    n_indyear  = int(iy_id_arr.max()) + 1

    # --- Demean y (log sale) ---
    y_raw = df["log_sale"].values.astype(np.float64)
    y_dm  = _demean_two_way(y_raw, firm_id_arr, iy_id_arr, n_firms, n_indyear)

    # --- Year-specific regressors: x_t = log(m_stock) * 1[year == t] ---
    years   = np.sort(df["date"].unique())
    T       = len(years)
    year_arr = df["date"].values
    lm      = df["log_m_stock"].values.astype(np.float64)

    # Build demeaned regressor matrix
    X_dm = np.empty((N, T), dtype=np.float64)
    for j, yr in enumerate(years):
        x_t = np.where(year_arr == yr, lm, 0.0)
        X_dm[:, j] = _demean_two_way(x_t, firm_id_arr, iy_id_arr, n_firms, n_indyear)

    # --- OLS: β = (X'X)^{-1} X'y ---
    XtX  = X_dm.T @ X_dm   # (T, T)
    Xty  = X_dm.T @ y_dm   # (T,)
    beta = np.linalg.solve(XtX, Xty)

    return np.column_stack([years.astype(np.float64), beta])


# ---------------------------------------------------------------------------
# Warm-up: trigger JIT compilation on module import so first calibration
# iteration isn't slow due to compilation overhead.
# ---------------------------------------------------------------------------

def _warmup_jit():
    """Pre-compile JIT functions with small dummy data."""
    n = 20
    v   = np.random.randn(n)
    fid = np.zeros(n, dtype=np.int64)
    fid[n//2:] = 1
    iy  = np.arange(n, dtype=np.int64) % 5
    _demean_two_way(v, fid, iy, 2, 5)

    m_inv = np.abs(np.random.randn(n)) + 0.01
    ages  = np.ones(n, dtype=np.float64) * 5.0
    fs    = np.array([0, 10], dtype=np.int64)
    fl    = np.array([10, 10], dtype=np.int64)
    _construct_m_stock(m_inv, ages, fs, fl, 2, 0.15, 0.2)


_warmup_jit()


# ---------------------------------------------------------------------------
# CLI entry point for quick verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    delta_m_arg = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
    print(f"Computing coefficients with delta_m = {delta_m_arg}")
    coefs = compute_coefs_byyear_numpy(delta_m_arg)
    print("year       coef")
    for row in coefs:
        print(f"  {int(row[0])}    {row[1]:.6f}")
