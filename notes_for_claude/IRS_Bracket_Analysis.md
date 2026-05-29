# IRS SOI Bracket Analysis — Context for Claude

## Core Question

These scripts investigate the **"missing middle"**: the rise of superstar firms appears to have come at the expense of mid-size firms. The brackets that have *shrunk* in terms of their share of returns/receipts are the middle receipt/asset brackets — those that held the majority of firms as of 1980. The bottom (very small) bracket and top (large) bracket grow, while the middle hollows out.

This is distinct from the paper's main result about negative earnings — it's a separate distributional story about where the action is in the size distribution, using IRS data that covers all US corporations (not just Compustat).

---

## Data Sources

All raw data lives in `data/raw/` under the Google Drive working directory.

| File | Used by | Contents |
|------|---------|----------|
| `agg_brackets_receipts_R5.dta` | `distributional_decomp.R`, `lognormal_code.R` | Aggregate (all-sector) corporate returns, binned by receipt (revenue) size bracket, by year |
| `agg_brackets_assets_R5.dta` | `distributional_decomp_assets.R` | Aggregate corporate returns, binned by **asset** size bracket, by year |
| `sector_brackets_receipts_R5.dta` | `distributional_decomp_sector.R` | Corporate returns by **sector**, binned by receipt size, by year |

All `.dta` files were pulled/processed by `pull_IRS.R` and `proc_IRS.py` / `digitize_irs_pdfs.py`.

---

## `distributional_decomp.R` — Receipts Brackets, All Sectors

**Core brackets (8, consistent across all 39 post-1980 years):**
`$0–$25K | $25K–$100K | $100K–$500K | $500K–$1M | $1M–$5M | $5M–$10M | $10M–$50M | $50M+`

The raw data has more fine-grained brackets in some years; these are collapsed to the 8 core brackets via `findInterval`.

**Figures produced (`figures/empirical/`):**

| Figure | Description |
|--------|-------------|
| `bracket_return_shares_all.pdf` | All 8 brackets, % of total returns, 1980–2019 |
| `bracket_return_shares_3group.pdf` | Collapsed to 3 groups: `<$25K`, `$25K–$1M`, `>$1M` — most direct view of the missing middle |
| `bracket_flow_firms.pdf` | Stacked area: share of firms by 6 collapsed groups over time |
| `bracket_pct_receipts_change.pdf` | % change in receipts share relative to 1980 baseline, by bracket |
| `bracket_return_shares_4group.pdf` | 5 finer groups |

**1983+ robustness section:** Re-runs the same plots using a 9th bracket (`$100K–$250K` split out) available from 1983 onward, producing `_1983` suffix versions of each figure.

---

## `distributional_decomp_assets.R` — Asset Brackets, All Sectors

Same analysis but stratified by **total assets** rather than receipts. Confirms the missing middle pattern is not a receipts measurement artifact.

**Core brackets (10):**
`$0 | <$500K | $500K–$1M | $1M–$5M | $5M–$10M | $10M–$25M | $25M–$50M | $50M–$100M | $100M–$250M | $250M+`

The `$0` bracket (firms with zero reported assets) is tracked separately — these are likely pure intangible/service firms and their growth is itself interesting.

**Figures produced:**

| Figure | Description |
|--------|-------------|
| `bracket_return_shares_all_assets.pdf` | All 10 brackets, 1980–2019 |
| `bracket_return_shares_3group_assets.pdf` | Collapsed: `No Assets`, `<$500K`, `>$500K` |
| `bracket_return_shares_4group_assets.pdf` | 6-group collapsed version |
| `bracket_return_shares_all_assets_1970_2000.pdf` | 12-bracket version, 1970–2000 subsample |
| `bracket_return_shares_6group_assets_1970_2000.pdf` | 6-group 1970–2000 version |

---

## `distributional_decomp_sector.R` — Receipts Brackets, By Sector

Replicates the receipts-bracket analysis separately for each IRS sector (excluding Finance, Utilities, and aggregate "All"). Shows whether the missing middle pattern is common across industries or concentrated in specific sectors.

Drops sector × year cells where IRS suppressed or merged brackets (`bracket_deletion_total == "yes"`). Uses the same 8 core receipt brackets.

**Figures produced (all faceted by sector):**

| Figure | Description |
|--------|-------------|
| `bracket_return_shares_all_bysector.pdf` | All 8 brackets, faceted |
| `bracket_return_shares_3group_bysector.pdf` | 3-group collapsed, faceted — clearest view of the missing middle by sector |
| `bracket_return_shares_4group_bysector.pdf` | 5-group collapsed, faceted |

---

## `lognormal_code.R` — Piecewise Log-Normal Fit to the Receipts Distribution

Fits a **piecewise log-normal distribution** to the bracketed receipt data to recover a smooth CDF and compute top-share statistics (top 50%, 10%, 1%, 0.01%) for each year. This interpolates within and above the top bracket where raw bracket shares are too coarse.

**Methodology:**
- For intermediate brackets: fits log-normal(μ, σ) per bin using two CDF values (`log_normal_fit_2` — exactly identified from two bracket boundaries)
- For the top (open-ended) bracket: fits using the threshold and empirical tail mean (`log_normal_fit` — minimizes squared distance between implied and observed conditional mean)
- `final_fit` combines both: `log_normal_fit_2` for all bins except last, `log_normal_fit` for the last
- `top_1_pct` uses the piecewise fit to compute the receipts share of the top X% of firms

**Key functions:**
- `ln_cond_mean(lower, upper, mu, sigma)`: conditional mean of a log-normal on `[lower, upper]`
- `sigma_given_mu(mu, cdf_min, thresh_low)`: recovers σ consistent with the empirical CDF at a bracket boundary, given a μ guess
- `log_normal_fit` / `log_normal_fit_2`: two identification strategies (tail mean vs. two CDF points)
- `final_fit`: combines both strategies across all bins for a given year × sector
- `top_1_pct`: computes top-X% receipts share from the piecewise fit
- `piecewise_lognormal_cdf`: generates a smooth (x, CDF) curve from the fitted bins for plotting
- `main_output`: main loop — iterates over year × sector cells, fits distribution, computes top shares, stores per-bin fit parameters

**Outputs:**
- Per-year CDF plots: `figures/empirical/log_normal_receipts/lognormal_all_{year}.pdf`
- 1980 vs. 2018 overlay (full distribution + right-tail zoom): `lognormal_all_1980_vs_2018.pdf`

**Note:** `main_output` supports sector-level iteration but is currently run on aggregate data only (`sector_main = "All"`). Sector-level log-normal fitting is not yet wired up.

---

## Relationship to the Paper

- The IRS percent-negative-earnings series feeds **Figure 2** and **Table 1** directly.
- The bracket/distributional decomp work is exploratory/appendix material. The missing-middle framing connects to the paper's result that the sales and earnings distribution has *spread* (Section 2, Figures 4–6 in the draft) — the IRS data shows the same hollowing-out of the middle in a broader universe of firms.
- The log-normal fitting is a methodological extension to recover continuous distributional statistics from coarse bracket data, in the spirit of Pareto/log-normal interpolation from the inequality literature.
