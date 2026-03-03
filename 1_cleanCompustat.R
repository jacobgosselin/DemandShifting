setwd("/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings")
# load libraries
library(fixest)
library(tidyr)
library(dplyr)
library(haven)
library(readxl)
library(stringr)
load("data/raw/compustat_raw.RData")

# clean data (pre-mark-up) ----------------------------------------------------

# download FRED price deflator
# library(fredr)
# fredr_set_key("d6eb0bdcb3327f31c7a0e35e48b60df9") 
# # IPDNBS, annual
# deflator <- fredr(series_id = "GDPDEF", observation_start = as.Date("1961-01-01"), observation_end = as.Date("2024-01-01"), frequency = "a") %>%
#   select(date, value) %>%
#   rename(deflator = value) %>%
#   mutate(fyear = as.numeric(format(date, "%Y"))) %>%
#   select(fyear, deflator)
# save(deflator, file = "data/raw/deflator.RData")
load("data/raw/deflator.RData")
compustat <- merge(compustat, deflator, by = "fyear")

# keep variables of interest
vars_of_interest <- c("gvkey", "conm", "fyear", "datadate", "indfmt", "state", "naics", "ebitda", "ni", "pi",  "sale", "cogs", "xsga", "xrd", "rdip", "capx", "k_int", "k_int_org", "k_int_know", "ppegt", "ppent", "deflator", "oiadp", "oibdp", "revt", "xopr", "txpd", "at", "dp", "xint", "mkt_val", "cusip", "ipodate", "permno")

# select variables of interest
# and put all nominal variables in real terms (deflate by GDP deflator, then *100)
compustat_preMU <- compustat %>%
  # deflate all nominal variables
  mutate(mkt_val = prcc_f * csho + dltt + dlc - act,
         mkt_val = mkt_val / deflator * 100,
         ebitda = ebitda/deflator * 100,
         ni = ni / deflator * 100,
         pi = pi / deflator * 100,
         oiadp = oiadp / deflator * 100,
         oibdp = oibdp / deflator * 100,
         dp = dp / deflator * 100,
         xint = xint/deflator * 100,
         sale = sale / deflator * 100,
         cogs = cogs / deflator * 100,
         xsga = xsga / deflator * 100,
         xrd = xrd / deflator * 100,
         rdip = rdip / deflator * 100,
         capx = capx / deflator * 100,
         txpd = txpd / deflator * 100,
         k_int = k_int / deflator * 100,
         k_int_org = k_int_org / deflator * 100,
         k_int_know = k_int_know / deflator * 100,
         ppegt = ppegt / deflator * 100,
         ppent = ppent / deflator * 100,
         at = at / deflator * 100) %>%
  select(all_of(vars_of_interest))


# generate intangible investment + investment rates
# according to Peters&Taylor, org_inv = .3*sga_PandT, know_inv = xrd + rdip
# compute etr according to Dyreng et al (2008), etr = txpd/pi, winsorized at 0 and 1
compustat_preMU <- compustat_preMU %>%
  arrange(gvkey, fyear) %>%
  group_by(gvkey) %>%
  mutate(xrd = ifelse(is.na(xrd), 0, xrd),
         rdip = ifelse(is.na(rdip), 0, rdip),
         sga_PandT = ifelse((xrd>xsga & xrd<cogs) | is.na(xsga), xsga, xsga - xrd - rdip),
         naics_3digit = substring(naics, 1, 3),
         inv_org = .3*sga_PandT,
         inv_know = xrd + rdip,
         inv_phys = capx,
         K_Tot = k_int + ppegt,
         etr = txpd/pi,
         etr = ifelse(etr < 0, 0, ifelse(etr > 1, 1, etr)))
         

# load 100y tax data
tax_100y <- read_dta("data/raw/100Y_DATA.dta")
tax_100y <- tax_100y %>%
  select(year, state, corporate_income_tax_rate, corporate_min_tax_rate)

compustat_preMU <- merge(compustat_preMU, tax_100y, by.x = c("fyear", "state"), by.y = c("year", "state"), all.x = TRUE)

# clean for mark-up estimation ------------------------------------------

# R code matching the stata cleaning code from DGM 
# Parameters from the .do file 
indvar   <- "naics"  # industry variable used in Stata
ind_lev  <- 2        # keep first 2 digits (line 164)
firm_var <- "gvkey"  # local firm (line 130)
year_var <- "fyear"  # local year (line 134)
min_year <- 1950     # (line 176)
max_year <- 2019     # (line 177)
min_obs  <- 100      # (line 190)

# Flags in the do-file we mirror here
drop_estimates <- TRUE     # drop orbil/orcr=="E" if exist
interpolate_one_gap <- TRUE
apply_deflator <- "deflator" %in% names(compustat_preMU)  # only if already merged

# Helper: single-gap interpolation (lead/lag average) within firm ----
interp_single_gap <- function(x) {
  # replace NA at t with avg of lead/lag when both exist; otherwise keep x
  lagx  <- dplyr::lag(x)
  leadx <- dplyr::lead(x)
  fill  <- ifelse(is.na(x) & is.finite(lagx) & is.finite(leadx), (lagx + leadx)/2, x)
  fill
}

compustat_procMU <- compustat_preMU %>%
  # 1) Year window & firm-year duplicates drop
  filter(.data[[year_var]] >= min_year, .data[[year_var]] <= max_year) %>%
  arrange(.data[[firm_var]], .data[[year_var]]) %>%
  distinct(.data[[firm_var]], .data[[year_var]], .keep_all = TRUE) %>%
  # 2) Build sector from NAICS prefix of length ind_lev
  mutate(
    naics_chr = as.character(.data[[indvar]]),
    sector_raw = if_else(nchar(naics_chr) >= ind_lev, str_sub(naics_chr, 1L, ind_lev), NA_character_),
    sector = suppressWarnings(as.integer(sector_raw))   # mirror `destring`
  ) %>%
  # 3) Optional: drop estimated values
  {
    if (drop_estimates && all(c("orbil", "orcr") %in% names(.))) {
      filter(., !(orbil == "E" | orcr == "E"))
    } else .
  } %>%
  # 4) Deflate (already done)
  # 5) Interpolate single-year holes for selected vars within firm
  {
    if (interpolate_one_gap) {
      group_by(., .data[[firm_var]]) |>
        arrange(.data[[year_var]], .by_group = TRUE) |>
        mutate(
          cogs = interp_single_gap(cogs),
          sale = interp_single_gap(sale),
          xsga = interp_single_gap(xsga)
        ) |>
        ungroup()
    } else .
  } %>%
  # 6) --- INDUSTRY-LEVEL COUNT FILTER (lines 247–251) ---
  # keep only sectors with at least `min_obs` observations (over the whole sample, like the .do)
  group_by(sector) %>%
  mutate(count_sector = n()) %>%
  ungroup() %>%
  filter(is.na(sector) | count_sector >= min_obs) %>%  # drop tiny sectors; keep NA sector unchanged if desired
  # 7) --- Multiple industry rows per firm-year clean-up (lines 253–257) ---
  # Create an industry group id (egen group) then drop extra rows where firm-year repeats
  mutate(ind = as.integer(factor(sector))) %>%
  group_by(.data[[firm_var]], .data[[year_var]]) %>%
  mutate(nrobs = n()) %>%
  ungroup() %>%
  # Drop if (nrobs==2 | 3) & indfmt=="FS" (mirror Stata)
  {
    if ("indfmt" %in% names(.)) {
      filter(., !(nrobs %in% c(2L, 3L) & indfmt == "FS"))
    } else .
  } %>%
  arrange(.data[[firm_var]], .data[[year_var]]) %>%
  # Final firm-year dedup in case anything slipped through
  distinct(.data[[firm_var]], .data[[year_var]], .keep_all = TRUE) %>%
  # 8) Create analysis variables and logs
  mutate(
    varcost = cogs, 
    capital = ppent,
    fixcost = xsga
  ) %>%
  # 9) --- Trim on sales/varcost within year ---
  mutate(s_g = sale / varcost) %>%
  group_by(.data[[year_var]]) %>%
  mutate(
    s_g_p_1  = quantile(s_g, 0.01, na.rm = TRUE, type = 7),
    s_g_p_99 = quantile(s_g, 0.99, na.rm = TRUE, type = 7)
  ) %>%
  ungroup() %>%
  filter(is.finite(s_g), s_g > 0, s_g > s_g_p_1, s_g < s_g_p_99) %>%
  select(-s_g, -s_g_p_1, -s_g_p_99) %>%
  # 10) Final variables (logs)
  mutate(
    firmid  = .data[[firm_var]],
    date    = .data[[year_var]],
    v = if_else(varcost > 0, log(varcost), NA_real_),
    k = if_else(capital > 0, log(capital), NA_real_),
    x = if_else(fixcost > 0, log(fixcost), NA_real_),
    y = if_else(sale    > 0, log(sale),    NA_real_)
  ) %>%
  arrange(sector, firmid, date)

write.csv(compustat_procMU, "data/intermediate/compustat_preMU.csv", row.names = FALSE)
