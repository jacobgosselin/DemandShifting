setwd("/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings")
# load libraries
library(fixest)
library(tidyr)
library(dplyr)
library(stringr)
library(texreg)
library(readr)

# load data ----------------------------------------------------

load("data/clean/analysis_data.RData")

# Build sector-year dataset (3-digit NAICS, n_firms >= 5) -----------------

sector_year_data <- analysis_data %>%
  group_by(naics_3digit, date) %>%
  reframe(
    med_mu = median(mu, na.rm = TRUE),
    pct_negative = mean(ebitda < 0),
    p5 = quantile(ebitda, 0.05),
    median_ebitda = median(ebitda),
    p95 = quantile(ebitda, 0.95),
    median_capx_sale = median(capx_sale),
    median_cogs_sale = median(cogs_sale),
    median_sga_sale = median(sga_sale),
    cust_capital = first(cust_capital),
    sd_ebitda = sd(ebitda),
    n_firms = n(),
    .groups = "drop"
  ) %>%
  mutate(
    right_tail = abs(p95 - median_ebitda),
    left_tail = abs(median_ebitda - p5)
  ) %>%
  filter(n_firms >= 5)

# Build sector changes: late (2015-2019) minus early (1980-1984) -----------

sector_changes <- sector_year_data %>%
  mutate(period = case_when(
    date >= 1980 & date <= 1984 ~ "early",
    date >= 2015 & date <= 2019 ~ "late",
    TRUE ~ NA_character_
  )) %>%
  filter(!is.na(period)) %>%
  group_by(naics_3digit, period) %>%
  summarize(
    pct_negative = mean(pct_negative, na.rm = TRUE),
    sd_ebitda = mean(sd_ebitda, na.rm = TRUE),
    left_tail = mean(left_tail, na.rm = TRUE),
    right_tail = mean(right_tail, na.rm = TRUE),
    med_mu = mean(med_mu, na.rm = TRUE),
    cust_capital = first(cust_capital),
    .groups = "drop"
  ) %>%
  pivot_wider(
    names_from = period,
    values_from = c(pct_negative, sd_ebitda, left_tail, right_tail, med_mu)
  ) %>%
  mutate(
    delta_pct_negative = pct_negative_late - pct_negative_early,
    delta_sd_ebitda    = log(sd_ebitda_late) - log(sd_ebitda_early),
    delta_left_tail    = log(left_tail_late) - log(left_tail_early),
    delta_right_tail   = log(right_tail_late) - log(right_tail_early),
    delta_med_mu       = log(med_mu_late) - log(med_mu_early)
  ) %>%
  filter(!is.na(delta_pct_negative))

# Sector-level panel regressions (year FE) --------------------------------

reg_neg_earnings  <- feols(pct_negative ~ cust_capital | date, data = sector_year_data)
reg_sd_ebitda     <- feols(log(sd_ebitda) ~ cust_capital | date, data = sector_year_data)
reg_left_tail     <- feols(log(left_tail) ~ cust_capital | date, data = sector_year_data)
reg_right_tail    <- feols(log(right_tail) ~ cust_capital | date, data = sector_year_data)
reg_med_mu        <- feols(log(med_mu) ~ cust_capital | date, data = sector_year_data)

# Sector-level change regressions (cross-sectional) -----------------------

reg_change_neg_earnings <- feols(delta_pct_negative ~ cust_capital, data = sector_changes)
reg_change_sd_ebitda    <- feols(delta_sd_ebitda ~ cust_capital,    data = sector_changes)
reg_change_left_tail    <- feols(delta_left_tail ~ cust_capital,    data = sector_changes)
reg_change_right_tail   <- feols(delta_right_tail ~ cust_capital,   data = sector_changes)
reg_change_med_mu       <- feols(delta_med_mu ~ cust_capital,       data = sector_changes)

# Output tables -----------------------------------------------------------

texreg(list(
  reg_neg_earnings,
  reg_sd_ebitda
), custom.model.names = c("Neg. EBITDA", "Std. Dev. EBITDA (logged)"),
  omit.coef = "Constant",
  custom.coef.map = list("cust_capital" = "$I^{SM}$/Rev. (He et. al. 2025)"),
  file = "tables/sector_level_regs.tex",
  table = FALSE,
  include.rsquared = FALSE,
  include.adjrs = FALSE,
  include.groups = FALSE)

texreg(list(
  reg_change_neg_earnings,
  reg_change_sd_ebitda
), custom.coef.map = list("cust_capital" = "$I^{SM}$/Rev. (He et. al. 2025)"),
   custom.model.names = c("Neg. EBITDA", "Std. Dev. EBITDA (logged)"),
   omit.coef = "Constant",
   file = "tables/sector_level_change_regs.tex",
   table = FALSE,
   include.rsquared = FALSE,
   include.adjrs = FALSE,
   include.ngroups = FALSE)
