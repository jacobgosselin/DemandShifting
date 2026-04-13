setwd("/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings")
# load libraries
library(fixest)
library(tidyr)
library(dplyr)
library(stringr)
library(DescTools)
library(readr)
library(readxl)
library(lubridate)

# Load data ---------------------------------------------------------------

compustat_postMU <- read.csv("data/intermediate/compustat_postMU.csv")

ritter_data <- read_excel("data/raw/IPO-age.xlsx") %>%
  mutate(
    permno = `CRSP Perm`,
    founding_century = floor(Founding / 100),
    founding = ifelse(founding_century %in% 18:20, Founding, NA)
  ) %>%
  select(permno, founding) %>%
  filter(!is.na(founding)) %>%
  group_by(permno) %>%
  filter(n() == 1)

cust_capital <- read.csv("data/raw/misc/cust_capital_he_NEW.csv") %>%
  rename(naics_3digit = NAICS,
         cust_capital = ISM.Rev..median.) %>%
  select(naics_3digit, cust_capital)

# Step 0: Merge Compustat with Ritter Founding Data -----------------------

permno_IPO <- compustat_postMU %>%
  mutate(
    ipo_year = year(as.Date(ipodate))
  ) %>%
  filter(ipo_year == date) %>%
  select(gvkey, permno) %>%
  group_by(gvkey) %>%
  filter(n_distinct(permno) == 1 & !is.na(permno))

gvkey_founding <- merge(permno_IPO, ritter_data, by = "permno")

analysis_data <- merge(compustat_postMU, gvkey_founding %>% select(gvkey, founding), by = "gvkey", all.x = TRUE)

# Step 1: Construct m_inv, sga spending normalized by industry-year total -

analysis_data <- analysis_data %>%
  filter(!is.na(sga_PandT) & !is.na(naics)) %>%
  mutate(
    naics_3digit = as.numeric(str_sub(naics, 1, 3)),
    naics_2digit = as.numeric(str_sub(naics, 1, 2)),
    sga = sga_PandT
  ) %>%
  group_by(gvkey) %>%
  mutate(
    has_IPO = ifelse(any(!is.na(ipodate)), 1, 0),
    ipo_first = first(ipodate),
    ipo_year = year(as.Date(ipodate)),
    n_permno = n_distinct(permno),
    obs_pre_founding = first(date) < founding
  ) %>%
  group_by(naics_2digit, date) %>%
  mutate(
    m_inv = sga / sum(sga, na.rm = TRUE)
    # m_inv = sga
  ) %>%
  group_by(gvkey) %>%
  arrange(gvkey, date)

# Step 2: Pre-IPO growth rate of m_inv ------------------------------------

preIPO_subset <- analysis_data %>%
  filter(has_IPO == 1) %>%
  group_by(gvkey) %>%
  arrange(gvkey, date) %>%
  mutate(
    m_inv_growth = (m_inv - lag(m_inv)) / lag(m_inv),
    m_inv_growth = ifelse(is.infinite(m_inv_growth) | is.na(m_inv_growth), NA, m_inv_growth)
  ) %>%
  filter(date < ipo_first) %>%
  group_by(gvkey) %>%
  filter(n() > 1) %>%
  reframe(
    n_obs_preIPO = n(),
    m_inv_growth_preIPO = mean(m_inv_growth, na.rm = TRUE)
  )

med_preIPO_growth <- median(preIPO_subset$m_inv_growth_preIPO, na.rm = TRUE)
avg_preIPO_growth <- mean(preIPO_subset$m_inv_growth_preIPO, trim = 0.05, na.rm = TRUE)

# Step 3: Compute founding_imputed, age, and m_stock on full panel --------
# IMPORTANT: m_stock must be built on full firm history before analysis filters

ritter_subset <- analysis_data %>%
  filter(!is.na(founding)) %>%
  group_by(gvkey) %>%
  mutate(dist_firstobs_founding = min(date) - founding) %>%
  filter(date >= 1980 & date < 2020) %>%
  group_by(gvkey) %>%
  reframe(dist_firstobs_founding = first(dist_firstobs_founding))

med_dist_firstobs <- median(ritter_subset$dist_firstobs_founding, na.rm = TRUE)
avg_dist_firstobs <- mean(ritter_subset$dist_firstobs_founding, trim = 0.05, na.rm = TRUE)
delta_m <- 0.15  # fixed; not a calibration target

analysis_data <- analysis_data %>%
  group_by(gvkey) %>%
  arrange(gvkey, date) %>%
  mutate(
    founding_imputed = ifelse(is.na(founding), date - avg_dist_firstobs, founding),
    age = date - founding_imputed,
    m_stock = {
      delta <- delta_m
      g     <- avg_preIPO_growth
      r     <- (1 - delta) / (1 + g)
      init_val <- first(m_inv) * (1 - r^first(age)) / (1 - r)
      Reduce(function(prev, curr) (1 - delta) * prev + curr,
             m_inv[-1],
             init = init_val,
             accumulate = TRUE)
    }
  ) 
  # %>%
  # ungroup() %>%
  # group_by(naics_2digit, date) %>%
  # mutate( 
  #   m_stock = m_stock / sum(m_stock, na.rm = TRUE) # normalize to sum to 1 within industry-year
  # )


# Step 4: Apply analysis filters ------------------------------------------

analysis_data <- analysis_data %>%
  filter(!is.na(ebitda) & !is.na(sale) & !is.na(cogs) & !is.na(capx)) %>%
  filter(!(naics_2digit %in% c(22, 52, 99)))

# Step 5: Add empirical variables -----------------------------------------

analysis_data <- analysis_data %>%
  mutate(
    neg_ebitda    = as.numeric(ebitda < 0),
    neg_pi        = as.numeric(pi < 0),
    neg_ni        = as.numeric(ni < 0),
    neg_profits   = as.numeric((sale - cogs - xsga - capx) < 0),
    rd            = xrd + rdip,
    rd_sale_raw   = rd / sale,
    sga_sale_raw  = sga / sale,
    cogs_sale_raw = cogs / sale,
    capx_sale_raw = capx / sale,
    mu            = mu_v
  ) %>%
  group_by(date) %>%
  mutate(
    ebitda    = Winsorize(ebitda,          na.rm = TRUE, probs = c(0, 1)),
    ebitda_sale = Winsorize(ebitda / sale, na.rm = TRUE, probs = c(0, 1)),
    rd_sale   = Winsorize(rd_sale_raw,     na.rm = TRUE, probs = c(0, 1)),
    sga_sale  = Winsorize(sga_sale_raw,    na.rm = TRUE, probs = c(0, 1)),
    cogs_sale = Winsorize(cogs_sale_raw,   na.rm = TRUE, probs = c(0, 1)),
    capx_sale = Winsorize(capx_sale_raw,   na.rm = TRUE, probs = c(0, 1))
  ) %>%
  ungroup() %>%
  group_by(naics_2digit, date) %>%
  mutate(
    sales_share = sale / sum(sale, na.rm = TRUE)
  ) %>%
  ungroup() %>%
  group_by(gvkey) %>%
  arrange(gvkey, date) %>%
  mutate(
    # generate spells and cap since only have data going back to 1961
    neg_spell = sequence(rle(neg_ebitda)$lengths) * neg_ebitda,
    neg_spell_cap_flag = ifelse(neg_spell > 19, 1, 0), # cap at 20 years, but keep track of whether cap was hit
    neg_spell = ifelse(neg_spell > 19, 19, neg_spell), 
    neg_pi_spell = sequence(rle(neg_pi)$lengths) * neg_pi,
    neg_pi_spell_cap_flag = ifelse(neg_pi_spell > 19, 1, 0), # cap at 20 years, but keep track of whether cap was hit
    neg_pi_spell = ifelse(neg_pi_spell > 19, 19, neg_pi_spell),
    neg_ni_spell = sequence(rle(neg_ni)$lengths) * neg_ni,
    neg_ni_spell_cap_flag = ifelse(neg_ni_spell > 19, 1, 0), # cap at 20 years, but keep track of whether cap was hit
    neg_ni_spell = ifelse(neg_ni_spell > 19, 19, neg_ni_spell),
    neg_profits_spell = sequence(rle(neg_profits)$lengths) * neg_profits,
    neg_profits_spell_cap_flag = ifelse(neg_profits_spell > 19, 1, 0), # cap at 20 years, but keep track of whether cap was hit
    neg_profits_spell = ifelse(neg_profits_spell > 19, 19, neg_profits_spell)
  ) %>%
  ungroup() %>%
  left_join(cust_capital, by = "naics_3digit") %>%
  # Don't filter date until AFTER computing m_stock and neg_spell, which rely on full firm history
  filter(date >= 1980 & date < 2020)

# Save --------------------------------------------------------------------

save(analysis_data, med_preIPO_growth, file = "data/clean/analysis_data.RData")

cat("3a_build_analysis_data.R complete.\n")
cat("Rows:", nrow(analysis_data), "\n")
cat("Firms:", n_distinct(analysis_data$gvkey), "\n")
cat("Years:", min(analysis_data$date), "-", max(analysis_data$date), "\n")
cat("delta_m (fixed):", delta_m, "\n")
cat("med_preIPO_growth:", med_preIPO_growth, "\n")
cat("avg_preIPO_growth:", avg_preIPO_growth, "\n")
