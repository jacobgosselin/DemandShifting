setwd("/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings")
# load libraries
library(fixest)
library(tidyr)
library(dplyr)
library(stringr)
library(readr)
library(readxl)
library(lubridate)

# load data and filter ----------------------------------------------------

load("data/intermediate/compustat_procMU.RData")
ritter_data <- read_excel("data/raw/IPO-age.xlsx") %>%
  mutate(
    permno = `CRSP Perm`,
    # extract founding century as first two digits of founding year
    founding_century = floor(Founding / 100),
    founding = ifelse(founding_century %in% 18:20, Founding, NA)
  ) %>%
  select(permno, founding) %>%
  filter(!is.na(founding)) %>%
  # keep only unique permno-founding pairs
  group_by(permno) %>%
  filter(n() == 1)

# Step 0: Merge Compustat with Ritter Founding Data ----

# merge with Compustat
# ONLY WHEN first permno matches ritter permno
# Get first observed permno per gvkey
permno_IPO <- compustat_procMU %>%
  mutate(
    ipo_year = year(as.Date(ipodate))
  ) %>%
  filter(ipo_year == date) %>%
  select(gvkey, permno) %>%
  # keep only 1-1 matches between gvkey and permno
  group_by(gvkey) %>%
  filter(n_distinct(permno) == 1 & !is.na(permno))

# Match ritter only where first permno corresponds
# and we have a 1-1 match
gvkey_founding <- merge(permno_IPO, ritter_data, by = "permno")

# Merge founding back to full panel by gvkey
analysis_data <- merge(compustat_procMU, gvkey_founding %>% select(gvkey, founding), by = "gvkey", all.x = TRUE)

# Step 1: Construct m_inv, sga spending normalized by industry-year total ----
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
  )

# Step 2: Extract (1) distance from first observation to founding date, and (2) pre-IPO growth rate of m_inv ----

# (1) Distance from founding to first observation and to IPO
ritter_subset <- analysis_data %>%
  filter(!is.na(founding)) %>%
  group_by(gvkey) %>%
  # first, check if gvkey is associated with >1 permno
  mutate(
    n_permno = n_distinct(permno)
  ) %>%
  mutate(
    dist_firstobs_founding = min(date) - founding,
    dist_IPO_founding = ipo_year - founding
  ) %>%
  filter(date >= 1980 & date < 2020) %>%
  group_by(gvkey) %>%
  reframe(
    n_permno = first(n_permno),
    permno = first(permno),
    conm = first(conm),
    dist_firstobs_founding = first(dist_firstobs_founding),
    dist_IPO_founding = first(dist_IPO_founding)
  )

med_dist_IPO <- median(ritter_subset$dist_IPO_founding, na.rm = TRUE)
avg_dist_IPO <- mean(ritter_subset$dist_IPO_founding, trim = 0.05, na.rm = TRUE)

med_dist_firstobs <- median(ritter_subset$dist_firstobs_founding, na.rm = TRUE)
avg_dist_firstobs <- mean(ritter_subset$dist_firstobs_founding, trim = 0.01, na.rm = TRUE)

# (2) Pre-IPO growth rate of m_inv (for firms with at least 2 pre-IPO observations to calculate growth)
preIPO_subset <- analysis_data %>%
  filter(has_IPO == 1) %>%
  group_by(gvkey) %>%
  arrange(gvkey, date) %>%
  mutate(
    m_inv_growth = (m_inv - lag(m_inv)) / lag(m_inv),
    m_inv_growth = ifelse(is.infinite(m_inv_growth) | is.na(m_inv_growth), NA, m_inv_growth)
  ) %>%
  # pre-IPO
  filter(date < ipo_first) %>%
  group_by(gvkey) %>%
  filter(n() > 1) %>%
  reframe(
    n_obs_preIPO = n(),
    m_inv_growth_preIPO = mean(m_inv_growth, na.rm = TRUE)
  )

med_preIPO_growth <- median(preIPO_subset$m_inv_growth_preIPO, na.rm = TRUE)
avg_preIPO_growth <- mean(preIPO_subset$m_inv_growth_preIPO, trim = 0.01, na.rm = TRUE)

# Step 3: Compute founding_imputed and age on the FULL panel (after Step 1 filter only) ----
# IMPORTANT: m_stock must be computed on the full history before applying analysis filters,
# matching the original 3b_Est_Structural.R behavior.
analysis_data_mstock_input <- analysis_data %>%
  group_by(gvkey) %>%
  mutate(
    founding_imputed = ifelse(is.na(founding), date - med_dist_firstobs, founding),
    age = date - founding_imputed
  ) %>%
  ungroup() %>%
  arrange(gvkey, date) %>%
  select(gvkey, date, naics_2digit, naics_3digit, sale, ebitda, m_inv, sga_PandT, cogs, capx, age)

# Step 4: Analysis-filtered version for 3c_exog_params.R (exogenous param targets) ----
analysis_data_prepped <- analysis_data_mstock_input %>%
  filter(!is.na(ebitda) & !is.na(sale) & !is.na(sga_PandT) & !is.na(cogs) & !is.na(naics_2digit)) %>%
  filter(date >= 1980 & date < 2020) %>%
  filter(!(naics_2digit %in% c(22, 52, 99)))

# Save outputs for downstream use (Python JIT + R scripts)
# analysis_data_mstock_input: full history per firm, used by compute_coefs_byyear functions
# analysis_data_prepped: analysis-filtered version, used by 3c_exog_params.R
write.csv(analysis_data_mstock_input, "data/intermediate/analysis_data_mstock_input.csv", row.names = FALSE)
write.csv(analysis_data_prepped,      "data/intermediate/analysis_data_prepped.csv",      row.names = FALSE)
write.csv(
  data.frame(med_preIPO_growth = med_preIPO_growth),
  "data/intermediate/prepped_scalars.csv",
  row.names = FALSE
)
save(
  analysis_data_mstock_input, analysis_data_prepped, med_preIPO_growth, med_dist_firstobs,
  file = "data/intermediate/analysis_data_prepped.RData"
)

cat("3b_mstock_prep.R complete.\n")
cat("Rows in mstock_input (full history):", nrow(analysis_data_mstock_input), "\n")
cat("Rows in prepped (analysis-filtered):", nrow(analysis_data_prepped), "\n")
cat("Firms in mstock_input:", n_distinct(analysis_data_mstock_input$gvkey), "\n")
cat("med_preIPO_growth:", med_preIPO_growth, "\n")
cat("med_dist_firstobs:", med_dist_firstobs, "\n")
