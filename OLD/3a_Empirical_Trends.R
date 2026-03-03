setwd("/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings")
# load libraries
library(fixest)
library(tidyr)
library(dplyr)
library(stringr)
library(DescTools)
library(ggplot2)
library(viridis)
library(texreg)
library(readr)

# load data and filter ----------------------------------------------------

compustat_clean <- read.csv("data/clean/compustat_clean.csv")
# save as RDS for faster loading next time
# saveRDS(compustat_clean, "data/clean/compustat_clean.rds")
compustat_clean <- readRDS("data/clean/compustat_clean.rds")

cust_capital <- read.csv("data/raw/misc/cust_capital_he_NEW.csv")
cust_capital <- cust_capital %>%
  rename(naics_3digit = NAICS,
         cust_capital = ISM.Rev..median.) %>%
  select(naics_3digit, cust_capital)

analysis_data <- compustat_clean %>%
  filter(!is.na(ebitda) & !is.na(sale) & !is.na(sga_PandT) & !is.na(cogs) & !is.na(naics)) %>%
  mutate(
    neg_ebitda = as.numeric(ebitda < 0),
    naics_3digit = as.numeric(str_sub(naics, 1, 3)),
    naics_2digit = as.numeric(str_sub(naics, 1, 2)),
    sga = sga_PandT, # cleansing SGA of R&D
    rd = xrd + rdip, # combining R&D
    rd_sale_raw = rd/sale,
    sga_sale_raw = sga/sale,
    cogs_sale_raw = cogs/sale,
    capx_sale_raw = capx/sale,
    mu = mu_v) %>%
  # Winsorize by year cross-section (not pooled across all years)
  group_by(date) %>%
  mutate(
    ebitda = Winsorize(ebitda, na.rm = TRUE, probs = c(0, 1)),
    ebitda_sale = Winsorize(ebitda/sale, na.rm = TRUE, probs = c(0, 1)),
    rd_sale = Winsorize(rd_sale_raw, na.rm = TRUE, probs = c(0, 1)),
    sga_sale = Winsorize(sga_sale_raw, na.rm = TRUE, probs = c(0, 1)),
    cogs_sale = Winsorize(cogs_sale_raw, na.rm = TRUE, probs = c(0, 1)),
    capx_sale = Winsorize(capx_sale_raw, na.rm = TRUE, probs = c(0, 1))
  ) %>%
  ungroup() %>%
  group_by(naics_2digit, date) %>%
  mutate(
    sales_share = sale / sum(sale, na.rm = TRUE),
  ) %>%
  ungroup() %>% 
  group_by(gvkey) %>% arrange(gvkey, date) %>%
  mutate(
    neg_spell = sequence(rle(neg_ebitda)$lengths) * neg_ebitda,
    neg_spell = ifelse(neg_spell > 10, 10, neg_spell)
  ) %>%
  left_join(cust_capital, by = "naics_3digit") %>%
  filter(date >= 1980 & date <= 2019) %>%
  filter(!(naics_2digit %in% c(22, 52, 99)))

# print number of firms and number of observations 
num_firms <- n_distinct(analysis_data$gvkey)
num_observations <- nrow(analysis_data)
cat("Number of firms:", num_firms, "\n")
cat("Number of observations:", num_observations, "\n")

# Common theme for all plots
theme_common <- theme_minimal(base_size = 18) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "bottom"
  )

# 2a. Plot percent of firms with negative earnings by year ---------------

neg_earnings_by_year <- analysis_data %>%
  group_by(date) %>%
  reframe(
    pct_negative = mean(ebitda < 0) * 100,
    n_firms = n()
  )

ggplot(neg_earnings_by_year, aes(x = date, y = pct_negative)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  labs(
    title = "",
    x = "Year",
    y = "Percent of Firms with EBITDA < 0 (%)",
    caption = paste0("N firms ranges from ", min(neg_earnings_by_year$n_firms),
                     " to ", max(neg_earnings_by_year$n_firms))
  ) +
  theme_common

ggsave("figures/pct_negative_earnings_by_year.pdf", width = 10, height = 10)

# compare to IRS data
irs_corp_returns_post94 <- read.csv("data/clean/irs_corp_returns.csv") %>%
  mutate(perc_neg_earnings = perc_neg_earnings * 100) %>%
  filter(year >= 1994 & year <= 2019)

irs_corps_returns_pre94 <- read.csv("data/clean/irs_corp_returns_pre94.csv") %>%
  mutate(perc_neg_earnings = perc_neg_earnings * 100) %>%
  filter(year >= 1980 & year < 1994)

irs_corps_returns <- bind_rows(irs_corps_returns_pre94, irs_corp_returns_post94)

# Calculate offset to align starting points
compustat_1980 <- neg_earnings_by_year %>% filter(date == 1980) %>% pull(pct_negative)
irs_1980 <- irs_corps_returns %>% filter(year == 1980) %>% pull(perc_neg_earnings)
offset <- compustat_1980 - irs_1980

# Transform IRS data to align visually with Compustat
irs_corps_returns <- irs_corps_returns %>%
  mutate(perc_neg_adjusted = perc_neg_earnings + offset)

ggplot() +
  geom_line(data = neg_earnings_by_year, aes(x = date, y = pct_negative, color = "Compustat"), linewidth = 2) +
  geom_point(data = neg_earnings_by_year, aes(x = date, y = pct_negative, color = "Compustat"), size = 3) +
  geom_line(data = irs_corps_returns, aes(x = year, y = perc_neg_adjusted, color = "IRS"), linewidth = 2) +
  geom_point(data = irs_corps_returns, aes(x = year, y = perc_neg_adjusted, color = "IRS"), size = 3) +
  scale_y_continuous(
    name = "Compustat: % Firms with EBITDA < 0",
    sec.axis = sec_axis(~ . - offset, name = "IRS: % Returns with No Net Income (Excluding 1120-S)")
  ) +
  scale_color_manual(values = c("Compustat" = "black", "IRS" = "steelblue")) +
  labs(
    x = "Year",
    color = "Data Source"
  ) +
  theme_common

ggsave("figures/pct_negative_earnings_compustat_vs_irs.pdf", width = 10, height = 10)

# 2b. Plot average and median neg_spell among negative earning firms over time ----

neg_spell_by_year <- analysis_data %>%
  filter(neg_ebitda == 1) %>%
  group_by(date) %>%
  reframe(
    mean_neg_spell = mean(neg_spell, na.rm = TRUE),
    n_firms = n()
  )

ggplot(neg_spell_by_year, aes(x = date, y = mean_neg_spell)) +
  geom_line(linewidth = 2, color = "black") +
  geom_point(size = 3, color = "black") +
  labs(
    title = "",
    x = "Year",
    y = "Average Negative Earnings Spell Length (Years)",
    caption = paste0("N negative earning firms ranges from ", min(neg_spell_by_year$n_firms),
                     " to ", max(neg_spell_by_year$n_firms))
  ) +
  theme_common

ggsave("figures/neg_spell_over_time.pdf", width = 10, height = 10)

# 3a. Plot log change in SD of sales over time ------------------------------------

sd_sales_by_year <- analysis_data %>%
  group_by(date) %>%
  reframe(
    sd_sales = sd(sale)
  ) %>%
  mutate(
    log_sd = log(sd_sales)
  )
# Get 1980 values for normalization
base_1980_sales_sd <- sd_sales_by_year %>% filter(date == 1980)
# Normalize to 1980 (so line starts at 0)
sd_sales_normalized <- sd_sales_by_year %>%
  mutate(
    log_change_sd = log_sd - base_1980_sales_sd$log_sd
  )
ggplot(sd_sales_normalized, aes(x = date, y = log_change_sd)) +
  geom_line(linewidth = 2, color = "black") +
  geom_point(size = 3, color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_x_continuous(breaks = seq(1980, 2020, 5)) +
  labs(
    title = "",
    subtitle = "",
    x = "Year",
    y = "Std. Deviation (Log Change from 1980)"
  ) +
  theme_common

ggsave("figures/sd_sales_by_year.pdf", width = 10, height = 10)

# 3b. Top Percentiles of sales over time ------------------------

top_decile_sales_measures <- analysis_data %>%
  group_by(date) %>%
  reframe(
    median = median(sale),
    p75 = quantile(sale, 0.75),
    p90 = quantile(sale, 0.90),
    p95 = quantile(sale, 0.95),
    p99 = quantile(sale, 0.99)
  ) %>%
  mutate(
    log_dist_p75 = log(abs(p75 - median)),
    log_dist_p90 = log(abs(p90 - median)),
    log_dist_p95 = log(abs(p95 - median)),
    log_dist_p99 = log(abs(p99 - median))
  )

# Get 1980 values for normalization
base_1980_top_sales <- top_decile_sales_measures %>% filter(date == 1980)

# Normalize to 1980 (so all lines start at 0)
top_decile_sales_normalized <- top_decile_sales_measures %>%
  mutate(
    log_change_p75 = log_dist_p75 - base_1980_top_sales$log_dist_p75,
    log_change_p90 = log_dist_p90 - base_1980_top_sales$log_dist_p90,
    log_change_p95 = log_dist_p95 - base_1980_top_sales$log_dist_p95,
    log_change_p99 = log_dist_p99 - base_1980_top_sales$log_dist_p99
  )

# Reshape for plotting
top_decile_sales_long <- top_decile_sales_normalized %>%
  select(date, log_change_p75, log_change_p90, log_change_p95, log_change_p99) %>%
  pivot_longer(cols = starts_with("log_change"),
               names_to = "percentile",
               values_to = "log_change")

ggplot(top_decile_sales_long, aes(x = date, y = log_change, color = percentile, group = percentile)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_x_continuous(breaks = seq(1980, 2020, 5)) +
  scale_color_manual(
    values = setNames(viridis::plasma(4, begin = 0.1, end = 0.9),
                      c("log_change_p75", "log_change_p90", "log_change_p95", "log_change_p99")),
    labels = c("log_change_p75" = "|P75 - Median|", "log_change_p90" = "|P90 - Median|", "log_change_p95" = "|P95 - Median|", "log_change_p99" = "|P99 - Median|"),
    breaks = c("log_change_p75", "log_change_p90", "log_change_p95", "log_change_p99")
  ) +
  labs(
    title = "",
    subtitle = "",
    x = "Year",
    y = "Log Change from 1980",
    color = ""
  ) +
  theme_common

ggsave("figures/top_quantile_sales_over_time.pdf", width = 10, height = 10)

# 3c. Bottom percentiles of sales over time -------------------

bottom_quantile_sales_measures <- analysis_data %>%
  group_by(date) %>%
  reframe(
    median = median(sale),
    p25 = quantile(sale, 0.25),
    p10 = quantile(sale, 0.10),
    p5 = quantile(sale, 0.05),
    p1 = quantile(sale, 0.01)
  ) %>%
  mutate(
    log_dist_p25 = log(abs(median - p25)),
    log_dist_p10 = log(abs(median - p10)),
    log_dist_p5 = log(abs(median - p5)),
    log_dist_p1 = log(abs(median - p1))
  )

# Get 1980 values for normalization
base_1980_bottom_sales <- bottom_quantile_sales_measures %>% filter(date == 1980)

# Normalize to 1980 (so all lines start at 0)
bottom_quantile_sales_normalized <- bottom_quantile_sales_measures %>%
  mutate(
    log_change_p25 = log_dist_p25 - base_1980_bottom_sales$log_dist_p25,
    log_change_p10 = log_dist_p10 - base_1980_bottom_sales$log_dist_p10,
    log_change_p5 = log_dist_p5 - base_1980_bottom_sales$log_dist_p5,
    log_change_p1 = log_dist_p1 - base_1980_bottom_sales$log_dist_p1
  )

# Reshape for plotting
bottom_quantile_sales_long <- bottom_quantile_sales_normalized %>%
  select(date, log_change_p25, log_change_p10, log_change_p5, log_change_p1) %>%
  pivot_longer(cols = starts_with("log_change"),
               names_to = "percentile",
               values_to = "log_change")

ggplot(bottom_quantile_sales_long, aes(x = date, y = log_change, color = percentile, group = percentile)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_x_continuous(breaks = seq(1980, 2020, 5)) +
  scale_color_manual(
    values = setNames(viridis::plasma(4, begin = 0.1, end = 0.9),
                      c("log_change_p25", "log_change_p10", "log_change_p5", "log_change_p1")),
    labels = c("log_change_p25" = "|Median - P25|", "log_change_p10" = "|Median - P10|", "log_change_p5" = "|Median - P5|", "log_change_p1" = "|Median - P1|"),
    breaks = c("log_change_p25", "log_change_p10", "log_change_p5", "log_change_p1")
  ) +
  labs(
    title = "",
    subtitle = "",
    x = "Year",
    y = "Log Change from 1980",
    color = ""
  ) +
  theme_common

ggsave("figures/bottom_quantile_sales_over_time.pdf", width = 10, height = 10)

# 4a. Plot log change in SD of earnings over time -------------------------

sd_by_year <- analysis_data %>%
  group_by(date) %>%
  reframe(
    sd_ebitda = sd(ebitda)
  ) %>%
  mutate(
    log_sd = log(sd_ebitda)
  )

# Get 1980 values for normalization
base_1980_sd <- sd_by_year %>% filter(date == 1980)

# Normalize to 1980 (so line starts at 0)
sd_normalized <- sd_by_year %>%
  mutate(
    log_change_sd = log_sd - base_1980_sd$log_sd
  )

ggplot(sd_normalized, aes(x = date, y = log_change_sd)) +
  geom_line(linewidth = 2, color = "black") +
  geom_point(size = 3, color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_x_continuous(breaks = seq(1980, 2020, 5)) +
  labs(
    title = "",
    subtitle = "",
    x = "Year",
    y = "Std. Deviation (Log Change from 1980)"
  ) +
  theme_common

ggsave("figures/sd_earnings_by_year.pdf", width = 10, height = 10)

# 4b. Top Percentiles of earnings over time ------------------------

top_decile_measures <- analysis_data %>%
  group_by(date) %>%
  reframe(
    median = median(ebitda_sale),
    p75 = quantile(ebitda_sale, 0.75),
    p90 = quantile(ebitda_sale, 0.90),
    p95 = quantile(ebitda_sale, 0.95),
    p99 = quantile(ebitda_sale, 0.99)
  ) %>%
  mutate(
    log_dist_p75 = log(abs(p75 - median)),
    log_dist_p90 = log(abs(p90 - median)),
    log_dist_p95 = log(abs(p95 - median)),
    log_dist_p99 = log(abs(p99 - median))
  )

# Get 1980 values for normalization
base_1980_top <- top_decile_measures %>% filter(date == 1980)

# Normalize to 1980 (so all lines start at 0)
top_decile_normalized <- top_decile_measures %>%
  mutate(
    log_change_p75 = log_dist_p75 - base_1980_top$log_dist_p75,
    log_change_p90 = log_dist_p90 - base_1980_top$log_dist_p90,
    log_change_p95 = log_dist_p95 - base_1980_top$log_dist_p95,
    log_change_p99 = log_dist_p99 - base_1980_top$log_dist_p99
  )

# Reshape for plotting
top_decile_long <- top_decile_normalized %>%
  select(date, log_change_p75, log_change_p90, log_change_p95, log_change_p99) %>%
  pivot_longer(cols = starts_with("log_change"),
               names_to = "percentile",
               values_to = "log_change")

ggplot(top_decile_long, aes(x = date, y = log_change, color = percentile, group = percentile)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_x_continuous(breaks = seq(1980, 2020, 5)) +
  scale_color_manual(
    values = setNames(viridis::plasma(4, begin = 0.1, end = 0.9),
                      c("log_change_p75", "log_change_p90", "log_change_p95", "log_change_p99")),
    labels = c("log_change_p75" = "|P75 - Median|", "log_change_p90" = "|P90 - Median|", "log_change_p95" = "|P95 - Median|", "log_change_p99" = "|P99 - Median|"),
    breaks = c("log_change_p75", "log_change_p90", "log_change_p95", "log_change_p99")
  ) +
  labs(
    title = "",
    subtitle = "",
    x = "Year",
    y = "Log Change from 1980",
    color = ""
  ) +
  theme_common

ggsave("figures/top_quantile_earnings_over_time.pdf", width = 10, height = 10)

# 4c. Bottom percentiles of earnings over time -------------------

bottom_quantile_measures <- analysis_data %>%
  group_by(date) %>%
  reframe(
    median = median(ebitda_sale),
    p25 = quantile(ebitda_sale, 0.25),
    p10 = quantile(ebitda_sale, 0.10),
    p5 = quantile(ebitda_sale, 0.05),
    p1 = quantile(ebitda_sale, 0.01)
  ) %>%
  mutate(
    log_dist_p25 = log(abs(median - p25)),
    log_dist_p10 = log(abs(median - p10)),
    log_dist_p5 = log(abs(median - p5)),
    log_dist_p1 = log(abs(median - p1))
  )

# Get 1980 values for normalization
base_1980_bottom <- bottom_quantile_measures %>% filter(date == 1980)

# Normalize to 1980 (so all lines start at 0)
bottom_quantile_normalized <- bottom_quantile_measures %>%
  mutate(
    log_change_p25 = log_dist_p25 - base_1980_bottom$log_dist_p25,
    log_change_p10 = log_dist_p10 - base_1980_bottom$log_dist_p10,
    log_change_p5 = log_dist_p5 - base_1980_bottom$log_dist_p5,
    log_change_p1 = log_dist_p1 - base_1980_bottom$log_dist_p1
  )

# Reshape for plotting
bottom_quantile_long <- bottom_quantile_normalized %>%
  select(date, log_change_p25, log_change_p10, log_change_p5, log_change_p1) %>%
  pivot_longer(cols = starts_with("log_change"),
               names_to = "percentile",
               values_to = "log_change")

ggplot(bottom_quantile_long, aes(x = date, y = log_change, color = percentile, group = percentile)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_x_continuous(breaks = seq(1980, 2020, 5)) +
  scale_color_manual(
    values = setNames(viridis::plasma(4, begin = 0.1, end = 0.9),
                      c("log_change_p25", "log_change_p10", "log_change_p5", "log_change_p1")),
    labels = c("log_change_p25" = "|Median - P25|", "log_change_p10" = "|Median - P10|", "log_change_p5" = "|Median - P5|", "log_change_p1" = "|Median - P1|"),
    breaks = c("log_change_p25", "log_change_p10", "log_change_p5", "log_change_p1")
  ) +
  labs(
    title = "",
    subtitle = "",
    x = "Year",
    y = "Log Change from 1980",
    color = ""
  ) +
  theme_common

ggsave("figures/bottom_quantile_earnings_over_time.pdf", width = 10, height = 10)

# 5. Plot cost ratios over time ------------------------------

# Calculate median cost ratios by year
cost_ratios_by_year <- analysis_data %>%
  group_by(date) %>%
  reframe(
    median_rd_sale = median(rd_sale, na.rm = TRUE),
    median_sga_sale = median(sga_sale, na.rm = TRUE),
    median_cogs_sale = median(cogs_sale, na.rm = TRUE),
    median_capx_sale = median(capx_sale, na.rm = TRUE),
    .groups = "drop"
  )

# Plot cost ratios over time
cost_ratios_long <- cost_ratios_by_year %>%
  pivot_longer(cols = c(median_rd_sale, median_sga_sale, median_cogs_sale, median_capx_sale),
               names_to = "cost_type",
               values_to = "median_ratio") 

ggplot(cost_ratios_long, aes(x = date, y = median_ratio, color = cost_type, group = cost_type)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  scale_x_continuous(breaks = seq(1975, 2020, 5)) +
  scale_color_manual(
    values = c("median_rd_sale" = "#E31A1C", "median_sga_sale" = "#1F78B4", "median_cogs_sale" = "#33A02C", "median_capx_sale" = "#FF7F00"),
    labels = c("median_rd_sale" = "R&D/Sales", "median_sga_sale" = "SG&A/Sales", "median_cogs_sale" = "COGS/Sales", "median_capx_sale" = "CapEx/Sales")
  ) +
  labs(
    title = "",
    x = "Year",
    y = "Median Ratio",
    color = "Cost Type"
  ) +
  theme_common

ggsave("figures/cost_ratios_by_year.pdf", width = 10, height = 10)

# same thing but only for ebitda < 0
cost_ratios_neg_ebitda_by_year <- analysis_data %>%
  filter(ebitda < 0) %>%
  group_by(date) %>%
  reframe(
    median_rd_sale = median(rd_sale, na.rm = TRUE),
    median_sga_sale = median(sga_sale, na.rm = TRUE),
    median_cogs_sale = median(cogs_sale, na.rm = TRUE),
    median_capx_sale = median(capx_sale, na.rm = TRUE),
    .groups = "drop"
  )

cost_ratios_neg_long <- cost_ratios_neg_ebitda_by_year %>%
  pivot_longer(cols = c(median_rd_sale, median_sga_sale, median_cogs_sale, median_capx_sale),
               names_to = "cost_type",
               values_to = "median_ratio")

ggplot(cost_ratios_neg_long, aes(x = date, y = median_ratio, color = cost_type, group = cost_type)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  scale_x_continuous(breaks = seq(1975, 2020, 5)) +
  scale_color_manual(
    values = c("median_rd_sale" = "#E31A1C", "median_sga_sale" = "#1F78B4", "median_cogs_sale" = "#33A02C", "median_capx_sale" = "#FF7F00"),
    labels = c("median_rd_sale" = "R&D/Sales", "median_sga_sale" = "SG&A/Sales", "median_cogs_sale" = "COGS/Sales", "median_capx_sale" = "CapEx/Sales")
  ) +
  labs(
    title = "",
    x = "Year",
    y = "Median Ratio",
    color = "Cost Type"
  ) +
  theme_common

ggsave("figures/cost_ratios_neg_ebitda_by_year.pdf", width = 10, height = 10)

# 6. Create sector-year dataset and analyze -----------------------------

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
filter(n_firms >= 5)  # Filter out sector-years with too few firms

# Calculate change in key variables: late (2015-2019) minus early (1980-1984)
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
    delta_sd_ebitda = log(sd_ebitda_late) - log(sd_ebitda_early),
    delta_left_tail = log(left_tail_late) - log(left_tail_early),
    delta_right_tail = log(right_tail_late) - log(right_tail_early),
    delta_med_mu = log(med_mu_late) - log(med_mu_early)
  ) %>%
  filter(!is.na(delta_pct_negative))  # Keep only sectors with both periods

# Sector regs
reg_neg_earnings <- feols(pct_negative ~ cust_capital | date,
                          data = sector_year_data)
reg_sd_ebitda <- feols(log(sd_ebitda) ~ cust_capital | date,
                       data = sector_year_data)
reg_left_tail <- feols(log(left_tail) ~ cust_capital | date,
                       data = sector_year_data)
reg_right_tail <- feols(log(right_tail) ~ cust_capital | date,
                        data = sector_year_data)
reg_med_mu <- feols(log(med_mu) ~ cust_capital | date,
                        data = sector_year_data)

# Sector change regs
reg_change_neg_earnings <- feols(delta_pct_negative ~ cust_capital,
                                 data = sector_changes)
reg_change_sd_ebitda <- feols(delta_sd_ebitda ~ cust_capital,
                              data = sector_changes)
reg_change_left_tail <- feols(delta_left_tail ~ cust_capital,
                              data = sector_changes)
reg_change_right_tail <- feols(delta_right_tail ~ cust_capital,
                               data = sector_changes)
reg_change_med_mu <- feols(delta_med_mu ~ cust_capital,
                               data = sector_changes)

# Texreg results (display)
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

# OLD: Sector-specific earnings distribution analysis ---------------------

# Bar plot of negative earnings by 2-digit NAICS sector
sector_year_data_2digit <- analysis_data %>%
  mutate(naics_2digit = as.numeric(str_sub(naics_3digit, 1, 2))) %>%
  group_by(naics_2digit, date) %>%
  reframe(
    pct_negative = mean(ebitda < 0),
    cust_capital = median(cust_capital, na.rm = TRUE),
    n_firms = n(),
    .groups = "drop"
  ) %>%
  filter(n_firms >= 5)

sector_averages_2digit <- sector_year_data_2digit %>%
  group_by(naics_2digit) %>%
  summarise(
    avg_pct_negative    = mean(pct_negative, na.rm = TRUE),
    pct_negative_late   = mean(pct_negative[date >= 2014], na.rm = TRUE),
    pct_negative_early  = mean(pct_negative[date <= 1980], na.rm = TRUE),
    cust_capital        = first(cust_capital),
    n_years             = n(),
    .groups = "drop"
  ) %>%
  mutate(change_pct_negative = pct_negative_late - pct_negative_early) %>%
  arrange(desc(avg_pct_negative))

# Create long format data for plotting three metrics
sector_2digit_long <- sector_averages_2digit %>%
  select(naics_2digit, pct_negative_late, change_pct_negative, cust_capital) %>%
  pivot_longer(cols = c(pct_negative_late, change_pct_negative, cust_capital),
               names_to = "metric",
               values_to = "value") %>%
  mutate(metric = factor(metric, levels = c("pct_negative_late", "change_pct_negative", "cust_capital")))

ggplot(sector_2digit_long, aes(x = reorder(as.factor(naics_2digit),
                                            -value * (metric == "pct_negative_late")),
                                y = value, fill = metric)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  scale_fill_manual(
    breaks = c("pct_negative_late", "change_pct_negative", "cust_capital"),
    values = c("pct_negative_late" = "steelblue", "change_pct_negative" = "darkgreen", "cust_capital" = "coral"),
    labels = c("pct_negative_late" = "% Negative EBITDA (2014-2019)",
               "change_pct_negative" = "Change 1975-2019",
               "cust_capital" = "Customer Capital")
  ) +
  labs(
    title = "Negative Earnings, Customer Capital, and Change by Sector (2-digit NAICS)",
    subtitle = "2019 values and change from 1975-2019",
    x = "NAICS 2-digit Sector",
    y = "Value",
    fill = "Metric"
  ) +
  theme_common

ggsave("figures/neg_earnings_custcap_by_sector_2digit.pdf", width = 16, height = 9)

# OLD: Earnings concentration among top 10%, 5%, 1%, and 0.1% over time --------

earnings_concentration <- analysis_data %>%
  filter(ebitda > 0) %>%  # Focus on positive earnings for concentration
  group_by(date) %>%
  arrange(desc(ebitda)) %>%
  mutate(
    rank = row_number(),
    n_firms = n(),
    cumsum_ebitda = cumsum(ebitda),
    total_ebitda = sum(ebitda)
  ) %>%
  reframe(
    # Top 10% share
    top10_share = sum(ebitda[rank <= ceiling(0.10 * n_firms)]) / total_ebitda[1] * 100,
    # Top 5% share
    top5_share = sum(ebitda[rank <= ceiling(0.05 * n_firms)]) / total_ebitda[1] * 100,
    # Top 1% share
    top1_share = sum(ebitda[rank <= ceiling(0.01 * n_firms)]) / total_ebitda[1] * 100,
    # Top 0.1% share
    top01_share = sum(ebitda[rank <= ceiling(0.001 * n_firms)]) / total_ebitda[1] * 100,
    n_firms = n_firms[1]
  )

# Reshape for plotting
concentration_long <- earnings_concentration %>%
  select(date, top10_share, top5_share, top1_share, top01_share) %>%
  pivot_longer(cols = c(top10_share, top5_share, top1_share, top01_share),
               names_to = "percentile",
               values_to = "share")

ggplot(concentration_long, aes(x = date, y = share, color = percentile, group = percentile)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  scale_x_continuous(breaks = seq(1980, 2020, 5)) +
  scale_color_manual(
    values = c("top10_share" = "#6A3D9A", "top5_share" = "#1F78B4", "top1_share" = "#E31A1C", "top01_share" = "#33A02C"),
    labels = c("top10_share" = "Top 10%", "top5_share" = "Top 5%", "top1_share" = "Top 1%", "top01_share" = "Top 0.1%")
  ) +
  labs(
    title = "Earnings Concentration Over Time",
    subtitle = "Share of total positive EBITDA held by top earners",
    x = "Year",
    y = "Share of Total Earnings (%)",
    color = "Percentile"
  ) +
  theme_common

ggsave("figures/earnings_concentration_over_time.pdf", width = 16, height = 9)

# 7. Overlayed histograms: early (1980-1984) vs late (2015-2019) earnings distribution
earnings_early_late <- analysis_data %>%
  filter((date >= 1980 & date <= 1984) | (date >= 2015 & date <= 2019)) %>%
  mutate(period = ifelse(date <= 1984, "1980-1984", "2015-2019"))

ggplot(earnings_early_late, aes(x = ebitda, fill = period, color = period)) +
  geom_histogram(aes(y = after_stat(density)), alpha = 0.4, position = "identity", bins = 10000) +
  scale_fill_manual(values = c("1980-1984" = "#1F78B4", "2015-2019" = "#E31A1C")) +
  scale_color_manual(values = c("1980-1984" = "#1F78B4", "2015-2019" = "#E31A1C")) +
  coord_cartesian(xlim = c(earnings_early_late$ebitda %>% quantile(0.1),
                           earnings_early_late$ebitda %>% quantile(0.9))) +
  labs(
    title = "Earnings Distribution: Early vs Late Period",
    subtitle = "1980-1984 vs 2015-2019",
    x = "EBITDA",
    y = "Density",
    fill = "Period",
    color = "Period"
  ) +
  theme_common

ggsave("figures/earnings_distribution_early_vs_late.pdf", width = 16, height = 9)

# Same for Information Sector only
earnings_early_late_info <- analysis_data %>%
  mutate(naics_2digit = as.numeric(str_sub(naics_3digit, 1, 2))) %>%
  filter(naics_2digit == 51) %>%
  filter((date >= 1980 & date <= 1984) | (date >= 2015 & date <= 2019)) %>%
  mutate(period = ifelse(date <= 1984, "1980-1984", "2015-2019"))

ggplot(earnings_early_late_info, aes(x = ebitda, fill = period, color = period)) +
  geom_histogram(aes(y = after_stat(density)), alpha = 0.4, position = "identity", bins = 10000) +
  scale_fill_manual(values = c("1980-1984" = "#1F78B4", "2015-2019" = "#E31A1C")) +
  scale_color_manual(values = c("1980-1984" = "#1F78B4", "2015-2019" = "#E31A1C")) +
  # xlim at the 5th and 95th percentiles
  coord_cartesian(xlim = c(earnings_early_late_info$ebitda %>% quantile(0.1),
                           earnings_early_late_info$ebitda %>% quantile(0.9))) +
  labs(
    title = "Earnings Distribution: Early vs Late Period (Information Sector - NAICS 51)",
    subtitle = "1980-1984 vs 2015-2019",
    x = "EBITDA",
    y = "Density",
    fill = "Period",
    color = "Period"
  ) +
  theme_common

ggsave("figures/earnings_distribution_early_vs_late_info_sector.pdf", width = 16, height = 9)

# OLD: Median markup -----

# Calculate median markup by year
markup_by_year <- analysis_data %>%
  group_by(date) %>%
  reframe(
    median_mu = median(mu, na.rm = TRUE),
    .groups = "drop"
  )

# Plot markup over time
ggplot(markup_by_year, aes(x = date, y = median_mu)) +
  geom_line(linewidth = 2, color = "black") +
  geom_point(size = 3, color = "black") +
  scale_x_continuous(breaks = seq(1975, 2020, 5)) +
  labs(
    title = "",
    x = "Year",
    y = "Median Markup"
  ) +
  theme_common

ggsave("figures/markup_by_year.pdf", width = 10, height = 10)