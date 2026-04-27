setwd("/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings")
# load libraries
library(tidyr)
library(dplyr)
library(stringr)
library(ggplot2)
library(ggpubr)
library(viridis)
library(readr)

# load data ----------------------------------------------------

load("data/clean/analysis_data.RData")

# print number of firms and number of observations
num_firms <- n_distinct(analysis_data$gvkey)
num_observations <- nrow(analysis_data)
cat("Number of firms:", num_firms, "\n")
cat("Number of observations:", num_observations, "\n")

# Common theme and palette for all plots
theme_common_2panel <- theme_minimal(base_size = 24) +
  theme(
    text = element_text(family = "serif", size = 24),
    legend.position = "bottom",
    legend.text = element_text(size = 24)
  )

theme_common <- theme_minimal(base_size = 18) +
  theme(
    text = element_text(family = "serif", size = 18),
    legend.position = "bottom"
  )

palette_2 <- viridis::inferno(2, begin = 0.0, end = 0.9)
palette_3 <- viridis::inferno(3, begin = 0.0, end = 0.9)
palette_4 <- viridis::inferno(4, begin = 0.0, end = 0.9)

# 1 Get Paper Stats ---------------

# neg_earnings_1980 and neg_earnings_2019
neg_earnings_1980 <- mean(analysis_data$neg_ebitda[analysis_data$date == 1980], na.rm = TRUE) * 100
neg_earnings_2019 <- mean(analysis_data$neg_ebitda[analysis_data$date == 2019], na.rm = TRUE) * 100

# neg_spell_1980 and neg_spell_2019
neg_spell_1980 <- mean(analysis_data$neg_spell[analysis_data$date == 1980 & analysis_data$neg_ebitda == 1], na.rm = TRUE)
neg_spell_2019 <- mean(analysis_data$neg_spell[analysis_data$date == 2019 & analysis_data$neg_ebitda == 1], na.rm = TRUE)

# avg_earnings_percchange and med_earnings_percchange (EBITDA)
ebitda_1980 <- analysis_data$ebitda[analysis_data$date == 1980]
ebitda_2019 <- analysis_data$ebitda[analysis_data$date == 2019]
avg_earnings_percchange <- (mean(ebitda_2019, na.rm = TRUE) - mean(ebitda_1980, na.rm = TRUE)) / abs(mean(ebitda_1980, na.rm = TRUE)) * 100
med_earnings_percchange <- (median(ebitda_2019, na.rm = TRUE) - median(ebitda_1980, na.rm = TRUE)) / abs(median(ebitda_1980, na.rm = TRUE)) * 100
avg_earnings_factorchange <- mean(ebitda_2019, na.rm = TRUE) / mean(ebitda_1980, na.rm = TRUE)
med_earnings_factorchange <- median(ebitda_2019, na.rm = TRUE) / median(ebitda_1980, na.rm = TRUE)

# num_obs_analysisdata and num_firms_analysisdata
num_obs_analysisdata <- nrow(analysis_data) 
num_firms_analysisdata <- n_distinct(analysis_data$gvkey)

# sd_sales_percchange and sd_ebitda_percchange
sd_sales_1980 <- sd(analysis_data$sale[analysis_data$date == 1980], na.rm = TRUE)
sd_sales_2019 <- sd(analysis_data$sale[analysis_data$date == 2019], na.rm = TRUE)
sd_ebitda_1980 <- sd(analysis_data$ebitda[analysis_data$date == 1980], na.rm = TRUE)
sd_ebitda_2019 <- sd(analysis_data$ebitda[analysis_data$date == 2019], na.rm = TRUE)
sd_sales_percchange <- (sd_sales_2019 - sd_sales_1980) / abs(sd_sales_1980) * 100
sd_ebitda_percchange <- (sd_ebitda_2019 - sd_ebitda_1980) / abs(sd_ebitda_1980) * 100

# cv_sales_percchange and cv_ebitda_percchange
cv_sales_1980 <- sd_sales_1980 / abs(mean(analysis_data$sale[analysis_data$date == 1980], na.rm = TRUE))
cv_sales_2019 <- sd_sales_2019 / abs(mean(analysis_data$sale[analysis_data$date == 2019], na.rm = TRUE))
cv_ebitda_1980 <- sd_ebitda_1980 / abs(mean(analysis_data$ebitda[analysis_data$date == 1980], na.rm = TRUE))
cv_ebitda_2019 <- sd_ebitda_2019 / abs(mean(analysis_data$ebitda[analysis_data$date == 2019], na.rm = TRUE))
cv_sales_percchange <- (cv_sales_2019 - cv_sales_1980) / abs(cv_sales_1980) * 100
cv_ebitda_percchange <- (cv_ebitda_2019 - cv_ebitda_1980) / abs(cv_ebitda_1980) * 100

# iqr_sales_percchange and iqr_ebitda_percchange
iqr_sales_1980 <- IQR(analysis_data$sale[analysis_data$date == 1980], na.rm = TRUE)
iqr_sales_2019 <- IQR(analysis_data$sale[analysis_data$date == 2019], na.rm = TRUE)
iqr_ebitda_1980 <- IQR(analysis_data$ebitda[analysis_data$date == 1980], na.rm = TRUE)
iqr_ebitda_2019 <- IQR(analysis_data$ebitda[analysis_data$date == 2019], na.rm = TRUE)
iqr_sales_percchange <- (iqr_sales_2019 - iqr_sales_1980) / abs(iqr_sales_1980) * 100
iqr_ebitda_percchange <- (iqr_ebitda_2019 - iqr_ebitda_1980) / abs(iqr_ebitda_1980) * 100

# cogs_change_all and capex_change_all and sga_change_all (% change in medians, all firms)
cogs_1980 <- analysis_data$cogs_sale[analysis_data$date == 1980]
cogs_2019 <- analysis_data$cogs_sale[analysis_data$date == 2019]
cogs_change_all <- (median(cogs_2019, na.rm = TRUE) - median(cogs_1980, na.rm = TRUE)) * 100

capex_1980 <- analysis_data$capx_sale[analysis_data$date == 1980]
capex_2019 <- analysis_data$capx_sale[analysis_data$date == 2019]
capex_change_all <- (median(capex_2019, na.rm = TRUE) - median(capex_1980, na.rm = TRUE)) * 100

sga_1980 <- analysis_data$sga_sale[analysis_data$date == 1980]
sga_2019 <- analysis_data$sga_sale[analysis_data$date == 2019]
sga_change_all <- (median(sga_2019, na.rm = TRUE) - median(sga_1980, na.rm = TRUE)) * 100

# cogs_change_neg and capex_change_neg and sga_change_neg and rd_change_neg (% change in medians, neg earnings firms only)
neg_1980 <- analysis_data$neg_ebitda == 1 & analysis_data$date == 1980
neg_2019 <- analysis_data$neg_ebitda == 1 & analysis_data$date == 2019
cogs_change_neg <- (median(analysis_data$cogs_sale[neg_2019], na.rm = TRUE) - median(analysis_data$cogs_sale[neg_1980], na.rm = TRUE)) * 100
capex_change_neg <- (median(analysis_data$capx_sale[neg_2019], na.rm = TRUE) - median(analysis_data$capx_sale[neg_1980], na.rm = TRUE)) * 100
sga_change_neg <- (median(analysis_data$sga_sale[neg_2019], na.rm = TRUE) - median(analysis_data$sga_sale[neg_1980], na.rm = TRUE)) * 100
rd_change_neg <- (median(analysis_data$rd_sale[neg_2019], na.rm = TRUE) - median(analysis_data$rd_sale[neg_1980], na.rm = TRUE)) * 100

# Write paper stats to CSV
paper_stats <- read.csv("paper/paper_stats.csv", stringsAsFactors = FALSE)
stats_computed <- list(
  neg_earnings_1980       = neg_earnings_1980,
  neg_earnings_2019       = neg_earnings_2019,
  neg_spell_1980          = neg_spell_1980,
  neg_spell_2019          = neg_spell_2019,
  avg_earnings_percchange = avg_earnings_percchange,
  med_earnings_percchange = med_earnings_percchange,
  num_obs_analysisdata    = num_obs_analysisdata,
  num_firms_analysisdata  = num_firms_analysisdata,
  sd_sales_percchange     = sd_sales_percchange,
  sd_ebitda_percchange    = sd_ebitda_percchange,
  cogs_change_all         = cogs_change_all,
  capex_change_all        = capex_change_all,
  sga_change_all          = sga_change_all,
  cogs_change_neg         = cogs_change_neg,
  capex_change_neg        = capex_change_neg,
  sga_change_neg          = sga_change_neg,
  rd_change_neg           = rd_change_neg,
  avg_earnings_factorchange = avg_earnings_factorchange,
  med_earnings_factorchange = med_earnings_factorchange,
  iqr_sales_percchange    = iqr_sales_percchange,
  iqr_ebitda_percchange   = iqr_ebitda_percchange
)
for (key in names(stats_computed)) {
  # if integer, round to 0 decimal places, otherwise round to 2 decimal places
  if (is.integer(stats_computed[[key]])) {
    paper_stats$value[paper_stats$key == key] <- round(stats_computed[[key]], 0)
  } else {
  paper_stats$value[paper_stats$key == key] <- round(stats_computed[[key]], 2)
  }
}
write.csv(paper_stats, "paper/paper_stats.csv", row.names = FALSE)

# 2 Plots for Rise of Neg. Earnings ---------------

neg_earnings_byyear <- analysis_data %>%
  group_by(date) %>%
  reframe(
    neg_ebitda_spell = mean(neg_spell[neg_ebitda == 1], na.rm = TRUE),
    neg_pi_spell = mean(neg_pi_spell[neg_pi == 1], na.rm = TRUE),
    neg_ni_spell = mean(neg_ni_spell[neg_ni == 1], na.rm = TRUE),
    neg_profits_spell = mean(neg_profits_spell[neg_profits == 1], na.rm = TRUE),
    neg_ebitda = mean(neg_ebitda) * 100,
    neg_pi = mean(neg_pi, na.rm = TRUE) * 100,
    neg_ni = mean(neg_ni, na.rm = TRUE) * 100,
    neg_profits = mean(neg_profits) * 100,
    n_firms = n()
  ) 

neg_ebitda_plot <- ggplot(neg_earnings_byyear, aes(x = date, y = neg_ebitda)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  labs(
    x = "Year",
    y = "",
    title = "Percent of Firms with EBITDA < 0"
  ) +
  theme_common_2panel

ggsave("figures/empirical/pct_negative_earnings_by_year.pdf", neg_ebitda_plot, width = 8, height = 6)

neg_ebitda_spell_plot <- ggplot(neg_earnings_byyear, aes(x = date, y = neg_ebitda_spell)) +
  geom_line(linewidth = 2, color = "black") +
  geom_point(size = 3, color = "black") +
  labs(
    x = "Year",
    y = "",
    title = "Average Negative Spell Length (Years)"
  ) +
  theme_common_2panel

ggarrange(neg_ebitda_plot, neg_ebitda_spell_plot, ncol = 2, nrow = 1)
ggsave("figures/empirical/neg_earnings_and_spell_over_time.pdf", width = 16, height = 9)
ggarrange(neg_ebitda_plot, neg_ebitda_spell_plot, ncol = 1, nrow = 2)
ggsave("figures/empirical/neg_earnings_and_spell_over_time_slides.pdf", width = 8, height = 12)
  
# 2a Robustness to IRS Data ---------------

# compare to IRS data
irs_corp_returns <- read.csv("data/clean/irs_corp_returns_combined.csv") %>%
  mutate(perc_neg_earnings = perc_neg_earnings * 100) %>%
  filter(year >= 1980 & year <= 2019)

ggplot() +
  geom_line(data = neg_earnings_byyear, aes(x = date, y = neg_ebitda, color = "EBITDA < 0 (Compustat)"), linewidth = 2) +
  geom_point(data = neg_earnings_byyear, aes(x = date, y = neg_ebitda, color = "EBITDA < 0 (Compustat)"), size = 3) +
  geom_line(data = irs_corp_returns, aes(x = year, y = perc_neg_earnings, color = "Net Income < 0 (IRS)"), linewidth = 2) +
  geom_point(data = irs_corp_returns, aes(x = year, y = perc_neg_earnings, color = "Net Income < 0 (IRS)"), size = 3) +
  scale_color_manual(
  values = setNames(palette_2, c("EBITDA < 0 (Compustat)", "Net Income < 0 (IRS)")),
  labels = c("EBITDA < 0 (Compustat)", expression("Net Income" <= "0 (IRS)"))
  ) +
  labs(
    x = "Year",
    y = "% with Negative Earnings",
    color = ""
  ) +
  theme_common

ggsave("figures/empirical/pct_negative_earnings_compustat_vs_irs.pdf", width = 8, height = 6)

# 2b Robustness to alternative earnings ---------------


neg_earnings_alt <- neg_earnings_byyear %>%
  select(date, neg_ebitda, neg_pi, neg_ni, neg_profits) %>%
  rename(
    "EBITDA" = neg_ebitda,
    "Pretax Income" = neg_pi,
    "Net Income" = neg_ni,
    "Profits" = neg_profits
  ) %>%
  pivot_longer(cols = -date, names_to = "measure", values_to = "pct_negative") 

neg_spells_alt <- neg_earnings_byyear %>%
  select(date, neg_ebitda_spell, neg_pi_spell, neg_ni_spell, neg_profits_spell) %>%
  rename(
    "EBITDA" = neg_ebitda_spell,
    "Pretax Income" = neg_pi_spell,
    "Net Income" = neg_ni_spell,
    "Profits" = neg_profits_spell
  ) %>%
  pivot_longer(cols = -date, names_to = "measure", values_to = "mean_neg_spell")

neg_earnings_plot <- ggplot(neg_earnings_alt, aes(x = date, y = pct_negative, color = measure)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  scale_color_manual(values = setNames(palette_4,
    c("EBITDA", "Net Income", "Pretax Income", "Profits"))) +
  labs(x = "Year", y = "", title = "Percent Firms with Negative Earnings", color = "") +
  theme_common_2panel

neg_spells_plot <- ggplot(neg_spells_alt, aes(x = date, y = mean_neg_spell, color = measure)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  scale_color_manual(values = setNames(palette_4,
    c("EBITDA", "Net Income", "Pretax Income", "Profits"))) +
  labs(x = "Year", y = "", title = "Average Negative Spell Length (Years)", color = "") +
  theme_common_2panel

ggarrange(neg_earnings_plot, neg_spells_plot, ncol = 2, nrow = 1, common.legend = TRUE, legend = "bottom")
ggsave("figures/empirical/negative_earnings_alt_measures.pdf", width = 16, height = 9)

ggarrange(neg_earnings_plot, neg_spells_plot, ncol = 1, nrow = 2, common.legend = TRUE, legend = "bottom")
ggsave("figures/empirical/negative_earnings_alt_measures_slides.pdf", width = 8, height = 12)


# 2c (mean/median). Average and median EBITDA/Sales over time ----------------

ebitda_stats_by_year <- analysis_data %>%
  group_by(date) %>%
  reframe(
    Mean   = log(mean(ebitda, na.rm = TRUE)),
    Median = log(median(ebitda, na.rm = TRUE))
  ) %>%
  pivot_longer(cols = c(Mean, Median), names_to = "stat", values_to = "log_ebitda")

ggplot(ebitda_stats_by_year, aes(x = date, y = log_ebitda, color = stat)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  scale_color_manual(values = setNames(palette_2, c("Mean", "Median"))) +
  labs(x = "Year", y = "EBITDA (logged)", color = "") +
  theme_common

ggsave("figures/empirical/mean_median_log_ebitda_by_year.pdf", width = 8, height = 6)

# 3a. Plot log change in SD of sales over time ------------------------------------

sd_sales_by_year <- analysis_data %>%
  group_by(date) %>%
  reframe(
    log_sd = log(sd(sale))
  )

# Get 1980 values for normalization
base_1980_sales_sd <- sd_sales_by_year %>% filter(date == 1980)
# Normalize to 1980 (so line starts at 0)
sd_sales_normalized <- sd_sales_by_year %>%
  mutate(
    log_change_sd = log_sd - base_1980_sales_sd$log_sd
  )

ggplot(sd_sales_by_year, aes(x = date, y = log_sd)) +
  geom_line(linewidth = 2, color = "black") +
  geom_point(size = 3, color = "black") +
  scale_x_continuous(breaks = seq(1980, 2020, 5)) +
  labs(
    title = "",
    subtitle = "",
    x = "Year",
    y = "Std. Deviation Sales (Logged)"
  ) +
  theme_common

ggsave("figures/empirical/sd_sales_by_year.pdf", width = 8, height = 6)

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
    values = setNames(palette_4,
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

ggsave("figures/empirical/top_quantile_sales_over_time.pdf", width = 8, height = 6)

top_quantiles_sales <- ggplot(top_decile_sales_long, aes(x = date, y = log_change, color = percentile, group = percentile)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_x_continuous(breaks = seq(1980, 2020, 5)) +
  scale_color_manual(
    values = setNames(palette_4,
                      c("log_change_p75", "log_change_p90", "log_change_p95", "log_change_p99")),
    labels = c("log_change_p75" = "|P75 - Median|", "log_change_p90" = "|P90 - Median|", "log_change_p95" = "|P95 - Median|", "log_change_p99" = "|P99 - Median|"),
    breaks = c("log_change_p75", "log_change_p90", "log_change_p95", "log_change_p99")
  ) +
  labs(
    title = "Top Quantiles of Sales",
    subtitle = "",
    x = "Year",
    y = "Log Change from 1980",
    color = ""
  ) +
  theme_common_2panel

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
    values = setNames(palette_4,
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

ggsave("figures/empirical/bottom_quantile_sales_over_time.pdf", width = 8, height = 6)

bottom_quantiles_sales <- ggplot(bottom_quantile_sales_long, aes(x = date, y = log_change, color = percentile, group = percentile)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_x_continuous(breaks = seq(1980, 2020, 5)) +
  scale_color_manual(
    values = setNames(palette_4,
                      c("log_change_p25", "log_change_p10", "log_change_p5", "log_change_p1")),
    labels = c("log_change_p25" = "|Median - P25|", "log_change_p10" = "|Median - P10|", "log_change_p5" = "|Median - P5|", "log_change_p1" = "|Median - P1|"),
    breaks = c("log_change_p25", "log_change_p10", "log_change_p5", "log_change_p1")
  ) +
  labs(
    title = "Bottom Quantiles of Sales",
    subtitle = "",
    x = "Year",
    y = "Log Change from 1980",
    color = ""
  ) +
  theme_common_2panel

# 4a. Plot log change in SD of earnings over time -------------------------

sd_by_year <- analysis_data %>%
  group_by(date) %>%
  reframe(
    log_sd = log(sd(ebitda))
  )

# Get 1980 values for normalization
base_1980_sd <- sd_by_year %>% filter(date == 1980)

# Normalize to 1980 (so line starts at 0)
sd_normalized <- sd_by_year %>%
  mutate(
    log_change_sd = log_sd - base_1980_sd$log_sd
  )

ggplot(sd_by_year, aes(x = date, y = log_sd)) +
  geom_line(linewidth = 2, color = "black") +
  geom_point(size = 3, color = "black") +
  # geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_x_continuous(breaks = seq(1980, 2020, 5)) +
  labs(
    title = "",
    subtitle = "",
    x = "Year",
    y = "Std. Deviation EBITDA (Logged)"
  ) +
  theme_common

ggsave("figures/empirical/sd_earnings_by_year.pdf", width = 8, height = 6)

# 4b. Top Percentiles of earnings over time ------------------------

top_decile_measures <- analysis_data %>%
  group_by(date) %>%
  reframe(
    median = median(ebitda),
    p75 = quantile(ebitda, 0.75),
    p90 = quantile(ebitda, 0.90),
    p95 = quantile(ebitda, 0.95),
    p99 = quantile(ebitda, 0.99)
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
    values = setNames(palette_4,
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

ggsave("figures/empirical/top_quantile_earnings_over_time.pdf", width = 8, height = 6)

top_quantiles_earnings <- ggplot(top_decile_long, aes(x = date, y = log_change, color = percentile, group = percentile)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_x_continuous(breaks = seq(1980, 2020, 5)) +
  scale_color_manual(
    values = setNames(palette_4,
                      c("log_change_p75", "log_change_p90", "log_change_p95", "log_change_p99")),
    labels = c("log_change_p75" = "|P75 - Median|", "log_change_p90" = "|P90 - Median|", "log_change_p95" = "|P95 - Median|", "log_change_p99" = "|P99 - Median|"),
    breaks = c("log_change_p75", "log_change_p90", "log_change_p95", "log_change_p99")
  ) +
  labs(
    title = "Top Quantiles of EBITDA",
    subtitle = "",
    x = "Year",
    y = "Log Change from 1980",
    color = ""
  ) +
  theme_common_2panel

# 4c. Bottom percentiles of earnings over time -------------------

bottom_quantile_measures <- analysis_data %>%
  group_by(date) %>%
  reframe(
    median = median(ebitda),
    p25 = quantile(ebitda, 0.25),
    p10 = quantile(ebitda, 0.10),
    p5 = quantile(ebitda, 0.05),
    p1 = quantile(ebitda, 0.01)
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
    values = setNames(palette_4,
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
  theme_common_2panel

ggsave("figures/empirical/bottom_quantile_earnings_over_time.pdf", width = 8, height = 6)

bottom_quantiles_earnings <- ggplot(bottom_quantile_long, aes(x = date, y = log_change, color = percentile, group = percentile)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_x_continuous(breaks = seq(1980, 2020, 5)) +
  scale_color_manual(
    values = setNames(palette_4,
                      c("log_change_p25", "log_change_p10", "log_change_p5", "log_change_p1")),
    labels = c("log_change_p25" = "|Median - P25|", "log_change_p10" = "|Median - P10|", "log_change_p5" = "|Median - P5|", "log_change_p1" = "|Median - P1|"),
    breaks = c("log_change_p25", "log_change_p10", "log_change_p5", "log_change_p1")
  ) +
  labs(
    title = "Bottom Quantiles of EBITDA",
    subtitle = "",
    x = "Year",
    y = "Log Change from 1980",
    color = ""
  ) +
  theme_common_2panel

# 4d: Combined plot of quantiles of earnings and sales ----------------

ggarrange(top_quantiles_sales, top_quantiles_earnings, ncol = 2, nrow = 1, common.legend = TRUE, legend = "bottom")
ggsave("figures/empirical/top_quantiles_earnings_sales.pdf", width = 16, height = 9)
ggarrange(bottom_quantiles_sales, bottom_quantiles_earnings, ncol = 2, nrow = 1, common.legend = TRUE, legend = "bottom")
ggsave("figures/empirical/bottom_quantiles_earnings_sales.pdf", width = 16, height = 9)

# 4e: Joint plot of log sd earnings + log sd sales over time ----------------
# add dashed lines for log iqr of earnings and sales
iqr_sd_joint <- analysis_data %>%
  group_by(date) %>%
  reframe(
    log_iqr_ebitda = log(quantile(ebitda, 0.75) - quantile(ebitda, 0.25)),
    log_iqr_sales = log(quantile(sale, 0.75) - quantile(sale, 0.25)),
    log_sd_ebitda = log(sd(ebitda)),
    log_sd_sales = log(sd(sale))
  ) %>%
  pivot_longer(cols = c(log_iqr_ebitda, log_iqr_sales, log_sd_ebitda, log_sd_sales), names_to = "series", values_to = "value")

# sd_joint <- bind_rows(
#   sd_by_year       %>% mutate(series = "EBITDA"),
#   sd_sales_by_year %>% mutate(series = "Sales")
# )

# ggplot(sd_joint, aes(x = date, y = log_sd, color = series, group = series)) +
#   geom_line(linewidth = 2) +
#   geom_point(size = 3) +
#   scale_x_continuous(breaks = seq(1980, 2020, 5)) +
#   scale_color_manual(
#     values = setNames(palette_2, c("EBITDA", "Sales")),
#     labels = c("EBITDA" = "EBITDA", "Sales" = "Sales")
#   ) +
#   labs(
#     title = "",
#     subtitle = "",
#     x = "Year",
#     y = "Std. Deviation (Logged)",
#     color = ""
#   ) +
#   theme_common

# ggsave("figures/empirical/sd_joint_by_year.pdf", width = 8, height = 6)

iqr_sd_joint %>%
  mutate(
    variable = ifelse(grepl("ebitda", series), "EBITDA", "Sales"),
    measure  = ifelse(grepl("iqr",    series), "IQR",    "SD"),
    label    = factor(paste(variable, measure),
                      levels = c("EBITDA IQR", "EBITDA SD", "Sales IQR", "Sales SD"))
  ) %>%
  ggplot(aes(x = date, y = value, color = label, linetype = label, group = label)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  scale_x_continuous(breaks = seq(1980, 2020, 5)) +
  scale_color_manual(
    name = "",
    values = c(
      "EBITDA IQR" = palette_2[1], "EBITDA SD" = palette_2[1],
      "Sales IQR"  = palette_2[2], "Sales SD"  = palette_2[2]
    )
  ) +
  scale_linetype_manual(
    name = "",
    values = c(
      "EBITDA IQR" = "dashed", "EBITDA SD" = "solid",
      "Sales IQR"  = "dashed", "Sales SD"  = "solid"
    )
  ) +
  guides(
    color    = guide_legend(keywidth = unit(1.75, "cm")),
    linetype = guide_legend(keywidth = unit(1.75, "cm"))
  ) +
  labs(
    title = "",
    subtitle = "",
    x = "Year",
    y = "Dispersion (Logged)"
  ) +
  theme_common

ggsave("figures/empirical/iqr_joint_by_year.pdf", width = 8, height = 6)


# 5: Bar plot of negative earnings by 2-digit NAICS sector ----------------------

sector_year_data_2digit <- analysis_data %>%
  mutate(naics_2digit = as.numeric(str_sub(naics_3digit, 1, 2))) %>%
  group_by(naics_2digit, date) %>%
  reframe(
    pct_negative = mean(ebitda < 0),
    n_firms = n(),
    .groups = "drop"
  ) %>%
  filter(n_firms >= 5)

sector_averages_2digit <- sector_year_data_2digit %>%
  group_by(naics_2digit) %>%
  summarise(
    avg_pct_negative   = mean(pct_negative, na.rm = TRUE),
    pct_negative_late  = mean(pct_negative[date >= 2014], na.rm = TRUE),
    pct_negative_early = mean(pct_negative[date <= 1980], na.rm = TRUE),
    n_years            = n(),
    .groups = "drop"
  ) %>%
  mutate(change_pct_negative = pct_negative_late - pct_negative_early) %>%
  arrange(desc(avg_pct_negative))

sector_2digit_long <- sector_averages_2digit %>%
  select(naics_2digit, pct_negative_late, change_pct_negative) %>%
  pivot_longer(cols = c(pct_negative_late, change_pct_negative),
               names_to = "metric",
               values_to = "value") %>%
  mutate(value = 100*value,
         metric = factor(metric, levels = c("change_pct_negative", "pct_negative_late"))) %>%
  filter(!is.na(value))

ggplot(sector_2digit_long, aes(x = as.factor(naics_2digit),
                                # x = reorder(as.factor(naics_2digit), -value * (metric == "pct_negative_late")),
                                y = value, fill = metric)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  scale_fill_manual(
    breaks = c("pct_negative_late", "change_pct_negative"),
    values = setNames(palette_2, c("pct_negative_late", "change_pct_negative")),
    labels = c("pct_negative_late" = "% Neg. Earnings \n (2014-2019)",
               "change_pct_negative" = "Change in % Neg. Earnings \n (1980-1984 to 2014-2019)")
  ) +
  labs(
    title = "",
    subtitle = "",
    x = "Sector",
    y = "",
    fill = ""
  ) +
  theme_common

ggsave("figures/empirical/neg_earnings_by_sector_2digit.pdf", width = 8, height = 6)

# Earnings concentration among top 10%, 5%, 1%, and 0.1% over time --------

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
    values = setNames(palette_4, c("top10_share", "top5_share", "top1_share", "top01_share")),
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

ggsave("figures/empirical/earnings_concentration_over_time.pdf", width = 16, height = 9)

# 7. Overlayed histograms: early (1980-1984) vs late (2015-2019) earnings distribution
earnings_early_late <- analysis_data %>%
  filter((date >= 1980 & date <= 1984) | (date >= 2015 & date <= 2019)) %>%
  mutate(period = ifelse(date <= 1984, "1980-1984", "2015-2019"))

ggplot(earnings_early_late, aes(x = ebitda, fill = period, color = period)) +
  geom_histogram(aes(y = after_stat(density)), alpha = 0.4, position = "identity", bins = 10000) +
  scale_fill_manual(values = setNames(palette_2, c("1980-1984", "2015-2019"))) +
  scale_color_manual(values = setNames(palette_2, c("1980-1984", "2015-2019"))) +
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

ggsave("figures/empirical/earnings_distribution_early_vs_late.pdf", width = 16, height = 9)

# Same for Information Sector only
earnings_early_late_info <- analysis_data %>%
  mutate(naics_2digit = as.numeric(str_sub(naics_3digit, 1, 2))) %>%
  filter(naics_2digit == 51) %>%
  filter((date >= 1980 & date <= 1984) | (date >= 2015 & date <= 2019)) %>%
  mutate(period = ifelse(date <= 1984, "1980-1984", "2015-2019"))

ggplot(earnings_early_late_info, aes(x = ebitda, fill = period, color = period)) +
  geom_histogram(aes(y = after_stat(density)), alpha = 0.4, position = "identity", bins = 10000) +
  scale_fill_manual(values = setNames(palette_2, c("1980-1984", "2015-2019"))) +
  scale_color_manual(values = setNames(palette_2, c("1980-1984", "2015-2019"))) +
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

ggsave("figures/empirical/earnings_distribution_early_vs_late_info_sector.pdf", width = 10, height = 9)

# Median markup -----

markup_by_year <- analysis_data %>%
  group_by(date) %>%
  reframe(
    median_mu = median(mu, na.rm = TRUE),
    .groups = "drop"
  )

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

ggsave("figures/empirical/markup_by_year.pdf", width = 8, height = 6)

# APPENDIX: Scale invariant measure of distribution spread for earnings

# Sector-normalized IQR: divide each firm by its sector-year median, then take IQR across all firms
sector_norm_spread <- analysis_data %>%
  group_by(date) %>%
  reframe(
    iqr_norm_sale = (quantile(sale, 0.75) - quantile(sale, 0.25)) / median(sale),
    iqr_norm_ebitda = (quantile(ebitda, 0.75) - quantile(ebitda, 0.25)) / median(ebitda),
    # check movements on both sides
    p50_p25_norm_ebitda = (quantile(ebitda, 0.50) - quantile(ebitda, 0.25)) / median(ebitda),
    p75_p50_norm_ebitda = (quantile(ebitda, 0.75) - quantile(ebitda, 0.50)) / median(ebitda),
    p50_p25_norm_sale = (quantile(sale, 0.50) - quantile(sale, 0.25)) / median(sale),
    p75_p50_norm_sale = (quantile(sale, 0.75) - quantile(sale, 0.50)) / median(sale)
  ) %>%
  mutate(
    log_iqr_norm_sales = log(iqr_norm_sale),
    log_iqr_norm_ebitda = log(iqr_norm_ebitda),
    log_p50_p25_norm_ebitda = log(p50_p25_norm_ebitda),
    log_p75_p50_norm_ebitda = log(p75_p50_norm_ebitda),
    log_p50_p25_norm_sale = log(p50_p25_norm_sale),
    log_p75_p50_norm_sale = log(p75_p50_norm_sale)
  )

base_1980_spread <- sector_norm_spread %>% filter(date == 1980)

scale_inv_iqr <- sector_norm_spread %>%
  mutate(
    log_change_iqr_norm_sales = log_iqr_norm_sales - base_1980_spread$log_iqr_norm_sales,
    log_change_iqr_norm_ebitda = log_iqr_norm_ebitda - base_1980_spread$log_iqr_norm_ebitda, 
    log_change_p50_p25_norm_ebitda = log_p50_p25_norm_ebitda - base_1980_spread$log_p50_p25_norm_ebitda,
    log_change_p75_p50_norm_ebitda = log_p75_p50_norm_ebitda - base_1980_spread$log_p75_p50_norm_ebitda,
    log_change_p50_p25_norm_sale = log_p50_p25_norm_sale - base_1980_spread$log_p50_p25_norm_sale,
    log_change_p75_p50_norm_sale = log_p75_p50_norm_sale - base_1980_spread$log_p75_p50_norm_sale
  ) %>%
  select(date, log_change_iqr_norm_sales, log_change_iqr_norm_ebitda, log_change_p50_p25_norm_ebitda, log_change_p75_p50_norm_ebitda, log_change_p50_p25_norm_sale, log_change_p75_p50_norm_sale) %>%
  pivot_longer(cols = starts_with("log_change"),
               names_to = "variable",
               values_to = "log_change")

plot_main_measures <- ggplot(scale_inv_iqr %>% filter(variable == "log_change_iqr_norm_ebitda" | variable == "log_change_iqr_norm_sales"), aes(x = date, y = log_change, color = variable, group = variable)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_x_continuous(breaks = seq(1980, 2020, 5)) +
  scale_color_manual(
    values = setNames(palette_2, c( "log_change_iqr_norm_ebitda", "log_change_iqr_norm_sales")),
    labels = c("log_change_iqr_norm_ebitda" = "EBITDA", "log_change_iqr_norm_sales" = "Sales")
  ) +
  labs(
    x = "Year",
    y = "IQR/Med. (Log Change from 1980)",
    color = "",
    title = ""
  ) +
  theme_common

ggsave("figures/empirical/scale_inv_main_measures.pdf", width = 8, height = 6)