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
theme_common <- theme_minimal(base_size = 18) +
  theme(
    text = element_text(family = "serif", size = 18),
    plot.title = element_text(face = "bold"),
    legend.position = "bottom"
  )

palette_2 <- viridis::inferno(2, begin = 0.0, end = 0.9)
palette_3 <- viridis::inferno(3, begin = 0.0, end = 0.9)
palette_4 <- viridis::inferno(4, begin = 0.0, end = 0.9)

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

ggsave("figures/empirical/pct_negative_earnings_by_year.pdf", width = 8, height = 6)

# compare to IRS data
irs_corp_returns_post94 <- read.csv("data/clean/irs_corp_returns.csv") %>%
  mutate(perc_neg_earnings = perc_neg_earnings * 100) %>%
  filter(year >= 1994 & year <= 2019)

irs_corps_returns_pre94 <- read.csv("data/clean/irs_corp_returns_pre94.csv") %>%
  mutate(perc_neg_earnings = perc_neg_earnings * 100) %>%
  filter(year >= 1980 & year < 1994)

irs_corps_returns <- bind_rows(irs_corps_returns_pre94, irs_corp_returns_post94)

ggplot() +
  geom_line(data = neg_earnings_by_year, aes(x = date, y = pct_negative, color = "EBITDA < 0 (Compustat)"), linewidth = 2) +
  geom_point(data = neg_earnings_by_year, aes(x = date, y = pct_negative, color = "EBITDA < 0 (Compustat)"), size = 3) +
  geom_line(data = irs_corps_returns, aes(x = year, y = perc_neg_earnings, color = "Net Income < 0 (IRS)"), linewidth = 2) +
  geom_point(data = irs_corps_returns, aes(x = year, y = perc_neg_earnings, color = "Net Income < 0 (IRS)"), size = 3) +
  scale_color_manual(values = setNames(palette_2, c("EBITDA < 0 (Compustat)", "Net Income < 0 (IRS)"))) +
  labs(
    x = "Year",
    y = "% with Negative Earnings",
    color = ""
  ) +
  theme_common

ggsave("figures/empirical/pct_negative_earnings_compustat_vs_irs.pdf", width = 8, height = 6)

# Overlay EBITDA, Net Income, Pretax Income

neg_earnings_alt <- analysis_data %>%
  group_by(date) %>%
  reframe(
    `EBITDA < 0`        = mean(ebitda < 0, na.rm = TRUE) * 100,
    `Net Income < 0`    = mean(ni < 0, na.rm = TRUE) * 100,
    `Pretax Income < 0` = mean(pi < 0, na.rm = TRUE) * 100
  ) %>%
  pivot_longer(cols = -date, names_to = "measure", values_to = "pct_negative")

ggplot(neg_earnings_alt, aes(x = date, y = pct_negative, color = measure)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  scale_color_manual(values = setNames(palette_3,
    c("EBITDA < 0", "Net Income < 0", "Pretax Income < 0"))) +
  labs(x = "Year", y = "% Firms with Negative Earnings", color = "") +
  theme_common

ggsave("figures/empirical/pct_negative_earnings_alt_measures.pdf", width = 8, height = 6)

# 2b (mean/median). Average and median EBITDA/Sales over time ----------------

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

# 2c Plot average neg_spell among negative earning firms over time ----

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

ggsave("figures/empirical/neg_spell_over_time.pdf", width = 8, height = 6)

# 2 panel figure with percent negative and average neg spell over time
neg_earnings_panel <- ggplot(neg_earnings_by_year, aes(x = date, y = pct_negative)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  labs(
    title = "",
    x = "Year",
    y = "Percent of Firms with EBITDA < 0 (%)",
    # caption = paste0("N firms ranges from ", min(neg_earnings_by_year$n_firms),
                    #  " to ", max(neg_earnings_by_year$n_firms))
  ) +
  theme_common

neg_spell_panel <- ggplot(neg_spell_by_year, aes(x = date, y = mean_neg_spell)) +
  geom_line(linewidth = 2, color = "black") +
  geom_point(size = 3, color = "black") +
  labs(
    title = "",
    x = "Year",
    y = "Average Negative Earnings Spell Length (Years)",
    # caption = paste0("N negative earning firms ranges from ", min(neg_spell_by_year$n_firms),
                    #  " to ", max(neg_spell_by_year$n_firms))
  ) +
  theme_common

ggarrange(neg_earnings_panel, neg_spell_panel, ncol = 2, nrow = 1)
ggsave("figures/empirical/neg_earnings_and_spell_over_time.pdf", width = 16, height = 9)

# 2d Plots using neg_profits (expensing capital) ----
# profits = sale - cogs - sga - capx (built in 3_build_analysis_data.R)

neg_profits_by_year <- analysis_data %>%
  group_by(date) %>%
  reframe(
    pct_negative = mean(neg_profits, na.rm = TRUE) * 100,
    n_firms = n()
  )

neg_profits_spell_by_year <- analysis_data %>%
  filter(neg_profits == 1) %>%
  group_by(date) %>%
  reframe(
    mean_neg_spell = mean(neg_profits_spell, na.rm = TRUE),
    n_firms = n()
  )

neg_profits_panel <- ggplot(neg_profits_by_year, aes(x = date, y = pct_negative)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  labs(x = "Year", y = "Percent of Firms with Profits < 0 (%)") +
  theme_common

neg_profits_spell_panel <- ggplot(neg_profits_spell_by_year, aes(x = date, y = mean_neg_spell)) +
  geom_line(linewidth = 2, color = "black") +
  geom_point(size = 3, color = "black") +
  labs(x = "Year", y = "Average Negative Profits Spell Length (Years)") +
  theme_common

ggarrange(neg_profits_panel, neg_profits_spell_panel, ncol = 2, nrow = 1)
ggsave("figures/empirical/neg_profits_and_spell_over_time.pdf", width = 16, height = 9)

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

ggsave("figures/empirical/bottom_quantile_earnings_over_time.pdf", width = 8, height = 6)


# 4d: Joint plot of log sd earnings + log sd sales over time ----------------

sd_joint <- bind_rows(
  sd_by_year       %>% mutate(series = "EBITDA"),
  sd_sales_by_year %>% mutate(series = "Sales")
)

ggplot(sd_joint, aes(x = date, y = log_sd, color = series, group = series)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  scale_x_continuous(breaks = seq(1980, 2020, 5)) +
  scale_color_manual(
    values = setNames(palette_2, c("EBITDA", "Sales")),
    labels = c("EBITDA" = "EBITDA", "Sales" = "Sales")
  ) +
  labs(
    title = "",
    subtitle = "",
    x = "Year",
    y = "Std. Deviation (Logged)",
    color = ""
  ) +
  theme_common

ggsave("figures/empirical/sd_joint_by_year.pdf", width = 8, height = 6)


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

ggplot(sector_2digit_long, aes(x = reorder(as.factor(naics_2digit),
                                            -value * (metric == "pct_negative_late")),
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
    x = "NAICS 2-digit Sector",
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
