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
# filter to non-biotech firms
analysis_data <- analysis_data %>% filter(biotech_flag == 0)

# print number of firms and number of observations
num_firms <- n_distinct(analysis_data$gvkey)
num_observations <- nrow(analysis_data)
cat("Number of firms:", num_firms, "\n")
cat("Number of observations:", num_observations, "\n")

# Common theme and palette for all plots
theme_common <- theme_minimal(base_size = 18) +
  theme(
    text = element_text(family = "serif", size = 18),
    legend.position = "bottom"
  )

palette_2 <- viridis::inferno(2, begin = 0.0, end = 0.9)
palette_3 <- viridis::inferno(3, begin = 0.0, end = 0.9)
palette_4 <- viridis::inferno(4, begin = 0.0, end = 0.9)

# Restrict to negative-earnings firms for all neg_spell analyses
neg_data <- analysis_data %>%
  filter(neg_ebitda == 1) %>%
  mutate(naics_2digit = as.numeric(str_sub(naics_3digit, 1, 2)))

# -----------------------------------------------------------------------
# 1. Aggregate trend: mean neg_spell among neg-earnings firms
# -----------------------------------------------------------------------

spell_byyear <- neg_data %>%
  group_by(date) %>%
  reframe(mean_spell = mean(neg_spell, na.rm = TRUE))

ggplot(spell_byyear, aes(x = date, y = mean_spell)) +
  geom_line(linewidth = 2, color = "black") +
  geom_point(size = 3, color = "black") +
  labs(
    x = "Year",
    y = "Average Negative Spell Length (Years)",
    title = "Rise in Persistence of Negative Earnings"
  ) +
  theme_common

ggsave("figures/empirical/alt_neg_spell_aggregate.pdf", width = 8, height = 6)

# -----------------------------------------------------------------------
# 2. Within/Between Sector Decomposition of mean neg_spell
#
# Overall mean  = weighted sum of sector means
#   mu_t        = sum_s  w_{s,t} * mu_{s,t}
#
# Decompose change relative to a base year (1980):
#   Delta mu_t  = within_t + between_t
#
#   within_t    = sum_s  w_{s,0} * (mu_{s,t} - mu_{s,0})   [spell change, fixed weights]
#   between_t   = sum_s  (w_{s,t} - w_{s,0}) * mu_{s,t}   [reallocation of firms across sectors]
#
# Weights w_{s,t} = share of neg-earning firms in sector s in year t.
# -----------------------------------------------------------------------

sector_spell <- neg_data %>%
  group_by(naics_2digit, date) %>%
  reframe(
    sector_mean_spell = mean(neg_spell, na.rm = TRUE),
    n_firms_sector    = n(),
    .groups           = "drop"
  )

total_neg_byyear <- neg_data %>%
  group_by(date) %>%
  reframe(n_total = n())

sector_spell <- sector_spell %>%
  left_join(total_neg_byyear, by = "date") %>%
  mutate(w = n_firms_sector / n_total)

# Base-year (1980) sector means and weights
base_sector <- sector_spell %>%
  filter(date == 1980) %>%
  select(naics_2digit, mu_s0 = sector_mean_spell, w_s0 = w)

# Overall mean from data (for verification)
overall_mean <- sector_spell %>%
  group_by(date) %>%
  reframe(mean_spell_agg = sum(w * sector_mean_spell, na.rm = TRUE))

# Decomposition — only for sectors present in base year
decomp <- sector_spell %>%
  inner_join(base_sector, by = "naics_2digit") %>%
  group_by(date) %>%
  reframe(
    within_component  = sum(w_s0 * (sector_mean_spell - mu_s0), na.rm = TRUE),
    between_component = sum((w - w_s0) * sector_mean_spell,    na.rm = TRUE)
  ) %>%
  mutate(total_decomp = within_component + between_component)

# Pivot for plotting
decomp_long <- decomp %>%
  select(date, within_component, between_component) %>%
  pivot_longer(cols = c(within_component, between_component),
               names_to  = "component",
               values_to = "change_in_spell") %>%
  mutate(component = factor(component,
                             levels = c("within_component", "between_component"),
                             labels = c("Within Sector", "Between Sector")))

ggplot(decomp_long, aes(x = date, y = change_in_spell, color = component)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  scale_color_manual(values = setNames(palette_2, c("Within Sector", "Between Sector"))) +
  labs(
    x     = "Year",
    y     = "Change in Avg. Spell (Years, rel. to 1980)",
    color = "",
    title = "Within vs. Between Sector Decomposition of Spell Rise"
  ) +
  theme_common

ggsave("figures/empirical/alt_neg_spell_within_between.pdf", width = 8, height = 6)

# Also plot the total decomposed change vs the raw aggregate change side by side
decomp_check <- decomp %>%
  left_join(overall_mean, by = "date") %>%
  left_join(spell_byyear, by = "date") %>%
  mutate(
    base_mean     = mean_spell[date == 1980],
    raw_change    = mean_spell - mean_spell[date == 1980]
  )

base_mean_1980 <- spell_byyear$mean_spell[spell_byyear$date == 1980]

decomp_check <- decomp_check %>%
  mutate(raw_change = mean_spell - base_mean_1980)

decomp_verify_long <- decomp_check %>%
  select(date, raw_change, total_decomp) %>%
  pivot_longer(cols = c(raw_change, total_decomp),
               names_to  = "series",
               values_to = "value") %>%
  mutate(series = recode(series,
                         "raw_change"  = "Raw Change",
                         "total_decomp" = "Decomposed Total"))

ggplot(decomp_verify_long, aes(x = date, y = value, color = series, linetype = series)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  scale_color_manual(values = setNames(palette_2, c("Decomposed Total", "Raw Change"))) +
  labs(
    x        = "Year",
    y        = "Change from 1980 (Years)",
    color    = "",
    linetype = "",
    title    = "Decomposition Check: Raw vs. Decomposed Total"
  ) +
  theme_common

ggsave("figures/empirical/alt_neg_spell_decomp_check.pdf", width = 8, height = 6)

# -----------------------------------------------------------------------
# 3. New-listings check: does the rise hold for established firms?
#
# Firm age = date - ipo_year.
# Groups: <5 yrs since IPO ("new lists"), >=5 yrs ("established").
# Only firms with a non-missing ipo_year are included.
# -----------------------------------------------------------------------

neg_ipo <- neg_data %>%
  filter(!is.na(ipo_year)) %>%
  mutate(
    firm_age   = date - ipo_year,
    age_group  = case_when(
      firm_age <  5 ~ "< 5 Years Post-IPO",
      firm_age >= 5 ~ ">= 5 Years Post-IPO"
    )
  ) %>%
  filter(firm_age >= 0)  # drop pre-IPO rows (data artefacts)

spell_by_agegroup_year <- neg_ipo %>%
  group_by(age_group, date) %>%
  reframe(
    mean_spell = mean(neg_spell, na.rm = TRUE),
    n_firms    = n()
  ) %>%
  filter(n_firms >= 10)

ggplot(spell_by_agegroup_year, aes(x = date, y = mean_spell, color = age_group)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  scale_color_manual(values = setNames(palette_2, c("< 5 Years Post-IPO", ">= 5 Years Post-IPO"))) +
  labs(
    x     = "Year",
    y     = "Average Negative Spell Length (Years)",
    color = "",
    title = "Persistence of Negative Earnings by Firm Age Since IPO"
  ) +
  theme_common

ggsave("figures/empirical/alt_neg_spell_by_ipo_age.pdf", width = 8, height = 6)

# -----------------------------------------------------------------------
# 4. Combined two-panel: within/between + new-lists check
# -----------------------------------------------------------------------

p_decomp <- ggplot(decomp_long, aes(x = date, y = change_in_spell, color = component)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  scale_color_manual(values = setNames(palette_2, c("Within Sector", "Between Sector"))) +
  labs(
    x     = "Year",
    y     = "Change in Avg. Spell (Years, rel. to 1980)",
    color = "",
    title = "Within vs. Between Sector"
  ) +
  theme_common

p_ipo <- ggplot(spell_by_agegroup_year, aes(x = date, y = mean_spell, color = age_group)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  scale_color_manual(values = setNames(palette_2, c("< 5 Years Post-IPO", ">= 5 Years Post-IPO"))) +
  labs(
    x     = "Year",
    y     = "Average Negative Spell Length (Years)",
    color = "",
    title = "By Firm Age Since IPO"
  ) +
  theme_common

ggarrange(p_decomp, p_ipo, ncol = 2, nrow = 1)
ggsave("figures/empirical/alt_neg_spell_decomp_and_ipo.pdf", width = 16, height = 9)

# -----------------------------------------------------------------------
# 5. Within/Between Age Decomposition of mean neg_spell
#
# Same structure as the sector decomposition; grouping variable is
# factor(age) so each integer age value is its own group.
#
#   within_t    = sum_a  w_{a,0} * (mu_{a,t} - mu_{a,0})
#   between_t   = sum_a  (w_{a,t} - w_{a,0}) * mu_{a,t}
#
# Weights w_{a,t} = share of neg-earning firms with age a in year t.
# -----------------------------------------------------------------------

age_spell <- neg_data %>%
  group_by(age, date) %>%
  reframe(
    age_mean_spell = mean(neg_spell, na.rm = TRUE),
    n_firms_age    = n()
  )

total_neg_byyear_age <- neg_data %>%
  group_by(date) %>%
  reframe(n_total_age = n())

age_spell <- age_spell %>%
  left_join(total_neg_byyear_age, by = "date") %>%
  mutate(w_age = n_firms_age / n_total_age)

base_age <- age_spell %>%
  filter(date == 1980) %>%
  select(age, mu_a0 = age_mean_spell, w_a0 = w_age)

decomp_age <- age_spell %>%
  inner_join(base_age, by = "age") %>%
  group_by(date) %>%
  reframe(
    within_component  = sum(w_a0 * (age_mean_spell - mu_a0), na.rm = TRUE),
    between_component = sum((w_age - w_a0) * age_mean_spell, na.rm = TRUE)
  ) %>%
  mutate(total_decomp = within_component + between_component)

decomp_age_long <- decomp_age %>%
  select(date, within_component, between_component) %>%
  pivot_longer(cols = c(within_component, between_component),
               names_to  = "component",
               values_to = "change_in_spell") %>%
  mutate(component = factor(component,
                             levels = c("within_component", "between_component"),
                             labels = c("Within Age", "Between Age")))

ggplot(decomp_age_long, aes(x = date, y = change_in_spell, color = component)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  scale_color_manual(values = setNames(palette_2, c("Within Age", "Between Age"))) +
  labs(
    x     = "Year",
    y     = "Change in Avg. Spell (Years, rel. to 1980)",
    color = "",
    title = "Within vs. Between Age Decomposition of Spell Rise"
  ) +
  theme_common

ggsave("figures/empirical/alt_neg_spell_within_between_age.pdf", width = 8, height = 6)

# -----------------------------------------------------------------------
# 6. Combined two-panel: within/between sector + within/between age
# -----------------------------------------------------------------------

p_decomp_age <- ggplot(decomp_age_long, aes(x = date, y = change_in_spell, color = component)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  scale_color_manual(values = setNames(palette_2, c("Within Age", "Between Age"))) +
  labs(
    x     = "Year",
    y     = "Change in Avg. Spell (Years, rel. to 1980)",
    color = "",
    title = "Within vs. Between Age"
  ) +
  theme_common

ggarrange(p_decomp, p_decomp_age, ncol = 2, nrow = 1)
ggsave("figures/empirical/alt_neg_spell_decomp_sector_and_age.pdf", width = 16, height = 9)
