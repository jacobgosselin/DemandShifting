setwd("/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings")
# load libraries
library(tidyr)
library(dplyr)
library(stringr)
library(ggplot2)
library(ggpubr)
library(viridis)
library(readr)
# load stata data
library(haven)
library(ggpubr)
library(fixest)

# load data ----------------------------------------------------

load("data/clean/analysis_data.RData")
palette_3 <- viridis::inferno(3, begin = 0.0, end = 0.9)
theme_common <- theme_minimal(base_size = 12) +
  theme(
    text = element_text(family = "serif", size = 12),
    legend.position = "bottom"
  )

theme_slides <- theme_minimal(base_size = 32) +
  theme(
    text = element_text(family = "serif", size = 32),
    legend.position = "bottom"
  )

# looking at ma ----------------------------------------------------

# get ma total receipts
data_ma <- read_stata("data/raw/agg_raw_receipts_R5.dta")
data_ma <- as.data.frame(data_ma)
total_receipts <- data_ma %>% filter(thres_low == "Total")
total_receipts <- total_receipts %>% select(year, total_receipts = breceipts)

# deflate with GDP Deflator
load("data/raw/deflator.RData")
deflator <- deflator %>% rename(year = fyear)
total_receipts <- merge(total_receipts, deflator, by = "year")
total_receipts <- total_receipts %>%
  mutate(total_receipts = total_receipts / deflator * 100,
         total_receipts_rel_1980 = total_receipts / total_receipts[year == 1980]) %>%
  select(year, total_receipts, total_receipts_rel_1980)

# get total sales by top 500 firms and total sales by top 500 firms as share of total sales
sales_top500 <- analysis_data %>%
  group_by(date) %>%
  arrange(desc(sale)) %>%
  summarise(sales_top500 = sum(sale[1:500], na.rm = TRUE),
            sales_total_compustat = sum(sale, na.rm = TRUE)) %>%
mutate(sales_top500_share = sales_top500 / sales_total_compustat) %>%
rename(year = date)

sales_top500 <- merge(sales_top500, total_receipts, by = "year")
sales_top500 <- sales_top500 %>%
  mutate(sales_top500_share_total_receipts = sales_top500 / total_receipts,
         sales_total_compustat_share_total_receipts = sales_total_compustat / total_receipts)

# plot all three series
plot_data <- sales_top500 %>%
  select(year,
         `Top 500 Share (Total Receipts)`   = sales_top500_share_total_receipts,
         `Compustat Share (Total Receipts)` = sales_total_compustat_share_total_receipts,
         `Top 500 Share (Compustat)`        = sales_top500_share) %>%
  pivot_longer(-year, names_to = "series", values_to = "value") %>%
  mutate(series = factor(series, levels = c("Top 500 Share (Total Receipts)", "Compustat Share (Total Receipts)", "Top 500 Share (Compustat)")))

ggplot(plot_data, aes(x = year, y = value, color = series)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  scale_color_manual(values = palette_3) +
  labs(x = NULL, y = "Share", color = NULL) +
  theme_common

ggsave("figures/empirical/for_kunal.pdf", width = 8, height = 6)

# digging into neg_profits ----------------------------------------------------

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

neg_earnings_byage_bycohort <- analysis_data %>%
  mutate(
    age = date - first_date,
    cohort = case_when(
      # 1980-1989
      first_date >= 1980 & first_date < 1990 ~ "1980-1989",
      # 1990-1999
      first_date >= 1990 & first_date < 2000 ~ "1990-1999",
      # 2000-2009
      first_date >= 2000 & first_date < 2010 ~ "2000-2009"
      )
  ) %>%
  group_by(gvkey) %>% 
  mutate(
    max_age = max(age, na.rm = TRUE)
  ) %>%
  filter(!is.na(cohort) & age >= 0 & age <= 10) %>%
  group_by(cohort, age) %>%
  summarise(
    pct_neg_ebitda = mean(neg_ebitda, na.rm = TRUE) * 100,
    pct_neg_profits = mean(neg_profits, na.rm = TRUE) * 100,
    n_firms = n(),
    .groups = "drop"
  )

palette_3 <- viridis::inferno(3, begin = 0.0, end = 0.9)
cohort_colors <- setNames(palette_3, c("1980-1989", "1990-1999", "2000-2009"))

cohort_profits <- ggplot(neg_earnings_byage_bycohort, aes(x = age, y = pct_neg_profits, color = cohort)) +
  geom_line(linewidth = 1) +
  geom_point(aes(size = n_firms)) +
  # geom_smooth(method = "lm", se = FALSE, linetype = "dashed", linewidth = 0.8) +
  scale_x_continuous(breaks = 1:10) +
  scale_color_manual(values = cohort_colors) +
  scale_size_continuous(name = "", range = c(1, 5)) +
  labs(x = "Firm Age", y = "", title = "Percent of Firms with Negative Profits", color = "") +
  theme_common

cohort_ebitda <- ggplot(neg_earnings_byage_bycohort, aes(x = age, y = pct_neg_ebitda, color = cohort)) +
  geom_line(linewidth = 2) +
  geom_point(aes(size = n_firms)) +
  # geom_smooth(method = "lm", se = FALSE, linetype = "dashed", linewidth = 0.8) +
  scale_x_continuous(breaks = 1:10) +
  scale_color_manual(values = cohort_colors) +
  scale_size_continuous(name = "", range = c(1, 5)) +
  labs(x = "Firm Age", y = "", title = "Percent of Firms with Negative EBITDA", color = "") +
  theme_common

ggarrange(cohort_ebitda, cohort_profits, nrow = 1, common.legend = TRUE, legend = "bottom")
ggsave("figures/empirical/neg_earnings_by_ageANDcohort.pdf", width = 10, height = 5)

# plot the average negative profits spell length by 5-year cohort, 1962-2010
# (9 x 5-year windows + 1 x 4-year window: 2007-2010)
neg_spell_bycohort <- analysis_data %>%
  mutate(
    # cap at 10 years
    neg_spell = ifelse(neg_spell > 10, 10, neg_spell),
    neg_profits_spell = ifelse(neg_profits_spell > 10, 10, neg_profits_spell),
    cohort = case_when(
      first_date >= 1962 & first_date <= 1966 ~ "1962-1966",
      first_date >= 1967 & first_date <= 1971 ~ "1967-1971",
      first_date >= 1972 & first_date <= 1976 ~ "1972-1976",
      first_date >= 1977 & first_date <= 1981 ~ "1977-1981",
      first_date >= 1982 & first_date <= 1986 ~ "1982-1986",
      first_date >= 1987 & first_date <= 1991 ~ "1987-1991",
      first_date >= 1992 & first_date <= 1996 ~ "1992-1996",
      first_date >= 1997 & first_date <= 2001 ~ "1997-2001",
      first_date >= 2002 & first_date <= 2006 ~ "2002-2006",
      first_date >= 2007 & first_date <= 2010 ~ "2007-2010"
    )
  ) %>%
  filter(!is.na(cohort)) %>%
  group_by(cohort) %>%
  summarise(
    avg_neg_earnings_spell = mean(neg_spell[neg_ebitda == 1], na.rm = TRUE),
    avg_neg_profits_spell = mean(neg_profits_spell[neg_profits == 1], na.rm = TRUE),
    n_firms = n_distinct(gvkey),
    .groups = "drop"
  )

cohort_levels <- c("1962-1966","1967-1971","1972-1976","1977-1981","1982-1986",
                   "1987-1991","1992-1996","1997-2001","2002-2006","2007-2010")
neg_spell_bycohort$cohort <- factor(neg_spell_bycohort$cohort, levels = cohort_levels)

neg_spell_bycohort_long <- neg_spell_bycohort %>%
  pivot_longer(cols = c(avg_neg_earnings_spell, avg_neg_profits_spell), names_to = "variable", values_to = "value") %>%
  mutate(variable = recode(variable, avg_neg_earnings_spell = "Negative EBITDA", avg_neg_profits_spell = "Negative Profits"))

palette_2 <- viridis::inferno(2, begin = 0.0, end = 0.9)
theme_common <- theme_minimal(base_size = 18) +
  theme(
    text = element_text(family = "serif", size = 18),
    legend.position = "bottom"
  )
ggplot(neg_spell_bycohort_long, aes(x = cohort, y = value, color = variable, group = variable)) +
  geom_line(linewidth = 2) +
  geom_point(aes(size = n_firms)) +
  scale_color_manual(values = c("Negative EBITDA" = palette_2[1], "Negative Profits" = palette_2[2])) +
  scale_size_continuous(name = "", range = c(1, 5)) +
  labs(x = "Cohort (Entry Window)", y = "Average Spell Length (capped at 10)", color = "") +
  theme_common +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("figures/empirical/neg_spell_bycohort.pdf", width = 8, height = 6)

ggplot(neg_spell_bycohort_long, aes(x = cohort, y = value, color = variable, group = variable)) +
  geom_line(linewidth = 2) +
  geom_point(aes(size = n_firms)) +
  scale_color_manual(values = c("Negative EBITDA" = palette_2[1], "Negative Profits" = palette_2[2])) +
  scale_size_continuous(name = "", range = c(3, 6)) +
  labs(x = "Cohort (Entry Window)", y = "Average Spell Length", color = "") +
  theme_slides +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("figures/empirical/neg_spell_bycohort_slides.pdf", width = 10, height = 10)


# negative earnings age coefficients ----------------------------------------------------

# only date+sector FEs, no firm FEs
analysis_data$age <- analysis_data$date - analysis_data$first_date
neg_ebitda_reg <- feols(neg_ebitda ~ age + age:relevel(factor(date), ref = "1980") | date + sector, data = analysis_data)
neg_ebitda_reg <- summary(neg_ebitda_reg)
neg_profits_reg <- feols(neg_profits ~ age + age:relevel(factor(date), ref = "1980") | date + sector, data = analysis_data)
neg_profits_reg <- summary(neg_profits_reg)
# overlayed plot of coefficients +/- 95% confidence intervals
extract_year_coefs <- function(reg) {
  nms <- rownames(reg$coeftable)
  idx <- grepl("^age:relevel", nms)
  years <- as.numeric(gsub(".*\\)(\\d{4})$", "\\1", nms[idx]))
  data.frame(year = years, coef = reg$coeftable[idx, "Estimate"], se = reg$coeftable[idx, "Std. Error"])
}

coef_data <- rbind(
  cbind(extract_year_coefs(neg_ebitda_reg), variable = "neg_ebitda"),
  cbind(extract_year_coefs(neg_profits_reg), variable = "neg_profits")
) 

palette_2 <- viridis::inferno(2, begin = 0.0, end = 0.9)
theme_common <- theme_minimal(base_size = 18) +
  theme(
    text = element_text(family = "serif", size = 18),
    legend.position = "bottom"
  )

without_firm_fes <- ggplot(coef_data, aes(x = year, y = coef, color = variable)) +
  geom_line(linewidth = 1) +
  geom_point() +
  geom_errorbar(aes(ymin = coef - 1.65 * se, ymax = coef + 1.65 * se), width = 0.2) +
  scale_color_manual(values = c("neg_ebitda" = palette_2[1], "neg_profits" = palette_2[2]), labels = c("Negative EBITDA", "Negative Profits")) +
  labs(x = "Year", y = "Coefficient", color = "", title = "") +
  theme_common

# add firm FEs
neg_ebitda_reg <- feols(neg_ebitda ~ age + age:relevel(factor(date), ref = "1980") | gvkey + date, data = analysis_data)
neg_ebitda_reg <- summary(neg_ebitda_reg)
neg_profits_reg <- feols(neg_profits ~ age + age:relevel(factor(date), ref = "1980") | gvkey + date, data = analysis_data)
neg_profits_reg <- summary(neg_profits_reg)
coef_data <- rbind(
  cbind(extract_year_coefs(neg_ebitda_reg), variable = "neg_ebitda"),
  cbind(extract_year_coefs(neg_profits_reg), variable = "neg_profits")
) 

with_firm_fes <- ggplot(coef_data, aes(x = year, y = coef, color = variable)) +
  geom_line(linewidth = 1) +
  geom_point() +
  geom_errorbar(aes(ymin = coef - 1.65 * se, ymax = coef + 1.65 * se), width = 0.2) +
  scale_color_manual(values = c("neg_ebitda" = palette_2[1], "neg_profits" = palette_2[2]), labels = c("Negative EBITDA", "Negative Profits")) +
  labs(x = "Year", y = "Coefficient", color = "", title = "") +
  theme_common

ggarrange(without_firm_fes, with_firm_fes, nrow = 1, common.legend = TRUE, legend = "bottom")
ggsave("figures/empirical/neg_earnings_coefficients_dual.pdf", width = 8, height = 6)
ggsave("figures/empirical/neg_earnings_coefficients_solo.pdf", without_firm_fes, width = 8, height = 6)

