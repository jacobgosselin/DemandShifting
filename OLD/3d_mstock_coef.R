setwd("/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings")
library(fixest)
library(dplyr)
library(ggplot2)

# Common plot theme
theme_common <- theme_minimal(base_size = 24) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "bottom"
  )

# Load calibrated delta_m (or default to 0.15) ----
cal_path <- "code/3_ComputationalEx/calibrated_investment_params.csv"
if (file.exists(cal_path)) {
  cal <- read.csv(cal_path)
  delta_m_use <- cal$delta_m[1]
  cat("Loaded calibrated delta_m:", delta_m_use, "\n")
} else {
  delta_m_use <- 0.15
  cat("calibrated_investment_params.csv not found; using delta_m =", delta_m_use, "\n")
}

# Load prepped data ----
# analysis_data_mstock_input: full firm histories (pre-analysis-filter) for m_stock construction
load("data/intermediate/analysis_data_prepped.RData")

# Define coefficient computation function ----
# Accepts delta_m as argument; replicates feols(log(sale) ~ log(m_stock):i(date) | date:naics_2digit + gvkey)
compute_coefs_byyear <- function(data, med_preIPO_growth, delta_m = 0.15) {
  # Construct m_stock on full firm history, then apply analysis filters.
  # Matching original 3b_Est_Structural.R: m_stock built before filters.
  analysis <- data %>%
    group_by(gvkey) %>%
    arrange(gvkey, date) %>%
    mutate(
      m_stock = {
      delta <- delta_m
      g <- med_preIPO_growth
      r <- (1 - delta) / (1 + g)
      init_val <- first(m_inv) * (1 - r^first(age)) / (1 - r) # M_0 = m_inv_0 * (1 - r^age)(1 - r)
      # Perpetual inventory: m_t = (1 - delta) * m_{t-1} + m_inv_t
      Reduce(function(prev, curr) (1 - delta) * prev + curr,
             m_inv[-1],
             init = init_val,
             accumulate = TRUE)
      }
    ) %>%
    ungroup() %>%
    filter(!is.na(ebitda) & !is.na(sale) & !is.na(cogs)) %>%
    filter(date >= 1980 & date < 2020) %>%
    filter(!(naics_2digit %in% c(22, 52, 99)))

  # Singleton and finiteness filters for regression
  reg_data <- analysis %>%
    mutate(
      log_sale    = log(sale),
      log_m_stock = log(m_stock)
    ) %>%
    filter(
      !is.na(log_sale)    & !is.na(log_m_stock) &
      !is.infinite(log_sale) & !is.infinite(log_m_stock)
    ) %>%
    group_by(gvkey) %>%
    filter(n() > 1) %>%
    ungroup() %>%
    group_by(date, naics_2digit) %>%
    filter(n() > 1) %>%
    ungroup()

  # Run two-way FE regression with year-interacted log(m_stock)
  reg <- feols(
    log(sale) ~ log(m_stock):i(date) | date:naics_2digit + gvkey,
    data = reg_data
  )
  reg <- summary(reg, se = "hetero")

  # Extract coefficients and confidence intervals
  coef_year <- data.frame(
    year = as.numeric(gsub("log\\(m_stock\\):date::", "", names(coef(reg)))),
    coef = coef(reg),
    se   = se(reg)
  ) %>%
    mutate(
      ci_lower = coef - 1.65 * se,
      ci_upper = coef + 1.65 * se
    ) %>%
    select(year, coef, ci_lower, ci_upper)

  return(coef_year)
}

# Compute coefficients using full firm histories ----
coef_year_delta15_check <- compute_coefs_byyear(analysis_data_mstock_input, med_preIPO_growth, delta_m = 0.15)
coef_year <- compute_coefs_byyear(analysis_data_mstock_input, med_preIPO_growth, 0.15)
cat("Years in coefficient table:", nrow(coef_year), "\n")

# Print coef_year for verification
print(coef_year)

# Plot: sales elasticity of customer capital over time ----
ggplot(coef_year, aes(x = year, y = coef)) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), alpha = 0.2) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  labs(
    title = "",
    x     = "Year",
    y     = "Sales Elasticity of Customer Capital"
  ) +
  theme_common

ggsave("figures/sales_elasticity_m_by_year.pdf", width = 10, height = 10)

# Save coefficients for reference
write.csv(coef_year, "data/clean/sales_elasticity_m_by_year.csv", row.names = FALSE)
write.csv(coef_year_delta15_check, "data/clean/coefs_byyear_delta_m_15.csv", row.names = FALSE)

cat("3d_mstock_coef.R complete. Used delta_m =", delta_m_use, "\n")
cat("Saved: figures/sales_elasticity_m_by_year.pdf\n")
cat("Saved: data/clean/sales_elasticity_m_by_year.csv\n")
