setwd("/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings")
REPO_DIR <- "/Users/jacobgosselin/Documents(local)/GitHub/DemandShifting"
library(fixest)
library(dplyr)
library(ggplot2)

# Common plot theme
theme_common <- theme_minimal(base_size = 24) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "bottom"
  )

# Load analysis data (m_stock already constructed by 3a_build_analysis_data.R) ----
load("data/clean/analysis_data.RData")

# Singleton and finiteness filters for regression ----
reg_data <- analysis_data %>%
  filter(
    !is.na(sale) & !is.na(m_stock) &
    !is.infinite(sale) & !is.infinite(m_stock) & 
    m_stock > 0 & sale > 0
  ) %>%
  mutate(
    log_sale    = log(sale),
    log_m_stock = log(m_stock),
    log_k_stock = log(ppegt)
  ) %>%
  filter(
    !is.na(log_sale)    & !is.na(log_m_stock) & !is.na(log_k_stock) &
    !is.infinite(log_sale) & !is.infinite(log_m_stock) & !is.infinite(log_k_stock)
  ) %>%
  group_by(gvkey) %>%
  filter(n() > 1) %>%
  ungroup()

# Two-way FE regression with year-interacted log(m_stock) ----
reg <- feols(
  log_sale ~ log_m_stock:i(date) + log_k_stock | gvkey + sector:date,
  data = reg_data
)
reg <- summary(reg, se = "hetero")

# Extract coefficients and confidence intervals ----
coef_year <- data.frame(
  year = as.numeric(gsub("log_m_stock:date::", "", names(coef(reg)))),
  coef = coef(reg),
  se   = se(reg)
) %>%
  mutate(
    ci_lower = coef - 1.65 * se,
    ci_upper = coef + 1.65 * se
  ) %>%
  select(year, coef, ci_lower, ci_upper) %>%
  filter( 
    !is.na(year)
  )

print(coef_year)

# Plot ----
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

ggsave("figures/empirical/sales_elasticity_m_by_year.pdf", width = 10, height = 10)

# Save coefficients ----
write.csv(coef_year, file.path(REPO_DIR, "6_ComputationalEx", "sales_elasticity_m_by_year.csv"), row.names = FALSE)

cat("4b_mstock_coef.R complete.\n")
cat("Years in coefficient table:", nrow(coef_year), "\n")
cat("Saved: figures/sales_elasticity_m_by_year.pdf\n")
cat("Saved: data/clean/sales_elasticity_m_by_year.csv\n")
