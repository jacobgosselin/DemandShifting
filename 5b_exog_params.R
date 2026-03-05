setwd("/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings")
REPO_DIR <- "/Users/jacobgosselin/Documents(local)/GitHub/DemandShifting"
library(dplyr)
library(lubridate)

# Load analysis data from 3a_build_analysis_data.R ----
load("data/clean/analysis_data.RData")

# AR(1) and production function parameters (sector-weighted) ----
acf_data <- read.csv("data/clean/ACF_bysector.csv") %>%
  mutate(naics_2digit = as.double(naics_2digit))

sector_weights <- analysis_data %>%
  group_by(naics_2digit) %>%
  summarise(weight = n()) %>%
  ungroup() %>%
  mutate(weight = weight / sum(weight))

acf_data <- merge(acf_data, sector_weights, by = "naics_2digit")

struct_data <- acf_data %>%
  summarise(
    rho      = sum(rho      * weight),
    sigma_xi = sum(sigma_xi * weight),
    gamma_l  = sum(beta_l   * weight),
    gamma_k  = sum(beta_k   * weight)
  )

# Exogenous exit rate ----
exit_data <- analysis_data %>%
  group_by(gvkey) %>%
  arrange(date) %>%
  mutate(
    exit = ifelse(is.na(lead(sale)), 1, 0)
  ) %>%
  ungroup()

exit_rate_byyear <- exit_data %>%
  group_by(date) %>%
  summarise(exit_rate = mean(exit, na.rm = TRUE)) %>%
  filter(date < 2019)

exit_rate <- mean(exit_rate_byyear$exit_rate, na.rm = TRUE)
struct_data$exit_rate <- exit_rate

# Entry rate ----
entry_data <- analysis_data %>%
  group_by(gvkey) %>%
  arrange(date) %>%
  mutate(
    entry = ifelse(is.na(lag(sale)), 1, 0)
  ) %>%
  ungroup()

entry_rate_byyear <- entry_data %>%
  group_by(date) %>%
  summarise(entry_rate = mean(entry, na.rm = TRUE)) %>%
  filter(date > 1980)

entry_rate <- mean(entry_rate_byyear$entry_rate, na.rm = TRUE)
struct_data$entry_rate <- entry_rate

# Calibration moment targets ----
base_year  <- analysis_data %>% filter(date == 1980)
final_year <- analysis_data %>% filter(date == 2019)

sga_sale_median              <- median(base_year$sga_sale,                               na.rm = TRUE)
capx_sale_median             <- median(base_year$capx_sale,                              na.rm = TRUE)
sga_sale_negebitda_median    <- median(base_year$sga_sale[base_year$neg_ebitda == 1],    na.rm = TRUE)
capx_sale_negebitda_median   <- median(base_year$capx_sale[base_year$neg_ebitda == 1],   na.rm = TRUE)
neg_ebitda_base              <- mean(base_year$neg_ebitda,                               na.rm = TRUE)
neg_ebitda_final             <- mean(final_year$neg_ebitda,                              na.rm = TRUE)

struct_data$med_sales               <- sga_sale_median
struct_data$med_capx_sale           <- capx_sale_median
struct_data$med_sales_negebitda     <- sga_sale_negebitda_median
struct_data$med_capx_sale_negebitda <- capx_sale_negebitda_median
struct_data$neg_ebitda_base         <- neg_ebitda_base
struct_data$neg_ebitda_final        <- neg_ebitda_final

write.csv(struct_data, file.path(REPO_DIR, "6_ComputationalEx", "structural_parameters.csv"), row.names = FALSE)

cat("4c_exog_params.R complete.\n")
cat("rho:", struct_data$rho, "  sigma_xi:", struct_data$sigma_xi, "\n")
cat("gamma_l:", struct_data$gamma_l, "  gamma_k:", struct_data$gamma_k, "\n")
cat("exit_rate:", exit_rate, "  entry_rate:", entry_rate, "\n")
cat("neg_ebitda_base:", neg_ebitda_base, "  neg_ebitda_final:", neg_ebitda_final, "\n")
