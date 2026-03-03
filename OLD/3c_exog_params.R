setwd("/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings")
library(dplyr)
library(lubridate)

# Load prepped data from 3b_mstock_prep.R ----
analysis_data_prepped <- read.csv("data/intermediate/analysis_data_prepped.csv")

# Compute ratios needed for moments
analysis_data_prepped <- analysis_data_prepped %>%
  mutate(
    sga_sale = sga_PandT / sale,
    capx_sale = capx / sale
  )

# AR(1) productivity parameters (sector-weighted) ----
ar1_data <- read.csv("data/clean/ar1_productivity.csv") %>%
  mutate(sector = as.double(sector))

sector_weights <- analysis_data_prepped %>%
  mutate(sector = as.double(naics_2digit)) %>%
  group_by(sector) %>%
  summarise(weight = n()) %>%
  ungroup() %>%
  mutate(weight = weight / sum(weight))

ar1_data <- merge(ar1_data, sector_weights, by = "sector")
struct_data <- ar1_data %>%
  summarise(
    rho = sum(rho * weight),
    sigma_xi = sum(sigma_xi * weight)
  )

# Production function parameters (sector-weighted) ----
prod_params <- read.csv("data/clean/prod_fncts_params.csv") %>%
  mutate(sector = as.double(sector))

prod_params <- merge(prod_params, sector_weights, by = "sector")
struct_data$gamma_l <- sum(prod_params$beta_l * prod_params$weight)
struct_data$gamma_k <- sum(prod_params$beta_k * prod_params$weight)

# Exogenous exit rate ----
exit_data <- analysis_data_prepped %>%
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
entry_data <- analysis_data_prepped %>%
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
base_year <- analysis_data_prepped %>%
  filter(date == 1980) %>%
  mutate(neg_ebitda = ifelse(ebitda < 0, 1, 0))

final_year <- analysis_data_prepped %>%
  filter(date == 2019) %>%
  mutate(neg_ebitda = ifelse(ebitda < 0, 1, 0))

sga_sale_median              <- median(base_year$sga_sale, na.rm = TRUE)
capx_sale_median             <- median(base_year$capx_sale, na.rm = TRUE)
sga_sale_negebitda_median    <- median(base_year$sga_sale[base_year$neg_ebitda == 1], na.rm = TRUE)
capx_sale_negebitda_median   <- median(base_year$capx_sale[base_year$neg_ebitda == 1], na.rm = TRUE)
neg_ebitda_base              <- mean(base_year$neg_ebitda, na.rm = TRUE)
neg_ebitda_final             <- mean(final_year$neg_ebitda, na.rm = TRUE)

struct_data$med_sales               <- sga_sale_median
struct_data$med_capx_sale           <- capx_sale_median
struct_data$med_sales_negebitda     <- sga_sale_negebitda_median
struct_data$med_capx_sale_negebitda <- capx_sale_negebitda_median
struct_data$neg_ebitda_base         <- neg_ebitda_base
struct_data$neg_ebitda_final        <- neg_ebitda_final

write.csv(struct_data, "data/clean/structural_parameters.csv", row.names = FALSE)

cat("3c_exog_params.R complete.\n")
cat("rho:", struct_data$rho, "  sigma_xi:", struct_data$sigma_xi, "\n")
cat("exit_rate:", exit_rate, "  entry_rate:", entry_rate, "\n")
cat("neg_ebitda_base:", neg_ebitda_base, "  neg_ebitda_final:", neg_ebitda_final, "\n")
