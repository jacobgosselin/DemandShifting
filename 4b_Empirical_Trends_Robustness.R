setwd("/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings")
library(tidyr)
library(dplyr)
library(stringr)
library(ggplot2)
library(ggpubr)
library(viridis)
library(readr)

# load data ----------------------------------------------------

load("data/clean/analysis_data.RData")
analysis_data <- analysis_data %>% filter(biotech_flag == 0)

# Common theme and palette
theme_common <- theme_minimal(base_size = 24) +
  theme(
    text            = element_text(family = "serif", size = 24),
    legend.position = "bottom"
  )

palette_2 <- viridis::inferno(2, begin = 0.0, end = 0.9)

# Add naics_2digit
analysis_data <- analysis_data %>%
  mutate(naics_2digit = as.numeric(str_sub(naics_3digit, 1, 2)))

# -----------------------------------------------------------------------
# Helper: within/between decomposition (base year = 1980)
#
#   mu_t        = sum_g  w_{g,t} * mu_{g,t}
#   within_t    = sum_g  w_{g,0} * (mu_{g,t}  - mu_{g,0})
#   between_t   = sum_g (w_{g,t} - w_{g,0})   * mu_{g,t}
#
# group_var : bare column name (symbol) of the grouping variable
# value_var : bare column name (symbol) of the outcome to average
# subset    : logical vector (or TRUE) selecting the denominator population
# -----------------------------------------------------------------------

decompose_within_between <- function(df, group_var, value_var, base_year = 1980) {
  gv <- ensym(group_var)
  vv <- ensym(value_var)

  # group-level means and firm counts
  grp <- df %>%
    group_by(date, !!gv) %>%
    summarise(grp_mean = mean(!!vv, na.rm = TRUE),
              n_grp    = n(),
              .groups  = "drop")

  # total counts per year for weights
  tot <- df %>%
    group_by(date) %>%
    summarise(n_tot = n(), .groups = "drop")

  grp <- grp %>%
    left_join(tot, by = "date") %>%
    mutate(w = n_grp / n_tot)

  # base-year anchors
  base <- grp %>%
    filter(date == base_year) %>%
    select(!!gv, mu0 = grp_mean, w0 = w)

  grp %>%
    inner_join(base, by = as_label(gv)) %>%
    group_by(date) %>%
    summarise(
      within_component  = sum(w0  * (grp_mean - mu0), na.rm = TRUE),
      between_component = sum((w - w0) * grp_mean,    na.rm = TRUE),
      .groups = "drop"
    )
}

# -----------------------------------------------------------------------
# Helper: build a single two-panel decomposition figure
#
# sector_decomp / age_decomp : output of decompose_within_between()
# y_label                    : y-axis label string
# within_sector_label        : legend label for within-sector component
# between_sector_label       : legend label for between-sector component
# within_age_label           : legend label for within-age component
# between_age_label          : legend label for between-age component
# -----------------------------------------------------------------------

make_decomp_figure <- function(sector_decomp, age_decomp,
                               y_label,
                               within_sector_label  = "Within Sector",
                               between_sector_label = "Between Sector",
                               within_age_label     = "Within Age",
                               between_age_label    = "Between Age") {

  # sector panel
  sector_long <- sector_decomp %>%
    pivot_longer(cols      = c(within_component, between_component),
                 names_to  = "component",
                 values_to = "value") %>%
    mutate(component = factor(component,
                              levels = c("within_component", "between_component"),
                              labels = c(within_sector_label, between_sector_label)))

  p_sector <- ggplot(sector_long, aes(x = date, y = value, color = component)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    geom_line(linewidth = 2) +
    geom_point(size = 3) +
    scale_color_manual(values = setNames(palette_2,
                                         c(within_sector_label, between_sector_label))) +
    labs(x = "Year", y = y_label, color = "", title = "Sector Decomposition") +
    theme_common

  # age panel
  age_long <- age_decomp %>%
    pivot_longer(cols      = c(within_component, between_component),
                 names_to  = "component",
                 values_to = "value") %>%
    mutate(component = factor(component,
                              levels = c("within_component", "between_component"),
                              labels = c(within_age_label, between_age_label)))

  p_age <- ggplot(age_long, aes(x = date, y = value, color = component)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    geom_line(linewidth = 2) +
    geom_point(size = 3) +
    scale_color_manual(values = setNames(palette_2,
                                         c(within_age_label, between_age_label))) +
    labs(x = "Year", y = y_label, color = "", title = "Age Decomposition") +
    theme_common

  ggarrange(p_sector, p_age, ncol = 2, nrow = 1)
}

# -----------------------------------------------------------------------
# Figure 1: neg_ebitda (share of firms with EBITDA < 0)
#   Outcome  = neg_ebitda (0/1); mean = share
#   Universe = full sample
# -----------------------------------------------------------------------

decomp_negebitda_sector <- decompose_within_between(
  analysis_data, naics_2digit, neg_ebitda)

decomp_negebitda_age    <- decompose_within_between(
  analysis_data, age, neg_ebitda)

fig1 <- make_decomp_figure(
  decomp_negebitda_sector, decomp_negebitda_age,
  y_label = ""
)

ggsave("figures/empirical/decomp_neg_ebitda.pdf", fig1, width = 16, height = 9)

# -----------------------------------------------------------------------
# Figure 2: neg_profits (share of firms with profits < 0)
#   Outcome  = neg_profits (0/1); mean = share
#   Universe = full sample
# -----------------------------------------------------------------------

decomp_negprofits_sector <- decompose_within_between(
  analysis_data, naics_2digit, neg_profits)

decomp_negprofits_age    <- decompose_within_between(
  analysis_data, age, neg_profits)

fig2 <- make_decomp_figure(
  decomp_negprofits_sector, decomp_negprofits_age,
  y_label = ""
)

ggsave("figures/empirical/decomp_neg_profits.pdf", fig2, width = 16, height = 9)

# -----------------------------------------------------------------------
# Figure 3: neg_spell (average spell length among neg_ebitda == 1 firms)
#   Outcome  = neg_spell; mean = average spell length
#   Universe = neg_ebitda == 1
# -----------------------------------------------------------------------

neg_ebitda_data <- analysis_data %>% filter(neg_ebitda == 1)

decomp_negspell_sector <- decompose_within_between(
  neg_ebitda_data, naics_2digit, neg_spell)

decomp_negspell_age    <- decompose_within_between(
  neg_ebitda_data, age, neg_spell)

fig3 <- make_decomp_figure(
  decomp_negspell_sector, decomp_negspell_age,
  y_label = ""
)

ggsave("figures/empirical/decomp_neg_spell.pdf", fig3, width = 16, height = 9)

# -----------------------------------------------------------------------
# Figure 4: neg_profits_spell (average spell length among neg_profits == 1 firms)
#   Outcome  = neg_profits_spell; mean = average spell length
#   Universe = neg_profits == 1
# -----------------------------------------------------------------------

neg_profits_data <- analysis_data %>% filter(neg_profits == 1)

decomp_negprofitsspell_sector <- decompose_within_between(
  neg_profits_data, naics_2digit, neg_profits_spell)

decomp_negprofitsspell_age    <- decompose_within_between(
  neg_profits_data, age, neg_profits_spell)

fig4 <- make_decomp_figure(
  decomp_negprofitsspell_sector, decomp_negprofitsspell_age,
  y_label = ""
)

ggsave("figures/empirical/decomp_neg_profits_spell.pdf", fig4, width = 16, height = 9)
