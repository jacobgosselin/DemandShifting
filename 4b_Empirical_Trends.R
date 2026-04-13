setwd("/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings")
# load libraries
library(tidyr)
library(dplyr)
library(ggplot2)
library(readr)
library(ggpubr)
library(viridis)

# load data ----------------------------------------------------

load("data/clean/analysis_data.RData")

# Common theme and palette for all plots
theme_common <- theme_minimal(base_size = 24) +
  theme(
    text = element_text(family = "serif", size = 24),
    # plot.title = element_text(face = "bold"),
    legend.position = "bottom"
  )

# a. Plot cost ratios over time (all firms) ------------------------------

palette_4 <- viridis::inferno(4, begin = 0.0, end = 0.9)

cost_ratios_by_year <- analysis_data %>%
  group_by(date) %>%
  reframe(
    median_rd_sale   = median(rd_sale, na.rm = TRUE),
    median_sga_sale  = median(sga_sale, na.rm = TRUE),
    median_cogs_sale = median(cogs_sale, na.rm = TRUE),
    median_capx_sale = median(capx_sale, na.rm = TRUE),
    .groups = "drop"
  )

cost_ratios_long <- cost_ratios_by_year %>%
  pivot_longer(cols = c(median_rd_sale, median_sga_sale, median_cogs_sale, median_capx_sale),
               names_to = "cost_type",
               values_to = "median_ratio")

p_all <- ggplot(cost_ratios_long, aes(x = date, y = median_ratio, color = cost_type, group = cost_type)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  scale_y_continuous(limits = c(0, 0.8)) +
  scale_x_continuous(breaks = seq(1975, 2020, 5)) +
  # scale_color_manual(
  #   values = c("median_rd_sale" = "#E31A1C", "median_sga_sale" = "#1F78B4", "median_cogs_sale" = "#33A02C", "median_capx_sale" = "#FF7F00"),
  #   labels = c("median_rd_sale" = "R&D/Sales", "median_sga_sale" = "SG&A/Sales", "median_cogs_sale" = "COGS/Sales", "median_capx_sale" = "CapEx/Sales")
  # ) +
  scale_color_manual(
    values = palette_4,
    labels = c("median_rd_sale" = "R&D/Sales", "median_sga_sale" = "SG&A/Sales", "median_cogs_sale" = "COGS/Sales", "median_capx_sale" = "CapEx/Sales")
  ) +
  labs(
    title = "",
    x = "Year",
    y = "Median Ratio",
    color = ""
  ) +
  theme_common

ggsave("figures/empirical/cost_ratios_by_year.pdf", width = 8, height = 6)

# b. Same but only for firms with EBITDA < 0 ----------------------------

cost_ratios_neg_ebitda_by_year <- analysis_data %>%
  filter(ebitda < 0) %>%
  group_by(date) %>%
  reframe(
    median_rd_sale   = median(rd_sale, na.rm = TRUE),
    median_sga_sale  = median(sga_sale, na.rm = TRUE),
    median_cogs_sale = median(cogs_sale, na.rm = TRUE),
    median_capx_sale = median(capx_sale, na.rm = TRUE),
    .groups = "drop"
  )

cost_ratios_neg_long <- cost_ratios_neg_ebitda_by_year %>%
  pivot_longer(cols = c(median_rd_sale, median_sga_sale, median_cogs_sale, median_capx_sale),
               names_to = "cost_type",
               values_to = "median_ratio")

p_neg <- ggplot(cost_ratios_neg_long, aes(x = date, y = median_ratio, color = cost_type, group = cost_type)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  scale_x_continuous(breaks = seq(1975, 2020, 5)) +
  # y-axis limits 
  scale_y_continuous(limits = c(0, 0.8)) +
  # scale_color_manual(
  #   values = c("median_rd_sale" = "#E31A1C", "median_sga_sale" = "#1F78B4", "median_cogs_sale" = "#33A02C", "median_capx_sale" = "#FF7F00"),
  #   labels = c("median_rd_sale" = "R&D/Sales", "median_sga_sale" = "SG&A/Sales", "median_cogs_sale" = "COGS/Sales", "median_capx_sale" = "CapEx/Sales")
  # ) +
  scale_color_manual(
    values = palette_4,
    labels = c("median_rd_sale" = "R&D/Sales", "median_sga_sale" = "SG&A/Sales", "median_cogs_sale" = "COGS/Sales", "median_capx_sale" = "CapEx/Sales")
  ) +
  labs(
    title = "",
    x = "Year",
    y = "Median Ratio",
    color = ""
  ) +
  theme_common

ggsave("figures/empirical/cost_ratios_neg_ebitda_by_year.pdf", width = 8, height = 6)

# c. 2-panel figure ----------------------------

ggarrange(p_all, p_neg,
          labels = c("All Firms", "Negative Earnings Firms"),
          font.label = list(size = 24, face = "bold", family = "serif"),
          ncol = 2, nrow = 1,
          common.legend = TRUE, legend = "bottom")

ggsave("figures/empirical/cost_ratios_2panel.pdf", width = 16, height = 9)

