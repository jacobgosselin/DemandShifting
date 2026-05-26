setwd("/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings")
library(data.table)
library(haven)
library(tidyverse)
library(ggplot2)
library(viridis)

theme_common <- theme_minimal(base_size = 24) +
  theme(
    text = element_text(family = "serif", size = 24),
    legend.position = "bottom"
  )

# ---- load and compute bracket shares ---------------------------------------
d <- as.data.table(read_dta("data/raw/agg_brackets_assets_R5.dta"))

# restrict to post-1980 where bracket structure is most consistent
d <- d[year >= 1980]

# 10 core brackets present in all 39 post-1980 years
# (thres_low=0: $0 assets; thres_low=1: $1 to $500K; 100K/250K only in 21 years → collapse; 500M/2.5B only in 18 years → collapse)
CORE_BRACKETS <- c(0, 1, 5e5, 1e6, 5e6, 1e7, 2.5e7, 5e7, 1e8, 2.5e8)

# assign every non-core bracket to the highest core threshold <= its thres_low
d[, core_bracket := CORE_BRACKETS[findInterval(thres_low, CORE_BRACKETS)]]

# sum firm counts within each core bracket x year
d_agg <- d[, .(number = sum(number), number_total = number_total[1]),
             by = .(year, core_bracket)]

# bracket labels
bracket_labels <- c(
  "0"     = "$0",
  "1"     = "< $500K",
  "5e+05" = "$500K to $1M",
  "1e+06" = "$1M to $5M",
  "5e+06" = "$5M to $10M",
  "1e+07" = "$10M to $25M",
  "2.5e+07" = "$25M to $50M",
  "5e+07" = "$50M to $100M",
  "1e+08" = "$100M to $250M",
  "2.5e+08" = "$250M+"
)

d_agg[, pct_returns := number / number_total * 100]
d_agg[, bracket := factor(as.character(core_bracket),
                           levels = as.character(CORE_BRACKETS),
                           labels = unname(bracket_labels))]

# ---- Plot A: all brackets, percent of total returns -----------------------
pA <- ggplot(d_agg, aes(x = year, y = pct_returns, colour = bracket)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  scale_colour_manual(values = magma(10, begin = 0, end = 1),
                      name = "") +
  labs(
    title = "",
    x = "Year",
    y = "% of total returns"
  ) +
  theme_common +
  theme(text = element_text(size = 24))

ggsave("figures/empirical/bracket_return_shares_all_assets.pdf", pA,
       width = 10, height = 10)

# ---- Plot B: bottom vs. middle vs. top, collapsed -------------------------
d_agg[, group := fcase(
  core_bracket %in% c(0), "No Assets",
  core_bracket %in% c(1), "< $500K",
  core_bracket %in% c(5e5, 1e6, 5e6, 1e7, 2.5e7, 5e7, 1e8, 2.5e8), "> $500K"
)]

group_shares <- d_agg[, .(pct_returns = sum(pct_returns)), by = .(year, group)]
group_shares[, group := factor(group,
  levels = c("No Assets", "< $500K", "> $500K"))]

pB <- ggplot(group_shares, aes(x = year, y = pct_returns, colour = group)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  scale_colour_manual(
    values = c("No Assets" = "#2166ac",
               "< $500K" = "#4dac26",
               "> $500K" = "#d73027"),
    name = NULL
  ) +
  labs(
    title = "",
    x = "Year",
    y = "% of total returns"
  ) +
  theme_common

ggsave("figures/empirical/bracket_return_shares_3group_assets.pdf", pB,
       width = 8, height = 6)

ggsave("figures/empirical/bracket_return_shares_3group_assets_slides.pdf", pB,
       width = 10, height = 10)

# ---- Plot C: 5-group collapsed -------------------------
d_agg[, group_c := fcase(
  core_bracket %in% c(0),               "$0",
  core_bracket %in% c(1),               "< $500K",
  core_bracket %in% c(5e5, 1e6),        "$500K to $5M",
  core_bracket %in% c(5e6, 1e7),        "$5M to $25M",
  core_bracket %in% c(2.5e7, 5e7, 1e8), "$25M to $250M",
  core_bracket %in% c(2.5e8),           "$250M+"
)]

group_c_shares <- d_agg[, .(pct_returns = sum(pct_returns)), by = .(year, group_c)]
group_c_shares[, group_c := factor(group_c,
  levels = c("$0", "< $500K", "$500K to $5M", "$5M to $25M", "$25M to $250M", "$250M+"))]

pC <- ggplot(group_c_shares, aes(x = year, y = pct_returns, colour = group_c)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  scale_colour_manual(
    values = c("$0"            = "#762a83",
               "< $500K"       = "#2166ac",
               "$500K to $5M"  = "lightgreen",
               "$5M to $25M"   = "green",
               "$25M to $250M" = "darkgreen",
               "$250M+"        = "#d73027"),
    name = NULL
  ) +
  labs(
    title = "",
    x = "Year",
    y = "% of total returns"
  ) +
  theme_common

ggsave("figures/empirical/bracket_return_shares_4group_assets.pdf", pC,
       width = 8, height = 6)
