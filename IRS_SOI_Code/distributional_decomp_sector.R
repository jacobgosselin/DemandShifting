setwd("/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings")
library(data.table)
library(haven)
library(tidyverse)
library(ggplot2)
library(viridis)

theme_common <- theme_minimal(base_size = 18) +
  theme(
    text = element_text(family = "serif", size = 18),
    legend.position = "bottom"
  )

# ---- load and compute bracket shares ---------------------------------------
d <- as.data.table(read_dta("data/raw/sector_brackets_receipts_R5.dta"))

# drop the aggregate "All" sector and Finance (bracket structure differs)
d <- d[!sector_main %in% c("All", "Finance", "Utilities")]

# restrict to post-1980
d <- d[year >= 1980]

# drop sector × year observations where IRS suppressed/merged brackets
bad_sectorYears <- unique(d[bracket_deletion_total == "yes", .(sector_main, year)])
d <- d[!bad_sectorYears, on = .(sector_main, year)]

# 8 core brackets present in all post-1980 years
CORE_BRACKETS <- c(0, 25e3, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7)

d[, core_bracket := CORE_BRACKETS[findInterval(thres_low, CORE_BRACKETS)]]

# sum firm counts within each core bracket × sector × year
# number_total is sector-specific (same for all rows within sector × year)
d_agg <- d[, .(number = sum(number), number_total = number_total[1]),
             by = .(sector_main, year, core_bracket)]

# bracket labels
bracket_labels <- c(
  "0"     = "$0 to $25K",
  "25000" = "$25K to $100K",
  "1e+05" = "$100K to $500K",
  "5e+05" = "$500K to $1M",
  "1e+06" = "$1M to $5M",
  "5e+06" = "$5M to $10M",
  "1e+07" = "$10M to $50M",
  "5e+07" = "$50M+"
)

d_agg[, pct_returns := number / number_total * 100]
d_agg[, bracket := factor(as.character(core_bracket),
                           levels = as.character(CORE_BRACKETS),
                           labels = unname(bracket_labels))]

SECTORS <- sort(unique(d_agg$sector_main))

# ---- Plot A: all brackets, percent of total returns, faceted by sector -----
pA <- ggplot(d_agg, aes(x = year, y = pct_returns, colour = bracket)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.5) +
  scale_colour_manual(values = viridis(8, begin = 0.1, end = 0.9)) +
  facet_wrap(~ sector_main, scales = "free_y") +
  labs(
    title = "",
    x = "Year",
    y = "% of total returns"
  ) +
  theme_common +
  theme(strip.text = element_text(size = 13))

ggsave("figures/empirical/bracket_return_shares_all_bysector.pdf", pA,
       width = 14, height = 10)

# ---- Plot B: bottom vs. middle vs. top, collapsed, faceted by sector -------
d_agg[, group := fcase(
  core_bracket %in% c(0),                   "Bottom (< $25K)",
  core_bracket %in% c(25e3, 1e5, 5e5),      "Middle ($25K to $1M)",
  core_bracket %in% c(1e6, 5e6, 1e7, 5e7), "Top (> $1M)"
)]

group_shares <- d_agg[, .(pct_returns = sum(pct_returns)), by = .(sector_main, year, group)]
group_shares[, group := factor(group,
  levels = c("Bottom (< $25K)", "Middle ($25K to $1M)", "Top (> $1M)"))]

pB <- ggplot(group_shares, aes(x = year, y = pct_returns, colour = group)) +
  geom_line(linewidth = 1.5) +
  geom_point(size = 2) +
  scale_colour_manual(
    values = c("Bottom (< $25K)"      = "#2166ac",
               "Middle ($25K to $1M)" = "#4dac26",
               "Top (> $1M)"          = "#d73027"),
    name = NULL
  ) +
  facet_wrap(~ sector_main, scales = "free_y") +
  labs(
    title = "",
    x = "Year",
    y = "% of total returns"
  ) +
  theme_common +
  theme(strip.text = element_text(size = 13))

ggsave("figures/empirical/bracket_return_shares_3group_bysector.pdf", pB,
       width = 14, height = 10)

# ---- Plot C: 5-group split, faceted by sector ------------------------------
d_agg[, group_c := fcase(
  core_bracket %in% c(0),        "< $25K",
  core_bracket %in% c(25e3),     "$25K to $100K",
  core_bracket %in% c(1e5),      "$100K to $500K",
  core_bracket %in% c(5e5),      "$500K to $1M",
  core_bracket %in% c(1e6, 5e6, 1e7, 5e7), "$1M+"
)]

group_c_shares <- d_agg[, .(pct_returns = sum(pct_returns)), by = .(sector_main, year, group_c)]
group_c_shares[, group_c := factor(group_c,
  levels = c("< $25K", "$25K to $100K", "$100K to $500K", "$500K to $1M", "$1M+"))]

pC <- ggplot(group_c_shares, aes(x = year, y = pct_returns, colour = group_c)) +
  geom_line(linewidth = 1.5) +
  geom_point(size = 2) +
  scale_colour_manual(
    values = c("< $25K"         = "#2166ac",
               "$25K to $100K"  = "lightgreen",
               "$100K to $500K" = "green",
               "$500K to $1M"   = "darkgreen",
               "$1M+"           = "#d73027"),
    name = NULL
  ) +
  facet_wrap(~ sector_main, scales = "free_y") +
  labs(
    title = "",
    x = "Year",
    y = "% of total returns"
  ) +
  theme_common +
  theme(strip.text = element_text(size = 13))

ggsave("figures/empirical/bracket_return_shares_4group_bysector.pdf", pC,
       width = 14, height = 10)
