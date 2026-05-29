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

# ---- load and aggregate to 10 core brackets --------------------------------
CORE_BRACKETS <- c(0, 1, 5e5, 1e6, 5e6, 1e7, 2.5e7, 5e7, 1e8, 2.5e8)

d <- as.data.table(read_dta("data/raw/agg_brackets_assets_R5.dta"))
d <- d[year >= 1980]
d[, core_bracket := CORE_BRACKETS[findInterval(thres_low, CORE_BRACKETS)]]

d_agg <- d[, .(number = sum(number), number_total = number_total[1]),
             by = .(year, core_bracket)]

# 5-group collapse: individual bins up through $1M–$5M, everything above → "$5M+"
d_agg[, group := fcase(
  core_bracket %in% c(0),                                          "$0",
  core_bracket %in% c(1),                                          "< $500K",
  core_bracket %in% c(5e5),                                        "$500K to $1M",
  core_bracket %in% c(1e6),                                        "$1M to $5M",
  core_bracket %in% c(5e6, 1e7, 2.5e7, 5e7, 1e8, 2.5e8),         "$5M+"
)]

group_dt <- d_agg[, .(pct_firms = sum(number) / number_total[1] * 100),
                  by = .(year, group)]
group_dt[, group := factor(group, levels = c("$0", "< $500K", "$500K to $1M",
                                              "$1M to $5M", "$5M+"))]

fill_vals <- c(
  "$0"           = "#762a83",
  "< $500K"      = "#2166ac",
  "$500K to $1M" = "#74add1",
  "$1M to $5M"   = "#4dac26",
  "$5M+"         = "#d73027"
)

# ---- Plot 1: all 5 groups --------------------------------------------------
# Split at 2001/2002 reporting-change discontinuity so areas don't interpolate across the break
group_dt_pre  <- group_dt[year <= 2001]
group_dt_post <- group_dt[year >= 2002]

p_all <- ggplot(mapping = aes(x = year, y = pct_firms, fill = group)) +
  geom_area(data = group_dt_pre,  position = "stack", colour = "white", linewidth = 0.5) +
  geom_area(data = group_dt_post, position = "stack", colour = "white", linewidth = 0.5) +
  geom_vline(xintercept = 2001.5, linetype = "dashed", colour = "black", linewidth = 0.8) +
  scale_fill_manual(values = fill_vals, name = NULL) +
  labs(x = "Year", y = "% of total firms") +
  theme_common

ggsave("figures/empirical/bracket_flow_firms_assets.pdf", p_all,
       width = 10, height = 10)

# ---- Plot 2: exclude $0 assets, rescale to sum to 100 ----------------------
d_ex <- d_agg[group != "$0"]
total_ex <- d_agg[, .(total_ex = sum(number[group != "$0"])), by = year]
d_ex <- merge(d_ex, total_ex, by = "year")
group_ex <- d_ex[, .(pct_firms = sum(number) / total_ex[1] * 100),
                 by = .(year, group)]
group_ex[, group := factor(group, levels = c("< $500K", "$500K to $1M",
                                              "$1M to $5M", "$5M+"))]

p_ex <- ggplot(group_ex, aes(x = year, y = pct_firms, fill = group)) +
  geom_area(position = "stack", colour = "white", linewidth = 0.5) +
  scale_fill_manual(values = fill_vals, name = NULL) +
  labs(x = "Year", y = "% of firms (excl. $0 assets)") +
  theme_common

ggsave("figures/empirical/bracket_flow_firms_assets_ex_zero.pdf", p_ex,
       width = 10, height = 10)
