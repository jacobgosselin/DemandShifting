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

# ---- load and aggregate to 8 core brackets ---------------------------------
CORE_BRACKETS <- c(0, 25e3, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7)

d <- as.data.table(read_dta("data/raw/sector_brackets_receipts_R5.dta"))
d <- d[!sector_main %in% c("All", "Finance", "Utilities")]
d <- d[year >= 1980]

bad_sectorYears <- unique(d[bracket_deletion_total == "yes", .(sector_main, year)])
d <- d[!bad_sectorYears, on = .(sector_main, year)]

d[, core_bracket := CORE_BRACKETS[findInterval(thres_low, CORE_BRACKETS)]]

d_agg <- d[, .(number = sum(number), number_total = number_total[1]),
             by = .(sector_main, year, core_bracket)]

# 6-group collapse matching distributional_decomp.R
d_agg[, group := fcase(
  core_bracket %in% c(0),              "< $25K",
  core_bracket %in% c(25e3),           "$25K to $100K",
  core_bracket %in% c(1e5),            "$100K to $500K",
  core_bracket %in% c(5e5),            "$500K to $1M",
  core_bracket %in% c(1e6),            "$1M to $5M",
  core_bracket %in% c(5e6, 1e7, 5e7), "$5M+"
)]

group_dt <- d_agg[, .(pct_firms = sum(number) / number_total[1] * 100),
                  by = .(sector_main, year, group)]
group_dt[, group := factor(group, levels = c("< $25K", "$25K to $100K",
                                              "$100K to $500K", "$500K to $1M",
                                              "$1M to $5M", "$5M+"))]

fill_vals <- c(
  "< $25K"         = "#2166ac",
  "$25K to $100K"  = "#74add1",
  "$100K to $500K" = "#4dac26",
  "$500K to $1M"   = "#fee090",
  "$1M to $5M"     = "#f46d43",
  "$5M+"           = "#d73027"
)

# ---- Plot 1: all 6 groups, faceted by sector --------------------------------
p_all <- ggplot(group_dt, aes(x = year, y = pct_firms, fill = group)) +
  geom_area(position = "stack", colour = "white", linewidth = 0.3) +
  scale_fill_manual(values = fill_vals, name = NULL, guide = guide_legend(reverse = TRUE)) +
  facet_wrap(~ sector_main, scales = "free_y") +
  labs(x = "Year", y = "% of total firms") +
  theme_common +
  theme(strip.text = element_text(size = 13))

ggsave("figures/empirical/bracket_flow_firms_bysector.pdf", p_all,
       width = 14, height = 10)

# ---- Plot 2: exclude < $25K, rescale to sum to 100, faceted by sector ------
d_ex <- d_agg[group != "< $25K"]
total_ex <- d_agg[, .(total_ex = sum(number[group != "< $25K"])),
                  by = .(sector_main, year)]
d_ex <- merge(d_ex, total_ex, by = c("sector_main", "year"))
group_ex <- d_ex[, .(pct_firms = sum(number) / total_ex[1] * 100),
                 by = .(sector_main, year, group)]
group_ex[, group := factor(group, levels = c("$5M+", "$1M to $5M", "$500K to $1M",
                                              "$100K to $500K", "$25K to $100K"))]

# compute stacked y-midpoints at the last year for % change labels, per sector
last_year  <- max(group_ex$year)
first_year <- min(group_ex$year)

last_stack <- group_ex[year == last_year][order(sector_main, -as.integer(group))]
last_stack[, y_top := cumsum(pct_firms), by = sector_main]
last_stack[, y_mid := y_top - pct_firms / 2]

pct_change_dt <- merge(
  group_ex[year == last_year,  .(sector_main, group, pct_last  = pct_firms)],
  group_ex[year == first_year, .(sector_main, group, pct_first = pct_firms)],
  by = c("sector_main", "group")
)
pct_change_dt[, pct_chg := pct_last - pct_first]
pct_change_dt[, label := ifelse(pct_chg >= 0,
                                paste0("+", round(pct_chg, 1), "%"),
                                paste0(round(pct_chg, 1), "%"))]
label_dt <- merge(last_stack[, .(sector_main, group, y_mid)],
                  pct_change_dt[, .(sector_main, group, label)],
                  by = c("sector_main", "group"))
label_dt[, x := last_year + 0.5]

p_ex <- ggplot(group_ex, aes(x = year, y = pct_firms, fill = group)) +
  geom_area(position = "stack", colour = "white", linewidth = 0.3) +
  scale_fill_manual(values = fill_vals, name = NULL) +
  geom_text(data = label_dt, aes(x = x, y = y_mid, label = label),
            inherit.aes = FALSE, hjust = 0, size = 3, family = "serif") +
  coord_cartesian(clip = "off") +
  facet_wrap(~ sector_main, scales = "free_y") +
  labs(x = "Year", y = "% of firms (excl. < $25K)") +
  guides(fill = guide_legend(nrow = 2, reverse = TRUE)) +
  theme_common +
  theme(strip.text = element_text(size = 13),
        plot.margin = margin(5.5, 60, 5.5, 5.5))

ggsave("figures/empirical/bracket_flow_firms_bysector_ex_bottom.pdf", p_ex,
       width = 14, height = 10)
