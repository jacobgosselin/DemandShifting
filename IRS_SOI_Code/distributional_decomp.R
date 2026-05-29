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

d <- as.data.table(read_dta("data/raw/agg_brackets_receipts_R5.dta"))
d <- d[year >= 1980]
d[, core_bracket := CORE_BRACKETS[findInterval(thres_low, CORE_BRACKETS)]]

d_agg <- d[, .(number = sum(number), number_total = number_total[1]),
             by = .(year, core_bracket)]

# 6-group collapse used for stacked area
d_agg[, group := fcase(
  core_bracket %in% c(0),              "< $25K",
  core_bracket %in% c(25e3),           "$25K to $100K",
  core_bracket %in% c(1e5),            "$100K to $500K",
  core_bracket %in% c(5e5),            "$500K to $1M",
  core_bracket %in% c(1e6),            "$1M to $5M",
  core_bracket %in% c(5e6, 1e7, 5e7), "$5M+"
)]

group_dt <- d_agg[, .(pct_firms = sum(number) / number_total[1] * 100),
                  by = .(year, group)]
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

# ---- Plot 1: all 6 groups --------------------------------------------------
p_all <- ggplot(group_dt, aes(x = year, y = pct_firms, fill = group)) +
  geom_area(position = "stack", colour = "white", linewidth = 0.5) +
  scale_fill_manual(values = fill_vals, name = NULL, guide = guide_legend(reverse = TRUE)) +
  labs(x = "", y = "% of total firms") +
  theme_common

ggsave("figures/empirical/bracket_flow_firms.pdf", p_all,
       width = 10, height = 10)

# ---- Plot 2: exclude < $25K, rescale to sum to 100 ------------------------
group_ex <- group_dt[group != "< $25K"]

# recompute raw firm counts to rescale among the 5 remaining groups
d_ex <- d_agg[group != "< $25K"]
total_ex <- d_agg[, .(total_ex = sum(number[group != "< $25K"])), by = year]
d_ex <- merge(d_ex, total_ex, by = "year")
group_ex2 <- d_ex[, .(pct_firms = sum(number) / total_ex[1] * 100),
                  by = .(year, group)]
group_ex2[, group := factor(group, levels = c("$5M+", "$1M to $5M", "$500K to $1M",
                                               "$100K to $500K", "$25K to $100K"))]

# compute stacked y-midpoints at the last year for % change labels
# ggplot stacks first factor level at the bottom, so reverse factor order before cumsum
group_ex2_ordered <- group_ex2[order(year, group)]
last_year <- max(group_ex2_ordered$year)
first_year <- min(group_ex2_ordered$year)

last_stack <- group_ex2_ordered[year == last_year][order(-as.integer(group))]
last_stack[, y_top := cumsum(pct_firms)]
last_stack[, y_mid := y_top - pct_firms / 2]

pct_change_dt <- merge(
  group_ex2_ordered[year == last_year, .(group, pct_last = pct_firms)],
  group_ex2_ordered[year == first_year, .(group, pct_first = pct_firms)],
  by = "group"
)
pct_change_dt[, pct_chg := (pct_last - pct_first)] #/ pct_first * 100]
pct_change_dt[, label := ifelse(pct_chg >= 0,
                                paste0("+", round(pct_chg, 1), "%"),
                                paste0(round(pct_chg, 1), "%"))]
label_dt <- merge(last_stack[, .(group, y_mid)], pct_change_dt[, .(group, label)], by = "group")
label_dt[, x := last_year + 0.5]

p_ex <- ggplot(group_ex2, aes(x = year, y = pct_firms, fill = group)) +
  geom_area(position = "stack", colour = "white", linewidth = 0.5) +
  scale_fill_manual(values = fill_vals, name = NULL) +
  geom_text(data = label_dt, aes(x = x, y = y_mid, label = label),
            inherit.aes = FALSE, hjust = 0, size = 5, family = "serif") +
  coord_cartesian(clip = "off") +
  labs(x = "", y = "% of Filings") +
  guides(fill = guide_legend(nrow = 2, reverse = TRUE)) +
  theme_common +
  theme(plot.margin = margin(5.5, 80, 5.5, 5.5))

ggsave("figures/empirical/bracket_flow_firms_ex_bottom.pdf", p_ex,
       width = 10, height = 10)

# ---- Plot 3: % of receipts (excl. < $25K), all remaining groups -------------
d_rec <- d[core_bracket != 0,
           .(breceipts = sum(breceipts), breceipts_total = breceipts_total[1]),
           by = .(year, core_bracket)]

d_rec[, group := fcase(
  core_bracket %in% c(25e3),           "$25K to $100K",
  core_bracket %in% c(1e5),            "$100K to $500K",
  core_bracket %in% c(5e5),            "$500K to $1M",
  core_bracket %in% c(1e6),            "$1M to $5M",
  core_bracket %in% c(5e6, 1e7, 5e7), "$5M+"
)]

# recompute total receipts excluding < $25K bracket
total_rec_ex <- d[core_bracket != 0,
                  .(breceipts_total_ex = sum(breceipts)),
                  by = year]
d_rec <- merge(d_rec, total_rec_ex, by = "year")

rec_dt <- d_rec[, .(pct_receipts = sum(breceipts) / breceipts_total_ex[1] * 100),
                by = .(year, group)]
rec_dt[, group := factor(group, levels = c("$5M+", "$1M to $5M", "$500K to $1M",
                                            "$100K to $500K", "$25K to $100K"))]

# % change labels for receipts plots (first factor level stacks at bottom)
rec_last_year  <- max(rec_dt$year)
rec_first_year <- min(rec_dt$year)

rec_last_stack <- rec_dt[year == rec_last_year][order(-as.integer(group))]
rec_last_stack[, y_top := cumsum(pct_receipts)]
rec_last_stack[, y_mid := y_top - pct_receipts / 2]

rec_pct_chg <- merge(
  rec_dt[year == rec_last_year,  .(group, pct_last  = pct_receipts)],
  rec_dt[year == rec_first_year, .(group, pct_first = pct_receipts)],
  by = "group"
)
rec_pct_chg[, pct_chg := (pct_last - pct_first) / pct_first * 100]
rec_pct_chg[, label := ifelse(pct_chg >= 0,
                               paste0("+", round(pct_chg, 1), "%"),
                               paste0(round(pct_chg, 1), "%"))]
rec_label_dt <- merge(rec_last_stack[, .(group, y_mid)],
                      rec_pct_chg[, .(group, label)], by = "group")
rec_label_dt[, x := rec_last_year + 0.5]

p_rec_full <- ggplot(rec_dt, aes(x = year, y = pct_receipts, fill = group)) +
  geom_area(position = "stack", colour = "white", linewidth = 0.5) +
  scale_fill_manual(values = fill_vals, name = NULL) +
  geom_text(data = rec_label_dt, aes(x = x, y = y_mid, label = label),
            inherit.aes = FALSE, hjust = 0, size = 5, family = "serif") +
  coord_cartesian(clip = "off") +
  labs(x = "", y = "% of Receipts", title = "") +
  guides(fill = guide_legend(nrow = 2, reverse = TRUE)) +
  theme_common +
  theme(plot.margin = margin(5.5, 80, 5.5, 5.5))

ggsave("figures/empirical/bracket_flow_receipts_ex_bottom.pdf", p_rec_full,
       width = 10, height = 10)

# ---- Plot 3a: same as Plot 3 but exclude $5M+ (shares still out of total excl. <$25K) ----
rec_dt_ex5m <- rec_dt[group != "$5M+"]

rec3a_last_stack <- rec_dt_ex5m[year == rec_last_year][order(-as.integer(group))]
rec3a_last_stack[, y_top := cumsum(pct_receipts)]
rec3a_last_stack[, y_mid := y_top - pct_receipts / 2]

rec3a_pct_chg <- merge(
  rec_dt_ex5m[year == rec_last_year,  .(group, pct_last  = pct_receipts)],
  rec_dt_ex5m[year == rec_first_year, .(group, pct_first = pct_receipts)],
  by = "group"
)
rec3a_pct_chg[, pct_chg := (pct_last - pct_first) / pct_first * 100]
rec3a_pct_chg[, label := ifelse(pct_chg >= 0,
                                paste0("+", round(pct_chg, 1), "%"),
                                paste0(round(pct_chg, 1), "%"))]
rec3a_label_dt <- merge(rec3a_last_stack[, .(group, y_mid)],
                        rec3a_pct_chg[, .(group, label)], by = "group")
rec3a_label_dt[, x := rec_last_year + 0.5]

p_rec_ex5m <- ggplot(rec_dt_ex5m, aes(x = year, y = pct_receipts, fill = group)) +
  geom_area(position = "stack", colour = "white", linewidth = 0.5) +
  scale_fill_manual(values = fill_vals, name = NULL) +
  geom_text(data = rec3a_label_dt, aes(x = x, y = y_mid, label = label),
            inherit.aes = FALSE, hjust = 0, size = 5, family = "serif") +
  coord_cartesian(clip = "off") +
  labs(x = "", y = "% of Receipts", title = "") +
  guides(fill = guide_legend(nrow = 2, reverse = TRUE)) +
  theme_common +
  theme(plot.margin = margin(5.5, 80, 5.5, 5.5))

ggsave("figures/empirical/bracket_flow_receipts_ex_bottom_ex5m.pdf", p_rec_ex5m,
       width = 10, height = 10)

# ---- Plot 4: % change in aggregate receipts, plot-2 brackets ----------------
d_rec2 <- d[core_bracket != 0,
            .(breceipts = sum(breceipts)),
            by = .(year, core_bracket)]

d_rec2[, group := fcase(
  core_bracket %in% c(25e3),           "$25K to $100K",
  core_bracket %in% c(1e5),            "$100K to $500K",
  core_bracket %in% c(5e5),            "$500K to $1M",
  core_bracket %in% c(1e6),            "$1M to $5M",
  core_bracket %in% c(5e6, 1e7, 5e7), "$5M+"
)]

rec2_dt <- d_rec2[, .(breceipts = sum(breceipts)), by = .(year, group)]
rec2_dt[, group := factor(group, levels = c("$25K to $100K", "$100K to $500K",
                                             "$500K to $1M", "$1M to $5M", "$5M+"))]

total_rec2 <- rec2_dt[, .(total = sum(breceipts)), by = year]
rec2_dt    <- merge(rec2_dt, total_rec2, by = "year")
rec2_dt[, share := breceipts / total * 100]

base_rec <- rec2_dt[year == min(year), .(group, base_share = share)]
rec2_dt  <- merge(rec2_dt, base_rec, by = "group")
rec2_dt[, pct_chg := (share/base_share - 1) * 100]

fill_vals2 <- fill_vals[c("$25K to $100K", "$100K to $500K",
                           "$500K to $1M", "$1M to $5M", "$5M+")]

p_rec_chg <- ggplot(rec2_dt, aes(x = year, y = pct_chg, colour = group)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  scale_colour_manual(values = fill_vals2, name = NULL,
                      guide = guide_legend(reverse = TRUE)) +
  labs(x = "", y = "% change in aggregate receipts") +
  guides(colour = guide_legend(nrow = 2, reverse = TRUE)) +
  theme_common +
  theme(plot.margin = margin(5.5, 20, 5.5, 5.5))

# combined plot
p_combined <- ggarrange(p_ex + labs(title = "Recomposition of Filings"), 
                        p_rec_ex5m + labs(title = "Recomposition of Receipts"),
                        ncol = 2, common.legend = TRUE, legend = "bottom")
ggsave("figures/empirical/receipt_recomposition.pdf", p_combined, width = 16, height = 9)
p_combined_slides <- ggarrange(p_ex + labs(title = "Recomposition of Filings"), 
                                p_rec_ex5m + labs(title = "Recomposition of Receipts"),
                                nrow = 2, common.legend = TRUE, legend = "bottom")
ggsave("figures/empirical/receipt_recomposition_slides.pdf", p_combined_slides, width = 10, height = 10)
