# pulling IRS data, round 2

# Setup
library(httr)

# Create download directory if it doesn't exist
download_dir <- "/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings/data/raw/irs_data_downloaded/irs_post94_excel/"
if (!dir.exists(download_dir)) {
  dir.create(download_dir)
}

# ==============================================================================
# Table 22: 1994-2013
# Returns of Active Corporations other than Forms 1120S, 1120-REIT, and 1120-RIC
# ==============================================================================

table22_urls <- list(
  "2013" = "https://www.irs.gov/pub/irs-soi/13co22ccr.xls",
  "2012" = "https://www.irs.gov/pub/irs-soi/12co22ccr.xls",
  "2011" = "https://www.irs.gov/pub/irs-soi/11co22ccr.xls",
  "2010" = "https://www.irs.gov/pub/irs-soi/10co22ccr.xls",
  "2009" = "https://www.irs.gov/pub/irs-soi/09co22ccr.xls",
  "2008" = "https://www.irs.gov/pub/irs-soi/08co22ccr.xls",
  "2007" = "https://www.irs.gov/pub/irs-soi/07co22ccr.xls",
  "2006" = "https://www.irs.gov/pub/irs-soi/06co22ccr.xls",
  "2005" = "https://www.irs.gov/pub/irs-soi/05co22ccr.xls",
  "2004" = "https://www.irs.gov/pub/irs-soi/04co22ccr.xls",
  "2003" = "https://www.irs.gov/pub/irs-soi/03co22nr.xls",
  "2002" = "https://www.irs.gov/pub/irs-soi/02co22nr.xls",
  "2001" = "https://www.irs.gov/pub/irs-soi/01co22nr.xls",
  "2000" = "https://www.irs.gov/pub/irs-soi/00co22nr.xls",
  "1999" = "https://www.irs.gov/pub/irs-soi/99co22nr.xls",
  "1998" = "https://www.irs.gov/pub/irs-soi/98co22nr.xls",
  "1997" = "https://www.irs.gov/pub/irs-soi/97co22tabl.xls",
  "1996" = "https://www.irs.gov/pub/irs-soi/96crtb22.xls",
  "1995" = "https://www.irs.gov/pub/irs-soi/95crtb22.xls",
  "1994" = "https://www.irs.gov/pub/irs-soi/94co22ac.xls"
)

# Download Table 22 files (1994-2013)
cat("Downloading Table 22 files (1994-2013)...\n")
for (year in names(table22_urls)) {
  url <- table22_urls[[year]]
  dest_file <- file.path(download_dir, paste0(year, "_Table22.xls"))

  if (!file.exists(dest_file)) {
    cat(sprintf("  Downloading %s...\n", year))
    tryCatch({
      GET(url, write_disk(dest_file, overwrite = TRUE))
      cat(sprintf("    Success: %s\n", dest_file))
    }, error = function(e) {
      cat(sprintf("    ERROR downloading %s: %s\n", year, e$message))
    })
    Sys.sleep(0.5)  # Be polite to the server
  } else {
    cat(sprintf("  Skipping %s (already exists)\n", year))
  }
}

# ==============================================================================
# Table 5.3: 2014-2022
# Returns of Active Corporations other than Forms 1120S, 1120-REIT, and 1120-RIC
# ==============================================================================

table53_urls <- list(
  "2022" = "https://www.irs.gov/pub/irs-soi/22co53ccr.xlsx",
  "2021" = "https://www.irs.gov/pub/irs-soi/21co53ccr.xlsx",
  "2020" = "https://www.irs.gov/pub/irs-soi/20co53ccr.xlsx",
  "2019" = "https://www.irs.gov/pub/irs-soi/19co53ccr.xlsx",
  "2018" = "https://www.irs.gov/pub/irs-soi/18co53ccr.xlsx",
  "2017" = "https://www.irs.gov/pub/irs-soi/17co53ccr.xlsx",
  "2016" = "https://www.irs.gov/pub/irs-soi/16co53ccr.xlsx",
  "2015" = "https://www.irs.gov/pub/irs-soi/15co53ccr.xlsx",
  "2014" = "https://www.irs.gov/pub/irs-soi/14co53ccr.xlsx"
)

# Download Table 5.3 files (2014-2022)
cat("\nDownloading Table 5.3 files (2014-2022)...\n")
for (year in names(table53_urls)) {
  url <- table53_urls[[year]]
  dest_file <- file.path(download_dir, paste0(year, "_Table53.xlsx"))

  if (!file.exists(dest_file)) {
    cat(sprintf("  Downloading %s...\n", year))
    tryCatch({
      GET(url, write_disk(dest_file, overwrite = TRUE))
      cat(sprintf("    Success: %s\n", dest_file))
    }, error = function(e) {
      cat(sprintf("    ERROR downloading %s: %s\n", year, e$message))
    })
    Sys.sleep(0.5)
  } else {
    cat(sprintf("  Skipping %s (already exists)\n", year))
  }
}

# ==============================================================================
# Table 5.4: 2014-2022
# Returns with Net Income (Active Corps other than 1120S, 1120-REIT, 1120-RIC)
# ==============================================================================

table54_urls <- list(
  "2022" = "https://www.irs.gov/pub/irs-soi/22co54ccr.xlsx",
  "2021" = "https://www.irs.gov/pub/irs-soi/21co54ccr.xlsx",
  "2020" = "https://www.irs.gov/pub/irs-soi/20co54ccr.xlsx",
  "2019" = "https://www.irs.gov/pub/irs-soi/19co54ccr.xlsx",
  "2018" = "https://www.irs.gov/pub/irs-soi/18co54ccr.xlsx",
  "2017" = "https://www.irs.gov/pub/irs-soi/17co54ccr.xlsx",
  "2016" = "https://www.irs.gov/pub/irs-soi/16co54ccr.xlsx",
  "2015" = "https://www.irs.gov/pub/irs-soi/15co54ccr.xlsx",
  "2014" = "https://www.irs.gov/pub/irs-soi/14co54ccr.xlsx"
)

# Download Table 5.4 files (2014-2022)
cat("\nDownloading Table 5.4 files (2014-2022)...\n")
for (year in names(table54_urls)) {
  url <- table54_urls[[year]]
  dest_file <- file.path(download_dir, paste0(year, "_Table54.xlsx"))

  if (!file.exists(dest_file)) {
    cat(sprintf("  Downloading %s...\n", year))
    tryCatch({
      GET(url, write_disk(dest_file, overwrite = TRUE))
      cat(sprintf("    Success: %s\n", dest_file))
    }, error = function(e) {
      cat(sprintf("    ERROR downloading %s: %s\n", year, e$message))
    })
    Sys.sleep(0.5)
  } else {
    cat(sprintf("  Skipping %s (already exists)\n", year))
  }
}

cat("\nDownloads complete!\n")
