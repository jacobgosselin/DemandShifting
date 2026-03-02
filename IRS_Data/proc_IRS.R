# processing IRS data

library(reticulate)

# Use xlrd for reading old .xls files
xlrd <- import("xlrd")

# Directory with downloaded files
data_dir <- "/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings/code/IRS_Data/irs_data_downloaded"

# ==============================================================================
# Process Table 22 files (1994-2013)
# ==============================================================================

years_table22 <- 1994:2013

# Initialize results dataframe
results <- data.frame(
  year = integer(),
  total_returns = numeric(),
  returns_with_net_income = numeric(),
  stringsAsFactors = FALSE
)

for (year in years_table22) {
  file_path <- file.path(data_dir, paste0(year, "_Table22.xls"))

  if (!file.exists(file_path)) {
    cat(sprintf("File not found: %s\n", file_path))
    next
  }

  # Read using xlrd
  workbook <- xlrd$open_workbook(file_path)
  sheet <- workbook$sheet_by_index(0L)

  # Search for "Returns with net income" in column A
  net_income_row <- NULL
  for (row_idx in 0:(sheet$nrows - 1)) {
    cell_value <- tryCatch({
      as.character(sheet$cell_value(as.integer(row_idx), 0L))
    }, error = function(e) "")

    if (grepl("Returns with net income", cell_value, ignore.case = TRUE)) {
      net_income_row <- row_idx
      break
    }
  }

  if (is.null(net_income_row)) {
    cat(sprintf("Could not find 'Returns with net income' row in %d\n", year))
    next
  }

  # The "Total" row should be directly above
  total_row <- net_income_row - 1L

  # Extract Column B (index 1) values
  total_returns <- as.numeric(sheet$cell_value(as.integer(total_row), 1L))
  returns_with_net_income <- as.numeric(sheet$cell_value(as.integer(net_income_row), 1L))

  cat(sprintf("%d: total_returns = %s, returns_with_net_income = %s\n",
              year, format(total_returns, big.mark = ","),
              format(returns_with_net_income, big.mark = ",")))

  # Add to results
  results <- rbind(results, data.frame(
    year = year,
    total_returns = total_returns,
    returns_with_net_income = returns_with_net_income
  ))
}

# Print results
cat("\n=== Results for 1994-2013 ===\n")
print(results)
