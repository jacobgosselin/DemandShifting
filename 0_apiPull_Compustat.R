setwd("/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings")
# load libraries
library(RPostgres)
library(data.table)
library(tidyr)
library(dplyr)

wrds <- dbConnect(Postgres(),
                  host='wrds-pgdata.wharton.upenn.edu',
                  port=9737,
                  dbname='wrds',
                  sslmode='require',
                  user='jacobgosselin')

# request Compustat variables
res <- dbSendQuery(wrds, "select funda.conm, company.fic, company.sic, company.ipodate, funda.gvkey, funda.cusip, funda.datadate, funda.fyear,
company.naics, funda.ebitda, funda.aqc, funda.at, funda.che, funda.dlc, funda.dltt, funda.dvp, funda.lct, funda.lt, funda.oiadp,
funda.ppegt, funda.ppent, funda.sale, funda.xint, funda.mkvalt, funda.ni,  funda.dp, funda.invfg, funda.cogs, funda.dd1, funda.xopr,
funda.xsga, funda.np, funda.intan, funda.capx, funda.xrd, funda.rdip, funda.fyear, funda.xstfws, funda.emp, funda.DVPSP_F, funda.MKVALT, 
funda.PRCC_F, funda.INVRM, company.state, funda.indfmt, funda.revt, funda.csho, funda.pi, funda.oiadp, funda.xint, funda.spi, funda.nopi, 
funda.oibdp, funda.revt, funda.xopr, funda.txpd, funda.dlc, funda.act,
ccm.lpermno as permno
                    FROM comp.funda
                    JOIN comp.company
                    ON funda.gvkey = company.gvkey
                    LEFT JOIN crsp.ccmxpf_linktable as ccm -- adds CRSP permno; LEFT JOIN preserves firms with no CCM match
                    ON funda.gvkey = ccm.gvkey
                    AND ccm.linktype IN ('LU', 'LC') -- valid link types only
                    AND ccm.linkprim IN ('P', 'C') -- primary permno for the gvkey only
                    AND (ccm.linkdt IS NULL OR funda.datadate >= ccm.linkdt) -- permno active at datadate
                    AND (ccm.linkenddt IS NULL OR funda.datadate <= ccm.linkenddt)
                    WHERE funda.indfmt = 'INDL' AND funda.consol = 'C' AND funda.popsrc = 'D'
                    AND funda.datafmt = 'STD' AND funda.curcd = 'USD'")

compustat <- data.table(dbFetch(res, n=-1))

# get Peters and Taylor TOTALQ
res <- dbSendQuery(wrds, "select *
                    from totalq_all.total_q")
totalq <- data.table(dbFetch(res, n=-1))
compustat <- merge(compustat, totalq, by = c("gvkey", "fyear", "datadate"), all.x = TRUE)


# duplicate gvkey-year combinations exist; merge (if obs are same use both, if missing obs use existing one)
compustat <- compustat %>%
  distinct(gvkey, fyear, .keep_all = TRUE)

save(compustat, file="data/raw/compustat_raw.RData")

