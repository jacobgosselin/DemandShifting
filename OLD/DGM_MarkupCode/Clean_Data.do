*** COMPUSTAT Cleaning
* Maarten de Ridder & Basile Grassi
* Modified by Giovanni Morzenti

* This version: 27/02/2023


** This Do File is prepared to clean COMPUSTAT firm level data


	/*
	The file is structured as follows:
		1. Preamble: set working directory, identify correct datasets 
		2. Select Variables: choose what variables should be used in the analysis.
		3. Select Method: select some parameters for cleaning
		4. Open data and perform cleaning + export. 
		
	The file produces the following output:
		- COMPUSTAT_Clean.csv	: cleaned COMPUSTAT dataset 
		
	*/ 

	
/*
Use stata file downloaded from Compustat using the following protocol:
Access compustat (WRDS)
> Compustat - Capital IQ
> NORTH AMERICA
> FUNDAMENTALS ANNUAL
> DATE RANGE 1955 - 2019
> GVKEY CODE - search the entire database
> Consolidation Level - C
> Industry Format - INDL, FS
> Data Format - STD
> Population Source - D
> Currency - USD
> Company Status - A,I
>> VARIABLES  TO BE SELECTED:
- GVKEY
- NAICS
- SIC
- STATE
- FYEAR
- FQUARTER
- INVRM - Inventories Raw Materials
- PPEGT - Property, Plant and Equipment - Total (Gross)
- PPENT - Property, Plant and Equipment - Total (Net)
- COGS - Cost of Goods Sold
- SALE - Sales/Turnover (Net)
- XSGA - Selling, General and Administrative Expense
- XSTFWS - Staff Expense - Wages and Salaries
- CAPX - Capital Expenditures
- EMP - Employees
- DVPSP_F - Dividends per Share - Pay Date - Fiscal
- MKVALT - Market Value - Total - Fiscal
- PRCC_F - Price Close - Annual - Fiscal

>>Actual Variables in final database:
gvkey     indfmt    datafmt   capx      invrm     sale      costat    prcc_f    state
datadate  consol    tic       cogs      ppegt     xsga      dvpsp_f   naics
fyear     popsrc    curcd     emp       ppent     xstfws    mkvalt    sic	
	
	* external datasets:
		1. DEFL_Y.dta (taken from FRED)
		
*/	

	
************************************************************************************************************************************************************************************
************************************************************************************************************************************************************************************
	
	
***** 1. Preamble ******

* clean up 
clear all 
set max_memory 10g
set matsize 11000

*set working directory. Must contain:
	// 1) single dataset with all firm data 
	// 2) dataset with deflators 
	local path "/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings/To_Share_final"

*set dataset names (actual files need to be .dta) 
	// main firm-level dataset
	local dataset compustat_raw.dta
	// dataset with deflators 
	local deflator DEFL_Y.dta
	
	
************************************************************************************************************************************************************************************
************************************************************************************************************************************************************************************	
	
	
	
***** 2: Select Variables *****

*variable for sales (revenue) 
	//this variable must be sales (p*q) in levels
	local sale sale 

*variable for book-value of capital 
	//this variable must contain fixed assets in levels ()
	local capital ppent 
	
*variable for gross investment 
	//gross investment = investment not corrected for replacement investment  (this is used to derive capital through the perpetual inventory method 
	local gross_investment capx 

*variable for fixed input
	//measure of Overhead Costs, booked in Compustat under ԓelling, General and Administrative Expensesԍ
	local fixcost xsga 

*variable for variable input
	//this must be an input that the firm sets (in levels) when observing contemporaneous productivity shocks. DLW: labor DLE: cost of good sold
	//note: the variable must denote expenditure on the variable input
	local varcost cogs //acha1 acha2 ... up to 5, sal if wagebilll 

*variable with quantity of variable input 
	//if estimating the markups and you have seperate variables for expenditure for
	//example: total wagebill and labor, labor would be entered here, wagebill would be the variable input in local 'varcost' 
	local varquant emp 
	
*variable with industry-classification  
	//industry classification must have stucture of Naics code: nth digit is subset of industries from (n-1)th digit 
	local indvar naics //use naf2003_single if you only want 1 code per firm

*variable for firm identifiers
	local firm gvkey

*variable for year
	//check whether time-of-year reporting is comparable across firms. If adjustment for fiscal year end dates needs to be made, additional code must be added in the file 
	local year fyear 
  
*variable investment good deflator
	//variable should contain the investment good deflator in the .dta file with deflators (only used if estimating capital with PIM) 
	//note: this should be the actual deflator, not the relative price of investment goods 
	local deflate_var_i defl_inv
	local capdeflate_var defl_cap 

*variables to deflate inputs in production function\
	//variable used to deflate variable input (wagebill) 
	local deflate_var_v defl_gdp // sectordeflator_GO_P
	//variable used to deflate output 
	local deflate_var_s defl_gdp // sectordeflator_GO_P  //CAUTION: choose value added or gross output deflator based on variable put in local 'sale'  
	//variable used to deflate fised inputs
	local deflate_var_m defl_gdp //  sectordeflator_II_P
	
	
************************************************************************************************************************************************************************************
************************************************************************************************************************************************************************************



***** 3: Select Method *****
	
*select industry level 
	//this is the digit level at which the GMM estimation is executed (usually 2-digit) 
	local ind_lev 3  //choose 1, 2, 3, .. 
	
*choose to trim yes/no
	//trim 2% tails of cost-revenue ratio (as it is done in DEU 2020) 
	local trim 1 //choose 1,0 
		
*choose to deflate yes/no 
	//will lead all variables to be deflated by GDP deflator 
	//if capital needs to be deflated by differently, either 1) estimate with PIM (see below) or 2) manually change code in Section 5 
	local deflate 1 //choose 1,0 
 
*choose years to be considered for analysis 
	local min_year 1955
	local max_year 2019
 	
*interpolate neighboring values
	//interpolate missing values from direct neighbours if neighbour on both side exists and original value is missing 
	//applied to capital, net capital, variable input and sales
	local interpolate 1 //choose 1,0 

*drop estimated values
	//drops BNC-based firm-year obs if identifier variables are estimated assuming that BIC firms in same sector with same sales bin have same balance sheets\
	local dropestimates 0 //choose 1,0

*choose minimum observatios
	//choose the minimum number of firm-year observations in each industry. Too few will cause script to stop running. 
	local min_obs 100 //141; for 13 years that means 11 firms/yaers, which is export minimum 
 
*choose depreciation rate capital 
	//only used when approximating capital with the PIM
	local delta 0.095
		
************************************************************************************************************************************************************************************
************************************************************************************************************************************************************************************ 
 
***** 4: Open Dataset, Generate Variables and Perform Cleaning  *****


** 4.1 Open dataset and perform cleaning 

* open dataset 
cd "`path'"
use `dataset', clear 

* initial cleaning 
drop if `year' > `max_year' 
drop if `year' < `min_year' 
duplicates drop `year' `firm' , force  //firms that switch fiscal year 
destring `firm' `indvar', replace 
rename `year' year
set more off 

* Generate high-level industries
tostring `indvar', replace
gen `indvar'_length =strlen(`indvar')
gen `indvar'`ind_lev' = substr(`indvar',1,`ind_lev') if `indvar'_length >= `ind_lev' //generate 2-digit code (if industry level is 2), require initial code to have > 1 digit 
destring `indvar'*, replace 

* deflate dataset (optional) 
if "`deflate'" == "1" { 
	merge m:1 year using `deflator'
	replace `varcost' = (`varcost')   / `deflate_var_v' *100 
	replace `sale' 	  = (`sale') 	  / `deflate_var_s' *100
	replace `fixcost' = (`fixcost') 	  / `deflate_var_m' *100
	drop if `firm' == . 
}
xtset `firm' year
sort `firm' year

* drop estimated values (optional)
if "`dropestimates'" == "1" {
	drop if orbil == "E" | orcr == "E"
}

*interpolate missing values 
if "`interpolate'" == "1" {

	local intlist `varcost' `sale' `fixcost' 
	foreach var in `intlist'  { 
		replace `var' = (f.`var'+l.`var')/2 if `var' == . & f.`var' != .  & l.`var'  != . & f.`firm' == l.`firm' // check same firm for consistency
	} 
}	


//only keep industries with more than min_obs observations (leaving too few obs per industry causes script to stop running) 
bysort `indvar'`ind_lev': gen count = _N
drop if count < `min_obs' 
xtset `firm' year 
egen ind = group(`indvar'`ind_lev')

* Keep only observation for one industry (necessary depending on downloaded file)
bysort `firm' year : gen nrobs = _N
drop if (nrobs == 2 | nrobs == 3) & indfmt == "FS"
sort `firm' year
drop if `firm'==`firm'[_n-1] & year==year[_n-1]

* trim based on several ratios
if "`trim'" == "1" {
	
	* trim on sales to cost ratio
	gen s_g = `sale'/`varcost'
	keep if s_g>0
	bysort year: egen s_g_p_1  = pctile(s_g), p(1)
	bysort year: egen s_g_p_99  = pctile(s_g), p(99)
	keep if s_g> s_g_p_1 & s_g< s_g_p_99
	drop s_g*

	/* trim on costshare ratios
	gen costshare1 = `varcost'/(`varcost'+`capital')
	gen costshare2 = `varcost'/(`varcost'+`capital'+`fixcost')
	keep if costshare1>0 & costshare2>0
	forvalues s=1(1)2 {
		bysort year: egen costshare`s'_p_1=pctile(costshare`s'), p(1)
		bysort year: egen costshare`s'_p_99=pctile(costshare`s'), p(99)
		drop if costshare`s'==0 | costshare`s'==.
		keep if costshare`s'> costshare`s'_p_1 & costshare`s'< costshare`s'_p_99
	}
	drop costshare*
	*/
}

gen v = log(`varcost')
gen k = log(`capital')
gen x = log(`fixcost')
gen y = log(`sale')

* rename variables for the python script
ren `firm' firmid
ren year date
ren `indvar'`ind_lev' sector
ren `varcost' varcost
ren `capital' capital
ren `fixcost' fixcost

sort sector firmid date 

** 4.2 Export 

* export CSV file
export delim firmid date sector sale varcost capital fixcost v k x y  ///
		using "COMPUSTAT_Clean.csv" ///
		, replace
