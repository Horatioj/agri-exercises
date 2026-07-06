*--------------------------------------------------------------------
* STEP 0 (Stata) — assemble the base county production panel, merged on
* countyid + year:  nominal GVP <- agr_GVP.dta, labour <- agr_labor.dta,
* land/capital/intermediate <- cty_prod_account_agg_forAXN.dta.
* Saves data/base_panel.dta for the cleaning/correction step.
*--------------------------------------------------------------------
clear all
set more off

* --- GVP (master) ---
use "E:/TFP_weather/data/agr_GVP.dta", clear
duplicates drop countyid year, force
tempfile gvp
save `gvp'

* --- labour COUNT (agr_labor1 is number of workers, NOT labour-days) ---
use "E:/TFP_weather/data/agr_labor.dta", clear
duplicates drop countyid year, force
rename agr_labor1 agr_labor_num
keep countyid year agr_labor_num
tempfile lab
save `lab'

* --- labour-DAYS + land / capital / intermediate (from the agg account) ---
use "E:/TFP_weather/data/cty_prod_account_agg_forAXN.dta", clear
duplicates drop countyid year, force
keep countyid year SID state county_name Laborday_impute Land_serv_q capital_serv_q Inter_all_real Inter_all_nom
tempfile agg
save `agg'

* --- merge on countyid + year ---
use `gvp', clear
merge 1:1 countyid year using `lab', nogen
merge 1:1 countyid year using `agg', nogen update
order countyid year SID state county_name GVP_allagr_impute Laborday_impute ///
      agr_labor_num Land_serv_q capital_serv_q Inter_all_real Inter_all_nom
sort countyid year
save "E:/TFP_weather/data/base_panel.dta", replace

count
di "base_panel.dta saved: " _N " rows"
codebook countyid year, compact
display "DONE_BASE_PANEL"
