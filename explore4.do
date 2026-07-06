capture log close
log using "E:\TFP_weather\output4.txt", replace text

use "E:\TFP_weather\data\cty_prod_account_agg_forAXN.dta", clear

codebook countyid, compact
codebook year, compact
tab SID, sort

count if !missing(GVP_allagr_impute) & !missing(Laborday_impute) & !missing(Land_serv_q) & !missing(capital_serv_q) & !missing(Inter_all_real)
count

tab year if !missing(GVP_allagr_impute) & !missing(Land_serv_q) & !missing(capital_serv_q) & !missing(Inter_all_real)

log close
