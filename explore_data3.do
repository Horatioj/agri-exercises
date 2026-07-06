capture log close
log using "E:\TFP_weather\explore_data3.log", replace text

use "E:\TFP_weather\data\cty_prod_account_agg_forAXN.dta", clear

* Count distinct counties
codebook countyid, compact
codebook year, compact
tab SID, sort

* Check coverage of all key variables
count if !missing(GVP_allagr_impute) & !missing(Laborday_impute) & !missing(Land_serv_q) & !missing(capital_serv_q) & !missing(Inter_all_real)
count

* Year range with good coverage
tab year if !missing(GVP_allagr_impute) & !missing(Land_serv_q) & !missing(capital_serv_q) & !missing(Inter_all_real)

log close
