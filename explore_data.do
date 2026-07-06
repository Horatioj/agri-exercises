capture log close
log using "E:\TFP_weather\explore_data2.log", replace text

use "E:\TFP_weather\data\cty_prod_account_agg_forAXN.dta", clear

* Check county types by merging with climate panel info
* First check distinct counties and years
distinct countyid
distinct year
tab SID, sort

* Check how many counties per province
bysort SID: egen n_cty = tag(countyid)
tab SID if n_cty==1

* Simple TFP proxy: output per unit of total input
gen ln_GVP = ln(GVP_allagr_impute)
gen ln_Labor = ln(Laborday_impute)
gen ln_Land = ln(Land_serv_q)
gen ln_Capital = ln(capital_serv_q)
gen ln_Inter = ln(Inter_all_real)

* Check coverage
count if !missing(GVP_allagr_impute) & !missing(Laborday_impute) & !missing(Land_serv_q) & !missing(capital_serv_q) & !missing(Inter_all_real)
count

* Year range with reasonable coverage
tab year if !missing(GVP_allagr_impute) & !missing(Land_serv_q) & !missing(capital_serv_q)

log close
