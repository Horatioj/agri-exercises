capture log close
log using "E:\TFP_weather\data_description.log", replace text

use "E:\TFP_weather\data\cty_prod_account_agg_forAXN.dta", clear
describe
summarize
tab year

use "E:\TFP_weather\data\cty_prod_account_q&p_forAXN.dta", clear
describe
summarize

use "E:\TFP_weather\data\county_irri_elect_fert_forXZ.dta", clear
describe
summarize

use "E:\TFP_weather\data\mac_power.dta", clear
describe
summarize

use "E:\TFP_weather\data\priceindex_province.dta", clear
describe
summarize

log close
