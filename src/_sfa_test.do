clear all
set more off
cap ssc install sfpanel, replace

import delimited "E:/TFP_weather/src/clean/county_panel_clean.csv", clear varnames(1) case(preserve)
destring countyid year SID, replace force

gen double ly    = ln(real_gvp)
gen double lL    = ln(Laborday_impute)
gen double lLand = ln(Land_serv_q)
gen double lK    = ln(capital_serv_q)
gen double lM    = ln(Inter_all_real)
keep if ly<. & lL<. & lLand<. & lK<. & lM<.

* small representative subset: first 150 counties
egen cid = group(countyid)
keep if cid <= 150
xtset countyid year
di "N obs = " _N
qui levelsof countyid, local(cc)
di "N counties = " `: word count `cc''

timer clear 1
timer on 1
sfpanel ly lL lLand lK lM i.year, model(tfe) distribution(hnormal)
timer off 1
timer list 1
di "converged = " e(converged)
