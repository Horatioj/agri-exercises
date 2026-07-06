*===============================================================
* 17_sfa_sub1000.do
* EXACT SFA.do spec on a 1000-county random subsample (Inner Mongolia 15,
* Tibet 54, Qinghai 63, Xinjiang 65 excluded; seed 42, built in Python ->
* county_panel_sub1000.csv).  Visible iteration log.
*   sfpanel ln_y_s ln_L_s ln_K_s ln_M_s ib2005.year,
*           model(bc92) distribution(tnormal) vce(cluster countyid)
*===============================================================
clear all
set more off
set matsize 800

local ROOT "E:/TFP_weather"
import delimited "`ROOT'/src/clean/county_panel_sub1000.csv", clear varnames(1) case(preserve)
destring countyid year SID real_gvp Laborday_impute Land_serv_q capital_serv_q Inter_all_real, replace force
xtset countyid year

gen double ln_S   = ln(Land_serv_q)
gen double ln_y_s = ln(real_gvp)        - ln_S
gen double ln_L_s = ln(Laborday_impute) - ln_S
gen double ln_K_s = ln(capital_serv_q)  - ln_S
gen double ln_M_s = ln(Inter_all_real)  - ln_S

di "N obs = " _N
qui levelsof countyid, local(cc)
di "N counties = " `: word count `cc''

timer clear 1
timer on 1
sfpanel ln_y_s ln_L_s ln_K_s ln_M_s ib2005.year, model(bc92) distribution(tnormal) vce(cluster countyid)
timer off 1
timer list 1
di "converged = " e(converged)
estimates store SFA_SUB
gen byte sfa_sample = e(sample)
matrix list e(b)

predict double xb_s  if sfa_sample, xb
predict double u_hat if sfa_sample, u
gen double te_jlms = exp(-u_hat) if sfa_sample
gen double input_part_s = _b[ln_L_s]*ln_L_s + _b[ln_K_s]*ln_K_s + _b[ln_M_s]*ln_M_s if sfa_sample
gen double ln_frontier  = xb_s - input_part_s if sfa_sample
gen double ln_tfp_chen  = ln_frontier - u_hat if sfa_sample

* did inefficiency actually vary? (wrong-skewness check)
sum u_hat te_jlms, detail

preserve
keep if sfa_sample
keep countyid year SID real_gvp ln_tfp_chen u_hat te_jlms
order countyid year SID real_gvp ln_tfp_chen u_hat te_jlms
export delimited "`ROOT'/src/clean/sfa_sub1000_county_year.csv", replace
restore

di "DONE 17_sfa_sub1000.do"
