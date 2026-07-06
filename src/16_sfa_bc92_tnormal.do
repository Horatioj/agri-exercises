*===============================================================
* 16_sfa_bc92_tnormal.do
* EXACT SFA.do spec, run verbatim with VISIBLE iteration log:
*   sfpanel ln_y_s ln_L_s ln_K_s ln_M_s ib2005.year,
*           model(bc92) distribution(tnormal) vce(cluster countyid)
* then Chen-style TFP (ln_frontier - u) exported for aggregation.
*===============================================================
clear all
set more off
set matsize 800

local ROOT "E:/TFP_weather"
import delimited "`ROOT'/src/clean/county_panel_clean.csv", clear varnames(1) case(preserve)
destring countyid year SID real_gvp Laborday_impute Land_serv_q capital_serv_q Inter_all_real, replace force

gen byte any_invalid = ///
    missing(real_gvp)        | real_gvp        <= 0 | ///
    missing(Laborday_impute) | Laborday_impute <= 0 | ///
    missing(Land_serv_q)     | Land_serv_q     <= 0 | ///
    missing(capital_serv_q)  | capital_serv_q  <= 0 | ///
    missing(Inter_all_real)  | Inter_all_real  <= 0
drop if any_invalid == 1
drop any_invalid
xtset countyid year

gen double ln_S   = ln(Land_serv_q)
gen double ln_y_s = ln(real_gvp)        - ln_S
gen double ln_L_s = ln(Laborday_impute) - ln_S
gen double ln_K_s = ln(capital_serv_q)  - ln_S
gen double ln_M_s = ln(Inter_all_real)  - ln_S

*--------------------------------------------------*
* EXACT SFA.do spec (visible log so we can see convergence behavior)
*--------------------------------------------------*
timer clear 1
timer on 1
sfpanel ln_y_s ln_L_s ln_K_s ln_M_s ib2005.year, model(bc92) distribution(tnormal) vce(cluster countyid)
timer off 1
timer list 1
di "converged = " e(converged)
estimates store SFA_BC92TN
gen byte sfa_sample = e(sample)
matrix list e(b)

*--------------------------------------------------*
* Chen-style TFP
*--------------------------------------------------*
predict double xb_s  if sfa_sample, xb
predict double u_hat if sfa_sample, u
gen double te_jlms = exp(-u_hat) if sfa_sample

gen double input_part_s = _b[ln_L_s]*ln_L_s + _b[ln_K_s]*ln_K_s + _b[ln_M_s]*ln_M_s if sfa_sample
gen double ln_frontier  = xb_s - input_part_s if sfa_sample
gen double ln_tfp_chen  = ln_frontier - u_hat if sfa_sample

* diagnostics: does inefficiency actually vary? (wrong-skewness check)
sum u_hat te_jlms, detail
bys year: egen double sd_frontier = sd(ln_frontier) if sfa_sample
sum sd_frontier if sfa_sample
drop sd_frontier

preserve
keep if sfa_sample
keep countyid year SID real_gvp ln_tfp_chen u_hat te_jlms
order countyid year SID real_gvp ln_tfp_chen u_hat te_jlms
export delimited "`ROOT'/src/clean/sfa_bc92tn_county_year.csv", replace
restore

di "DONE 16_sfa_bc92_tnormal.do"
