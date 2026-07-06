*===============================================================
* 10_sfa_tfe_tfp.do
* SFA True-Fixed-Effects (Greene 2005) production frontier on the
* CLEANED county panel, then Chen-style TFP per year + variations.
*
* Model spec taken from src/_sfa_test.do (full sample this time):
*     sfpanel ly lL lLand lK lM i.year, model(tfe) distribution(hnormal)
* TFP construction taken from SFA.do:
*     ln_tfp_it = (year/technology effect lambda_t) - u_it
*     alpha_i (county FE) and input contributions are NOT part of TFP;
*     alpha_i cancels in growth / per-county base normalization.
*
* Outputs (src/clean/):
*   sfa_tfe_county_year.csv   countyid year ln_tfp u_hat te_bc  (county-level)
*   sfa_tfe_annual.csv        year national mean/sd/n of dln_tfp + cum index
*===============================================================
clear all
set more off
set matsize 800

local ROOT "E:/TFP_weather"
import delimited "`ROOT'/src/clean/county_panel_clean.csv", clear varnames(1) case(preserve)
destring countyid year SID, replace force

*--- logs of output and the 5 inputs (levels; NO land normalization, per _sfa_test) ---
gen double ly    = ln(real_gvp)
gen double lL    = ln(Laborday_impute)
gen double lLand = ln(Land_serv_q)
gen double lK    = ln(capital_serv_q)
gen double lM    = ln(Inter_all_real)
keep if ly<. & lL<. & lLand<. & lK<. & lM<.

xtset countyid year
di "N obs = " _N
qui levelsof year, local(yy)
di "years: `yy'"

*--------------------------------------------------*
* SFA estimation: True Fixed Effects, half-normal
*--------------------------------------------------*
timer clear 1
timer on 1
sfpanel ly lL lLand lK lM i.year, model(tfe) distribution(hnormal) nolog
timer off 1
timer list 1
di "converged = " e(converged)
estimates store SFA_TFE

gen byte sfa_sample = e(sample)
matrix list e(b)

*--------------------------------------------------*
* Predict inefficiency & efficiency
*--------------------------------------------------*
predict double u_hat  if sfa_sample, u
predict double te_bc  if sfa_sample, bc

*--------------------------------------------------*
* Extract the technology (year) effect lambda_t
* i.year base level = smallest year -> coef 0.  _b[`t'.year] for others.
*--------------------------------------------------*
gen double lambda_t = 0
qui levelsof year, local(yrs)
foreach t of local yrs {
    capture local bt = _b[`t'.year]
    if _rc==0 {
        qui replace lambda_t = `bt' if year==`t'
    }
}

*--------------------------------------------------*
* Chen-style TFP:  ln_tfp = lambda_t - u_hat
* (county FE alpha_i excluded: heterogeneity, cancels in growth)
*--------------------------------------------------*
gen double ln_tfp = lambda_t - u_hat if sfa_sample
label var ln_tfp "Log SFA-TFE TFP (year effect - inefficiency)"

xtset countyid year
* year-to-year log growth (per county)
gen double dln_tfp = ln_tfp - L.ln_tfp if sfa_sample
label var dln_tfp "Year-on-year log TFP growth (SFA-TFE)"

*--- save county-level ---
preserve
keep if sfa_sample
keep countyid year SID ln_tfp u_hat te_bc dln_tfp lambda_t
order countyid year SID lambda_t u_hat te_bc ln_tfp dln_tfp
export delimited "`ROOT'/src/clean/sfa_tfe_county_year.csv", replace
restore

*--------------------------------------------------*
* National annual TFP growth (mean of county dln) + variation
*--------------------------------------------------*
preserve
keep if sfa_sample & dln_tfp<.
collapse (mean) dln_mean=dln_tfp (sd) dln_sd=dln_tfp (count) n=dln_tfp, by(year)
gen double growth_pct = 100*(exp(dln_mean)-1)
gen double se = dln_sd/sqrt(n)
* cumulative log index (base = first growth year = 0)
sort year
gen double cum_lntfp = sum(dln_mean)
label var dln_mean  "Mean county log TFP growth"
label var growth_pct "Annual TFP growth (%)"
label var cum_lntfp  "Cumulative log TFP (base year=0)"
list year n dln_mean growth_pct dln_sd, sep(0)
export delimited "`ROOT'/src/clean/sfa_tfe_annual.csv", replace
restore

di "DONE 10_sfa_tfe_tfp.do"
