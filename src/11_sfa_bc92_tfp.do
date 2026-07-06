*===============================================================
* 11_sfa_bc92_tfp.do
* SFA per SFA.do: LAND-NORMALIZED (CRS) Battese-Coelli (1992) time-decay
* frontier on the CLEANED county panel, then Chen-style TFP.
*
*   sfpanel ln_y_s ln_L_s ln_K_s ln_M_s ib2005.year,
*           model(bc92) distribution(tnormal) vce(cluster countyid)
*   ln_tfp_chen = (alpha + year effect) - u_hat
*
* Exports county-year TFP + real_gvp so national aggregation (simple mean
* AND output-share Tornqvist weighting) can be built in Python, alongside DEA.
*
* Output (src/clean/): sfa_bc92_county_year.csv
*   countyid year SID real_gvp ln_tfp_chen u_hat te_bc te_jlms
*===============================================================
clear all
set more off
set matsize 800

local ROOT "E:/TFP_weather"
import delimited "`ROOT'/src/clean/county_panel_clean.csv", clear varnames(1) case(preserve)
destring countyid year SID real_gvp Laborday_impute Land_serv_q capital_serv_q Inter_all_real, replace force

*--- keep valid (all 5 strictly positive & non-missing) ---
gen byte any_invalid = ///
    missing(real_gvp)        | real_gvp        <= 0 | ///
    missing(Laborday_impute) | Laborday_impute <= 0 | ///
    missing(Land_serv_q)     | Land_serv_q     <= 0 | ///
    missing(capital_serv_q)  | capital_serv_q  <= 0 | ///
    missing(Inter_all_real)  | Inter_all_real  <= 0
tab any_invalid
drop if any_invalid == 1
drop any_invalid

xtset countyid year

*--- land normalization (impose CRS): everything per unit land service ---
gen double ln_S   = ln(Land_serv_q)
gen double ln_y_s = ln(real_gvp)        - ln_S
gen double ln_L_s = ln(Laborday_impute) - ln_S
gen double ln_K_s = ln(capital_serv_q)  - ln_S
gen double ln_M_s = ln(Inter_all_real)  - ln_S
label var ln_y_s "ln real output per land service"
label var ln_L_s "ln labor days per land service"
label var ln_K_s "ln capital service per land service"
label var ln_M_s "ln intermediate inputs per land service"

sum ln_y_s ln_L_s ln_K_s ln_M_s ln_S

*--------------------------------------------------*
* SFA: Battese-Coelli 1992, truncated-normal, year dummies (base 2005)
*--------------------------------------------------*
timer clear 1
timer on 1
* Try BC92 (time-decay) with half-normal (more robust than tnormal), difficult
* optimizer and a hard iteration cap so it cannot hang blind.
capture noisily sfpanel ln_y_s ln_L_s ln_K_s ln_M_s ib2005.year, ///
    model(bc92) distribution(hnormal) difficult iterate(50) vce(cluster countyid)
local okbc = (_rc==0)
if `okbc' local okbc = (e(converged)==1)
scalar used_bc92 = `okbc'
if !`okbc' {
    di as err "BC92 did not converge (rc=" _rc ", conv=" e(converged) ") -> pooled frontier fallback"
    * Pooled cross-sectional SFA with year dummies: robust, fast; same Chen
    * construction (alpha+year effects - u_it), u_it varies by observation.
    frontier ln_y_s ln_L_s ln_K_s ln_M_s ib2005.year, ///
        distribution(hnormal) vce(cluster countyid)
}
timer off 1
timer list 1
di "used_bc92 = " used_bc92 "   converged = " e(converged)
estimates store SFA_USED
gen byte sfa_sample = e(sample)
matrix list e(b)

*--------------------------------------------------*
* Predict frontier, inefficiency, efficiency
*--------------------------------------------------*
* xb and u (E[u|e]) are supported by BOTH sfpanel and frontier; te via exp(-u)
predict double xb_s   if sfa_sample, xb
predict double u_hat  if sfa_sample, u
gen double te_jlms = exp(-u_hat) if sfa_sample

* frontier = alpha + year effects  (strip input contribution)
gen double input_part_s = ///
    _b[ln_L_s]*ln_L_s + _b[ln_K_s]*ln_K_s + _b[ln_M_s]*ln_M_s if sfa_sample
gen double ln_frontier = xb_s - input_part_s if sfa_sample
label var ln_frontier "Frontier technology term: alpha + year effects"

* check frontier only varies by year (should be ~0 within-year sd)
bys year: egen double sd_frontier = sd(ln_frontier) if sfa_sample
sum sd_frontier if sfa_sample
drop sd_frontier

*--------------------------------------------------*
* Chen-style SFA-TFP
*--------------------------------------------------*
gen double ln_tfp_chen = ln_frontier - u_hat if sfa_sample
label var ln_tfp_chen "Log SFA-TFP, Chen-style (alpha+year-u)"
sum ln_tfp_chen u_hat te_jlms, detail

*--- export county-year for Python aggregation (simple + weighted) ---
preserve
keep if sfa_sample
keep countyid year SID real_gvp ln_tfp_chen u_hat te_jlms
order countyid year SID real_gvp ln_tfp_chen u_hat te_jlms
export delimited "`ROOT'/src/clean/sfa_bc92_county_year.csv", replace
restore

di "DONE 11_sfa_bc92_tfp.do"
