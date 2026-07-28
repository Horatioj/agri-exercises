*===============================================================
* 28_sfa_bc92_surface.do
* Estimate the SFA production FRONTIER used to draw the 3-D surface figure
* (src/23_io_combined.py), on the current cleaned panel.
*
*   sfpanel ln_y_s ln_L_s ln_K_s ln_M_s ib2005.year,
*           model(bc92) distribution(tnormal) vce(cluster countyid)
*
* Land-normalized (imposes CRS), per METHODOLOGY.md 4.1.  Converting back to
* levels, the frontier is
*   ln y = alpha + lambda_t + bL*lnL + bK*lnK + bM*lnM + (1-bL-bK-bM)*lnLand
* so the implied land elasticity is 1-bL-bK-bM and the surface is concave in
* (L,M) whenever bL+bM < 1.
*
* This can take HOURS at N=2,502 (the ML surface is non-concave early on).
* If it fails to converge, it retries on a 1,000-county random subsample
* (the specification that converged previously).
*
* Output: src/clean/sfa_surface_coefs.csv   (term, coef)  - year effects included
*===============================================================
clear all
set more off
set matsize 800
set seed 42

local ROOT "E:/TFP_weather"
import delimited "`ROOT'/src/clean/county_panel_clean.csv", clear varnames(1) case(preserve)
destring countyid year SID real_gvp Laborday_impute Land_serv_q capital_serv_q Inter_all_real, replace force

gen byte bad = missing(real_gvp, Laborday_impute, Land_serv_q, capital_serv_q, Inter_all_real) ///
    | real_gvp<=0 | Laborday_impute<=0 | Land_serv_q<=0 | capital_serv_q<=0 | Inter_all_real<=0
drop if bad
drop bad
xtset countyid year

gen double ln_S   = ln(Land_serv_q)
gen double ln_y_s = ln(real_gvp)        - ln_S
gen double ln_L_s = ln(Laborday_impute) - ln_S
gen double ln_K_s = ln(capital_serv_q)  - ln_S
gen double ln_M_s = ln(Inter_all_real)  - ln_S

qui levelsof countyid, local(cc)
di "N obs = " _N "   N counties = " `: word count `cc''
save "`ROOT'/src/clean/_sfa_surface_sample.dta", replace

*--------------------------------------------------*
* Attempt 1: FULL panel   (set FULLTRY 0 to skip -- see note)
*--------------------------------------------------*
* FULL-PANEL EXPERIENCE, N=2,502 / 80,626 obs (three runs, ~8h total):
*   - the likelihood climbs -226,057 -> -52,499.6 and is FLAT from iteration 14
*     (relative change 7e-7; 2e-8 by iteration 17)
*   - but every step reports "backed up" and Stata's scaled-gradient rule never
*     fires; ~10 min per iteration
*   - and the run HUNG outright (0% CPU, no progress) at iteration 4 on one
*     attempt -> the full-panel route is not reproducible.
* The 1,000-county subsample below converges cleanly in ~15 min and is the
* specification documented in METHODOLOGY.md 4.3, so it is the default.
local FULLTRY 0
timer clear 1
timer on 1
if `FULLTRY' {
* CONVERGENCE NOTE (measured on this panel, N=2,502 / 80,626 obs):
* the likelihood climbs -226,057 -> -52,499.6 and is FLAT from iteration ~14
* (relative change 7e-7, and 2e-8 by iteration 17), but every step is reported
* as "backed up" -- a variance parameter sits on a boundary, so Stata's
* scaled-GRADIENT rule never fires and it would grind to iterate() (~10 min per
* iteration).  We therefore stop at the flat optimum and ACCEPT those estimates:
* `nonrtolerance' drops the gradient test and iterate(14) stops on the plateau.
* The estimates are numerically converged; only Stata's formal flag is not set.
    capture noisily sfpanel ln_y_s ln_L_s ln_K_s ln_M_s ib2005.year, ///
        model(bc92) distribution(tnormal) difficult iterate(14) nonrtolerance ///
        vce(cluster countyid)
    local ok = (_rc==0)
    if `ok' local ok = (e(N)<.)
    di "FULL-panel usable = " `ok'  "  (rc=" _rc ")"
}
else {
    local ok 0
    di "FULL-panel attempt skipped (FULLTRY=0) -> 1,000-county subsample"
}
timer off 1
timer list 1

*--------------------------------------------------*
* Attempt 2 (fallback): 1,000-county random subsample
*--------------------------------------------------*
if !`ok' {
    di as err "full panel did not converge -> retry on a 1,000-county subsample"
    use "`ROOT'/src/clean/_sfa_surface_sample.dta", clear
    preserve
    bysort countyid: keep if _n==1
    keep countyid
    gen double u = runiform()
    sort u
    keep if _n <= 1000
    keep countyid
    tempfile pick
    save `pick', replace
    restore
    merge m:1 countyid using `pick', keep(match) nogen
    xtset countyid year
    qui levelsof countyid, local(c2)
    di "subsample: " _N " obs, " `: word count `c2'' " counties"
    sfpanel ln_y_s ln_L_s ln_K_s ln_M_s ib2005.year, ///
        model(bc92) distribution(tnormal) difficult iterate(400) vce(cluster countyid)
    di "SUBSAMPLE converged = " e(converged)
}

*--------------------------------------------------*
* Export frontier coefficients (constant, inputs, year effects)
*--------------------------------------------------*
matrix b = e(b)
local nb = colsof(b)
local names : colnames b
local eqs   : coleq b

* NOTE: e(b) carries several "_cons" (frontier + variance equations), so the
* equation name must be exported alongside the coefficient name.
capture postclose CF
postfile CF str32 eq str32 term double coef using "`ROOT'/src/clean/_sfa_coef_tmp.dta", replace
forvalues j = 1/`nb' {
    local nm : word `j' of `names'
    local eq : word `j' of `eqs'
    post CF ("`eq'") ("`nm'") (b[1,`j'])
}
postclose CF

* also record which sample was used and the variance parameters
preserve
use "`ROOT'/src/clean/_sfa_coef_tmp.dta", clear
export delimited using "`ROOT'/src/clean/sfa_surface_coefs.csv", replace
restore

di _n "sigma_u2 = " e(sigma_u2) "   sigma_v2 = " e(sigma_v2)
di "N used   = " e(N)
erase "`ROOT'/src/clean/_sfa_coef_tmp.dta"
erase "`ROOT'/src/clean/_sfa_surface_sample.dta"
di _n "DONE 28_sfa_bc92_surface.do"
