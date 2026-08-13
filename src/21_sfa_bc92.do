*===============================================================
* 21_sfa_bc92.do
* LAND-NORMALIZED (CRS) Battese-Coelli (1992) time-decay stochastic frontier on
* the cleaned county panel, then Chen-style TFP -- ONE file for every sample and
* distribution the analysis uses.  The four earlier do-files (11 / 16 / 17 / 28)
* were the same specification run on different samples and differed only in
* which CSV they wrote, so they are folded into the arguments below.
*
*   sfpanel ln_y_s ln_L_s ln_K_s ln_M_s ib2005.year,
*           model(bc92) distribution(`DIST') vce(cluster countyid)
*   ln_tfp_chen = (alpha + year effect) - u_hat
*
* USAGE
*   do src/21_sfa_bc92.do [SAMPLE] [DIST] [NFULLTRY]
*     SAMPLE   full (default) = ALL ~1,800 agricultural counties
*              sub1000 | <integer N>  = random N-county subsample of them
*     DIST     tnormal (default) | hnormal
*     NFULLTRY 1 = attempt BC92 on the full panel; 0 = go straight to sub1000
*              (only consulted when SAMPLE == full).  Default 1.
*
*   do src/21_sfa_bc92.do full     tnormal      // headline spec (was 11 / 16)
*   do src/21_sfa_bc92.do sub1000  tnormal      // converged spec (was 17)
*   do src/21_sfa_bc92.do full     tnormal  0   // coefficients only (was 28)
*
* WHY THE SAMPLE ARGUMENT EXISTS -- measured on the OLD ~2,500-county panel,
* N=2,502 / 80,626 obs, over three runs, ~8h total:
*   - the likelihood climbs -226,057 -> -52,499.6 and is FLAT from iteration ~14
*     (relative change 7e-7; 2e-8 by iteration 17)
*   - but every step reports "backed up": a variance parameter sits on a
*     boundary, so Stata's scaled-gradient rule never fires and it grinds to
*     iterate() at ~10 min per iteration
*   - and one attempt HUNG outright (0% CPU) at iteration 4, so the full-panel
*     route is not reproducible.
* The 1,000-county subsample converges cleanly in ~15 min.  The agricultural
* filter cuts the full sample to ~1,800, which is between the two, so the full
* run is attempted under ORDINARY convergence rules (no `nonrtolerance') and
* should be given a wall-clock cap by the caller.  If it does not converge,
* fall back to sub1000 -- the specification documented in METHODOLOGY.md 4.3.
*
* OUTPUTS (src/clean/)
*   sfa_bc92_county_year.csv      SAMPLE=full (all ag)  countyid year SID real_gvp
*   sfa_sub1000_county_year.csv   SAMPLE=sub1000    ln_tfp_chen u_hat te_jlms
*   sfa_surface_coefs.csv         always: eq, term, coef -- the frontier
*                                 coefficients the 3-D surface figure
*                                 (50_io_surface.py) draws
*===============================================================
clear all
set more off
set matsize 800
set seed 42

local ROOT "E:/TFP_weather"
local SAMPLE  "`1'"
local DIST    "`2'"
local FULLTRY "`3'"
if "`SAMPLE'"  == "" local SAMPLE  "full"
if "`DIST'"    == "" local DIST    "tnormal"
if "`FULLTRY'" == "" local FULLTRY 1

if "`SAMPLE'" == "full"    local NSUB 0
else if "`SAMPLE'" == "sub1000" local NSUB 1000
else                       local NSUB = real("`SAMPLE'")
if `NSUB' == . {
    di as err "SAMPLE must be full, sub1000 or an integer; got `SAMPLE'"
    exit 198
}

if "`SAMPLE'" == "full"         local OUTCSV "sfa_bc92_county_year.csv"
else if "`SAMPLE'" == "sub1000" local OUTCSV "sfa_sub1000_county_year.csv"
else                            local OUTCSV "sfa_sub`NSUB'_county_year.csv"

di _n "=== 21_sfa_bc92.do  SAMPLE=`SAMPLE'  DIST=`DIST'  FULLTRY=`FULLTRY' ==="

*--------------------------------------------------*
* Load + analysis sample
*--------------------------------------------------*
import delimited "`ROOT'/src/clean/county_panel_clean.csv", clear varnames(1) case(preserve)
destring countyid year SID ag_county real_gvp Laborday_impute Land_serv_q capital_serv_q Inter_all_real, replace force

*--- AGRICULTURAL SAMPLE ---------------------------------------------------*
* The cleaned panel holds ALL resolved rural county units so it can be reused;
* every estimate here is on the cropland>=15% agricultural subset.
qui levelsof countyid, local(_all)
keep if ag_county == 1
qui levelsof countyid, local(_ag)
di "agricultural sample: " `: word count `_ag'' " of " `: word count `_all'' " counties"

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

*--- optional random county subsample ---
* Inner Mongolia (15), Tibet (54), Qinghai (63) and Xinjiang (65) are excluded:
* huge pastoral units whose land service is not comparable, and which dominate
* the frontier tail that stalls the optimizer.
if `NSUB' > 0 {
    drop if inlist(SID, 15, 54, 63, 65)
    preserve
        bysort countyid: keep if _n == 1
        keep countyid
        gen double u = runiform()
        sort u
        keep if _n <= `NSUB'
        keep countyid
        tempfile pick
        save `pick', replace
    restore
    merge m:1 countyid using `pick', keep(match) nogen
}

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

qui levelsof countyid, local(cc)
di "N obs = " _N "   N counties = " `: word count `cc''
sum ln_y_s ln_L_s ln_K_s ln_M_s ln_S

*--------------------------------------------------*
* Estimate
*--------------------------------------------------*
timer clear 1
timer on 1
local ok 0
if `NSUB' > 0 | `FULLTRY' {
    if `NSUB' > 0 {
        capture noisily sfpanel ln_y_s ln_L_s ln_K_s ln_M_s ib2005.year, ///
            model(bc92) distribution(`DIST') difficult iterate(400) vce(cluster countyid)
    }
    else {
        * full agricultural sample: ordinary convergence rules, visible log.
        * Give this a wall-clock cap from outside -- see the header note.
        capture noisily sfpanel ln_y_s ln_L_s ln_K_s ln_M_s ib2005.year, ///
            model(bc92) distribution(`DIST') difficult iterate(100) ///
            vce(cluster countyid)
    }
    local ok = (_rc == 0)
    if `ok' local ok = (e(N) < .)
    di "BC92 usable = `ok'   (rc=" _rc ", converged=" e(converged) ")"
}
else di "BC92 full-panel attempt skipped (FULLTRY=0)"

if !`ok' {
    di as err "BC92 unusable -> pooled frontier fallback"
    * Pooled cross-sectional SFA with year dummies: robust, fast; same Chen
    * construction (alpha + year effects - u_it), u_it varies by observation.
    * NOTE: on the cleaned panel this collapses to sigma_u -> 0 (wrong skewness,
    * METHODOLOGY.md 5.1), so its TFP path is neutral technical change only.
    frontier ln_y_s ln_L_s ln_K_s ln_M_s ib2005.year, ///
        distribution(hnormal) vce(cluster countyid)
}
timer off 1
timer list 1
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

* frontier = alpha + year effects  (strip the input contribution)
gen double input_part_s = ///
    _b[ln_L_s]*ln_L_s + _b[ln_K_s]*ln_K_s + _b[ln_M_s]*ln_M_s if sfa_sample
gen double ln_frontier = xb_s - input_part_s if sfa_sample
label var ln_frontier "Frontier technology term: alpha + year effects"

* check the frontier only varies by year (should be ~0 within-year sd)
bys year: egen double sd_frontier = sd(ln_frontier) if sfa_sample
sum sd_frontier if sfa_sample
drop sd_frontier

*--------------------------------------------------*
* Chen-style SFA-TFP
*--------------------------------------------------*
gen double ln_tfp_chen = ln_frontier - u_hat if sfa_sample
label var ln_tfp_chen "Log SFA-TFP, Chen-style (alpha + year effect - u)"
* did inefficiency actually vary?  (the wrong-skewness check)
sum ln_tfp_chen u_hat te_jlms, detail

*--- export county-year for the Python aggregation (simple + weighted) ---
preserve
keep if sfa_sample
keep countyid year SID real_gvp ln_tfp_chen u_hat te_jlms
order countyid year SID real_gvp ln_tfp_chen u_hat te_jlms
export delimited "`ROOT'/src/clean/`OUTCSV'", replace
restore

*--------------------------------------------------*
* Export frontier coefficients (constant, inputs, year effects)
*--------------------------------------------------*
* Converting back to levels, the frontier is
*   ln y = alpha + lambda_t + bL*lnL + bK*lnK + bM*lnM + (1-bL-bK-bM)*lnLand
* so the implied land elasticity is 1-bL-bK-bM and the surface is concave in
* (L,M) whenever bL+bM < 1.  e(b) carries several "_cons" (frontier AND variance
* equations), so the equation name must be exported alongside the term.
matrix b = e(b)
local nb = colsof(b)
local names : colnames b
local eqs   : coleq b
capture postclose CF
postfile CF str32 eq str32 term double coef using "`ROOT'/src/clean/_sfa_coef_tmp.dta", replace
forvalues j = 1/`nb' {
    local nm : word `j' of `names'
    local eq : word `j' of `eqs'
    post CF ("`eq'") ("`nm'") (b[1,`j'])
}
postclose CF
preserve
use "`ROOT'/src/clean/_sfa_coef_tmp.dta", clear
export delimited using "`ROOT'/src/clean/sfa_surface_coefs.csv", replace
restore
erase "`ROOT'/src/clean/_sfa_coef_tmp.dta"

di _n "sigma_u2 = " e(sigma_u2) "   sigma_v2 = " e(sigma_v2)
di "N used   = " e(N)
di _n "DONE 21_sfa_bc92.do  (SAMPLE=`SAMPLE', DIST=`DIST') -> src/clean/`OUTCSV'"
