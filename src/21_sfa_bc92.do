*===============================================================
* 21_sfa_bc92.do
* Battese-Coelli (1992) time-decay stochastic frontier, COBB-DOUGLAS, on the
* full cleaned agricultural county panel (cropland>=15%, ~1,975 counties),
* under the SAME returns-to-scale assumption the DEA side uses: NIRS
* (non-increasing returns to scale, RTS <= 1) rather than unconditional CRS.
*
* HOW NIRS IS IMPOSED (KKT logic -- NIRS is an inequality, not an equality)
*   1. estimate UNRESTRICTED (variable returns to scale);
*   2. RTS at the sample mean = sum of the four input coefficients (logs are
*      mean-centred, so this is exact -- Cobb-Douglas RTS is one global
*      number, unlike translog where it varies by observation);
*   3. RTS <= 1 -> constraint is SLACK: the unrestricted fit IS the NIRS fit,
*      nothing further to do;
*   4. RTS >  1 -> constraint BINDS: re-estimate on the boundary RTS = 1
*      (CRS) via land normalization -- sfpanel rejects constraints() together
*      with vce(cluster), and clustered SEs on 1,975 counties matter more
*      than writing the restriction out formally.
*
* CONFIRMED THIS RUN (2026-08-14 log): RTS = 0.4416, comfortably inside NIRS
* -- the CRS branch never fires, so ln_tfp_chen below is the plain
* unrestricted Cobb-Douglas frontier residual. Worth sanity-checking this
* number against DEA's own county-level implied RTS distribution -- 0.44 is
* a fairly strong DRS reading; could be real, could be the four (unnormalized,
* merely centred) inputs absorbing each other's variation.
*
* TRANSLOG -- TRIED AND DROPPED (2026-08-14). A parallel translog spec (14
* input terms: 4 squares + 6 cross-products) was estimated alongside this
* one; its likelihood got stuck at -44403.019 for at least 12 straight
* iterations, flagged "(not concave)" throughout and never clearing -- unlike
* this Cobb-Douglas run, which cleared "not concave" at iteration 15 and
* converged cleanly by 18. Removed rather than forced through more
* iterations. The translog / point-varying-elasticity work continues
* separately in 50b_io_surface_translog.py, outside the SFA/BC92 estimation.
*
* USAGE
*   do src/21_sfa_bc92.do
*
* CONVERGENCE NOTE -- measured on the OLD ~2,500-county panel (before the
* agricultural filter existed), N=2,502/80,626 obs, ~8h across three runs:
* likelihood climbed -226,057 -> -52,499.6, flat from iteration ~14, but kept
* "backing up" (a variance parameter sat on a boundary) so the stopping rule
* never fired; one attempt hung outright. On the current ~1,975-county
* agricultural sample this Cobb-Douglas spec converged cleanly in 18
* iterations, a few minutes -- kept here for reference in case a future,
* slower cut of the sample reproduces the old symptom.
*
* OUTPUTS (src/clean/)
*   sfa_bc92_county_year.csv  countyid year SID real_gvp ln_tfp_chen u_hat
*                             te_jlms rts sfa_model_used
*   sfa_surface_coefs.csv     eq term coef -- frontier coefficients for the
*                             3-D surface figure (50_io_surface.py)
*   sfa_nirs_summary.csv      one row: rts, nirs_binds, sigma_u2, converged,
*                             model_used
*===============================================================
clear all
set more off
set matsize 800

local ROOT "E:/TFP_weather"

di _n "=== 21_sfa_bc92.do  (BC92, Cobb-Douglas, NIRS) ==="

*--------------------------------------------------*
* Load + analysis sample
*--------------------------------------------------*
import delimited "`ROOT'/src/clean/county_panel_clean.csv", clear varnames(1) case(preserve)
destring countyid year SID ag_county real_gvp Laborday_impute Land_serv_q capital_serv_q Inter_all_real, replace force

*--- YEAR RESTRICTION: 1985 onward -----------------------------------------*
count
local n0 = r(N)
keep if year >= 1985
di "year restriction: kept " _N " of `n0' obs (dropped " `n0' - _N ", years 1981-1984)"

*--- AGRICULTURAL SAMPLE ---------------------------------------------------*
qui levelsof countyid, local(_all)
keep if ag_county == 1
qui levelsof countyid, local(_ag)
di "agricultural sample: " `: word count `_ag'' " of " `: word count `_all'' " counties"

*--- exclude composition/leverage outliers identified in 59_/60_ audits ---*
local excl_coastal  370634 330322 330921 371082 320981 350122   // island/coastal specialty ag
local excl_district 110106 110114 310112 310118 321202 330104 330903 360104 370213 430111  // peri-urban districts
local excl_ids `excl_coastal' `excl_district'   // TODO: append your remaining IDs here

gen byte excl_frontier = 0
foreach id of local excl_ids {
    replace excl_frontier = 1 if countyid == `id'
}
count if excl_frontier

drop if excl_frontier == 1
drop excl_frontier

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

*--------------------------------------------------*
* Logs, mean-centred -- centring makes the first-order coefficients readable
* directly as output elasticities AT the sample mean, so RTS_mean is simply
* their sum. Pure reparameterisation: shifts only the constant, changes
* nothing about the fit, residuals, or elasticities.
*--------------------------------------------------*
gen double ln_y = ln(real_gvp)
gen double ln_L = ln(Laborday_impute)
gen double ln_S = ln(Land_serv_q)
gen double ln_K = ln(capital_serv_q)
gen double ln_M = ln(Inter_all_real)

foreach v in L S K M {
    qui sum ln_`v', meanonly
    gen double c`v' = ln_`v' - r(mean)
    label var c`v' "ln input `v', centred at sample mean"
}
local CD "cL cS cK cM"

* CRS (land-normalized) version, used ONLY if the NIRS constraint binds
gen double n_y = ln_y - ln_S
gen double nL  = cL - cS
gen double nK  = cK - cS
gen double nM  = cM - cS
local CDn "nL nK nM"

qui levelsof countyid, local(cc)
di "N obs = " _N "   N counties = " `: word count `cc''
sum ln_y cL cS cK cM

*===============================================================
* Step 1: estimate UNRESTRICTED (variable returns to scale)
*===============================================================
timer clear 1
timer on 1
capture noisily sfpanel ln_y `CD' ib2005.year, ///
    model(bc92) distribution(tnormal) difficult iterate(400) vce(cluster countyid)
local ok = (_rc == 0)
if `ok' local ok = (e(N) < .)
local conv = cond(`ok', e(converged), 0)
timer off 1
timer list 1
di "unrestricted usable = `ok'  (rc=" _rc ", converged=`conv')"

local model_used ""
local rts = .
local binds = 0

if `ok' {
    local rts = _b[cL] + _b[cS] + _b[cK] + _b[cM]
    di _n "RTS at sample mean = " %6.4f `rts'
    capture noisily test cL + cS + cK + cM = 1

    if `rts' <= 1 {
        di as txt "NIRS constraint is SLACK (RTS <= 1) -- unrestricted fit IS the NIRS fit."
        local model_used "bc92_cd_nirs_slack"
    }
    else {
        di as txt "NIRS constraint BINDS (RTS > 1) -> re-estimating on the boundary RTS = 1 (CRS, land-normalized)."
        local binds = 1
        capture noisily sfpanel n_y `CDn' ib2005.year, ///
            model(bc92) distribution(tnormal) difficult iterate(400) vce(cluster countyid)
        local ok = (_rc == 0)
        if `ok' local ok = (e(N) < .)
        local conv = cond(`ok', e(converged), 0)
        local rts = 1
        local model_used "bc92_cd_nirs_crs"
    }
}

if !`ok' {
    di as err "BC92 did not converge -> POOLED frontier fallback (WEAKER MODEL -- see header)."
    frontier ln_y `CD' ib2005.year, distribution(hnormal) vce(cluster countyid)
    local rts = _b[cL] + _b[cS] + _b[cK] + _b[cM]
    local model_used "pooled_hnormal_fallback"
    local conv = 0
}

di "model used: `model_used'"
matrix list e(b)

*--------------------------------------------------*
* Predict frontier, inefficiency, efficiency
*--------------------------------------------------*
gen byte sfa_sample = e(sample)
predict double xb_s  if sfa_sample, xb
predict double u_hat if sfa_sample, u
gen double te_jlms = exp(-u_hat) if sfa_sample

* frontier = constant + year effects, stripped of every INPUT term. Built
* from the actual regressor list so slack vs binds needs no special-casing --
* EXCEPT that the binds branch regressed ln(y/S), so its ln_frontier is still
* in per-land-unit terms and must be shifted by +ln_S to land back on the
* same output-space scale as the slack branch and as DEA/FE downstream.
* (This +ln_S step was MISSING in the previous version -- see review notes.)
gen double input_part = 0 if sfa_sample
local USED = cond(`binds', "`CDn'", "`CD'")
foreach t of local USED {
    qui replace input_part = input_part + _b[`t']*`t' if sfa_sample
}
gen double ln_frontier = xb_s - input_part if sfa_sample
if `binds' {
    qui replace ln_frontier = ln_frontier + ln_S if sfa_sample
}
label var ln_frontier "Frontier technology term: constant + year effects, output scale"

* sanity: the frontier must vary by YEAR ONLY (within-year sd ~ 0)
bys year: egen double sd_frontier = sd(ln_frontier) if sfa_sample
sum sd_frontier if sfa_sample
drop sd_frontier

*--- Chen-style SFA-TFP ---
gen double ln_tfp_chen = ln_frontier - u_hat if sfa_sample
label var ln_tfp_chen "Log SFA-TFP, Chen-style (constant + year effect - u), output scale"
sum ln_tfp_chen u_hat te_jlms, detail

*--- variance parameters (e()-names differ between sfpanel and frontier) ---
local su2 = .
capture local su2 = e(sigma_u2)
if missing(`su2') capture local su2 = e(sigma_u)^2

*--- export county-year ---
gen str24 sfa_model_used = "`model_used'"
gen double rts = `rts'
label var sfa_model_used "bc92_cd_nirs_* = real BC92; pooled_hnormal_fallback = degraded fallback"
preserve
keep if sfa_sample
keep countyid year SID real_gvp ln_tfp_chen u_hat te_jlms rts sfa_model_used
order countyid year SID real_gvp ln_tfp_chen u_hat te_jlms rts sfa_model_used
export delimited "`ROOT'/src/clean/sfa_bc92_county_year.csv", replace
restore

*--------------------------------------------------*
* Export frontier coefficients (constant, inputs, year effects)
*--------------------------------------------------*
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

*--------------------------------------------------*
* One-row NIRS / convergence summary
*--------------------------------------------------*
preserve
clear
set obs 1
gen double rts = `rts'
gen byte nirs_binds = `binds'
gen double sigma_u2 = `su2'
gen byte converged = `conv'
gen str24 model_used = "`model_used'"
export delimited using "`ROOT'/src/clean/sfa_nirs_summary.csv", replace
restore

di _n "sigma_u2 = `su2'"
di "N used   = " e(N)
di _n "DONE 21_sfa_bc92.do  (model_used=`model_used', rts=" %6.4f `rts' ") -> src/clean/sfa_bc92_county_year.csv"
