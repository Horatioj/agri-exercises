*===============================================================
* 23_dea_sfa_framework.do
* Systematic DEA x SFA framework analysis on the cleaned panel.
*
* A. DEA Malmquist DECOMPOSITION  M = EFFCH x TECHCH, aggregated to national
*    with output-share Tornqvist weights, by year and by the five reform stages.
* B. Efficiency LEVELS and their dispersion over time (sigma-convergence).
* C. CATCH-UP (beta-convergence) regression: do initially-inefficient counties
*    grow faster?   dlnTFP_it = a + b*ln(eff_{i,t-1}) + FE + e
* D. Stage regression with county FE and clustered SEs (is the five-stage
*    pattern statistically significant?)
* E. DEA vs SFA comparison table (complementarity: the gap = efficiency change).
*
* Inputs : src/clean/dea_malmquist_county.csv (lnM lnEC lnTC eff_prev eff_now)
*          src/clean/county_panel_clean.csv    (real_gvp for weights)
*          src/clean/sfa_sub1000_county_year.csv (converged BC92 SFA), optional
* Outputs: src/clean/dea_decomp_byyear.csv  dea_decomp_bystage.csv
*          src/clean/framework_compare.csv
*===============================================================
clear all
set more off

local ROOT "E:/TFP_weather"

*--------------------------------------------------*
* 0. Load DEA county-level Malmquist + decomposition
*--------------------------------------------------*
import delimited "`ROOT'/src/clean/dea_malmquist_county.csv", clear varnames(1) case(preserve)
destring countyid year lnM lnEC lnTC eff_prev eff_now, replace force
tempfile dea
save `dea', replace

*--------------------------------------------------*
* 1. Bring in real GVP for the Tornqvist output-share weights
*--------------------------------------------------*
import delimited "`ROOT'/src/clean/county_panel_clean.csv", clear varnames(1) case(preserve)
destring countyid year ag_county real_gvp, replace force
keep if ag_county == 1                      // agricultural sample
keep countyid year real_gvp
keep if real_gvp > 0 & !missing(real_gvp)
tempfile gvp
save `gvp', replace

use `dea', clear
merge 1:1 countyid year using `gvp', keep(match) nogen         // GVP in year t
rename real_gvp gvp_now
gen year_lag = year - 1
rename year year_keep
rename year_lag year
merge 1:1 countyid year using `gvp', keep(match) nogen         // GVP in year t-1
rename real_gvp gvp_lag
rename year year_lag
rename year_keep year
drop year_lag

* output-share Tornqvist weight  w = 0.5*(s_{t-1} + s_t), renormalised within year
bysort year: egen double tot_now = total(gvp_now)
bysort year: egen double tot_lag = total(gvp_lag)
gen double w = 0.5*(gvp_now/tot_now + gvp_lag/tot_lag)
bysort year: egen double wsum = total(w)
replace w = w/wsum
drop tot_now tot_lag wsum

* winsorize county growth 1/99 WITHIN year (heavy tails; cf. SFA.do)
foreach v in lnM lnEC lnTC {
    bysort year: egen double p01_`v' = pctile(`v'), p(1)
    bysort year: egen double p99_`v' = pctile(`v'), p(99)
    replace `v' = p01_`v' if `v' < p01_`v'
    replace `v' = p99_`v' if `v' > p99_`v'
    drop p01_`v' p99_`v'
}

* five reform stages
gen byte stage = .
replace stage = 1 if inrange(year,1981,1984)
replace stage = 2 if inrange(year,1985,1988)
replace stage = 3 if inrange(year,1989,1996)
replace stage = 4 if inrange(year,1997,2003)
replace stage = 5 if inrange(year,2004,2016)
label define stg 1 "1 Reform take-off" 2 "2 Stagnation" 3 "3 Recovery" ///
                 4 "4 Adjustment" 5 "5 Subsidy era"
label values stage stg
xtset countyid year
save "`ROOT'/src/clean/_dea_work.dta", replace

*--------------------------------------------------*
* A. National decomposition by YEAR (weighted)
*--------------------------------------------------*
preserve
gen double wM = w*lnM
gen double wE = w*lnEC
gen double wT = w*lnTC
collapse (sum) lnM_w=wM lnEC_w=wE lnTC_w=wT (mean) lnM_s=lnM lnEC_s=lnEC lnTC_s=lnTC ///
         (mean) eff=eff_now (sd) eff_sd=eff_now (count) n=lnM, by(year)
sort year
gen double tfp_pct  = 100*(exp(lnM_w)-1)
gen double effch_pct= 100*(exp(lnEC_w)-1)
gen double tech_pct = 100*(exp(lnTC_w)-1)
gen double cum_tfp  = sum(lnM_w)
gen double cum_ec   = sum(lnEC_w)
gen double cum_tc   = sum(lnTC_w)
label var tfp_pct   "DEA TFP growth % (weighted)"
label var effch_pct "Efficiency change % (catch-up)"
label var tech_pct  "Technical change % (frontier shift)"
di _n "=== A. DEA Malmquist decomposition by year (output-weighted) ==="
list year n tfp_pct effch_pct tech_pct eff eff_sd, sep(0) noobs
export delimited using "`ROOT'/src/clean/dea_decomp_byyear.csv", replace
restore

*--------------------------------------------------*
* A2. By REFORM STAGE
*--------------------------------------------------*
preserve
gen double wM = w*lnM
gen double wE = w*lnEC
gen double wT = w*lnTC
collapse (sum) lnM_w=wM lnEC_w=wE lnTC_w=wT (count) n=lnM, by(year stage)
collapse (mean) lnM_w lnEC_w lnTC_w (sd) sd_tfp=lnM_w (sum) nyr=n, by(stage)
gen double tfp_pct   = 100*(exp(lnM_w)-1)
gen double effch_pct = 100*(exp(lnEC_w)-1)
gen double tech_pct  = 100*(exp(lnTC_w)-1)
gen double ec_share  = 100*lnEC_w/lnM_w
di _n "=== A2. DEA decomposition by REFORM STAGE (mean %/yr) ==="
list stage tfp_pct effch_pct tech_pct ec_share, sep(0) noobs
export delimited using "`ROOT'/src/clean/dea_decomp_bystage.csv", replace
restore

*--------------------------------------------------*
* B. Efficiency levels & sigma-convergence
*--------------------------------------------------*
di _n "=== B. Technical efficiency: level and dispersion (sigma-convergence) ==="
preserve
collapse (mean) eff=eff_now (sd) sd=eff_now (p10) p10=eff_now (p90) p90=eff_now, by(year)
gen double cv = sd/eff
list year eff sd cv p10 p90, sep(0) noobs
restore

*--------------------------------------------------*
* C. beta-CONVERGENCE: do inefficient counties catch up?
*--------------------------------------------------*
di _n "=== C. Catch-up regression: dlnTFP on lagged (log) efficiency ==="
gen double ln_eff_lag = ln(eff_prev)
* pooled with year FE, clustered by county
reghdfe lnM ln_eff_lag, absorb(year) vce(cluster countyid)
estimates store CONV_A
* + county FE
reghdfe lnM ln_eff_lag, absorb(countyid year) vce(cluster countyid)
estimates store CONV_B
* efficiency-change component only (the pure catch-up margin)
reghdfe lnEC ln_eff_lag, absorb(countyid year) vce(cluster countyid)
estimates store CONV_C
di "NOTE: negative coefficient = counties further BELOW the frontier grow faster (catch-up)."

*--------------------------------------------------*
* D. Are the five stages statistically distinct?
*--------------------------------------------------*
di _n "=== D. Stage regression (county FE, clustered SE), base = stage 4 ==="
reghdfe lnM ib4.stage [aw=w], absorb(countyid) vce(cluster countyid)
estimates store STAGE_TFP
test 1.stage = 2.stage = 3.stage = 5.stage
reghdfe lnEC ib4.stage [aw=w], absorb(countyid) vce(cluster countyid)
estimates store STAGE_EC
reghdfe lnTC ib4.stage [aw=w], absorb(countyid) vce(cluster countyid)
estimates store STAGE_TC

*--------------------------------------------------*
* E. DEA vs SFA comparison (complementarity)
*--------------------------------------------------*
capture confirm file "`ROOT'/src/clean/sfa_sub1000_county_year.csv"
if _rc == 0 {
    di _n "=== E. DEA vs SFA on the SAME (1000-county) sample ==="
    preserve
    import delimited "`ROOT'/src/clean/sfa_sub1000_county_year.csv", clear varnames(1) case(preserve)
    destring countyid year ln_tfp_chen u_hat, replace force
    xtset countyid year
    gen double dln_sfa = ln_tfp_chen - L.ln_tfp_chen
    keep countyid year dln_sfa
    drop if missing(dln_sfa)
    tempfile sfa
    save `sfa', replace
    use "`ROOT'/src/clean/_dea_work.dta", clear
    merge 1:1 countyid year using `sfa', keep(match) nogen
    gen byte stg = stage
    collapse (mean) dea_tfp=lnM dea_ec=lnEC dea_tc=lnTC sfa_tfp=dln_sfa, by(stg)
    foreach v in dea_tfp dea_ec dea_tc sfa_tfp {
        replace `v' = 100*(exp(`v')-1)
    }
    gen double gap = dea_tfp - sfa_tfp
    label var gap "DEA - SFA (= efficiency component SFA misses)"
    list stg dea_tfp dea_ec dea_tc sfa_tfp gap, sep(0) noobs
    export delimited using "`ROOT'/src/clean/framework_compare.csv", replace
    restore
}

erase "`ROOT'/src/clean/_dea_work.dta"
di _n "DONE 23_dea_sfa_framework.do"
