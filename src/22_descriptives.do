*===============================================================
* 22_descriptives.do
* Descriptive statistics of the cleaned county agricultural I-O panel.
*
* Table 1  full-sample moments, LEVELS  (N mean sd CV min p10 p25 p50 p75 p90
*          max skewness kurtosis)  -> heavy right tails expected
* Table 2  same in LOGS               -> should be near-Gaussian (kurtosis ~3)
*          if the cleaning worked: this is the key data-quality diagnostic and
*          the justification for the log-linear (Cobb-Douglas) frontier.
* Table 3  panel structure (counties, years, balance, coverage)
* Table 4  by five reform stages: means + growth of output/inputs, and the
*          input mix (K/L, land & labour productivity)
* Table 5  correlation matrix of the logged I-O variables
*
* Outputs (src/clean/): desc_levels.csv desc_logs.csv desc_bystage.csv
*                       desc_corr.csv
*===============================================================
clear all
set more off

local ROOT "E:/TFP_weather"
import delimited "`ROOT'/src/clean/county_panel_clean.csv", clear varnames(1) case(preserve)
destring countyid year SID ag_county real_gvp Laborday_impute Land_serv_q capital_serv_q Inter_all_real, replace force

*--- AGRICULTURAL SAMPLE ---------------------------------------------------*
* The cleaned panel holds ALL resolved rural county units so it can be reused;
* every estimate here is on the cropland>=15% agricultural subset.
qui levelsof countyid, local(_all)
keep if ag_county == 1
qui levelsof countyid, local(_ag)
di "agricultural sample: " `: word count `_ag'' " of " `: word count `_all'' " counties"

* analysis sample = all five I-O variables strictly positive & non-missing
gen byte ok = !missing(real_gvp, Laborday_impute, Land_serv_q, capital_serv_q, Inter_all_real) ///
    & real_gvp>0 & Laborday_impute>0 & Land_serv_q>0 & capital_serv_q>0 & Inter_all_real>0
keep if ok
drop ok
xtset countyid year

label var real_gvp        "Output: real GVP (10k yuan, 2005 prices)"
label var Laborday_impute "Labour (man-days)"
label var Land_serv_q     "Land service (quality-adj.)"
label var capital_serv_q  "Capital service"
label var Inter_all_real  "Intermediate inputs (real)"

*--------------------------------------------------*
* Table 3: panel structure
*--------------------------------------------------*
di _n "=== PANEL STRUCTURE ==="
qui levelsof countyid, local(cc)
qui levelsof year, local(yy)
di "counties     = " `: word count `cc''
di "years        = " `: word count `yy''  "  (" `=r(min)' ")"
qui su year
di "year range   = " r(min) " - " r(max)
di "observations = " _N
qui by countyid: gen byte first = (_n==1)
qui count if first
local ncty = r(N)
qui su year
local nyr = r(max)-r(min)+1
di "balance      = " %4.1f 100*_N/(`ncty'*`nyr') "% of a fully balanced panel"
drop first
xtdes, patterns(5)

*--------------------------------------------------*
* Tables 1 & 2: moments in LEVELS and LOGS
*--------------------------------------------------*
local VARS real_gvp Laborday_impute Land_serv_q capital_serv_q Inter_all_real
foreach v of local VARS {
    qui gen double ln_`v' = ln(`v')
}
* derived productivity / intensity ratios (levels only)
qui gen double y_per_land  = real_gvp/Land_serv_q
qui gen double y_per_labor = real_gvp/Laborday_impute
qui gen double k_per_labor = capital_serv_q/Laborday_impute
label var y_per_land  "Land productivity (output/land service)"
label var y_per_labor "Labour productivity (output/man-day)"
label var k_per_labor "Capital intensity (K/L)"

capture postclose PL
postfile PL str24 variable str6 scale double(N mean sd cv min p10 p25 p50 p75 p90 max skew kurt) ///
    using "`ROOT'/src/clean/_desc_tmp.dta", replace

foreach v of local VARS {
    * ---- levels ----
    qui su `v', detail
    post PL ("`v'") ("level") (r(N)) (r(mean)) (r(sd)) (r(sd)/r(mean)) (r(min)) ///
        (r(p10)) (r(p25)) (r(p50)) (r(p75)) (r(p90)) (r(max)) (r(skewness)) (r(kurtosis))
    * ---- logs ----
    qui su ln_`v', detail
    post PL ("ln_`v'") ("log") (r(N)) (r(mean)) (r(sd)) (abs(r(sd)/r(mean))) (r(min)) ///
        (r(p10)) (r(p25)) (r(p50)) (r(p75)) (r(p90)) (r(max)) (r(skewness)) (r(kurtosis))
}
foreach v in y_per_land y_per_labor k_per_labor {
    qui su `v', detail
    post PL ("`v'") ("ratio") (r(N)) (r(mean)) (r(sd)) (r(sd)/r(mean)) (r(min)) ///
        (r(p10)) (r(p25)) (r(p50)) (r(p75)) (r(p90)) (r(max)) (r(skewness)) (r(kurtosis))
}
postclose PL

preserve
use "`ROOT'/src/clean/_desc_tmp.dta", clear
di _n "=== TABLE 1/2: MOMENTS (levels, logs, ratios) ==="
format mean sd cv min p10 p25 p50 p75 p90 max skew kurt %12.3g
list variable scale N mean sd cv p50 skew kurt, sep(0) noobs
export delimited using "`ROOT'/src/clean/desc_levels.csv", replace
restore

*--------------------------------------------------*
* Table 4: by five reform stages
*--------------------------------------------------*
gen byte stage = .
replace stage = 1 if inrange(year,1981,1984)
replace stage = 2 if inrange(year,1985,1988)
replace stage = 3 if inrange(year,1989,1996)
replace stage = 4 if inrange(year,1997,2003)
replace stage = 5 if inrange(year,2004,2016)
label define stg 1 "1 Reform take-off 81-84" 2 "2 Stagnation 85-88" ///
                 3 "3 Recovery 89-96" 4 "4 Adjustment 97-03" 5 "5 Subsidy era 04-16"
label values stage stg

di _n "=== TABLE 4: MEANS BY REFORM STAGE ==="
tabstat real_gvp Laborday_impute Land_serv_q capital_serv_q Inter_all_real ///
        y_per_land y_per_labor k_per_labor, by(stage) stat(mean) format(%12.3g) nototal

preserve
collapse (mean) real_gvp Laborday_impute Land_serv_q capital_serv_q Inter_all_real ///
                y_per_land y_per_labor k_per_labor (count) n=real_gvp, by(stage)
export delimited using "`ROOT'/src/clean/desc_bystage.csv", replace
restore

* kurtosis of logs by stage (data-quality over time)
di _n "=== log-output kurtosis & skewness by stage (cleaning quality over time) ==="
forvalues s = 1/5 {
    qui su ln_real_gvp if stage==`s', detail
    di "stage `s': N=" %7.0f r(N) "  skew=" %6.2f r(skewness) "  kurt=" %6.2f r(kurtosis)
}

*--------------------------------------------------*
* Table 5: correlations of logged I-O variables
*--------------------------------------------------*
di _n "=== TABLE 5: CORRELATION MATRIX (logs) ==="
correlate ln_real_gvp ln_Laborday_impute ln_Land_serv_q ln_capital_serv_q ln_Inter_all_real
matrix Cm = r(C)
preserve
clear
svmat double Cm, names(col)
export delimited using "`ROOT'/src/clean/desc_corr.csv", replace
restore

erase "`ROOT'/src/clean/_desc_tmp.dta"
di _n "DONE 22_descriptives.do"
