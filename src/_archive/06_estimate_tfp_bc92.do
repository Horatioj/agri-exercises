*--------------------------------------------------------------------
* BC92 (Battese-Coelli 1992) Cobb-Douglas frontier, 3 inputs (L,K,M; no land).
* NOTE: putting ib2005.year inside sfpanel bc92 is non-concave (BC92 already
*   models time via eta) and crashes -> instead estimate the BC92 frontier on
*   inputs, then recover TFP-by-year as the year-mean output residual
*   (index-number TFP with SFA elasticities), base 2005.
*   sfpanel ln_y_s ln_L_s ln_K_s ln_M_s, model(bc92) distribution(tnormal) vce(cluster countyid)
* Runs CLEANED and RAW arms; writes tfp_fe_{tag}.csv (year, gamma) for 05_plot.
*--------------------------------------------------------------------
clear all
set more off

program define _bc92arm
    args ytag gvp lab cap inter outcsv
    import delimited "E:/TFP_weather/src/clean/county_panel_clean.csv", clear varnames(1) case(preserve)
    quietly destring countyid year SID, replace force
    gen double ln_y_s = ln(`gvp')
    gen double ln_L_s = ln(`lab')
    gen double ln_K_s = ln(`cap')
    gen double ln_M_s = ln(`inter')
	winsor2 ln_y_s, replace cuts(1 99)
	winsor2 ln_L_s, replace cuts(1 99)
	winsor2 ln_K_s, replace cuts(1 99)
	winsor2 ln_M_s, replace cuts(1 99)
    keep if ln_y_s<. & ln_L_s<. & ln_K_s<. & ln_M_s<.
    quietly summarize ln_y_s
    quietly replace ln_y_s = ln_y_s - r(mean)
    quietly summarize ln_L_s
    quietly replace ln_L_s = ln_L_s - r(mean)
    quietly summarize ln_K_s
    quietly replace ln_K_s = ln_K_s - r(mean)
    quietly summarize ln_M_s
    quietly replace ln_M_s = ln_M_s - r(mean)
    xtset countyid year
    di "==== BC92 arm: `ytag'  (N=" _N ") ===="
    sfpanel ln_y_s ln_L_s ln_K_s ln_M_s, model(bc92) distribution(tnormal) vce(cluster countyid)
**# Bookmark #1
    scalar bL = _b[ln_L_s]
    scalar bK = _b[ln_K_s]
    scalar bM = _b[ln_M_s]
    di "RTS_`ytag' = " %5.3f (bL+bK+bM) "  bL=" %5.3f bL " bK=" %5.3f bK " bM=" %5.3f bM
    capture predict te_`ytag', bc
    quietly summarize te_`ytag'
    di "meanTE_`ytag' = " %5.3f r(mean)
    * TFP-by-year = year-mean output residual (technical change + efficiency), base 2005
    gen double tfp_resid = ln_y_s - bL*ln_L_s - bK*ln_K_s - bM*ln_M_s
    preserve
    collapse (mean) tfp_resid, by(year)
    quietly summarize tfp_resid if year==2005
    gen double gamma = tfp_resid - r(mean)
    keep year gamma
    sort year
    export delimited "`outcsv'", replace
    di "WROTE `outcsv'"
    restore
end

_bc92arm cleaned real_gvp Laborday_impute capital_serv_q Inter_all_real "E:/TFP_weather/src/clean/tfp_fe_cleaned.csv"
_bc92arm raw real_gvp_raw Laborday_impute_raw capital_serv_q_raw Inter_all_real_raw "E:/TFP_weather/src/clean/tfp_fe_raw.csv"

di "ALL_DONE_BC92"
