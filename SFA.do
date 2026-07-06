 
 
 cd /Users/zhangxinzhen/Desktop/TFP/Code/output
 
 import delimited "climate_tfp_irrigation_panel_1982_2015.csv", clear
 xtset countyid year
 
 
 gen any_invalid = ///
    missing(gvp_real_wanyuan)      | gvp_real_wanyuan      <= 0 | ///
    missing(laborday_impute)  | laborday_impute  <= 0 | ///
    missing(land_serv_q_wanyuan)      | land_serv_q_wanyuan      <= 0 | ///
    missing(capital_serv_q_wanyuan)   | capital_serv_q_wanyuan   <= 0 | ///
    missing(inter_all_real_wanyuan)   | inter_all_real_wanyuan   <= 0

tab any_invalid
drop if any_invalid == 1
drop any_invalid
	

		
/*winsor2 gvp_real_wanyuan laborday_impute inter_all_real_wanyuan capital_serv_q_wanyuan land_serv_q_wanyuan, cuts(1 99) trim replace

drop if missing(gvp_real_wanyuan, laborday_impute, inter_all_real_wanyuan, ///
    capital_serv_q_wanyuan, land_serv_q_wanyuan)
	
	
summarize gvp_real_wanyuan laborday_impute capital_serv_q_wanyuan inter_all_real_wanyuan, detail
*/





*--------------------------------------------------*
* Province-year subgroup distributions
* One variable = one multi-page PDF
* Each subplot = one province
* Each box = distribution within province-year
*--------------------------------------------------*

































* 分年分省 trim 极端值
winsor2 gvp_real_wanyuan laborday_impute inter_all_real_wanyuan ///
    capital_serv_q_wanyuan land_serv_q_wanyuan, ///
    cuts(1 99) trim replace by(year countyid)

* 删除被 trim 成 missing 的观测
drop if missing(gvp_real_wanyuan, laborday_impute, inter_all_real_wanyuan, ///
    capital_serv_q_wanyuan, land_serv_q_wanyuan)


* 要看的变量
local vars gvp_real_wanyuan laborday_impute inter_all_real_wanyuan ///
           capital_serv_q_wanyuan land_serv_q_wanyuan

* 先看描述统计
foreach v of local vars {
    di "=============================="
    di "`v'"
    summarize `v', detail
}

* 水平值分布
local i = 1
foreach v of local vars {
    histogram `v', percent kdensity ///
        title("`v'", size(small)) ///
        xtitle("") ytitle("Percent") ///
        graphregion(color(white)) ///
        name(h_level_`i', replace)
    local ++i
}

graph combine h_level_1 h_level_2 h_level_3 h_level_4 h_level_5, ///
    cols(2) graphregion(color(white)) ///
    title("Distribution of Core Variables: Levels")

graph export "distribution_core_variables_level.png", replace width(2400)


* log值分布
foreach v of local vars {
    gen ln_`v' = ln(`v')
}

local i = 1
foreach v of local vars {
    histogram ln_`v', percent kdensity ///
        title("ln(`v')", size(small)) ///
        xtitle("") ytitle("Percent") ///
        graphregion(color(white)) ///
        name(h_log_`i', replace)
    local ++i
}

graph combine h_log_1 h_log_2 h_log_3 h_log_4 h_log_5, ///
    cols(2) graphregion(color(white)) ///
    title("Distribution of Core Variables: Logs")

graph export "distribution_core_variables_log.png", replace width(2400)
	
	

gen ln_S = ln(land_serv_q_wanyuan)

gen ln_y_s = ln(gvp_real_wanyuan)      - ln_S
gen ln_L_s = ln(laborday_impute)  - ln_S
gen ln_K_s = ln(capital_serv_q_wanyuan)   - ln_S
gen ln_M_s = ln(inter_all_real_wanyuan)   - ln_S

label var ln_S   "ln quality-adjusted real land service"
label var ln_y_s "ln real output per land service"
label var ln_L_s "ln labor days per land service"
label var ln_K_s "ln capital service per land service"
label var ln_M_s "ln intermediate inputs per land service"

sum ln_y_s ln_L_s ln_K_s ln_M_s ln_S, detail


*--------------------------------------------------*
*  SFA Estimation
*--------------------------------------------------*

sfpanel ln_y_s ln_L_s ln_K_s ln_M_s ib2005.year, model(bc92) distribution(tnormal) vce(cluster countyid) nolog

estimates store SFA_BC92_Chen_landserv

gen byte sfa_sample = e(sample)
label var sfa_sample "Sample used in BC92 SFA estimation"

matrix list e(b)


*--------------------------------------------------*
*  Predict frontier, inefficiency, and efficiency
*--------------------------------------------------*
* xb = alpha + beta_L ln(L/S) + beta_K ln(K/S)
*      + beta_M ln(M/S) + year effects
predict xb_s if sfa_sample, xb

* u_hat = E[u | epsilon]，对应 Chen 公式中的 u 的预测值
predict u_hat if sfa_sample, u

* JLMS efficiency = exp(-E[u|epsilon])
predict te_jlms if sfa_sample, jlms

* Battese-Coelli efficiency = E[exp(-u)|epsilon]
predict te_bc if sfa_sample, bc

label var xb_s    "Linear prediction from BC92 SFA"
label var u_hat   "Predicted technical inefficiency, E[u|epsilon]"
label var te_jlms "Technical efficiency, JLMS exp(-E[u|epsilon])"
label var te_bc   "Technical efficiency, BC E[exp(-u)|epsilon]"

*--------------------------------------------------*
* 7. Extract alpha + year effects
*--------------------------------------------------*
* xb_s 里面包含：alpha + year effects + input contribution
* 扣掉投入项，只留下 alpha + lambda_t
gen input_part_s = ///
    _b[ln_L_s] * ln_L_s + ///
    _b[ln_K_s] * ln_K_s + ///
    _b[ln_M_s] * ln_M_s if sfa_sample

gen ln_frontier = xb_s - input_part_s if sfa_sample
label var ln_frontier "Frontier technology term: alpha + year effects"

* 检查 ln_frontier 是否只随年份变化
bys year: egen sd_frontier = sd(ln_frontier) if sfa_sample
sum sd_frontier if sfa_sample
drop sd_frontier

*--------------------------------------------------*
* 8. Construct Chen-style SFA-TFP
*--------------------------------------------------*
* Chen 公式：lnTFP = alpha + lambda_t - u_it
gen ln_tfp_chen = ln_frontier - u_hat if sfa_sample
gen tfp_chen    = exp(ln_tfp_chen)     if sfa_sample

label var ln_tfp_chen "Log SFA-TFP, Chen-style, alpha+year-u"
label var tfp_chen    "SFA-TFP, Chen-style"

sum ln_tfp_chen tfp_chen, detail


histogram ln_tfp_chen, normal kdensity
histogram tfp_chen, normal kdensity

* 确认面板设定
xtset countyid year

* 相邻年份 TFP 指数
gen tfp_index_chen = exp(ln_tfp_chen - L.ln_tfp_chen) if sfa_sample

* 相邻年份 TFP 增长率
gen tfp_growth_chen = tfp_index_chen - 1 if sfa_sample

label var tfp_index_chen "Year-to-year SFA-TFP index, Chen-style"
label var tfp_growth_chen "Year-to-year SFA-TFP growth rate, Chen-style"

sum tfp_index_chen tfp_growth_chen, detail


gen tfp_growth_pct_chen = 100 * tfp_growth_chen
label var tfp_growth_pct_chen "Year-to-year SFA-TFP growth rate (%)"

* 提取每个县1982年的lnTFP
bysort countyid (year): gen temp_lntfp_1982 = ln_tfp_chen if year == 1982 & sfa_sample

* 把1982年的lnTFP填充到该县所有年份
bysort countyid: egen ln_tfp_1982 = max(temp_lntfp_1982)

* 构造1982年为基期的累计TFP指数
gen tfp_base1982 = exp(ln_tfp_chen - ln_tfp_1982) if sfa_sample & !missing(ln_tfp_1982)

label var tfp_base1982 "Cumulative SFA-TFP index, 1982=1"

sum tfp_base1982, detail







*===============================================================
* Figure: China Agricultural SFA-TFP, 1982-2016
* Top panel:    TFP index (1982=1) with 95% CI band
* Bottom panel: Year-on-year growth rate (%)
* Style: economics top-journal grade
*===============================================================

*--- Aggregate to national level by year -----------------------
preserve

* 国家层面年度统计:TFP指数和增长率的均值 + 标准误
collapse (mean) tfp_idx = tfp_base1982 ///
                tfp_grw = tfp_growth_pct_chen ///
         (sd)   sd_idx  = tfp_base1982 ///
                sd_grw  = tfp_growth_pct_chen ///
         (count) n_idx  = tfp_base1982 ///
                 n_grw  = tfp_growth_pct_chen, ///
         by(year)

* 95% 置信带(用 1.96 * SE)
gen se_idx = sd_idx / sqrt(n_idx)
gen se_grw = sd_grw / sqrt(n_grw)

gen lb_idx = tfp_idx - 1.96 * se_idx
gen ub_idx = tfp_idx + 1.96 * se_idx
gen lb_grw = tfp_grw - 1.96 * se_grw
gen ub_grw = tfp_grw + 1.96 * se_grw

*--- Set up academic plot scheme -------------------------------
* 用 plotplain 或 s2color, 顶刊偏好简洁
set scheme s2color

*--- Top panel: TFP index --------------------------------------
twoway ///
    (rarea ub_idx lb_idx year, ///
        color(navy%15) lwidth(none)) ///
    (line tfp_idx year, ///
        lcolor(navy) lwidth(medthick) lpattern(solid)), ///
    yline(1, lcolor(gs10) lpattern(dash) lwidth(thin)) ///
    ytitle("TFP Index (1982 = 1)", size(medsmall) margin(small)) ///
    xtitle("") ///
    xlabel(1982(5)2016, labsize(small) grid glcolor(gs15) glwidth(vthin)) ///
    ylabel(, labsize(small) angle(horizontal) format(%4.2f) ///
        grid glcolor(gs15) glwidth(vthin)) ///
    legend(off) ///
    title("Panel A: TFP Index (1982 = 1)", ///
        size(medsmall) color(black) position(11) ring(0) justification(left)) ///
    graphregion(color(white) margin(medsmall)) ///
    plotregion(color(white) margin(zero) lcolor(black) lwidth(thin)) ///
    name(panel_a, replace) ///
    nodraw

*--- Bottom panel: growth rate ---------------------------------
twoway ///
    (rarea ub_grw lb_grw year, ///
        color(maroon%15) lwidth(none)) ///
    (line tfp_grw year, ///
        lcolor(maroon) lwidth(medthick) lpattern(solid)) ///
    (scatter tfp_grw year, ///
        mcolor(maroon) msymbol(circle) msize(vsmall)), ///
    yline(0, lcolor(gs10) lpattern(dash) lwidth(thin)) ///
    ytitle("Annual Growth Rate (%)", size(medsmall) margin(small)) ///
    xtitle("Year", size(medsmall)) ///
    xlabel(1982(5)2016, labsize(small) grid glcolor(gs15) glwidth(vthin)) ///
    ylabel(, labsize(small) angle(horizontal) format(%3.1f) ///
        grid glcolor(gs15) glwidth(vthin)) ///
    legend(off) ///
    title("Panel B: Year-on-Year Growth Rate", ///
        size(medsmall) color(black) position(11) ring(0) justification(left)) ///
    graphregion(color(white) margin(medsmall)) ///
    plotregion(color(white) margin(zero) lcolor(black) lwidth(thin)) ///
    name(panel_b, replace) ///
    nodraw

*--- Combine ---------------------------------------------------
graph combine panel_a panel_b, ///
    cols(1) ///
    xsize(7) ysize(8) ///
    graphregion(color(white) margin(medlarge)) ///
    iscale(*1.0) ///
    name(tfp_combined, replace)

*--- Export ----------------------------------------------------
graph export "fig_tfp_china_1982_2016.pdf", ///
    replace as(pdf)
graph export "fig_tfp_china_1982_2016.png", ///
    replace as(png) width(2400)

restore

di "Figure saved to: fig_tfp_china_1982_2016.pdf and .png"

winsor2 tfp_growth_pct_chen, cuts(1 99) trim replace


gen SM2 = rzsm_gs*rzsm_gs

gen GDD2 = gdd_growing_season*gdd_growing_season

reghdfe tfp_growth_pct_chen rzsm_gs SM2 gdd_growing_season GDD2, absorb(countyid year) vce(cluster countyid)











