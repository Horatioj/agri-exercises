# Methodology
### China county-level agricultural TFP × weather, 1981–2016

Everything about *how* the numbers are made lives here. [README.md](README.md) is the
pipeline map (what to run, in what order).

| § | |
|---|---|
| 1–8 | frontier framework: DEA and SFA as complementary approaches |
| 9 | data cleaning of the county input–output panel (in Chinese) |
| 10 | weather-variable construction: GDD/HDD, growing season, soil moisture |
| 11 | figure captions: single-year frontier and isoquant figures |

---

# Part I — Frontier framework: DEA and SFA as complementary approaches

---

## 1. Conceptual framework

Let county *i* in year *t* use inputs **x**<sub>it</sub> = (L, Land, K, M) to produce output
y<sub>it</sub> (real GVP). The **production technology** is

&nbsp;&nbsp;&nbsp;&nbsp;T<sup>t</sup> = { (**x**, y) : **x** can produce y in period t }

with output-oriented **distance function** D<sup>t</sup>(**x**,y) = y / f<sup>t</sup>(**x**) ≤ 1,
where f<sup>t</sup>(**x**) is the maximum feasible output (the frontier). D = 1 means the
county is *on* the frontier; D < 1 measures technical inefficiency.

**TFP change decomposes into two economically distinct sources:**

&nbsp;&nbsp;&nbsp;&nbsp;**ΔTFP = frontier shift (technical change) × movement toward the frontier (efficiency change)**

This decomposition is the analytical core of the paper. A county can raise TFP either
because the *technology* improved (new seeds, mechanization, infrastructure) or because
it *caught up* to best practice (better management, institutions, incentives). Chinese
agricultural reform plausibly operated through **both** channels, and separating them is
the reason a frontier approach is preferred to a simple Solow residual — the latter
conflates the two and, under a within estimator, is contaminated by biased input
elasticities (here FE-RTS ≈ 0.39, far below the ≈ 1 that CRS requires).

---

## 2. Why DEA and SFA are complementary (not competing)

The two methods make **opposite trade-offs**. Neither dominates; agreement between them
is the credibility check, and disagreement is itself informative.

| | **DEA** (nonparametric) | **SFA** (parametric) |
|---|---|---|
| Frontier form | None assumed — piecewise-linear envelope of the data | Cobb-Douglas (or translog) imposed |
| Noise | **No** stochastic term: all deviation = inefficiency | Composed error v − u **separates noise from inefficiency** |
| Weather/outliers | Bad year ⇒ mismeasured as inefficiency | Bad year absorbed into v (noise) |
| Estimation | Linear programming, deterministic | Maximum likelihood, distributional assumptions on u |
| Statistical inference | Bootstrap needed | Standard errors, LR tests |
| Sensitivity | To outliers & dimensionality (curse of dimensionality) | To functional form & distribution of u |
| Handles many DMUs | Yes (LP scales) | Convergence problems with large N (incidental parameters) |

**The complementarity in one line:** DEA is flexible about *technology* but fragile about
*noise*; SFA is robust to *noise* but restrictive about *technology*. Reporting both
brackets the truth. In this application the contrast is sharp and diagnostic:

- **DEA** yields TFP growth of **+4.60%/yr** (output-weighted) and, because it treats all
  deviation as inefficiency, retains large cross-county efficiency dispersion.
- **SFA** (pooled, land-normalized) yields **+2.89%/yr** but collapses to a *neutral
  technical-change* path because of **wrong skewness** (see §5): after rigorous cleaning
  the residuals are symmetric (within-county skewness +0.04), so no one-sided inefficiency
  component is identified and σ<sub>u</sub> → 0.
- The **gap between them (~1.7 pp/yr) is precisely the efficiency-change component** that
  DEA attributes to catch-up and SFA does not identify. This is not a contradiction: it is
  the decomposition made visible by using both methods.

*(Sample note. Those figures are from `30_aggregate_tfp.py` on the UNION sample. The
agricultural-county rule has since been tightened to cropland ≥ 15% in EVERY raster year
(1,975 → **1,718 counties**, README §1), and the estimation window now starts in **1985**,
so both numbers await re-estimation. What is already measured on the new footing: DEA TFP
growth is **4.24 %/yr** on the full sample and **4.15 %/yr** with the sixteen audited
leverage counties removed — see §8b.)*

---

## 3. DEA: sequential-NIRS Malmquist index

### 3.1 Frontier specification
- **Output-oriented**: how much more output is feasible with the same inputs (appropriate
  where farmers face quasi-fixed land and quotas rather than free input choice).
- **NIRS** (non-increasing returns to scale, Σλ ≤ 1): allows decreasing returns at large
  scale without imposing full CRS. Unrestricted VRS over-fits with 2,500 DMUs; CRS was
  rejected empirically (a CRS frontier gave a nonsensical TFP path driven by a handful of
  extreme counties, mean efficiency 0.08).
- **Sequential (non-regressive) frontier**: the reference set for year *t* is *all*
  observations up to *t*. This rules out "technical regress", which in this data is a
  measurement artifact (weather-driven bad years) rather than lost knowledge — the
  standard fix in agricultural applications.

### 3.2 The Malmquist index
For adjacent years a → b (Färe, Grosskopf, Norris & Zhang 1994):

&nbsp;&nbsp;&nbsp;&nbsp;M = [ (D<sup>a</sup>(x<sub>b</sub>,y<sub>b</sub>) / D<sup>a</sup>(x<sub>a</sub>,y<sub>a</sub>)) × (D<sup>b</sup>(x<sub>b</sub>,y<sub>b</sub>) / D<sup>b</sup>(x<sub>a</sub>,y<sub>a</sub>)) ]<sup>1/2</sup>

the geometric mean of the index evaluated against period-a and period-b technology
(avoiding an arbitrary base-year choice). **M > 1 = TFP growth.** It decomposes as

&nbsp;&nbsp;&nbsp;&nbsp;**M = EFFCH × TECHCH**

&nbsp;&nbsp;&nbsp;&nbsp;EFFCH = D<sup>b</sup>(x<sub>b</sub>,y<sub>b</sub>) / D<sup>a</sup>(x<sub>a</sub>,y<sub>a</sub>) &nbsp;&nbsp; *(catch-up: movement toward the frontier)*

&nbsp;&nbsp;&nbsp;&nbsp;TECHCH = [ (f<sup>b</sup>(x<sub>b</sub>)/f<sup>a</sup>(x<sub>b</sub>)) × (f<sup>b</sup>(x<sub>a</sub>)/f<sup>a</sup>(x<sub>a</sub>)) ]<sup>1/2</sup> &nbsp;&nbsp; *(frontier shift: innovation)*

Under a sequential frontier TECHCH ≥ 1 by construction, so **all measured TFP declines are
efficiency losses**, which is the economically meaningful reading (a drought pushes
counties away from best practice; it does not destroy the technology).

Each requires four LPs per county-year: f<sup>a</sup>(x<sub>a</sub>), f<sup>b</sup>(x<sub>b</sub>),
f<sup>a</sup>(x<sub>b</sub>), f<sup>b</sup>(x<sub>a</sub>). Implemented in Python
(`src/20_dea_tfp.py`) because ~350k LPs are infeasible in Stata's ado-based DEA;
all **econometric analysis** of the resulting indices is done in Stata.

---

## 4. SFA: NIRS stochastic frontier

### 4.1 Returns to scale — matched to DEA, not imposed as CRS
The SFA now carries the **same returns-to-scale assumption as §3: NIRS**. Earlier drafts
imposed **CRS by land-normalisation** (dividing output and every input by land service)
while §3 solved the DEA LP under Σλ ≤ 1. NIRS *permits* decreasing returns and only rules
out increasing returns, so the two frontiers were answering different questions and the
gap fell straight into measured TFP.

NIRS is an **inequality**, so it is handled by the KKT argument rather than imposed:

1. estimate **unrestricted** (variable returns);
2. read RTS off the fit — with the logs mean-centred it is the sum of the four
   first-order coefficients, which is exact for Cobb-Douglas;
3. **RTS ≤ 1 → the constraint is slack**, and the unrestricted fit *is* the NIRS fit;
4. **RTS > 1 → it binds**, and the optimum sits on the boundary RTS = 1, applied by
   land-normalisation (`sfpanel` refuses `constraints()` together with `vce(cluster)`,
   and clustered SEs over ~1,700 counties are worth more than writing the restriction out).

On every run so far step 3 fires: RTS ≈ 0.44–0.49, far inside NIRS, so the CRS branch has
never been used.

&nbsp;&nbsp;&nbsp;&nbsp;ln y<sub>it</sub> = α + β<sub>L</sub> l̃n L + β<sub>S</sub> l̃n S + β<sub>K</sub> l̃n K + β<sub>M</sub> l̃n M + λ<sub>t</sub> + v<sub>it</sub> − u<sub>it</sub>

with v ~ N(0, σ<sub>v</sub><sup>2</sup>) noise, u ≥ 0 inefficiency, and l̃n x mean-centred so
the first-order coefficients read as elasticities *at the sample mean*. Year dummies
λ<sub>t</sub> (base 2005) trace **neutral technical change**; SEs clustered by county.
Land is now estimated **directly** (`cS`) rather than recovered as the 1 − β<sub>L</sub> −
β<sub>K</sub> − β<sub>M</sub> residual it was under normalisation, and `_tfp.BETA_AG` reads
the coefficients live from `sfa_surface_coefs.csv`.

**Translog was tried and dropped.** A 14-term specification (4 squares + 6
cross-products) stuck at log-likelihood −44,403 for twelve straight iterations, flagged
`(not concave)` throughout and never clearing, while the Cobb-Douglas run converged by
iteration 18. Point-varying elasticities are pursued outside the BC92 estimation.

### 4.2 TFP construction (Chen-style)
&nbsp;&nbsp;&nbsp;&nbsp;**ln TFP<sub>it</sub> = (α + λ<sub>t</sub>) − u<sub>it</sub>**

i.e. the technology level net of inefficiency — the direct SFA analogue of
M = TECHCH × EFFCH. Input contributions are stripped out so TFP is not mechanically
driven by input growth.

### 4.3 Panel models and the identification problem
| Model | Heterogeneity control | Outcome on this panel |
|---|---|---|
| Pooled `frontier` | none | Converges instantly; **σ<sub>u</sub> → 0** (wrong skewness) |
| `sfpanel, bc92` | none (persistent heterogeneity → u) | **Converges.** Last completed run (1,959 counties, sixteen leverage counties excluded, N=60,389, 15 iterations, ~27 min): RTS=0.4864, σ²<sub>u</sub>=0.2920, σ²<sub>v</sub>=0.0796, γ=0.786, η=+0.0001 (p=0.86), mean TE=0.235. On the full union sample (1,975 counties): RTS=0.4549, mean TE=0.123. **Both predate the strict 1,718-county sample and need re-running.** |
| `sfpanel, tfe` | full county FE (Greene 2005) | **Never converges** (incidental parameters, 2,505 nuisance parameters) |

**Key methodological point:** BC92 is a *panel time-decay* model — it identifies u from the
**persistent panel component**, not from the skewness of the composed error. Hence it
recovers σ<sub>u</sub> > 0 even where pooled SFA cannot. Its efficiency *levels* are
implausibly low because without a heterogeneity term persistent county differences are
mislabelled as inefficiency; and η is statistically zero (+0.0001, p = 0.86), so the model
has degenerated to **time-invariant** inefficiency — what it calls inefficiency is a county
fixed effect. **Therefore: use BC92 for TFP growth, not for efficiency levels.**

**Why RTS ≈ 0.45 rather than the ≈ 0.85 the pooled data show.** BC92 puts persistent county
heterogeneity into u<sub>i</sub>, so the slopes are identified from WITHIN-county variation.
On the same sample, pooled OLS gives RTS 0.844 and the between estimator 0.859, but the
within estimator gives **0.483 with a negative labour coefficient**. The reason is
measurable: only **13.3%** of labour's variance and **16.8%** of land's is within-county, so
at that signal-to-noise ratio classical measurement error attenuates both toward zero.
Capital is the opposite problem — 61.6% within, but its year-on-year change correlates
**+0.03** with farm machinery horsepower and **+0.10** with intermediates, i.e. it carries
almost no county-specific annual signal, which is why β<sub>K</sub> is indistinguishable
from zero in every specification.

---

## 5. Diagnostics that discipline the choice

1. **Wrong skewness (Green & Mayes 1991).** OLS residual skewness is **+1.19** (should be
   negative for a production frontier). Within-county+year it is **+0.04** — essentially
   symmetric. Interpretation: the data cleaning removed the one-sided negative-skew
   "shortfall" that identifies inefficiency in cross-section; residual deviations are
   two-sided noise. This is a *property of the cleaned data*, not a solver failure, and it
   is why pooled/TFE SFA cannot identify inefficiency here while DEA (which assumes all
   deviation is inefficiency) still can.
2. **Distributional shape.** In **levels** the I-O variables are extremely leptokurtic
   (kurtosis 7–700); in **logs** they are close to Gaussian (output: skew −0.56,
   kurtosis 3.34). This justifies the log-linear (Cobb-Douglas) frontier and confirms the
   cleaning worked — log-kurtosis is stable across all five reform stages (3.4–3.8).
3. **Volatility decomposition.** National annual TFP swings of ±20% are *not* an averaging
   artifact (the aggregate national-totals index is equally volatile, sd ≈ 8.5) and are not
   removed by winsorizing (8.82 → 8.83). They are a **common annual factor**: a single
   national deflator (sd 7.7%/yr) plus noisy input series (corr(Δinput, ΔTFP) = −0.62).
   Smoothed to the frequency the literature reports, the series has sd ≈ 3.1% — inside the
   published ±2–5% band, with the mean unchanged.

---

## 6. Aggregation from county to national

County indices are aggregated with **output-share Törnqvist weights**

&nbsp;&nbsp;&nbsp;&nbsp;w<sub>it</sub> = ½ ( s<sub>i,t−1</sub> + s<sub>i,t</sub> ),&nbsp;&nbsp; s<sub>it</sub> = GVP<sub>it</sub> / Σ<sub>j</sub> GVP<sub>jt</sub>

the standard growth-accounting aggregator (averaging adjacent-year shares prevents any
single year dominating). Equal weighting would give a tiny Tibetan county the same weight
as a large Henan grain producer. Weighting **matters for DEA** (+3.21% → +3.97%/yr: larger
counties grew faster) and is **irrelevant for the degenerate pooled SFA** (TFP common
across counties). County growth rates are winsorized 1/99 **within year** before
aggregation.

---

## 7. Empirical design and the five reform stages

Results are organized around the standard periodization of Chinese agricultural reform
(Lin 1992; Huang & Rozelle; Jin, Huang, Rozelle & Rosegrant; Gong 2018):

| Stage | Years | Driver |
|---|---|---|
| 1 Reform take-off | 1978–1984 | Household Responsibility System, procurement price rises |
| 2 Stagnation | 1985–1988 | One-off HRS gains exhausted; 1985 grain output fell |
| 3 Recovery | 1989–1996 | Price liberalization, TVE boom, input/technology diffusion |
| 4 Adjustment | 1997–2003 | Grain glut and deflation, low prices, rural burden, WTO |
| 5 Subsidy era | 2004–2016 | Ag-tax abolition, four subsidies, minimum prices, mechanization |

The expected TFP signature is **high → low → moderate → flat → high**, and all three
estimators (DEA, SFA, Solow-CRS) reproduce it — the strongest available validation that
the cleaned panel carries real economic signal.

---

## 8. Implementation map

| Step | Script | Engine |
|---|---|---|
| Cropland share → ag-county definition | `src/00_cropland_share.py` | Python (raster) |
| Data cleaning | `src/00`–`src/04` | Python |
| Weather variables | `src/10`–`src/14` | Python (raster) |
| DEA sequential-NIRS Malmquist + EFFCH/TECHCH | `src/20_dea_tfp.py` | Python (LP) |
| SFA BC92, Cobb-Douglas, NIRS | `src/21_sfa_bc92.do` | **Stata** |
| Descriptive statistics & moments | `src/22_descriptives.do` | **Stata** |
| I-O + index descriptives, plotted ranges, scale check | `src/24_io_index_descriptives.py` | Python |
| Framework analysis, decomposition & comparison tables | `src/23_dea_sfa_framework.do` | **Stata** |
| Aggregation & stage analysis | `src/30_aggregate_tfp.py` | Python |
| Volatility diagnostics | `src/31_volatility.py` | Python |
| CPS four-component decomposition | `src/40`, `src/41` | Python (LP) |
| Frontier & isoquant figures (one engine) | `src/52_frontier_figs.py` + `src/_frontier.py` | Python |
| Weather figures incl. the f(x, w) surface | `src/56_weather_figs.py` | Python |
| Frontier composition audit | `src/59_frontier_audit.py` | Python (LP) |
| Leave-one-out sensitivity | `src/60_leave_one_out.py` | Python (LP) |
| Screening dashboards (HTML maps) | `src/61_screening_dashboard.py` | Python |

Econometric estimation and inference are done in Stata; linear programming (DEA) and
data engineering in Python.

The input index, the winsorizing rule, the elasticities, the five reform stages and the
period cuts live in `src/_tfp.py`; **every frontier estimator** lives in `src/_frontier.py`.
Both are imported everywhere, so a change cannot apply to one table of the paper and not
another. Three scripts (`52_`, `53_`, `58_`) once held private copies of the same boundary
and had silently drifted apart — that is why the estimators were centralised.

**One naming correction that matters for the text.** What the code builds is a
**fixed-weight geometric (Cobb-Douglas) index**, not a Törnqvist. A Törnqvist is a chained
bilateral index between adjacent periods using period-specific cost shares averaged across
the two; this is a cross-sectional level index with one constant weight vector from a
single BC92 estimate. They coincide only if the shares do not move over time.

---

## 8b. How much of this rests on a handful of counties

`59_frontier_audit.py` and `60_leave_one_out.py` answer that directly, and the answer
disciplines which numbers can be quoted.

| | full sample | 16 leverage counties removed | change |
|---|---|---|---|
| DEA TFP growth | 4.24 %/yr | 4.15 %/yr | **−2.1%** |
| DEA mean efficiency | 0.187 | 0.228 | **+22%** |
| SFA mean efficiency | 0.123 | 0.235 | **+91%** |
| EFFCH | −1.72 %/yr | −1.04 %/yr | **−39%** |
| TECHCH | +6.01 %/yr | +5.24 %/yr | **−13%** |

**Growth is robust; levels and the decomposition are not.** The Malmquist index is a ratio
between adjacent years, so a frontier held up by the same counties in both years cancels
out. The EFFCH/TECHCH split does not cancel: its two halves move 39% and 13% in opposite
directions and nearly offset, which is why the total moves only 2%. The split is the
headline advantage of DEA over SFA in §2 and it is the number most exposed here.

Who these counties are: peri-urban districts (丰台, 闵行, 青浦, 江干, 普陀 …) and
island/coastal specialty counties (长岛, 洞头, 岱山, 荣成 …). Their values are **not**
data errors — the series are smooth in time and internally consistent — they are genuinely
specialised units whose output-per-input ratio sits at the 99.8th percentile. The strict
cropland rule removes 6 of the 16; the rest clear ≥15% cropland every year and can only be
separated by county-type indicators, not by a cropland threshold.

---

# Part II — Data

---

## 9. 县级农业投入产出数据清洗

本文件汇总县级农业投入产出面板（GVP、劳动、土地、资本、中间投入，1981–2016）的清洗流程与边界案例。
代码脚本流水线：`00_build_base_panel.py` → `01_resolve_roster.py`（县码/县名订正）
→ `02_apply_resolution.py`（生成订正后原始数据）→ `03_clean_panel.py`（数值清洗 + 自动 spike 插值
+ 人工多年插值，同一次运行内完成）。每一步的改动都保留 `<var>_raw` 原值、`<var>_imp` 标记，可逆。

---

### 9.0 数据合并（`00_build_base_panel.py`，按 countyid+year）
| 变量 | 来源 |
|---|---|
| 名义 GVP `GVP_allagr_impute` | `agr_GVP.dta` |
| 劳动工日 `Laborday_impute` | `cty_prod_account_agg_forAXN.dta` |
| 劳动人数 `agr_labor_num` | `agr_labor.dta` |
| 土地/资本/中间投入 `Land_serv_q` `capital_serv_q` `Inter_all_real`(+`Inter_all_nom`) | `cty_prod_account_agg_forAXN.dta` |

### 9.1 县级单位名称与县级代码清洗（`01_resolve_roster.py`，权威表 `_county_corrections.py`）
- **更名合并（无重叠，拼接为一条连续序列）**：通什市→五指山市、波阳县→鄱阳县、孝县→孝义市、郎县(错字)→朗县 等。
- **行政合并（有重叠年份，`02_apply_resolution.py` 按年**求和**为一致口径）**：宁冈县 2000 年并入井冈山市 → 两码逐年相加。额尔古纳左旗、牙克石市；额尔古纳右旗，额尔古纳市，根河市等行政区划名称的演变。
- **重复码（同一地方两码）**：横县、赤壁市、武穴市 → 保留一码、删除另一码，或按县名归一。
- **错误县名（typo）**：毫县→亳县、林高县→临高县（临高在 ag 表码 469024，故重编码）、杨凌区→杨陵区（码在 ag 表则以码为准采用 ag 名）。
- **删除**：地级市及其市辖区/市郊/市中区/“辖”（如 济南市中区、株洲市 430200、南昌市辖、马鞍山辖）；升格为地级市的原县级（随州、菏泽 372901、周口、驻马店、临汾、淮安、宁德、榆林、固原、通辽 152301、齐齐哈尔、呼和浩特、碑林区 610103）；金门县 350527；`_county_corrections.REMOVE` 内全部重复/无用码（含康县 621224，保留 622625）；县名全空的县。
- **保留不设区的市**（不设区地级市按县级处理）：441101 中山市、469003 儋州市、441900 东莞市。

### 9.2 筛选农村地区 + ag 建制区（`01_resolve_roster.py`）
- **口径**：所有**农村县级单位**（县/县级市/旗 等）**＋** 在耕地表中（2010年耕地面积$\approx \geq$ 15%）的**城区**（如萧山区、华州区、番禺区、杨陵区）。
- 城区默认删除，只有出现在 `ag_counties_crop15.csv`（2010 shapefile × 耕地≥15%）中的`区`才保留。
- 生产数据县码可能有误 → 以**县名（同省内）**匹配 ag 表的**正确码**；改市设区（番禺市→番禺区）用“同名前两字”建议、人工确认（见 `roster_resolution.csv`）。

### 9.3 数值清洗（`03_clean_panel.py` 第 3–6 步）
顺序：÷10000 订正 → 平减 → 4b/4c/4d 置 NA → 省级中位数 → spike 插值 → 全 NA 删除。

#### 9.3a ÷10000 录入错误订正（平减前）
某县某年名义 GVP 与中间投入**同时**约小 10000 倍（记成亿元而非万元）且次年恢复 → ×10000 还原。
- 例：**皋兰县 1993**（GVP 1→10000，real≈21784，邻年 29995/35077）；**福建 2004**（闽侯/连江/罗源/福清，GVP 15/38/12/55 → ×10000）。

#### 9.3b 置 NA 的三类错误值（4b）
- **=0**：大量级投入不应为 0。
- **<1**：这些量纲下 <1 不可信（如 久治县 land 全<1）。
- **相对本县中位数 <1/1000 的孤立小值**（低值后跳升）。

#### 9.3c 首/末端**多年**边界跳变（4c）
首（或末）**≤5 年**的短段，与稳定主体之间存在 **>5倍**的年际跳变、且该段值离跳变后核心中位数 **>3倍** → 置 NA。
- 例：**伊宁市 土地** 603/712/672/4227 然后 ~70000（1981–84 全 NA）；**临淄区** 1981–84；**玛多县 土地**边界高值。

#### 9.3d 跨县占位值（4d）
某（省, 年, 变量）**>50% 的县共用同一个数值** = 填充占位，非真实数据 → 置 NA。
- 例：**青海 资本服务 1981–84 各县完全相同**、1985 才跳变 → 1981–84 全省资本置 NA。

#### 9.3e Spike-and-revert 检测 + 省级趋势插值（第 6 步）
- **检测**：值上跳/下跳后在 1/2/3 年内回落（**single/double/triple**）。相邻同段年份差 <2倍视作平台（如**连南** 2005/06=128k/228k 相差1.77倍，回落于 2007 → double）。
- **不对称阈值**：**上跳（更可能是错误）阈值 3倍**；**下跳（可能是灾害/气候真实低产）阈值 5倍**。
- **插值方法**：把缺口年份**贴着省级中位数趋势**插值，两端锚定“最后正常年”与“下一正常年”：
  `ln C(t) = R(t) + [d0 + (d3-d0)·(t-t0)/(t3-t0)]`，R=同省其他县 ln 值的**中位数**。


### 9.4 统计口径变化：连续多年高位/低位（`03_clean_panel.py` 第 7 步，人工指定区间）
自动规则只处理会“回落”的 spike。**多年**的口径性异常（如 1997–2000 连续偏低）需人工识别：
1. 看 `src/figures/county_review/` 的 SVG（灰=原始、绿=清洗后、红点=已插值、紫虚线=省级中位数趋势）；county 偏离紫线一段又回归即候选。
2. 在 `data/manual_impute_list.csv` 填 `countyid,var,year_start,year_end`（var 大小写不敏感）。
3. 重跑 `python src/03_clean_panel.py` → 与自动步骤**同一省级趋势插值方法**填补，记录于 `manual_impute_log.csv`。
- 例：根河市/额尔古纳市/满洲里市 gvp 1996–2004 偏低 → 贴省趋势插值。

### 9.5 删除不可用县（4c 之后）
任一 I-O 变量**全期全 NA**的县无法使用 → 删除。

---

### 9.6 一些边界案例
| 案例 | 现象 | 处理 |
|---|---|---|
| 皋兰县1993 / 福建4县2004 | GVP、中间投入同时小 10000 倍后恢复 | ÷10000 订正 (3a) |
| 西藏久治县 土地 | 全序列 <1 | <1→NA (3b) |
| 伊宁市 / 临淄区 土地 | 开头几年极低后跳升 | 边界段裁剪 (3c) |
| 青海 资本 1981–84 | 各县同值后跳变 | 跨县占位→NA (3d) |
| 柘荣县 1983/2001 | 单年约 10 倍尖峰后回落（且落在口径年） | spike 插值 (3e)（不再因口径年而保留） |
| 广东连南 2005–06 | 两年高位（相差1.77倍）后回落 | double spike (3e) |
| 根河市/额尔古纳市/陈巴尔虎旗 1997–2004 | 连续多年偏低（口径性） | 人工省趋势插值 (4) |
| 遵义县 520321 | 全序列跳动、非单一 spike | **疑难**：建议人工核查/分段，或按需剔除 |

> 原始气候数据（栅格来源、坐标系、缺失值、已知问题）的清单见 `data/数据说明.md`。

---

## 10. Weather-variable construction: GDD / HDD and the growing season

How GDD and HDD are computed per county per year, and why the current method is what it is.
This section began as a review of the ORIGINAL degree-day step; the defects it identifies
have since been fixed (in `_degree_days.py` and `10_monthly_gdd_hdd.py`), but the reasoning
is kept because it is what justifies the present specification.

---

### 10.0 The finding that drove the rewrite

The **growing-season step is faithful to Ortiz-Bobea**; the **original degree-day step was
not**. It averaged the day before applying the temperature response, which (i) biased GDD
and (ii) made **HDD impossible to recover** — it was not computed anywhere in the pipeline.

---

### 10.1 The critical methodological problem

The original B step collapsed each day to its mean *first*, then applies a piecewise function:

```python
tmean = (tmax + tmin) / 2
gdd   = 0 if tmean < 8 else (24 if tmean > 32 else tmean - 8)
```

Two consequences:

**(a) Heat above 32 °C is capped into GDD, not separated out.** A day that peaks at 33 °C and one
that peaks at 45 °C both contribute `24`. The distinction between beneficial and harmful heat —
the entire reason for having GDD *and* HDD — is destroyed at source.

**(b) The within-day temperature path is discarded.** A day with `tmin=5, tmax=35` has
`tmean=20`, so it is scored as 12 beneficial degree-days and **zero** harmful exposure, even
though the crop spent several hours above 32 °C.

Ortiz-Bobea (following Schlenker & Roberts 2009, Snyder 1985) does the opposite: apply the
nonlinear response **within the day**, then integrate. Temperature is modelled as a sine wave
running between `tmin` and `tmax`,

&nbsp;&nbsp;&nbsp;&nbsp;`T(t) = M + A·sin(t)`, &nbsp; `M = (tmax+tmin)/2`, &nbsp; `A = (tmax−tmin)/2`

and the time above a threshold is integrated analytically. **Degree days above threshold `b`:**

| case | value |
|---|---|
| `tmax ≤ b` | `0` |
| `tmin ≥ b` | `M − b` |
| `tmin < b < tmax` | `(1/π)·[ (M−b)·(π/2 − θ) + A·cos θ ]`, &nbsp; `θ = arcsin((b−M)/A)` |

From that one primitive both variables follow, with bounds 8 °C and 32 °C:

> **GDD (beneficial) = DD(8) − DD(32)**
> **HDD (harmful)  = DD(32)**

Note the **subtraction**: degrees above 32 °C are moved *out* of the growth measure and *into*
the harm measure — they are not capped and silently retained.

##### Implementation + validation
`src/_degree_days.py` implements this and validates it. Unit checks pass exactly
(`tmax≤b → 0`; `tmin≥b → M−b`; both boundary cases → 0). Measured against the current method on
real China rasters (0.1°, 18 sampled days of 1982):

| | current (daily mean) | sine method |
|---|---|---|
| mean GDD/day | 5.344 | **5.582 (+4.5%)** |
| mean HDD/day | **0 by construction** | 0.025 |

The GDD bias is *not* uniform: it is small mid-summer (+0.6 to +1.8%) but large in the shoulder
seasons — **+43% (1 Jan), +48% (2 Mar), +23% (21 Jan)** — because on cool days with a wide diurnal
range the afternoon rises above 8 °C while the daily mean does not, so the current method scores
those days as zero growth. This systematically distorts the *start and end* of the growing season,
which is precisely where planting/harvest decisions bind.

On heat: **12 of 18 sampled days had pixels above 32 °C** (up to 23,463 pixels on 9 Aug 1982),
all of which the current method records as zero harmful exposure.

---

### 10.2 Blockers that had to be cleared (all now fixed)

| # | Problem | Fix |
|---|---|---|
| 1 | `utils.py`: `DATA_ROOT = /Users/zhangxinzhen/Desktop/TFP/Climate data` (macOS) | point to `Z:/weather data` |
| 2 | Expects folders `01 temp_min_max/`, `04 NDVI_China/` | actual: `temperature/`, `NDVI_China/` |
| 3 | Expects **extracted** `1982_max/19820101_max.tif`; data ships as **`1982_max.zip`** (730 files each) | extract, or read from the zip (faster: open each zip once per year) |
| 4 | `get_temp_grid_info()` samples `1980_max/19800101_max.tif` — **1980 does not exist** (data starts 1981) → crashes at setup | sample the first available year |

Verified about the rasters themselves (these are all consistent with the code's assumptions):
0.1° grid, `EPSG:4326`, bounds 70–140 °E / 15–55 °N, 400×700, **units are °C** (July max 43.5 °C),
NoData `−3.4e38` — so the `SENTINEL = -1e37` masking is correct.

---

### 10.3 What was already correct (and is kept)

- **Growing season (step A) follows Ortiz-Bobea 2021 closely**: NDVI half-month climatology over
  35 years → wrap-padded to 30 periods → 7-period centred moving average → `argmax` = peak
  half-month → **5-month window = peak ± 2 months**. The wrap-padding and the `set`-based month
  parsing (never treating the list as a range) are both right.
- **County aggregation (step E)** uses `exact_extract(..., ops=['weighted_mean'], weights=cropland_weight)`
  on the matching grid — cropland-weighted area aggregation, which is the correct approach and
  matches the literature. Keeping GDD on the temperature grid and RZSM on the soil grid (never
  mixing) is also right.
- NoData/sentinel handling and the strict NaN-propagation rule are sound and documented.

---

### 10.4 Accurate GDD/HDD per county per year — the recipe, as implemented

**Step 1 — daily, per pixel (0.1° grid).** For each day, from `tmin`/`tmax` *(never from the mean,
and never from the `_avg` files)*:
```
dd8  = dd_above(tmin, tmax, 8)
dd32 = dd_above(tmin, tmax, 32)
gdd_day = dd8 - dd32          # beneficial, 8-32 C
hdd_day = dd32                # harmful,  > 32 C
```

**Step 2 — accumulate over the growing season.** Sum `gdd_day` and `hdd_day` over the days of the
zone's 5-month window (peak ± 2). Accumulate **both** variables over the *same* window, so they
are directly comparable in the regression. Keep the existing NaN rule.

**Step 3 — aggregate to county, cropland-weighted.** Exactly as step E does today:
```
exact_extract(gdd_raster, counties, ops=['weighted_mean'],
              weights=cropland_weight_<slice>_tempgrid.tif, include_cols=['PAC'])
```
using the LUCC slice mapped to that year. HDD uses the *same* weights and grid.

**Step 4 — join to the TFP panel** on the county code, with the same layered matching used in
`_county_match.py` (code → name → 曾用名 → stem), since production codes and boundary codes differ.

**Result:** `PAC × year × {gdd_growing_season, hdd_growing_season}`, both in °C·day,
cropland-weighted, over a phenologically-defined season — the Ortiz-Bobea specification, ready to
enter the TFP regression with the expected opposite signs (GDD `+`, HDD `−`).

### 10.5 Two judgement calls worth making explicitly
- **Thresholds.** 8/32 °C follows the AJAE paper. Schlenker–Roberts use 29 °C for US maize; the
  right upper bound is crop- and region-specific. Because `dd_above` is a primitive, you can emit
  several bounds cheaply and let robustness checks decide.
- **Season resolution.** The window is currently defined at **37-zone** resolution and in whole
  **calendar months**. Pixel-level `peak_month.tif` already exists, so a county-specific window is
  available at no extra cost and would remove within-zone phenology error — worth at least a
  robustness check.

---

# Part III — Figures

---

## 11. Figure captions — single-year frontier and isoquant figures

Years: **1986, 2000, 2015**, chosen on county-count density rather than fixed
endpoints (1,718 / 1,732 / 1,702 counties). 1981–84 were excluded: coverage there
is 489–818 counties, less than half the later years.

Each year gets its **own cross-sectional DEA frontier** over the counties observed
that year — these are standalone figures, not the sequential accumulating
technology used in the main CPS decomposition. Raw disaggregated inputs are used
throughout Figures A and B; the Törnqvist scalar appears only in C and D, where the
second axis is a weather variate.

All axes are **logarithmic**: output-per-input and input-per-output ratios span
three or more orders of magnitude across 1,800 counties, so a linear axis collapses
either the county cloud or the DEA boundary. The frontiers themselves are computed
in **levels** — only the display is logarithmic.

---

**Figure A. Output per labour day and capital per labour day, 1986 / 2000 / 2015.**
Source: cleaned county agricultural panel (1,800 agricultural counties, 2005
prices). Axes: real GVP per labour day against capital service per labour day, both
log scales. The frontier is the DEA best-practice boundary under non-increasing
returns and **free disposability** of the input ratio; the flat right-hand segment
is the free-disposability tail beyond the largest observed ratio. Points are
counties. *File:* `figA_output_vs_capital_per_labour.{png,pdf}`

**Figure A (variant). Output per labour day and land per labour day.** As above
with land service replacing capital. *File:* `figA_output_vs_land_per_labour.{png,pdf}`

**Figure A (density variant).** 2015 counties as a log-scaled hexbin density with
all three frontiers overlaid, for readability where the scatter saturates.
*File:* `figA_hexbin_capital_per_labour.{png,pdf}`

**Figure B. Unit isoquants, labour and capital, 1986 / 2000 / 2015.** Unit
isoquants (y = 1) are built by dividing each county's inputs by its own observed
output and taking the lower-left input-requirement boundary of the normalised cloud
(Kumar & Russell 2002 convention), separately for each year. Both conventional
inputs are **freely disposable**, so the boundary carries flat extensions on both
axes. An isoquant closer to the origin is the more productive technology.
*File:* `figB_unit_isoquant_labour_capital.{png,pdf}`

**Figure B (variant). Unit isoquants, labour and land.**
*File:* `figB_unit_isoquant_labour_land.{png,pdf}`

**Figure C. Unit isoquants in input–weather space: growing degree days.** Source as
above; weather from ERA5 aggregated to counties over the cropland-NDVI growing
season. Output-normalised as in Figure B. The aggregate input is freely disposable
but the weather variate is **weakly disposable** — the equality constraint used in
the main CPS decomposition — so, unlike Figures A and B, the curve has **no flat
extension in the weather direction**: it terminates at the extreme feasible weather
level, marked by horizontal end bars. Weather cannot be freely disposed of.
*File:* `figC_isoquant_input_gdd.{png,pdf}`

**Figure D. Unit isoquants in input–weather space: soil moisture.** As Figure C with
root-zone (0–28 cm) soil moisture as the weather variate. One weather variable per
figure; the 2-vector w is never plotted jointly.
*File:* `figD_isoquant_input_soilmoisture.{png,pdf}`

---

### 11.1 The disposability contrast, stated plainly

The difference between Figures A/B and C/D is the point of the pair, not a plotting
detail:

| | disposability | visual consequence |
|---|---|---|
| A, B (L, K, land) | **free** | boundary extends flat — using more input is always feasible |
| C, D (GDD, soil moisture) | **weak** (equality) | boundary **stops**; end bars mark the terminating feasible level |

### 11.2 Colour and accessibility

Palette: dataviz categorical slots 1–3 (blue `#2a78d6`, orange `#eb6834`, aqua
`#1baf7a`), assigned in fixed order. Validated: worst adjacent CVD ΔE 9.2 (deutan),
normal-vision ΔE 27.6 — both above the required floors. The aqua slot falls below
3:1 contrast against the surface, which obliges secondary encoding, so **every
series also carries a direct label** on its curve in addition to the legend;
identity is never colour-alone. These are static publication figures, so no
hover layer or dark-mode variant is generated.
