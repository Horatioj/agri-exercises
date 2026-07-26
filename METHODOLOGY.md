# Methodology Framework: DEA and SFA as Complementary Frontier Approaches
### China county-level agricultural TFP, 1981–2016

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

- **DEA** yields TFP growth of **+3.97%/yr** (output-weighted) and, because it treats all
  deviation as inefficiency, retains large cross-county efficiency dispersion.
- **SFA** (pooled, land-normalized) yields **+2.89%/yr** but collapses to a *neutral
  technical-change* path because of **wrong skewness** (see §5): after rigorous cleaning
  the residuals are symmetric (within-county skewness +0.04), so no one-sided inefficiency
  component is identified and σ<sub>u</sub> → 0.
- The **gap between them (~1.1 pp/yr) is precisely the efficiency-change component** that
  DEA attributes to catch-up and SFA does not identify. This is not a contradiction: it is
  the decomposition made visible by using both methods.

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
(`src/09_dea_vs_fe_tfp.py`) because ~350k LPs are infeasible in Stata's ado-based DEA;
all **econometric analysis** of the resulting indices is done in Stata.

---

## 4. SFA: land-normalized stochastic frontier

### 4.1 Specification
Imposing **CRS by land-normalization** (dividing output and all inputs by land service)
avoids the biased returns-to-scale that plague within-estimators here:

&nbsp;&nbsp;&nbsp;&nbsp;ln(y/S)<sub>it</sub> = α + β<sub>L</sub> ln(L/S) + β<sub>K</sub> ln(K/S) + β<sub>M</sub> ln(M/S) + λ<sub>t</sub> + v<sub>it</sub> − u<sub>it</sub>

with v ~ N(0, σ<sub>v</sub><sup>2</sup>) noise and u ≥ 0 inefficiency. Year dummies λ<sub>t</sub>
(base 2005) trace **neutral technical change**; SEs clustered by county.

### 4.2 TFP construction (Chen-style)
&nbsp;&nbsp;&nbsp;&nbsp;**ln TFP<sub>it</sub> = (α + λ<sub>t</sub>) − u<sub>it</sub>**

i.e. the technology level net of inefficiency — the direct SFA analogue of
M = TECHCH × EFFCH. Input contributions are stripped out so TFP is not mechanically
driven by input growth.

### 4.3 Panel models and the identification problem
| Model | Heterogeneity control | Outcome on this panel |
|---|---|---|
| Pooled `frontier` | none | Converges instantly; **σ<sub>u</sub> → 0** (wrong skewness) |
| `sfpanel, bc92` | none (persistent heterogeneity → u) | **Does not converge** at N=2,505 (non-concave); **converges on a 1,000-county subsample**: σ²<sub>u</sub>=0.242, σ²<sub>v</sub>=0.111, λ=1.48 |
| `sfpanel, tfe` | full county FE (Greene 2005) | **Never converges** (incidental parameters, 2,505 nuisance parameters) |

**Key methodological point:** BC92 is a *panel time-decay* model — it identifies u from the
**persistent panel component**, not from the skewness of the composed error. Hence it
recovers σ<sub>u</sub> > 0 even where pooled SFA cannot. Its efficiency *levels* are
implausibly low (mean TE ≈ 0.064) because without a heterogeneity term persistent county
differences are mislabeled as inefficiency; but with η ≈ −0.008 (near time-invariant
inefficiency) the *growth* series is unaffected. **Therefore: use BC92 for TFP growth, not
for efficiency levels.**

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
| Data cleaning | `src/00*`–`src/02*` | Python |
| Descriptive statistics & moments | `src/24_descriptives.do` | **Stata** |
| DEA sequential-NIRS Malmquist + EFFCH/TECHCH | `src/09_dea_vs_fe_tfp.py` | Python (LP) |
| SFA pooled / BC92 / subsample | `src/11`, `16`, `17_*.do` | **Stata** |
| Framework analysis, decomposition & comparison tables | `src/25_dea_sfa_framework.do` | **Stata** |
| Aggregation, volatility, stage analysis | `src/12`–`15`, `18`, `19` | Python |
| Production surface figures | `src/23_io_combined.py` | Python |

Econometric estimation and inference are done in Stata; linear programming (DEA) and
data engineering in Python.
