"""Detect per-county outliers in one panel variable (county-year panel).

VAR is inferred from the parent folder name (.../cleaning/<VAR>/code_output/),
so the same script serves every variable folder.

Because these variables span ~3 orders of magnitude across counties, outliers
are judged *relative to each county's own time series* on the log scale, not by
any cross-county absolute threshold. Non-positive values (a few 0s in some
variables) cannot be log-transformed and are skipped, not flagged.

Detector — isolated spike / dip (robust to genuine monotone growth)
-------------------------------------------------------------------
For each interior year t of a county's log series y = log(value):
    a = y[t] - y[t-1]      (jump in from the left)
    b = y[t] - y[t+1]      (jump in from the right)
If a and b have the SAME sign, year t sticks out above (peak) or below
(dip) BOTH neighbours, so it is an isolated one-year excursion:
    spike = sign(a) * min(|a|, |b|)
A genuine trend break (level shift) gives a and b opposite signs -> spike 0,
so steady growth is never flagged. A point is an outlier when

    |spike| > DETECT_THRESHOLD           (default ln 5 = nearer neighbour ratio > 5x)

The 5x bar is deliberately loose so genuine scale-up of a county's output is
not mistaken for an error; only sharp one-year excursions survive.

Endpoints (first / last year) have only one neighbour, so they cannot be
cross-checked; we flag them on the same single-step jump
(|dlog| > DETECT_THRESHOLD, default ln 5) and mark them lower-confidence.

(The gentler SYSTEMIC_SCAN_THRESHOLD, 2x, is passed to detect_county as
collect_threshold only to gather candidates for the systemic-year scan; it
never enters the final table.)

Severity (by the smaller of the two neighbour ratios, exp|spike|):
    >= 10x  -> 'order-of-magnitude'   (almost surely a data error)
    >=  3x  -> 'large'
    else    -> 'moderate'

This version loops over ALL the input-output variables (see VARS) in one run,
each detected on its original <var>_raw series, writing one outlier table per
variable into src/dq/outliers/ for one-by-one manual review.

Run
---
    python detect_outliers.py                  # all provinces, all vars, single, 5x
    python detect_outliers.py Quanguo 3        # all provinces, all vars, single, 3x
    python detect_outliers.py Yunnan 5 double  # one province, all vars, plateau, 5x
Args: <Scope> <fold> <method>. Scope = province name or Quanguo (default Quanguo).
fold (default 5) sets DETECT_THRESHOLD = ln(fold); method single|double (default
single). Filename tag = <method>_spike<fold>. The 2x systemic-year scan is fixed.

Outputs (src/dq/outliers/), one set PER VARIABLE; <Tag> = METHOD_TAG:
    outliers_<var>_<Scope>_<Tag>.csv         one row per flagged (countyid, year);
        carries `variable`, `imputed_by_pipeline` (was it auto-imputed already?),
        `category`, `severity`, fold/dlog vs neighbours, prev/next values.
    outlier_year_counts_<var>_<Scope>_<Tag>.csv  #counties flagged/year (systemic)
    nonpositive_<var>_<Scope>.csv            values present but <=0 (excluded from log)
    frozen_<var>_<Scope>.csv                 frozen runs: >= FROZEN_MIN_RUN (default
        4) consecutive years with an IDENTICAL value (基层敷衍填报). One row per run
        with start/end year, length, value, and share of the county's series. GVP
        is scanned on its NOMINAL column (deflation hides a freeze in real_gvp);
        adjustable via the 4th arg <frozen_min_run>.
Provinces are detected independently (systemic years judged per province) then
concatenated; tables carry a leading `state` column.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Project paths (this script lives in the repo root, E:/TFP_weather).
SCRIPT_DIR = Path(__file__).resolve().parent           # repo root
PANEL_CSV = SCRIPT_DIR / "src" / "clean" / "county_panel_clean.csv"
OUT_DIR = SCRIPT_DIR / "src" / "dq" / "outliers"       # one CSV per variable

# The input-output variables to check (same five used downstream). Detection runs
# on the ORIGINAL pre-imputation <var>_raw series (if present) so nothing is hidden
# by the auto-cleaner -- e.g. 基层敷衍填报 (careless grassroots reporting: copied /
# placeholder / one-off keyed numbers) surfaces for one-by-one human review. The
# <var>_imp flag (whether the 01_clean_panel pipeline already imputed that point)
# is appended to every output row so already-handled cases are obvious.
VARS = ["real_gvp", "Laborday_impute", "Land_serv_q", "capital_serv_q", "Inter_all_real"]
RAW_SUFFIX = "_raw"
VAR = VARS[0]                                          # current variable; set in main()

# Frozen-run detector (基层敷衍填报: the same number keyed year after year). A run
# is >= FROZEN_MIN_RUN consecutive years with an identical value. Frozen *reporting*
# lives in NOMINAL GVP -- deflation hides it in real_gvp -- so GVP is scanned on its
# nominal source column; the other variables on their <var>_raw series.
FROZEN_MIN_RUN = 4
FROZEN_SRC = {"real_gvp": "GVP_allagr_impute"}

# Two decoupled thresholds:
#   DETECT — what enters the cleaning table (user-set 5x). Loose on purpose so
#            genuine scale-up is not flagged.
#   SCAN   — a gentler 2x sweep used ONLY to recognise "systemic years" (many
#            counties spiking in the same calendar year). It never adds points
#            to the table; it only decides the systemic-year label.
# Detection fold = how many times a neighbour a point must exceed to be flagged.
# Set from the command line in main() (2nd arg); DEFAULT_FOLD used if omitted.
# The 2x systemic-year scan is fixed and independent of the detection fold.
DEFAULT_FOLD = 5.0
DEFAULT_METHOD = "single"   # "single" = one-year spike; "double" = two-year plateau


def _fold_str(fold: float) -> str:
    """5.0 -> '5', 2.5 -> '2.5' (for filenames and column names)."""
    return str(int(fold)) if float(fold) == int(fold) else str(fold)


DETECT_FOLD = DEFAULT_FOLD
DETECT_METHOD = DEFAULT_METHOD
DETECT_THRESHOLD = np.log(DETECT_FOLD)
SYSTEMIC_SCAN_THRESHOLD = np.log(2.0)
# "double" method: the two plateau years must agree within ~50% (ratio < 1.5).
PLATEAU_TOL = np.log(1.5)
# Tag for this method/fold in output filenames, so other methods or folds write
# to distinct files: outliers_<Scope>_<METHOD_TAG>.csv and
# outlier_year_counts_<Scope>_<METHOD_TAG>.csv carry it; nonpositive_<Scope>.csv
# does NOT (a <=0 value is a data property, shared by every method).
METHOD_TAG = f"{DETECT_METHOD}_spike{_fold_str(DETECT_FOLD)}"


def signed_fold(value: float, neighbor: float) -> float:
    """Fold-change of value vs a neighbour year, sign = direction.

    +N  -> value is N times the neighbour (higher)
    -N  -> value is 1/N of the neighbour, i.e. neighbour is N times value (lower)
    abs() is always the fold magnitude ("差几倍"). NaN if not computable.
    """
    if not np.isfinite([value, neighbor]).all() or value <= 0 or neighbor <= 0:
        return np.nan
    r = value / neighbor
    return round(r, 2) if r >= 1 else round(-1.0 / r, 2)


def severity(abs_spike: float) -> str:
    ratio = np.exp(abs_spike)
    if ratio >= 10:
        return "order-of-magnitude"
    if ratio >= 3:
        return "large"
    return "moderate"


def detect_county(g: pd.DataFrame, collect_threshold: float) -> list[dict]:
    g = g.sort_values("year")
    yr = g["year"].to_numpy()
    val = g[VAR].to_numpy(dtype=float)
    # log; guard non-positive (none expected, but be safe)
    with np.errstate(divide="ignore"):
        y = np.where(val > 0, np.log(val), np.nan)
    cid = int(g["countyid"].iloc[0])
    name = g["county_name"].dropna()
    name = name.iloc[0] if len(name) else ""
    out = []
    n = len(y)

    for i in range(1, n - 1):
        if not np.isfinite(y[i - 1: i + 2]).all():
            continue
        a = y[i] - y[i - 1]
        b = y[i] - y[i + 1]
        if a * b <= 0:
            continue  # not isolated (trend / level shift)
        spike = np.sign(a) * min(abs(a), abs(b))
        if abs(spike) <= collect_threshold:
            continue
        out.append(dict(
            countyid=cid, county_name=name, year=int(yr[i]),
            value=val[i], prev_value=val[i - 1], next_value=val[i + 1],
            dlog_prev=round(a, 3), dlog_next=round(b, 3),
            spike=round(spike, 3),
            direction="peak" if spike > 0 else "dip",
            position="interior",
            severity=severity(abs(spike)),
            confidence="high",
        ))

    # endpoints
    for i, nb in ((0, 1), (n - 1, n - 2)):
        if n < 3 or not np.isfinite([y[i], y[nb]]).all():
            continue
        step = y[i] - y[nb]
        if abs(step) <= collect_threshold:
            continue
        out.append(dict(
            countyid=cid, county_name=name, year=int(yr[i]),
            value=val[i], prev_value=(val[nb] if i > 0 else np.nan),
            next_value=(val[nb] if i == 0 else np.nan),
            dlog_prev=round(step, 3) if i > 0 else np.nan,
            dlog_next=round(step, 3) if i == 0 else np.nan,
            spike=round(step, 3),
            direction="peak" if step > 0 else "dip",
            position="first" if i == 0 else "last",
            severity=severity(abs(step)),
            confidence="low",
        ))

    # add direct fold-change labels (符号=涨跌, 绝对值=倍数)
    for d in out:
        d["fold_prev"] = signed_fold(d["value"], d["prev_value"])
        d["fold_next"] = signed_fold(d["value"], d["next_value"])
    return out


def detect_county_double(g: pd.DataFrame, collect_threshold: float) -> list[dict]:
    """Two-year plateau spike: years (t, t+1) sit within ~50% of each other but
    BOTH stand > fold from their outer neighbours (t-1 and t+2), in the same
    direction. Catches a two-year data error a one-year spike would miss.
    Each plateau yields two rows (its first and second year)."""
    g = g.sort_values("year")
    yr = g["year"].to_numpy()
    val = g[VAR].to_numpy(dtype=float)
    with np.errstate(divide="ignore"):
        y = np.where(val > 0, np.log(val), np.nan)
    cid = int(g["countyid"].iloc[0])
    name = g["county_name"].dropna()
    name = name.iloc[0] if len(name) else ""
    out = []
    n = len(y)

    # plateau occupies indices i, i+1; needs outer neighbours i-1 and i+2
    for i in range(1, n - 2):
        if not np.isfinite(y[i - 1: i + 3]).all():
            continue
        a = y[i] - y[i - 1]        # first plateau year vs the year before
        c = y[i + 1] - y[i + 2]    # second plateau year vs the year after
        if abs(y[i] - y[i + 1]) >= PLATEAU_TOL:
            continue               # the two years differ by >= ~50%: not a plateau
        if abs(a) <= collect_threshold or abs(c) <= collect_threshold:
            continue               # an end jump is below the fold
        if a * c <= 0:
            continue               # ends disagree -> trend, not a plateau spike
        mag = min(abs(a), abs(c))  # plateau is only as extreme as its weaker end
        direction = "peak" if a > 0 else "dip"
        sev = severity(mag)
        out.append(dict(
            countyid=cid, county_name=name, year=int(yr[i]),
            value=val[i], prev_value=val[i - 1], next_value=val[i + 1],
            dlog_prev=round(a, 3), dlog_next=round(y[i] - y[i + 1], 3),
            spike=round(np.sign(a) * mag, 3), direction=direction,
            position="double-first", severity=sev, confidence="high"))
        out.append(dict(
            countyid=cid, county_name=name, year=int(yr[i + 1]),
            value=val[i + 1], prev_value=val[i], next_value=val[i + 2],
            dlog_prev=round(y[i + 1] - y[i], 3), dlog_next=round(c, 3),
            spike=round(np.sign(c) * mag, 3), direction=direction,
            position="double-second", severity=sev, confidence="high"))

    for d in out:
        d["fold_prev"] = signed_fold(d["value"], d["prev_value"])
        d["fold_next"] = signed_fold(d["value"], d["next_value"])
    return out


DETECTORS = {"single": detect_county, "double": detect_county_double}


# A year is "systemic" for a state when many counties spike in the SAME year —
# that points to a calendar-year data/coverage issue, not random county errors.
SYSTEMIC_MIN_COUNT = 4
SYSTEMIC_MIN_SHARE = 0.05


def assign_category(df: pd.DataFrame, systemic_years: set,
                    scan_per_year: pd.Series) -> pd.DataFrame:
    if df.empty:
        df["category"] = []
        df["year_n_flagged_2x"] = []
        return df
    # how many counties spiked (>=2x) in this year — the systemic-year evidence
    df["year_n_flagged_2x"] = df["year"].map(scan_per_year).astype("Int64")

    def _cat(r):
        if r["year"] in systemic_years:
            return "systemic-year"
        if r["position"] in ("first", "last"):
            return "endpoint-jump"
        double = r["position"] in ("double-first", "double-second")
        if r["severity"] == "order-of-magnitude":
            return "double-magnitude-error" if double else "isolated-magnitude-error"
        return "double-spike" if double else "isolated-spike"

    df["category"] = df.apply(_cat, axis=1)
    return df


COLS = ["countyid", "county_name", "year", "value", "prev_value",
        "next_value", "fold_prev", "fold_next", "dlog_prev", "dlog_next",
        "spike", "direction", "position", "severity", "confidence"]


def detect_state(panel: pd.DataFrame, state: str):
    """Return (cleaning_table, year_counts) for one state."""
    sub = panel[panel["state"] == state][
        ["countyid", "county_name", "year", VAR]
    ].dropna(subset=[VAR])
    n_counties = sub.countyid.nunique()

    # Gentle 2x sweep: every candidate excursion, used for systemic detection.
    detector = DETECTORS[DETECT_METHOD]
    rows: list[dict] = []
    for _, g in sub.groupby("countyid"):
        rows.extend(detector(g, SYSTEMIC_SCAN_THRESHOLD))
    cand = pd.DataFrame(rows, columns=COLS)

    if cand.empty:
        empty = assign_category(cand.copy(), set(), pd.Series(dtype=int))
        return empty, pd.DataFrame(
            columns=["year", "n_counties_2x",
                     f"n_counties_{_fold_str(DETECT_FOLD)}x"])

    scan_per_year = cand.groupby("year").countyid.nunique()
    thr = max(SYSTEMIC_MIN_COUNT, SYSTEMIC_MIN_SHARE * n_counties)
    systemic_years = set(scan_per_year[scan_per_year >= thr].index)

    # Cleaning table keeps only excursions above the detection fold.
    kept = cand[cand["spike"].abs() > DETECT_THRESHOLD].copy()
    kept = assign_category(kept, systemic_years, scan_per_year)
    if not kept.empty:
        kept = kept.sort_values(
            ["category", "severity", "spike"],
            key=lambda s: (
                s.map({"systemic-year": 0, "isolated-magnitude-error": 1,
                       "isolated-spike": 2, "endpoint-jump": 3})
                if s.name == "category"
                else s.map({"order-of-magnitude": 0, "large": 1,
                            "moderate": 2}) if s.name == "severity"
                else s.abs()),
            ascending=[True, True, False],
        ).reset_index(drop=True)

    kept_per_year = kept.groupby("year").countyid.nunique() if not kept.empty \
        else pd.Series(dtype=int)
    year_counts = (
        pd.DataFrame({"n_counties_2x": scan_per_year,
                      f"n_counties_{_fold_str(DETECT_FOLD)}x": kept_per_year})
        .fillna(0).astype(int).reset_index()
        .rename(columns={"index": "year"})
        .sort_values("year")
    )
    return kept, year_counts


def _nonpositive(panel: pd.DataFrame, state: str) -> pd.DataFrame:
    """Values present but <=0 — excluded from the log detector, kept for audit
    (e.g. capital_serv_q has 70 zeros, all in Hainan — likely missing-as-0)."""
    sr = panel[panel.state == state]
    return sr.loc[sr[VAR].notna() & (sr[VAR] <= 0),
                  ["countyid", "county_name", "year", VAR]
                  ].sort_values(["countyid", "year"])


FROZEN_COLS = ["countyid", "county_name", "run_start", "run_end", "run_length",
               "frozen_value", "n_obs", "frac_of_series", "source_column"]


def _frozen_runs(panel: pd.DataFrame, state: str, col: str) -> pd.DataFrame:
    """Runs of >= FROZEN_MIN_RUN consecutive years with an IDENTICAL value — a
    county keying the same number year after year (基层敷衍填报). One row per run:
    its start/end year, length, the frozen value, and what share of the county's
    series it covers (a whole-series freeze is the strongest signal)."""
    sr = panel[panel.state == state][["countyid", "county_name", "year", col]].dropna(subset=[col])
    rows = []
    for cid, g in sr.groupby("countyid"):
        g = g.sort_values("year")
        yr = g["year"].to_numpy()
        val = g[col].to_numpy(dtype=float)
        nm = g["county_name"].dropna()
        nm = nm.iloc[0] if len(nm) else ""
        n = len(val)
        i = 0
        while i < n:
            j = i + 1
            while j < n and yr[j] == yr[j - 1] + 1 and val[j] == val[i]:
                j += 1
            run = j - i
            if run >= FROZEN_MIN_RUN:
                rows.append(dict(
                    countyid=int(cid), county_name=nm,
                    run_start=int(yr[i]), run_end=int(yr[j - 1]),
                    run_length=int(run), frozen_value=val[i],
                    n_obs=int(n), frac_of_series=round(run / n, 3),
                    source_column=col))
            i = j
    return pd.DataFrame(rows, columns=FROZEN_COLS)


def _state_first(d):
    return d[["state"] + [c for c in d.columns if c != "state"]]


def _concat(frames):  # skip empty frames -> avoids all-NA concat warning
    nonempty = [f for f in frames if not f.empty]
    return (pd.concat(nonempty, ignore_index=True) if nonempty
            else pd.DataFrame(columns=frames[0].columns))


def process_variable(panel: pd.DataFrame, var: str, states: list, quanguo: bool):
    """Detect outliers for ONE variable across `states`, write its own CSVs.

    Detection runs on `<var>_raw` (the original pre-imputation series) when it
    exists, so values the auto-cleaner already fixed still surface for review.
    """
    global VAR
    VAR = var
    detect_col = var + RAW_SUFFIX if (var + RAW_SUFFIX) in panel.columns else var
    imp_col = var + "_imp"
    work = panel.copy()
    work[var] = work[detect_col]                 # detector reads raw values under base name

    out_frames, yc_frames, np_frames = [], [], []
    for state in states:
        df, yr_counts = detect_state(work, state)
        nonpos = _nonpositive(work, state)
        out_frames.append(df.assign(state=state))
        yc_frames.append(yr_counts.assign(state=state))
        np_frames.append(nonpos.assign(state=state))

    outliers = _state_first(_concat(out_frames))
    # flag points the 01_clean_panel pipeline already imputed (so they stand out)
    if imp_col in panel.columns and not outliers.empty:
        flag = (panel[["countyid", "year", imp_col]]
                .rename(columns={imp_col: "imputed_by_pipeline"}))
        outliers = outliers.merge(flag, on=["countyid", "year"], how="left")
    outliers.insert(0, "variable", var)
    outliers["source_column"] = detect_col

    scope = "Quanguo" if quanguo else states[0].replace(" ", "_")
    outliers.to_csv(OUT_DIR / f"outliers_{var}_{scope}_{METHOD_TAG}.csv",
                    index=False, encoding="utf-8-sig")
    _state_first(_concat(yc_frames)).to_csv(
        OUT_DIR / f"outlier_year_counts_{var}_{scope}_{METHOD_TAG}.csv",
        index=False, encoding="utf-8-sig")
    _state_first(_concat(np_frames)).to_csv(
        OUT_DIR / f"nonpositive_{var}_{scope}.csv",
        index=False, encoding="utf-8-sig")

    # frozen-run scan (敷衍填报): GVP on its nominal source, others on <var>_raw
    frozen_col = FROZEN_SRC.get(var, detect_col)
    if frozen_col not in panel.columns:
        frozen_col = detect_col
    frozen = _concat([_frozen_runs(panel, s, frozen_col).assign(state=s) for s in states])
    if not frozen.empty:
        frozen = _state_first(frozen.sort_values(["run_length", "frac_of_series"],
                                                 ascending=False))
    frozen.to_csv(OUT_DIR / f"frozen_{var}_{scope}.csv",
                  index=False, encoding="utf-8-sig")

    cats = outliers.category.value_counts().to_dict() if not outliers.empty else {}
    ncty = outliers.countyid.nunique() if not outliers.empty else 0
    print(f"[{var:16}] {len(outliers):5} spike-outliers (>{_fold_str(DETECT_FOLD)}x) "
          f"in {ncty:4} cty  |  {len(frozen):4} frozen runs (>={FROZEN_MIN_RUN}y, on {frozen_col})  {cats}")
    return outliers


def main() -> None:
    global DETECT_FOLD, DETECT_METHOD, DETECT_THRESHOLD, METHOD_TAG, FROZEN_MIN_RUN
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL_CSV, encoding="utf-8-sig")

    # Args: <Scope> <fold> <method> <frozen_min_run>.  Scope defaults to Quanguo.
    target = sys.argv[1] if len(sys.argv) > 1 else "Quanguo"
    DETECT_FOLD = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_FOLD
    if DETECT_FOLD <= 1:
        raise SystemExit(f"fold must be > 1, got {DETECT_FOLD}")
    DETECT_METHOD = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_METHOD
    if DETECT_METHOD not in DETECTORS:
        raise SystemExit(
            f"method must be one of {list(DETECTORS)}, got {DETECT_METHOD!r}")
    if len(sys.argv) > 4:
        FROZEN_MIN_RUN = int(sys.argv[4])
        if FROZEN_MIN_RUN < 2:
            raise SystemExit(f"frozen_min_run must be >= 2, got {FROZEN_MIN_RUN}")
    DETECT_THRESHOLD = np.log(DETECT_FOLD)
    METHOD_TAG = f"{DETECT_METHOD}_spike{_fold_str(DETECT_FOLD)}"

    quanguo = target.upper() in ("QUANGUO", "ALL", "全国")
    states = sorted(panel["state"].dropna().unique()) if quanguo else [target]
    scope = "Quanguo" if quanguo else target

    print(f"Scope={scope}  fold={_fold_str(DETECT_FOLD)}x  method={DETECT_METHOD}  "
          f"detect on <var>{RAW_SUFFIX}  ->  {OUT_DIR.relative_to(SCRIPT_DIR)}\n")
    for var in VARS:
        process_variable(panel, var, states, quanguo)
    print(f"\nDone. Review one CSV per variable: "
          f"outliers_<var>_{scope}_{METHOD_TAG}.csv  in {OUT_DIR.relative_to(SCRIPT_DIR)}")


if __name__ == "__main__":
    main()
