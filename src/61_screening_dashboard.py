# -*- coding: utf-8 -*-
"""
County screening dashboards: three self-contained HTML maps, no server.

PURPOSE.  Not a data viewer -- a SAMPLE-DEFINITION tool.  59_frontier_audit.py
and 60_leave_one_out.py established that the frontier is set by a handful of
peri-urban districts and island/coastal counties, and that excluding sixteen of
them moves mean DEA efficiency +22% and mean SFA efficiency +91% while leaving
TFP GROWTH almost unchanged (4.24 -> 4.15 %/yr).  The open question is therefore
not "is the sample fragile" but "which counties should the study be about".
These maps exist to answer that by eye, county by county and year by year.

THREE FILES
  dash_raw.html     the source series with NOTHING done to them except the 2005
                    deflation -- GVP_allagr_impute / PPI_CCD_2005 and the four
                    inputs as they arrive.  The baseline against which every
                    cleaning decision can be judged.
  dash_clean.html   the same five variables from county_panel_clean.csv, i.e.
                    after unit fixes, boundary-run rescaling, spike imputation
                    and the extreme-low floor.
  dash_screen.html  the indicators that actually decide sample membership:
                    cropland share, the physical crop-vs-livestock split, the
                    horticulture share, irrigation, the urban-district label,
                    and how often the county sits on a DEA hull.

WHY PYTHON AND NOT R/SHINY.  The plotting is the easy half.  The hard half is
matching 1,975 county codes to 2010 boundaries, and that already exists here in
_county_match.py -- SHP_CODE_FIX for the malformed 7-digit code, the MANUAL
rename table (满州里市, 霍县, 布特哈旗, 江浦县, 白郎县 ...), the 撤县设区 merges in
_county_corrections.py, and the EPSG:3857 -> 4326 reprojection that silently
returned all-NaN when it was missing.  Rebuilding that in R would duplicate the
part where the bugs live, to save effort on the part where they do not.  Shiny
also needs a running R process; these files open in a browser and can be sent to
someone.  If the interactivity outgrows a slider and a dropdown -- cross-filter,
linked brushing, on-the-fly re-estimation -- Shiny becomes the right answer and
the matching can be exported once to a lookup CSV rather than reimplemented.

Run:  python src/61_screening_dashboard.py
      python src/61_screening_dashboard.py --years 1985,1995,2005,2015
Out:  src/figures/dashboard/dash_{raw,clean,screen}.html
      src/dq/county_screening_panel.csv
"""
from __future__ import annotations
import os, sys, json, argparse
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
import _common as C
import _tfp as T
import _frontier as F

OUT = os.path.join(C.FIG_DIR, "dashboard")
os.makedirs(OUT, exist_ok=True)
os.makedirs(C.DQ_DIR, exist_ok=True)


def build_panel(from_year):
    """One long panel carrying the raw series, the cleaned series and every
    screening indicator, keyed county-year."""
    d = pd.read_csv(C.CLEAN_PANEL)
    d = d[(d.ag_county == 1) & (d.year >= from_year)].copy()

    # --- RAW, deflated only ------------------------------------------------
    # <var>_raw is the snapshot taken before any cleaning rule fired, so these
    # columns ARE the source series.  GVP needs the same deflation the cleaned
    # real_gvp got, and nothing else.
    d["raw_real_gvp"] = pd.to_numeric(d.get("real_gvp_raw"), errors="coerce")
    for c in ["Laborday_impute", "Land_serv_q", "capital_serv_q", "Inter_all_real"]:
        d["raw_" + c] = pd.to_numeric(d.get(c + "_raw"), errors="coerce")

    # --- physical crop vs livestock ---------------------------------------
    q = pd.read_stata(os.path.join(C.DATA, "cty_prod_account_q&p_forAXN.dta"),
                      convert_categoricals=False)
    keep = ["countyid", "year", "totalmeat_impute", "rice_impute", "wheat_impute",
            "maize_impute", "cotton_impute", "oilseeds_impute",
            "vege_area_impute", "fruit_area_impute"]
    q = q[[c for c in keep if c in q.columns]].drop_duplicates(["countyid", "year"])
    d = d.merge(q, on=["countyid", "year"], how="left")
    grain = d[["rice_impute", "wheat_impute", "maize_impute"]].sum(axis=1, min_count=1)
    d["grain_t"] = grain
    # A PHYSICAL share, in tonnes, so no price index enters and the crop/
    # livestock split cannot be contaminated by the deflator problems that
    # affect the value series.  It is a proxy, not an accounting share: meat and
    # grain are not commensurable, so read it as an ordering, not a percentage.
    d["meat_vs_grain"] = d.totalmeat_impute / (d.totalmeat_impute + grain)
    # Vegetable and fruit area are SEPARATE columns and must stay separate:
    # corr(vege, fruit) = -0.011 across 103,315 county-years, i.e. they are
    # independent specialisations, so a combined "horticulture" number hides
    # exactly the distinction that identifies peri-urban vegetable belts as
    # against orchard counties.  Both cover 1981-2016 (>0 in 88-98% and 85-93%
    # of county-years respectively).
    d["vege_area"] = pd.to_numeric(d.get("vege_area_impute"), errors="coerce")
    d["fruit_area"] = pd.to_numeric(d.get("fruit_area_impute"), errors="coerce")

    # --- irrigation --------------------------------------------------------
    ir = pd.read_stata(os.path.join(C.DATA, "county_irri_elect_fert_forXZ.dta"),
                       convert_categoricals=False)
    ir = ir[["countyid", "year", "irrigated_area"]].drop_duplicates(["countyid", "year"])
    d = d.merge(ir, on=["countyid", "year"], how="left")

    # --- cropland share, the ag-county definition itself -------------------
    cs = pd.read_csv(os.path.join(C.DATA, "ag_county", "cropland_share_by_year.csv"))
    cs = cs.rename(columns={"code": "countyid", "crop_pct": "cropland_pct"})
    cs = cs[["countyid", "year", "cropland_pct"]].drop_duplicates(["countyid", "year"])
    d = d.merge(cs, on=["countyid", "year"], how="left")
    # Per-COUNTY summary of the same rasters (ag_counties_union.csv): how many
    # years the county cleared the 15% test, and its best and mean share.  These
    # are constants, so they say what the per-year layer cannot -- whether a
    # county is agricultural throughout or only crossed the line briefly, which
    # is exactly the margin the 15% threshold is fragile on (160 counties sit
    # within +/-3 percentage points of it).
    un = pd.read_csv(os.path.join(C.DATA, "ag_county", "ag_counties_union.csv"))
    un = un.rename(columns={"code": "countyid", "max_pct": "cropland_max_pct",
                            "mean_pct": "cropland_mean_pct",
                            "n_yrs_above": "cropland_yrs_above15"})
    un["is_ag_always"] = un["is_ag_always"].astype(int)
    d = d.merge(un[["countyid", "cropland_max_pct", "cropland_mean_pct",
                    "cropland_yrs_above15", "is_ag_always"]],
                on="countyid", how="left")


    # --- county type and frontier leverage ---------------------------------
    d["is_district"] = d.county_name.map(C.is_urban_district).astype(int)
    d = F.add_region(d)
    # How often does this county sit ON a single-input DEA hull?  Cheap (a hull,
    # not an LP) and it is the same leverage 60_leave_one_out.py measured the
    # consequences of -- a county with a high count is one whose removal moves
    # everyone else's efficiency score.
    lev = {}
    for yr, g in d.groupby("year"):
        for c in ["Laborday_impute", "Land_serv_q", "capital_serv_q", "Inter_all_real"]:
            x = pd.to_numeric(g[c], errors="coerce").values
            y = pd.to_numeric(g["real_gvp"], errors="coerce").values
            cid = g.countyid.values
            ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            if ok.sum() < 10:
                continue
            xs, ys, cs_ = x[ok], y[ok], cid[ok]
            o = np.argsort(xs)
            H = F._upper_hull([(0.0, 0.0, -1)] + list(zip(xs[o], ys[o], cs_[o])))
            for h in H:
                if h[2] > 0:
                    lev[(int(h[2]), int(yr))] = lev.get((int(h[2]), int(yr)), 0) + 1
    d["hull_vertex_count"] = [lev.get((int(a), int(b)), 0)
                              for a, b in zip(d.countyid, d.year)]
    return d


HTML = r"""<meta charset="utf-8">
<title>__TITLE__</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
 body{margin:0;font:13px/1.45 -apple-system,Segoe UI,Roboto,"Microsoft YaHei",sans-serif;color:#1a1a19}
 #bar{padding:8px 14px;border-bottom:1px solid #e6e5e0;display:flex;gap:18px;align-items:center;flex-wrap:wrap}
 #bar b{font-size:15px} select{font:13px inherit;padding:3px 6px}
 #yr{width:420px;vertical-align:middle} #ylab{font-variant-numeric:tabular-nums;min-width:3.2em;display:inline-block}
 #note{color:#5c5b55;font-size:12px;padding:2px 14px 8px}
 #map{width:100vw;height:calc(100vh - 108px)}
 button{font:13px inherit;padding:2px 9px}
</style>
<div id="bar">
  <b>__TITLE__</b>
  <label>variable <select id="var"></select></label>
  <label>year <input type="range" id="yr"> <span id="ylab"></span></label>
  <button id="play">▶</button>
</div>
<div id="note">__NOTE__</div>
<div id="map"></div>
<script>
const D = __DATA__;
const nC = D.cid.length, nT = D.years.length;
const sel = document.getElementById('var'), sl = document.getElementById('yr');
D.vars.forEach((v,i)=>{const o=document.createElement('option');o.value=i;o.text=v.label;sel.add(o)});
sl.min=0; sl.max=nT-1; sl.value=nT-1;

// Hover carries EVERY variable for that county-year, not just the mapped one:
// the point of the map is choosing a sample, and that decision needs the whole
// row in one place rather than four passes over four maps.
function hover(t){
  const out=[];
  for(let i=0;i<nC;i++){
    let s='<b>'+D.name[i]+'</b>  '+D.cid[i];
    for(const v of D.vars){
      const x=v.z[i*nT+t];
      s+='<br>'+v.label+': '+(x===null||x!==x?'—':fmt(x));
    }
    out.push(s);
  }
  return out;
}
function fmt(x){
  if(x===null||x!==x) return '—';
  const a=Math.abs(x);
  if(a>=1e6) return (x/1e6).toFixed(2)+'M';
  if(a>=1e3) return x.toLocaleString(undefined,{maximumFractionDigits:0});
  return (+x.toPrecision(4)).toString();
}
function zvals(k,t){
  const v=D.vars[k], out=new Array(nC);
  for(let i=0;i<nC;i++){
    const x=v.z[i*nT+t];
    out[i]=(x===null||x!==x)?null:(v.log?(x>0?Math.log10(x):null):x);
  }
  return out;
}

const HATCH = D.hatch ? [{
  type:'scattergeo', mode:'lines', lon:D.hatch.lon, lat:D.hatch.lat,
  line:{color:'#9a9892',width:.5}, hoverinfo:'skip', showlegend:false
}] : [];

// A conic equal-area projection on the 25N/47N standard parallels with the
// central meridian at 105E -- the same Krasovsky Albers the project's raster
// work uses, so county AREAS on this map are comparable with the cropland
// shares computed from those rasters.  A plain equirectangular view would
// stretch the north and make Xinjiang and Heilongjiang look far larger.
const GEO = {
  projection:{type:'conic equal area', parallels:[25,47], rotation:{lon:105}},
  showcountries:true, countrycolor:'#5c5b55', countrywidth:1.1,
  showcoastlines:true, coastlinecolor:'#8a8880', coastlinewidth:.7,
  showland:true, landcolor:'#f4f3f0', showocean:true, oceancolor:'#eaf1f7',
  showlakes:false, showframe:false, resolution:50,
  lonaxis:{range:[73,136]}, lataxis:{range:[17,54]}
};

function draw(k,t,init){
  const v=D.vars[k];
  const tr={type:'choropleth', geojson:D.geo, locations:D.cid.map(String),
    featureidkey:'properties.cid', z:zvals(k,t), text:hover(t),
    hovertemplate:'%{text}<extra></extra>',
    colorscale:'Viridis', marker:{line:{width:0}},
    colorbar:{title:{text:(v.log?'log10 ':'')+v.label,side:'right'},thickness:14,len:.72}};
  if(init){
    Plotly.newPlot('map',[...HATCH,tr],
      {geo:GEO,margin:{l:0,r:0,t:0,b:0},dragmode:'pan'},
      {responsive:true,displaylogo:false,scrollZoom:true});
  }else{
    Plotly.restyle('map',{z:[tr.z],text:[tr.text],colorbar:[tr.colorbar]},[HATCH.length]);
  }
  document.getElementById('ylab').textContent=D.years[t];
}
let k=0,t=nT-1,timer=null;
draw(k,t,true);
sel.onchange=()=>{k=+sel.value;draw(k,t)};
sl.oninput=()=>{t=+sl.value;draw(k,t)};
document.getElementById('play').onclick=function(){
  if(timer){clearInterval(timer);timer=null;this.textContent='▶';return}
  this.textContent='❚❚';
  timer=setInterval(()=>{t=(t+1)%nT;sl.value=t;draw(k,t)},420);
};
</script>
"""


def _hatch_each(geoms, step):
    """Fallback: hatch polygon by polygon when the union cannot be formed."""
    from shapely.geometry import LineString
    lon, lat = [], []
    for gm in geoms:
        x0, y0, x1, y1 = gm.bounds
        c = x0 - (y1 - y0)
        while c <= x1 + (y1 - y0):
            try:
                inter = LineString([(c, y0 - 1),
                                    (c + (y1 - y0) + 2, y1 + 1)]).intersection(gm)
            except Exception:
                c += step; continue
            for pg in getattr(inter, "geoms", [inter]):
                if pg.is_empty or pg.geom_type != "LineString":
                    continue
                xs, ys = pg.xy
                lon += [round(v, 3) for v in xs] + [None]
                lat += [round(v, 3) for v in ys] + [None]
            c += step
    return {"lon": lon, "lat": lat} if lon else None


def hatch_lines(gexc, step=0.28):
    """Diagonal hatch for counties with NO data, so "not in the sample" is
    visually distinct from "in the sample, value near zero".

    Plotly choropleths have no fill pattern, so the hatch is drawn as real
    geometry: parallel 45-degree lines clipped to each excluded polygon and
    emitted as one Scattergeo trace with NaN separators.  A blank white gap
    would read as missing rendering; a hatch reads as a deliberate exclusion.
    """
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    lon, lat = [], []
    # Four of the 890 excluded polygons are invalid after simplification and
    # unary_union then raises "unable to assign free hole to a shell".
    # buffer(0) repairs them; if the union still fails the polygons are hatched
    # individually, which is slower but never loses the layer silently.
    geoms = [gm.buffer(0) if not gm.is_valid else gm
             for gm in gexc.geometry.values if gm is not None and not gm.is_empty]
    try:
        geom = unary_union(geoms)
    except Exception:
        geom = None
    if geom is None or geom.is_empty:
        return _hatch_each(geoms, step)
    x0, y0, x1, y1 = geom.bounds
    c = x0 - (y1 - y0)
    while c <= x1 + (y1 - y0):
        seg = LineString([(c, y0 - 1), (c + (y1 - y0) + 2, y1 + 1)])
        try:
            inter = seg.intersection(geom)
        except Exception:
            c += step; continue
        parts = getattr(inter, "geoms", [inter])
        for pgeom in parts:
            if pgeom.is_empty or pgeom.geom_type != "LineString":
                continue
            xs, ys = pgeom.xy
            lon += [round(v, 3) for v in xs] + [None]
            lat += [round(v, 3) for v in ys] + [None]
        c += step
    return {"lon": lon, "lat": lat} if lon else None


def write_map(d, cols, gj, stub, title, note, hatch=None):
    """One self-contained HTML, driven by hand-written JS rather than Plotly
    frames.

    Frames were the first attempt and do not scale to the full panel: they carry
    one z array per variable per year, so 8 variables x 36 years x 1,975
    counties came to roughly 80 MB of JSON per file.  The payload here is the
    minimum -- one flat array per variable over county-major order -- and the
    year slider and variable dropdown recompute the displayed layer in the
    browser, which also lets the tooltip show EVERY variable at once instead of
    only the mapped one.
    """
    years = sorted(d.year.unique())
    ids = sorted(d.countyid.unique())
    P = d.set_index(["countyid", "year"])
    nm = d.drop_duplicates("countyid").set_index("countyid").county_name
    idx = pd.MultiIndex.from_product([ids, years], names=["countyid", "year"])
    P = P.reindex(idx)

    def flat(c):
        v = pd.to_numeric(P[c], errors="coerce").values
        # 6 significant digits is well inside the precision of any of these
        # series and roughly halves the JSON
        return [None if not np.isfinite(x) else float(f"{x:.6g}") for x in v]

    data = {"cid": [int(c) for c in ids],
            "name": [str(nm.get(c, "")) for c in ids],
            "years": [int(y) for y in years],
            "vars": [{"label": lab, "log": bool(lg), "z": flat(c)}
                     for c, lab, lg in cols],
            "geo": gj, "hatch": hatch}
    html = (HTML.replace("__TITLE__", title).replace("__NOTE__", note)
                .replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    p = os.path.join(OUT, stub + ".html")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"  saved {os.path.basename(p)}  ({os.path.getsize(p)/1e6:.1f} MB, "
          f"{len(cols)} variables, {len(years)} years)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-year", type=int, default=1981)
    ap.add_argument("--years", default=None,
                    help="comma-separated subset; fewer years = much smaller HTML")
    ap.add_argument("--simplify", type=float, default=0.02,
                    help="polygon simplification tolerance in degrees")
    a = ap.parse_args()

    d = build_panel(a.from_year)
    if a.years:
        d = d[d.year.isin([int(y) for y in a.years.split(",")])]
    print(f"panel: {len(d):,d} county-years, {d.countyid.nunique():,d} counties, "
          f"{d.year.min()}-{d.year.max()}")

    import _county_match as M
    g = M.load_boundaries()                       # GeoDataFrame, key column `code`
    # match_counties annotates the PANEL with map_code; the geometry stays in g,
    # so the join runs panel.map_code -> g.code and the polygons are relabelled
    # by countyid (several 2010 polygons can carry one panel county after the
    # 撤县设区 merges, hence the dissolve).
    cty = d.drop_duplicates("countyid")[["countyid", "county_name"]]
    cty = M.match_counties(cty, g)
    cty = cty[cty.map_code.notna()].copy()
    cty["map_code"] = cty.map_code.astype(int)
    gg = g.merge(cty[["countyid", "map_code"]], left_on="code", right_on="map_code",
                 how="inner")
    gg = gg.dissolve(by="countyid", as_index=False, aggfunc="first")
    gg["cid"] = gg.countyid.astype(int).astype(str)
    gg["geometry"] = gg.geometry.simplify(a.simplify, preserve_topology=True)
    gj = json.loads(gg[["cid", "geometry"]].to_json())
    for f in gj["features"]:
        f["id"] = f["properties"]["cid"]
    matched = set(gg.countyid.astype(int))
    print(f"boundaries: {len(gg):,d} polygons for {d.countyid.nunique():,d} "
          f"counties ({d.countyid.nunique() - len(matched):,d} without geometry)")
    d = d[d.countyid.isin(matched)]

    # Counties present in the 2010 boundaries but NOT in the agricultural panel.
    # These are what the map would otherwise leave white, which is ambiguous:
    # hatching says "excluded from the sample", not "value missing".
    used = set(cty.map_code.astype(int))
    gexc = g[~g["code"].isin(used)].copy()
    gexc["geometry"] = gexc.geometry.simplify(a.simplify, preserve_topology=True)
    hatch = hatch_lines(gexc)
    print(f"hatched: {len(gexc):,d} counties outside the agricultural sample"
          + (f", {sum(1 for v in hatch['lon'] if v is None):,d} hatch segments"
             if hatch else " (hatch skipped)"))

    RAW = [("raw_real_gvp", "real GVP (raw, 2005 prices)", True),
           ("raw_Laborday_impute", "labour days (raw)", True),
           ("raw_Land_serv_q", "land service (raw)", True),
           ("raw_capital_serv_q", "capital service (raw)", True),
           ("raw_Inter_all_real", "intermediates (raw)", True)]
    # Cropland share belongs on the CLEANED map too, not only the screening one:
    # it is the variable that decided whether the county is on the map at all,
    # so reading an input value without it invites treating a marginal county as
    # if it were a core one.  NOTE the per-year layer covers 1986-2015 only --
    # the rasters do not run to the panel's 1981-2016 — so it is blank at both
    # ends while the per-county summary columns are not.
    CLEAN = [("real_gvp", "real GVP (cleaned)", True),
             ("Laborday_impute", "labour days", True),
             ("Land_serv_q", "land service", True),
             ("capital_serv_q", "capital service", True),
             ("Inter_all_real", "intermediates", True),
             ("cropland_pct", "cropland share this year, %", False),
             ("cropland_max_pct", "cropland share, county max %", False),
             ("cropland_yrs_above15", "years above 15% (of 30)", False),
             ("is_ag_always", "above 15% in EVERY year = 1", False)]
    SCREEN = [("cropland_pct", "cropland share of county area, %", False),
              ("meat_vs_grain", "meat / (meat + grain), physical", False),
              ("grain_t", "grain output, tonnes", True),
              ("vege_area", "vegetable area, ha", True),
              ("fruit_area", "fruit area, ha", True),
              ("irrigated_area", "irrigated area, ha", True),
              ("hull_vertex_count", "DEA hull vertex count (0-4)", False),
              ("is_district", "urban district (区) = 1", False)]

    print("writing maps ...")
    sub = (f"{d.countyid.nunique():,d} agricultural counties, "
           f"{d.year.min()}-{d.year.max()}; hover shows every variable; "
           "hatched = outside the sample")
    write_map(d, RAW, gj, "dash_raw", "RAW county series, 2005 deflation only",
              "no unit fix, no imputation, no rescaling — the source as it "
              "arrives. " + sub, hatch)
    write_map(d, CLEAN, gj, "dash_clean", "CLEANED county panel",
              "county_panel_clean.csv — unit fixes, boundary-run rescaling, "
              "spike imputation, extreme-low floor. " + sub, hatch)
    write_map(d, SCREEN, gj, "dash_screen", "Sample-screening indicators",
              "what decides membership: cropland share, crop-vs-livestock mix, "
              "vegetable and fruit area (kept separate: corr = -0.011), "
              "irrigation, county type, frontier leverage. " + sub, hatch)

    keep = ["countyid", "county_name", "SID", "region", "year", "is_district",
            "cropland_pct", "cropland_max_pct", "cropland_mean_pct",
            "cropland_yrs_above15", "is_ag_always", "meat_vs_grain", "grain_t", "vege_area", "fruit_area",
            "irrigated_area", "hull_vertex_count", "real_gvp",
            "Laborday_impute", "Land_serv_q", "capital_serv_q", "Inter_all_real"]
    d[[c for c in keep if c in d.columns]].to_csv(
        os.path.join(C.DQ_DIR, "county_screening_panel.csv"),
        index=False, encoding="utf-8-sig")
    print(f"saved -> dq/county_screening_panel.csv\ndashboards -> {OUT}")


if __name__ == "__main__":
    main()
