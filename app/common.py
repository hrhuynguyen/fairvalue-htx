"""Shared data loaders, palette, and chart helpers for every dashboard page."""
from pathlib import Path
import html
import json
import re
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from build_dataset import HUBS, haversine_mi  # noqa: E402
from features import FEATURES, LISTING_FLAGS, CompsIndex, prepare  # noqa: E402

# Palette (validated reference palette). Blue/red are reserved for under/over fair value.
BLUE, RED, GRAY = "#2a78d6", "#e34948", "#a3a29c"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#c3c2b7", "#e8e7e3"
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # one-hue sequential
FLAG_COLOR = {"Underpriced": BLUE, "Fair": GRAY, "Overpriced": RED, "Verify first": INK_2}
UNUSUAL_GAP = (-0.40, 0.60)  # beyond ~1st/99th pct: mostly foreclosures, fixer-uppers, acreage, data errors

HUB_LABELS = {"downtown": "Downtown", "tmc": "Texas Medical Center", "uh": "University of Houston",
              "galleria": "Galleria / Uptown", "energy_corridor": "Energy Corridor"}

# SHAP features rolled up into explanations a buyer understands (SHAP values are additive).
DRIVER_GROUPS = {
    "Living area": ["livingArea", "sqft_per_bed"],
    "Beds & baths": ["bedrooms", "bathrooms", "bed_bath_ratio"],
    "Lot size": ["lot_sqft"],
    "Year built": ["yearBuilt"],
    "Home type": ["homeType"],
    "Location & commute": ["latitude", "longitude", "zipcode", "dist_downtown_mi", "dist_tmc_mi",
                           "dist_galleria_mi", "dist_energy_corridor_mi", "dist_uh_mi"],
    "Nearby comparable homes": ["knn_log_ppsf", "knn_log_price", "knn_dist_mi"],
    "ZIP income & market trend": ["zip_avg_agi", "zip_tax_returns", "zip_zhvi", "zip_zhvi_1y_growth",
                                  "zip_zhvi_5y_growth", "zip_zori"],
    "New build / builder / foreclosure": LISTING_FLAGS,
}

# Zillow's published nationwide Zestimate accuracy (zillow.com/zestimate, accessed Oct 2026).
ZILLOW_ON_MARKET_MDAPE = 1.83
ZILLOW_OFF_MARKET_MDAPE = 7.01


@st.cache_data
def load_scored():
    df = pd.read_parquet(ROOT / "data/processed/scored.parquet")
    for hub in ("galleria", "energy_corridor"):
        lat, lon = HUBS[hub]
        df[f"dist_{hub}_mi"] = haversine_mi(df.latitude, df.longitude, lat, lon)
    df.loc[~df.gap_pct.between(*UNUSUAL_GAP), "flag"] = "Verify first"
    df["ppsf"] = df.price / df.livingArea
    return df


@st.cache_data
def load_listings():
    return pd.read_parquet(ROOT / "data/processed/listings.parquet")


@st.cache_resource
def load_model():
    bundle = joblib.load(ROOT / "models/fair_value.joblib")
    bundle["comps"] = CompsIndex(load_listings())
    return bundle


@st.cache_data
def load_json(name):
    return json.load(open(ROOT / "reports" / name))


def default_pool(df, n=500):
    """Listings to pick from before the user has searched: starter-home bargains (Maya's story)."""
    return list(df[(df.flag == "Underpriced") & (df.price <= 400_000)].sort_values("gap_pct").index[:n])


def money(x):
    return f"${x:,.0f}"


def md_money(x):
    """Dollar amount safe for st.markdown/caption, where a bare $...$ pair renders as LaTeX."""
    return money(x).replace("$", "\\$")


def style(fig, height=320, **layout):
    """House style: transparent surface, recessive grid, muted ink, no chart junk."""
    base = dict(height=height, margin=dict(l=10, r=40, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color=INK_2, size=12),
                showlegend=False, hoverlabel=dict(bgcolor="white", font_color=INK, font_size=13))
    fig.update_layout(**{**base, **layout})
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=MUTED)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=MUTED)
    return fig


def fmt_value(v, fmt):
    """Python format spec plus a leading '$' (d3-style), e.g. '$,.0f'."""
    return "$" + format(v, fmt[1:]) if fmt.startswith("$") else format(v, fmt)


def hbar(labels, values, fmt, color=BLUE, height=None, hover=None, colors=None):
    """Horizontal bar chart with value labels at the bar ends."""
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors or color,
                           text=[fmt_value(v, fmt) for v in values], textposition="outside", cliponaxis=False,
                           hovertemplate=hover or "%{y}: %{x:" + fmt + "}<extra></extra>"))
    return style(fig, height or 34 * len(labels) + 50, yaxis=dict(autorange="reversed"))


def monthly_cost(price, rate, down, tax_rate, ins_rate, hoa, flood_yr):
    """All-in monthly cost of owning: principal & interest, tax, insurance, PMI, HOA, flood."""
    loan = price * (1 - down)
    r = rate / 12
    return {
        "Principal & interest": loan * r / (1 - (1 + r) ** -360),
        "Property tax": price * tax_rate / 12,
        "Homeowners insurance": price * ins_rate / 12,
        "PMI": loan * 0.005 / 12 if down < 0.20 else 0 * price,
        "HOA": hoa + 0 * price,
        "Flood insurance": flood_yr / 12 + 0 * price,
    }


DEFAULT_COSTS = dict(rate=0.065, down=0.10, tax_rate=0.022, ins_rate=0.010, hoa=50, flood_yr=800)


def max_affordable_price(income, share=0.28, **assume):
    """Highest price whose all-in monthly cost fits within `share` of gross income."""
    a = {**DEFAULT_COSTS, **assume}
    lo, hi = 0.0, 10_000_000.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sum(monthly_cost(mid, **a).values()) <= income * share / 12:
            lo = mid
        else:
            hi = mid
    return lo


def cost_inputs(key):
    with st.expander("Ownership cost assumptions (Texas defaults, edit to match your quote)", expanded=False):
        c = st.columns(3)
        rate = c[0].number_input("Mortgage rate %", 2.0, 12.0, 6.5, 0.125, key=f"{key}_rate") / 100
        down = c[1].number_input("Down payment %", 0.0, 100.0, 10.0, 0.5, key=f"{key}_down") / 100
        tax = c[2].number_input("Property tax % / yr", 0.5, 4.0, 2.2, 0.1, key=f"{key}_tax",
                                help="Harris County effective rates are typically ~2.0–2.8% incl. school & MUD districts") / 100
        c = st.columns(3)
        ins = c[0].number_input("Home insurance % of price / yr", 0.2, 3.0, 1.0, 0.1, key=f"{key}_ins",
                                help="Houston windstorm/hail risk makes premiums well above the national average") / 100
        hoa = c[1].number_input("HOA $ / month", 0, 2000, 50, 10, key=f"{key}_hoa")
        flood = c[2].number_input("Flood insurance $ / yr", 0, 10000, 800, 50, key=f"{key}_flood",
                                  help="Lenders require it in FEMA flood zones; many Houston homes outside them flooded in Harvey")
    return dict(rate=rate, down=down, tax_rate=tax, ins_rate=ins, hoa=hoa, flood_yr=flood)


def driver_waterfall(base, phi: dict, fair_value):
    """Turn log-space SHAP into dollar steps that add up exactly to the fair value."""
    grouped = {g: sum(phi.get(f, 0.0) for f in feats) for g, feats in DRIVER_GROUPS.items()}
    level, steps = base, []
    for g in sorted(grouped, key=lambda g: -abs(grouped[g])):
        steps.append((g, np.exp(level + grouped[g]) - np.exp(level)))
        level += grouped[g]
    assert abs(np.exp(level) - fair_value) / fair_value < 1e-3
    return np.exp(base), steps


def plot_waterfall(start, steps, fair_value):
    labels = ["Average listing (starting point)"] + [s[0] for s in steps] + ["Fair value"]
    values = [start] + [s[1] for s in steps] + [fair_value]
    fig = go.Figure(go.Waterfall(
        orientation="h", y=labels, x=values,
        measure=["absolute"] + ["relative"] * len(steps) + ["total"],
        text=[money(start)] + [("+" if v >= 0 else "−") + money(abs(v)) for _, v in steps] + [money(fair_value)],
        textposition="outside", cliponaxis=False,
        # Neutral grays: blue/red already mean under/overpriced elsewhere in the app.
        increasing={"marker": {"color": INK_2}}, decreasing={"marker": {"color": MUTED}},
        totals={"marker": {"color": INK}}, connector={"line": {"color": GRID, "width": 1}},
        hovertemplate="%{y}: %{text}<extra></extra>",
    ))
    fig = style(fig, 34 * len(labels) + 50, yaxis=dict(autorange="reversed"))
    fig.update_xaxes(tickformat="$,.0s", nticks=5)
    fig.update_layout(margin=dict(l=10, r=90, t=10, b=10))
    return fig


def plot_range(price, lo, mid, hi):
    """Listing price against the model's 80% fair-value range."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[0, 0], mode="lines", line=dict(color=GRAY, width=14),
                             hovertemplate="80% fair range<extra></extra>", name="80% fair range"))
    fig.add_trace(go.Scatter(x=[mid], y=[0], mode="markers", name="Fair value",
                             marker=dict(color=INK_2, size=14, symbol="line-ns-open", line=dict(width=3)),
                             hovertemplate="Fair value %{x:$,.0f}<extra></extra>"))
    color = BLUE if price < lo else RED if price > hi else INK
    fig.add_trace(go.Scatter(x=[price], y=[0], mode="markers", name="List price",
                             marker=dict(color=color, size=16, symbol="diamond", line=dict(color="white", width=2)),
                             hovertemplate="List price %{x:$,.0f}<extra></extra>"))
    pad = (max(hi, price) - min(lo, price)) * 0.15
    fig = style(fig, 150, showlegend=True, legend=dict(orientation="h", y=1.3, x=0))
    fig.update_xaxes(range=[min(lo, price) - pad, max(hi, price) + pad], tickformat="$,.0f")
    fig.update_yaxes(visible=False)
    return fig


CSS = """
<style>
.block-container {max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem;}
h1 {letter-spacing: -0.02em;}
h3 {margin-top: 1.2rem;}
[data-testid="stSidebarNav"] span, [data-testid="stSidebarNavLink"] span {font-size: 1rem;}
[data-testid="stMetricLabel"] p {font-size: 1rem !important; color: #52514e;}
[data-testid="stPageLink-NavLink"] p {color: #2a78d6; font-weight: 600;}
[data-testid="stCaptionContainer"] p {font-size: 0.95rem;}
table {font-size: 0.95rem;}
.fv-kicker {font-size: .8rem; letter-spacing: .14em; text-transform: uppercase; color: #8a877f; font-weight: 600; margin-bottom: -.6rem;}
.fv-lead {font-size: 1.1rem; line-height: 1.55; color: #52514e; max-width: 54rem; margin: .1rem 0 1.4rem;}
.fv-insight {border-left: 4px solid #2a78d6; background: #ffffff; padding: 1rem 1.3rem; border-radius: 0 .75rem .75rem 0;
             margin: .4rem 0 1.8rem; color: #2d2c2a; line-height: 1.6; box-shadow: 0 1px 2px rgba(0,0,0,.05);}
.fv-tag {display: block; font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; font-weight: 700; margin-bottom: .35rem;}
.fv-insight .fv-tag {color: #2a78d6;}
.fv-takeaway {background: #1a1a19; color: #e9e7e1; padding: 1.5rem 1.8rem; border-radius: 1rem; margin: 2.2rem 0 1rem;
              font-size: 1.05rem; line-height: 1.55;}
.fv-takeaway .fv-tag {color: #86b6ef;}
.fv-takeaway b {color: #ffffff;}
.fv-stats {display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: .8rem 0 1.6rem;}
.fv-stat {background: #ffffff; border: 1px solid #e6e3dc; border-radius: 1rem; padding: 1.1rem 1.3rem;}
.fv-stat-value {font-family: Fraunces, Georgia, serif; font-size: 1.85rem; line-height: 1.1; color: #1a1a19; font-weight: 600;}
.fv-stat-value.accent {color: #2a78d6;}
.fv-stat-label {font-size: .98rem; color: #52514e; margin-top: .4rem; font-weight: 500;}
.fv-stat-note {font-size: .82rem; color: #8a877f; margin-top: .2rem;}
.fv-steps {display: flex; gap: 1rem; flex-wrap: wrap; margin: .5rem 0 1rem;}
.fv-step {flex: 1 1 105px; border-top: 3px solid #2a78d6; padding: .7rem .1rem;}
.fv-step-n {font-family: Fraunces, Georgia, serif; font-size: 1.5rem; color: #2a78d6; line-height: 1;}
.fv-step-t {font-weight: 600; margin-top: .35rem;}
.fv-step-d {font-size: .88rem; color: #6b6964; line-height: 1.4;}
.fv-example {background: #eef4fc; border: 1px solid #cfe0f7; border-radius: .75rem; padding: .85rem 1.2rem;
              margin: .4rem 0 1.4rem; color: #1f3b63; line-height: 1.55;}
.fv-example .fv-tag {color: #2a78d6;}
.fv-guide .fv-step-d {font-size: .95rem; color: #2d2c2a;}
.fv-card {background: #ffffff; border: 1px solid #e6e3dc; border-radius: 1rem; padding: 1.1rem 1.3rem; height: 100%;}
.fv-card-t {font-weight: 600; margin-bottom: .3rem;}
.fv-card-d {font-size: .92rem; color: #52514e; line-height: 1.5;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def _inline(text):
    """Tiny markdown -> HTML for **bold** and *italic*, with $ made safe from LaTeX rendering."""
    text = html.escape(str(text), quote=False).replace("\\$", "&#36;").replace("$", "&#36;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)


def html_block(markup):
    st.markdown(markup, unsafe_allow_html=True)


def page_header(kicker, title, lead=None):
    html_block(f'<div class="fv-kicker">{_inline(kicker)}</div>')
    st.title(title)
    if lead:
        html_block(f'<p class="fv-lead">{_inline(lead)}</p>')


def step_header(step, total, title, one_liner=None):
    page_header(f"Step {step} of {total} · Data Science Task", title, one_liner)


def takeaway(text):
    html_block(f'<div class="fv-takeaway"><span class="fv-tag">Key takeaway</span>{_inline(text)}</div>')


def insight(text):
    html_block(f'<div class="fv-insight"><span class="fv-tag">What we learned</span>{_inline(text)}</div>')


def stat_tiles(items):
    """items: (value, label, note or None, accent?) tuples rendered as big-number tiles."""
    cells = []
    for value, label, note, *accent in items:
        cls = "fv-stat-value accent" if accent and accent[0] else "fv-stat-value"
        cells.append(f'<div class="fv-stat"><div class="{cls}">{_inline(value)}</div>'
                     f'<div class="fv-stat-label">{_inline(label)}</div>'
                     + (f'<div class="fv-stat-note">{_inline(note)}</div>' if note else "") + "</div>")
    html_block(f'<div class="fv-stats">{"".join(cells)}</div>')


def table(df):
    """Static, wrapping table (st.table) with $ escaped so cell text never renders as LaTeX."""
    st.table(df.map(lambda v: v.replace("$", "\\$") if isinstance(v, str) else v)
             .rename(index=lambda i: i.replace("$", "\\$") if isinstance(i, str) else i))


REDS = ["#fde8e6", "#f8c3bd", "#f09189", "#e35d55", "#c23a33", "#8e2420"]  # one-hue ramp: light = low, dark = high


@st.cache_data
def zip_shapes():
    """Houston ZIP (Census ZCTA) boundaries, trimmed to our 159 ZIP codes."""
    return json.load(open(ROOT / "data/external/houston_zip_geo.json"))


def zip_choropleth(stats: pd.DataFrame, value: str, fmt, label: str, min_n=10, height=460):
    """ZIP map shaded light -> dark red by `value` (6 quantile bins) plus a legend.

    stats: indexed by ZIP with columns [value, "n", "city"]. ZIPs with fewer than `min_n` listings are gray,
    because a median from a handful of homes is noise. Returns the bin edges.
    """
    solid = stats[stats.n >= min_n]
    edges = solid[value].quantile(np.linspace(0, 1, 7)).to_numpy()
    bins = pd.cut(stats[value].clip(edges[0], edges[-1]), edges, labels=False, include_lowest=True)
    geo = zip_shapes()
    for f in geo["features"]:
        z = f["properties"]["zipcode"]
        r = stats.loc[z]
        few = r["n"] < min_n
        f["properties"].update(fill=[200, 198, 192, 120] if few else hex_rgb(REDS[int(bins[z])], 215), city=r["city"],
                               val=fmt(r[value]) + (" (too few listings)" if few else ""), n=int(r["n"]))
    st.pydeck_chart(pdk.Deck(
        map_provider="carto", map_style="light",
        initial_view_state=pdk.ViewState(latitude=29.80, longitude=-95.45, zoom=8.3),
        layers=[pdk.Layer("GeoJsonLayer", geo, get_fill_color="properties.fill", get_line_color=[255, 255, 255],
                          line_width_min_pixels=1, stroked=True, filled=True, pickable=True, auto_highlight=True,
                          highlight_color=[11, 11, 11, 70])],
        tooltip={"html": f"<b>{{zipcode}} · {{city}}</b><br/>{label} {{val}}<br/>{{n}} listings"},
    ), height=height)
    swatch = '<span style="width:22px;height:14px;background:{};border-radius:3px;margin-right:6px;display:inline-block"></span>'
    items = [f'<span style="display:inline-flex;align-items:center;margin-right:14px">{swatch.format(c)}'
             f'{fmt(lo).replace("$", "&#36;")}–{fmt(hi).replace("$", "&#36;")}</span>'
             for c, lo, hi in zip(REDS, edges[:-1], edges[1:])]
    items.append(f'<span style="display:inline-flex;align-items:center">{swatch.format("#c8c6c0")}fewer than {min_n} listings</span>')
    html_block(f'<div style="font-size:.9rem;color:#52514e;margin:-.4rem 0 .8rem">{"".join(items)}</div>')
    return edges


def blend_mind_map(final: dict, members: list) -> str:
    """Radial SVG: the final blend in the center, one branch per model.

    final: {"mdape", "r2", "within10"}; members: dicts with name, mdape, r2, weight, trait, pos (x, y).
    Branch thickness is proportional to the model's weight; a weight of 0 draws a dashed, muted branch.
    """
    cx, cy, w, h = 500, 290, 300, 128
    parts = []
    for m in members:  # connectors first so cards sit on top
        x, y = m["pos"]
        mid = cx + (x - cx) * 0.55
        dropped = m["weight"] < 0.005
        stroke = MUTED if dropped else BLUE
        width = 1.5 if dropped else 2 + 20 * m["weight"]
        dash = ' stroke-dasharray="6 6"' if dropped else ""
        parts.append(f'<path d="M{cx},{cy} C{mid},{cy} {mid},{y} {x},{y}" fill="none" stroke="{stroke}" '
                     f'stroke-width="{width:.1f}" stroke-linecap="round" opacity="{0.9 if not dropped else 1}"{dash}/>')
    parts.append(f'<rect x="{cx - 125}" y="{cy - 72}" width="250" height="144" rx="18" fill="{INK}"/>'
                 f'<text x="{cx}" y="{cy - 38}" class="mm-kick" fill="#86b6ef">THE BLEND</text>'
                 f'<text x="{cx}" y="{cy + 12}" class="mm-big" fill="#ffffff">{final["mdape"]:.2f}%</text>'
                 f'<text x="{cx}" y="{cy + 38}" class="mm-sub mm-mid" fill="#c3c2b7">median error · R² {final["r2"]:.3f}</text>'
                 f'<text x="{cx}" y="{cy + 58}" class="mm-sub mm-mid" fill="#c3c2b7">{final["within10"]:.0f}% of homes within ±10%</text>')
    for m in members:
        x, y = m["pos"]
        dropped = m["weight"] < 0.005
        chip_fill, chip_text = (GRID, INK_2) if dropped else ("#e5effb", BLUE)
        chip = "dropped (0%)" if dropped else f"{m['weight']:.0%} of the blend"
        parts.append(
            f'<g><title>{m["name"]}: median error {m["mdape"]:.2f}%, R² {m["r2"]:.3f}, weight {m["weight"]:.1%}</title>'
            f'<rect x="{x - w / 2}" y="{y - h / 2}" width="{w}" height="{h}" rx="16" fill="#ffffff" '
            f'stroke="{MUTED if dropped else "#e6e3dc"}"{" stroke-dasharray=" + chr(34) + "5 5" + chr(34) if dropped else ""}/>'
            f'<text x="{x - w / 2 + 20}" y="{y - h / 2 + 32}" class="mm-name" fill="{INK_2 if dropped else INK}">{m["name"]}</text>'
            f'<rect x="{x + w / 2 - 20 - (len(chip) * 6.6 + 18)}" y="{y - h / 2 + 15}" width="{len(chip) * 6.6 + 18}" height="24" '
            f'rx="12" fill="{chip_fill}"/>'
            f'<text x="{x + w / 2 - 20 - (len(chip) * 6.6 + 18) / 2}" y="{y - h / 2 + 31.5}" class="mm-chip" fill="{chip_text}">{chip}</text>'
            f'<text x="{x - w / 2 + 20}" y="{y + 16}" class="mm-score" fill="{INK_2 if dropped else INK}">{m["mdape"]:.2f}%</text>'
            f'<text x="{x - w / 2 + 132}" y="{y + 9}" class="mm-sub" fill="{INK_2}">median error alone</text>'
            f'<text x="{x - w / 2 + 132}" y="{y + 27}" class="mm-sub" fill="{INK_2}">R² {m["r2"]:.3f}</text>'
            f'<text x="{x - w / 2 + 20}" y="{y + h / 2 - 16}" class="mm-trait" fill="#8a877f">{m["trait"]}</text></g>')
    style_block = ("<style>.mm text{font-family:Inter,sans-serif}"
                   ".mm .mm-big,.mm .mm-score{font-family:Fraunces,Georgia,serif;font-weight:600}"
                   ".mm .mm-big{font-size:44px;text-anchor:middle}.mm .mm-score{font-size:32px}"
                   ".mm .mm-kick{font-size:12px;letter-spacing:.14em;font-weight:700;text-anchor:middle}"
                   ".mm .mm-sub{font-size:13px}.mm .mm-mid{text-anchor:middle}"
                   ".mm .mm-name{font-size:19px;font-weight:600}"
                   ".mm .mm-chip{font-size:11.5px;font-weight:600;text-anchor:middle}.mm .mm-trait{font-size:12.5px}</style>")
    return (f'<svg class="mm" viewBox="0 0 1000 580" width="100%" role="img" '
            f'aria-label="Mind map of the model blend">{style_block}{"".join(parts)}</svg>')


def app_guide(steps, example):
    """Three numbered how-to steps and a worked example, shown at the top of every Application page."""
    html_block('<div class="fv-steps fv-guide">' + "".join(
        f'<div class="fv-step"><div class="fv-step-n">{i}</div><div class="fv-step-d">{_inline(t)}</div></div>'
        for i, t in enumerate(steps, 1)) + "</div>")
    html_block(f'<div class="fv-example"><span class="fv-tag">Example</span>{_inline(example)}</div>')


@st.cache_data
def zip_centers():
    """ZIP codes with at least 10 listings: center point (median coordinates) and main city."""
    z = load_listings().groupby("zipcode").agg(lat=("latitude", "median"), lon=("longitude", "median"),
                                               city=("city", lambda c: c.mode().iat[0]), n=("zpid", "size"))
    return z[z.n >= 10]


def price_homes(specs):
    """Fair value and 80% range for described homes (listed or not).

    specs: dicts with zipcode, homeType, sqft, beds, baths, year and optional lot, new.
    Returns (model-ready X, fair value, low, high) as arrays in the order of `specs`.
    """
    model, zc = load_model(), zip_centers()
    template = load_listings().drop_duplicates("zipcode").set_index("zipcode")  # carries the ZIP-level features
    rows = pd.DataFrame([{**template.loc[sp["zipcode"]].to_dict(), "zipcode": sp["zipcode"],
                          "latitude": zc.at[sp["zipcode"], "lat"], "longitude": zc.at[sp["zipcode"], "lon"],
                          "homeType": sp["homeType"], "livingArea": sp["sqft"], "bedrooms": sp["beds"],
                          "bathrooms": sp["baths"], "yearBuilt": sp["year"], "lot_sqft": sp.get("lot") or np.nan,
                          "is_new_construction": sp.get("new", False), "has_builder": sp.get("new", False),
                          "is_premier_builder": False, "is_bank_owned": False, "is_auction": False} for sp in specs])
    for name, (lat, lon) in HUBS.items():
        rows[f"dist_{name}_mi"] = haversine_mi(rows.latitude, rows.longitude, lat, lon)
    rows["bed_bath_ratio"] = rows.bedrooms / rows.bathrooms.replace(0, np.nan)
    rows["sqft_per_bed"] = rows.livingArea / rows.bedrooms.replace(0, np.nan)
    rows = rows.drop(columns=["knn_log_ppsf", "knn_log_price", "knn_dist_mi"], errors="ignore")
    rows = rows.join(model["comps"].query(rows, exclude_self=False))
    X = prepare(rows, model["categories"])[FEATURES]
    mid_log = model["point"].predict(X)
    band = np.searchsorted(model["band_edges"], mid_log)  # interval calibration is per price band
    mid = np.exp(mid_log)
    lo = np.minimum(np.exp(model["lo"].predict(X) - model["band_adj"][band, 0]), mid)
    hi = np.maximum(np.exp(model["hi"].predict(X) + model["band_adj"][band, 1]), mid)
    return X, mid, lo, hi


@st.cache_data
def zip_value_history():
    """Zillow typical home value per ZIP, monthly (Houston-area extract)."""
    z = pd.read_parquet(ROOT / "data/processed/zhvi_houston.parquet").set_index("RegionName")
    months = [c for c in z.columns if c[:2] == "20"]
    out = z[months].T
    out.index = pd.to_datetime(out.index)
    return out


def cards(items):
    """items: (title, description) tuples rendered as a row of quiet cards."""
    cols = st.columns(len(items))
    for col, (t, d) in zip(cols, items):
        with col:
            html_block(f'<div class="fv-card"><div class="fv-card-t">{_inline(t)}</div><div class="fv-card-d">{_inline(d)}</div></div>')


def hex_rgb(h, alpha=210):
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)] + [alpha]
