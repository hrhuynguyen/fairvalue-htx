import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from common import (BLUE, INK_2, REDS, ROOT, SEQ, hbar, hex_rgb, html_block, insight, load_listings, load_scored,
                    max_affordable_price, md_money, money, stat_tiles, step_header, style, takeaway, zip_choropleth, zip_shapes)

df = load_listings().copy()
df["ppsf"] = df.price / df.livingArea

step_header(2, 7, "Exploring the data")

# ---------------------------------------------------------------- Q1
st.markdown("### Q1. What does a typical Houston listing cost?")
stat_tiles([
    (money(df.price.median()), "median asking price", None, True),
    (money(df.price.mean()), "average asking price", "pulled up by a few mansions"),
    (f"${df.price.quantile(.25) / 1e3:.0f}k–{df.price.quantile(.75) / 1e3:.0f}k", "middle 50% of homes", None),
    (f"{df.livingArea.median():,.0f}", "median size (sqft)", None),
])

st.code(f"""df["price"].median()                # -> {df.price.median():,.0f}
df["price"].mean()                  # -> {df.price.mean():,.0f}
df["price"].quantile([.25, .75])    # -> {df.price.quantile(.25):,.0f} / {df.price.quantile(.75):,.0f}
df["price"].skew()                  # -> {df.price.skew():.1f}   (very right-skewed)
np.log(df["price"]).skew()          # -> {np.log(df.price).skew():.1f}   (close to symmetric, so we model log price)""",
        language="python")

edges = np.logspace(np.log10(50_000), np.log10(15_000_000), 60)
counts, _ = np.histogram(df.price, bins=edges)
mids = np.sqrt(edges[:-1] * edges[1:])
fig = go.Figure(go.Bar(x=mids, y=counts, width=np.diff(edges) * 0.92, marker_color=BLUE,
                       customdata=np.stack([edges[:-1], edges[1:]], axis=1),
                       hovertemplate="%{customdata[0]:$,.0f} – %{customdata[1]:$,.0f}<br>%{y} homes<extra></extra>"))
for val, name in [(df.price.median(), "median"), (df.price.mean(), "average")]:
    fig.add_vline(x=val, line=dict(color=INK_2, width=1.5, dash="dot"))
    fig.add_annotation(x=np.log10(val), y=counts.max(), text=f"{name} {money(val)}", showarrow=False,
                       yshift=10, xanchor="left" if name == "average" else "right", font=dict(color=INK_2))
style(fig, 300)
ticks = [50e3, 100e3, 200e3, 300e3, 500e3, 1e6, 2e6, 5e6, 10e6]
fig.update_xaxes(type="log", tickvals=ticks, ticktext=["$50k", "$100k", "$200k", "$300k", "$500k", "$1M", "$2M", "$5M", "$10M"],
                 title="List price (log scale)")
fig.update_yaxes(title="Number of homes")
st.plotly_chart(fig, width="stretch")
insight(f"Prices are very **right-skewed**: a few mansions pull the average ({md_money(df.price.mean())}) far above the median "
        f"({md_money(df.price.median())}). Skewness drops from **{df.price.skew():.1f}** to **{np.log(df.price).skew():.1f}** on a "
        "log scale. So the model predicts the *log* of price, which makes it care about % errors, not dollar errors on mansions.")

# ---------------------------------------------------------------- Q2
st.markdown("### Q2. Which home features move with price?")
labels = {"livingArea": "Living area (sqft)", "bathrooms": "Bathrooms", "zip_zhvi": "ZIP typical home value",
          "zip_avg_agi": "ZIP average income", "bedrooms": "Bedrooms", "yearBuilt": "Year built",
          "lot_sqft": "Lot size", "dist_downtown_mi": "Miles to downtown", "zip_zhvi_5y_growth": "ZIP 5-year price growth"}
corr = pd.Series({v: df[k].corr(np.log(df.price), method="spearman") for k, v in labels.items()}).sort_values(ascending=False)
corr_code = "\n".join(f"# {k:<18} {df[k].corr(np.log(df.price), method='spearman'):+.2f}"
                      for k in sorted(labels, key=lambda k: -df[k].corr(np.log(df.price), method="spearman")))
st.code(f"""cols = {list(labels)[:5]}
cols += {list(labels)[5:]}
df[cols].corrwith(np.log(df["price"]), method="spearman")
{corr_code}""", language="python")

c1, c2 = st.columns([1, 1.2])
with c1:
    st.plotly_chart(hbar(list(corr.index), list(corr.values), "+.2f",
                         hover="%{y}<br>rank correlation with price %{x:+.2f}<extra></extra>"), width="stretch")
    st.caption("Spearman rank correlation with price: +1 means 'always higher together', 0 means 'no simple relationship'.")
with c2:
    s = df[df.livingArea.notna()].sample(4000, random_state=0)
    fig = go.Figure(go.Scattergl(x=s.livingArea, y=s.price, mode="markers",
                                 marker=dict(color=BLUE, size=5, opacity=0.35),
                                 hovertemplate="%{x:,.0f} sqft · %{y:$,.0f}<extra></extra>"))
    style(fig, 340)
    fig.update_xaxes(type="log", title="Living area (sqft, log scale)")
    fig.update_yaxes(type="log", title="List price (log scale)", tickformat="$,.0s")
    st.plotly_chart(fig, width="stretch")
    st.caption("4,000 random homes. Bigger homes cost more, but the cloud is wide: same size, very different prices.")
insight("**Size is king** (0.79), and bathrooms matter more than bedrooms. But *miles to downtown* shows almost no "
        "simple correlation, which doesn't mean location is unimportant. It means the relationship isn't a straight line (Q3).")

# ---------------------------------------------------------------- Q3
st.markdown("### Q3. Where are homes expensive?")
bins = [0, 5, 10, 15, 20, 30, 45]
names = ["0–5 mi", "5–10", "10–15", "15–20", "20–30", "30–45"]
df["band"] = pd.cut(df.dist_downtown_mi, bins, labels=names)
g = df.groupby("band", observed=True).agg(ppsf=("ppsf", "median"), sqft=("livingArea", "median"),
                                          year=("yearBuilt", "median"), n=("price", "size"))
band_code = "\n".join(f"# {b:<7} ${r.ppsf:,.0f}/sqft   {r.sqft:,.0f} sqft   built {r.year:.0f}" for b, r in g.iterrows())
st.code(f"""df["ppsf"] = df["price"] / df["livingArea"]
df["band"] = pd.cut(df["dist_downtown_mi"], [0, 5, 10, 15, 20, 30, 45])   # miles from downtown
df.groupby("band")[["ppsf", "livingArea", "yearBuilt"]].median()
{band_code}""", language="python")

c1, c2 = st.columns(2)
with c1:
    fig = go.Figure(go.Bar(x=names, y=g.ppsf, marker_color=BLUE, text=[f"${v:.0f}" for v in g.ppsf], textposition="outside",
                           cliponaxis=False, customdata=g.n, hovertemplate="%{x} from downtown<br>median $%{y:.0f}/sqft<br>%{customdata:,} homes<extra></extra>"))
    style(fig, 300, title=dict(text="Median price per sqft", font=dict(size=14)))
    fig.update_xaxes(title="Distance from downtown")
    st.plotly_chart(fig, width="stretch")
with c2:
    fig = go.Figure(go.Bar(x=names, y=g.sqft, marker_color=SEQ[3], text=[f"{v:,.0f}" for v in g.sqft], textposition="outside",
                           cliponaxis=False, hovertemplate="%{x} from downtown<br>median %{y:,.0f} sqft<extra></extra>"))
    style(fig, 300, title=dict(text="Median home size (sqft)", font=dict(size=14)))
    fig.update_xaxes(title="Distance from downtown")
    st.plotly_chart(fig, width="stretch")

st.code('df.groupby("zipcode")["ppsf"].median()   # map below: median price per sqft by ZIP code', language="python")
zs = df.groupby("zipcode").agg(ppsf=("ppsf", "median"), n=("price", "size"), city=("city", lambda c: c.mode().iat[0]))
zip_choropleth(zs, "ppsf", lambda v: f"${v:,.0f}/sqft", "median price per sqft")
insight(f"Close to downtown you pay the most **per square foot** (about \\${g.ppsf.iloc[0]:.0f}), and it's cheapest 10–15 miles out "
        f"(about \\${g.ppsf.iloc[2]:.0f}). Farther suburbs climb again because their homes are **bigger and newer** "
        f"(median {g.sqft.iloc[4]:,.0f} sqft vs {g.sqft.iloc[2]:,.0f}). This U-shape is exactly why we chose a tree-based model, "
        "which can learn curves that a straight-line model cannot.")

# ---------------------------------------------------------------- Q4
st.markdown("### Q4. New construction vs. resale, and home types")
nc = df.groupby("is_new_construction").ppsf.median()
st.code(f"""df.groupby("is_new_construction")["ppsf"].median()
# resale            ${nc[False]:,.0f}/sqft
# new construction  ${nc[True]:,.0f}/sqft   (+{nc[True] / nc[False] - 1:.0%})
df.groupby("homeType")[["price", "ppsf", "livingArea"]].median()   # table on the right""", language="python")

c1, c2 = st.columns([1, 1.3])
with c1:
    n = df.groupby("is_new_construction").agg(ppsf=("ppsf", "median"), price=("price", "median"), n=("price", "size"))
    fig = go.Figure(go.Bar(x=["Resale", "New construction"], y=n.ppsf, marker_color=[SEQ[2], SEQ[4]],
                           text=[f"${v:.0f}/sqft" for v in n.ppsf], textposition="outside", cliponaxis=False,
                           customdata=np.stack([n.n, n.price], axis=1),
                           hovertemplate="%{x}<br>%{customdata[0]:,} homes · median %{customdata[1]:$,.0f}<extra></extra>"))
    style(fig, 300, title=dict(text="Median price per sqft", font=dict(size=14)))
    st.plotly_chart(fig, width="stretch")
with c2:
    t = df.groupby("homeType").agg(Homes=("price", "size"), Median_price=("price", "median"),
                                   Price_per_sqft=("ppsf", "median"), Median_sqft=("livingArea", "median")).sort_values("Homes", ascending=False)
    t.index = t.index.str.replace("_", " ").str.title()
    st.dataframe(t, width="stretch", column_config={
        "Median_price": st.column_config.NumberColumn("Median price", format="$%,.0f"),
        "Price_per_sqft": st.column_config.NumberColumn("$/sqft", format="$%.0f"),
        "Median_sqft": st.column_config.NumberColumn("Median sqft", format="%,.0f")})
share_new = df.is_new_construction.mean()
insight(f"**{share_new:.0%} of listings are brand-new builds**, a sign of Houston's suburban building boom. They carry about a "
        f"**{n.ppsf.iloc[1] / n.ppsf.iloc[0] - 1:.0%} premium per sqft**. Single-family homes are 85% of the market; condos are the "
        "cheapest way in.")

# ---------------------------------------------------------------- Q5
st.markdown("### Q5. Who can afford these homes?")
incomes = [50_000, 75_000, 100_000, 150_000, 200_000]
share = [(df.price <= max_affordable_price(i)).mean() for i in incomes]
afford_code = "\n".join(f"# {'$' + str(i // 1000) + 'k':<6} -> {s_:.0%} of listings" for i, s_ in zip(incomes, share))
st.code(f"""# max price whose all-in monthly cost (mortgage + tax + insurance + PMI + HOA + flood) <= 28% of income
for income in [{", ".join(f"{i:_}" for i in incomes)}]:
    (df["price"] <= max_affordable_price(income)).mean()
{afford_code}""", language="python")

c1, c2 = st.columns([1.2, 1])
with c1:
    fig = go.Figure(go.Bar(x=[f"${i // 1000}k" for i in incomes], y=share, marker_color=BLUE,
                           text=[f"{s:.0%}" for s in share], textposition="outside", cliponaxis=False,
                           hovertemplate="Income %{x}/yr<br>%{y:.0%} of listings affordable<extra></extra>"))
    style(fig, 300, title=dict(text="Share of listings affordable, by household income", font=dict(size=14)))
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_xaxes(title="Gross household income per year")
    st.plotly_chart(fig, width="stretch")
with c2:
    st.markdown("""
**How we define "affordable":** the *all-in* monthly cost is at most **28% of gross income**, the standard
lender rule of thumb.

All-in means mortgage (6.5%, 30 years, 10% down) **+ property tax (2.2%) + insurance + PMI + HOA + flood insurance**.

In Texas, tax and insurance alone are about **37% of the monthly payment** on a median-priced home.
""")
insight(f"On Maya's \\$75k salary, only **{share[1]:.0%} of listings** fit the 28% rule. Even at \\$100k it's {share[2]:.0%}. "
        "That's why our apps lead with *all-in monthly cost*, not sticker price.")

# ---------------------------------------------------------------- Q6
st.markdown("### Q6. How have Houston home values moved over time?")


@st.cache_data
def texas_zhvi():
    # Houston-area slice of Zillow's ZIP file, written by src/build_dataset.py
    return pd.read_parquet(ROOT / "data/processed/zhvi_houston.parquet").set_index("RegionName")


@st.cache_data
def metro_zhvi():
    z = texas_zhvi()
    z = z[z.Metro.fillna("").str.startswith("Houston")]
    months = [c for c in z.columns if c[:2] == "20"]
    q = z[months].quantile([0.25, 0.5, 0.75]).T
    q.index = pd.to_datetime(q.index)
    return q, len(z)


@st.cache_data
def zip_values_by_year(zips):
    """Typical home value (Zillow ZHVI) for each of our ZIP codes, every June 2000-2026."""
    z = texas_zhvi().reindex(list(zips))
    return pd.DataFrame({y: z[f"{y}-06-30"] for y in range(2000, 2027)})


ts, n_zips = metro_zhvi()
med = ts[0.5]
snap, last = med.loc["2024-05-31"], med.iloc[-1]
st.code(f"""zhvi = pd.read_csv("zhvi_zip.csv")                      # Zillow Home Value Index, one row per ZIP
houston = zhvi[zhvi["Metro"].str.startswith("Houston")]   # {n_zips} metro ZIP codes
houston[month_columns].median()                          # typical value, every month since 2000
# 2012: {money(med.loc["2012-01-31"])}   peak {med.idxmax():%b %Y}: {money(med.max())}
# our snapshot (May 2024): {money(snap)}   latest ({med.index[-1]:%b %Y}): {money(last)}""", language="python")
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts.index, y=ts[0.75], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
fig.add_trace(go.Scatter(x=ts.index, y=ts[0.25], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(57,135,229,0.15)",
                         name="middle 50% of ZIP codes", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=ts.index, y=med, mode="lines", line=dict(color=BLUE, width=2.5), name="median ZIP code",
                         hovertemplate="%{x|%b %Y}: %{y:$,.0f}<extra></extra>"))
for when, text in [(pd.Timestamp("2024-05-31"), "our data snapshot"), (med.idxmax(), "peak")]:
    fig.add_trace(go.Scatter(x=[when], y=[med.loc[when]], mode="markers", marker=dict(color=INK_2, size=9),
                             hoverinfo="skip", showlegend=False))
    fig.add_annotation(x=when, y=med.loc[when], text=text, showarrow=False, yshift=16,
                       xanchor="right" if text == "peak" else "left", font=dict(color=INK_2, size=13))
style(fig, 340, showlegend=True, legend=dict(orientation="h", y=1.1, x=0), hovermode="x unified")
fig.update_yaxes(tickformat="$.3~s", title="Typical home value")
st.plotly_chart(fig, width="stretch")
insight(f"Houston's typical home value went from {md_money(med.loc['2012-01-31'])} in 2012 to a peak of {md_money(med.max())} in "
        f"{med.idxmax():%B %Y}, **{med.max() / med.loc['2012-01-31'] - 1:+.0%} in ten years**, then went flat. Since our June 2024 snapshot "
        f"it has moved **{last / snap - 1:+.1%}**, so today's fair values would be slightly lower. The model is easy to retrain on fresh listings.")

# ---------------------------------------------------------------- Q7
st.markdown("#### …and ZIP code by ZIP code")
yv = zip_values_by_year(tuple(sorted(df.zipcode.unique())))
growth = (yv[2024] / yv[2012] - 1).dropna().sort_values()
city = df.groupby("zipcode").city.agg(lambda c: c.mode().iat[0])
fast, slow = growth.tail(3)[::-1], growth.head(3)
st.code(f"""by_year = zhvi.set_index("RegionName")[june_columns]   # one value per ZIP per year
by_year[2024] / by_year[2012] - 1                       # growth 2012 -> 2024
# fastest: {", ".join(f"{z} {g:+.0%}" for z, g in fast.items())}
# slowest: {", ".join(f"{z} {g:+.0%}" for z, g in slow.items())}
# {(growth >= 1).mean():.0%} of ZIP codes at least doubled""", language="python")

YEAR_EDGES = [0, 100_000, 150_000, 200_000, 300_000, 500_000, np.inf]  # fixed across years, so the map darkens as prices rise
YEAR_LABELS = ["under $100k", "$100k–150k", "$150k–200k", "$200k–300k", "$300k–500k", "over $500k"]
year = st.slider("Year", 2000, 2026, 2024, help="Zillow Home Value Index for each ZIP code, June of each year")
geo = zip_shapes()
vals = yv[year]
bins = pd.cut(vals, YEAR_EDGES, labels=False)
for f in geo["features"]:
    z = f["properties"]["zipcode"]
    v = vals.get(z, np.nan)
    chg = v / yv.at[z, 2012] - 1 if pd.notna(v) and pd.notna(yv.at[z, 2012]) else np.nan
    f["properties"].update(fill=[200, 198, 192, 120] if pd.isna(v) else hex_rgb(REDS[int(bins[z])], 215),
                           city=city.get(z, ""), val="no data" if pd.isna(v) else money(v),
                           chg="" if pd.isna(chg) else f" · {chg:+.0%} vs 2012")
c1, c2 = st.columns([1, 1.1], gap="large")
with c1:
    st.pydeck_chart(pdk.Deck(
        map_provider="carto", map_style="light",
        initial_view_state=pdk.ViewState(latitude=29.80, longitude=-95.45, zoom=7.9),
        layers=[pdk.Layer("GeoJsonLayer", geo, get_fill_color="properties.fill", get_line_color=[255, 255, 255],
                          line_width_min_pixels=0.8, stroked=True, filled=True, pickable=True, auto_highlight=True,
                          highlight_color=[11, 11, 11, 70])],
        tooltip={"html": f"<b>{{zipcode}} · {{city}}</b><br/>{year}: {{val}}{{chg}}"},
    ), height=420)
    swatch = '<span style="width:18px;height:12px;background:{};border-radius:3px;margin-right:5px;display:inline-block"></span>'
    html_block('<div style="font-size:.85rem;color:#52514e;margin:-.4rem 0 .6rem">' + "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:10px">{swatch.format(c)}{t.replace("$", "&#36;")}</span>'
        for c, t in zip(REDS, YEAR_LABELS)) + f'<span style="display:inline-flex;align-items:center">{swatch.format("#c8c6c0")}no data</span></div>')
    st.caption(f"Typical home value by ZIP code in June {year}. Drag the slider to watch Houston's prices climb.")
with c2:
    label = {z: f"{z} · {city.get(z, '')}" for z in yv.index}
    default = [fast.index[0], slow.index[0], yv[2024].idxmax()]
    pick = st.multiselect("Compare ZIP codes (up to 4)", list(yv.index), default=default, max_selections=4, format_func=label.get)
    fig = go.Figure()
    for z, color in zip(pick, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]):  # categorical slots 1-4, in fixed order
        ser = yv.loc[z].dropna()
        fig.add_trace(go.Scatter(x=ser.index, y=ser.values, mode="lines", name=label[z], line=dict(color=color, width=2.5),
                                 hovertemplate=label[z] + "<br>%{x}: %{y:$,.0f}<extra></extra>"))
        fig.add_annotation(x=ser.index[-1], y=np.log10(ser.values[-1]), text=z, showarrow=False, xanchor="left", xshift=6,
                           font=dict(color=color, size=12))
    fig.add_vline(x=year, line=dict(color=INK_2, width=1, dash="dot"))
    style(fig, 420, showlegend=True, legend=dict(orientation="h", y=-0.18, x=0), margin=dict(l=10, r=50, t=10, b=10),
          hovermode="x unified")
    fig.update_yaxes(type="log", tickvals=[5e4, 1e5, 2e5, 5e5, 1e6, 2e6], ticktext=["$50k", "$100k", "$200k", "$500k", "$1M", "$2M"],
                     title="Typical home value (log scale)")
    st.plotly_chart(fig, width="stretch")
fast_txt = ", ".join(f"{z} ({city.get(z, '')}) {g:+.0%}" for z, g in fast.items())
slow_txt = ", ".join(f"{z} {g:+.0%}" for z, g in slow.items())
insight(f"From 2012 to 2024, the **cheaper neighborhoods grew fastest**: {fast_txt}. "
        f"The pricey, condo-heavy core grew slowest ({slow_txt}). "
        f"Overall **{(growth >= 1).mean():.0%} of ZIP codes at least doubled**. Since the 2022 peak, "
        f"{((yv[2026] / yv[2022] - 1) < 0).mean():.0%} of ZIP codes have slipped.")

st.markdown("### Q7. Do more bedrooms mean a pricier home?")
bd = df[df.bedrooms.between(1, 6)]
by_bed = bd.groupby("bedrooms").agg(price=("price", "median"), ppsf=("ppsf", "median"), n=("price", "size"))
bed_code = "\n".join(f"# {b:.0f} bd  median {money(r.price):>9}   {money(r.ppsf)}/sqft   ({int(r.n):,} homes)" for b, r in by_bed.iterrows())
st.code(f"""df.groupby("bedrooms")[["price", "ppsf"]].median()
{bed_code}""", language="python")
fig = go.Figure()
for b in range(1, 7):
    fig.add_trace(go.Box(y=bd.price[bd.bedrooms == b], name=f"{b} bd", marker_color=BLUE, line=dict(width=1.5),
                         fillcolor="rgba(42,120,214,0.18)", boxpoints=False, hoverinfo="y"))
    fig.add_annotation(x=f"{b} bd", y=np.log10(bd.price[bd.bedrooms == b].quantile(0.95)), yshift=14, showarrow=False,
                       text=f"{money(by_bed.ppsf[b])}/sqft", font=dict(color=INK_2, size=12))
style(fig, 360, title=dict(text="Asking price by bedrooms (box = middle 50%, line = median, whiskers = typical range)",
                           font=dict(size=14)))
fig.update_yaxes(type="log", tickvals=[5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6],
                 ticktext=["$50k", "$100k", "$200k", "$500k", "$1M", "$2M", "$5M"],
                 title="Asking price (log scale)")
st.plotly_chart(fig, width="stretch")
insight(f"Each extra bedroom raises the typical price (from {md_money(by_bed.price.iloc[0])} for 1 bedroom to "
        f"{md_money(by_bed.price.iloc[-1])} for 6), but **price per sqft barely moves** ({md_money(by_bed.ppsf.min())}–"
        f"{md_money(by_bed.ppsf.max())}). Buyers pay for **space**; bedrooms matter because they come with square feet. "
        "The boxes also overlap a lot: bedroom count alone is a weak predictor.")

# ---------------------------------------------------------------- Q8
st.markdown("### Q8. Which features carry the same information?")
cmap = {"price": "log price", "livingArea": "living area", "bathrooms": "bathrooms", "bedrooms": "bedrooms",
        "lot_sqft": "lot size", "yearBuilt": "year built", "dist_downtown_mi": "miles to downtown",
        "zip_avg_agi": "ZIP income", "zip_zhvi": "ZIP home value"}
cm = df[list(cmap)].assign(price=np.log(df.price)).corr(method="spearman")
st.code(f"""cols = {list(cmap)[:5]}
cols += {list(cmap)[5:]}
df[cols].corr(method="spearman")   # heatmap below
# strongest pair: zip_avg_agi vs zip_zhvi = {cm.loc["zip_avg_agi", "zip_zhvi"]:.2f}""", language="python")
labels = [cmap[c] for c in cm.columns]
fig = go.Figure(go.Heatmap(z=cm.values, x=labels, y=labels, zmin=-1, zmax=1, zmid=0,
                           colorscale=[[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]],
                           text=cm.round(2).values, texttemplate="%{text:.2f}", textfont=dict(size=12),
                           hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>", xgap=2, ygap=2,
                           colorbar=dict(title="ρ", thickness=12, tickvals=[-1, 0, 1])))
style(fig, 470, margin=dict(l=10, r=10, t=10, b=10))
fig.update_yaxes(autorange="reversed", showgrid=False)
fig.update_xaxes(showgrid=False, tickangle=-35)
st.plotly_chart(fig, width="stretch")
insight(f"Size features move together (living area vs bathrooms {cm.loc['livingArea', 'bathrooms']:.2f}, vs bedrooms "
        f"{cm.loc['livingArea', 'bedrooms']:.2f}), and **ZIP income and ZIP home value are almost the same signal "
        f"({cm.loc['zip_avg_agi', 'zip_zhvi']:.2f})**. That's why adding ZIP income barely helped the model in Step 6. "
        "Overlapping features would confuse a linear model, but tree models handle them fine.")

# ---------------------------------------------------------------- Q9
st.markdown("### Q9. How long do homes sit on the market, and who cuts prices?")
dom = df.assign(kind=np.where(df.is_new_construction, "New construction", "Resale"), cut=df.priceReduction.notna())
cuts = dom.groupby("kind").agg(days=("daysOnZillow", "median"), cut=("cut", "mean"))
st.code(f"""df.groupby("is_new_construction")["daysOnZillow"].median()
# resale {cuts.days["Resale"]:.0f} days, new construction {cuts.days["New construction"]:.0f} days
df["priceReduction"].notna().groupby(df["is_new_construction"]).mean()
# resale {cuts.cut["Resale"]:.0%} had a price cut, new construction {cuts.cut["New construction"]:.0%}""", language="python")
c1, c2 = st.columns([1.3, 1], gap="large")
with c1:
    fig = go.Figure()
    for kind, color in [("Resale", BLUE), ("New construction", "#eb6834")]:
        d = np.sort(dom.daysOnZillow[dom.kind == kind].clip(upper=365).to_numpy())
        y = np.arange(1, len(d) + 1) / len(d)
        fig.add_trace(go.Scatter(x=d, y=y, mode="lines", name=kind, line=dict(color=color, width=2.5),
                                 hovertemplate=kind + ": %{y:.0%} listed ≤ %{x} days<extra></extra>"))
    fig.add_hline(y=0.5, line=dict(color=INK_2, width=1, dash="dot"))
    style(fig, 330, showlegend=True, legend=dict(orientation="h", y=1.12, x=0),
          title=dict(text="Share of listings on the market for at most N days", font=dict(size=14), y=0.98),
          margin=dict(l=10, r=20, t=70, b=10))
    fig.update_xaxes(title="Days on Zillow (capped at 365)")
    fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
    st.plotly_chart(fig, width="stretch")
with c2:
    flagged = load_scored().assign(cut=lambda d: d.priceReduction.notna()).groupby("flag").cut.mean()
    stat_tiles([(f"{cuts.cut['Resale']:.0%}", "of resale listings have cut their price", None),
                (f"{cuts.cut['New construction']:.0%}", "of new builds have cut their price", None),
                (f"{flagged['Underpriced']:.0%}", "of homes our model flags as underpriced already had a price cut",
                 f"vs {flagged['Overpriced']:.0%} of those flagged overpriced", True)])
insight(f"New builds sit longer (median {cuts.days['New construction']:.0f} vs {cuts.days['Resale']:.0f} days) but cut prices less "
        f"often. And a sneak peek at Step 7: **{flagged['Underpriced']:.0%} of the homes our model calls underpriced already had a "
        f"price cut**, versus {flagged['Overpriced']:.0%} of those it calls overpriced. Many bargains are homes whose sellers already "
        "lowered the price, and our model finds them without ever seeing the price history.")

takeaway("Price depends on **size** (not bedroom count), **location in a non-linear way**, and **new vs. resale**. Prices are "
         "skewed, so we model log price; features overlap, so we use trees. Values peaked in 2022 and have eased since, and "
         "affordability is the real story for Houston buyers.")
st.page_link("views/cleaning.py", label="Next: cleaning the data →")
