import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from common import (BLUE, INK_2, SEQ, hbar, hex_rgb, insight, load_listings, max_affordable_price, md_money, money,
                    stat_tiles, step_header, style, takeaway)

df = load_listings().copy()
df["ppsf"] = df.price / df.livingArea

step_header(2, 6, "Exploring the data",
            "Before any modeling, we asked the data simple questions. The answers shaped every later decision.")

# ---------------------------------------------------------------- Q1
st.markdown("### Q1. What does a typical Houston listing cost?")
stat_tiles([
    (money(df.price.median()), "median asking price", None, True),
    (money(df.price.mean()), "average asking price", "pulled up by a few mansions"),
    (f"${df.price.quantile(.25) / 1e3:.0f}k–{df.price.quantile(.75) / 1e3:.0f}k", "middle 50% of homes", None),
    (f"{df.livingArea.median():,.0f}", "median size (sqft)", None),
])

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

z = df.groupby("zipcode").agg(lat=("latitude", "median"), lon=("longitude", "median"), ppsf=("ppsf", "median"),
                              n=("price", "size"), city=("city", lambda s: s.mode().iat[0]))
z = z[z.n >= 15].reset_index()
cuts = [0, 125, 150, 175, 225, 300, np.inf]
z["bucket"] = pd.cut(z.ppsf, cuts, labels=False)
z["color"] = z.bucket.map(lambda b: hex_rgb(SEQ[int(b)], 220))
z["label"] = z.ppsf.map(lambda v: f"${v:.0f}/sqft")
st.pydeck_chart(pdk.Deck(
    map_provider="carto", map_style="light",
    initial_view_state=pdk.ViewState(latitude=29.78, longitude=-95.42, zoom=8.6),
    layers=[pdk.Layer("ScatterplotLayer", z, get_position=["lon", "lat"], get_fill_color="color", get_radius=1100,
                      radius_min_pixels=6, pickable=True, stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1)],
    tooltip={"html": "<b>{zipcode} {city}</b><br/>{label} · {n} homes"},
), height=400)
st.caption("Median price per sqft by ZIP code, light → dark: under \\$125 · \\$125–150 · \\$150–175 · \\$175–225 · \\$225–300 · over \\$300.")
insight(f"Close to downtown you pay the most **per square foot** (about \\${g.ppsf.iloc[0]:.0f}), and it's cheapest 10–15 miles out "
        f"(about \\${g.ppsf.iloc[2]:.0f}). Farther suburbs climb again because their homes are **bigger and newer** "
        f"(median {g.sqft.iloc[4]:,.0f} sqft vs {g.sqft.iloc[2]:,.0f}). This U-shape is exactly why we chose a tree-based model, "
        "which can learn curves that a straight-line model cannot.")

# ---------------------------------------------------------------- Q4
st.markdown("### Q4. New construction vs. resale, and home types")
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

takeaway("Price depends on **size**, **location in a non-linear way**, and **new vs. resale**. Prices are skewed, so "
         "we model log price. And affordability is the real story for Houston buyers.")
st.page_link("views/cleaning.py", label="Next: cleaning the data →", icon="🧹")
