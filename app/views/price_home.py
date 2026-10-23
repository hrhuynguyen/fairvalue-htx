import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import (BLUE, FEATURES, INK_2, app_guide, driver_waterfall, load_model, load_scored, money, page_header,
                    plot_waterfall, price_homes, stat_tiles, style, zip_centers)

page_header("APPLICATION 1 · HOUSING PRICE PREDICTION", "Price any home",
            "For buyers and sellers: a fair price for any Houston home, listed or not, in seconds.")
app_guide(["Pick the **ZIP code** and **home type**.", "Enter **size, bedrooms, bathrooms** and **year built**.",
           "Read the **fair value**, its **80% range**, and **what drives the price**."],
          "A 3-bed, 2-bath, 1,800 sqft house built in 2005 in **77004** (Third Ward, next to UH). Its fair value is "
          "shown below. Change any input and everything updates instantly.")

zc = zip_centers()
c = st.columns(4)
z = c[0].selectbox("ZIP code", list(zc.index), index=list(zc.index).index("77004"), format_func=lambda z: f"{z} · {zc.at[z, 'city']}")
htype = c[1].selectbox("Home type", ["SINGLE_FAMILY", "TOWNHOUSE", "CONDO"], format_func=lambda t: t.replace("_", " ").title())
sqft = c[2].number_input("Living area (sqft)", 400, 10_000, 1800, 50)
year = c[3].number_input("Year built", 1900, 2025, 2005)
c = st.columns(4)
beds = c[0].number_input("Bedrooms", 1, 8, 3)
baths = c[1].number_input("Bathrooms", 1.0, 8.0, 2.0, 0.5)
with c[2]:
    with st.expander("More details"):
        lot = st.number_input("Lot size (sqft)", 0, 200_000, 6000, 250)
        new = st.checkbox("New construction", value=year >= 2023)

spec = dict(zipcode=z, homeType=htype, sqft=sqft, beds=beds, baths=baths, year=year, lot=lot, new=new)
X, mid, lo, hi = price_homes([spec])
stat_tiles([(money(mid[0]), "fair value", None, True),
            (f"${lo[0] / 1e3:,.0f}k – {hi[0] / 1e3:,.0f}k", "80% range", "8 in 10 similar homes list inside it"),
            (f"${mid[0] / sqft:,.0f}", "per sqft", f"ZIP {z}")])

c1, c2 = st.columns([1.1, 1], gap="large")
with c1:
    st.markdown("**Why this price**")
    base, contrib = load_model()["point"].contributions(X)
    start, steps = driver_waterfall(float(base[0]), dict(zip(FEATURES, contrib[0])), mid[0])
    st.plotly_chart(plot_waterfall(start, steps, mid[0]), width="stretch")
with c2:
    st.markdown("**How size moves the price**")
    sizes = np.arange(max(400, sqft - 1200), sqft + 1250, 100)
    _, curve_mid, curve_lo, curve_hi = price_homes([{**spec, "sqft": int(v)} for v in sizes])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sizes, y=curve_hi, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=sizes, y=curve_lo, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(42,120,214,0.15)", hoverinfo="skip", name="80% range"))
    fig.add_trace(go.Scatter(x=sizes, y=curve_mid, mode="lines", line=dict(color=BLUE, width=2.5), name="Fair value",
                             hovertemplate="%{x:,} sqft: %{y:$,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[sqft], y=[mid[0]], mode="markers", marker=dict(color=INK_2, size=10), name="This home",
                             hovertemplate="This home: %{y:$,.0f}<extra></extra>"))
    style(fig, 360, showlegend=True, legend=dict(orientation="h", y=1.1, x=0), margin=dict(l=10, r=20, t=40, b=10))
    fig.update_xaxes(title="Living area (sqft)")
    fig.update_yaxes(tickformat="$,.0f")
    st.plotly_chart(fig, width="stretch")

st.markdown("**Similar homes listed in this ZIP code**")
scored = load_scored()
comps = scored[(scored.zipcode == z) & (scored.homeType == htype) & scored.livingArea.between(sqft * 0.75, sqft * 1.25)]
comps = comps.assign(diff=(comps.livingArea - sqft).abs()).nsmallest(5, "diff")
if len(comps):
    st.dataframe(comps[["address", "price", "fair_value", "livingArea", "bedrooms", "bathrooms", "yearBuilt", "detailUrl"]],
                 hide_index=True, width="stretch", column_config={
                     "address": "Address", "price": st.column_config.NumberColumn("Asking", format="$%,.0f"),
                     "fair_value": st.column_config.NumberColumn("Our fair value", format="$%,.0f"),
                     "livingArea": st.column_config.NumberColumn("Sqft", format="%,d"),
                     "bedrooms": st.column_config.NumberColumn("Beds", format="%d"), "bathrooms": "Baths",
                     "yearBuilt": st.column_config.NumberColumn("Built", format="%d"),
                     "detailUrl": st.column_config.LinkColumn("Listing", display_text="Zillow ↗")})
else:
    st.caption("No listings of this type and size in this ZIP code.")
