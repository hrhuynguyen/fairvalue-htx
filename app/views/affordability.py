import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from common import (BLUE, GRAY, HUBS, HUB_LABELS, INK_2, cost_inputs, hex_rgb, load_scored, max_affordable_price,
                    md_money, money, monthly_cost, page_header, style)

scored = load_scored()

page_header("APPLICATION 2 · AFFORDABLE HOUSING", "What can I afford, and where?",
            "Enter a household income. We find the highest price whose **all-in** monthly cost (mortgage + tax + "
            "insurance + PMI + HOA + flood) stays within a share of income, then show where those homes are.")

c = st.columns([1, 1, 1.2])
income = c[0].number_input("Household income ($/yr)", 20_000, 1_000_000, 75_000, 5_000)
share = c[1].slider("Max share of income for housing", 0.20, 0.40, 0.28, 0.01, format="%.2f",
                    help="Lenders' rule of thumb is 28% of gross income")
min_beds = c[2].selectbox("Minimum bedrooms", [1, 2, 3, 4], index=1)
assume = cost_inputs("afford")

max_price = max_affordable_price(income, share, **assume)
ok = scored[(scored.price <= max_price) & (scored.bedrooms.fillna(0) >= min_beds)]
k = st.columns(4)
k[0].metric("Max affordable price", money(max_price))
k[1].metric("Monthly budget", money(income * share / 12))
k[2].metric("Homes that fit", f"{len(ok):,}", help=f"{len(ok) / len(scored):.1%} of all listings")
k[3].metric("…priced at or below fair value", f"{(ok.gap_pct <= 0).sum():,}")

c1, c2 = st.columns([1.1, 1])
with c1:
    incomes = np.arange(30_000, 300_001, 5_000)
    curve = [((scored.price <= max_affordable_price(i, share, **assume)) & (scored.bedrooms.fillna(0) >= min_beds)).mean()
             for i in incomes]
    fig = go.Figure(go.Scatter(x=incomes, y=curve, mode="lines", line=dict(color=BLUE, width=2),
                               hovertemplate="Income %{x:$,.0f}<br>%{y:.0%} of listings affordable<extra></extra>"))
    you = len(ok) / len(scored)
    fig.add_trace(go.Scatter(x=[income], y=[you], mode="markers+text", marker=dict(color=INK_2, size=10, line=dict(color="white", width=2)),
                             text=[f"  You: {you:.0%}"], textposition="middle right", hoverinfo="skip"))
    style(fig, 320, title=dict(text="Share of Houston listings you could afford, by income", font=dict(size=14)))
    fig.update_xaxes(tickformat="$,.0s", title="Household income per year")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig, width="stretch")
with c2:
    zz = scored.groupby("zipcode").agg(city=("city", lambda s: s.mode().iat[0]), n=("zpid", "size"),
                                       fits=("price", lambda p: (p <= max_price).sum()), median=("price", "median"))
    zz = zz[zz.n >= 20]
    zz["share_fit"] = zz.fits / zz.n
    top = zz.sort_values("fits", ascending=False).head(10)
    st.markdown("**ZIP codes with the most homes you can afford**")
    st.dataframe(top.assign(place=[f"{z} {top.at[z, 'city']}" for z in top.index]).set_index("place").rename_axis("ZIP code")[["fits", "share_fit", "median"]],
                 width="stretch", column_config={
                     "fits": st.column_config.NumberColumn("Homes that fit"),
                     "share_fit": st.column_config.ProgressColumn("Share of ZIP", format="percent", min_value=0, max_value=1),
                     "median": st.column_config.NumberColumn("ZIP median price", format="$%,.0f")})

if len(ok):
    m = ok.assign(color=[hex_rgb(BLUE if g <= 0 else GRAY, 200) for g in ok.gap_pct],
                  price_s=ok.price.map(money), cost_s=[money(sum(monthly_cost(p, **assume).values())) for p in ok.price])
    st.pydeck_chart(pdk.Deck(
        map_provider="carto", map_style="light",
        initial_view_state=pdk.ViewState(latitude=29.78, longitude=-95.42, zoom=8.6),
        layers=[pdk.Layer("ScatterplotLayer", m, get_position=["longitude", "latitude"], get_fill_color="color", get_radius=90,
                          radius_min_pixels=2.5, radius_max_pixels=9, pickable=True),
                pdk.Layer("ScatterplotLayer", pd.DataFrame([{"lat": HUBS[h][0], "lon": HUBS[h][1], "address": HUB_LABELS[h],
                                                              "price_s": "job hub", "cost_s": "–"} for h in HUBS]),
                          get_position=["lon", "lat"], get_fill_color=[11, 11, 11], get_radius=250, radius_min_pixels=5, pickable=True)],
        tooltip={"html": "<b>{address}</b><br/>{price_s} · all-in {cost_s}/mo"},
    ), height=420)
    st.caption("● Blue: affordable AND priced at or below our fair value · ● Gray: affordable, priced above fair value · "
               "⬤ Black: job hubs (Downtown, Medical Center, UH, Galleria, Energy Corridor)")
else:
    st.warning(f"No listings fit a {md_money(income * share / 12)}/month budget. Try a higher share, lower rate, or bigger down payment.")
