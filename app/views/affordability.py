import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from common import (BLUE, DEFAULT_COSTS, GRAY, INK_2, app_guide, cost_inputs, hex_rgb, load_scored, max_affordable_price,
                    money, monthly_cost, page_header, stat_tiles, style)

scored = load_scored()
page_header("APPLICATION 2 · AFFORDABLE HOUSING", "Affordability by income",
            "For households and first-time buyers: how much home can you afford, and where are those homes?")
maya_price = max_affordable_price(75_000)
app_guide(["Enter your **household income**.", "Pick the **bedrooms** you need.",
           "See your **max price**, how much of the market fits, and **where** those homes are."],
          f"Maya earns **$75,000**. At 28% of income she can spend **{money(75_000 * 0.28 / 12)}/month** all-in "
          f"(mortgage, tax, insurance), which buys a home up to **{money(maya_price)}**: only "
          f"**{(scored.price <= maya_price).mean():.0%} of Houston listings**.")

c = st.columns([1, 1, 1.4])
income = c[0].number_input("Household income ($/yr)", 20_000, 1_000_000, 75_000, 5_000)
beds = c[1].selectbox("Bedrooms (min)", [1, 2, 3, 4], index=1)
with c[2]:
    with st.expander("More options"):
        share = st.slider("Max share of income for housing", 0.20, 0.40, 0.28, 0.01)
        assume = cost_inputs("afford")

max_price = max_affordable_price(income, share, **assume)
fits = scored[(scored.price <= max_price) & (scored.bedrooms.fillna(0) >= beds) & (scored.flag != "Verify first")]
stat_tiles([(money(max_price), "max home price", None, True),
            (money(income * share / 12), "monthly budget, all-in", None),
            (f"{len(fits):,}", "homes that fit", f"{len(fits) / len(scored):.0%} of listings"),
            (f"{(fits.flag == 'Underpriced').sum():,}", "of them below fair value", None)])

c1, c2 = st.columns([1.1, 1], gap="large")
with c1:
    incomes = np.arange(30_000, 300_001, 5_000)
    curve = [((scored.price <= max_affordable_price(i, share, **assume)) & (scored.bedrooms.fillna(0) >= beds)).mean()
             for i in incomes]
    you = ((scored.price <= max_price) & (scored.bedrooms.fillna(0) >= beds)).mean()
    fig = go.Figure(go.Scatter(x=incomes, y=curve, mode="lines", line=dict(color=BLUE, width=2.5),
                               hovertemplate="Income %{x:$,.0f}: %{y:.0%} of listings<extra></extra>"))
    fig.add_trace(go.Scatter(x=[income], y=[you], mode="markers+text", marker=dict(color=INK_2, size=10),
                             text=[f"  you: {you:.0%}"], textposition="middle right", hoverinfo="skip"))
    style(fig, 320, title=dict(text="Share of listings you can afford, by income", font=dict(size=15)))
    fig.update_xaxes(tickformat="$,.0s", title="Household income")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig, width="stretch")
with c2:
    parts = monthly_cost(max_price, **assume)
    fig = go.Figure(go.Bar(x=list(parts.values()), y=list(parts), orientation="h", marker_color=BLUE,
                           text=[money(v) for v in parts.values()], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:$,.0f}/mo<extra></extra>"))
    style(fig, 320, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=60, t=40, b=10),
          title=dict(text=f"Where your {money(sum(parts.values()))}/month goes", font=dict(size=15)))
    fig.update_xaxes(tickformat="$,.0f")
    st.plotly_chart(fig, width="stretch")

st.markdown("**Where the homes you can afford are**")
if len(fits):
    m = fits.assign(color=[hex_rgb(BLUE if f == "Underpriced" else GRAY, 200) for f in fits.flag], price_s=fits.price.map(money))
    st.pydeck_chart(pdk.Deck(
        map_provider="carto", map_style="light", initial_view_state=pdk.ViewState(latitude=29.78, longitude=-95.42, zoom=8.6),
        layers=[pdk.Layer("ScatterplotLayer", m, get_position=["longitude", "latitude"], get_fill_color="color", get_radius=90,
                          radius_min_pixels=2.5, radius_max_pixels=9, pickable=True)],
        tooltip={"html": "<b>{address}</b><br/>{price_s}"}), height=400)
    top = fits.groupby("zipcode").agg(city=("city", lambda c: c.mode().iat[0]), homes=("zpid", "size"),
                                      median=("price", "median")).nlargest(5, "homes")
    st.caption("● Blue: affordable and listed below our fair value · ● Gray: affordable. Most options: "
               + ", ".join(f"{z} {r.city} ({r.homes})" for z, r in top.iterrows()) + ".")
else:
    st.warning("No listings fit this budget. Try a higher share of income or a bigger down payment under More options.")
