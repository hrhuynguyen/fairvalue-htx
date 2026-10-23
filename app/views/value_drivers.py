import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (INK_2, MUTED, app_guide, load_json, money, page_header, price_homes, stat_tiles, style, zip_centers)

page_header("APPLICATION 6 · PROPERTY VALUE ANALYSIS", "What adds value",
            "For owners planning a renovation: which upgrade adds the most value to *your* home?")
app_guide(["Describe **your home**.", "See what each **upgrade** is worth for this home, in dollars.",
           "Compare with the **Houston-wide** average below."],
          "A 3-bed, 2-bath, 1,800 sqft house built in 2005 in **77004**. The chart shows, for this exact home, the change "
          "in fair value from each upgrade. Try another ZIP code: the same upgrade is worth different amounts in different places.")

zc = zip_centers()
c = st.columns(5)
z = c[0].selectbox("ZIP code", list(zc.index), index=list(zc.index).index("77004"), format_func=lambda z: f"{z} · {zc.at[z, 'city']}")
sqft = c[1].number_input("Living area (sqft)", 400, 10_000, 1800, 50)
beds = c[2].number_input("Bedrooms", 1, 8, 3)
baths = c[3].number_input("Bathrooms", 1.0, 8.0, 2.0, 0.5)
year = c[4].number_input("Year built", 1900, 2025, 2005)

home = dict(zipcode=z, homeType="SINGLE_FAMILY", sqft=sqft, beds=beds, baths=baths, year=year, lot=6000, new=False)
upgrades = {
    "Add a bathroom": {**home, "baths": baths + 1},
    "Add 250 sqft": {**home, "sqft": sqft + 250},
    "Add 500 sqft": {**home, "sqft": sqft + 500},
    "Add a bedroom (same sqft)": {**home, "beds": beds + 1},
    "Full renovation (like 10 yrs newer)": {**home, "year": min(year + 10, 2025)},
}
_, vals, _, _ = price_homes([home] + list(upgrades.values()))
base_value = vals[0]
delta = pd.Series(vals[1:] - base_value, index=list(upgrades)).sort_values(ascending=False)
stat_tiles([(money(base_value), "fair value today", None, True),
            (delta.index[0], "most valuable upgrade", f"{'+' if delta.iloc[0] >= 0 else '−'}{money(abs(delta.iloc[0]))}")])

fig = go.Figure(go.Bar(x=delta.values, y=delta.index, orientation="h",
                       marker_color=[INK_2 if v >= 0 else MUTED for v in delta.values],
                       text=[f"{'+' if v >= 0 else '−'}{money(abs(v))} ({v / base_value:+.1%})" for v in delta.values],
                       textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:$,.0f}<extra></extra>"))
style(fig, 300, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=40, t=10, b=10))
fig.add_vline(x=0, line=dict(color=INK_2, width=1))
span = max(abs(delta.min()), abs(delta.max()))
fig.update_xaxes(tickformat="$,.0f", title="Change in fair value for this home",
                 range=[min(delta.min(), 0) - 0.9 * span, max(delta.max(), 0) + 0.6 * span])
st.plotly_chart(fig, width="stretch")
st.caption("Each bar re-prices this home with one change and everything else held fixed. It shows how the market values a "
           "feature, not the cost of building it: compare the gain with your contractor's quote.")

with st.expander("Houston-wide average (3,000 single-family homes)"):
    vd = pd.DataFrame(load_json("diagnostics.json")["value_drivers"]).sort_values("median_dollars", ascending=False)
    st.dataframe(vd.rename(columns={"change": "Change", "median_dollars": "Median $ change", "median_pct": "Median % change"}),
                 hide_index=True, width="stretch", column_config={
                     "Median $ change": st.column_config.NumberColumn(format="$%,.0f"),
                     "Median % change": st.column_config.NumberColumn(format="%+.1f%%")})
