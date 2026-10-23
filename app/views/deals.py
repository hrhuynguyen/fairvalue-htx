import pandas as pd
import pydeck as pdk
import streamlit as st

from common import (FLAG_COLOR, HUB_LABELS, HUBS, app_guide, cost_inputs, hex_rgb, load_scored, money, monthly_cost,
                    page_header, stat_tiles)

scored = load_scored().copy()
page_header("APPLICATION 4 · PROPERTY VALUE ANALYSIS", "Under- & overpriced homes",
            "For buyers: which listings in your budget are priced below their fair value?")
app_guide(["Set your **monthly budget** and **bedrooms**.", "Pick **where you work** and how far you'll commute.",
           "Browse the **map and table**: blue = below fair value, red = above."],
          "Maya: **$1,750/month**, 2+ bedrooms, within 15 miles of the **Texas Medical Center**. The table lists her "
          "best-value homes first; open one in **Home report** for the full analysis.")

c = st.columns(4)
budget = c[0].number_input("Monthly budget, all-in ($)", 500, 30_000, 1750, 50)
beds = c[1].selectbox("Bedrooms (min)", [1, 2, 3, 4, 5], index=1)
hub = c[2].selectbox("Work location", list(HUB_LABELS), format_func=HUB_LABELS.get, index=1)
max_mi = c[3].slider("Max miles", 2, 40, 15)
with st.expander("More filters"):
    f1, f2 = st.columns(2)
    types = f1.multiselect("Home type", sorted(scored.homeType.unique()), default=["SINGLE_FAMILY", "TOWNHOUSE", "CONDO"])
    show = f2.radio("Show", ["Underpriced only", "All homes", "Overpriced only"], horizontal=True)
    include_unusual = st.checkbox("Include extreme gaps (often foreclosures, fixer-uppers or listing errors)")
    assume = cost_inputs("deals")

scored["monthly_cost"] = sum(monthly_cost(scored.price, **assume).values())
view = scored[(scored.monthly_cost <= budget) & (scored.bedrooms.fillna(0) >= beds) & scored.homeType.isin(types)
              & (scored[f"dist_{hub}_mi"] <= max_mi)]
if not include_unusual:
    view = view[view.flag != "Verify first"]
counts = view.flag.value_counts()
stat_tiles([(f"{len(view):,}", "homes in budget", None),
            (f"{counts.get('Underpriced', 0):,}", "below fair value", None, True),
            (f"{counts.get('Fair', 0):,}", "fairly priced", None),
            (f"{counts.get('Overpriced', 0):,}", "above fair value", None)])
view = {"Underpriced only": view[view.flag == "Underpriced"].sort_values("gap_pct"),
        "Overpriced only": view[view.flag == "Overpriced"].sort_values("gap_pct", ascending=False),
        "All homes": view.sort_values("gap_pct")}[show]
st.session_state["deal_view"] = list(view.index[:500])
if len(view):
    st.session_state["selected_home"] = view.index[0]

if len(view):
    mv = view.assign(color=view.flag.map(lambda f: hex_rgb(FLAG_COLOR[f])), price_s=view.price.map(money),
                     fair_s=view.fair_value.map(money), gap_s=(view.gap_pct * 100).map(lambda g: f"{g:+.0f}%"))
    hlat, hlon = HUBS[hub]
    st.pydeck_chart(pdk.Deck(
        map_provider="carto", map_style="light",
        initial_view_state=pdk.ViewState(latitude=hlat, longitude=hlon, zoom=9.5 if max_mi <= 20 else 8.5),
        layers=[pdk.Layer("ScatterplotLayer", mv, get_position=["longitude", "latitude"], get_fill_color="color",
                          get_radius=120, radius_min_pixels=4, radius_max_pixels=12, pickable=True,
                          stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1),
                pdk.Layer("ScatterplotLayer", pd.DataFrame({"lat": [hlat], "lon": [hlon]}), get_position=["lon", "lat"],
                          get_fill_color=[11, 11, 11], get_radius=300, radius_min_pixels=7)],
        tooltip={"html": "<b>{address}</b><br/>Asking {price_s} · fair {fair_s} ({gap_s})"}), height=400)
    table = view.head(10)[["address", "price", "fair_value", "gap_pct", "monthly_cost", "bedrooms", "livingArea", "flag",
                           "detailUrl"]].assign(gap_pct=lambda d: d.gap_pct * 100)
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "address": "Address", "price": st.column_config.NumberColumn("Asking", format="$%,.0f"),
        "fair_value": st.column_config.NumberColumn("Fair value", format="$%,.0f"),
        "gap_pct": st.column_config.NumberColumn("vs fair", format="%+.0f%%"),
        "monthly_cost": st.column_config.NumberColumn("All-in / mo", format="$%,.0f"),
        "bedrooms": st.column_config.NumberColumn("Beds", format="%d"),
        "livingArea": st.column_config.NumberColumn("Sqft", format="%,d"), "flag": "Verdict",
        "detailUrl": st.column_config.LinkColumn("Listing", display_text="Zillow ↗")})
    st.caption("Top 10 shown. A big discount can also mean repairs or flood history the data can't see: inspect before you offer.")
else:
    st.warning("No homes match. Try a higher budget, a longer commute, or fewer bedrooms.")
