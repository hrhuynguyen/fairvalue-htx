import pandas as pd
import pydeck as pdk
import streamlit as st

from common import FLAG_COLOR, HUBS, HUB_LABELS, cost_inputs, hex_rgb, load_scored, money, monthly_cost, page_header

scored = load_scored().copy()

page_header("APPLICATION 3 · PROPERTY VALUE ANALYSIS", "Under- and overpriced homes",
            "Every listing is compared with its **fair value** from a model that never saw that home's price. "
            "If the asking price falls **below the 80% fair range**, it's flagged as underpriced; **above it**, overpriced.")

c = st.columns([1, 1, 1, 1.4])
budget = c[0].number_input("Max all-in monthly cost ($)", 500, 30000, 1750, 50)
beds = c[1].selectbox("Min bedrooms", [1, 2, 3, 4, 5], index=1)
hub = c[2].selectbox("Near", list(HUB_LABELS), format_func=HUB_LABELS.get, index=1)
max_mi = c[3].slider(f"Max miles to {HUB_LABELS[hub]}", 2, 60, 15)
c = st.columns([2, 1])
types = c[0].multiselect("Home type", sorted(scored.homeType.unique()), default=["SINGLE_FAMILY", "TOWNHOUSE", "CONDO"])
show = c[1].selectbox("Show", ["Underpriced only", "Overpriced only", "All homes"])
include_unusual = st.checkbox("Include listings with an unusually large gap (often foreclosures, fixer-uppers or data errors)")
assume = cost_inputs("deals")

scored["monthly_cost"] = sum(monthly_cost(scored.price, **assume).values())
view = scored[(scored.monthly_cost <= budget) & (scored.bedrooms.fillna(0) >= beds)
              & scored.homeType.isin(types) & (scored[f"dist_{hub}_mi"] <= max_mi)]
if not include_unusual:
    view = view[view.flag != "Verify first"]
counts = view.flag.value_counts()
k = st.columns(4)
k[0].metric("Homes matching", f"{len(view):,}")
k[1].metric("🔵 Underpriced", f"{counts.get('Underpriced', 0):,}")
k[2].metric("⚪ Fairly priced", f"{counts.get('Fair', 0):,}")
k[3].metric("🔴 Overpriced", f"{counts.get('Overpriced', 0):,}")

if show == "Underpriced only":
    view = view[view.flag == "Underpriced"].sort_values("gap_pct")
elif show == "Overpriced only":
    view = view[view.flag == "Overpriced"].sort_values("gap_pct", ascending=False)
else:
    view = view.sort_values("gap_pct")
st.session_state["deal_view"] = list(view.index[:500])

if len(view):
    mv = view.assign(color=view.flag.map(lambda f: hex_rgb(FLAG_COLOR[f])), price_s=view.price.map(money),
                     fair_s=view.fair_value.map(money), gap_s=(view.gap_pct * 100).map(lambda g: f"{g:+.0f}%"),
                     cost_s=view.monthly_cost.map(money))
    hlat, hlon = HUBS[hub]
    st.pydeck_chart(pdk.Deck(
        map_provider="carto", map_style="light",
        initial_view_state=pdk.ViewState(latitude=hlat, longitude=hlon, zoom=9.5 if max_mi <= 20 else 8.5),
        layers=[
            pdk.Layer("ScatterplotLayer", mv, get_position=["longitude", "latitude"], get_fill_color="color",
                      get_radius=120, radius_min_pixels=4, radius_max_pixels=12, pickable=True,
                      stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1),
            pdk.Layer("ScatterplotLayer", pd.DataFrame({"lat": [hlat], "lon": [hlon]}), get_position=["lon", "lat"],
                      get_fill_color=[11, 11, 11], get_radius=300, radius_min_pixels=7),
        ],
        tooltip={"html": "<b>{address}</b><br/>List {price_s} · Fair {fair_s} ({gap_s})<br/>All-in {cost_s}/mo"},
    ), height=420)
    st.caption("● Blue: below the fair range · ● Gray: within · ● Red: above · ● Dark gray: verify first · ⬤ Black: "
               f"{HUB_LABELS[hub]}")
    table = view.head(200)[["address", "price", "fair_value", "gap_pct", "monthly_cost", "bedrooms", "bathrooms",
                            "livingArea", "yearBuilt", "statusText", "flag", "detailUrl"]].copy()
    table["gap_pct"] *= 100
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "address": "Address", "price": st.column_config.NumberColumn("List price", format="$%,.0f"),
        "fair_value": st.column_config.NumberColumn("Fair value", format="$%,.0f"),
        "gap_pct": st.column_config.NumberColumn("List vs fair", format="%+.0f%%"),
        "monthly_cost": st.column_config.NumberColumn("All-in / mo", format="$%,.0f"),
        "bedrooms": st.column_config.NumberColumn("Beds", format="%d"), "bathrooms": "Baths",
        "livingArea": st.column_config.NumberColumn("Sqft", format="%,d"),
        "yearBuilt": st.column_config.NumberColumn("Built", format="%d"), "statusText": "Status", "flag": "Verdict",
        "detailUrl": st.column_config.LinkColumn("Listing", display_text="Zillow ↗"),
    })
    st.caption("Why do budget searches show more blue than red? A budget filter keeps homes whose asking price is low *for what they "
               "are*, which is exactly what underpriced means. Across all listings, every price band has about 10% blue and 10% red.")
    st.info("A price well below fair value can be a real deal, or a sign of something the data can't see "
            "(condition, flood history, foundation). Treat a blue dot as a reason to look closer, not a verdict. "
            "Open **Home report** in the sidebar to see why the model values any of these homes the way it does.")
else:
    st.warning("No homes match. Try a higher budget, a larger radius, or fewer bedrooms.")
