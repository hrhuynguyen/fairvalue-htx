import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

import ai
from common import (DEFAULT_COSTS, DRIVER_GROUPS, FEATURES, FLAG_COLOR, HUBS, HUB_LABELS, driver_waterfall, hex_rgb,
                    load_scored, max_affordable_price, money, monthly_cost, page_header)

scored = load_scored().copy()
scored["monthly_cost"] = sum(monthly_cost(scored.price, **DEFAULT_COSTS).values())
CITIES = sorted(scored.city.unique())
ZIPS = sorted(scored.zipcode.unique())
EXAMPLES = [
    "3-bedroom house under $1,800 a month within 10 miles of the Medical Center, good deals only",
    "We're a family of four earning $95k. Show us bargains in Katy or Sugar Land",
    "Cheapest newer townhouse near downtown, built after 2015",
]

page_header("CREATIVE 1 · AI-POWERED SEARCH AND EXPLANATIONS", "Ask XMarket",
            "Describe the home you want in plain English. **Claude** (Anthropic's AI) turns your words into search filters, "
            "and our model supplies every price. The AI never makes up numbers.")

cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, width="stretch"):
        st.session_state.ask_query = ex
query = st.text_input("What are you looking for?", key="ask_query",
                      placeholder="e.g. 2-bed condo near UH under $250k")


def apply_filters(f):
    d = scored[scored.flag != "Verify first"]
    budget = f["max_monthly_cost"]
    if f["annual_income"]:
        budget = min(budget or np.inf, f["annual_income"] * 0.28 / 12)
    if budget:
        d = d[d.monthly_cost <= budget]
    if f["max_price"]:
        d = d[d.price <= f["max_price"]]
    if f["min_bedrooms"]:
        d = d[d.bedrooms >= f["min_bedrooms"]]
    if f["min_bathrooms"]:
        d = d[d.bathrooms >= f["min_bathrooms"]]
    if f["min_sqft"]:
        d = d[d.livingArea >= f["min_sqft"]]
    if f["home_types"]:
        d = d[d.homeType.isin(f["home_types"])]
    if f["near"] != "none":
        d = d[d[f"dist_{f['near']}_mi"] <= (f["max_miles"] or 10)]
    if f["zip_codes"] or f["cities"]:
        d = d[d.zipcode.isin(f["zip_codes"]) | d.city.str.lower().isin([c.lower() for c in f["cities"]])]
    if f["new_construction"] != "any":
        d = d[d.is_new_construction == (f["new_construction"] == "only")]
    if f["built_after"]:
        d = d[d.yearBuilt >= f["built_after"]]
    if f["deals_only"]:
        d = d[d.flag == "Underpriced"]
    order = {"best_deal": ("gap_pct", True), "lowest_price": ("price", True), "lowest_monthly_cost": ("monthly_cost", True),
             "largest": ("livingArea", False), "newest": ("yearBuilt", False)}[f["sort_by"]]
    return d.sort_values(order[0], ascending=order[1]), budget


if query:
    try:
        with st.spinner("Claude is reading your request…"):
            f = ai.parse_search(query, CITIES, ZIPS)
    except ai.AIUnavailable as e:
        st.error(f"The AI assistant is unavailable: {e} You can still search manually on **Under- & overpriced homes**.")
        st.stop()
    results, budget = apply_filters(f)
    st.success(f"**Claude understood:** {f['interpretation']}")
    chips = [f"budget ≤ {money(budget)}/mo" if budget else None,
             f"price ≤ {money(f['max_price'])}" if f["max_price"] else None,
             f"{f['min_bedrooms']}+ beds" if f["min_bedrooms"] else None,
             f"{f['min_bathrooms']:g}+ baths" if f["min_bathrooms"] else None,
             f"{f['min_sqft']:,.0f}+ sqft" if f["min_sqft"] else None,
             ", ".join(t.replace("_", " ").lower() for t in f["home_types"]) or None,
             f"≤ {f['max_miles'] or 10:g} mi from {HUB_LABELS[f['near']]}" if f["near"] != "none" else None,
             "in " + ", ".join(f["cities"] + f["zip_codes"]) if f["cities"] or f["zip_codes"] else None,
             {"only": "new construction only", "exclude": "no new construction"}.get(f["new_construction"]),
             f"built {f['built_after']}+" if f["built_after"] else None,
             "underpriced only" if f["deals_only"] else None,
             "sorted by " + f["sort_by"].replace("_", " ")]
    st.caption("Filters: " + " · ".join(c for c in chips if c).replace("$", "\\$"))

    if results.empty:
        st.warning("No homes match all of that. Try a bigger budget or a wider area.")
        st.stop()
    k = st.columns(3)
    k[0].metric("Homes found", f"{len(results):,}")
    k[1].metric("Median asking price", money(results.price.median()))
    k[2].metric("Listed below fair value", f"{(results.flag == 'Underpriced').mean():.0%}")

    st.session_state["deal_view"] = list(results.index[:500])  # Home report / Rent or buy pick from these
    top = results.head(300)
    mv = top.assign(color=top.flag.map(lambda x: hex_rgb(FLAG_COLOR[x])), price_s=top.price.map(money),
                    fair_s=top.fair_value.map(money), gap_s=(top.gap_pct * 100).map(lambda g: f"{g:+.0f}%"))
    center = HUBS[f["near"]] if f["near"] != "none" else (top.latitude.median(), top.longitude.median())
    st.pydeck_chart(pdk.Deck(
        map_provider="carto", map_style="light",
        initial_view_state=pdk.ViewState(latitude=center[0], longitude=center[1], zoom=9.5),
        layers=[pdk.Layer("ScatterplotLayer", mv, get_position=["longitude", "latitude"], get_fill_color="color",
                          get_radius=120, radius_min_pixels=4, radius_max_pixels=12, pickable=True,
                          stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1)],
        tooltip={"html": "<b>{address}</b><br/>{price_s} · fair {fair_s} ({gap_s})"},
    ), height=360)
    table = top.head(15)[["address", "price", "fair_value", "gap_pct", "monthly_cost", "bedrooms", "bathrooms", "livingArea",
                          "yearBuilt", "flag", "detailUrl"]].assign(gap_pct=lambda t: t.gap_pct * 100)
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "address": "Address", "price": st.column_config.NumberColumn("Asking", format="$%,.0f"),
        "fair_value": st.column_config.NumberColumn("Fair value", format="$%,.0f"),
        "gap_pct": st.column_config.NumberColumn("vs fair", format="%+.0f%%"),
        "monthly_cost": st.column_config.NumberColumn("All-in / mo", format="$%,.0f"),
        "bedrooms": st.column_config.NumberColumn("Beds", format="%d"), "bathrooms": "Baths",
        "livingArea": st.column_config.NumberColumn("Sqft", format="%,d"),
        "yearBuilt": st.column_config.NumberColumn("Built", format="%d"), "flag": "Verdict",
        "detailUrl": st.column_config.LinkColumn("Listing", display_text="Zillow ↗")})

    st.markdown("#### Explain a home in plain English")
    pick = st.selectbox("Pick one of the results", list(top.index[:15]),
                        format_func=lambda i: f"{scored.at[i, 'address']} · {money(scored.at[i, 'price'])}")
    st.session_state["selected_home"] = pick
    if st.button("✨ Explain this home", type="primary"):
        h = scored.loc[pick]
        _, steps = driver_waterfall(h.shap_base, {x: h[f"shap__{x}"] for x in FEATURES}, h.fair_value)
        parts = monthly_cost(h.price, **DEFAULT_COSTS)
        total = sum(parts.values())
        facts = {
            "address": h.address, "home": f"{h.bedrooms:.0f} bed, {h.bathrooms:g} bath, {h.livingArea:,.0f} sqft "
                                          f"{h.homeType.replace('_', ' ').lower()}, built {h.yearBuilt:.0f}",
            "asking_price": money(h.price), "fair_value": money(h.fair_value),
            "fair_range_80pct": f"{money(h.fair_lo)} to {money(h.fair_hi)}",
            "verdict": {"Underpriced": "below the fair range", "Overpriced": "above the fair range",
                        "Fair": "within the fair range"}[h.flag],
            "asking_vs_fair": f"{h.gap_pct:+.1%}",
            "price_drivers_vs_average_listing": {g: ("+" if v >= 0 else "-") + money(abs(v)) for g, v in steps[:4]},
            "all_in_monthly_cost": money(total),
            "monthly_breakdown": {k: money(v) for k, v in parts.items()},
            "assumptions": "6.5% 30-year mortgage, 10% down, 2.2% property tax, 1% insurance, $50 HOA, $800/yr flood insurance",
            "income_needed_28pct_rule": money(total * 12 / 0.28) + "/yr",
            "zip_average_household_income": money(h.zip_avg_agi) + "/yr",
            "rent_estimate_similar_home": money(h.rentZestimate) + "/mo" if pd.notna(h.rentZestimate) else "unknown",
            "listing_status": h.statusText,
        }
        try:
            with st.spinner("Claude is writing a summary…"):
                text = ai.explain_home(facts)
            st.info(text.replace("$", "\\$"))
            st.caption("Written by Claude from our model's numbers above. Not financial advice.")
        except ai.AIUnavailable as e:
            st.error(f"The AI assistant is unavailable: {e} See **Home report** for the full numbers.")

st.divider()
st.caption(f"Powered by Claude ({ai.MODEL}) via the Anthropic API. The AI only translates requests into filters and explains our "
           "model's output; all prices come from the XMarket model. Answers are cached so repeat questions are instant.")
