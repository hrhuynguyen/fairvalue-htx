import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (BLUE, FEATURES, cost_inputs, default_pool, driver_waterfall, hbar, load_scored, md_money, money,
                    monthly_cost, page_header, plot_range, plot_waterfall)

scored = load_scored()

page_header("APPLICATIONS 1 + 2 + 3 FOR ONE HOME", "Home report")
pool = st.session_state.get("deal_view") or default_pool(scored)
st.caption("Listings come from your latest search on *Under- & overpriced homes* (or, before any search, starter-home bargains under \\$400k).")
sel = st.session_state.get("selected_home")
pick = st.selectbox("Choose a listing", pool, index=pool.index(sel) if sel in pool else 0,
                    format_func=lambda i: f"{scored.at[i, 'address']} · {money(scored.at[i, 'price'])} ({scored.at[i, 'gap_pct'] * 100:+.0f}% vs fair)")
st.session_state["selected_home"] = pick
h = scored.loc[pick]

left, right = st.columns([1, 1.6])
with left:
    if isinstance(h.imgSrc, str):
        st.image(h.imgSrc, width="stretch")
    st.markdown(f"**{h.address}**  \n{h.bedrooms:.0f} bd · {h.bathrooms:g} ba · {h.livingArea:,.0f} sqft · "
                f"built {h.yearBuilt:.0f} · {h.homeType.replace('_', ' ').title()}  \n[View on Zillow ↗]({h.detailUrl})")
    verdict = {"Underpriced": "🔵 Listed **below** its fair-value range",
               "Overpriced": "🔴 Listed **above** its fair-value range",
               "Fair": "⚪ Listed **within** its fair-value range",
               "Verify first": "⚠️ Gap is **unusually large**. Likely a distressed sale, major repairs, "
                               "or a listing data error. Verify before trusting it."}[h.flag]
    st.markdown(f"### {verdict}")
    k = st.columns(2)
    k[0].metric("List price", money(h.price))
    k[1].metric("Fair value", money(h.fair_value), delta=f"{h.gap_pct * 100:+.1f}% list vs fair", delta_color="off")
    st.caption(f"80% fair range: {md_money(h.fair_lo)} – {md_money(h.fair_hi)}. "
               f"Zillow's Zestimate: {md_money(h.zestimate) if pd.notna(h.zestimate) else 'n/a'}.")
with right:
    st.plotly_chart(plot_range(h.price, h.fair_lo, h.fair_value, h.fair_hi), width="stretch")
    st.markdown("**Why the model values it at this price**")
    start, steps = driver_waterfall(h.shap_base, {f: h[f"shap__{f}"] for f in FEATURES}, h.fair_value)
    st.plotly_chart(plot_waterfall(start, steps, h.fair_value), width="stretch")

st.markdown("#### The real monthly cost (the sticker price is not the payment)")
parts = monthly_cost(h.price, **cost_inputs("report"))
total = sum(parts.values())
c1, c2 = st.columns([1.4, 1])
with c1:
    st.plotly_chart(hbar(list(parts), list(parts.values()), "$,.0f", color=BLUE, height=270,
                         hover="%{y}: %{x:$,.0f}/mo<extra></extra>").update_xaxes(tickformat="$,.0f"), width="stretch")
with c2:
    st.metric("All-in monthly cost", money(total),
              delta=f"{(total - parts['Principal & interest']) / total:.0%} is tax, insurance & fees", delta_color="off")
    st.metric("Income needed (28% rule)", money(total * 12 / 0.28) + "/yr")
    st.metric(f"Avg household income in {h.zipcode}", money(h.zip_avg_agi) + "/yr", help="IRS average adjusted gross income per return, 2021")
    if pd.notna(h.rentZestimate):
        cheaper = "renting" if h.rentZestimate < total else "owning"
        st.metric("Renting a similar home", money(h.rentZestimate) + "/mo",
                  delta=f"{cheaper} is {money(abs(total - h.rentZestimate))}/mo cheaper", delta_color="off")
if pd.notna(h.zip_value_change_since_listing):
    st.caption(f"Market since this snapshot: typical home values in {h.zipcode} changed "
               f"{h.zip_value_change_since_listing:+.1%} from June 2024 to Sept 2026 (Zillow Home Value Index). "
               f"In the year before the listing: {h.zip_zhvi_1y_growth:+.1%}.")
