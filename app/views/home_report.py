import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (BLUE, FEATURES, INK_2, MUTED, app_guide, cost_inputs, default_pool, driver_waterfall, hbar, load_scored,
                    md_money, money, monthly_cost, page_header, plot_range, plot_waterfall, stat_tiles, style,
                    zip_value_history)

scored = load_scored()
page_header("APPLICATION 5 · ONE HOME, END TO END", "Home report",
            "For a buyer about to make an offer: is the price fair, what will it cost each month, and where is the market heading?")
app_guide(["**Choose a listing** (from your latest search, or a starter-home bargain by default).",
           "Check the **verdict** and **what drives** its fair value.", "Switch tabs for the **monthly cost** and the **ZIP's price trend**."],
          "The default is a starter home listed well below its fair value. Its report shows why the model values it higher, "
          "what Maya would pay each month, and how home values in that ZIP code have moved.")

pool = st.session_state.get("deal_view") or default_pool(scored)
sel = st.session_state.get("selected_home")
pick = st.selectbox("Listing", pool, index=pool.index(sel) if sel in pool else 0,
                    format_func=lambda i: f"{scored.at[i, 'address']} · {money(scored.at[i, 'price'])} ({scored.at[i, 'gap_pct'] * 100:+.0f}% vs fair)")
st.session_state["selected_home"] = pick
h = scored.loc[pick]
verdict = {"Underpriced": "Below fair range", "Overpriced": "Above fair range", "Fair": "Within fair range",
           "Verify first": "Unusual gap: verify"}[h.flag]
stat_tiles([(money(h.price), "asking price", None), (money(h.fair_value), "our fair value", f"{h.gap_pct:+.0%} asking vs fair", True),
            (verdict, "verdict", f"80% range {money(h.fair_lo)} – {money(h.fair_hi)}")])

tab1, tab2, tab3 = st.tabs(["Price check", "Monthly cost", "ZIP price trend"])
with tab1:
    c1, c2 = st.columns([1, 1.6], gap="large")
    with c1:
        if isinstance(h.imgSrc, str):
            st.image(h.imgSrc, width="stretch")
        st.markdown(f"{h.bedrooms:.0f} bd · {h.bathrooms:g} ba · {h.livingArea:,.0f} sqft · built {h.yearBuilt:.0f}  \n"
                    f"[View on Zillow ↗]({h.detailUrl})")
    with c2:
        st.plotly_chart(plot_range(h.price, h.fair_lo, h.fair_value, h.fair_hi), width="stretch")
        start, steps = driver_waterfall(h.shap_base, {f: h[f"shap__{f}"] for f in FEATURES}, h.fair_value)
        st.plotly_chart(plot_waterfall(start, steps, h.fair_value), width="stretch")
with tab2:
    parts = monthly_cost(h.price, **cost_inputs("report"))
    total = sum(parts.values())
    c1, c2 = st.columns([1.4, 1], gap="large")
    with c1:
        st.plotly_chart(hbar(list(parts), list(parts.values()), "$,.0f", color=BLUE, height=260,
                             hover="%{y}: %{x:$,.0f}/mo<extra></extra>").update_xaxes(tickformat="$,.0f"), width="stretch")
    with c2:
        stat_tiles([(money(total), "all-in per month", None, True), (money(total * 12 / 0.28), "income needed / yr", "28% rule")])
        if pd.notna(h.rentZestimate):
            st.caption(f"Renting a similar home: about {md_money(h.rentZestimate)}/month.")
with tab3:
    hist = zip_value_history()
    since = hist.loc["2012-01-01":]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=since.index, y=since.median(axis=1), mode="lines", name="Houston-area median",
                             line=dict(color=MUTED, width=2), hovertemplate="%{x|%b %Y}: %{y:$,.0f}<extra></extra>"))
    if h.zipcode in since.columns:
        fig.add_trace(go.Scatter(x=since.index, y=since[h.zipcode], mode="lines", name=f"ZIP {h.zipcode}",
                                 line=dict(color=BLUE, width=2.5), hovertemplate="%{x|%b %Y}: %{y:$,.0f}<extra></extra>"))
    fig.add_vline(x=pd.Timestamp("2024-06-01"), line=dict(color=INK_2, width=1, dash="dot"))
    style(fig, 320, showlegend=True, legend=dict(orientation="h", y=1.12, x=0), hovermode="x unified",
          margin=dict(l=10, r=20, t=40, b=10))
    fig.update_yaxes(tickformat="$,.0f", title="Typical home value (Zillow)")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Dotted line: our June 2024 listing snapshot. Since then, typical values in {h.zipcode} changed "
               f"{h.zip_value_change_since_listing:+.1%}.")
