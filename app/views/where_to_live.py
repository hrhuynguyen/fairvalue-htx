import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from common import (app_guide, HUB_LABELS, HUBS, SEQ, cost_inputs, haversine_mi, hex_rgb, insight, load_scored, money, monthly_cost,
                    page_header, stat_tiles, style)

# Categorical slots 1-4 of the reference palette, in fixed order (one per scoring criterion).
CRITERIA_COLORS = {"Affordability": "#2a78d6", "Short commute": "#eb6834", "Neighborhood strength": "#1baf7a",
                   "Bargains available": "#eda100"}
scored = load_scored()

page_header("APPLICATION 3 · RECOMMENDATION", "Where should I live?",
            "Tell us your income and where you work. We score every Houston ZIP code on **affordability**, **commute** "
            "and **neighborhood strength**, then recommend the best fits for you.")
app_guide(["Enter your **income** and **where you work**.", "Choose **buy or rent**, bedrooms and your longest commute.",
           "Get your **top 3 ZIP codes**, a map, and the best-value homes in #1."],
          "Maya earns **$75,000** and works at the **Texas Medical Center**. Her top match is shown below with how many "
          "homes fit her budget and how close it is to work. Open *Adjust priorities* to weight commute or price more.")

zips = scored.groupby("zipcode").agg(city=("city", lambda s: s.mode().iat[0]), lat=("latitude", "median"),
                                     lon=("longitude", "median"))
c = st.columns([1, 1.2, 0.9, 0.9])
income = c[0].number_input("Household income ($/yr)", 20_000, 1_000_000, 75_000, 5_000)
work_opts = list(HUBS) + ["zip"]
work = c[1].selectbox("Where do you work?", work_opts, index=work_opts.index("tmc"),
                      format_func=lambda k: HUB_LABELS.get(k, "Another ZIP code…"))
if work == "zip":
    wz = c[1].selectbox("Work ZIP code", list(zips.index), format_func=lambda z: f"{z} · {zips.at[z, 'city']}")
    (wlat, wlon), work_label = (zips.at[wz, "lat"], zips.at[wz, "lon"]), f"ZIP {wz}"
else:
    (wlat, wlon), work_label = HUBS[work], HUB_LABELS[work]
mode = c[2].radio("I want to", ["Buy", "Rent"], horizontal=True)
beds = c[3].selectbox("Bedrooms (min)", [1, 2, 3, 4], index=1)
max_mi = st.slider("Longest commute you'd accept (straight-line miles)", 3, 40, 15)

with st.expander("Adjust priorities (0 = ignore, 5 = very important)"):
    w = st.columns(4)
    pri = {"Affordability": w[0].slider("Affordability", 0, 5, 3),
           "Short commute": w[1].slider("Short commute", 0, 5, 3),
           "Neighborhood strength": w[2].slider("Neighborhood strength", 0, 5, 2,
                                                help="Home values grew in the 5 years before the listing and held up since 2024 (Zillow)"),
           "Bargains available": w[3].slider("Bargains available", 0, 5, 1 if mode == "Buy" else 0, disabled=mode == "Rent",
                                             help="Share of the ZIP's listings priced below our fair value")}
if mode == "Rent":
    pri["Bargains available"] = 0
assume = cost_inputs("where") if mode == "Buy" else None

# ---------------------------------------------------------------- score every ZIP code
homes = scored[(scored.flag != "Verify first") & (scored.bedrooms.fillna(0) >= beds)
               & scored.homeType.isin(["SINGLE_FAMILY", "TOWNHOUSE", "CONDO"])].copy()
buy_budget, rent_budget = income * 0.28 / 12, income * 0.30 / 12
if mode == "Buy":
    homes["monthly"] = sum(monthly_cost(homes.price, **assume).values())
    homes["fits"] = homes.monthly <= buy_budget
g = homes.groupby("zipcode").agg(listings=("zpid", "size"), rent=("zip_zori", "first"),
                                 growth=("zip_zhvi_5y_growth", "first"), since=("zip_value_change_since_listing", "first"),
                                 under=("flag", lambda f: (f == "Underpriced").mean()),
                                 **({"fits": ("fits", "sum")} if mode == "Buy" else {}))
g = g[g.listings >= 10].join(zips)
g["miles"] = haversine_mi(g.lat, g.lon, wlat, wlon)
ok = g.miles <= max_mi
if mode == "Buy":
    g["fit_share"] = g.fits / g.listings
    ok &= g.fits >= 5
else:
    g["burden"] = g.rent * 12 / income
    ok &= g.burden <= 0.35
cand = g[ok].copy()
if len(cand) < 3:
    st.warning(f"Only {len(cand)} ZIP codes within {max_mi} miles of {work_label} fit this budget. "
               "Try a longer commute, fewer bedrooms, or renting instead of buying.")
    st.stop()


def pct(s, higher_is_better=True):
    """Percentile rank among the candidate ZIP codes, 0-100."""
    return s.rank(pct=True, ascending=higher_is_better) * 100


# Buyers: both the share of homes that fit and how many there are to choose from.
cand["s_Affordability"] = ((pct(cand.fit_share) + pct(cand.fits)) / 2 if mode == "Buy"
                           else pct(cand.burden, higher_is_better=False))
cand["s_Short commute"] = pct(cand.miles, higher_is_better=False)
cand["s_Neighborhood strength"] = (pct(cand.growth.fillna(cand.growth.median()))
                                   + pct(cand.since.fillna(cand.since.median()))) / 2
cand["s_Bargains available"] = pct(cand.under)
total_w = sum(pri.values()) or 1
for k, v in pri.items():
    cand[f"c_{k}"] = cand[f"s_{k}"] * v / total_w  # each criterion's share of the final score
cand["score"] = sum(cand[f"c_{k}"] for k in pri)
cand = cand.sort_values("score", ascending=False)
top = cand.head(3)

# ---------------------------------------------------------------- recommendations
st.markdown(f"### Best ZIP codes for you, working at {work_label}")


def detail(r):
    afford = (f"{int(r.fits)} of {int(r.listings)} homes fit your budget" if mode == "Buy"
              else f"typical rent {money(r.rent)}/mo ({r.burden:.0%} of income)")
    growth = f" · values {r.growth:+.0%} in 5 yrs" if pd.notna(r.growth) else ""
    return f"{r.miles:.1f} mi to work · {afford}{growth}"


stat_tiles([(f"{z}", f"#{i + 1} · {r.city} · score {r.score:.0f}/100", detail(r), i == 0)
            for i, (z, r) in enumerate(top.iterrows())])

c1, c2 = st.columns([1.15, 1], gap="large")
with c1:
    m = cand.reset_index().assign(
        color=lambda d: [hex_rgb(SEQ[min(5, int(s // 17))], 220) for s in d.score],
        score_s=lambda d: d.score.round(0).astype(int).astype(str),
        miles_s=lambda d: d.miles.round(1).astype(str))
    labels = top.reset_index().assign(rank=[f"#{i + 1}" for i in range(len(top))])
    st.pydeck_chart(pdk.Deck(
        map_provider="carto", map_style="light",
        initial_view_state=pdk.ViewState(latitude=wlat, longitude=wlon, zoom=8.9 if max_mi <= 15 else 8.2),
        layers=[
            pdk.Layer("ScatterplotLayer", m, get_position=["lon", "lat"], get_fill_color="color", get_radius=900,
                      radius_min_pixels=7, pickable=True, stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1),
            pdk.Layer("ScatterplotLayer", labels, get_position=["lon", "lat"], get_fill_color=[0, 0, 0, 0], get_radius=1300,
                      radius_min_pixels=11, stroked=True, get_line_color=[11, 11, 11], line_width_min_pixels=2.5),
            pdk.Layer("TextLayer", labels, get_position=["lon", "lat"], get_text="rank", get_size=15, get_color=[11, 11, 11],
                      get_pixel_offset=[0, -24], font_weight=700),
            pdk.Layer("ScatterplotLayer", pd.DataFrame({"lat": [wlat], "lon": [wlon]}), get_position=["lon", "lat"],
                      get_fill_color=[11, 11, 11], get_radius=400, radius_min_pixels=8),
        ],
        tooltip={"html": "<b>{zipcode} {city}</b><br/>score {score_s}/100 · {miles_s} mi to work"},
    ), height=500)
    st.caption(f"Each dot is a ZIP code that fits your budget and commute; darker = better score. Ringed = top 3. "
               f"⬤ Black = {work_label}.")
with c2:
    t10 = cand.head(10).iloc[::-1]
    fig = go.Figure()
    for k, color in CRITERIA_COLORS.items():
        if pri[k]:
            fig.add_trace(go.Bar(y=[f"{z} {r.city}" for z, r in t10.iterrows()], x=t10[f"c_{k}"], name=k, orientation="h",
                                 marker=dict(color=color, line=dict(color="#fbfaf7", width=2)),
                                 hovertemplate=k + ": %{x:.0f} points<extra></extra>"))
    fig.add_trace(go.Scatter(y=[f"{z} {r.city}" for z, r in t10.iterrows()], x=t10.score + 4, mode="text",
                             text=[f"{s:.0f}" for s in t10.score], textposition="middle right", showlegend=False,
                             hoverinfo="skip", textfont=dict(color="#1a1a19", size=14)))
    style(fig, 500, barmode="stack", showlegend=True,
          legend=dict(orientation="h", y=-0.2, x=0, traceorder="normal", font=dict(size=13)),
          title=dict(text="Top 10: what makes up each score", font=dict(size=15)), margin=dict(l=10, r=30, t=40, b=10))
    fig.update_xaxes(range=[0, 108], title="Score (0-100)")
    st.plotly_chart(fig, width="stretch")

best_zip, best = top.index[0], top.iloc[0]
insight(f"For a **{money(income)}** household working at **{work_label}**, the best balance is **{best_zip} ({best.city})**: "
        f"{detail(best)}. Change the sliders to see how the ranking shifts when you care more about commute or price.")

if mode == "Buy":
    st.markdown(f"### Homes to look at in {best_zip}")
    picks = homes[(homes.zipcode == best_zip) & homes.fits].sort_values("gap_pct")
    st.session_state["deal_view"] = list(picks.index[:500])  # Home report and Rent or buy pick from these
    if len(picks):
        st.session_state["selected_home"] = picks.index[0]
    tbl = picks.head(10)[["address", "price", "fair_value", "gap_pct", "monthly", "bedrooms", "bathrooms", "livingArea",
                          "flag", "detailUrl"]].assign(gap_pct=lambda d: d.gap_pct * 100)
    st.dataframe(tbl, hide_index=True, width="stretch", column_config={
        "address": "Address", "price": st.column_config.NumberColumn("Asking", format="$%,.0f"),
        "fair_value": st.column_config.NumberColumn("Fair value", format="$%,.0f"),
        "gap_pct": st.column_config.NumberColumn("vs fair", format="%+.0f%%"),
        "monthly": st.column_config.NumberColumn("All-in / mo", format="$%,.0f"),
        "bedrooms": st.column_config.NumberColumn("Beds", format="%d"), "bathrooms": "Baths",
        "livingArea": st.column_config.NumberColumn("Sqft", format="%,d"), "flag": "Verdict",
        "detailUrl": st.column_config.LinkColumn("Listing", display_text="Zillow ↗")})
    st.caption("These homes are now preselected on **Home report** and **Rent or buy?** in the sidebar.")

with st.expander(f"All {len(cand)} ZIP codes that fit, ranked"):
    cols = ["city", "score", "miles"] + (["fits", "listings"] if mode == "Buy" else ["rent", "burden"]) + ["growth", "since", "under"]
    st.dataframe(cand[cols], width="stretch", column_config={
        "score": st.column_config.ProgressColumn("Score", format="%.0f", min_value=0, max_value=100),
        "miles": st.column_config.NumberColumn("Miles to work", format="%.1f"),
        "fits": st.column_config.NumberColumn("Homes that fit"),
        "listings": st.column_config.NumberColumn("Listings"),
        "rent": st.column_config.NumberColumn("Typical rent", format="$%,.0f"),
        "burden": st.column_config.NumberColumn("Rent / income", format="percent"),
        "growth": st.column_config.NumberColumn("5-yr value growth", format="percent"),
        "since": st.column_config.NumberColumn("Since June 2024", format="percent"),
        "under": st.column_config.NumberColumn("Share underpriced", format="percent")})
st.caption("How it works: each ZIP gets a 0-100 percentile on every criterion (among ZIP codes that fit your budget and "
           "commute), then a weighted average using your sliders. Distances are straight-line miles, not drive times. "
           "'Neighborhood strength' is home-value growth (Zillow), not school or crime data, so check those separately.")
