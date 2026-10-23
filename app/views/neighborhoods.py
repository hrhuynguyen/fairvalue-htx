import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (BLUE, RED, app_guide, fmt_value, hbar, load_scored, max_affordable_price, page_header, style, table,
                    zip_value_history)

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order
scored = load_scored()
page_header("APPLICATION 7 · NEIGHBORHOOD COMPARISONS", "Neighborhood comparison",
            "For anyone choosing where to search: compare Houston ZIP codes on price, affordability and momentum.")
app_guide(["Pick **2–4 ZIP codes** to compare.", "Read the **side-by-side table**.",
           "Check the **price trend**, then rank every ZIP code at the bottom."],
          "Midtown/Third Ward (**77004**) vs the Heights (**77008**) vs Katy (**77494**): the Heights costs the most per sqft, "
          "Katy gives the most space for the money, and the trend chart shows how each market moved since 2012.")


@st.cache_data
def zip_table(df):
    g = df.groupby("zipcode").agg(
        city=("city", lambda s: s.mode().iat[0]), listings=("zpid", "size"), median_price=("price", "median"),
        median_ppsf=("ppsf", "median"), median_sqft=("livingArea", "median"), median_gap=("gap_pct", "median"),
        income=("zip_avg_agi", "first"), growth_5y=("zip_zhvi_5y_growth", "first"), since=("zip_value_change_since_listing", "first"),
        rent=("zip_zori", "first"), miles_downtown=("dist_downtown_mi", "median"))
    g = g[g.listings >= 30]
    g["price_to_income"] = g.median_price / g.income
    g["income_needed"] = [_income_needed(p) for p in g.median_price]
    return g


def _income_needed(price):
    lo, hi = 10_000, 3_000_000
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if max_affordable_price(mid) >= price else (mid, hi)
    return hi


g = zip_table(scored)
label = {z: f"{z} · {g.at[z, 'city']}" for z in g.index}
pick = st.multiselect("ZIP codes to compare", list(g.index), default=[z for z in ["77004", "77008", "77494"] if z in g.index],
                      max_selections=4, format_func=label.get)
if pick:
    rows = [("Median asking price", "median_price", "${:,.0f}"), ("Median $/sqft", "median_ppsf", "${:,.0f}"),
            ("Median size (sqft)", "median_sqft", "{:,.0f}"), ("Income needed for the median home", "income_needed", "${:,.0f}"),
            ("Price-to-income ratio", "price_to_income", "{:.1f}×"), ("Typical rent (Zillow)", "rent", "${:,.0f}/mo"),
            ("Value change, 5 yrs before 2024", "growth_5y", "{:+.0%}"), ("Value change since June 2024", "since", "{:+.1%}"),
            ("Median asking vs fair value", "median_gap", "{:+.1%}"), ("Miles to downtown", "miles_downtown", "{:.0f}")]
    table(pd.DataFrame({label[z]: [fmt.format(g.at[z, col]) if pd.notna(g.at[z, col]) else "–" for _, col, fmt in rows]
                        for z in pick}, index=[r[0] for r in rows]))

    hist = zip_value_history().loc["2012-01-01":]
    fig = go.Figure()
    for z, color in zip(pick, COLORS):
        if z in hist.columns:
            fig.add_trace(go.Scatter(x=hist.index, y=hist[z], mode="lines", name=label[z], line=dict(color=color, width=2.5),
                                     hovertemplate=label[z] + ": %{y:$,.0f}<extra></extra>"))
    style(fig, 340, showlegend=True, legend=dict(orientation="h", y=1.12, x=0), hovermode="x unified",
          title=dict(text="Typical home value since 2012 (Zillow)", font=dict(size=15), y=0.98), margin=dict(l=10, r=20, t=70, b=10))
    fig.update_yaxes(tickformat="$,.0f")
    st.plotly_chart(fig, width="stretch")

with st.expander(f"Rank all {len(g)} ZIP codes"):
    options = {"Price-to-income (lower = more affordable)": ("price_to_income", ".1f"),
               "Income needed to buy the median home": ("income_needed", "$,.0f"),
               "Median price per sqft": ("median_ppsf", "$,.0f"),
               "Median asking vs fair value": ("median_gap", "+.1%"),
               "Value change since June 2024": ("since", "+.1%")}
    choice = st.radio("Rank by", list(options), horizontal=True)
    col, f = options[choice]
    top = pd.concat([g.nsmallest(10, col), g.nlargest(10, col)]).drop_duplicates().sort_values(col)
    mid = g[col].median()
    colors = [BLUE if v < mid else RED for v in top[col]] if col in ("median_gap", "since") else [BLUE] * len(top)
    st.plotly_chart(hbar([label[z] for z in top.index], list(top[col]), f, colors=colors, height=26 * len(top) + 60,
                         hover="%{y}<br>%{x:" + f + "}<extra></extra>").update_xaxes(tickformat=f), width="stretch")
    st.caption(f"The 10 lowest and 10 highest ZIP codes (≥ 30 listings). Median: {fmt_value(mid, f).replace('$', chr(92) + '$')}.")
