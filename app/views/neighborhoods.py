import pandas as pd
import streamlit as st

from common import BLUE, RED, fmt_value, hbar, load_scored, max_affordable_price, page_header

scored = load_scored()

page_header("APPLICATION 4 · NEIGHBORHOOD COMPARISONS", "Compare Houston neighborhoods")


@st.cache_data
def zip_table(df):
    g = df.groupby("zipcode").agg(
        city=("city", lambda s: s.mode().iat[0]), listings=("zpid", "size"), median_price=("price", "median"),
        median_ppsf=("ppsf", "median"), median_sqft=("livingArea", "median"), median_gap=("gap_pct", "median"),
        share_under=("flag", lambda f: (f == "Underpriced").mean()), income=("zip_avg_agi", "first"),
        growth_5y=("zip_zhvi_5y_growth", "first"), since=("zip_value_change_since_listing", "first"),
        rent=("zip_zori", "first"), new_share=("is_new_construction", "mean"), miles_downtown=("dist_downtown_mi", "median"))
    g = g[g.listings >= 30]
    g["price_to_income"] = g.median_price / g.income
    g["income_needed"] = [_income_needed(p) for p in g.median_price]
    g.index = [f"{z} {c}" for z, c in zip(g.index, g.city)]
    return g


def _income_needed(price):
    lo, hi = 10_000, 3_000_000
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if max_affordable_price(mid) >= price else (mid, hi)
    return hi


g = zip_table(scored)
st.markdown(f"{len(g)} ZIP codes with at least 30 listings.")

st.markdown("### Side by side")
default = [z for z in g.index if z.startswith(("77004", "77008", "77494"))][:3]
pick = st.multiselect("Choose up to 4 ZIP codes", list(g.index), default=default, max_selections=4)
if pick:
    side = g.loc[pick, ["median_price", "median_ppsf", "median_sqft", "income", "income_needed", "price_to_income",
                        "growth_5y", "since", "rent", "new_share", "miles_downtown", "median_gap", "listings"]].T
    labels = {"median_price": "Median list price", "median_ppsf": "Median $/sqft", "median_sqft": "Median size (sqft)",
              "income": "Avg household income (IRS)", "income_needed": "Income needed for median home",
              "price_to_income": "Price-to-income ratio", "growth_5y": "Home values, 5 yrs before listing",
              "since": "Home values since June 2024", "rent": "Typical rent (Zillow)", "new_share": "New construction share",
              "miles_downtown": "Miles to downtown", "median_gap": "Median list vs. fair value", "listings": "Listings"}
    fmt = {"median_price": "${:,.0f}", "median_ppsf": "${:,.0f}", "median_sqft": "{:,.0f}", "income": "${:,.0f}",
           "income_needed": "${:,.0f}", "price_to_income": "{:.1f}×", "growth_5y": "{:+.0%}", "since": "{:+.1%}",
           "rent": "${:,.0f}/mo", "new_share": "{:.0%}", "miles_downtown": "{:.0f}", "median_gap": "{:+.1%}", "listings": "{:,.0f}"}
    shown = side.apply(lambda r: r.map(lambda v: fmt[r.name].format(v) if pd.notna(v) else "–"), axis=1)
    shown.index = [labels[i] for i in shown.index]
    st.table(shown)

st.markdown("### Rank every ZIP code")
options = {
    "Price-to-income (lower = more affordable)": ("price_to_income", ".1f"),
    "Income needed to buy the median home": ("income_needed", "$,.0f"),
    "Median price per sqft": ("median_ppsf", "$,.0f"),
    "Median list price vs. fair value (are sellers asking too much?)": ("median_gap", "+.1%"),
    "Home value change since June 2024": ("since", "+.1%"),
}
choice = st.radio("Compare by", list(options), horizontal=False)
col, f = options[choice]
top = pd.concat([g.nsmallest(10, col), g.nlargest(10, col)]).drop_duplicates().sort_values(col)
mid = g[col].median()
colors = [BLUE if v < mid else RED for v in top[col]] if col in ("median_gap", "since") else [BLUE] * len(top)
fig = hbar(list(top.index), list(top[col]), f, colors=colors, height=26 * len(top) + 60,
           hover="%{y}<br>%{x:" + f + "}<extra></extra>")
fig.update_xaxes(tickformat=f)
st.plotly_chart(fig, width="stretch")
st.caption(f"The 10 lowest and 10 highest of {len(g)} ZIP codes. Median across all: {fmt_value(mid, f).replace('$', chr(92) + '$')}.")
with st.expander("Full table (all ZIP codes)"):
    st.dataframe(g.sort_values(col), width="stretch", column_config={
        "median_price": st.column_config.NumberColumn("Median price", format="$%,.0f"),
        "median_ppsf": st.column_config.NumberColumn("$/sqft", format="$%.0f"),
        "income": st.column_config.NumberColumn("Avg income", format="$%,.0f"),
        "income_needed": st.column_config.NumberColumn("Income needed", format="$%,.0f"),
        "price_to_income": st.column_config.NumberColumn("Price / income", format="%.1f"),
        "median_gap": st.column_config.NumberColumn("List vs fair", format="percent"),
        "share_under": st.column_config.NumberColumn("Share underpriced", format="percent"),
        "growth_5y": st.column_config.NumberColumn("5y growth", format="percent"),
        "since": st.column_config.NumberColumn("Since 2024", format="percent"),
        "new_share": st.column_config.NumberColumn("New builds", format="percent"),
        "rent": st.column_config.NumberColumn("Rent", format="$%,.0f"),
    })
