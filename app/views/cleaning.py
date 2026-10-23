import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import BLUE, INK_2, MUTED, ROOT, insight, load_json, load_listings, step_header, style, table, takeaway

facts = load_json("cleaning.json")
iss = facts["issues"]

step_header(3, 6, "Cleaning the data",
            "Garbage in, garbage out: we found four hidden problems in the 'clean' dataset, rebuilt it from the raw "
            "Zillow data, and removed rows that aren't real homes for sale.")

st.markdown("### Problems we found (and why they matter)")
cards = [
    ("🏗️", f"{iss['raw_builder_plans']:,}", "builder floor plans",
     "Listings like 'Custom Design 15218 Plan' show a *starting-from* price and the builder's sales-office location. "
     "They're not real homes."),
    ("🌳", f"{iss['raw_lots_land']:,}", "vacant land lots",
     "Land has 0 beds and 0 sqft. Mixing it in teaches the model that '0 sqft' is a normal house."),
    ("📏", f"{iss['raw_lot_in_acres']:,}", "lot sizes in acres",
     "The clean CSV dropped the unit column, so '0.3' (acres) and '6,769' (sqft) sat in the same column."),
    ("📅", f"{iss['clean_csv_year_2001_5']:,}", "fake build years",
     "Missing years were filled with 2001.5, the median. A made-up value looks real to a model."),
]
cols = st.columns(4)
for col, (icon, n, what, why) in zip(cols, cards):
    with col:
        with st.container(border=True):
            st.markdown(f"### {icon} {n}")
            st.markdown(f"**{what}**")
            st.caption(why)
PROBLEM = "#ec835a"  # status 'serious': marks the bad data, never used for series


@st.cache_data
def load_clean_csv():
    return pd.read_csv(ROOT / "data/raw/properties_clean_v2.csv", usecols=["yearBuilt", "lotAreaValue", "homeType"])


clean = load_clean_csv()
after = load_listings()
st.markdown("### Seeing the problems")
c1, c2 = st.columns(2, gap="large")
with c1:
    yr = clean.yearBuilt.value_counts().sort_index()
    yr = yr[yr.index >= 1950]
    fig = go.Figure(go.Bar(x=yr.index, y=yr.values, marker_color=[PROBLEM if y == 2001.5 else MUTED for y in yr.index],
                           width=[2.5 if y == 2001.5 else 0.8 for y in yr.index],
                           hovertemplate="built %{x}: %{y:,} rows<extra></extra>"))
    fig.add_annotation(x=2001.5, y=int(yr.get(2001.5, 0)), text="3,057 rows set to '2001.5'<br>(a filled-in value, not a real year)",
                       showarrow=True, arrowhead=0, ax=-120, ay=-10, font=dict(color=INK_2, size=13))
    style(fig, 300, title=dict(text="Before: year built in the 'clean' CSV", font=dict(size=15)))
    fig.update_yaxes(title="Listings")
    st.plotly_chart(fig, width="stretch")
with c2:
    edges = np.logspace(-1, 6, 70)
    before, _ = np.histogram(clean.lotAreaValue[clean.homeType != "LOT"], bins=edges)
    after_c, _ = np.histogram(after.lot_sqft.dropna(), bins=edges)
    mids = np.sqrt(edges[:-1] * edges[1:])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mids, y=before, mode="lines", line=dict(color=PROBLEM, width=2), name="Before (mixed units)",
                             hovertemplate="value %{x:,.1f}: %{y:,}<extra>before</extra>"))
    fig.add_trace(go.Scatter(x=mids, y=after_c, mode="lines", line=dict(color=BLUE, width=2), name="After (all in sqft)",
                             hovertemplate="%{x:,.0f} sqft: %{y:,}<extra>after</extra>"))
    fig.add_annotation(x=np.log10(0.25), y=max(before) * 0.55, text="acres hiding<br>in a sqft column", showarrow=False,
                       font=dict(color=PROBLEM, size=13))
    style(fig, 300, showlegend=True, legend=dict(orientation="h", y=-0.35, x=0),
          title=dict(text="Lot size: before vs. after fixing units", font=dict(size=15)), margin=dict(l=10, r=20, t=40, b=10))
    fig.update_xaxes(type="log", title="Lot size value (log scale)", tickvals=[0.1, 10, 1e3, 1e5],
                     ticktext=["0.1", "10", "1k", "100k"], tickangle=0)
    fig.update_yaxes(title="Homes")
    st.plotly_chart(fig, width="stretch")
insight("A model would have learned that thousands of homes were built in the year 2001.5, and that a 0.3-acre lot is "
        "'smaller' than a 600-sqft patio. Charts like these are how we caught both.")

st.markdown("**Fix:** instead of trusting the pre-cleaned CSV, we rebuilt the dataset from the **raw Zillow JSON**, kept the lot "
            "unit and converted acres → sqft (× 43,560), and borrowed only `yearBuilt` from the CSV (turning 2001.5 back into *missing*).")

st.markdown("### Step by step: from 25,948 listings to 19,430 homes")
funnel = facts["funnel"]
why = {
    "raw listings": "Everything Zillow returned for Greater Houston.",
    "drop builder floor plans": "'From $X' floor-plan ads, not real homes.",
    "drop vacant land": "Land is a different market from homes; it needs its own model.",
    "drop missing price/coords": "Can't learn or map a home without a price or location.",
    "price in [$50k, $15M]": "Below $50k is usually a rental or data typo; above $15M is too rare to learn from.",
    "living area in [300, 20k] sqft": "Smaller than a studio or larger than a mansion = data-entry error.",
    "price/sqft in [$25, $2000]": "Physically implausible price per sqft = wrong price or wrong size.",
}
labels = [f for f, _ in funnel]
values = [n for _, n in funnel]
fig = go.Figure(go.Bar(x=values, y=labels, orientation="h",
                       marker_color=[MUTED] + [BLUE] * (len(values) - 1),
                       text=[f"{v:,}" + (f"  (−{values[i - 1] - v:,})" if i else "") for i, v in enumerate(values)],
                       textposition="outside", cliponaxis=False,
                       customdata=[why[f] for f, _ in funnel],
                       hovertemplate="%{y}: %{x:,} rows left<br>%{customdata}<extra></extra>"))
style(fig, 330, yaxis=dict(autorange="reversed"))
fig.update_layout(margin=dict(l=10, r=120, t=10, b=10))
st.plotly_chart(fig, width="stretch")
table(pd.DataFrame([{"Step": f, "Rows left": f"{n:,}", "Why": why[f]} for f, n in funnel]).set_index("Step"))

st.markdown("### Other preprocessing decisions")
decisions = pd.DataFrame([
    ("Missing values (beds, sqft, lot, year)",
     "Left as *missing*. No fake zeros or medians.",
     "Our model (LightGBM) learns the best way to handle a missing value. Filling with 0 would claim a home has 0 sqft."),
    ("Price target", "Predict log(price), then convert back.",
     "Prices are heavily skewed (Step 2). Logs turn '$20k off' into '% off', fair to cheap and expensive homes alike."),
    ("Leaky columns", "Removed Zestimate, rent Zestimate, tax-assessed value from the inputs.",
     "They are other people's estimates of the price. We tested it: with the Zestimate as an input, error 'drops' to 1.2% and R² to 0.995: fake, it just copies Zillow."),
    ("Seller-behavior columns", "Days on Zillow and price cuts are not inputs.",
     "We want the value of the *home*, not how the seller is behaving. We show them as context instead."),
    ("Categories", "Home type and ZIP passed as categories.",
     "Tree models can split on categories directly, no need to invent 160 dummy columns."),
    ("Implausible values", "Lot < 200 sqft or > 50 acres, > 15 beds/baths → missing.",
     "Typos in a single field shouldn't delete the whole home, so we blank only the bad field."),
], columns=["Decision", "What we did", "Why"])
table(decisions.set_index("Decision"))

takeaway("We kept **19,430 real homes**, fixed units, and never invented data. "
         "Every removed row has a written reason, so anyone can audit it.")
st.page_link("views/features.py", label="Next: feature engineering →", icon="🛠️")
