import pandas as pd
import pydeck as pdk
import streamlit as st

import plotly.graph_objects as go

from common import BLUE, SEQ, hbar, hex_rgb, load_json, load_listings, stat_tiles, step_header, style, table, takeaway

facts = load_json("cleaning.json")
listings = load_listings()

step_header(1, 6, "The data",
            "We combined one listings dataset with two public datasets, so every home has its size, location, "
            "and the income and price trends of its ZIP code.")

st.markdown("#### Where the data comes from")
sources = pd.DataFrame([
    {"Dataset": "Houston Housing Market 2024 (raw Zillow scrape)", "Source": "Kaggle · datadetective08",
     "What it gives us": "Price, beds, baths, sqft, lot, home type, exact location, listing status, Zestimate",
     "Size": "25,948 listings (143 MB JSON)", "Time period": "Snapshot, early June 2024",
     "Coverage": "Greater Houston: 159 ZIPs, 62 cities"},
    {"Dataset": "Houston Housing Market 2024 (clean CSV)", "Source": "Kaggle · mohamedalmansop",
     "What it gives us": "Year built (not in the raw file)", "Size": "25,922 rows",
     "Time period": "Same snapshot", "Coverage": "Same listings, joined by Zillow ID"},
    {"Dataset": "IRS Statistics of Income by ZIP code", "Source": "irs.gov",
     "What it gives us": "Average household income and number of tax returns per ZIP",
     "Size": "~27,000 ZIPs (we use Houston's)", "Time period": "Tax year 2021", "Coverage": "Every US ZIP"},
    {"Dataset": "Zillow Home Value Index (ZHVI) & Observed Rent Index (ZORI)", "Source": "zillow.com/research",
     "What it gives us": "Typical home value and rent per ZIP, every month",
     "Size": "Monthly series per ZIP", "Time period": "2000 – Sept 2026", "Coverage": "Every US ZIP Zillow tracks"},
])
table(sources.set_index("Dataset"))

stat_tiles([
    (f"{facts['n_final']:,}", "homes after cleaning", "from 25,948 raw listings", True),
    (str(facts["n_zips"]), "ZIP codes", None),
    (str(facts["n_cities"]), "cities and towns", None),
    (f"{facts['new_construction_share']:.0%}", "brand-new construction", None),
])

c1, c2, c3 = st.columns(3, gap="large")
with c1:
    t = listings.homeType.value_counts()
    st.markdown("**Home types**")
    st.plotly_chart(hbar([x.replace("_", " ").title() for x in t.index], list(t.values), ",d", height=240), width="stretch")
with c2:
    t = listings.city.value_counts().head(6)
    st.markdown("**Biggest cities in the data**")
    st.plotly_chart(hbar(list(t.index), list(t.values), ",d", color=SEQ[3], height=240), width="stretch")
with c3:
    cols = {"Location (lat/lon)": "latitude", "ZIP income (IRS)": "zip_avg_agi", "Year built": "yearBuilt",
            "Living area": "livingArea", "Bedrooms": "bedrooms", "ZIP rent (Zillow)": "zip_zori", "Lot size": "lot_sqft"}
    comp = {k: listings[v].notna().mean() for k, v in cols.items()}
    st.markdown("**How complete each column is**")
    st.plotly_chart(hbar(list(comp), list(comp.values()), ".0%", color=SEQ[4], height=240).update_xaxes(range=[0, 1.15], tickformat=".0%"),
                    width="stretch")

st.markdown("#### Geographic coverage")
st.pydeck_chart(pdk.Deck(
    map_provider="carto", map_style="light",
    initial_view_state=pdk.ViewState(latitude=29.80, longitude=-95.45, zoom=8.3),
    layers=[pdk.Layer("ScatterplotLayer", listings, get_position=["longitude", "latitude"], get_fill_color=hex_rgb(BLUE, 120),
                      get_radius=80, radius_min_pixels=1.5, pickable=False)],
), height=380)
st.caption("Every dot is one home for sale. Coverage runs from Conroe in the north to Galveston Bay in the south, "
           "and from Waller and Fulshear in the west to Baytown in the east: mainly Harris, Fort Bend, Montgomery, Brazoria and Galveston counties.")

st.markdown("#### What the important columns mean")
dictionary = pd.DataFrame([
    ("price", "Asking (list) price in USD", "Target: what we predict"),
    ("bedrooms, bathrooms", "Number of bedrooms and bathrooms", "Feature"),
    ("livingArea", "Heated/cooled interior space in square feet", "Feature"),
    ("lotAreaValue + lotAreaUnit", "Land size, in sqft OR acres (we convert all to sqft)", "Feature"),
    ("yearBuilt", "Year the home was built", "Feature"),
    ("homeType", "Single family, townhouse, condo, multi-family, manufactured", "Feature"),
    ("latitude, longitude, zipcode", "Exact location and 5-digit ZIP", "Feature + used to join other data"),
    ("is_newHome / builderName", "New construction listing and the builder's name", "Feature"),
    ("is_bankOwned", "Foreclosure / bank-owned sale", "Feature"),
    ("zestimate, rentZestimate", "Zillow's own value and rent estimates", "Benchmark only, never a feature"),
    ("taxAssessedValue", "County tax appraisal", "Benchmark only, never a feature"),
    ("daysOnZillow, priceReduction", "How long it's been listed, any price cut", "Context only (seller behavior, not home value)"),
], columns=["Column", "Meaning", "How we use it"])
table(dictionary.set_index("Column"))

st.markdown("#### The two questions we asked of the data")
c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.markdown("**Does it have location or neighborhood info?**")
        st.markdown("✅ **Yes.** Every home has exact latitude/longitude and a ZIP code. That lets us measure "
                    "commute distance, find each home's nearest neighbors, and compare neighborhoods.")
with c2:
    with st.container(border=True):
        st.markdown("**Does it have income or affordability data?**")
        st.markdown("❌ **Not in the listings**, so we joined it by ZIP code: IRS average household income, plus Zillow's "
                    "typical home value and rent for that ZIP. (The Census API now requires a key; the IRS file is open.)")

takeaway("The listings tell us *what* the home is and *where* it is. The public data tells us *who can afford it* "
         "and *how its neighborhood's prices are moving*.")
st.page_link("views/eda.py", label="Next: exploring the data →", icon="🔍")
