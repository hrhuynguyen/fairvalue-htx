import numpy as np
import streamlit as st

from common import (FEATURES, HUBS, driver_waterfall, haversine_mi, load_listings, load_model, md_money, money,
                    page_header, plot_waterfall)
from features import prepare

model = load_model()
listings = load_listings()

page_header("APPLICATION 1 · HOUSING PRICE PREDICTION", "Price any home",
            "Describe a home, even one that isn't for sale, and get its fair price, an honest range, and the reasons behind it.")

zips = listings.groupby("zipcode").agg(lat=("latitude", "median"), lon=("longitude", "median"),
                                       city=("city", lambda s: s.mode().iat[0]), n=("zpid", "size"))
zips = zips[zips.n >= 10]
c = st.columns(4)
z = c[0].selectbox("ZIP code", zips.index, index=list(zips.index).index("77004") if "77004" in zips.index else 0,
                   format_func=lambda z: f"{z} · {zips.at[z, 'city']}")
htype = c[1].selectbox("Home type", ["SINGLE_FAMILY", "TOWNHOUSE", "CONDO", "MULTI_FAMILY", "MANUFACTURED"],
                       format_func=lambda t: t.replace("_", " ").title())
sqft = c[2].number_input("Living area (sqft)", 300, 20000, 1800, 50)
year = c[3].number_input("Year built", 1900, 2025, 2005)
c = st.columns(4)
bd = c[0].number_input("Bedrooms", 0, 10, 3)
ba = c[1].number_input("Bathrooms", 0.0, 10.0, 2.0, 0.5)
lot = c[2].number_input("Lot size (sqft)", 0, 500000, 6000, 250)
newb = c[3].checkbox("New construction", value=year >= 2023)

row = listings[listings.zipcode == z].iloc[[0]].copy()
row = row.assign(latitude=zips.at[z, "lat"], longitude=zips.at[z, "lon"], bedrooms=bd, bathrooms=ba,
                 livingArea=sqft, lot_sqft=lot or np.nan, yearBuilt=year, homeType=htype,
                 is_new_construction=newb, has_builder=newb, is_premier_builder=False, is_bank_owned=False, is_auction=False)
for name, (lat, lon) in HUBS.items():
    row[f"dist_{name}_mi"] = haversine_mi(row.latitude, row.longitude, lat, lon)
row["bed_bath_ratio"] = bd / ba if ba else np.nan
row["sqft_per_bed"] = sqft / bd if bd else np.nan
row = row.join(model["comps"].query(row, exclude_self=False))
X = prepare(row, model["categories"])[FEATURES]
log_mid = model["point"].predict(X)[0]
band = np.searchsorted(model["band_edges"], log_mid)  # interval calibration is per price band
mid = float(np.exp(log_mid))
lo = min(float(np.exp(model["lo"].predict(X)[0] - model["band_adj"][band, 0])), mid)
hi = max(float(np.exp(model["hi"].predict(X)[0] + model["band_adj"][band, 1])), mid)

k = st.columns(3)
k[0].metric("Estimated fair value", money(mid))
k[1].metric("80% range", f"{md_money(lo)} – {md_money(hi)}")
k[2].metric("Per square foot", money(mid / sqft))
st.markdown("**Why this price?** Start from the average listing, then each factor pushes the price up or down:")
base, sv = model["point"].contributions(X)  # exact SHAP for the weighted ensemble
start, steps = driver_waterfall(float(base[0]), dict(zip(FEATURES, sv[0])), mid)
st.plotly_chart(plot_waterfall(start, steps, mid), width="stretch")
st.caption(f"Location uses the center of ZIP {z}. 'Nearby comparable homes' = the 10 closest listings. "
           "Try changing one input at a time to see what it's worth.")
