import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (BLUE, INK_2, MUTED, hbar, insight, load_json, load_listings, load_scored, step_header, style, table,
                    takeaway)
from features import FEATURES, HUB_DIST, LISTING_FLAGS, LOCATION, SPATIAL_LAG, ZIP_MARKET, CompsIndex

scored = load_scored()
metrics = load_json("metrics.json")
imp = pd.Series(metrics["shap_importance"])
rank = imp.rank(ascending=False).astype(int)

step_header(4, 7, "Feature engineering")

stages = metrics["stages"]
n_engineered = 2 + len(HUB_DIST) + len(SPATIAL_LAG) + len(ZIP_MARKET)  # 2 ratios + commute + comps + ZIP context
st.markdown("### Summary")
st.markdown(f"We started from the raw columns that describe the house itself (size, rooms, lot, year built, home type, exact "
            f"location and listing flags: {len(FEATURES) - n_engineered} inputs) and added **{n_engineered} engineered features** that "
            "describe it the way a buyer thinks: **ratios** (bedrooms per bathroom, sqft per bedroom), **commute** (straight-line "
            f"miles to {len(HUB_DIST)} job hubs), **nearby comps** (the median price and price per sqft of the 10 closest *other* "
            "listings, built without ever seeing the home's own price) and **neighborhood context** (ZIP income from the IRS; "
            f"typical home value, growth and rent from Zillow). That gives the model **{len(FEATURES)} inputs**. We then measured "
            "how much each input matters (below) and how much each idea cuts the error (Step 6): with one LightGBM model, going "
            f"from structure alone to the full feature set takes the typical error from **{stages[1]['MdAPE_%']:.1f}% to "
            f"{stages[6]['MdAPE_%']:.1f}%**.")

st.code("""# ratios
df["bed_bath_ratio"] = df["bedrooms"] / df["bathrooms"]
df["sqft_per_bed"] = df["livingArea"] / df["bedrooms"]

# commute: straight-line miles to 5 job hubs (Haversine formula)
for hub, (lat, lon) in HUBS.items():   # downtown, tmc, galleria, energy_corridor, uh
    df[f"dist_{hub}_mi"] = haversine_mi(df["latitude"], df["longitude"], lat, lon)

# nearby comps: the 10 closest OTHER listings (training homes only during cross-validation)
tree = BallTree(np.radians(train[["latitude", "longitude"]]), metric="haversine")
dist, idx = tree.query(np.radians(df[["latitude", "longitude"]]), k=11)   # 10 neighbors + self
idx = idx[:, 1:]                                                          # drop self
df["knn_log_ppsf"] = np.median(np.log(train_price / train_sqft)[idx], axis=1)
df["knn_log_price"] = np.median(np.log(train_price)[idx], axis=1)

# neighborhood context, joined by ZIP code
df = df.merge(irs_by_zip, on="zipcode").merge(zillow_by_zip, on="zipcode")""", language="python")

st.markdown("### Which inputs matter most?")
NICE = {"livingArea": "Living area", "knn_log_price": "Neighbors' median price", "knn_log_ppsf": "Neighbors' $/sqft",
        "lot_sqft": "Lot size", "yearBuilt": "Year built", "homeType": "Home type", "bathrooms": "Bathrooms",
        "sqft_per_bed": "Sqft per bedroom", "knn_dist_mi": "Distance to neighbors", "latitude": "Latitude",
        "longitude": "Longitude", "zipcode": "ZIP code", "zip_avg_agi": "ZIP income", "bed_bath_ratio": "Beds ÷ baths",
        "is_new_construction": "New construction", "bedrooms": "Bedrooms", "zip_zhvi": "ZIP typical value",
        "dist_tmc_mi": "Miles to Medical Center", "dist_downtown_mi": "Miles to downtown", "dist_galleria_mi": "Miles to Galleria",
        "dist_uh_mi": "Miles to UH", "dist_energy_corridor_mi": "Miles to Energy Corridor", "zip_zori": "ZIP rent",
        "zip_zhvi_5y_growth": "ZIP 5-yr growth", "zip_zhvi_1y_growth": "ZIP 1-yr growth", "zip_tax_returns": "ZIP households",
        "has_builder": "Builder listing", "is_bank_owned": "Foreclosure", "is_premier_builder": "Premier builder", "is_auction": "Auction"}
top = (imp / imp.sum()).sort_values(ascending=False).head(15)
engineered = {"knn_log_price", "knn_log_ppsf", "knn_dist_mi", "sqft_per_bed", "bed_bath_ratio", "zip_avg_agi", "zip_zhvi",
              "zip_zori", "zip_zhvi_5y_growth", "zip_zhvi_1y_growth", "zip_tax_returns"} | {f for f in imp.index if f.startswith("dist_")}
share = imp / imp.sum()
imp_code = "\n".join(f"# {f:<16} {v:.0%}" for f, v in share.sort_values(ascending=False).head(5).items())
st.code(f"""base, contrib = ensemble.contributions(X)   # SHAP value per home, per input
share = np.abs(contrib).mean(axis=0) / np.abs(contrib).mean(axis=0).sum()
share.sort_values(ascending=False).head(5)
{imp_code}""", language="python")
c1, c2 = st.columns([1.5, 1], gap="large")
with c1:
    st.plotly_chart(hbar([NICE.get(f, f) for f in top.index], list(top.values), ".0%", height=520,
                         colors=[BLUE if f in engineered else MUTED for f in top.index],
                         hover="%{y}: %{x:.1%} of the model's attention<extra></extra>").update_xaxes(tickformat=".0%"),
                    width="stretch")
with c2:
    st.markdown("<br>", unsafe_allow_html=True)
    comps_share = (imp[["knn_log_price", "knn_log_ppsf", "knn_dist_mi"]].sum() / imp.sum())
    insight(f"**Blue bars are features we engineered.** {sum(f in engineered for f in top.index)} of the top 15 inputs didn't "
            f"exist in the raw data. Together, the neighbor-price features (our 'comps') carry {comps_share:.0%} of the model's "
            "attention, the biggest signal after the size of the home itself.")
    st.caption("Share of the model's total attention, measured with SHAP: the average size of each input's push on the "
               "predicted price, across all 19,430 homes.")

st.markdown("### Every feature the model uses, and why")
rows = [
    # (feature, group, plain meaning, why)
    ("livingArea", "Raw", "Interior square feet", "The #1 driver of price (Step 2)."),
    ("bedrooms / bathrooms", "Raw", "Room counts", "Bathrooms signal a newer, larger, better-equipped home."),
    ("lot_sqft", "Raw (fixed)", "Land size in sqft (acres converted)", "Land has value, especially in the suburbs."),
    ("yearBuilt", "Raw", "Construction year (house age = 2024 − year)", "Newer homes have modern layouts and fewer repairs. Trees use year and age identically, so we keep one."),
    ("homeType", "Raw", "Single family / townhouse / condo / ...", "A condo and a house of equal size price differently."),
    ("latitude, longitude, ZIP", "Raw", "Exact location", "Lets the model learn price 'hot spots' directly."),
    ("bed_bath_ratio", "Engineered", "Bedrooms ÷ bathrooms", "4 beds with 1 bath is a very different home from 4 beds with 3."),
    ("sqft_per_bed", "Engineered", "Living area ÷ bedrooms", "Spacious rooms vs. chopped-up space."),
    ("dist_downtown / TMC / Galleria / Energy Corridor / UH", "Engineered", "Straight-line miles to 5 job hubs (Haversine formula)", "Houston is car-commute city: commute to jobs drives demand."),
    ("knn_log_ppsf", "Engineered", "Median price per sqft of the 10 nearest listings", "What appraisers call 'comps', the neighborhood-level average price."),
    ("knn_log_price, knn_dist_mi", "Engineered", "Median price of 10 nearest listings; how far away they are", "Neighborhood price level, and how dense/rural the area is."),
    ("zip_avg_agi", "External (IRS)", "Average household income in the ZIP", "Wealthier areas support higher prices."),
    ("zip_zhvi, 1-yr & 5-yr growth", "External (Zillow)", "Typical ZIP home value and its recent trend", "Is the neighborhood heating up or cooling down?"),
    ("zip_zori", "External (Zillow)", "Typical ZIP rent", "Rent is what a home would earn if not owner-occupied."),
    ("is_new_construction, has_builder, is_premier_builder", "Raw flags", "New build / builder listing", "New homes command a premium (Step 2)."),
    ("is_bank_owned, is_auction", "Raw flags", "Foreclosure / auction", "Distressed sales sell at a discount."),
]
feat_keys = {"livingArea": ["livingArea"], "bedrooms / bathrooms": ["bedrooms", "bathrooms"], "lot_sqft": ["lot_sqft"],
             "yearBuilt": ["yearBuilt"], "homeType": ["homeType"], "latitude, longitude, ZIP": ["latitude", "longitude", "zipcode"],
             "bed_bath_ratio": ["bed_bath_ratio"], "sqft_per_bed": ["sqft_per_bed"], "knn_log_ppsf": ["knn_log_ppsf"],
             "knn_log_price, knn_dist_mi": ["knn_log_price", "knn_dist_mi"], "zip_avg_agi": ["zip_avg_agi"],
             "zip_zhvi, 1-yr & 5-yr growth": ["zip_zhvi", "zip_zhvi_1y_growth", "zip_zhvi_5y_growth"], "zip_zori": ["zip_zori"],
             "is_new_construction, has_builder, is_premier_builder": ["is_new_construction", "has_builder", "is_premier_builder"],
             "is_bank_owned, is_auction": ["is_bank_owned", "is_auction"],
             "dist_downtown / TMC / Galleria / Energy Corridor / UH": ["dist_downtown_mi", "dist_tmc_mi", "dist_galleria_mi",
                                                                      "dist_energy_corridor_mi", "dist_uh_mi"]}
feat_table = pd.DataFrame(rows, columns=["Feature", "Type", "Meaning", "Why it should matter"])
feat_table["Best importance rank"] = [int(min(rank[k] for k in feat_keys[f])) for f in feat_table.Feature]
table(feat_table.set_index("Feature"))
st.caption(f"Importance rank: position among all {len(imp)} model inputs by average SHAP impact (1 = most important). "
           "For grouped rows, the best-ranked member is shown.")

st.warning("**Why isn't the home's own price per square foot a feature?** Because it's calculated *from the price*. "
           "Giving the model price ÷ sqft is giving it the answer. Instead we use the **neighbors'** price per sqft, "
           "and never include the home itself (we explicitly drop it from its own neighbor list).")

st.markdown("### Does the 'nearby comps' feature actually work?")
s = scored[scored.livingArea.notna()].sample(4000, random_state=1)
# Neighbors come only from homes outside this sample, so no home is its own comp.
lst = load_listings()
comps = CompsIndex(lst[~lst.zpid.isin(s.zpid)]).query(s, exclude_self=False)
x, y = np.exp(comps.knn_log_ppsf), s.price / s.livingArea
r = np.corrcoef(np.log(x), np.log(y))[0, 1]
st.code(f"""sample = df.sample(4000, random_state=1)
neighbors = CompsIndex(df[~df.zpid.isin(sample.zpid)])   # comps from homes OUTSIDE the sample
comps = neighbors.query(sample)
np.corrcoef(comps["knn_log_ppsf"], np.log(sample["price"] / sample["livingArea"]))[0, 1]
# -> {r:.2f}""", language="python")
c1, c2 = st.columns([1.2, 1])
with c1:
    fig = go.Figure(go.Scattergl(x=x, y=y, mode="markers", marker=dict(color=BLUE, size=5, opacity=0.3),
                                 hovertemplate="neighbors $%{x:.0f}/sqft<br>this home $%{y:.0f}/sqft<extra></extra>"))
    fig.add_trace(go.Scatter(x=[50, 800], y=[50, 800], mode="lines", line=dict(color=INK_2, dash="dot", width=1), hoverinfo="skip"))
    style(fig, 360)
    ticks = dict(tickvals=[50, 100, 200, 400, 800], ticktext=["$50", "$100", "$200", "$400", "$800"])
    fig.update_xaxes(type="log", title="Median $/sqft of the 10 nearest listings", range=[np.log10(50), np.log10(800)], **ticks)
    fig.update_yaxes(type="log", title="This home's $/sqft", range=[np.log10(50), np.log10(800)], **ticks)
    st.plotly_chart(fig, width="stretch")
with c2:
    st.metric("Correlation (log scale)", f"{r:.2f}")
    insight("A home's price per sqft closely tracks its neighbors' (dots hug the dotted 'equal' line). This one feature "
            "packs in school zones, flood risk and street appeal, things that aren't in the data directly.")

takeaway("Location features turned raw coordinates into things humans understand: **commute**, **comparable homes**, "
         "and **neighborhood income**. On the next page, each one gets an error-reduction score.")
st.page_link("views/metric.py", label="Next: choosing our metric →")
