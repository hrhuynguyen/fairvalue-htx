"""Feature definitions shared by training and the app."""
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

K_NEIGHBORS = 10

STRUCTURAL = ["bedrooms", "bathrooms", "livingArea", "lot_sqft", "yearBuilt", "homeType",
              "bed_bath_ratio", "sqft_per_bed"]
LOCATION = ["latitude", "longitude", "zipcode"]
HUB_DIST = ["dist_downtown_mi", "dist_tmc_mi", "dist_galleria_mi", "dist_energy_corridor_mi", "dist_uh_mi"]
SPATIAL_LAG = ["knn_log_ppsf", "knn_log_price", "knn_dist_mi"]
ZIP_MARKET = ["zip_avg_agi", "zip_tax_returns", "zip_zhvi", "zip_zhvi_1y_growth", "zip_zhvi_5y_growth", "zip_zori"]
LISTING_FLAGS = ["is_new_construction", "has_builder", "is_premier_builder", "is_bank_owned", "is_auction"]

# Staged feature sets for the before/after table. Each stage adds one idea.
STAGES = [
    ("1. Structure only (beds, baths, sqft, lot, year, type)", STRUCTURAL),
    ("2. + Raw location (lat/lon, ZIP)", STRUCTURAL + LOCATION),
    ("3. + Distance to Houston job hubs", STRUCTURAL + LOCATION + HUB_DIST),
    ("4. + Neighborhood comps (10 nearest listings)", STRUCTURAL + LOCATION + HUB_DIST + SPATIAL_LAG),
    ("5. + ZIP income, value & rent trends (IRS, Zillow)", STRUCTURAL + LOCATION + HUB_DIST + SPATIAL_LAG + ZIP_MARKET),
    ("6. + Listing type (new build, builder, foreclosure)", STRUCTURAL + LOCATION + HUB_DIST + SPATIAL_LAG + ZIP_MARKET + LISTING_FLAGS),
]
FEATURES = STAGES[-1][1]
CATEGORICAL = ["homeType", "zipcode"]
EARTH_RADIUS_MI = 3958.8


class CompsIndex:
    """Nearest-neighbor comps from a reference set of listings (haversine BallTree)."""

    def __init__(self, ref: pd.DataFrame):
        self.tree = BallTree(np.radians(ref[["latitude", "longitude"]].to_numpy()), metric="haversine")
        self.log_price = np.log(ref.price.to_numpy())
        self.log_ppsf = np.log((ref.price / ref.livingArea).to_numpy())

    def query(self, df: pd.DataFrame, exclude_self: bool) -> pd.DataFrame:
        """exclude_self=True means df is the reference set itself (row i == ref row i)."""
        k = K_NEIGHBORS + int(exclude_self)
        dist, idx = self.tree.query(np.radians(df[["latitude", "longitude"]].to_numpy()), k=k)
        if exclude_self:
            # Drop the row's own entry, not just the first hit: co-located condos tie at distance 0.
            keep = idx != np.arange(len(df))[:, None]
            keep[keep.all(axis=1), -1] = False
            dist = dist[keep].reshape(len(df), K_NEIGHBORS)
            idx = idx[keep].reshape(len(df), K_NEIGHBORS)
        return pd.DataFrame({
            "knn_log_ppsf": np.nanmedian(self.log_ppsf[idx], axis=1),
            "knn_log_price": np.median(self.log_price[idx], axis=1),
            "knn_dist_mi": dist.mean(axis=1) * EARTH_RADIUS_MI,
        }, index=df.index)


def prepare(df: pd.DataFrame, categories: dict | None = None) -> pd.DataFrame:
    """Cast categoricals consistently so train and inference share category codes."""
    X = df.copy()
    for c in CATEGORICAL:
        cats = categories[c] if categories else sorted(X[c].dropna().unique())
        X[c] = pd.Categorical(X[c], categories=cats)
    for c in LISTING_FLAGS:
        X[c] = X[c].astype(int)
    return X
