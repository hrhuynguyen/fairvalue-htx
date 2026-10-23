"""Build the modeling dataset from the raw Zillow scrape plus public ZIP-level data.

Why not use properties_clean_v2.csv directly? It fills missing beds/sqft with 0,
mixes lot sizes in sqft and acres, imputes yearBuilt=2001.5, and keeps 3,400
builder "floor plan" entries that are not real homes. We rebuild from the raw
JSON and only borrow yearBuilt from the clean file.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
EXT = ROOT / "data" / "external"
OUT = ROOT / "data" / "processed"

SNAPSHOT = "2024-05-31"  # scrape date is early June 2024

HUBS = {
    "downtown": (29.7604, -95.3698),
    "tmc": (29.7079, -95.3978),
    "galleria": (29.7399, -95.4617),
    "energy_corridor": (29.7848, -95.6265),
    "uh": (29.7199, -95.3422),
}


def haversine_mi(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def load_raw() -> pd.DataFrame:
    records = json.load(open(RAW / "houston_housing market 2024_light.json"))
    rows = []
    for r in records:
        h = r["hdpData"]["homeInfo"]
        sub = h.get("listing_sub_type", {})
        lot, unit = h.get("lotAreaValue"), h.get("lotAreaUnit")
        lot_sqft = lot * 43560 if unit == "acres" else lot
        rows.append({
            "zpid": int(r["zpid"]),
            "price": h.get("price"),
            "bedrooms": h.get("bedrooms"),
            "bathrooms": h.get("bathrooms"),
            "livingArea": h.get("livingArea"),
            "lot_sqft": lot_sqft,
            "lot_unit": unit,
            "homeType": h.get("homeType"),
            "latitude": h.get("latitude"),
            "longitude": h.get("longitude"),
            "zipcode": h.get("zipcode"),
            "city": h.get("city"),
            "address": r.get("address"),
            "unit": h.get("unit"),
            "statusText": r.get("statusText"),
            "detailUrl": r.get("detailUrl"),
            "imgSrc": r.get("imgSrc"),
            "brokerName": r.get("brokerName"),
            "builderName": r.get("builderName"),
            "group_type": h.get("group_type"),
            "is_new_construction": bool(sub.get("is_newHome")),
            "is_bank_owned": bool(sub.get("is_bankOwned")) or bool(h.get("isPreforeclosureAuction")),
            "is_auction": bool(sub.get("is_forAuction")),
            "is_premier_builder": bool(h.get("isPremierBuilder")),
            "is_non_owner_occupied": bool(h.get("isNonOwnerOccupied")),
            "daysOnZillow": h.get("daysOnZillow"),
            "priceReduction": h.get("priceReduction"),
            "priceChange": h.get("priceChange"),
            # Benchmarks only. Never used as model features (target leakage).
            "zestimate": h.get("zestimate"),
            "rentZestimate": h.get("rentZestimate"),
            "taxAssessedValue": h.get("taxAssessedValue"),
        })
    return pd.DataFrame(rows)


def load_year_built() -> pd.Series:
    clean = pd.read_csv(RAW / "properties_clean_v2.csv", usecols=["zpid", "yearBuilt"])
    clean.loc[clean.yearBuilt == 2001.5, "yearBuilt"] = np.nan  # placeholder fill-in, not a real year
    return clean.set_index("zpid").yearBuilt


def load_irs() -> pd.DataFrame:
    irs = pd.read_csv(EXT / "irs_soi_21zpallnoagi.csv", usecols=["STATE", "ZIPCODE", "N1", "A00100"])
    irs = irs[(irs.STATE == "TX") & (irs.ZIPCODE.between(77000, 77999))]
    return pd.DataFrame({
        "zipcode": irs.ZIPCODE.astype(str),
        "zip_avg_agi": irs.A00100 * 1000 / irs.N1,  # A00100 is in $ thousands
        "zip_tax_returns": irs.N1,
    })


def load_zillow_zip() -> pd.DataFrame:
    zhvi = pd.read_csv(EXT / "zhvi_zip.csv", dtype={"RegionName": str})
    zhvi = zhvi[zhvi.State == "TX"]
    latest = zhvi.columns[-1]
    out = pd.DataFrame({
        "zipcode": zhvi.RegionName.str.zfill(5),
        "zip_zhvi": zhvi[SNAPSHOT],
        "zip_zhvi_1y_growth": zhvi[SNAPSHOT] / zhvi["2023-05-31"] - 1,
        "zip_zhvi_5y_growth": zhvi[SNAPSHOT] / zhvi["2019-05-31"] - 1,
        # For the app only: how the ZIP moved since the listing snapshot.
        "zip_zhvi_latest": zhvi[latest],
    })
    zori = pd.read_csv(EXT / "zori_zip.csv", dtype={"RegionName": str})
    zori = zori[zori.State == "TX"]
    zori = pd.DataFrame({"zipcode": zori.RegionName.str.zfill(5), "zip_zori": zori[SNAPSHOT]})
    return out.merge(zori, on="zipcode", how="left"), latest


def write_cleaning_report(log, df, latest):
    """Numbers for the Data and Preprocessing pages of the dashboard."""
    clean = pd.read_csv(RAW / "properties_clean_v2.csv")
    raw = load_raw()
    issues = {
        "clean_csv_rows": len(clean),
        "clean_csv_zero_beds": int((clean.bedrooms == 0).sum()),
        "clean_csv_zero_sqft": int((clean.livingArea == 0).sum()),
        "clean_csv_year_2001_5": int((clean.yearBuilt == 2001.5).sum()),
        "clean_csv_lot_under_100": int((clean.lotAreaValue < 100).sum()),
        "raw_lot_in_acres": int((raw.lot_unit == "acres").sum()),
        "raw_builder_plans": int((raw.group_type == "BUILDER_COMMUNITY").sum()),
        "raw_lots_land": int((raw.homeType == "LOT").sum()),
        "raw_missing_sqft": int(raw.livingArea.isna().sum()),
        "raw_missing_beds": int(raw.bedrooms.isna().sum()),
        "raw_missing_lot": int(raw.lot_sqft.isna().sum()),
        "raw_has_zestimate": int(raw.zestimate.notna().sum()),
        "raw_has_tax_value": int(raw.taxAssessedValue.notna().sum()),
    }
    facts = {
        "funnel": log,
        "issues": issues,
        "n_final": len(df),
        "n_zips": int(df.zipcode.nunique()),
        "n_cities": int(df.city.nunique()),
        "lat_range": [float(df.latitude.min()), float(df.latitude.max())],
        "lon_range": [float(df.longitude.min()), float(df.longitude.max())],
        "zhvi_latest_month": latest,
        "new_construction_share": float(df.is_new_construction.mean()),
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    json.dump(facts, open(ROOT / "reports/cleaning.json", "w"), indent=2, default=str)


def write_zhvi_extract(zips):
    """Houston-area slice of the 124 MB Zillow file (all months), small enough to ship with the app."""
    z = pd.read_csv(EXT / "zhvi_zip.csv", dtype={"RegionName": str})
    z["RegionName"] = z.RegionName.str.zfill(5)
    keep = (z.State == "TX") & (z.Metro.fillna("").str.startswith("Houston") | z.RegionName.isin(zips))
    z[keep].drop(columns=["RegionID", "SizeRank", "RegionType", "StateName"]).to_parquet(OUT / "zhvi_houston.parquet", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    n0 = len(df)
    df["yearBuilt"] = df.zpid.map(load_year_built())

    log = [("raw listings", n0)]
    df = df[df.group_type != "BUILDER_COMMUNITY"]; log.append(("drop builder floor plans", len(df)))
    df = df[df.homeType != "LOT"]; log.append(("drop vacant land", len(df)))
    df = df[df.latitude.notna() & df.price.notna()]; log.append(("drop missing price/coords", len(df)))
    df = df[df.price.between(50_000, 15_000_000)]; log.append(("price in [$50k, $15M]", len(df)))
    df = df[df.livingArea.isna() | df.livingArea.between(300, 20_000)]; log.append(("living area in [300, 20k] sqft", len(df)))
    ppsf = df.price / df.livingArea
    df = df[ppsf.isna() | ppsf.between(25, 2_000)]; log.append(("price/sqft in [$25, $2000]", len(df)))
    df.loc[~df.lot_sqft.between(200, 50 * 43560), "lot_sqft"] = np.nan
    df.loc[df.bedrooms > 15, "bedrooms"] = np.nan
    df.loc[df.bathrooms > 15, "bathrooms"] = np.nan

    for name, (lat, lon) in HUBS.items():
        df[f"dist_{name}_mi"] = haversine_mi(df.latitude, df.longitude, lat, lon)
    df["age"] = 2024 - df.yearBuilt
    df["bed_bath_ratio"] = df.bedrooms / df.bathrooms.replace(0, np.nan)
    df["sqft_per_bed"] = df.livingArea / df.bedrooms.replace(0, np.nan)
    df["has_builder"] = df.builderName.notna()

    df = df.merge(load_irs(), on="zipcode", how="left")
    zz, latest = load_zillow_zip()
    df = df.merge(zz, on="zipcode", how="left")
    df["zip_value_change_since_listing"] = df.zip_zhvi_latest / df.zip_zhvi - 1

    df = df.reset_index(drop=True)
    df.to_parquet(OUT / "listings.parquet", index=False)
    write_cleaning_report(log, df, latest)
    write_zhvi_extract(set(df.zipcode))

    print("Cleaning funnel:")
    for step, n in log:
        print(f"  {step:<35} {n:>7,}")
    print(f"ZHVI latest month available: {latest}")
    print("Coverage:", df[["livingArea", "yearBuilt", "lot_sqft", "zip_avg_agi", "zip_zhvi", "zip_zori"]].notna().mean().round(3).to_dict())
    print(df.homeType.value_counts().to_dict())


if __name__ == "__main__":
    main()
