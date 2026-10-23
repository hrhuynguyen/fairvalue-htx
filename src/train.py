"""Train and evaluate the fair-value model.

Outputs
  reports/metrics.json         every number quoted in the pitch
  reports/feature_stages.md    before/after table, one row per feature idea
  models/fair_value.joblib     final point + 80% interval models (fit on all data)
  data/processed/scored.parquet  out-of-fold fair value, interval, flag, SHAP per listing
"""
from pathlib import Path
import json
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from ensemble_model import MEMBERS, EnsembleRegressor, fit_member, member_contrib, member_predict, optimize_weights
from features import CATEGORICAL, FEATURES, SPATIAL_LAG, STAGES, CompsIndex, prepare

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
ALPHA = 0.20  # 80% prediction intervals
N_BANDS = 10  # price bands (deciles of predicted price) for interval calibration

PARAMS = dict(n_estimators=800, learning_rate=0.05, num_leaves=63, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              random_state=SEED, n_jobs=2, verbose=-1)


def metrics(y_log, pred_log) -> dict:
    price, pred = np.exp(y_log), np.exp(pred_log)
    ape = np.abs(pred - price) / price
    ss_res = np.sum((y_log - pred_log) ** 2)
    return {
        "MAE_$": float(np.mean(np.abs(pred - price))),
        "MdAE_$": float(np.median(np.abs(pred - price))),
        "MdAPE_%": float(np.median(ape) * 100),
        "within_10%": float(np.mean(ape <= 0.10) * 100),
        "R2_log": float(1 - ss_res / np.sum((y_log - y_log.mean()) ** 2)),
    }


def add_comps(train: pd.DataFrame, other: pd.DataFrame | None):
    comps = CompsIndex(train)
    tr = train.join(comps.query(train, exclude_self=True))
    return tr, (other.join(comps.query(other, exclude_self=False)) if other is not None else None)


def _fit_fold(df, y, features, tr_idx, va_idx, objective, alpha, explain):
    tr, va = df.iloc[tr_idx], df.iloc[va_idx]
    if any(f in features for f in SPATIAL_LAG):
        tr, va = add_comps(tr, va)
    params = dict(PARAMS, objective=objective, **({"alpha": alpha} if alpha is not None else {}))
    model = lgb.LGBMRegressor(**params).fit(tr[features], y[tr_idx])
    pred = model.predict(va[features])
    if not explain:
        return va_idx, pred, None, None
    ex = shap.TreeExplainer(model)
    return va_idx, pred, ex.shap_values(va[features]), ex.expected_value


def run_cv(df, y, features, folds, objective="regression", alpha=None, explain=False):
    """Out-of-fold predictions (log price). Comps are rebuilt inside each fold, so no leakage."""
    oof = np.zeros(len(df))
    shap_vals = np.zeros((len(df), len(features))) if explain else None
    base = np.zeros(len(df)) if explain else None
    jobs = (delayed(_fit_fold)(df, y, features, tr, va, objective, alpha, explain) for tr, va in folds)
    for va_idx, pred, sv, ev in Parallel(n_jobs=len(folds))(jobs):
        oof[va_idx] = pred
        if explain:
            shap_vals[va_idx], base[va_idx] = sv, ev
    return oof, shap_vals, base


def _ensemble_fold(df, y, name, tr_idx, va_idx, contrib):
    tr, va = add_comps(df.iloc[tr_idx], df.iloc[va_idx])
    model = fit_member(name, tr[FEATURES], y[tr_idx])
    return (name, va_idx, member_predict(name, model, va[FEATURES]),
            member_contrib(name, model, va[FEATURES]) if contrib else None)


def ensemble_cv(df, y, folds, contrib=True):
    """OOF predictions (n x members) and, optionally, OOF SHAP contributions (n x members x features+1)."""
    P = np.zeros((len(df), len(MEMBERS)))
    C = np.zeros((len(df), len(MEMBERS), len(FEATURES) + 1)) if contrib else None
    jobs = (delayed(_ensemble_fold)(df, y, n, tr, va, contrib) for n in MEMBERS for tr, va in folds)
    for name, va_idx, pred, c in Parallel(n_jobs=len(MEMBERS) * len(folds))(jobs):
        P[va_idx, MEMBERS.index(name)] = pred
        if contrib:
            C[va_idx, MEMBERS.index(name)] = c
    return P, C


def nested_weights(P, y, groups):
    """Per-row ensemble weights learned on the OTHER 4/5 of rows, so a home's score never tunes its own weights."""
    W = np.zeros_like(P)
    for tr, va in GroupKFold(n_splits=5, shuffle=True, random_state=7).split(P, y, groups):
        W[va] = optimize_weights(P[tr], y[tr])
    return W


def ridge_baseline(df, y, folds):
    num = ["bedrooms", "bathrooms", "livingArea", "lot_sqft", "yearBuilt"]
    X = df[num + ["homeType"]].copy()
    X["livingArea"] = np.log(X.livingArea)
    X["lot_sqft"] = np.log(X.lot_sqft)
    pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), ["homeType"])])
    oof = np.zeros(len(df))
    for tr_idx, va_idx in folds:
        m = make_pipeline(pre, RidgeCV()).fit(X.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = m.predict(X.iloc[va_idx])
    return oof


def main():
    t0 = time.time()
    raw = pd.read_parquet(ROOT / "data/processed/listings.parquet")
    categories = {c: sorted(raw[c].dropna().unique()) for c in CATEGORICAL}
    df = prepare(raw, categories)
    y = np.log(df.price.to_numpy())

    # Primary CV: 5 folds, grouped by exact coordinates so co-located units never straddle folds.
    loc_groups = df.latitude.round(4).astype(str) + "," + df.longitude.round(4).astype(str)
    folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(df, y, loc_groups))
    # Stress test: whole ZIP codes held out (model never saw the neighborhood).
    zip_folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(df, y, df.zipcode))

    results = {"n_listings": len(df), "cv": "5-fold, grouped by location", "stages": []}
    oof = ridge_baseline(df, y, folds)
    results["stages"].append({"stage": "0. Linear regression baseline (structure only)", **metrics(y, oof)})
    print(results["stages"][-1])

    for name, feats in STAGES:  # feature ablation with LightGBM alone
        oof, _, _ = run_cv(df, y, feats, folds)
        results["stages"].append({"stage": name, **metrics(y, oof)})
        print(results["stages"][-1], f"{time.time() - t0:.0f}s")

    # Final model: weighted ensemble of LightGBM + CatBoost + XGBoost (comparison of weighting methods: ensemble.py).
    P, C = ensemble_cv(df, y, folds)
    W = nested_weights(P, y, loc_groups.to_numpy())
    oof_point = (P * W).sum(axis=1)
    contrib = np.einsum("nm,nmf->nf", W, C)  # SHAP is linear, so the ensemble's SHAP is the weighted sum
    shap_vals, base = contrib[:, :-1], contrib[:, -1]
    assert np.allclose(shap_vals.sum(axis=1) + base, oof_point, atol=1e-4)
    weights = optimize_weights(P, y)
    results["stages"].append({"stage": "7. Ensemble: LightGBM + CatBoost + XGBoost", **metrics(y, oof_point)})
    results["ensemble"] = {"members": MEMBERS, "weights": dict(zip(MEMBERS, map(float, weights))),
                           "member_metrics": {n: metrics(y, P[:, i]) for i, n in enumerate(MEMBERS)}}
    print(results["stages"][-1], results["ensemble"]["weights"], f"{time.time() - t0:.0f}s")

    Pz, _ = ensemble_cv(df, y, zip_folds, contrib=False)
    zip_oof = Pz @ weights
    results["unseen_zip"] = metrics(y, zip_oof)
    print("unseen ZIP:", results["unseen_zip"])

    # Conformalized quantile regression (CQR) for 80% intervals, calibrated per price band (Mondrian)
    # and per tail. A single global adjustment left cheap homes' ranges too narrow on the low side and
    # expensive homes' too narrow on the high side; this gives every band ~10% below and ~10% above.
    lo, _, _ = run_cv(df, y, FEATURES, folds, objective="quantile", alpha=ALPHA / 2)
    hi, _, _ = run_cv(df, y, FEATURES, folds, objective="quantile", alpha=1 - ALPHA / 2)
    edges = np.quantile(oof_point, np.linspace(0, 1, N_BANDS + 1)[1:-1])
    band = np.searchsorted(edges, oof_point)

    def calibrate(mask):
        adj = np.zeros((N_BANDS, 2))
        for b in range(N_BANDS):
            m = mask & (band == b)
            level = min(1.0, (1 - ALPHA / 2) * (1 + 1 / m.sum()))
            adj[b] = np.quantile(lo[m] - y[m], level), np.quantile(y[m] - hi[m], level)
        return adj

    half = np.random.default_rng(SEED).random(len(df)) < 0.5  # calibrate on one half, verify on the other
    a = calibrate(half)
    lo_h, hi_h, test = lo - a[band, 0], hi + a[band, 1], ~half
    bands_check = [{"band": b + 1, "fair_value_from": float(np.exp(edges[b - 1])) if b else 0.0,
                    "below_%": float(np.mean(y[test & (band == b)] < lo_h[test & (band == b)]) * 100),
                    "above_%": float(np.mean(y[test & (band == b)] > hi_h[test & (band == b)]) * 100)}
                   for b in range(N_BANDS)]
    holdout_cov = np.mean((y[test] >= lo_h[test]) & (y[test] <= hi_h[test]))
    adj = calibrate(np.ones(len(df), dtype=bool))
    lo, hi = lo - adj[band, 0], hi + adj[band, 1]
    results["interval"] = {
        "nominal_coverage_%": (1 - ALPHA) * 100,
        "holdout_coverage_%": float(holdout_cov * 100),
        "median_width_%": float(np.median(np.exp(hi - lo) - 1) * 100),
        "bands_holdout": bands_check,
    }
    print("interval:", results["interval"])

    # Benchmarks on the same rows. Zestimate is NOT a fair fight: Zillow sees the list price.
    for col in ["zestimate", "taxAssessedValue"]:
        m = df[col].notna().to_numpy() & (df[col] > 0).to_numpy()
        results[f"benchmark_{col}"] = {"n": int(m.sum()),
                                       col: metrics(y[m], np.log(df[col].to_numpy()[m])),
                                       "our_model_same_rows": metrics(y[m], oof_point[m])}
    print("zestimate bench:", results["benchmark_zestimate"])

    # Score every listing with its out-of-fold fair value (no model ever saw that home's price).
    scored = raw[["zpid", "price", "address", "unit", "city", "zipcode", "latitude", "longitude",
                  "bedrooms", "bathrooms", "livingArea", "lot_sqft", "yearBuilt", "homeType",
                  "statusText", "detailUrl", "imgSrc", "builderName", "is_new_construction",
                  "is_bank_owned", "daysOnZillow", "priceReduction", "zestimate", "rentZestimate",
                  "taxAssessedValue", "zip_avg_agi", "zip_zhvi", "zip_zhvi_1y_growth",
                  "zip_zhvi_5y_growth", "zip_zori", "zip_value_change_since_listing",
                  "dist_downtown_mi", "dist_tmc_mi", "dist_uh_mi"]].copy()
    scored["fair_value"] = np.exp(oof_point)
    scored["fair_lo"] = np.exp(lo)
    scored["fair_hi"] = np.exp(hi)
    scored["gap_pct"] = scored.price / scored.fair_value - 1
    scored["flag"] = np.select([scored.price < scored.fair_lo, scored.price > scored.fair_hi],
                               ["Underpriced", "Overpriced"], "Fair")
    scored["shap_base"] = base
    for i, f in enumerate(FEATURES):
        scored[f"shap__{f}"] = shap_vals[:, i]
    scored.to_parquet(ROOT / "data/processed/scored.parquet", index=False)
    results["flags"] = scored.flag.value_counts().to_dict()
    imp = pd.Series(np.abs(shap_vals).mean(axis=0), index=FEATURES).sort_values(ascending=False)
    results["shap_importance"] = imp.round(4).to_dict()

    # Final models on all data for pricing homes that are not in the dataset.
    full, _ = add_comps(df, None)
    final = {
        "point": EnsembleRegressor({n: fit_member(n, full[FEATURES], y, n_jobs=8) for n in MEMBERS}, weights),
        "lo": lgb.LGBMRegressor(**dict(PARAMS, objective="quantile", alpha=ALPHA / 2, n_jobs=8)).fit(full[FEATURES], y),
        "hi": lgb.LGBMRegressor(**dict(PARAMS, objective="quantile", alpha=1 - ALPHA / 2, n_jobs=8)).fit(full[FEATURES], y),
        "band_edges": edges, "band_adj": adj, "features": FEATURES, "categories": categories,
    }
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(final, ROOT / "models/fair_value.joblib")

    (ROOT / "reports").mkdir(exist_ok=True)
    json.dump(results, open(ROOT / "reports/metrics.json", "w"), indent=2)
    table = pd.DataFrame(results["stages"]).set_index("stage")
    with open(ROOT / "reports/feature_stages.md", "w") as f:
        f.write(f"5-fold CV (grouped by location), n = {len(df):,} listings\n\n")
        f.write(table.round(3).to_markdown())
    print(table.round(3).to_string())
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
