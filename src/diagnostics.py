"""Evidence for the Model and Validation pages: model comparison, overfitting checks,
and what each home feature is worth according to the model.

Writes reports/diagnostics.json. Run after train.py.
"""
from pathlib import Path
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from features import CATEGORICAL, FEATURES, CompsIndex, prepare
from train import PARAMS, SEED, add_comps, metrics, ridge_baseline

ROOT = Path(__file__).resolve().parents[1]


def as_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """Random forest needs plain numbers: category codes, and -1 for missing."""
    X = X.copy()
    for c in CATEGORICAL:
        X[c] = X[c].cat.codes
    return X.astype(float).fillna(-1)


def fold_run(df, y, tr_idx, va_idx, record_curve):
    tr, va = add_comps(df.iloc[tr_idx], df.iloc[va_idx])
    curve = {}
    gbm = lgb.LGBMRegressor(**dict(PARAMS, objective="regression"))
    gbm.fit(tr[FEATURES], y[tr_idx], eval_set=[(tr[FEATURES], y[tr_idx]), (va[FEATURES], y[va_idx])],
            eval_names=["train", "validation"], eval_metric="l1",
            callbacks=[lgb.record_evaluation(curve)] if record_curve else None)
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.5, n_jobs=2, random_state=SEED)
    rf.fit(as_numeric(tr[FEATURES]), y[tr_idx])
    return {
        "va_idx": va_idx,
        "gbm_train": gbm.predict(tr[FEATURES]), "tr_idx": tr_idx,
        "gbm_val": gbm.predict(va[FEATURES]),
        "rf_val": rf.predict(as_numeric(va[FEATURES])),
        "curve": {k: v["l1"] for k, v in curve.items()} if record_curve else None,
    }


def value_drivers(raw: pd.DataFrame, n=3000) -> list:
    """Counterfactuals on real single-family homes: change one thing, re-price, compare."""
    bundle = joblib.load(ROOT / "models/fair_value.joblib")
    comps = CompsIndex(raw)
    sample = raw[(raw.homeType == "SINGLE_FAMILY") & raw.livingArea.notna() & raw.bedrooms.notna()
                 & raw.bathrooms.notna()].sample(n, random_state=SEED)
    sample = sample.join(comps.query(sample, exclude_self=True))

    def price(d):
        d = d.assign(bed_bath_ratio=d.bedrooms / d.bathrooms.replace(0, np.nan),
                     sqft_per_bed=d.livingArea / d.bedrooms.replace(0, np.nan))
        return np.exp(bundle["point"].predict(prepare(d, bundle["categories"])[FEATURES]))

    base = price(sample)
    scenarios = {
        "+1 bathroom": sample.assign(bathrooms=sample.bathrooms + 1),
        "+1 bedroom (same sqft)": sample.assign(bedrooms=sample.bedrooms + 1),
        "+250 sqft of living area": sample.assign(livingArea=sample.livingArea + 250),
        "+10% living area": sample.assign(livingArea=sample.livingArea * 1.1),
        "Built 10 years later": sample.assign(yearBuilt=sample.yearBuilt + 10),
        "Double the lot size": sample.assign(lot_sqft=sample.lot_sqft * 2),
        "Sold as new construction": sample.assign(is_new_construction=True, has_builder=True),
        "Foreclosure / bank-owned": sample.assign(is_bank_owned=True),
    }
    out = []
    for name, d in scenarios.items():
        new = price(d)
        out.append({"change": name, "median_pct": float(np.median(new / base - 1) * 100),
                    "median_dollars": float(np.median(new - base))})
    return out


def main():
    raw = pd.read_parquet(ROOT / "data/processed/listings.parquet")
    categories = {c: sorted(raw[c].dropna().unique()) for c in CATEGORICAL}
    df = prepare(raw, categories)
    y = np.log(df.price.to_numpy())
    loc_groups = df.latitude.round(4).astype(str) + "," + df.longitude.round(4).astype(str)
    folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(df, y, loc_groups))

    runs = Parallel(n_jobs=5)(delayed(fold_run)(df, y, tr, va, i == 0) for i, (tr, va) in enumerate(folds))
    gbm, rf = np.zeros(len(df)), np.zeros(len(df))
    gap = []
    for r in runs:
        gbm[r["va_idx"]], rf[r["va_idx"]] = r["gbm_val"], r["rf_val"]
        gap.append({"train_MdAPE_%": metrics(y[r["tr_idx"]], r["gbm_train"])["MdAPE_%"],
                    "validation_MdAPE_%": metrics(y[r["va_idx"]], r["gbm_val"])["MdAPE_%"]})
    ridge = ridge_baseline(df, y, folds)

    out = {
        "models": [
            {"model": "Linear regression (baseline)", **metrics(y, ridge)},
            {"model": "Random forest", **metrics(y, rf)},
            {"model": "LightGBM (gradient boosting)", **metrics(y, gbm)},
            {"model": "Ensemble: average of LightGBM + random forest", **metrics(y, (gbm + rf) / 2)},
        ],
        "fold_gap": gap,
        # Mean absolute error of log price is roughly the mean % error.
        "learning_curve": runs[0]["curve"],
        "value_drivers": value_drivers(raw),
    }
    json.dump(out, open(ROOT / "reports/diagnostics.json", "w"), indent=2)
    for m in out["models"]:
        print(f"{m['model']:<50} MdAPE {m['MdAPE_%']:.2f}%  R2 {m['R2_log']:.3f}")
    print(pd.DataFrame(gap).round(2))
    print(pd.DataFrame(out["value_drivers"]).round(1))


if __name__ == "__main__":
    main()
