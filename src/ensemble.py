"""Can an ensemble beat LightGBM alone? And how should the models be weighted?

1. Train four diverse models on the SAME 5 location-grouped folds -> out-of-fold (OOF) predictions.
2. Try several ways to choose weights. Weights are learned from data too, so every weighting
   method is scored with a second (nested) cross-validation over the OOF predictions.
3. Check the gain isn't noise: paired bootstrap CI and fold-by-fold wins.

Writes reports/ensemble.json and data/processed/oof_models.parquet.
"""
from pathlib import Path
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from joblib import Parallel, delayed
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from diagnostics import as_numeric
from features import CATEGORICAL, FEATURES, prepare
from train import PARAMS, SEED, add_comps, metrics

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["LightGBM", "CatBoost", "XGBoost", "Random forest"]


def as_catboost(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype(str)
    return X


def fit_predict(name, tr_X, tr_y, va_X):
    if name == "LightGBM":
        m = lgb.LGBMRegressor(**dict(PARAMS, objective="regression"))
        return m.fit(tr_X, tr_y).predict(va_X)
    if name == "CatBoost":
        m = CatBoostRegressor(iterations=2000, learning_rate=0.05, depth=8, l2_leaf_reg=3, random_seed=SEED,
                              verbose=0, thread_count=2, cat_features=CATEGORICAL)
        return m.fit(as_catboost(tr_X), tr_y).predict(as_catboost(va_X))
    if name == "XGBoost":
        m = XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=8, min_child_weight=3, subsample=0.8,
                         colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", enable_categorical=True,
                         max_cat_to_onehot=1, n_jobs=2, random_state=SEED)
        return m.fit(tr_X, tr_y).predict(va_X)
    m = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.5, n_jobs=2, random_state=SEED)
    return m.fit(as_numeric(tr_X), tr_y).predict(as_numeric(va_X))


def run_fold(df, y, name, tr_idx, va_idx):
    tr, va = add_comps(df.iloc[tr_idx], df.iloc[va_idx])
    t = time.time()
    return name, va_idx, fit_predict(name, tr[FEATURES], y[tr_idx], va[FEATURES]), time.time() - t


# ---------------------------------------------------------------- weighting methods
def _convex(P, y, loss):
    m = P.shape[1]
    res = minimize(lambda w: loss(P @ w - y), np.full(m, 1 / m), method="SLSQP", bounds=[(0, 1)] * m,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return res.x


WEIGHTERS = {
    "Equal weights (simple average)": lambda P, y: np.full(P.shape[1], 1 / P.shape[1]),
    "Inverse-error weights (1 / MSE)": lambda P, y: (lambda inv: inv / inv.sum())(1 / ((P - y[:, None]) ** 2).mean(0)),
    "Optimized weights, squared error": lambda P, y: _convex(P, y, lambda r: np.mean(r ** 2)),
    "Optimized weights, absolute error": lambda P, y: _convex(P, y, lambda r: np.mean(np.abs(r))),
}


def nested_cv(P, y, groups, weigher=None, stack=False):
    """Learn the combination on 4/5 of the OOF rows, score it on the other 1/5, rotate."""
    out = np.zeros(len(y))
    for tr, va in GroupKFold(n_splits=5, shuffle=True, random_state=7).split(P, y, groups):
        if stack:
            out[va] = Ridge(alpha=1.0).fit(P[tr], y[tr]).predict(P[va])
        else:
            out[va] = P[va] @ weigher(P[tr], y[tr])
    return out


def mdape(y, pred):
    return np.median(np.abs(np.exp(pred - y) - 1)) * 100


def main():
    t0 = time.time()
    raw = pd.read_parquet(ROOT / "data/processed/listings.parquet")
    categories = {c: sorted(raw[c].dropna().unique()) for c in CATEGORICAL}
    df = prepare(raw, categories)
    y = np.log(df.price.to_numpy())
    groups = (df.latitude.round(4).astype(str) + "," + df.longitude.round(4).astype(str)).to_numpy()
    folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(df, y, groups))

    jobs = [delayed(run_fold)(df, y, name, tr, va) for name in MODELS for tr, va in folds]
    P = np.zeros((len(df), len(MODELS)))
    secs = dict.fromkeys(MODELS, 0.0)
    for name, va_idx, pred, s in Parallel(n_jobs=10)(jobs):
        P[va_idx, MODELS.index(name)] = pred
        secs[name] += s
    pd.DataFrame(P, columns=MODELS).to_parquet(ROOT / "data/processed/oof_models.parquet", index=False)
    print(f"base models done in {time.time() - t0:.0f}s", {k: round(v) for k, v in secs.items()})

    results = {"base": [], "ensembles": []}
    for i, name in enumerate(MODELS):
        results["base"].append({"model": name, **metrics(y, P[:, i]), "train_seconds": secs[name]})
        print(f"{name:<45} MdAPE {results['base'][-1]['MdAPE_%']:.3f}")

    lgbm = P[:, 0]
    best_name, best_pred = None, None
    for wname, weigher in WEIGHTERS.items():
        pred = nested_cv(P, y, groups, weigher)
        w_all = weigher(P, y)
        results["ensembles"].append({"method": wname, **metrics(y, pred), "weights": dict(zip(MODELS, map(float, w_all)))})
        print(f"{wname:<45} MdAPE {results['ensembles'][-1]['MdAPE_%']:.3f}  weights {np.round(w_all, 3)}")
    pred = nested_cv(P, y, groups, stack=True)
    ridge = Ridge(alpha=1.0).fit(P, y)
    results["ensembles"].append({"method": "Stacking (ridge regression on predictions)", **metrics(y, pred),
                                 "weights": dict(zip(MODELS, map(float, ridge.coef_))), "intercept": float(ridge.intercept_)})
    print(f"{'Stacking (ridge)':<45} MdAPE {results['ensembles'][-1]['MdAPE_%']:.3f}  coef {np.round(ridge.coef_, 3)}")

    # The candidate we'd ship: best nested-CV MdAPE among the convex methods (stacking can extrapolate).
    convex = [e for e in results["ensembles"] if not e["method"].startswith("Stacking")]
    pick = min(convex, key=lambda e: e["MdAPE_%"])
    best_pred = nested_cv(P, y, groups, WEIGHTERS[pick["method"]])

    # Is the gain real? Paired bootstrap over homes, and fold-by-fold comparison.
    ape_l = np.abs(np.exp(lgbm - y) - 1)
    ape_e = np.abs(np.exp(best_pred - y) - 1)
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(1000):
        idx = rng.integers(0, len(y), len(y))
        diffs.append((np.median(ape_e[idx]) - np.median(ape_l[idx])) * 100)
    per_fold = [{"fold": k + 1, "LightGBM": mdape(y[va], lgbm[va]), "Ensemble": mdape(y[va], best_pred[va])}
                for k, (_, va) in enumerate(folds)]
    results["chosen"] = {
        "method": pick["method"], "weights": pick["weights"],
        "mdape_gain_pts": float(mdape(y, lgbm) - mdape(y, best_pred)),
        "bootstrap_ci_95_pts": [float(-np.percentile(diffs, 97.5)), float(-np.percentile(diffs, 2.5))],
        "folds_won": int(sum(f["Ensemble"] < f["LightGBM"] for f in per_fold)),
        "per_fold": per_fold,
    }
    resid = P - y[:, None]
    results["error_correlation"] = pd.DataFrame(resid, columns=MODELS).corr().round(3).to_dict()
    json.dump(results, open(ROOT / "reports/ensemble.json", "w"), indent=2)
    print(json.dumps(results["chosen"], indent=1))
    print("error correlation:\n", pd.DataFrame(resid, columns=MODELS).corr().round(3))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
