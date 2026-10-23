"""Weighted ensemble of LightGBM, CatBoost and XGBoost on log price, with exact SHAP explanations.

Each library computes exact per-feature contributions (TreeSHAP) natively. Because the ensemble is a
weighted average in log space, its contributions are the same weighted average, so the dollar
waterfall in the app still adds up exactly.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from scipy.optimize import minimize
from xgboost import DMatrix, XGBRegressor

from features import CATEGORICAL

SEED = 42
MEMBERS = ["LightGBM", "CatBoost", "XGBoost"]
LGBM_PARAMS = dict(n_estimators=800, learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=2, verbose=-1)


def _as_catboost(X):
    X = X.copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype(str)
    return X


def new_member(name, n_jobs=2):
    if name == "LightGBM":
        return lgb.LGBMRegressor(**dict(LGBM_PARAMS, n_jobs=n_jobs))
    if name == "CatBoost":
        return CatBoostRegressor(iterations=2000, learning_rate=0.05, depth=8, l2_leaf_reg=3, random_seed=SEED,
                                 verbose=0, thread_count=n_jobs, cat_features=CATEGORICAL)
    return XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=8, min_child_weight=3, subsample=0.8,
                        colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", enable_categorical=True,
                        max_cat_to_onehot=1, n_jobs=n_jobs, random_state=SEED)


def fit_member(name, X, y, n_jobs=2):
    model = new_member(name, n_jobs)
    return model.fit(_as_catboost(X) if name == "CatBoost" else X, y)


def member_predict(name, model, X):
    return model.predict(_as_catboost(X) if name == "CatBoost" else X)


def member_contrib(name, model, X):
    """Exact TreeSHAP contributions: n x (features + 1), last column = base value."""
    if name == "LightGBM":
        return model.predict(X, pred_contrib=True)
    if name == "CatBoost":
        return model.get_feature_importance(Pool(_as_catboost(X), cat_features=CATEGORICAL), type="ShapValues")
    return model.get_booster().predict(DMatrix(X, enable_categorical=True), pred_contribs=True)


def optimize_weights(P, y):
    """Non-negative weights summing to 1 that minimize mean absolute log error (~ the median % error we report)."""
    m = P.shape[1]
    res = minimize(lambda w: np.mean(np.abs(P @ w - y)), np.full(m, 1 / m), method="SLSQP",
                   bounds=[(0, 1)] * m, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return res.x


class EnsembleRegressor:
    def __init__(self, members: dict, weights: np.ndarray):
        self.members, self.weights = members, np.asarray(weights)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return sum(w * member_predict(n, m, X) for (n, m), w in zip(self.members.items(), self.weights))

    def contributions(self, X: pd.DataFrame):
        """(base value per row, SHAP matrix) for the weighted ensemble, in log-price units."""
        total = sum(w * member_contrib(n, m, X) for (n, m), w in zip(self.members.items(), self.weights))
        return total[:, -1], total[:, :-1]
