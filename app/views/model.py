import numpy as np
import pandas as pd
import streamlit as st

from common import (BLUE, DRIVER_GROUPS, ROOT, blend_mind_map, hbar, html_block, load_json, load_scored, step_header, table,
                    takeaway)

metrics = load_json("metrics.json")
diag = load_json("diagnostics.json")
ens = load_json("ensemble.json")
scored = load_scored()

step_header(6, 7, "Training the model")

w = metrics["ensemble"]["weights"]
final, base_line, unseen = metrics["stages"][-1], metrics["stages"][0], metrics["unseen_zip"]
st.markdown("### Summary")
st.markdown(f"""
- **Model:** a weighted ensemble of gradient-boosted decision trees: **CatBoost {w['CatBoost']:.2f} · XGBoost {w['XGBoost']:.2f} ·
  LightGBM {w['LightGBM']:.2f}**. Out-of-fold MdAPE **{final['MdAPE_%']:.2f}%**, R² {final['R2_log']:.3f} (linear baseline: {base_line['MdAPE_%']:.1f}%).
- **Target and features:** regression on log(price) with 30 features (14 raw, 16 engineered), {metrics['n_listings']:,} listings.
- **Training:** 5-fold `GroupKFold` cross-validation grouped by coordinates; ensemble weights fit on out-of-fold
  predictions (non-negative, sum to 1, minimizing absolute log error); final models refit on the full dataset.
- **Compute:** CPU only. Apple M2 Pro (10 cores), 16 GB RAM, about 5 minutes end to end. No GPU required.
- **Validation and overfitting control:** fold-local feature engineering (no target leakage), target proxies excluded,
  nested CV for the ensemble weights, regularized trees, and a spatial hold-out on unseen ZIP codes (MdAPE {unseen['MdAPE_%']:.1f}%).
  Details in Step 7.
""")

st.code(f"""y = np.log(df["price"])                                     # learn relative (%) errors
folds = list(GroupKFold(5, shuffle=True).split(X, y, groups=location))   # twins stay together

for name in ["LightGBM", "CatBoost", "XGBoost"]:              # same folds for every model
    for train, test in folds:                                 # comps rebuilt from `train` only
        oof[name][test] = fit(name, X[train], y[train]).predict(X[test])

weights = optimize_weights(oof, y)   # >= 0, sum to 1, minimize |log error|
# CatBoost {w['CatBoost']:.2f}, XGBoost {w['XGBoost']:.2f}, LightGBM {w['LightGBM']:.2f}
final = EnsembleRegressor({{n: fit(n, X, y) for n in MEMBERS}}, weights)   # refit on all homes""", language="python")

# ---------------------------------------------------------------- base models
st.markdown("### Base models")
st.caption("Out-of-fold scores on identical folds and features. Fit time = sum over the 5 folds, 2 CPU threads each.")
base_scores = {b["model"]: b["MdAPE_%"] for b in ens["base"]}
st.code(f"""oof = {{m: cross_val_oof(m, X, y, folds) for m in ["RandomForest", "LightGBM", "XGBoost", "CatBoost"]}}
{{m: mdape(y, p) for m, p in oof.items()}}
# {", ".join(f"{m} {v:.2f}%" for m, v in sorted(base_scores.items(), key=lambda kv: kv[1]))}""", language="python")
lin = next(m for m in diag["models"] if m["model"].startswith("Linear"))
base = {b["model"]: b for b in ens["base"]}
rows = [("Ridge regression (baseline)", lin, None, "Structure features only")]
rows += [(n, base[n], base[n]["train_seconds"], note) for n, note in [
    ("Random forest", "300 trees; weight 0 in the blend"), ("LightGBM", "Fastest booster"),
    ("XGBoost", "Native categorical splits"), ("CatBoost", "Best single model")]]
rows.append((f"Ensemble (Cat {w['CatBoost']:.2f} · XGB {w['XGBoost']:.2f} · LGBM {w['LightGBM']:.2f})", final, None, "Final model"))
table(pd.DataFrame([{"Model": n, "MdAPE": f"{m['MdAPE_%']:.2f}%", "R² (log)": f"{m['R2_log']:.3f}",
                     "Within ±10%": f"{m['within_10%']:.1f}%", "Fit time": f"{t:.0f} s" if t else "–", "Note": note}
                    for n, m, t, note in rows]).set_index("Model"))

TRAITS = {"CatBoost": "Symmetric trees · native categories", "XGBoost": "Depth-8 trees · histogram splits",
          "LightGBM": "Leaf-wise trees · fastest to train", "Random forest": "Independent trees, averaged"}
POS = {"CatBoost": (165, 100), "XGBoost": (835, 100), "LightGBM": (165, 480), "Random forest": (835, 480)}
html_block(blend_mind_map(
    {"mdape": final["MdAPE_%"], "r2": final["R2_log"], "within10": final["within_10%"]},
    [{"name": n, "mdape": base[n]["MdAPE_%"], "r2": base[n]["R2_log"], "weight": w.get(n, 0.0),
      "trait": TRAITS[n], "pos": POS[n]} for n in POS]))
st.caption("Blend at a glance: each card is a base model's own score; branch thickness is its weight.")

# ---------------------------------------------------------------- ensembling
st.markdown("### Ensembling")


@st.cache_data
def gain_vs_catboost():
    """Paired bootstrap (1,000 resamples) of the MdAPE gain of the final blend over CatBoost alone."""
    y = np.log(scored.price.to_numpy())
    cat = pd.read_parquet(ROOT / "data/processed/oof_models.parquet")["CatBoost"].to_numpy()
    ape_c, ape_e = np.abs(np.exp(cat - y) - 1), np.abs(scored.fair_value.to_numpy() / scored.price.to_numpy() - 1)
    rng = np.random.default_rng(42)
    d = [np.median(ape_c[i]) - np.median(ape_e[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(1000))]
    return (np.median(ape_c) - np.median(ape_e)) * 100, np.percentile(d, 2.5) * 100, np.percentile(d, 97.5) * 100


ch = ens["chosen"]
gain_c, lo_c, hi_c = gain_vs_catboost()
corr = pd.DataFrame(ens["error_correlation"])
rf_corr = corr.loc["Random forest", ["LightGBM", "CatBoost", "XGBoost"]]
booster_corr = pd.Series([corr.loc[a, b] for a, b in [("LightGBM", "XGBoost"), ("LightGBM", "CatBoost"), ("CatBoost", "XGBoost")]])
st.markdown(f"""
- **Level-2 inputs:** out-of-fold log-price predictions of the four base models. Residual correlation between boosters is
  {booster_corr.min():.2f}–{booster_corr.max():.2f}, and {rf_corr.min():.2f}–{rf_corr.max():.2f} with the random forest: enough diversity to blend.
- **Selection protocol:** every weighting scheme is scored with **nested 5-fold CV** over the OOF rows, so weights are never
  fit on the rows they are scored on.
- **Chosen:** convex weights minimizing MAE on log price (closest to the MdAPE objective). Random forest received weight 0
  and was removed: final 3-model blend **{final['MdAPE_%']:.2f}%**.
- **Significance (paired bootstrap, 1,000 resamples):** vs LightGBM **−{ch['mdape_gain_pts']:.2f} pts** (95% CI
  {ch['bootstrap_ci_95_pts'][0]:.2f}–{ch['bootstrap_ci_95_pts'][1]:.2f}), better on {ch['folds_won']}/5 folds; vs CatBoost
  **−{gain_c:.2f} pts** (95% CI {lo_c:.2f}–{hi_c:.2f}).
""")
cw = ch["weights"]  # 4-model optimization; dropping the 0-weight forest and refitting gives the final 3-model weights
st.code(f"""P = np.column_stack([oof[m] for m in MODELS])   # homes x models, out-of-fold log price
w = minimize(lambda w: np.abs(P @ w - y).mean(), np.full(4, 0.25), method="SLSQP",
             bounds=[(0, 1)] * 4, constraints={{"type": "eq", "fun": lambda w: w.sum() - 1}}).x
# LightGBM {cw['LightGBM']:.2f}, CatBoost {cw['CatBoost']:.2f}, XGBoost {cw['XGBoost']:.2f}, RandomForest {cw['Random forest']:.2f}
gains = [mdape(y[i], lgbm[i]) - mdape(y[i], blend[i]) for i in resamples]   # 1,000 bootstrap draws
np.percentile(gains, [2.5, 97.5])   # -> [{ch['bootstrap_ci_95_pts'][0]:.2f}, {ch['bootstrap_ci_95_pts'][1]:.2f}] points""", language="python")
em = pd.DataFrame(ens["ensembles"])
table(pd.DataFrame([{"Weighting scheme": e["method"].replace(" (simple average)", "").replace(" (ridge regression on predictions)", " (ridge)"),
                     "Nested-CV MdAPE": f"{e['MdAPE_%']:.3f}%", "R² (log)": f"{e['R2_log']:.4f}",
                     "Weights LGBM / Cat / XGB / RF": " / ".join(f"{e['weights'][m]:+.2f}" if "Stacking" in e["method"]
                                                                 else f"{e['weights'][m]:.2f}" for m in ["LightGBM", "CatBoost", "XGBoost", "Random forest"])}
                    for _, e in em.iterrows()]).set_index("Weighting scheme"))

# ---------------------------------------------------------------- ablation
st.markdown("### Feature ablation")
st.caption("Single LightGBM model, features added cumulatively. Spatial hold-out = whole ZIP codes held out (GroupKFold by ZIP).")
st.code("""for stage, cols in STAGES:                                    # features added cumulatively
    oof = cross_val_oof("LightGBM", X[cols], y, folds)          # location-grouped CV
    oof_zip = cross_val_oof("LightGBM", X[cols], y, zip_folds)  # spatial hold-out: GroupKFold by ZIP""",
        language="python")
s = pd.DataFrame(metrics["stages"])
uz = {u["stage"]: u["MdAPE_%"] for u in diag["unseen_zip_stages"]}
uz[s.stage.iloc[-1]] = unseen["MdAPE_%"]
abl = pd.DataFrame({"Stage": s.stage, "MdAPE": s["MdAPE_%"].map("{:.2f}%".format),
                    "Δ vs previous": s["MdAPE_%"].diff().map(lambda v: "" if pd.isna(v) else f"{v:+.2f}"),
                    "R² (log)": s["R2_log"].map("{:.3f}".format),
                    "Spatial hold-out MdAPE": s.stage.map(lambda t: f"{uz[t]:.2f}%" if t in uz else "–")})
table(abl.set_index("Stage"))
st.markdown(f"Stage 1 is the model switch (ridge → LightGBM). Among feature groups, location is the largest gain "
            f"({s['MdAPE_%'][2] - s['MdAPE_%'][1]:+.2f} pts). ZIP-level IRS/Zillow features are "
            f"neutral in random CV but improve the spatial hold-out ({uz[s.stage[4]]:.2f}% → {uz[s.stage[5]]:.2f}%), so they are kept.")

# ---------------------------------------------------------------- hyperparameters
st.markdown("### Hyperparameters")
table(pd.DataFrame([
    ("LightGBM", "800 trees · lr 0.05 · 63 leaves · min_child_samples 20 · subsample/colsample 0.8 · L2 1.0"),
    ("CatBoost", "2,000 iterations · lr 0.05 · depth 8 · l2_leaf_reg 3 · native categorical (ZIP, home type)"),
    ("XGBoost", "1,200 trees · lr 0.03 · max_depth 8 · min_child_weight 3 · subsample/colsample 0.8 · L2 1.0 · hist, native categorical"),
    ("Price ranges", "Quantile LightGBM (α = 0.10 / 0.90) + conformal calibration per price decile and tail"),
], columns=["Model", "Settings"]).set_index("Model"))
st.code("""LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=63, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0)
CatBoostRegressor(iterations=2000, learning_rate=0.05, depth=8, l2_leaf_reg=3, cat_features=["homeType", "zipcode"])
XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=8, min_child_weight=3, subsample=0.8,
             colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", enable_categorical=True)""", language="python")
st.caption("No hyperparameter search: conservative settings, with the time budget spent on validation instead.")

# ---------------------------------------------------------------- negative results
st.markdown("### What didn't work")
lgbm_rf = next(m for m in diag["models"] if "random forest" in m["model"].lower() and "Ensemble" in m["model"])
lgbm = next(m for m in diag["models"] if m["model"].startswith("LightGBM"))
stack = next(e for e in ens["ensembles"] if e["method"].startswith("Stacking"))
sq = next(e for e in ens["ensembles"] if "squared" in e["method"])
leak = diag["leakage_demo"]["with_zestimate_leak"]
st.markdown(f"""
- **Random forest:** weight 0 in every optimized blend; averaging it 50/50 with LightGBM lowered R² ({lgbm_rf['R2_log']:.3f} vs {lgbm['R2_log']:.3f}).
- **Ridge stacking:** no gain over convex weights ({stack['MdAPE_%']:.3f}%) and assigns a negative weight to the random forest ({stack['weights']['Random forest']:+.2f}).
- **Weights optimized for squared error:** worse on MdAPE ({sq['MdAPE_%']:.3f}%) than the absolute-error objective.
- **Zestimate as a feature:** MdAPE {leak['MdAPE_%']:.1f}%, R² {leak['R2_log']:.3f}. That is target leakage, not skill, so it (and the tax value) are benchmarks only.
""")

st.code(f"""Ridge(alpha=1.0).fit(P, y).coef_                    # stacking -> RF weight {stack['weights']['Random forest']:+.2f}
cross_val_oof("LightGBM", X.assign(zestimate=df["zestimate"]), y, folds)
# -> MdAPE {leak['MdAPE_%']:.1f}%, R² {leak['R2_log']:.3f}: the model copies Zillow (leakage)""", language="python")

# ---------------------------------------------------------------- importance
st.markdown("### Feature importance (mean |SHAP|, grouped)")
st.code("""base, contrib = final.contributions(X)   # exact TreeSHAP, weighted across the 3 models
share = {g: np.abs(contrib[:, cols].sum(axis=1)).mean() for g, cols in DRIVER_GROUPS.items()}""", language="python")
groups = {g: sum(scored[f"shap__{f}"] for f in feats if f"shap__{f}" in scored) for g, feats in DRIVER_GROUPS.items()}
imp = pd.Series({g: float(np.mean(np.abs(v))) for g, v in groups.items()}).sort_values(ascending=False)
pct = imp / imp.sum()
st.plotly_chart(hbar(list(pct.index), list(pct.values), ".0%", color=BLUE, height=320,
                     hover="%{y}<br>%{x:.0%} of total |SHAP|<extra></extra>").update_xaxes(tickformat=".0%"), width="stretch")
st.caption("Ensemble SHAP = weighted sum of each booster's exact TreeSHAP values, so the decomposition stays additive.")

takeaway(f"A 3-model gradient-boosting blend reaches **{final['MdAPE_%']:.2f}% MdAPE** out-of-fold "
         f"({base_line['MdAPE_%']:.1f}% baseline), with gains verified by nested CV and a paired bootstrap.")
st.page_link("views/validation.py", label="Next: validation →")
