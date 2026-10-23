import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (BLUE, DRIVER_GROUPS, GRAY, blend_mind_map, hbar, html_block, insight, load_json, load_scored,
                    step_header, style, table, takeaway)

metrics = load_json("metrics.json")
diag = load_json("diagnostics.json")
scored = load_scored()

step_header(5, 6, "Training the model",
            "We started with the simplest possible model as a baseline, then tried stronger ones and kept the one "
            "that was most accurate on homes it had never seen.")

c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.markdown("**🧱 Building a model** = choosing *how* it works.")
        st.caption("Like choosing a recipe. We compared a straight-line formula (linear regression), a forest of independent "
                   "decision trees (random forest), and a chain of trees that learn from each other's mistakes (gradient boosting).")
with c2:
    with st.container(border=True):
        st.markdown("**🎓 Training a model** = teaching it with examples.")
        st.caption("We show it ~15,500 homes *with* their prices. It adjusts itself until its guesses match. Then we test it "
                   "on ~3,900 homes it has never seen. We repeat this 5 times so every home gets tested once.")

ens = load_json("ensemble.json")
lin = next(m for m in diag["models"] if m["model"].startswith("Linear"))
st.markdown("### Which model? We tried four, then combined the best three")
rows = [("Linear regression (baseline)", lin["MdAPE_%"], GRAY)] + \
       [(b["model"], b["MdAPE_%"], GRAY) for b in sorted(ens["base"], key=lambda b: -b["MdAPE_%"])] + \
       [("Ensemble: CatBoost + XGBoost + LightGBM", metrics["stages"][-1]["MdAPE_%"], BLUE)]
st.plotly_chart(hbar([r[0] for r in rows], [r[1] for r in rows], ".2f", colors=[r[2] for r in rows], height=300,
                     hover="%{y}<br>typical error %{x:.2f}%<extra></extra>")
                .update_xaxes(ticksuffix="%", title="Typical (median) % error on unseen homes, lower is better"), width="stretch")
st.caption("All models use the same features and the same 5 cross-validation folds. CatBoost and XGBoost are two other "
           "gradient-boosting libraries; each builds its trees differently, so each makes slightly different mistakes.")

st.markdown("### The blend at a glance")
base = {b["model"]: b for b in ens["base"]}
weights = metrics["ensemble"]["weights"]
TRAITS = {"CatBoost": "Symmetric trees · reads ZIP & home type natively",
          "XGBoost": "Depth-limited trees (8 levels) · histogram splits",
          "LightGBM": "Leaf-wise trees (63 leaves) · fastest to train",
          "Random forest": "300 independent trees, averaged · added nothing"}
POS = {"CatBoost": (175, 105), "XGBoost": (825, 105), "LightGBM": (175, 475), "Random forest": (825, 475)}
final = metrics["stages"][-1]
html_block(blend_mind_map(
    {"mdape": final["MdAPE_%"], "r2": final["R2_log"], "within10": final["within_10%"]},
    [{"name": n, "mdape": base[n]["MdAPE_%"], "r2": base[n]["R2_log"], "weight": weights.get(n, 0.0),
      "trait": TRAITS[n], "pos": POS[n]} for n in POS]))
st.caption("Each card: the model's error on its own. Branch thickness = its weight in the blend. All scores are "
           "out-of-fold on the same 5 folds; the blend's weights are scored with nested cross-validation.")

st.markdown("### Can combining models beat the best single model?")
c1, c2 = st.columns([1.15, 1], gap="large")
with c1:
    st.markdown("""
An **ensemble** averages several models' predictions. It only helps if the models make *different* mistakes, and the
question is how much to trust each one. We tried five ways to choose the weights:

- **Equal weights:** a plain average.
- **Inverse-error weights:** more accurate models get more say.
- **Optimized weights (two versions):** search for the weights (each ≥ 0, summing to 1) that minimize squared or absolute error.
- **Stacking:** a small regression model learns how to combine the predictions.

**The honest part:** weights are learned from data, so they could overfit too. We scored every weighting method with a
*second* cross-validation: learn the weights on 4/5 of the homes, test them on the other 1/5, and rotate.
""")
with c2:
    em = pd.DataFrame(ens["ensembles"]).rename(columns={"method": "Weighting method"})
    em["Weighting method"] = em["Weighting method"].str.replace(" (simple average)", "").str.replace("Stacking (ridge regression on predictions)", "Stacking (ridge)")
    st.plotly_chart(hbar(list(em["Weighting method"]), list(em["MdAPE_%"]), ".2f", height=260,
                         colors=[BLUE if m.startswith("Optimized weights, abs") else GRAY for m in em["Weighting method"]],
                         hover="%{y}<br>%{x:.3f}% (nested CV)<extra></extra>")
                    .update_xaxes(range=[7.0, 7.2], ticksuffix="%", title="Median % error (nested CV)"), width="stretch")
    st.caption("Experiment with all four models. Dropping the random forest (weight 0) gives the final ensemble's "
               f"{metrics['stages'][-1]['MdAPE_%']:.2f}%.")
    w = metrics["ensemble"]["weights"]
    st.plotly_chart(hbar(list(w), list(w.values()), ".0%", color=BLUE, height=170,
                         hover="%{y}: weight %{x:.1%}<extra></extra>").update_xaxes(range=[0, 0.8], tickformat=".0%",
                                                                                    title="Final weights"), width="stretch")
ch = ens["chosen"]
lgbm_m = next(b for b in ens["base"] if b["model"] == "LightGBM")["MdAPE_%"]
insight(f"Every combination beat every single model, but the *way* of weighting barely mattered: a plain average scores "
        f"{ens['ensembles'][0]['MdAPE_%']:.2f}% and the optimized weights {metrics['stages'][-1]['MdAPE_%']:.2f}%. We kept the optimized "
        f"weights (CatBoost {w['CatBoost']:.0%}, XGBoost {w['XGBoost']:.0%}, LightGBM {w['LightGBM']:.0%}) because they minimize "
        f"*absolute* error, the cousin of our median % metric. The random forest earned weight 0 and was dropped.")
st.markdown(f"""
**Is the gain real or luck?** Versus LightGBM alone ({lgbm_m:.2f}%) the ensemble improves the typical error by
**{ch['mdape_gain_pts']:.2f} points**, and a bootstrap (re-sampling the homes 1,000 times) puts the gain between
**{ch['bootstrap_ci_95_pts'][0]:.2f} and {ch['bootstrap_ci_95_pts'][1]:.2f} points**. It won in **{ch['folds_won']} of 5** folds.
The gain is small but consistent, and it costs nothing in explainability: each model's SHAP values are exact, so the
ensemble's explanation is just their weighted sum.
""")

st.markdown("### How gradient boosting (all three models) works, in plain English")
c1, c2 = st.columns([1.3, 1])
with c1:
    st.markdown("""
1. **Tree #1** makes a rough guess using simple yes/no questions: *"Over 2,000 sqft? Within 10 miles of downtown?"*
2. **Tree #2** looks only at what tree #1 got wrong and learns to fix those mistakes.
3. **Repeat 800 times.** Each tree is small and weak; together they're very accurate.
4. The final price = the sum of all 800 small corrections.

**Why it fits this problem:** location effects are curved (the U-shape in Step 2), features interact (a pool matters more
in a big house), and it handles missing values natively, so no fake fill-ins.
""")
with c2:
    st.markdown("**LightGBM settings (hyperparameters)**")
    table(pd.DataFrame([
        ("Trees", "800", "Enough to keep improving; validation error still falling at 800"),
        ("Learning rate", "0.05", "Each tree corrects only 5% of the remaining error: slow & steady"),
        ("Leaves per tree", "63", "How many questions each tree can ask"),
        ("Min homes per leaf", "20", "No rule may be based on fewer than 20 homes"),
        ("Row / column sampling", "80% / 80%", "Each tree sees a random subset, which reduces memorizing"),
        ("L2 penalty", "1.0", "Discourages extreme corrections"),
    ], columns=["Setting", "Value", "Why"]).set_index("Setting"))
    st.caption("Standard, conservative values. We deliberately did **not** spend hours tuning: correctness over decimal points.")

st.markdown("### What did each idea add? (before vs. after)")
s = pd.DataFrame(metrics["stages"])
uz = diag["unseen_zip_stages"]  # stages 2-6 re-scored with whole ZIP codes held out
st.plotly_chart(hbar(list(s.stage), list(s["MdAPE_%"]), ".1f", height=370, colors=[GRAY] + [BLUE] * (len(s) - 1),
                     hover="%{y}<br>typical error %{x:.1f}%<extra></extra>").update_xaxes(ticksuffix="%", title="Typical (median) % error"),
                width="stretch")
insight(f"**Location is the biggest single win**: adding coordinates and ZIP cut the error from {s['MdAPE_%'][1]:.1f}% to {s['MdAPE_%'][2]:.1f}%. "
        f"Commute distance and neighborhood comps shaved off more, to {s['MdAPE_%'][4]:.1f}%. Honest note: ZIP income & market trends added "
        "almost nothing *here*, because the nearby-comps feature already captured it. But on ZIP codes the model has never seen, they cut "
        f"the error from {uz[2]['MdAPE_%']:.1f}% to {uz[3]['MdAPE_%']:.1f}%, so we kept them.")

st.markdown("### What the model pays attention to")
groups = {g: sum(scored[f"shap__{f}"] for f in feats if f"shap__{f}" in scored) for g, feats in DRIVER_GROUPS.items()}
imp = pd.Series({g: float(np.mean(np.abs(v))) for g, v in groups.items()}).sort_values(ascending=False)
pct = imp / imp.sum()
st.plotly_chart(hbar(list(pct.index), list(pct.values), ".0%", height=340,
                     hover="%{y}<br>%{x:.0%} of the model's total attention<extra></extra>").update_xaxes(tickformat=".0%"),
                width="stretch")
st.caption("Measured with SHAP values: for every home, SHAP splits the predicted price into the contribution of each feature. "
           "Shown: average size of each group's contribution, as a share of the total.")

takeaway(f"Our final model is a **weighted ensemble of three gradient-boosting models**: typical error "
         f"**{metrics['stages'][-1]['MdAPE_%']:.1f}%**, down from {metrics['stages'][0]['MdAPE_%']:.1f}% for the baseline, and it still "
         "explains every prediction feature by feature.")
st.page_link("views/validation.py", label="Next: can we trust it? →", icon="✅")
