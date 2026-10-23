import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import GroupKFold

from common import (BLUE, GRAY, INK_2, MUTED, RED, ZILLOW_OFF_MARKET_MDAPE, ZILLOW_ON_MARKET_MDAPE, hbar, load_json,
                    load_scored, md_money, money, plot_range, step_header, style, table, takeaway)

ORANGE = "#eb6834"  # categorical slot 2, used only for the 'training' series
metrics = load_json("metrics.json")
diag = load_json("diagnostics.json")
ens = load_json("ensemble.json")
scored = load_scored()
best, u, iv = metrics["stages"][-1], metrics["unseen_zip"], metrics["interval"]
leak = diag["leakage_demo"]["with_zestimate_leak"]
gap = pd.DataFrame(diag["fold_gap"])
curve = {k: np.array(v) for k, v in diag["learning_curve"].items()}
bands = pd.DataFrame(iv["bands_holdout"])


@st.cache_data
def fold_mdape():
    """MdAPE of the final out-of-fold predictions within each of the 5 location-grouped folds."""
    groups = scored.latitude.round(4).astype(str) + "," + scored.longitude.round(4).astype(str)
    ape = (scored.fair_value / scored.price - 1).abs().to_numpy()
    splits = GroupKFold(n_splits=5, shuffle=True, random_state=42).split(scored, groups=groups)
    return pd.Series([np.median(ape[va]) * 100 for _, va in splits])


fold_scores = fold_mdape()
ch = ens["chosen"]

step_header(7, 7, "Validation")

st.markdown("### Summary")
st.markdown(f"""
- **Protocol:** 5-fold `GroupKFold`, grouped by exact coordinates so co-located listings never straddle folds. Every
  reported metric is out-of-fold (n = {metrics['n_listings']:,}).
- **Out-of-fold performance:** MdAPE **{best['MdAPE_%']:.2f}%**, within ±10% {best['within_10%']:.1f}%, median absolute error
  {md_money(best['MdAE_$'])}, R² {best['R2_log']:.3f}. Fold-to-fold MdAPE {fold_scores.min():.2f}–{fold_scores.max():.2f}%.
- **Leakage control:** fold-local comps (training rows only, self excluded); target proxies and seller-behavior fields
  excluded; ensemble weights selected with nested CV.
- **Overfitting diagnostics:** validation loss keeps falling (minimum at tree {int(np.argmin(curve['validation'])) + 1} of
  {len(curve['validation'])}), with no divergence; the train/validation gap is stable across folds.
- **Generalization:** spatial hold-out (`GroupKFold` by ZIP code), MdAPE **{u['MdAPE_%']:.1f}%**, R² {u['R2_log']:.3f}.
- **Uncertainty:** 80% conformalized quantile intervals with **{iv['holdout_coverage_%']:.1f}%** held-out coverage, calibrated per price decile.
- **Benchmark:** Zillow's off-market Zestimate, {ZILLOW_OFF_MARKET_MDAPE}% MdAPE (national, sale prices).
""")
st.code("""groups = df["latitude"].round(4).astype(str) + "," + df["longitude"].round(4).astype(str)
# main CV: grouped by location
folds = list(GroupKFold(5, shuffle=True, random_state=42).split(X, y, groups))
# spatial hold-out: whole ZIP codes unseen in training
zip_folds = list(GroupKFold(5, shuffle=True, random_state=42).split(X, y, df["zipcode"]))""",
        language="python")

# ---------------------------------------------------------------- error distribution
st.markdown("### Out-of-fold error distribution")
err = (scored.fair_value / scored.price - 1).clip(-0.6, 0.6) * 100
counts, edges = np.histogram(err, bins=np.arange(-60, 62, 2.5))
mids = (edges[:-1] + edges[1:]) / 2
fig = go.Figure(go.Bar(x=mids, y=counts, marker_color=[BLUE if abs(m) <= 10 else MUTED for m in mids], width=2.2,
                       hovertemplate="%{x:+.1f}%: %{y:,} homes<extra></extra>"))
fig.add_annotation(x=0, y=counts.max(), yshift=18, showarrow=False, font=dict(color=BLUE, size=14),
                   text=f"{best['within_10%']:.1f}% within ±10%")
style(fig, 280, margin=dict(l=10, r=40, t=40, b=10))
fig.update_xaxes(title="Signed error, prediction / price − 1 (clipped at ±60%)", ticksuffix="%", dtick=10)
fig.update_yaxes(title="Homes")
st.plotly_chart(fig, width="stretch")
st.caption(f"Median signed error {np.median(scored.fair_value / scored.price - 1):+.1%}: no systematic over- or under-prediction.")

# ---------------------------------------------------------------- overfitting
st.markdown("### Overfitting diagnostics")
st.code("""model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_va, y_va)], eval_metric="l1",
          callbacks=[lgb.record_evaluation(curve)])   # train vs. validation loss per tree""", language="python")
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    for col, name, color in [("train_MdAPE_%", "Train", ORANGE), ("validation_MdAPE_%", "Validation", BLUE)]:
        fig.add_trace(go.Bar(x=[f"Fold {i + 1}" for i in range(len(gap))], y=gap[col], name=name, marker_color=color,
                             text=[f"{v:.1f}%" for v in gap[col]], textposition="outside", cliponaxis=False,
                             hovertemplate=name + " · %{x}: %{y:.2f}%<extra></extra>"))
    style(fig, 300, showlegend=True, barmode="group", bargap=0.3, bargroupgap=0.08,
          legend=dict(orientation="h", y=1.12, x=0), margin=dict(l=10, r=40, t=70, b=10),
          title=dict(text="MdAPE by fold (LightGBM member)", font=dict(size=14), y=0.97))
    fig.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig, width="stretch")
with c2:
    trees = np.arange(1, len(curve["train"]) + 1)
    fig = go.Figure()
    for key, name, color in [("train", "Train", ORANGE), ("validation", "Validation", BLUE)]:
        y = curve[key] * 100
        fig.add_trace(go.Scatter(x=trees, y=y, name=name, mode="lines", line=dict(color=color, width=2),
                                 hovertemplate=name + " · %{x} trees: %{y:.1f}%<extra></extra>"))
    style(fig, 300, showlegend=True, legend=dict(orientation="h", y=1.12, x=0), hovermode="x unified",
          margin=dict(l=10, r=40, t=70, b=10), title=dict(text="Learning curve, fold 1 (mean |log error|)", font=dict(size=14), y=0.97))
    fig.update_yaxes(ticksuffix="%")
    fig.update_xaxes(title="Trees")
    st.plotly_chart(fig, width="stretch")
best_tree = int(np.argmin(curve["validation"])) + 1
st.markdown(f"Validation loss keeps falling, apart from tiny tree-to-tree noise, to its minimum at tree {best_tree} of {len(trees)}: "
            f"no divergence point is reached. LightGBM validation "
            f"MdAPE ranges {gap['validation_MdAPE_%'].min():.2f}–{gap['validation_MdAPE_%'].max():.2f}% across folds, and the "
            f"train/validation gap ({gap['train_MdAPE_%'].mean():.1f}% vs {gap['validation_MdAPE_%'].mean():.1f}%) is consistent: "
            "expected variance for boosted trees, not overfitting.")

# ---------------------------------------------------------------- leakage
st.markdown("### Leakage controls")
st.code("""for tr, va in folds:
    comps = CompsIndex(df.iloc[tr])                      # neighbors from training rows only
    X_tr = comps.query(df.iloc[tr], exclude_self=True)   # a home is never its own neighbor
    X_va = comps.query(df.iloc[va])""", language="python")
table(pd.DataFrame([
    ("Co-located listings in train and test", "`GroupKFold` on exact coordinates", "–"),
    ("Comps feature built from the target", "Rebuilt per fold from training rows; self excluded", "–"),
    ("Target proxies (Zestimate, rent Zestimate, tax value)", "Excluded from features",
     f"With Zestimate: MdAPE {leak['MdAPE_%']:.1f}%, R² {leak['R2_log']:.3f} (leakage)"),
    ("Seller behavior (days on market, price cuts)", "Excluded from features", "–"),
    ("Ensemble weights tuned on scored rows", "Nested 5-fold CV over OOF predictions",
     f"Gain vs LightGBM {ch['mdape_gain_pts']:.2f} pts, 95% CI {ch['bootstrap_ci_95_pts'][0]:.2f}–{ch['bootstrap_ci_95_pts'][1]:.2f}"),
], columns=["Risk", "Control", "Check"]).set_index("Risk"))

# ---------------------------------------------------------------- intervals
st.markdown("### Prediction intervals")
st.code(f"""lo, hi = oof_quantile(alpha=0.10), oof_quantile(alpha=0.90)        # quantile LightGBM, out-of-fold
for band in fair_value_deciles:                                     # conformal adjustment per band and tail
    lo[band] -= np.quantile(lo[band] - y[band], 0.90)
    hi[band] += np.quantile(y[band] - hi[band], 0.90)
# held-out coverage {iv['holdout_coverage_%']:.1f}%, median width ±{iv['median_width_%'] / 2:.0f}%""", language="python")
labels = ["under $" + f"{bands.fair_value_from[1] / 1e3:.0f}k"] + [
    f"${a / 1e3:.0f}k+" if a < 1e6 else f"${a / 1e6:.1f}M+" for a in bands.fair_value_from[1:]]
fig = go.Figure()
for col, name, color in [("below_%", "Below interval", BLUE), ("above_%", "Above interval", RED)]:
    fig.add_trace(go.Bar(x=labels, y=bands[col], name=name, marker_color=color,
                         hovertemplate="Fair value %{x}<br>" + name + ": %{y:.1f}%<extra></extra>"))
fig.add_hline(y=10, line=dict(color=INK_2, width=1, dash="dot"), annotation_text="10% target", annotation_position="top right")
style(fig, 280, showlegend=True, barmode="group", bargap=0.25, bargroupgap=0.08, legend=dict(orientation="h", y=1.18, x=0),
      margin=dict(l=10, r=40, t=50, b=10))
fig.update_yaxes(ticksuffix="%", range=[0, 16], title="Held-out share")
fig.update_xaxes(title="Fair-value decile")
st.plotly_chart(fig, width="stretch")
st.caption(f"Each tail is {min(bands['below_%'].min(), bands['above_%'].min()):.1f}–"
           f"{max(bands['below_%'].max(), bands['above_%'].max()):.1f}% per decile (target 10%). A single global "
           "adjustment under-covered cheap homes; per-decile calibration fixed it.")

# ---------------------------------------------------------------- benchmark
st.markdown("### External benchmark")
bench = metrics["benchmark_zestimate"]
rows = [("Zillow, on-market (sees asking price)", ZILLOW_ON_MARKET_MDAPE, GRAY),
        ("Zillow, off-market", ZILLOW_OFF_MARKET_MDAPE, GRAY),
        ("XMarket, out-of-fold", best["MdAPE_%"], BLUE),
        ("XMarket, unseen ZIP codes", u["MdAPE_%"], BLUE),
        ("Ridge baseline", metrics["stages"][0]["MdAPE_%"], MUTED)]
st.plotly_chart(hbar([r[0] for r in rows], [r[1] for r in rows], ".2f", colors=[r[2] for r in rows], height=250,
                     hover="%{y}<br>MdAPE %{x:.2f}%<extra></extra>").update_xaxes(ticksuffix="%", title="MdAPE"),
                width="stretch")
st.caption(f"Off-market is the like-for-like comparison: on our listings the Zestimate sits within "
           f"{bench['zestimate']['MdAPE_%']:.1f}% of the asking price, i.e. it is anchored on it. Caveat: Zillow's figures are national "
           "and measured against sale prices; ours are Houston list prices. Source: zillow.com/zestimate.")

# ---------------------------------------------------------------- spot check
st.markdown("### Spot check")
if st.button("Sample a random out-of-fold prediction"):
    st.session_state["test_idx"] = int(np.random.default_rng().choice(scored.index))
h = scored.loc[st.session_state.get("test_idx", int(scored.index[123]))]
c1, c2 = st.columns([1, 1.6])
with c1:
    if isinstance(h.imgSrc, str):
        st.image(h.imgSrc, width="stretch")
with c2:
    st.markdown(f"**{h.address}**  \n{h.bedrooms:.0f} bd · {h.bathrooms:g} ba · {h.livingArea:,.0f} sqft · built {h.yearBuilt:.0f}")
    k = st.columns(3)
    k[0].metric("Asking price", money(h.price))
    k[1].metric("Out-of-fold estimate", money(h.fair_value), delta=f"{(h.fair_value / h.price - 1) * 100:+.1f}%", delta_color="off")
    k[2].metric("Zestimate", money(h.zestimate) if pd.notna(h.zestimate) else "n/a")
    st.plotly_chart(plot_range(h.price, h.fair_lo, h.fair_value, h.fair_hi), width="stretch")

takeaway(f"Out-of-fold MdAPE {best['MdAPE_%']:.2f}% with leakage controlled at every step, {u['MdAPE_%']:.1f}% on unseen ZIP codes, "
         f"and calibrated 80% intervals ({iv['holdout_coverage_%']:.1f}% coverage).")
st.page_link("views/price_home.py", label="Next: application →")
