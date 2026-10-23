import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (BLUE, GRAY, INK_2, MUTED, RED, ZILLOW_OFF_MARKET_MDAPE, ZILLOW_ON_MARKET_MDAPE, hbar, insight,
                    load_json, load_scored, md_money, money, plot_range, step_header, style, table, takeaway)

ORANGE = "#eb6834"  # categorical slot 2, used only for the 'training' series
metrics = load_json("metrics.json")
diag = load_json("diagnostics.json")
scored = load_scored()
best = metrics["stages"][-1]
leak = diag["leakage_demo"]["with_zestimate_leak"]

step_header(6, 6, "Can we trust it?",
            "We only ever grade the model on homes it has never seen, we block every way it could 'peek' at the answer, "
            "and then we compare it to Zillow.")

st.markdown("### Which metric, and why?")
table(pd.DataFrame([
    ("Median % error (MdAPE)", f"{best['MdAPE_%']:.1f}%", "**Main metric.** 'Half of our estimates are within X% of the price.' "
     "Easy to explain, not distorted by a few mansions, and the same metric Zillow publishes, so we can compare."),
    ("Within 10%", f"{best['within_10%']:.0f}%", "Share of homes where we're within 10%: a practical 'good enough to negotiate with' bar."),
    ("Median $ error", money(best["MdAE_$"]), "The same error in dollars for a typical home."),
    ("R² on log price", f"{best['R2_log']:.3f}", "Share of price variation explained (1.0 = perfect). Standard for regression."),
], columns=["Metric", "Our score", "Why we use it"]).set_index("Metric"))

err = (scored.fair_value / scored.price - 1).clip(-0.6, 0.6) * 100
counts, edges = np.histogram(err, bins=np.arange(-60, 62, 2.5))
mids = (edges[:-1] + edges[1:]) / 2
fig = go.Figure(go.Bar(x=mids, y=counts, marker_color=[BLUE if abs(m) <= 10 else MUTED for m in mids], width=2.2,
                       hovertemplate="%{x:+.1f}% off: %{y:,} homes<extra></extra>"))
fig.add_annotation(x=0, y=counts.max(), yshift=18, showarrow=False, font=dict(color=BLUE, size=14),
                   text=f"{best['within_10%']:.0f}% of homes within ±10%")
style(fig, 300, title=dict(text="How far off is each estimate? (all 19,430 homes, out-of-fold)", font=dict(size=15)))
fig.update_xaxes(title="Our estimate vs. asking price", ticksuffix="%", dtick=10)
fig.update_yaxes(title="Homes")
st.plotly_chart(fig, width="stretch")

st.markdown("### How we test: 5-fold cross-validation")
c1, c2 = st.columns([1.1, 1])
with c1:
    fig = go.Figure()
    for r in range(5):
        for c in range(5):
            test = r == c
            fig.add_shape(type="rect", x0=c + 0.05, x1=c + 0.95, y0=-r - 0.4, y1=-r + 0.4,
                          fillcolor=BLUE if test else "#eceae4", line=dict(width=0))
        fig.add_annotation(x=-0.1, y=-r, text=f"Round {r + 1}", showarrow=False, xanchor="right", font=dict(color=INK_2))
    fig.add_annotation(x=2.5, y=0.8, text="Each block = 1/5 of the homes", showarrow=False, font=dict(color=INK_2))
    style(fig, 260)
    fig.update_xaxes(visible=False, range=[-1.2, 5.1])
    fig.update_yaxes(visible=False, range=[-4.6, 1.1])
    st.plotly_chart(fig, width="stretch")
    st.caption("Gray = homes the model learns from. Blue = homes it is tested on. After 5 rounds, every home has been "
               "tested exactly once by a model that never saw its price.")
with c2:
    st.markdown("""
**Like a fair exam:** you can't grade a student on the exact questions they studied.

So we split the homes into 5 groups, train on 4, test on the 5th, and rotate.

**Every number in this dashboard, and every "fair value" shown in the apps, comes from the round where that home was
in the test group.**
""")

st.markdown("### How we prevent overfitting (memorizing instead of learning)")
checks = [
    ("Test only on unseen homes", "5-fold cross-validation; every score is out-of-fold."),
    ("No neighbors in two places", "Units in the same building (same coordinates) are always in the same fold, so a condo can't be "
                                   "'tested' while its identical twin next door is in training."),
    ("Comps can't peek", "The nearby-comps feature is rebuilt inside every round from training homes only, and a home is never its own neighbor."),
    ("No answer-copying columns", f"Zestimate, rent Zestimate and tax-assessed value are excluded. We tested it: adding the Zestimate makes "
                                  f"the error look like {leak['MdAPE_%']:.1f}% (R² {leak['R2_log']:.3f}), but that's just copying Zillow, not learning."),
    ("Built-in brakes on the model", "Small learning rate, ≥ 20 homes per rule, 80% row/column sampling, L2 penalty."),
    ("Ensemble weights tested too", "The weights that combine the three models are learned on 4/5 of the homes and scored on the other 1/5 (nested cross-validation)."),
    ("Stress test on new neighborhoods", "Hold out entire ZIP codes: can it price a neighborhood it has never seen?"),
    ("Honest uncertainty", "80% price ranges, checked on held-out homes to really contain 80% of prices."),
]
cols = st.columns(2)
for i, (title, body) in enumerate(checks):
    cols[i % 2].markdown(f"✅ **{title}.** {body}")

c1, c2 = st.columns(2)
with c1:
    gap = pd.DataFrame(diag["fold_gap"])
    fig = go.Figure()
    for col, name, color in [("train_MdAPE_%", "Training homes", ORANGE), ("validation_MdAPE_%", "Unseen homes", BLUE)]:
        fig.add_trace(go.Bar(x=[f"Round {i + 1}" for i in range(len(gap))], y=gap[col], name=name, marker_color=color,
                             text=[f"{v:.1f}%" for v in gap[col]], textposition="outside", cliponaxis=False,
                             hovertemplate=name + " · %{x}: %{y:.2f}%<extra></extra>"))
    style(fig, 320, showlegend=True, barmode="group", bargap=0.3, bargroupgap=0.08,
          legend=dict(orientation="h", y=1.12, x=0), margin=dict(l=10, r=40, t=80, b=10),
          title=dict(text="Error on training vs. unseen homes (LightGBM member)", font=dict(size=14), y=0.97))
    fig.update_yaxes(ticksuffix="%", title="Median % error")
    st.plotly_chart(fig, width="stretch")
with c2:
    curve = diag["learning_curve"]
    trees = np.arange(1, len(curve["train"]) + 1)
    fig = go.Figure()
    for key, name, color in [("train", "Training homes", ORANGE), ("validation", "Unseen homes", BLUE)]:
        y = np.array(curve[key]) * 100
        fig.add_trace(go.Scatter(x=trees, y=y, name=name, mode="lines", line=dict(color=color, width=2),
                                 hovertemplate=name + " · %{x} trees: %{y:.1f}%<extra></extra>"))
        fig.add_annotation(x=trees[-1], y=y[-1], text=f"{name} {y[-1]:.1f}%", showarrow=False, xanchor="right", yshift=10,
                           font=dict(color=color))
    style(fig, 320, showlegend=True, legend=dict(orientation="h", y=1.12, x=0), hovermode="x unified",
          margin=dict(l=10, r=40, t=80, b=10), title=dict(text="Learning curve (LightGBM member)", font=dict(size=14), y=0.97))
    fig.update_yaxes(ticksuffix="%", title="Average % error (mean |log error|)")
    fig.update_xaxes(title="Number of trees")
    st.plotly_chart(fig, width="stretch")
    st.caption("This chart shows the *average* error, which runs higher than the median because a few hard-to-price homes miss by a lot.")
g = pd.DataFrame(diag["fold_gap"])
insight(f"The model is more accurate on homes it trained on ({g['train_MdAPE_%'].mean():.1f}%) than on unseen homes "
        f"({g['validation_MdAPE_%'].mean():.1f}%). Some gap is normal. The warning sign of overfitting is the unseen-homes line "
        "turning back **up** as trees are added; it never does. It flattens out, and all 5 rounds agree within about 1 point, so the result is stable.")

st.markdown("### Stress tests")
u, iv = metrics["unseen_zip"], metrics["interval"]
k = st.columns(3)
k[0].metric("Error on ZIP codes never seen", f"{u['MdAPE_%']:.1f}%", help="Entire ZIP codes held out during training")
k[1].metric("80% ranges that contain the true price", f"{iv['holdout_coverage_%']:.1f}%", help="Target is 80%; measured on held-out homes")
k[2].metric("Typical width of the 80% range", f"±{iv['median_width_%'] / 2:.0f}%")
st.caption("Price ranges come from *conformalized quantile regression*: two extra models predict a low and a high price, then each "
           "end of the range is widened just enough that 10% of unseen homes fall below it and 10% above it, separately for each of 10 price bands.")
bands = pd.DataFrame(iv["bands_holdout"])
labels = ["under $" + f"{bands.fair_value_from[1] / 1e3:.0f}k"] + [
    f"${a / 1e3:.0f}k+" if a < 1e6 else f"${a / 1e6:.1f}M+" for a in bands.fair_value_from[1:]]
fig = go.Figure()
for col, name, color in [("below_%", "Listed below range (flagged underpriced)", BLUE), ("above_%", "Listed above range (flagged overpriced)", RED)]:
    fig.add_trace(go.Bar(x=labels, y=bands[col], name=name, marker_color=color,
                         hovertemplate="Fair value %{x}<br>" + name + ": %{y:.1f}%<extra></extra>"))
fig.add_hline(y=10, line=dict(color=INK_2, width=1, dash="dot"), annotation_text="10% target", annotation_position="top right")
style(fig, 300, showlegend=True, barmode="group", bargap=0.25, bargroupgap=0.08, legend=dict(orientation="h", y=1.18, x=0),
      margin=dict(l=10, r=40, t=60, b=10))
fig.update_yaxes(ticksuffix="%", range=[0, 16], title="Share of unseen homes")
fig.update_xaxes(title="Price band (by our fair value)")
st.plotly_chart(fig, width="stretch")
insight("The flags are fair at every price level: in each band about 10% of homes are flagged underpriced and 10% overpriced. "
        "A first version used one global adjustment and over-flagged cheap homes as bargains; calibrating per band fixed that.")

st.markdown("### How close are we to a professional tool? Zillow's Zestimate")
bench = metrics["benchmark_zestimate"]
rows = [
    ("Zillow, homes for sale (sees the asking price)", ZILLOW_ON_MARKET_MDAPE, GRAY),
    ("Zillow, homes NOT for sale (no asking price)", ZILLOW_OFF_MARKET_MDAPE, GRAY),
    ("FairValue HTX (never sees the asking price)", best["MdAPE_%"], BLUE),
    ("FairValue HTX, brand-new ZIP codes", u["MdAPE_%"], BLUE),
    ("Our linear baseline", metrics["stages"][0]["MdAPE_%"], MUTED),
]
st.plotly_chart(hbar([r[0] for r in rows], [r[1] for r in rows], ".2f", colors=[r[2] for r in rows], height=270,
                     hover="%{y}<br>median error %{x:.2f}%<extra></extra>").update_xaxes(ticksuffix="%", title="Median % error, lower is better"),
                width="stretch")
st.markdown(f"""
- When Zillow **can see the asking price**, its estimate is within {ZILLOW_ON_MARKET_MDAPE}% (nationwide). On our listings the Zestimate is
  within **{bench['zestimate']['MdAPE_%']:.1f}%** of the list price: it mostly echoes the asking price, so it **can't tell you whether that price is fair.**
- The fair comparison is Zillow **without** an asking price: **{ZILLOW_OFF_MARKET_MDAPE}%**. Built in one day from public data, our model reaches
  **{best['MdAPE_%']:.1f}%**, close to the professional benchmark.
- *Caveats:* Zillow's figures are nationwide and measured against **sale** prices; ours are Houston **list** prices. Zillow source: zillow.com/zestimate.
""")

st.markdown("### Test it yourself")
if st.button("🎲 Show me a random home the model never saw"):
    st.session_state["test_idx"] = int(np.random.default_rng().choice(scored.index))
idx = st.session_state.get("test_idx", int(scored.index[123]))
h = scored.loc[idx]
c1, c2 = st.columns([1, 1.6])
with c1:
    if isinstance(h.imgSrc, str):
        st.image(h.imgSrc, width="stretch")
with c2:
    st.markdown(f"**{h.address}**  \n{h.bedrooms:.0f} bd · {h.bathrooms:g} ba · {h.livingArea:,.0f} sqft · built {h.yearBuilt:.0f}")
    k = st.columns(3)
    k[0].metric("Actual list price", money(h.price))
    k[1].metric("Our estimate", money(h.fair_value), delta=f"{(h.fair_value / h.price - 1) * 100:+.1f}% vs. actual", delta_color="off")
    k[2].metric("Zillow Zestimate", money(h.zestimate) if pd.notna(h.zestimate) else "n/a")
    st.plotly_chart(plot_range(h.price, h.fair_lo, h.fair_value, h.fair_hi), width="stretch")
    st.caption(f"Our 80% range: {md_money(h.fair_lo)} – {md_money(h.fair_hi)}. This estimate came from the cross-validation "
               "round where this home was hidden from training.")

takeaway(f"Tested honestly on unseen homes, the model's typical error is **{best['MdAPE_%']:.1f}%**, close to Zillow's own "
         f"{ZILLOW_OFF_MARKET_MDAPE}% when it can't see the price. Its price ranges are honest too: 80% really means 80%.")
st.page_link("views/price_home.py", label="Next: put the model to work →", icon="🧮")
