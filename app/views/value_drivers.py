import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import BLUE, DRIVER_GROUPS, GRAY, INK_2, MUTED, insight, load_json, load_scored, page_header, style

diag = load_json("diagnostics.json")
scored = load_scored()

page_header("APPLICATION 3 · PROPERTY VALUE ANALYSIS", "What adds the most value?",
            "We asked the model *what-if* questions about 3,000 real single-family homes: change one thing, keep everything "
            "else the same, and re-price the home. The median change is the feature's value in today's market.")

vd = pd.DataFrame(diag["value_drivers"]).sort_values("median_dollars", ascending=False)
colors = [INK_2 if v >= 0 else MUTED for v in vd.median_dollars]  # neutral: red/blue mean over/underpriced
fig = go.Figure(go.Bar(x=vd.median_dollars, y=vd.change, orientation="h", marker_color=colors,
                       text=[f"{'+' if d >= 0 else '−'}${abs(d):,.0f}  ({p:+.1f}%)" for d, p in zip(vd.median_dollars, vd.median_pct)],
                       textposition="outside", cliponaxis=False,
                       hovertemplate="%{y}<br>median change %{x:$,.0f}<extra></extra>"))
style(fig, 360, yaxis=dict(autorange="reversed"))
lo_x, hi_x = vd.median_dollars.min(), vd.median_dollars.max()
fig.update_xaxes(tickformat="$,.0f", title="Median change in fair value for a typical single-family home",
                 range=[lo_x * 1.6, hi_x * 1.6])
fig.update_layout(margin=dict(l=10, r=150, t=10, b=10))
fig.add_vline(x=0, line=dict(color=INK_2, width=1))
st.plotly_chart(fig, width="stretch")
d = {r["change"]: r for r in diag["value_drivers"]}
insight(f"**Space beats rooms.** Another 250 sqft adds about \\${d['+250 sqft of living area']['median_dollars']:,.0f}, and an extra bathroom "
        f"about \\${d['+1 bathroom']['median_dollars']:,.0f}. But adding a bedroom *without* adding space slightly **lowers** value "
        f"({d['+1 bedroom (same sqft)']['median_pct']:+.1f}%): buyers pay for space, not for chopped-up rooms. A foreclosure sells "
        f"about {abs(d['Foreclosure / bank-owned']['median_pct']):.0f}% below an otherwise identical home.")

st.markdown("### How location changes value")
loc = sum(scored[f"shap__{f}"] for f in DRIVER_GROUPS["Location & commute"] + DRIVER_GROUPS["Nearby comparable homes"])
df = pd.DataFrame({"dist": scored.dist_downtown_mi, "effect": np.exp(loc) - 1})
bins = np.arange(0, 46, 3)
df["band"] = pd.cut(df.dist, bins)
g = df.groupby("band", observed=True).effect.median()
mids = [b.mid for b in g.index]
fig = go.Figure(go.Scatter(x=mids, y=g.values, mode="lines+markers", line=dict(color=BLUE, width=2), marker=dict(size=8),
                           hovertemplate="%{x:.0f} mi from downtown<br>location effect %{y:+.0%}<extra></extra>"))
fig.add_hline(y=0, line=dict(color=GRAY, width=1, dash="dot"))
style(fig, 320)
fig.update_xaxes(title="Miles from downtown Houston")
fig.update_yaxes(tickformat="+.0%", title="Location effect on price vs. average")
st.plotly_chart(fig, width="stretch")
insight("This isolates **location alone** (same house, different place). Within about 7 miles of downtown, location adds a big "
        "premium (up to +40%). Beyond 10 miles it matters much less, staying within roughly ±12% of average; there, size, age and "
        "lot drive the price.")

st.markdown("### Spot the bargains: share of listings flagged, by home type")
t = scored[scored.flag != "Verify first"].groupby("homeType").flag.value_counts(normalize=True).unstack().fillna(0)
t.index = t.index.str.replace("_", " ").str.title().rename("Home type")
st.dataframe(t[["Underpriced", "Fair", "Overpriced"]], width="stretch",
             column_config={c: st.column_config.ProgressColumn(c, format="percent", min_value=0, max_value=1)
                            for c in ["Underpriced", "Fair", "Overpriced"]})
st.caption("By design about 10% of homes fall below and 10% above an 80% range. Big departures from that tell you where "
           "asking prices run hot or cold.")
