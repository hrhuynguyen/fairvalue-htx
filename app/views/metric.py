import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (BLUE, INK_2, MUTED, ZILLOW_OFF_MARKET_MDAPE, insight, load_scored, md_money, money, stat_tiles,
                    step_header, style, table, takeaway)

scored = load_scored()
err = (scored.fair_value - scored.price).abs()
ape = err / scored.price
log_p, log_f = np.log(scored.price), np.log(scored.fair_value)
r2 = 1 - np.sum((log_p - log_f) ** 2) / np.sum((log_p - log_p.mean()) ** 2)

step_header(5, 7, "Choosing our metric")

st.markdown("### Summary")
st.markdown(f"Our main metric is the **median absolute percentage error (MdAPE)**: for every home, how far our estimate is "
            "from the asking price *in percent*, and then the middle value across all homes. In plain words: **half of our "
            f"estimates are within {ape.median():.1%} of the price.** We picked it because it's easy to understand, it treats a "
            "\\$150k starter home and a \\$1.5M mansion fairly, a few extreme homes can't distort it, and it's the same metric "
            "Zillow publishes, so we can compare ourselves with a professional tool (Step 7). We report three supporting metrics "
            "alongside it.")

st.code(f"""ape = (pred - price).abs() / price           # error in percent, one value per home
ape.median()                                 # MdAPE (main metric)    -> {ape.median():.1%}
(ape <= 0.10).mean()                         # within ±10%            -> {(ape <= 0.10).mean():.1%}
(pred - price).abs().median()                # median dollar error    -> {money(err.median())}
r2_score(np.log(price), np.log(pred))        # R² on log price        -> {r2:.3f}""",
        language="python")

st.markdown("### Our four metrics")
stat_tiles([(f"{ape.median():.1%}", "median % error (main metric)", "half of estimates are closer than this", True),
            (f"{(ape <= 0.10).mean():.0%}", "of homes within ±10%", "a practical 'good enough to negotiate' bar"),
            (f"${err.median() / 1e3:.1f}k", "median dollar error", "the same error, in dollars, for a typical home"),
            (f"{r2:.3f}", "R² on log price", "share of price differences explained (1 = perfect)")])

st.markdown("### Why percent, not dollars?")
seg = pd.DataFrame({"Starter (under $250k)": scored.price < 250_000, "Luxury (over $1M)": scored.price > 1_000_000})
rows = {name: (err[m].median(), ape[m].median(), int(m.sum())) for name, m in seg.items()}
c1, c2 = st.columns(2, gap="large")
for col, (title, idx, fmt, hover) in zip((c1, c2), [("Median error in dollars", 0, "$,.0f", "%{y:$,.0f}"),
                                                    ("Median error in percent", 1, ".1%", "%{y:.1%}")]):
    with col:
        vals = [v[idx] for v in rows.values()]
        fig = go.Figure(go.Bar(x=list(rows), y=vals, marker_color=[MUTED, BLUE] if idx == 0 else [BLUE, BLUE],
                               text=[format(v, fmt.lstrip("$")) if idx else money(v) for v in vals], textposition="outside",
                               cliponaxis=False, hovertemplate="%{x}<br>" + hover + "<extra></extra>"))
        style(fig, 300, title=dict(text=title, font=dict(size=14)), margin=dict(l=10, r=10, t=50, b=10))
        fig.update_yaxes(tickformat=fmt, rangemode="tozero")
        fig.update_xaxes(tickangle=0)
        st.plotly_chart(fig, width="stretch")
(cheap_d, cheap_p, n_cheap), (lux_d, lux_p, n_lux) = rows.values()
insight(f"In dollars, luxury homes look {lux_d / cheap_d:.0f}× harder to price ({md_money(lux_d)} vs {md_money(cheap_d)}), mostly "
        f"because they cost more. In percent the comparison is fair: {cheap_p:.1%} vs {lux_p:.1%}, so luxury homes are harder, but "
        f"about {lux_p / cheap_p:.1f}×, not {lux_d / cheap_d:.0f}×. And for Maya, an {md_money(cheap_d)} miss on a \\$150k home matters a lot, "
        "which a dollar metric would hide. That's also why the model predicts **log price**: it learns relative (%) errors.")

st.markdown("### Why the median, not the average?")
edges = np.arange(0, 0.62, 0.02)
counts, _ = np.histogram(ape.clip(upper=0.6), bins=edges)
fig = go.Figure(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=0.018, marker_color=BLUE,
                       hovertemplate="%{x:.0%} off: %{y:,} homes<extra></extra>"))
for val, name, pos in [(ape.median(), "median", "right"), (ape.mean(), "average", "left")]:
    fig.add_vline(x=val, line=dict(color=INK_2, width=1.5, dash="dot"))
    fig.add_annotation(x=val, y=counts.max(), text=f"{name} {val:.1%}", showarrow=False, yshift=12,
                       xanchor="right" if pos == "right" else "left", font=dict(color=INK_2, size=13))
style(fig, 300, margin=dict(l=10, r=10, t=30, b=10))
fig.update_xaxes(tickformat=".0%", title="How far off each estimate is (absolute % error, capped at 60%)")
fig.update_yaxes(title="Homes")
st.plotly_chart(fig, width="stretch")
insight(f"Most estimates are close, but a few hard-to-price homes (fixer-uppers, unusual estates, listing errors) miss by a lot. "
        f"They pull the **average** error up to {ape.mean():.1%}, while the **median** stays at {ape.median():.1%}, which describes a "
        f"typical home. Dollar metrics are hit even harder: the average dollar error is {md_money(err.mean())} and the root-mean-squared "
        f"error is {md_money(np.sqrt((err ** 2).mean()))}, both dominated by a handful of mansions.")

st.markdown("### The metrics we considered")
table(pd.DataFrame([
    ("Median % error (MdAPE)", "Typical miss, in percent", "✅ Main metric: fair across prices, robust to outliers, same as Zillow"),
    ("Within ±10%", "Share of homes we price within 10%", "✅ Supporting: a practical, easy-to-picture bar"),
    ("Median $ error", "Typical miss, in dollars", "✅ Supporting: makes the error concrete for a typical home"),
    ("R² on log price", "Share of price differences explained", "✅ Supporting: the standard regression score"),
    ("Average $ error (MAE)", "Average miss, in dollars", "❌ Dominated by expensive homes"),
    ("Root-mean-squared error (RMSE)", "Average squared miss, in dollars", "❌ Even more dominated by a few mansions and errors"),
], columns=["Metric", "What it measures", "Our decision"]).set_index("Metric"))

takeaway(f"We grade the model on its **median % error**: half of our estimates land within {ape.median():.1%} of the asking "
         f"price. It's fair to cheap and expensive homes alike, and directly comparable to Zillow's {ZILLOW_OFF_MARKET_MDAPE}%.")
st.page_link("views/model.py", label="Next: training the model →")
