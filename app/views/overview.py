import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import (BLUE, FLAG_COLOR, INK_2, ZILLOW_OFF_MARKET_MDAPE, html_block, load_json, load_scored,
                    max_affordable_price, page_header, stat_tiles, style)

metrics = load_json("metrics.json")
scored = load_scored()
best = metrics["stages"][-1]
afford_75k = (scored.price <= max_affordable_price(75_000)).mean()

page_header("UH CodeCoogs AI Hackathon 2026 · Track 2", ":blue[X]Market",
            "Is this Houston home fairly priced, and can I actually afford it? We trained a model on 19,430 listings to "
            "estimate every home's **fair price**, without ever looking at its asking price.")

stat_tiles([
    (f"{metrics['n_listings']:,}", "Houston homes analyzed", "Zillow listings, June 2024"),
    (f"{best['MdAPE_%']:.1f}%", "Typical prediction error", "on homes the model never saw", True),
    (f"{ZILLOW_OFF_MARKET_MDAPE:.2f}%", "Zillow's own error", "when it can't see the asking price"),
    (f"{afford_75k:.0%}", "of listings fit a $75k salary", "all-in cost ≤ 28% of income"),
])

c1, c2 = st.columns([1.55, 1], gap="large")
with c1:
    s = scored[scored.flag != "Verify first"].sample(9000, random_state=3)
    fig = go.Figure()
    for flag in ["Fair", "Underpriced", "Overpriced"]:
        d = s[s.flag == flag]
        fig.add_trace(go.Scattergl(x=d.fair_value, y=d.price, mode="markers", name=flag,
                                   marker=dict(color=FLAG_COLOR[flag], size=4, opacity=0.35 if flag == "Fair" else 0.6),
                                   customdata=d.address,
                                   hovertemplate="%{customdata}<br>fair %{x:$,.0f} · asking %{y:$,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[6e4, 6e6], y=[6e4, 6e6], mode="lines", line=dict(color=INK_2, width=1, dash="dot"),
                             hoverinfo="skip", showlegend=False))
    fig.add_annotation(x=np.log10(2.2e6), y=np.log10(2.5e5), text="asking below fair value ↘", showarrow=False,
                       font=dict(color=BLUE, size=14))
    fig.add_annotation(x=np.log10(1.6e5), y=np.log10(1.5e6), text="↖ asking above fair value", showarrow=False,
                       font=dict(color=FLAG_COLOR["Overpriced"], size=14))
    ticks = dict(tickvals=[1e5, 2e5, 5e5, 1e6, 2e6, 5e6], ticktext=["$100k", "$200k", "$500k", "$1M", "$2M", "$5M"])
    style(fig, 470, showlegend=True, legend=dict(orientation="h", y=1.08, x=0, itemsizing="constant"),
          margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(type="log", range=[np.log10(6e4), np.log10(6e6)], title="Our fair value", **ticks)
    fig.update_yaxes(type="log", range=[np.log10(6e4), np.log10(6e6)], title="Asking price", **ticks)
    st.plotly_chart(fig, width="stretch")
with c2:
    st.markdown("### Case study")
    st.markdown("A new UH graduate earning **\\$75,000** at the Texas Medical Center. She wants her first home within 15 miles "
                "of work for under **\\$1,750 a month**. In this project, we will analyze the Houston Housing Market.")
    st.markdown("#### Her three questions")
    st.markdown("1. **What's a fair price** for this home?\n"
                "2. **What can I afford**, once Texas taxes and insurance are included?\n"
                "3. **Which listings are bargains**, and which are overpriced?")
    counts = scored.flag.value_counts()
    share = counts / len(scored)
    st.markdown(f"Of all **{len(scored):,}** homes, **{counts['Fair']:,} ({share['Fair']:.0%})** are priced within our fair "
                f"range (gray), **{counts['Underpriced']:,} ({share['Underpriced']:.0%})** are listed below it (blue) and "
                f"**{counts['Overpriced']:,} ({share['Overpriced']:.0%})** above it (red). The other {counts['Verify first']:,} have "
                "gaps so extreme they're set aside to verify first.")
    st.caption("Chart: a random 9,000 of those homes. On the dotted line = asking price equals our fair value.")

st.markdown("### How we built it")
steps = [("1", "Data", "25,948 listings + IRS income + Zillow ZIP trends"), ("2", "Explore", "What drives Houston prices?"),
         ("3", "Clean", "Remove land, floor plans, errors"), ("4", "Features", "Commute, nearby comps, ZIP income"),
         ("5", "Metric", "Median % error: fair at every price"),
         ("6", "Model", "Linear baseline → blend of boosted trees"), ("7", "Validate", "Honest tests, no leakage, vs. Zillow"),
         ("8", "Apps", "Fair price, where to live, deals, AI search")]
html_block('<div class="fv-steps">' + "".join(
    f'<div class="fv-step"><div class="fv-step-n">{n}</div><div class="fv-step-t">{t}</div><div class="fv-step-d">{d}</div></div>'
    for n, t, d in steps) + "</div>")
st.page_link("views/data.py", label="Start the story: the data →")
