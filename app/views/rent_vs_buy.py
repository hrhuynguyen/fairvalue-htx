import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import BLUE, INK_2, default_pool, load_scored, md_money, money, page_header, style

ORANGE = "#eb6834"  # categorical slot 2: the 'rent' series
scored = load_scored()

page_header("CREATIVE 3 · RENT VS. BUY", "Rent or buy?",
            "For a home, after how many years does **buying** leave you wealthier than **renting** the same home and investing the "
            "money you didn't spend on a down payment? It's the question every first-time buyer asks.")

pool = st.session_state.get("deal_view") or default_pool(scored)
pool = [i for i in pool if pd.notna(scored.at[i, "rentZestimate"])]
sel = st.session_state.get("selected_home")
pick = st.selectbox("Choose a listing (from your latest search)", pool, index=pool.index(sel) if sel in pool else 0,
                    format_func=lambda i: f"{scored.at[i, 'address']} · {money(scored.at[i, 'price'])}")
st.session_state["selected_home"] = pick
h = scored.loc[pick]
past = (1 + h.zip_zhvi_5y_growth) ** (1 / 5) - 1 if pd.notna(h.zip_zhvi_5y_growth) else 0.03

c = st.columns(4)
price = c[0].number_input("Purchase price ($)", 50_000, 10_000_000, int(h.price), 5_000)
rent = c[1].number_input("Rent for a similar home ($/mo)", 300, 50_000, int(h.rentZestimate), 50,
                         help="Default: Zillow's rent estimate for this home")
appreciation = c[2].number_input("Home value growth %/yr", -5.0, 10.0, 3.0, 0.5,
                                 help=f"Long-run US average is ~3–4%. This ZIP grew {past:+.1%}/yr in the 5 years before the listing.") / 100
invest = c[3].number_input("Investment return %/yr", 0.0, 12.0, 5.0, 0.5, help="What the renter earns investing the down payment") / 100
with st.expander("More assumptions"):
    c = st.columns(4)
    rate = c[0].number_input("Mortgage rate %", 2.0, 12.0, 6.5, 0.125) / 100
    down = c[1].number_input("Down payment %", 0.0, 100.0, 10.0, 1.0) / 100
    tax = c[2].number_input("Property tax %/yr", 0.5, 4.0, 2.2, 0.1) / 100
    ins = c[3].number_input("Insurance %/yr", 0.2, 3.0, 1.0, 0.1) / 100
    c = st.columns(4)
    maint = c[0].number_input("Maintenance %/yr", 0.0, 4.0, 1.0, 0.25) / 100
    rent_growth = c[1].number_input("Rent growth %/yr", -2.0, 10.0, 3.0, 0.5) / 100
    buy_cost = c[2].number_input("Closing costs to buy %", 0.0, 10.0, 3.0, 0.5) / 100
    sell_cost = c[3].number_input("Costs to sell %", 0.0, 10.0, 6.0, 0.5) / 100

YEARS = 15
months = YEARS * 12
loan = price * (1 - down)
r = rate / 12
pmt = loan * r / (1 - (1 + r) ** -360)
balance, value, portfolio, cur_rent, owner_cash = loan, price, price * (down + buy_cost), rent, 0.0
buy_nw, rent_nw = [], []
for m in range(1, months + 1):
    interest = balance * r
    balance -= pmt - interest
    pmi = loan * 0.005 / 12 if balance > 0.8 * price else 0
    own_cost = pmt + price * (tax + ins + maint) / 12 + pmi  # tax/ins/maint on purchase price: conservative and simple
    # Whoever has the cheaper month invests the difference, so both paths spend the same cash.
    delta = own_cost - cur_rent
    portfolio = portfolio * (1 + invest / 12) + max(delta, 0)
    owner_cash = owner_cash * (1 + invest / 12) + max(-delta, 0)
    value *= (1 + appreciation) ** (1 / 12)
    if m % 12 == 0:
        buy_nw.append(value * (1 - sell_cost) - balance + owner_cash)
        rent_nw.append(portfolio)
        cur_rent *= 1 + rent_growth

years = np.arange(1, YEARS + 1)
diff = np.array(buy_nw) - np.array(rent_nw)
breakeven = next((int(y) for y, d in zip(years, diff) if d > 0), None)

k = st.columns(3)
k[0].metric("Buying wins after", f"{breakeven} years" if breakeven else f"> {YEARS} years")
k[1].metric("Owning cost, month 1", money(pmt + price * (tax + ins + maint) / 12 + (loan * 0.005 / 12 if down < 0.2 else 0)))
k[2].metric("Rent, month 1", money(rent))

fig = go.Figure()
for ys, name, color in [(buy_nw, "Buy: home equity after selling costs", BLUE), (rent_nw, "Rent: invested savings", ORANGE)]:
    fig.add_trace(go.Scatter(x=years, y=ys, mode="lines+markers", name=name, line=dict(color=color, width=2), marker=dict(size=6),
                             hovertemplate=name + "<br>Year %{x}: %{y:$,.0f}<extra></extra>"))
    fig.add_annotation(x=years[-1], y=ys[-1], text=name.split(":")[0], showarrow=False, xanchor="left", xshift=8, font=dict(color=color))
if breakeven:
    fig.add_vline(x=breakeven, line=dict(color=INK_2, width=1, dash="dot"))
    fig.add_annotation(x=breakeven, y=max(max(buy_nw), max(rent_nw)), text=f"break-even: year {breakeven}", showarrow=False,
                       xanchor="left", xshift=6, font=dict(color=INK_2))
style(fig, 380, showlegend=True, legend=dict(orientation="h", y=1.12, x=0), hovermode="x unified", margin=dict(l=10, r=60, t=50, b=10))
fig.update_xaxes(title="Years from today", dtick=1)
fig.update_yaxes(title="Wealth if you moved out that year", tickformat="$,.0f")
st.plotly_chart(fig, width="stretch")

if breakeven:
    st.success(f"If you'll stay **at least {breakeven} years**, buying this home likely builds more wealth than renting it. "
               "Leave sooner and the closing and selling costs make renting the better deal.")
else:
    st.warning(f"Under these assumptions renting stays ahead for {YEARS} years. High Texas property taxes and insurance make owning "
               "expensive relative to rent here. Try a lower price, a bigger down payment, or a home with a higher rent estimate.")
st.caption(f"Simplified model: ignores tax deductions and assumes the renter invests the {md_money(price * (down + buy_cost))} "
           "down payment and closing costs; each month, whichever option is cheaper invests the difference. Not financial advice.")
