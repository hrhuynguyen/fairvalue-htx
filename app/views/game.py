import numpy as np
import pandas as pd
import streamlit as st

from common import load_scored, money, page_header

ROUNDS = 5
scored = load_scored()
pool = scored[(scored.flag != "Verify first") & scored.imgSrc.notna() & scored.livingArea.notna()
              & scored.homeType.isin(["SINGLE_FAMILY", "TOWNHOUSE", "CONDO"]) & scored.price.between(100_000, 1_500_000)]

page_header("CREATIVE 2 · HOW GOOD IS THE MODEL, REALLY?", "Beat the model",
            f"You see what the model sees: photo, size, age and location. Guess the asking price. "
            f"After {ROUNDS} homes we compare your typical error with the model's. Every home here was hidden from the model during training.")


def new_game():
    st.session_state.game = {"homes": list(np.random.default_rng().choice(pool.index, ROUNDS, replace=False)),
                             "round": 0, "results": [], "revealed": False}


restart = st.button("🔄 New game")
if restart or "game" not in st.session_state:
    new_game()
g = st.session_state.game

if g["round"] >= ROUNDS:
    res = pd.DataFrame(g["results"])
    you, model = res.you_err.median(), res.model_err.median()
    st.subheader("🏁 Final score")
    k = st.columns(2)
    k[0].metric("Your typical error", f"{you:.0%}")
    k[1].metric("Model's typical error", f"{model:.0%}")
    st.markdown("### 🎉 You beat the model!" if you < model else "### 🤖 The model wins this time.")
    st.dataframe(res.assign(you_err=res.you_err * 100, model_err=res.model_err * 100), hide_index=True, width="stretch",
                 column_config={"address": "Home", "price": st.column_config.NumberColumn("Asking price", format="$%,.0f"),
                                "guess": st.column_config.NumberColumn("Your guess", format="$%,.0f"),
                                "model": st.column_config.NumberColumn("Model", format="$%,.0f"),
                                "you_err": st.column_config.NumberColumn("Your error", format="%.0f%%"),
                                "model_err": st.column_config.NumberColumn("Model error", format="%.0f%%")})
    st.caption("Over all 19,430 homes, the model's typical (median) error is 7.4%.")
    st.stop()

h = scored.loc[g["homes"][g["round"]]]
st.progress(g["round"] / ROUNDS, text=f"Home {g['round'] + 1} of {ROUNDS}")
c1, c2 = st.columns([1.3, 1])
with c1:
    st.image(h.imgSrc, width="stretch")
with c2:
    st.markdown(f"### {h.city}, TX {h.zipcode}")
    st.markdown(f"""
- **{h.homeType.replace('_', ' ').title()}**
- **{h.bedrooms:.0f}** bedrooms · **{h.bathrooms:g}** bathrooms
- **{h.livingArea:,.0f}** sqft living area
- Built in **{h.yearBuilt:.0f}**{' (new construction)' if h.is_new_construction else ''}
- **{h.dist_downtown_mi:.0f} miles** from downtown Houston
""")
    guess = st.number_input("Your guess ($)", 20_000, 5_000_000, 350_000, 5_000, key=f"guess_{g['round']}")
    if not g["revealed"]:
        if st.button("Reveal the price", type="primary"):
            g["revealed"] = True
            g["results"].append({"address": h.address, "price": h.price, "guess": guess, "model": h.fair_value,
                                 "you_err": abs(guess / h.price - 1), "model_err": abs(h.fair_value / h.price - 1)})
            st.rerun()
    else:
        r = g["results"][-1]
        k = st.columns(3)
        k[0].metric("Asking price", money(r["price"]))
        k[1].metric("You", money(r["guess"]), delta=f"{r['you_err']:.0%} off", delta_color="off")
        k[2].metric("Model", money(r["model"]), delta=f"{r['model_err']:.0%} off", delta_color="off")
        st.markdown("**You were closer! 🎉**" if r["you_err"] < r["model_err"] else "**The model was closer. 🤖**")
        if st.button("Next home →", type="primary"):
            g["round"] += 1
            g["revealed"] = False
            st.rerun()
