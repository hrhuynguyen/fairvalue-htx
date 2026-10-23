"""FairValue HTX: a beginner-friendly, end-to-end walk through a Houston home-price model.

Run:  .venv/bin/streamlit run app/app.py
"""
import streamlit as st

from common import inject_css

st.set_page_config(page_title="FairValue HTX", page_icon="🏠", layout="wide")
inject_css()

STORY = [
    st.Page("views/overview.py", title="Start here", icon="🏁", default=True),
    st.Page("views/data.py", title="1 · The data", icon="📦"),
    st.Page("views/eda.py", title="2 · Exploring the data", icon="🔍"),
    st.Page("views/cleaning.py", title="3 · Cleaning the data", icon="🧹"),
    st.Page("views/features.py", title="4 · Feature engineering", icon="🛠️"),
    st.Page("views/model.py", title="5 · Training the model", icon="🤖"),
    st.Page("views/validation.py", title="6 · Can we trust it?", icon="✅"),
]
APPS = [
    st.Page("views/price_home.py", title="Price any home", icon="🧮"),
    st.Page("views/affordability.py", title="Affordability by income", icon="💵"),
    st.Page("views/deals.py", title="Under- & overpriced homes", icon="🔎"),
    st.Page("views/home_report.py", title="Home report", icon="🏠"),
    st.Page("views/value_drivers.py", title="What adds value", icon="💎"),
    st.Page("views/neighborhoods.py", title="Neighborhood comparison", icon="🗺️"),
]

CREATIVE = [
    st.Page("views/ask.py", title="Ask FairValue (AI)", icon="🤖"),
    st.Page("views/game.py", title="Beat the model", icon="🎯"),
    st.Page("views/rent_vs_buy.py", title="Rent or buy?", icon="⚖️"),
]

nav = st.navigation({"The 5-minute story": STORY, "Put the model to work": APPS, "Creative extras": CREATIVE}, expanded=True)
with st.sidebar:
    st.caption("FairValue HTX · UH CodeCoogs AI Hackathon 2026 · Track 2")
nav.run()
