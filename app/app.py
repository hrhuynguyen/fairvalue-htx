"""XMarket: a beginner-friendly, end-to-end walk through a Houston home-price model.

Run:  .venv/bin/streamlit run app/app.py
"""
from pathlib import Path

import streamlit as st

from common import inject_css

ASSETS = Path(__file__).parent / "assets"
st.set_page_config(page_title="XMarket", page_icon=str(ASSETS / "logo_icon.svg"), layout="wide")
st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "logo_icon.svg"), size="large")
inject_css()

STORY = [
    st.Page("views/overview.py", title="0 · Introduction", default=True),
    st.Page("views/data.py", title="1 · The data"),
    st.Page("views/eda.py", title="2 · Exploring the data"),
    st.Page("views/cleaning.py", title="3 · Cleaning the data"),
    st.Page("views/features.py", title="4 · Feature engineering"),
    st.Page("views/metric.py", title="5 · Choosing our metric"),
    st.Page("views/model.py", title="6 · Training the model"),
    st.Page("views/validation.py", title="7 · Validation"),
]
APPS = [
    st.Page("views/price_home.py", title="1 · Price any home"),
    st.Page("views/affordability.py", title="2 · Affordability by income"),
    st.Page("views/where_to_live.py", title="3 · Where should I live?"),
    st.Page("views/deals.py", title="4 · Under- & overpriced homes"),
    st.Page("views/home_report.py", title="5 · Home report"),
    st.Page("views/value_drivers.py", title="6 · What adds value"),
    st.Page("views/neighborhoods.py", title="7 · Neighborhood comparison"),
]
CREATIVE = [
    st.Page("views/ask.py", title="1 · Ask XMarket (AI)"),
    st.Page("views/game.py", title="2 · Beat the model"),
    st.Page("views/rent_vs_buy.py", title="3 · Rent or buy?"),
]

nav = st.navigation({"Part I: Data Science Task": STORY, "Application": APPS, "Creative extras": CREATIVE}, expanded=True)
with st.sidebar:
    st.caption("XMarket · UH CodeCoogs AI Hackathon 2026 · Track 2")
nav.run()
