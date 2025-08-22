import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from datetime import date, timedelta

# ============================== Page ==============================
st.set_page_config(page_title="Copart (CPRT) Stock Chart", layout="wide")
st.title("Stock Price Chart")

TICKER = "CPRT"
today = date.today()

# ======================= Session defaults ========================
if "start_date" not in st.session_state:
    st.session_state.start_date = today - timedelta(days=30)  # default = 1M
if "end_date" not in st.session_state:
    st.session_state.end_date = today

# ============= One-line presets + Start/End date inputs ==========
row = st.columns([1,1,1,1,1,1,1,1,2.3,2.3])

presets = [
    ("1D", 1), ("5D", 5), ("1M", 30), ("3M", 90),
    ("6M", 180), ("1Y", 365), ("5Y", 365*5), ("YTD", "ytd"),
]

clicked = None
for i, (label, span) in enumerate(presets):
    if row[i].button(label):
        clicked = (label, span)

if clicked:
    lab, span = clicked
    if span == "ytd":
        st.session_state.start_date = date(today.year, 1, 1)
        st.session_state.end_date = today
    else:
        st.session_state.end_date = today
        st.session_state.start_date = today - timedelta(days=span)

start_input = row[8].date_input(
    "Start date", value=st.session_state.start_date,
    min_value=date(1990, 1, 1), max_value=today, label_visibility="collapsed",
)
end_input = row[9].date_input(
    "End date", value=st.session_state.end_date,
    min_value=start_input, ma
