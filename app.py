import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Copart (CPRT) Stock Chart", layout="wide")
st.title("Copart (CPRT) Stock Chart")

timeframes = {
    "All Time": "max",
    "5 Years": "5y",
    "1 Year": "1y",
    "6 Months": "6mo",
    "3 Months": "3mo",
    "1 Month": "1mo",
    "1 Week": "5d",
    "1 Day (Intraday 5m)": "1d"
}
choice = st.sidebar.selectbox("Select timeframe:", list(timeframes.keys()))
autorefresh = st.sidebar.checkbox("Auto-refresh intraday (30s)", value=False)

@st.cache_data(ttl=60) 
def get_prices(period: str) -> pd.DataFrame:
    ticker = yf.Ticker("CPRT")
    if period == "1d":
        return ticker.history(period="1d", interval="5m", auto_adjust=True)
    else:
        return ticker.history(period=period, auto_adjust=True)


data = get_prices(timeframes[choice])

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data.index, data["Close"], label="Close")
ax.set_title(f"CPRT - {choice}")
ax.set_xlabel("Date/Time")
ax.set_ylabel("Price ($)")
ax.legend()
ax.grid(True, alpha=0.2)
st.pyplot(fig, clear_figure=True)

st.caption(
    "Data source: Yahoo Finance via yfinance. Intraday data is typically delayed ~15 min."
)
