import streamlit as st
import yfinance as yf 
import matplotlib.pyplot as plt

st.title("Copart (CPRT) Stock Chart")

timeframes = {
  "All Time": "max",
  "5 Years": "5y",
  "1 Years": "1y",
  "6 Months": "6mo",
  "3 Months": "3mo",
  "1 Months": "1mo",
  "1 Week": "5d",
  "1 Day (Intraday)": "1d"
}
choice = st.sidebar.selectbox("Select timeframe:", list(timeframe.keys()))

ticker = yf.Ticker("CPRT")
period = timeframes[choice]

if period == "1d"
  data = ticker.history(period="1d", interval="5m")
else:
  data = ticker.history(period=period)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data.index, data["Close"], label="Close Price")
ax.set_title(f"CPRT Stock Price - {choice}")
ax.set_xlabel("Data")
ax.set_ylable("Price ($)")
ax.legend()

st.pyplot(fig)
  
