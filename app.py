import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
import numpy as np

# ---------------- Page Configuration ----------------
st.set_page_config(page_title="Copart (CPRT) Stock Chart", layout="wide")
st.title("Stock Price Chart")

# ---------------- Timeframe Configuration ----------------
TIMEFRAMES = {
    "1 Day": {"period": "1d", "interval": "5m"},
    "5 Days": {"period": "5d", "interval": "30m"},
    "1 Month": {"period": "1mo", "interval": "1d"},
    "3 Months": {"period": "3mo", "interval": "1d"},
    "6 Months": {"period": "6mo", "interval": "1d"},
    "1 Year": {"period": "1y", "interval": "1d"},
    "5 Years": {"period": "5y", "interval": "1wk"},
}

# Define MA periods for different timeframes
MA_PERIODS = {
    "1 Day": None,   # No MA for intraday
    "5 Days": None,  # No MA for 1 week
    "1 Month": 9,
    "3 Months": 9,
    "6 Months": 50,
    "1 Year": 50,
    "5 Years": 100
}

# Sidebar controls
selected_timeframe = st.sidebar.selectbox(
    "Select Timeframe:",
    list(TIMEFRAMES.keys()),
    index=2  # Default to 1 Month
)
period = TIMEFRAMES[selected_timeframe]["period"]
interval = TIMEFRAMES[selected_timeframe]["interval"]

# ---------------- Data Fetching ----------------
@st.cache_data(ttl=300)
def fetch_stock_data(ticker, period, interval):
    """Fetch stock data with error handling (adjusted prices)."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            df = yf.download(ticker, period=period, interval=interval,
                             auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().copy()
        # Force numeric (avoids weird plotting issues)
        for c in ['Open','High','Low','Close','Volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['Close'])
        # Make tz-naive for consistency
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        return df
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)[:100]}")
        return None

with st.spinner("Loading CPRT stock data..."):
    stock_data = fetch_stock_data("CPRT", period, interval)

if stock_data is None or stock_data.empty:
    st.error("❌ Unable to fetch stock data. Please try a different timeframe.")
    st.stop()

# ---------------- Display Current Info ----------------
latest_price = stock_data['Close'].iloc[-1]
prev_price = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else latest_price
price_change = latest_price - prev_price
pct_change = (price_change / prev_price * 100) if prev_price else 0

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Current Price", f"${latest_price:.2f}", f"{pct_change:+.2f}%")
with c2:
    st.metric("Day Range", f"${stock_data['Low'].iloc[-1]:.2f} – ${stock_data['High'].iloc[-1]:.2f}")
with c3:
    st.metric("Volume", f"{stock_data['Volume'].iloc[-1]:,.0f}")

# ---------------- Moving Average Configuration ----------------
ma_period = MA_PERIODS.get(selected_timeframe)
show_ma = False
ma_data = None
if ma_period is not None:
    show_ma = st.checkbox(f"Show {ma_period}-day Moving Average", value=False)
    if show_ma:
        if selected_timeframe == "5 Years":
            ma_data = stock_data['Close'].rolling(window=20).mean()  # weekly data ≈ 20 weeks
        else:
            ma_data = stock_data['Close'].rolling(window=ma_period).mean()

# ---------------- CPRT Chart ----------------
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=stock_data.index,
        y=stock_data['Close'],
        mode='lines',
        name='Close (Adj)',
        line=dict(width=2),
        hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    )
)
if show_ma and ma_data is not None:
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=ma_data,
            mode='lines',
            name=f'{ma_period}-day MA' if selected_timeframe != "5 Years" else '20-week MA',
            line=dict(width=2, dash='dash'),
            hovertemplate='Date: %{x}<br>MA: $%{y:.2f}<extra></extra>'
        )
    )

fig.update_layout(
    title=f"CPRT - {selected_timeframe}",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    height=520,
    template="plotly_white",
    hovermode='x unified',
    yaxis=dict(zeroline=False, rangemode="normal", tickprefix="$", separatethousands=True),
    xaxis=dict(rangeslider=dict(visible=selected_timeframe not in ("1 Day","5 Days")))
)
st.plotly_chart(fig, use_container_width=True)
