import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
import numpy as np
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
    "1 Day": None,  # No MA for intraday
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
    """Fetch stock data with error handling"""
    try:
        # Create ticker object
        stock = yf.Ticker(ticker)
        
        # Get historical data
        data = stock.history(period=period, interval=interval, auto_adjust=True)
        
        if data.empty:
            # Fallback to download method
            data = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False
            )
            
        if data.empty:
            return None
            
        # Clean up data
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        data = data.dropna()
        
        return data
        
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)[:100]}")
        return None

# Fetch the data
with st.spinner("Loading CPRT stock data..."):
    stock_data = fetch_stock_data("CPRT", period, interval)

if stock_data is None or stock_data.empty:
    st.error("❌ Unable to fetch stock data. Yahoo Finance might be experiencing issues.")
    st.info("Please try refreshing the page or selecting a different timeframe.")
    st.stop()

# Validate price range
price_min = stock_data['Close'].min()
price_max = stock_data['Close'].max()
if price_min < 20 or price_max > 100:
    st.warning(f"⚠️ Unusual price range detected: ${price_min:.2f} - ${price_max:.2f}")

# ---------------- Display Current Info ----------------
latest_price = stock_data['Close'].iloc[-1]
prev_price = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else latest_price
price_change = latest_price - prev_price
pct_change = (price_change / prev_price * 100) if prev_price != 0 else 0

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current Price", f"${latest_price:.2f}", f"{pct_change:+.2f}%")
with col2:
    st.metric("Day Range", f"${stock_data['Low'].iloc[-1]:.2f} - ${stock_data['High'].iloc[-1]:.2f}")
with col3:
    st.metric("Volume", f"{stock_data['Volume'].iloc[-1]:,.0f}")

# ---------------- Moving Average Configuration ----------------
ma_period = MA_PERIODS.get(selected_timeframe)
show_ma = False
ma_data = None

# Show checkbox only if MA is available for this timeframe
if ma_period is not None:
    show_ma = st.checkbox(f"Show {ma_period}-day Moving Average", value=False)
    
    if show_ma:
        # Calculate moving average
        if selected_timeframe == "5 Years":
            # For weekly data, adjust the period (100 days ≈ 20 weeks)
            ma_data = stock_data['Close'].rolling(window=20).mean()
        else:
            # For daily data
            ma_data = stock_data['Close'].rolling(window=ma_period).mean()

# ---------------- Create Chart ----------------
fig = go.Figure()

# Add line trace for closing prices
fig.add_trace(
    go.Scatter(
        x=stock_data.index,
        y=stock_data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue', width=2),
        hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    )
)

# Add moving average trace if enabled
if show_ma and ma_data is not None:
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=ma_data,
            mode='lines',
            name=f'{ma_period}-day MA',
            line=dict(color='yellow', width=2, dash='dash'),
            hovertemplate='Date: %{x}<br>MA: $%{y:.2f}<extra></extra>'
        )
    )

# Basic layout without complex updates
fig.update_layout(
    title=f"CPRT - {selected_timeframe}",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    height=600,
    template="plotly_white",
    showlegend=True,
    hovermode='x unified'  # Changed to unified for better hover display
)

# Display the chart
st.plotly_chart(fig, use_container_width=True)

# ---------------- Summary Statistics ----------------
st.title("Total Loss Frequency")

