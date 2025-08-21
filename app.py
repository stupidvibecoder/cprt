import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta

# ---------------- Page Configuration ----------------
st.set_page_config(page_title="Copart (CPRT) Stock Chart", layout="wide")
st.title("Copart (CPRT) Stock Chart")

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


# Using a simpler approach to avoid Plotly errors
fig = go.Figure()

# Add line trace for closing prices
fig.add_trace(
    go.Scatter(
        x=stock_data.index,
        y=stock_data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue', width=2)
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
    hovermode='x'
)

# Display the chart
st.plotly_chart(fig, use_container_width=True)

# ---------------- Summary Statistics ----------------
st.markdown("### Summary Statistics")
col1, col2, col3 = st.columns(4)

with col1:
    high = stock_data['High'].max()
    high_date = stock_data['High'].idxmax()
    st.metric("Period High", f"${high:.2f}")
    st.caption(f"{high_date.strftime('%Y-%m-%d')}")

with col2:
    low = stock_data['Low'].min()
    low_date = stock_data['Low'].idxmin()
    st.metric("Period Low", f"${low:.2f}")
    st.caption(f"{low_date.strftime('%Y-%m-%d')}")

with col3:
    avg = stock_data['Close'].mean()
    st.metric("Average Price", f"${avg:.2f}")

# -------------- Sidebar Info ----------------
st.sidebar.write("**Symbol:** CPRT")


# Refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()

# Footer
st.markdown("---")
st.caption("Data provided by Yahoo Finance. Prices are adjusted for splits and dividends.")
st.caption(f"Last updated: {datetime.now():%Y-%m-%d %H:%M:%S}")
