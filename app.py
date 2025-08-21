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
        # Download data
        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False
        )
        
        if data.empty:
            return None
            
        # Ensure we have single-level columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        # Keep only necessary columns
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        data = data.dropna()
        
        # Remove timezone for Plotly
        if hasattr(data.index, 'tz') and data.index.tz is not None:
            data.index = data.index.tz_localize(None)
            
        return data
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Fetch the data
with st.spinner("Loading CPRT stock data..."):
    stock_data = fetch_stock_data("CPRT", period, interval)

if stock_data is None or stock_data.empty:
    st.error("❌ Unable to fetch stock data. Please try again.")
    st.stop()

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

# ---------------- Create Chart ----------------
fig = go.Figure()

# Add candlestick chart
fig.add_trace(
    go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name='CPRT'
    )
)

# Update layout - simplified to avoid the error
fig.update_layout(
    title=f"CPRT - {selected_timeframe}",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    height=600,
    template="plotly_white",
    showlegend=False,
    hovermode='x unified'
)

# Update axes
fig.update_xaxis(
    rangeslider_visible=False,
    type='date'
)

fig.update_yaxis(
    tickprefix='$',
    side='right'
)

# Add range breaks for intraday
if selected_timeframe in ["1 Day", "5 Days"]:
    fig.update_xaxis(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(bounds=[16, 9.5], pattern="hour")
        ]
    )

# Display the chart
st.plotly_chart(fig, use_container_width=True)

# ---------------- Summary Statistics ----------------
st.markdown("### Summary Statistics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    high = stock_data['High'].max()
    st.metric("Period High", f"${high:.2f}")

with col2:
    low = stock_data['Low'].min()
    st.metric("Period Low", f"${low:.2f}")

with col3:
    avg = stock_data['Close'].mean()
    st.metric("Average Price", f"${avg:.2f}")

with col4:
    st.metric("Data Points", f"{len(stock_data):,}")

# ---------------- Data Preview ----------------
with st.expander("📊 View Raw Data"):
    # Show last 10 rows
    preview = stock_data.tail(10).copy()
    preview = preview.round(2)
    st.dataframe(preview)

# ---------------- Sidebar Info ----------------
st.sidebar.markdown("### Data Info")
st.sidebar.write(f"**Symbol:** CPRT")
st.sidebar.write(f"**Exchange:** NASDAQ")
st.sidebar.write(f"**Period:** {period}")
st.sidebar.write(f"**Interval:** {interval}")
st.sidebar.write(f"**Data Points:** {len(stock_data)}")
st.sidebar.write(f"**Price Range:** ${stock_data['Close'].min():.2f} - ${stock_data['Close'].max():.2f}")

# Auto-refresh option
if selected_timeframe == "1 Day":
    if st.sidebar.checkbox("Auto-refresh (30s)"):
        st.experimental_rerun()

# Footer
st.markdown("---")
st.caption("Data provided by Yahoo Finance")
st.caption(f"Last updated: {datetime.now():%Y-%m-%d %H:%M:%S}")
