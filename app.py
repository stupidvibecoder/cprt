import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from streamlit_plotly_events import plotly_events
from datetime import datetime, timedelta
import time

# ---------------- Page Configuration ----------------
st.set_page_config(page_title="Copart (CPRT) Stock Chart", layout="wide")
st.title("Copart (CPRT) Stock Chart")

# ---------------- Helper Functions ----------------
def validate_data(df, ticker):
    """Validate that the data looks reasonable for the given ticker"""
    if df.empty:
        return False, "No data returned"
    
    # For CPRT, reasonable price range based on recent history
    price_min = df['Close'].min()
    price_max = df['Close'].max()
    
    # CPRT has been trading between roughly $40-$65 in recent years
    if price_min < 20 or price_max > 100:
        return False, f"Suspicious price range: ${price_min:.2f} - ${price_max:.2f}"
    
    return True, "Data validated"

def get_stock_data(ticker, period, interval, retries=3):
    """Fetch stock data with retries and validation"""
    for attempt in range(retries):
        try:
            # Method 1: Try using download first
            st.sidebar.write(f"Attempt {attempt + 1}: Fetching {period}/{interval}")
            
            if period == "1d":
                # For intraday, calculate specific start/end times
                end = datetime.now()
                start = end - timedelta(days=1)
                df = yf.download(ticker, start=start, end=end, interval=interval, 
                               progress=False, auto_adjust=True, prepost=True)
            elif period == "5d":
                # For 1 week view
                end = datetime.now()
                start = end - timedelta(days=7)
                df = yf.download(ticker, start=start, end=end, interval=interval,
                               progress=False, auto_adjust=True)
            else:
                # For other periods
                df = yf.download(ticker, period=period, interval=interval,
                               progress=False, auto_adjust=True)
            
            if not df.empty:
                # Clean column names if multi-level
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Validate the data
                is_valid, message = validate_data(df, ticker)
                if is_valid:
                    st.sidebar.success(f"✓ {message}")
                    return df
                else:
                    st.sidebar.warning(f"⚠️ {message}")
            
            # Method 2: Try using Ticker.history if download fails
            if df.empty or attempt == 1:
                tkr = yf.Ticker(ticker)
                df = tkr.history(period=period, interval=interval, auto_adjust=True)
                
                if not df.empty:
                    is_valid, message = validate_data(df, ticker)
                    if is_valid:
                        return df
            
            time.sleep(1)  # Brief pause between attempts
            
        except Exception as e:
            st.sidebar.error(f"Attempt {attempt + 1} failed: {str(e)[:50]}")
            time.sleep(1)
    
    return pd.DataFrame()

# ---------------- Timeframe Configuration ----------------
# Simplified timeframes that work reliably
TIMEFRAMES = {
    "1 Day": {"period": "1d", "interval": "5m", "show_extended": True},
    "5 Days": {"period": "5d", "interval": "30m", "show_extended": False},
    "1 Month": {"period": "1mo", "interval": "1d", "show_extended": False},
    "3 Months": {"period": "3mo", "interval": "1d", "show_extended": False},
    "6 Months": {"period": "6mo", "interval": "1d", "show_extended": False},
    "1 Year": {"period": "1y", "interval": "1d", "show_extended": False},
    "5 Years": {"period": "5y", "interval": "1wk", "show_extended": False},
}

# Sidebar controls
selected_timeframe = st.sidebar.selectbox(
    "Select Timeframe:",
    list(TIMEFRAMES.keys()),
    index=2  # Default to 1 Month
)

tf_config = TIMEFRAMES[selected_timeframe]
period = tf_config["period"]
interval = tf_config["interval"]

# Auto-refresh for intraday
if selected_timeframe == "1 Day":
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    if auto_refresh:
        st.rerun()

# ---------------- Fetch Data ----------------
st.sidebar.markdown("### Data Fetching")
data = get_stock_data("CPRT", period, interval)

if data.empty:
    st.error("❌ Unable to fetch stock data. Please try a different timeframe or refresh the page.")
    st.info("💡 Yahoo Finance may be experiencing issues. Try these alternatives:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Retry with 1 Month"):
            st.rerun()
    with col2:
        if st.button("🔄 Retry with 6 Months"):
            st.rerun()
    st.stop()

# ---------------- Data Processing ----------------
# Ensure we have the right columns
price_column = 'Close' if 'Close' in data.columns else 'close'
if price_column not in data.columns:
    st.error(f"Price column not found. Available columns: {list(data.columns)}")
    st.stop()

# Create clean dataframe
chart_data = pd.DataFrame()
chart_data['Price'] = data[price_column]
chart_data['Volume'] = data.get('Volume', data.get('volume', 0))
chart_data = chart_data.dropna(subset=['Price'])

# Remove timezone for Plotly compatibility
if hasattr(chart_data.index, 'tz'):
    chart_data.index = chart_data.index.tz_localize(None)

# ---------------- Display Current Info ----------------
st.sidebar.markdown("### Current Data")
latest_price = chart_data['Price'].iloc[-1]
st.sidebar.metric("Latest Price", f"${latest_price:.2f}")
st.sidebar.write(f"Data points: {len(chart_data)}")
st.sidebar.write(f"Range: ${chart_data['Price'].min():.2f} - ${chart_data['Price'].max():.2f}")

# ---------------- Create Chart ----------------
fig = go.Figure()

# Add candlestick or line chart based on data availability
if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
    # Use candlestick if OHLC data is available
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='CPRT',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))
else:
    # Otherwise use line chart
    fig.add_trace(go.Scatter(
        x=chart_data.index,
        y=chart_data['Price'],
        mode='lines',
        name='CPRT Price',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))

# Add volume bars if available
if chart_data['Volume'].sum() > 0:
    fig.add_trace(go.Bar(
        x=chart_data.index,
        y=chart_data['Volume'],
        name='Volume',
        yaxis='y2',
        marker_color='lightgray',
        opacity=0.3,
        hovertemplate='Volume: %{y:,.0f}<extra></extra>'
    ))

# Configure layout
fig.update_layout(
    title=f"CPRT - {selected_timeframe}",
    xaxis=dict(
        title="Date/Time",
        rangeslider=dict(visible=False),
        type='date'
    ),
    yaxis=dict(
        title="Price ($)",
        titlefont=dict(color='#1f77b4'),
        tickfont=dict(color='#1f77b4'),
        tickprefix='$',
        side='left'
    ),
    yaxis2=dict(
        title="Volume",
        titlefont=dict(color='gray'),
        tickfont=dict(color='gray'),
        overlaying='y',
        side='right',
        showgrid=False
    ),
    hovermode='x unified',
    height=600,
    template='plotly_white',
    showlegend=False
)

# Add range breaks for intraday charts
if selected_timeframe in ["1 Day", "5 Days"]:
    fig.update_xaxis(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # Hide weekends
            dict(bounds=[16, 9.5], pattern="hour")  # Hide non-trading hours
        ]
    )

# Display chart
chart_placeholder = st.empty()
with chart_placeholder.container():
    st.plotly_chart(fig, use_container_width=True, key="main_chart")

# ---------------- Summary Statistics ----------------
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    current = chart_data['Price'].iloc[-1]
    if len(chart_data) > 1:
        prev = chart_data['Price'].iloc[-2]
        change = current - prev
        pct = (change / prev) * 100
        st.metric("Current Price", f"${current:.2f}", f"{pct:+.2f}%")
    else:
        st.metric("Current Price", f"${current:.2f}")

with col2:
    high = chart_data['Price'].max()
    high_date = chart_data['Price'].idxmax()
    st.metric("Period High", f"${high:.2f}", help=f"On {high_date:%m/%d}")

with col3:
    low = chart_data['Price'].min()
    low_date = chart_data['Price'].idxmin()
    st.metric("Period Low", f"${low:.2f}", help=f"On {low_date:%m/%d}")

with col4:
    avg = chart_data['Price'].mean()
    st.metric("Average", f"${avg:.2f}")

# ---------------- Additional Information ----------------
with st.expander("📊 Chart Data Preview"):
    # Show last 10 data points
    preview_data = chart_data.tail(10).copy()
    preview_data['Date'] = preview_data.index
    preview_data = preview_data[['Date', 'Price', 'Volume']]
    preview_data['Price'] = preview_data['Price'].apply(lambda x: f"${x:.2f}")
    preview_data['Volume'] = preview_data['Volume'].apply(lambda x: f"{x:,.0f}")
    st.dataframe(preview_data, use_container_width=True)

# Footer
st.markdown("---")
st.caption(f"Data from Yahoo Finance • Last updated: {datetime.now():%Y-%m-%d %H:%M:%S}")
st.caption("Note: If data appears incorrect, try selecting a different timeframe or refreshing the page.")
