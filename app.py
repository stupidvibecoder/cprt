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

# ---------------- High/Low readout (robust for DatetimeIndex) ----------------
if len(stock_data) >= 1:
    hi_ts = stock_data['Close'].idxmax()   # this is a Timestamp (index label)
    lo_ts = stock_data['Close'].idxmin()

    hi_price = float(stock_data.loc[hi_ts, 'Close'])
    lo_price = float(stock_data.loc[lo_ts, 'Close'])

    # If your index isn’t datetime, this formatting will still work (it will print the object)
    hi_ts_str = hi_ts.strftime("%Y-%m-%d %H:%M") if hasattr(hi_ts, "strftime") else str(hi_ts)
    lo_ts_str = lo_ts.strftime("%Y-%m-%d %H:%M") if hasattr(lo_ts, "strftime") else str(lo_ts)

    st.markdown(
        f"<div style='font-size:14px;'>"
        f"<b>High:</b> ${hi_price:,.2f} "
        f"(<span style='color:#888'>{hi_ts_str}</span>) "
        f"&nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Low:</b> ${lo_price:,.2f} "
        f"(<span style='color:#888'>{lo_ts_str}</span>)"
        f"</div>",
        unsafe_allow_html=True,
    )
else:
    st.info("Not enough data to compute high/low.")

# =====================================================================
#                     NHTSA CRASH DATA SECTION
# =====================================================================
st.header("NHTSA Crash Data (upload and visualize)")
st.caption("Upload a CSV export (e.g., FARS/CRSS monthly or yearly counts).")

# Controls for NHTSA data
nhtsa_col1, nhtsa_col2, nhtsa_col3 = st.columns([2,2,1])
with nhtsa_col1:
    uploaded = st.file_uploader("Upload NHTSA CSV", type=["csv"])
with nhtsa_col2:
    demo = st.button("Load demo data")

# Load data
nhtsa_df = None
if uploaded is not None:
    try:
        nhtsa_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")

elif demo:
    # Demo monthly series for structure preview
    dt_index = pd.date_range("2019-01-01", periods=72, freq="MS")
    vals = (1000 + np.sin(np.linspace(0, 12*np.pi, len(dt_index)))*80 + np.linspace(0, 150, len(dt_index))).astype(int)
    nhtsa_df = pd.DataFrame({"Date": dt_index, "Crashes": vals})

if nhtsa_df is not None:
    # Let user choose date/value columns
    cols = list(nhtsa_df.columns)
    with nhtsa_col1:
        date_col = st.selectbox("Date column", cols, index=0)
    with nhtsa_col2:
        value_col = st.selectbox("Value column", cols, index=min(1, len(cols)-1))

    # Parse dates and clean
    try:
        series = nhtsa_df[[date_col, value_col]].copy()
        series[date_col] = pd.to_datetime(series[date_col], errors="coerce")
        series[value_col] = pd.to_numeric(series[value_col], errors="coerce")
        series = series.dropna().sort_values(date_col)
    except Exception as e:
        st.error(f"Problem parsing your columns: {e}")
        st.stop()

    # Plot
    nfig = go.Figure()
    nfig.add_trace(
        go.Scatter(
            x=series[date_col],
            y=series[value_col],
            mode="lines",
            name=value_col,
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,}<extra></extra>",
        )
    )
    nfig.update_layout(
        title="NHTSA Crash Time Series",
        xaxis_title="Date",
        yaxis_title=value_col,
        template="plotly_white",
        hovermode="x unified",
        height=520,
        xaxis=dict(rangeslider=dict(visible=True)),
    )
    st.plotly_chart(nfig, use_container_width=True)

    # High/low
    hi_idx = series[value_col].idxmax()
    lo_idx = series[value_col].idxmin()
    st.markdown(
        f"<div style='font-size:14px;'>"
        f"<b>High:</b> {series.loc[hi_idx, value_col]:,} "
        f"(<span style='color:#888'>{series.loc[hi_idx, date_col]:%Y-%m}</span>) "
        f"&nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Low:</b> {series.loc[lo_idx, value_col]:,} "
        f"(<span style='color:#888'>{series.loc[lo_idx, date_col]:%Y-%m}</span>)"
        f"</div>", unsafe_allow_html=True
    )
else:
    st.info("Upload a CSV export from NHTSA (FARS/CRSS).")
