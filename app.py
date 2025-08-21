import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from streamlit_plotly_events import plotly_events
from datetime import datetime

# ---------------- Page ----------------
st.set_page_config(page_title="Copart (CPRT) Stock Chart", layout="wide")
st.title("Copart (CPRT) Stock Chart")

# Valid Yahoo period/interval pairs
TF = {
    "All Time": ("max", "1wk"),
    "5 Years": ("5y", "1d"),
    "1 Year": ("1y", "1d"),
    "6 Months": ("6mo", "1d"),
    "3 Months": ("3mo", "1h"),
    "1 Month": ("1mo", "30m"),
    "1 Week": ("7d", "5m"),
    "1 Day (Intraday)": ("1d", "5m"),
}
label = st.sidebar.selectbox("Select timeframe:", list(TF.keys()))
period, interval = TF[label]
autorefresh = st.sidebar.checkbox("Auto-refresh intraday (30s)", value=False) if "Intraday" in label else False

# ---------------- Data ----------------
@st.cache_data(ttl=60 if "Intraday" in label else 300, show_spinner=False)
def fetch_prices(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Fetch stock prices from Yahoo Finance.
    Using auto_adjust=False to get raw prices and handle them explicitly.
    """
    try:
        tkr = yf.Ticker(ticker)
        # Use auto_adjust=False to get raw data
        df = tkr.history(period=period, interval=interval, auto_adjust=False, actions=True)
        
        if df is None or df.empty:
            st.error(f"No data returned for {ticker}")
            return pd.DataFrame()
        
        # Use the Close price (raw close is what most people expect to see)
        df = df[['Close']].copy()
        df = df.dropna()
        
        if df.empty:
            st.error("No valid price data after cleaning")
            return pd.DataFrame()
        
        # Rename for clarity
        df.rename(columns={'Close': 'Price'}, inplace=True)
        
        # Make timezone-naive for Plotly
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Add timestamp column
        df['ts'] = df.index
        
        # Debug info
        st.sidebar.write(f"Data points: {len(df)}")
        st.sidebar.write(f"Latest price: ${df['Price'].iloc[-1]:.2f}")
        st.sidebar.write(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        
        return df[['ts', 'Price']]
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

if autorefresh:
    st.autorefresh(interval=30_000, key="refresh_intraday")

with st.spinner("Loading CPRT prices…"):
    data = fetch_prices("CPRT", period, interval)

if data.empty:
    st.error("No data returned from Yahoo Finance for this timeframe. Try another.")
    st.stop()

# Verify data integrity
st.sidebar.markdown("### Data Check")
st.sidebar.write(f"Min price: ${data['Price'].min():.2f}")
st.sidebar.write(f"Max price: ${data['Price'].max():.2f}")
st.sidebar.write(f"Mean price: ${data['Price'].mean():.2f}")

# ---------------- Plot (WebGL, forced money axis) ----------------
fig = go.Figure()

# Add the price line
fig.add_trace(
    go.Scattergl(
        x=data['ts'],
        y=data['Price'],
        mode='lines',
        name='Close Price',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Price: $%{y:.2f}<extra></extra>',
    )
)

# Hide weekends / off-hours on shorter windows (cosmetic)
rangebreaks = []
if label in ("1 Day (Intraday)", "1 Week", "1 Month", "3 Months"):
    rangebreaks = [
        dict(bounds=["sat", "mon"]),
        dict(bounds=[16, 9.5], pattern="hour"),  # US RTH approx (ET)
    ]

fig.update_layout(
    title=f"CPRT - {label}",
    xaxis_title="Date/Time",
    yaxis_title="Price ($)",
    hovermode="x unified",
    xaxis=dict(
        rangeslider=dict(visible=label not in ("1 Day (Intraday)", "1 Week")),
        rangebreaks=rangebreaks,
        type='date'
    ),
    yaxis=dict(
        autorange=True,
        rangemode='normal',
        tickprefix='$',
        tickformat=',.2f',
        zeroline=False,
        gridcolor='rgba(128,128,128,0.2)'
    ),
    plot_bgcolor='white',
    margin=dict(l=60, r=40, t=60, b=40),
    height=520
)

# ---------------- Display Chart ----------------
hover_pts = plotly_events(
    fig,
    click_event=False,
    select_event=False,
    hover_event=True,
    override_width="100%",
    override_height=520,
)

# ---------------- Hover readout ----------------
if hover_pts:
    try:
        x_val = hover_pts[-1]["x"]
        y_val = float(hover_pts[-1]["y"])
        st.markdown(
            f"<div style='font-family:monospace; font-size:14px; margin-top:10px;'>"
            f"<b>Hover</b> ⟶ {x_val} · <b>${y_val:,.2f}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )
    except:
        pass
else:
    st.markdown(
        "<div style='color:#888; font-size:12px; margin-top:10px;'>Hover over the chart to see price here.</div>",
        unsafe_allow_html=True,
    )

# ---------------- High / Low for the visible window ----------------
if not data.empty:
    price_series = data['Price']
    ts_series = data['ts']
    
    hi_idx = price_series.idxmax()
    lo_idx = price_series.idxmin()
    
    hi_price = price_series.loc[hi_idx]
    lo_price = price_series.loc[lo_idx]
    hi_ts = ts_series.loc[hi_idx]
    lo_ts = ts_series.loc[lo_idx]
    
    st.markdown(
        f"<div style='font-size:14px; margin-top:10px;'>"
        f"<b>Period High:</b> ${hi_price:,.2f} "
        f"(<span style='color:#888'>{hi_ts:%Y-%m-%d %H:%M}</span>)"
        f" &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Period Low:</b> ${lo_price:,.2f} "
        f"(<span style='color:#888'>{lo_ts:%Y-%m-%d %H:%M}</span>)"
        f"</div>",
        unsafe_allow_html=True,
    )

# ---------------- Additional Info ----------------
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if len(data) > 1:
        current_price = data['Price'].iloc[-1]
        prev_price = data['Price'].iloc[-2]
        change = current_price - prev_price
        change_pct = (change / prev_price) * 100
        color = "green" if change >= 0 else "red"
        arrow = "↑" if change >= 0 else "↓"
        st.markdown(
            f"<div style='text-align: center;'>"
            f"<h4>Latest Price</h4>"
            f"<p style='font-size: 24px; font-weight: bold;'>${current_price:.2f}</p>"
            f"<p style='color: {color};'>{arrow} ${abs(change):.2f} ({change_pct:+.2f}%)</p>"
            f"</div>",
            unsafe_allow_html=True
        )

with col2:
    st.markdown(
        f"<div style='text-align: center;'>"
        f"<h4>Time Period</h4>"
        f"<p style='font-size: 18px;'>{label}</p>"
        f"<p style='color: #888;'>{period} / {interval}</p>"
        f"</div>",
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"<div style='text-align: center;'>"
        f"<h4>Data Points</h4>"
        f"<p style='font-size: 24px; font-weight: bold;'>{len(data):,}</p>"
        f"<p style='color: #888;'>observations</p>"
        f"</div>",
        unsafe_allow_html=True
    )
