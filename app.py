import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, date, timedelta
import numpy as np

# ---------------- Page Configuration ----------------
st.set_page_config(page_title="Copart (CPRT) Stock Chart", layout="wide")
st.title("Stock Price Chart")

TICKER = "CPRT"

# ---------------- Session-state defaults ----------------
def _ensure_state():
    if "start_date" not in st.session_state or "end_date" not in st.session_state:
        end = date.today()
        start = end - timedelta(days=30)  # default = last 1 month
        st.session_state.start_date = start
        st.session_state.end_date = end
_ensure_state()

# ---------------- Preset buttons + date inputs ----------------
st.subheader("Time range")

bcols = st.columns(8)
presets = [
    ("1D", 1),
    ("5D", 5),
    ("1M", 30),
    ("3M", 90),
    ("6M", 180),
    ("1Y", 365),
    ("5Y", 365*5),
    ("YTD", "ytd"),
]

clicked = None
for (i, (label, span)) in enumerate(presets):
    if bcols[i].button(label):
        clicked = (label, span)

# Apply preset if clicked
today = date.today()
if clicked:
    label, span = clicked
    if span == "ytd":
        st.session_state.start_date = date(today.year, 1, 1)
        st.session_state.end_date = today
    else:
        st.session_state.end_date = today
        st.session_state.start_date = today - timedelta(days=span)

# Custom range inputs (you can type dates)
c1, c2 = st.columns(2)
start_input = c1.date_input("Start date", value=st.session_state.start_date)
end_input = c2.date_input("End date", value=st.session_state.end_date)

# Guard: swap if user enters reversed range
if start_input > end_input:
    start_input, end_input = end_input, start_input

st.session_state.start_date = start_input
st.session_state.end_date = end_input

# ---------------- Interval chooser (Yahoo-friendly) ----------------
def choose_interval(start_d: date, end_d: date) -> str:
    days = (end_d - start_d).days or 1
    # Yahoo limits: 1m <= 7d; 5m <= 60d; 30m <= 60d; 1h <= 730d; else daily/weekly/monthly
    if days <= 7:       return "5m"   # stable & light; change to "1m" if you need minute granularity
    if days <= 60:      return "30m"
    if days <= 365:     return "1d"
    if days <= 365*5:   return "1wk"
    return "1mo"

interval = choose_interval(st.session_state.start_date, st.session_state.end_date)

# ---------------- Data Fetching ----------------
@st.cache_data(ttl=120)
def fetch_stock_data_range(ticker: str, start_d: date, end_d: date, interval: str):
    """
    Fetch adjusted OHLCV between start/end with a safe interval.
    We extend 'end' by one day since Yahoo's end is exclusive for some intervals.
    """
    try:
        t = yf.Ticker(ticker)
        df = t.history(
            start=pd.Timestamp(start_d),
            end=pd.Timestamp(end_d + timedelta(days=1)),
            interval=interval,
            auto_adjust=True,
            actions=False,
        )
        if df is None or df.empty:
            # fallback
            df = yf.download(
                ticker, start=start_d, end=end_d + timedelta(days=1),
                interval=interval, auto_adjust=True, progress=False
            )
        if df is None or df.empty:
            return None

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
        # numeric + tz-naive
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Close"])
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        return df
    except Exception as e:
        st.error(f"Fetch error: {e}")
        return None

with st.spinner(f"Loading {TICKER} {interval} data…"):
    stock_data = fetch_stock_data_range(TICKER, st.session_state.start_date, st.session_state.end_date, interval)

if stock_data is None or stock_data.empty:
    st.error("❌ Unable to fetch stock data for the selected range.")
    st.stop()

# ---------------- Top metrics ----------------
latest_price = stock_data["Close"].iloc[-1]
prev_price = stock_data["Close"].iloc[-2] if len(stock_data) > 1 else latest_price
price_change = latest_price - prev_price
pct_change = (price_change / prev_price * 100) if prev_price else 0

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Current Price", f"${latest_price:.2f}", f"{pct_change:+.2f}%")
with c2:
    st.metric("Latest Bar Range", f"${stock_data['Low'].iloc[-1]:.2f} – ${stock_data['High'].iloc[-1]:.2f}")
with c3:
    st.metric("Volume", f"{stock_data['Volume'].iloc[-1]:,.0f}")

# ---------------- Moving Average (keep your logic, but driven by span) ----------------
days_span = (st.session_state.end_date - st.session_state.start_date).days or 1
# sensible default MA: intraday -> none, < 3m -> 9, < 1y -> 50, else 100
if   days_span <= 7:   default_ma = None
elif days_span <= 90:  default_ma = 9
elif days_span <= 365: default_ma = 50
else:                  default_ma = 100

show_ma = False
ma_data = None
if default_ma is not None:
    show_ma = st.checkbox(f"Show {default_ma}-period Moving Average", value=False)
    if show_ma:
        ma_data = stock_data["Close"].rolling(window=default_ma).mean()

# ---------------- Plotly chart ----------------
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=stock_data.index,
        y=stock_data["Close"],
        mode="lines",
        name="Close (Adj.)",
        line=dict(width=2),
        hovertemplate="Date: %{x}<br>Price: $%{y:.2f}<extra></extra>",
    )
)
if show_ma and ma_data is not None:
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=ma_data,
            mode="lines",
            name=f"{default_ma}-period MA",
            line=dict(width=2, dash="dash"),
            hovertemplate="Date: %{x}<br>MA: $%{y:.2f}<extra></extra>",
        )
    )

# range breaks (hide weekends/off-hours) for short spans
rangebreaks = []
if days_span <= 120:  # only apply on shorter ranges
    rangebreaks = [
        dict(bounds=["sat", "mon"]),
        dict(bounds=[16, 9.5], pattern="hour"),  # approx US RTH ET
    ]

fig.update_layout(
    title=f"{TICKER} — {st.session_state.start_date:%Y-%m-%d} → {st.session_state.end_date:%Y-%m-%d}",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    height=520,
    template="plotly_white",
    hovermode="x unified",
    yaxis=dict(zeroline=False, rangemode="normal", tickprefix="$", separatethousands=True),
    xaxis=dict(rangeslider=dict(visible=days_span > 7), rangebreaks=rangebreaks),
)
st.plotly_chart(fig, use_container_width=True)

# ---------------- High/Low readout (robust for DatetimeIndex) ----------------
hi_ts = stock_data["Close"].idxmax()
lo_ts = stock_data["Close"].idxmin()
hi_price = float(stock_data.loc[hi_ts, "Close"])
lo_price = float(stock_data.loc[lo_ts, "Close"])

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
