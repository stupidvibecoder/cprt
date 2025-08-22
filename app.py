import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from datetime import date, timedelta

# ---------------- Page ----------------
st.set_page_config(page_title="Copart (CPRT) Stock Chart", layout="wide")
st.title("Stock Price Chart")

TICKER = "CPRT"

# ---------------- Session defaults ----------------
today = date.today()
if "start_date" not in st.session_state:
    st.session_state.start_date = today - timedelta(days=30)  # default 1M
if "end_date" not in st.session_state:
    st.session_state.end_date = today

# ---------------- One-line controls: 8 presets + Start/End inputs ----------------
row = st.columns([1,1,1,1,1,1,1,1,2.2,2.2])

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
for i, (label, span) in enumerate(presets):
    if row[i].button(label):
        clicked = (label, span)

if clicked:
    lab, span = clicked
    if span == "ytd":
        st.session_state.start_date = date(today.year, 1, 1)
        st.session_state.end_date = today
    else:
        st.session_state.end_date = today
        st.session_state.start_date = today - timedelta(days=span)

# Start date (never allow future; hide label to save space)
start_input = row[8].date_input(
    "Start date",
    value=st.session_state.start_date,
    min_value=date(1990, 1, 1),
    max_value=today,  # <-- blocks future years/dates
    label_visibility="collapsed",
)

# End date (never allow future; min is start)
end_input = row[9].date_input(
    "End date",
    value=st.session_state.end_date,
    min_value=start_input,
    max_value=today,  # <-- blocks future years/dates
    label_visibility="collapsed",
)

# Keep dates sane
if start_input > end_input:
    start_input, end_input = end_input, start_input
st.session_state.start_date, st.session_state.end_date = start_input, end_input

# ---------------- Interval chooser ----------------
def choose_interval(start_d: date, end_d: date) -> str:
    days = (end_d - start_d).days or 1
    # Yahoo-friendly choices
    if days <= 7:    return "5m"   # (use "1m" if you truly need minute-by-minute)
    if days <= 60:   return "30m"
    if days <= 365:  return "1d"
    if days <= 365*5:return "1wk"
    return "1mo"

interval = choose_interval(st.session_state.start_date, st.session_state.end_date)

# ---------------- Data fetch ----------------
@st.cache_data(ttl=120)
def fetch_stock_data_range(ticker: str, start_d: date, end_d: date, interval: str):
    """
    Fetch adjusted OHLCV between start/end with a safe interval.
    Extend 'end' by one day because Yahoo's end can be exclusive.
    """
    t = yf.Ticker(ticker)
    df = t.history(
        start=pd.Timestamp(start_d),
        end=pd.Timestamp(end_d + timedelta(days=1)),
        interval=interval,
        auto_adjust=True,
        actions=False,
    )
    if df is None or df.empty:
        df = yf.download(
            ticker,
            start=start_d,
            end=end_d + timedelta(days=1),
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
    if df is None or df.empty:
        return None

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    # numeric + tz-naive
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"])
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    return df

with st.spinner(f"Loading {TICKER} {interval} data…"):
    stock_data = fetch_stock_data_range(TICKER, st.session_state.start_date, st.session_state.end_date, interval)

if stock_data is None or stock_data.empty:
    st.error("❌ Unable to fetch stock data for the selected range.")
    st.stop()

# ---------------- Metrics ----------------
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

# ---------------- Moving average (auto suggestion based on span) ----------------
days_span = (st.session_state.end_date - st.session_state.start_date).days or 1
if   days_span <= 7:    default_ma = None
elif days_span <= 90:   default_ma = 9
elif days_span <= 365:  default_ma = 50
else:                   default_ma = 100

show_ma = False
ma_data = None
if default_ma is not None:
    show_ma = st.checkbox(f"Show {default_ma}-period Moving Average", value=False)
    if show_ma:
        ma_data = stock_data["Close"].rolling(window=default_ma).mean()

# ---------------- Plot ----------------
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=stock_data.index,
        y=stock_data["Close"],
        mode="lines",
        name="Close (Adj.)",
        line=dict(width=2),
        connectgaps=True,  # draw through missing timestamps
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
            connectgaps=True,
            hovertemplate="Date: %{x}<br>MA: $%{y:.2f}<extra></extra>",
        )
    )

# Range breaks for short spans only
rangebreaks = []
if days_span <= 120:
    rangebreaks = [
        dict(bounds=["sat", "mon"]),
        dict(bounds=[16, 9.5], pattern="hour"),  # approx US RTH (ET)
    ]

fig.update_layout(
    title=f"{TICKER} — {st.session_state.start_date:%Y-%m-%d} → {st.session_state.end_date:%Y-%m-%d}",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    height=520,
    template="plotly_white",
    hovermode="x unified",
    yaxis=dict(
        zeroline=False,
        rangemode="normal",
        tickprefix="$",
        separatethousands=True
    ),
    xaxis=dict(
        rangeslider=dict(visible=days_span > 7),
        rangebreaks=rangebreaks
    ),
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- High/Low readout ----------------
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
