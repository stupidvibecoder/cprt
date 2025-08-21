import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from streamlit_plotly_events import plotly_events

# ---------------- Page ----------------
st.set_page_config(page_title="Copart (CPRT) Stock Chart", layout="wide")
st.title("Copart (CPRT) Stock Chart")

# ---------------- Controls ----------------
# Use only Yahoo-supported period/interval pairs
TF_MAP = {
    "All Time": ("max", "1wk"),
    "5 Years": ("5y", "1d"),
    "1 Year": ("1y", "1d"),
    "6 Months": ("6mo", "1d"),
    "3 Months": ("3mo", "1h"),     # intraday allowed ≤60d; yfinance stitches safely
    "1 Month": ("1mo", "30m"),
    "1 Week": ("7d", "5m"),
    "1 Day (Intraday)": ("1d", "5m"),  # 1m is throttled; 5m is stable + fast
}
label = st.sidebar.selectbox("Select timeframe:", list(TF_MAP.keys()))
period, interval = TF_MAP[label]
autorefresh = st.sidebar.checkbox("Auto-refresh intraday (30s)", value=False) if "Intraday" in label else False

# ---------------- Data ----------------
@st.cache_data(ttl=60 if "Intraday" in label else 300, show_spinner=False)
def fetch_prices(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Get split/div-adjusted OHLCV. We use yfinance.download for reliability.
    Ensures numeric dtypes and a 'ts' column for Plotly.
    """
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=True,      # 'Close' is adjusted; no 'Adj Close' column returned
        progress=False,
        group_by="ticker",
    )
    if df.empty:
        return df

    # For single ticker, columns are single level. Normalize & ensure numeric
    df = df.rename(columns=str.title)  # Open, High, Low, Close, Volume
    # Drop rows with all-NaN (can occur at start/end of requests)
    df = df.dropna(how="all")
    # Ensure datetime index is tz-naive for Plotly
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    # Critical: enforce float dtype (avoid category/object surprises)
    for col in ("Open", "High", "Low", "Close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close"]).copy()
    df["ts"] = df.index
    return df

if autorefresh:
    st.autorefresh(interval=30_000, key="refresh_intraday")

with st.spinner("Loading CPRT prices…"):
    data = fetch_prices("CPRT", period, interval)

if data.empty:
    st.error("No data returned from Yahoo Finance for this timeframe. Try another.")
    st.stop()

# ---------------- Plot (WebGL for speed) ----------------
fig = go.Figure()
fig.add_trace(
    go.Scattergl(
        x=data["ts"],
        y=data["Close"].astype(float),   # explicitly float
        mode="lines",
        name="Close",
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>$%{y:.2f}<extra></extra>",
    )
)

# Skip non-trading hours on intraday for nicer spacing
rangebreaks = []
if "Intraday" in label or label in ("1 Week", "1 Month", "3 Months"):
    rangebreaks = [
        dict(bounds=["sat", "mon"]),          # weekends
        dict(bounds=[16, 9.5], pattern="hour")  # non-RTH approx (ET)
    ]

fig.update_layout(
    title=f"CPRT - {label}",
    xaxis_title="Date/Time",
    yaxis_title="Price ($)",
    hovermode="x unified",
    xaxis=dict(rangeslider=dict(visible=label not in ("1 Day (Intraday)", "1 Week")),
               rangebreaks=rangebreaks),
    margin=dict(l=40, r=20, t=60, b=40),
)

# ---------------- Hover readout (bottom-left) ----------------
hover_pts = plotly_events(
    fig,
    click_event=False,
    select_event=False,
    hover_event=True,      # use flag (older versions don't accept events=[...])
    override_width="100%",
    override_height=520,
)

if hover_pts:
    x_val = hover_pts[-1]["x"]
    y_val = float(hover_pts[-1]["y"])
    st.markdown(
        f"<div style='font-family:monospace; font-size:14px;'>"
        f"<b>Hover</b> ⟶ {x_val} · <b>${y_val:,.2f}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<div style='color:#888; font-size:12px;'>Hover over the chart to see price here.</div>",
        unsafe_allow_html=True,
    )

# ---------------- High/Low (value + date) ----------------
# Use iloc to avoid any index dtype quirks
close_series = data["Close"].astype(float).reset_index(drop=True)
ts_series = pd.to_datetime(data["ts"]).reset_index(drop=True)

hi_i = int(close_series.idxmax())
lo_i = int(close_series.idxmin())
hi_price = float(close_series.iloc[hi_i])
lo_price = float(close_series.iloc[lo_i])
hi_ts = ts_series.iloc[hi_i]
lo_ts = ts_series.iloc[lo_i]

st.markdown(
    f"<div style='font-size:14px; margin-top:4px;'>"
    f"<b>High:</b> ${hi_price:,.2f} "
    f"(<span style='color:#888'>{hi_ts:%Y-%m-%d %H:%M}</span>)"
    f" &nbsp;&nbsp;|&nbsp;&nbsp; "
    f"<b>Low:</b> ${lo_price:,.2f} "
    f"(<span style='color:#888'>{lo_ts:%Y-%m-%d %H:%M}</span>)"
    f"</div>",
    unsafe_allow_html=True,
)
