import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from streamlit_plotly_events import plotly_events

# ---------------- Page ----------------
st.set_page_config(page_title="Copart (CPRT) Stock Chart", layout="wide")
st.title("Copart (CPRT) Stock Chart")

# Safe Yahoo period/interval pairs (intraday only for recent ranges)
TF = {
    "All Time": ("max", "1wk"),
    "5 Years": ("5y", "1d"),
    "1 Year": ("1y", "1d"),
    "6 Months": ("6mo", "1d"),
    "3 Months": ("3mo", "1d"),
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
    Robust fetch:
      - Avoid MultiIndex columns by using group_by='column'
      - Fall back to 'Adj Close' if 'Close' not present
      - Force numeric dtypes
      - Add 'ts' column for Plotly
    """
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=True,         # returns adjusted OHLC, usually no 'Adj Close'
        group_by="column",        # single-level columns
        progress=False,
        threads=True,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Normalize column names and ensure expected fields
    df.columns = [c.title() for c in df.columns]  # Open, High, Low, Close, Volume, Adj Close (sometimes)
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    # Drop rows with all NaN & coerce numerics
    df = df.dropna(how="all")
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close"]).copy()
    # Make tz-naive for Plotly
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert(None)

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
        y=data["Close"].astype(float),
        mode="lines",
        name="Close",
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>$%{y:.2f}<extra></extra>",
    )
)

# Hide weekends / non-RTH on shorter windows for better spacing
rangebreaks = []
if label in ("1 Day (Intraday)", "1 Week", "1 Month"):
    rangebreaks = [
        dict(bounds=["sat", "mon"]),
        dict(bounds=[16, 9.5], pattern="hour"),  # approx US RTH in ET
    ]

fig.update_layout(
    title=f"CPRT - {label}",
    xaxis_title="Date/Time",
    yaxis_title="Price ($)",
    hovermode="x unified",
    xaxis=dict(
        rangeslider=dict(visible=label not in ("1 Day (Intraday)", "1 Week")),
        rangebreaks=rangebreaks,
    ),
    margin=dict(l=40, r=20, t=60, b=40),
)

# ---------------- Hover readout (bottom-left) ----------------
hover_pts = plotly_events(
    fig,
    click_event=False,
    select_event=False,
    hover_event=True,
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
close_vals = data["Close"].astype(float).reset_index(drop=True)
ts_vals = pd.to_datetime(data["ts"]).reset_index(drop=True)

hi_i = int(close_vals.idxmax())
lo_i = int(close_vals.idxmin())
hi_price = float(close_vals.iloc[hi_i])
lo_price = float(close_vals.iloc[lo_i])
hi_ts = ts_vals.iloc[hi_i]
lo_ts = ts_vals.iloc[lo_i]

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
