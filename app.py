import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from streamlit_plotly_events import plotly_events

# ---------------- Page ----------------
st.set_page_config(page_title="Copart (CPRT) Stock Chart", layout="wide")
st.title("Copart (CPRT) Stock Chart")

# Safe period/interval pairs for Yahoo
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
def fetch_prices_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Use Ticker.history to avoid MultiIndex; return split/div-adjusted OHLCV."""
    tkr = yf.Ticker(ticker)
    df = tkr.history(period=period, interval=interval, auto_adjust=True, actions=False)
    if df is None or df.empty:
        return pd.DataFrame()
    # Normalize columns and dtypes
    df = df.rename(columns=str.title)  # Open, High, Low, Close, Volume
    df = df.dropna(how="all")
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close"]).copy()
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df["ts"] = df.index
    return df

if autorefresh:
    st.autorefresh(interval=30_000, key="refresh_intraday")

with st.spinner("Loading CPRT prices…"):
    data = fetch_prices_history("CPRT", period, interval)

if data.empty:
    st.error("No data returned from Yahoo Finance for this timeframe. Try another.")
    st.stop()

# ---------------- Plot (WebGL) ----------------
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

rangebreaks = []
if label in ("1 Day (Intraday)", "1 Week", "1 Month", "3 Months"):
    rangebreaks = [
        dict(bounds=["sat", "mon"]),              # hide weekends
        dict(bounds=[16, 9.5], pattern="hour"),   # hide non-RTH (approx ET)
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

# ---------------- Hover readout ----------------
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

# ---------------- High/Low ----------------
close_vals = data["Close"].astype(float).reset_index(drop=True)
ts_vals = pd.to_datetime(data["ts"]).reset_index(drop=True)
hi_i, lo_i = int(close_vals.idxmax()), int(close_vals.idxmin())
hi_price, lo_price = float(close_vals.iloc[hi_i]), float(close_vals.iloc[lo_i])
hi_ts, lo_ts = ts_vals.iloc[hi_i], ts_vals.iloc[lo_i]

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
