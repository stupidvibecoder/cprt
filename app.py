import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from streamlit_plotly_events import plotly_events

# ---------- Page ----------
st.set_page_config(page_title="Copart (CPRT) Stock Chart", layout="wide")
st.title("Copart (CPRT) Stock Chart")

# ---------- Controls ----------
TIMEFRAMES = {
    "All Time": ("max", "1wk"),
    "5 Years": ("5y", "1d"),
    "1 Year": ("1y", "1d"),
    "6 Months": ("6mo", "1h"),
    "3 Months": ("3mo", "1h"),
    "1 Month": ("1mo", "30m"),
    "1 Week": ("7d", "30m"),
    "1 Day (Intraday)": ("1d", "1m"),  # 1m works only for the most recent ~7 days
}
tf_label = st.sidebar.selectbox("Select timeframe:", list(TIMEFRAMES.keys()))
period, interval = TIMEFRAMES[tf_label]

autorefresh = False
if tf_label == "1 Day (Intraday)":
    autorefresh = st.sidebar.checkbox("Auto-refresh intraday (30s)", value=False)

# ---------- Data ----------
@st.cache_data(show_spinner=False)
def get_prices(period: str, interval: str) -> pd.DataFrame:
    """
    Use yfinance.download for reliability/perf.
    auto_adjust=True gives split/div adjusted close.
    """
    df = yf.download(
        "CPRT",
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if df.empty:
        return df
    # Normalize columns across intervals
    df = df.rename(columns=str.title)
    # Ensure tz-naive for Plotly
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df["ts"] = df.index
    return df

# Auto refresh intraday
if autorefresh:
    st.autorefresh(interval=30_000, key="refresh_intraday")

with st.spinner("Loading CPRT prices…"):
    data = get_prices(period, interval)

if data.empty:
    st.error("No data returned from Yahoo Finance for this timeframe. Try another.")
    st.stop()

# ---------- Plot (Plotly WebGL for speed) ----------
fig = go.Figure()

fig.add_trace(
    go.Scattergl(
        x=data["ts"],
        y=data["Close"],
        mode="lines",
        name="Close",
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>$%{y:.2f}<extra></extra>",
    )
)

# Add a (lightweight) rangeslider for longer windows
show_slider = tf_label not in ("1 Day (Intraday)", "1 Week")
fig.update_layout(
    title=f"CPRT - {tf_label}",
    xaxis_title="Date/Time",
    yaxis_title="Price ($)",
    hovermode="x unified",
    xaxis=dict(rangeslider=dict(visible=show_slider)),
    margin=dict(l=40, r=20, t=60, b=40),
)

# ---------- Hover readout (bottom-left, outside chart) ----------
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

# ---------- High / Low for current window ----------
hi_idx = data["Close"].idxmax()
lo_idx = data["Close"].idxmin()
hi_price, lo_price = data.loc[hi_idx, "Close"], data.loc[lo_idx, "Close"]

st.markdown(
    f"<div style='font-size:14px; margin-top:4px;'>"
    f"<b>High:</b> ${hi_price:,.2f} "
    f"(<span style='color:#888'>{hi_idx:%Y-%m-%d %H:%M}</span>)"
    f" &nbsp;&nbsp;|&nbsp;&nbsp; "
    f"<b>Low:</b> ${lo_price:,.2f} "
    f"(<span style='color:#888'>{lo_idx:%Y-%m-%d %H:%M}</span>)"
    f"</div>",
    unsafe_allow_html=True,
)
