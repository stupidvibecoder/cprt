import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="Copart (CPRT) Stock Chart", layout="wide")
st.title("Copart (CPRT) Stock Chart")

# ---- Controls ----
timeframes = {
    "All Time": "max",
    "5 Years": "5y",
    "1 Year": "1y",
    "6 Months": "6mo",
    "3 Months": "3mo",
    "1 Month": "1mo",
    "1 Week": "5d",
    "1 Day (Intraday 5m)": "1d",
}
choice = st.sidebar.selectbox("Select timeframe:", list(timeframes.keys()))
autorefresh = st.sidebar.checkbox("Auto-refresh intraday (30s)", value=False)

@st.cache_data(ttl=60)
def get_prices(period: str) -> pd.DataFrame:
    t = yf.Ticker("CPRT")
    if period == "1d":
        df = t.history(period="1d", interval="5m", auto_adjust=True)
    else:
        df = t.history(period=period, auto_adjust=True)
    # yfinance returns a DatetimeIndex; keep both index and a column for Plotly
    df = df.copy()
    df["ts"] = df.index
    return df

period = timeframes[choice]
if period == "1d" and autorefresh:
    st.autorefresh(interval=30_000, key="refresh")

data = get_prices(period)
if data.empty:
    st.warning("No data returned from Yahoo Finance. Try a different timeframe.")
    st.stop()

# ---- Plotly figure ----
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=data["ts"], y=data["Close"],
        mode="lines", name="Close"
    )
)
fig.update_layout(
    title=f"CPRT - {choice}",
    xaxis_title="Date/Time",
    yaxis_title="Price ($)",
    hovermode="x unified",                # vertical crosshair + unified tooltip
    xaxis=dict(rangeslider=dict(visible=True)),  # adds draggable range slider
    margin=dict(l=40, r=20, t=60, b=40),
)

# ---- Render & capture hover events ----
hover_points = plotly_events(
    fig,
    events=["hover"],          # listen to hover
    select_event=False,
    override_width="100%",
    override_height=520,
)

# ---- Bottom-left readout for hovered price ----
if hover_points:
    x_val = hover_points[-1]["x"]          # ISO datetime string
    y_val = hover_points[-1]["y"]          # price
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

# ---- Show high and low for the selected period ----
high_price = data["Close"].max()
low_price = data["Close"].min()

st.markdown(
    f"<div style='font-size:14px;'>"
    f"<b>High:</b> ${high_price:,.2f} &nbsp;&nbsp;|&nbsp;&nbsp; "
    f"<b>Low:</b> ${low_price:,.2f}"
    f"</div>",
    unsafe_allow_html=True,
)
