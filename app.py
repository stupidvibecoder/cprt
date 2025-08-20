import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
from streamlit_plotly_events import plotly_events

# --- build interactive figure ---
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=data.index, y=data["Close"], mode="lines", name="Close"
    )
)
fig.update_layout(
    title=f"CPRT - {choice}",
    xaxis_title="Date/Time",
    yaxis_title="Price ($)",
    hovermode="x unified",  # vertical hover line with unified tooltip
    margin=dict(l=40, r=20, t=60, b=40),
)

# --- render and capture hover events ---
hover_points = plotly_events(
    fig,
    events=["hover"],          # listen to hover
    select_event=False,
    override_width="100%",
    override_height=520,
)

# --- bottom-left readout for hovered price ---
# If user hovers, show the latest hovered point's time & price.
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
    # default hint shown when nothing hovered yet
    st.markdown(
        "<div style='color:#888; font-size:12px;'>Hover over the chart to see price here.</div>",
        unsafe_allow_html=True,
    )

import pandas as pd

st.set_page_config(page_title="Copart (CPRT) Stock Chart", layout="wide")
st.title("Copart (CPRT) Stock Chart")

timeframes = {
    "All Time": "max",
    "5 Years": "5y",
    "1 Year": "1y",
    "6 Months": "6mo",
    "3 Months": "3mo",
    "1 Month": "1mo",
    "1 Week": "5d",
    "1 Day (Intraday 5m)": "1d"
}
choice = st.sidebar.selectbox("Select timeframe:", list(timeframes.keys()))


@st.cache_data(ttl=60) 
def get_prices(period: str) -> pd.DataFrame:
    ticker = yf.Ticker("CPRT")
    if period == "1d":
        return ticker.history(period="1d", interval="5m", auto_adjust=True)
    else:
        return ticker.history(period=period, auto_adjust=True)


data = get_prices(timeframes[choice])

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data.index, data["Close"], label="Close")
ax.set_title(f"CPRT - {choice}")
ax.set_xlabel("Date/Time")
ax.set_ylabel("Price ($)")
ax.legend()
ax.grid(True, alpha=0.2)
st.pyplot(fig, clear_figure=True)

st.caption(
    "Data source: Yahoo Finance via yfinance. Intraday data is typically delayed ~15 min."
)
