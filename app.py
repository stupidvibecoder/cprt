import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from datetime import date, timedelta

# ---------------- Page ----------------
st.set_page_config(page_title="Copart (CPRT) Stock Chart", layout="wide")
st.title("Stock Price Chart")

TICKER = "CPRT"
today = date.today()

# ---------------- Session defaults ----------------
if "start_date" not in st.session_state:
    st.session_state.start_date = today - timedelta(days=30)  # default 1M
if "end_date" not in st.session_state:
    st.session_state.end_date = today

# ---------------- One-line controls: 8 presets + Start/End inputs ----------------
row = st.columns([1,1,1,1,1,1,1,1,2.3,2.3])

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

# Start date (block future)
start_input = row[8].date_input(
    "Start date",
    value=st.session_state.start_date,
    min_value=date(1990, 1, 1),
    max_value= today,
    label_visibility="collapsed",
)
# End date (block future; min is start)
end_input = row[9].date_input(
    "End date",
    value=st.session_state.end_date,
    min_value=start_input,
    max_value=today,
    label_visibility="collapsed",
)

# Keep dates sane
if start_input > end_input:
    start_input, end_input = end_input, start_input
st.session_state.start_date, st.session_state.end_date = start_input, end_input

# ---------------- Interval chooser ----------------
def choose_interval(start_d: date, end_d: date) -> str:
    days = (end_d - start_d).days or 1
    if days <= 7:    return "5m"   # (use "1m" only if you truly need minute data)
    if days <= 60:   return "30m"
    if days <= 365:  return "1d"
    if days <= 365*5:return "1wk"
    return "1mo"

interval = choose_interval(st.session_state.start_date, st.session_state.end_date)

# ---------------- Data fetch helpers ----------------
@st.cache_data(ttl=120)
def fetch_stock_data_range(ticker: str, start_d: date, end_d: date, interval: str):
    """Adjusted OHLCV between start/end. Extend end by 1 day (Yahoo exclusivity)."""
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

    # Normalize
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"])
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    return df

@st.cache_data(ttl=3600)
def fetch_earnings_df(ticker: str) -> pd.DataFrame:
    """Get earnings date table with estimate/reported/surprise. yfinance returns last/future events."""
    try:
        t = yf.Ticker(ticker)
        # limit large enough to cover several years
        df = t.get_earnings_dates(limit=60)
        if df is None or df.empty:
            return pd.DataFrame()
        # Standardize column names if present
        cols = {c.lower(): c for c in df.columns}
        # Expected columns: 'Earnings Date', 'EPS Estimate', 'Reported EPS', 'Surprise(%)'
        # Normalize names for easier access
        rename_map = {}
        for c in df.columns:
            cl = c.lower()
            if "earnings date" in cl: rename_map[c] = "Earnings Date"
            elif "eps estimate" in cl: rename_map[c] = "EPS Estimate"
            elif "reported eps" in cl: rename_map[c] = "Reported EPS"
            elif "surprise" in cl: rename_map[c] = "Surprise(%)"
        df = df.rename(columns=rename_map).copy()

        # Ensure datetime + numerics
        if "Earnings Date" in df.columns:
            df["Earnings Date"] = pd.to_datetime(df["Earnings Date"], errors="coerce").dt.date
        for c in ("EPS Estimate", "Reported EPS", "Surprise(%)"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["Earnings Date"])
    except Exception:
        return pd.DataFrame()

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

# ---------------- Moving average ----------------
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

# ---------------- Earnings calendar toggle ----------------
show_earn = st.checkbox("Earnings calendar", value=False)

# ---------------- Stock price chart ----------------
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=stock_data.index,
        y=stock_data["Close"],
        mode="lines",
        name="Close (Adj.)",
        line=dict(width=2),
        connectgaps=True,
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

# If earnings overlay is on, add marker dots with EPS tooltip
if show_earn:
    earn_df = fetch_earnings_df(TICKER)
    if not earn_df.empty:
        # Filter earnings between the chosen dates
        mask = (earn_df["Earnings Date"] >= st.session_state.start_date) & (earn_df["Earnings Date"] <= st.session_state.end_date)
        earn_win = earn_df.loc[mask].copy()

        if not earn_win.empty:
            # Build y-values for markers using daily resampled close (so it works for intraday windows too)
            daily_close = stock_data["Close"].resample("D").last().ffill()
            xs = []
            ys = []
            hovers = []

            for _, r in earn_win.iterrows():
                d = pd.Timestamp(r["Earnings Date"])
                if d in daily_close.index:
                    y = float(daily_close.loc[d])
                else:
                    # if the exact date isn't in index (holiday), try nearest previous day
                    y = float(daily_close.asof(d))
                xs.append(d)
                ys.append(y)

                est = r["EPS Estimate"] if "EPS Estimate" in r else None
                rep = r["Reported EPS"] if "Reported EPS" in r else None
                spr = r["Surprise(%)"] if "Surprise(%)" in r else None

                # Compute surprise if missing and we have est+rep
                if pd.isna(spr) and (pd.notna(est) and pd.notna(rep) and est != 0):
                    spr = (rep / est - 1.0) * 100.0

                hover_text = (
                    f"Earnings: {d.date()}<br>"
                    f"Est EPS: {est if pd.notna(est) else '—'}<br>"
                    f"Actual EPS: {rep if pd.notna(rep) else '—'}<br>"
                    f"Surprise: {spr:.2f}%"
                    if spr is not None and pd.notna(spr)
                    else f"Earnings: {d.date()}<br>"
                         f"Est EPS: {est if pd.notna(est) else '—'}<br>"
                         f"Actual EPS: {rep if pd.notna(rep) else '—'}"
                )
                hovers.append(hover_text)

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    name="Earnings",
                    marker=dict(size=8, symbol="circle", line=dict(width=1)),
                    hovertemplate="%{text}<extra></extra>",
                    text=hovers,
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

# =====================================================================
#                     Comparable performance (percent change)
# =====================================================================
st.header("Comparable performance")

# Mapping from user-facing choices -> Yahoo-compatible symbols
COMPARATORS = {
    "SPX": "^GSPC",   # S&P 500 index
    "RBA": "RBA",     # Ritchie Bros.
    "S5INDU": "XLI",  # Industrials sector proxy ETF
}

choices = st.multiselect(
    "Compare against (multi-select):",
    options=list(COMPARATORS.keys()),
    default=[],
)

@st.cache_data(ttl=120)
def fetch_close_series(ticker: str, start_d: date, end_d: date, interval: str) -> pd.Series | None:
    """Return a timezone-naive Close series (adjusted) indexed by timestamp."""
    df = fetch_stock_data_range(ticker, start_d, end_d, interval)
    if df is None or df.empty:
        return None
    s = df["Close"].copy()
    s = pd.to_numeric(s, errors="coerce").dropna()
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
        s.index = s.index.tz_convert(None)
    return s

# Build percent-change frame, base = 0% at first valid CPRT price
cprt_close = fetch_close_series(TICKER, st.session_state.start_date, st.session_state.end_date, interval)
if cprt_close is None or cprt_close.empty:
    st.warning("No CPRT data for the selected range.")
else:
    base = float(cprt_close.iloc[0])
    cprt_pct = (cprt_close / base - 1.0) * 100.0
    df_pct = pd.DataFrame({"CPRT": cprt_pct})

    # Add selected comparators
    for label in choices:
        sym = COMPARATORS[label]
        s = fetch_close_series(sym, st.session_state.start_date, st.session_state.end_date, interval)
        if s is None or s.empty:
            st.info(f"Could not load {label} data.")
            continue
        s = s.reindex(df_pct.index).ffill()
        base_c = float(s.iloc[0])
        df_pct[label] = (s / base_c - 1.0) * 100.0

    # Plot cumulative % change
    pfig = go.Figure()
    pfig.add_trace(
        go.Scatter(
            x=df_pct.index,
            y=df_pct["CPRT"],
            mode="lines",
            name="CPRT",
            connectgaps=True,
            hovertemplate="Date: %{x}<br>Change: %{y:.2f}%<extra></extra>",
        )
    )
    for label in [k for k in df_pct.columns if k != "CPRT"]:
        pfig.add_trace(
            go.Scatter(
                x=df_pct.index,
                y=df_pct[label],
                mode="lines",
                name=label,
                connectgaps=True,
                hovertemplate="Date: %{x}<br>Change: %{y:.2f}%<extra></extra>",
            )
        )

    rangebreaks2 = []
    if days_span <= 120:
        rangebreaks2 = [
            dict(bounds=["sat", "mon"]),
            dict(bounds=[16, 9.5], pattern="hour"),
        ]

    pfig.update_layout(
        title=f"Since {st.session_state.start_date:%Y-%m-%d} (cumulative % change)",
        xaxis_title="Date",
        yaxis_title="Change (%)",
        template="plotly_white",
        hovermode="x unified",
        height=520,
        yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor="#aaa", ticksuffix="%"),
        xaxis=dict(rangeslider=dict(visible=False), rangebreaks=rangebreaks2),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(pfig, use_container_width=True)

    st.caption(
        "Notes: SPX maps to Yahoo '^GSPC' (S&P 500). S5INDU maps to 'XLI' ETF as an Industrials proxy. "
        "All series are adjusted and rebased to 0% on the start date. Earnings data from Yahoo Finance via yfinance."
    )
