import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from datetime import date, timedelta

# ============================== Page ==============================
st.set_page_config(page_title="Apple (AAPL) Dashboard", layout="wide")
st.title("Stock Price Chart")

TICKER = "AAPL"
today = date.today()

# ======================= Session defaults ========================
if "start_date" not in st.session_state:
    st.session_state.start_date = today - timedelta(days=30)  # default = 1M
if "end_date" not in st.session_state:
    st.session_state.end_date = today

# ============= One-line presets + Start/End date inputs ==========
row = st.columns([1,1,1,1,1,1,1,1,2.3,2.3])

presets = [
    ("1D", 1), ("5D", 5), ("1M", 30), ("3M", 90),
    ("6M", 180), ("1Y", 365), ("5Y", 365*5), ("YTD", "ytd"),
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

start_input = row[8].date_input(
    "Start date", value=st.session_state.start_date,
    min_value=date(1990, 1, 1), max_value=today, label_visibility="collapsed",
)
end_input = row[9].date_input(
    "End date", value=st.session_state.end_date,
    min_value=start_input, max_value=today, label_visibility="collapsed",
)
if start_input > end_input:
    start_input, end_input = end_input, start_input
st.session_state.start_date, st.session_state.end_date = start_input, end_input

# ====================== Interval chooser =========================
def choose_interval(start_d: date, end_d: date) -> str:
    days = (end_d - start_d).days or 1
    if days <= 7:     return "5m"
    if days <= 60:    return "30m"
    if days <= 365:   return "1d"
    if days <= 365*5: return "1wk"
    return "1mo"

interval = choose_interval(st.session_state.start_date, st.session_state.end_date)

# ====================== Data fetch helpers =======================
@st.cache_data(ttl=120)
def fetch_stock_data_range(ticker: str, start_d: date, end_d: date, interval: str):
    """Adjusted OHLCV between start/end. Extend end by 1 day (Yahoo end can be exclusive)."""
    t = yf.Ticker(ticker)
    df = t.history(
        start=pd.Timestamp(start_d),
        end=pd.Timestamp(end_d + timedelta(days=1)),
        interval=interval, auto_adjust=True, actions=False,
    )
    if df is None or df.empty:
        df = yf.download(
            ticker, start=start_d, end=end_d + timedelta(days=1),
            interval=interval, auto_adjust=True, progress=False,
        )
    if df is None or df.empty:
        return None
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"])
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    return df

@st.cache_data(ttl=3600)
def fetch_earnings_df(ticker: str) -> pd.DataFrame:
    """Get earnings dates; normalize columns even if yfinance returns date as index."""
    try:
        t = yf.Ticker(ticker)
        df = t.get_earnings_dates(limit=60)
        if df is None or df.empty:
            return pd.DataFrame()
        if df.index.name and "date" in str(df.index.name).lower():
            df = df.reset_index()
        rename_map = {}
        for c in df.columns:
            lc = str(c).lower()
            if "earn" in lc and "date" in lc: rename_map[c] = "Earnings Date"
            elif "estimate" in lc and "eps" in lc: rename_map[c] = "EPS Estimate"
            elif ("reported" in lc or "actual" in lc) and "eps" in lc: rename_map[c] = "Reported EPS"
            elif "surprise" in lc: rename_map[c] = "Surprise(%)"
        df = df.rename(columns=rename_map)
        if "Earnings Date" not in df.columns:
            if "Date" in df.columns: df = df.rename(columns={"Date":"Earnings Date"})
            else: return pd.DataFrame()
        df["Earnings Date"] = pd.to_datetime(df["Earnings Date"], errors="coerce").dt.date
        for c in ("EPS Estimate","Reported EPS","Surprise(%)"):
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["Earnings Date"])
    except Exception:
        return pd.DataFrame()

# ========================== Load price data ============================
with st.spinner(f"Loading {TICKER} {interval} data…"):
    stock_data = fetch_stock_data_range(TICKER, st.session_state.start_date, st.session_state.end_date, interval)

if stock_data is None or stock_data.empty:
    st.error("❌ Unable to fetch stock data for the selected range.")
    st.stop()

# ============================ Metrics ============================
latest_price = stock_data["Close"].iloc[-1]
prev_price = stock_data["Close"].iloc[-2] if len(stock_data) > 1 else latest_price
price_change = latest_price - prev_price
pct_change = (price_change / prev_price * 100) if prev_price else 0

c1, c2, c3 = st.columns(3)
with c1: st.metric("Current Price", f"${latest_price:.2f}", f"{pct_change:+.2f}%")
with c2: st.metric("Latest Bar Range", f"${stock_data['Low'].iloc[-1]:.2f} – ${stock_data['High'].iloc[-1]:.2f}")
with c3: st.metric("Volume", f"{stock_data['Volume'].iloc[-1]:,.0f}")

# ====================== Options: MA & Earnings ===================
days_span = (st.session_state.end_date - st.session_state.start_date).days or 1
if   days_span <= 7:    default_ma = None
elif days_span <= 90:   default_ma = 9
elif days_span <= 365:  default_ma = 50
else:                   default_ma = 100

show_ma = False
ma_data = None
if default_ma is not None:
    show_ma = st.checkbox(f"Show {default_ma}-period Moving Average", value=False)
    if show_ma: ma_data = stock_data["Close"].rolling(window=default_ma).mean()

show_earn = st.checkbox("Earnings calendar", value=False)

# ========================= Price chart ===========================
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=stock_data.index, y=stock_data["Close"], mode="lines",
    name="Close (Adj.)", line=dict(width=2), connectgaps=True,
    hovertemplate="Date: %{x}<br>Price: $%{y:.2f}<extra></extra>",
))
if show_ma and ma_data is not None:
    fig.add_trace(go.Scatter(
        x=stock_data.index, y=ma_data, mode="lines",
        name=f"{default_ma}-period MA", line=dict(width=2, dash="dash"),
        connectgaps=True, hovertemplate="Date: %{x}<br>MA: $%{y:.2f}<extra></extra>",
))
if show_earn:
    earn_df = fetch_earnings_df(TICKER)
    if not earn_df.empty:
        mask = ((earn_df["Earnings Date"] >= st.session_state.start_date) &
                (earn_df["Earnings Date"] <= st.session_state.end_date))
        earn_win = earn_df.loc[mask].copy()
        if not earn_win.empty:
            daily_close = stock_data["Close"].resample("D").last().ffill()
            xs, ys, hovers = [], [], []
            for _, r in earn_win.iterrows():
                d_date = r["Earnings Date"]; d = pd.Timestamp(d_date)
                y = float(daily_close.asof(d)) if not daily_close.empty else float(stock_data["Close"].iloc[-1])
                est = r.get("EPS Estimate"); rep = r.get("Reported EPS"); spr = r.get("Surprise(%)")
                if (spr is None or pd.isna(spr)) and (pd.notna(est) and pd.notna(rep) and est):
                    spr = (rep / est - 1.0) * 100.0
                hover_text = (
                    f"Earnings: {d_date:%Y-%m-%d}<br>Est EPS: {est if pd.notna(est) else '—'}"
                    f"<br>Actual EPS: {rep if pd.notna(rep) else '—'}"
                    + (f"<br>Surprise: {spr:.2f}%"
                       if spr is not None and pd.notna(spr) else "")
                )
                xs.append(d); ys.append(y); hovers.append(hover_text)
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers", name="Earnings",
                marker=dict(size=9, color="crimson", symbol="diamond", line=dict(width=1)),
                hovertemplate="%{text}<extra></extra>", text=hovers,
            ))

rangebreaks = []
if days_span <= 120:
    rangebreaks = [dict(bounds=["sat","mon"]), dict(bounds=[16,9.5], pattern="hour")]
fig.update_layout(
    title=f"{TICKER} — {st.session_state.start_date:%Y-%m-%d} → {st.session_state.end_date:%Y-%m-%d}",
    xaxis_title="Date", yaxis_title="Price ($)", height=520,
    template="plotly_white", hovermode="x unified",
    yaxis=dict(zeroline=False, rangemode="normal", tickprefix="$", separatethousands=True),
    xaxis=dict(rangeslider=dict(visible=days_span > 7), rangebreaks=rangebreaks),
)
st.plotly_chart(fig, use_container_width=True)

# ======================== High/Low readout =======================
hi_ts = stock_data["Close"].idxmax()
lo_ts = stock_data["Close"].idxmin()
st.markdown(
    f"<div style='font-size:14px;'><b>High:</b> ${float(stock_data.loc[hi_ts,'Close']):,.2f} "
    f"(<span style='color:#888'>{hi_ts:%Y-%m-%d %H:%M}</span>) &nbsp;|&nbsp; "
    f"<b>Low:</b> ${float(stock_data.loc[lo_ts,'Close']):,.2f} "
    f"(<span style='color:#888'>{lo_ts:%Y-%m-%d %H:%M}</span>)</div>",
    unsafe_allow_html=True
)

# =================================================================
#                     Comparable performance
# =================================================================
st.header("Comparable performance")

# Mapping for the new comparator set
COMPARATORS = {
    "SPX": "^GSPC",
    "QQQ (Tech ETF)": "QQQ",
    "AMZN": "AMZN",
    "GOOGL": "GOOGL",
    "META": "META",
    "NVDA": "NVDA",
    "TSLA": "TSLA",
}

choices = st.multiselect("Compare against (multi-select):", options=list(COMPARATORS.keys()), default=[])

@st.cache_data(ttl=120)
def fetch_close_series(ticker: str, start_d: date, end_d: date, interval: str) -> pd.Series | None:
    df = fetch_stock_data_range(ticker, start_d, end_d, interval)
    if df is None or df.empty: return None
    s = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
        s.index = s.index.tz_convert(None)
    return s

aapl_close = fetch_close_series(TICKER, st.session_state.start_date, st.session_state.end_date, interval)
if aapl_close is None or aapl_close.empty:
    st.warning("No AAPL data for the selected range.")
else:
    base = float(aapl_close.iloc[0])
    df_pct = pd.DataFrame({"AAPL": (aapl_close/base - 1)*100})
    for label in choices:
        sym = COMPARATORS[label]
        s = fetch_close_series(sym, st.session_state.start_date, st.session_state.end_date, interval)
        if s is None or s.empty:
            st.info(f"Could not load {label} data."); continue
        s = s.reindex(df_pct.index).ffill()
        df_pct[label] = (s/float(s.iloc[0]) - 1)*100

    pfig = go.Figure()
    pfig.add_trace(go.Scatter(x=df_pct.index, y=df_pct["AAPL"], mode="lines",
                              name="AAPL", connectgaps=True,
                              hovertemplate="Date: %{x}<br>Change: %{y:.2f}%<extra></extra>"))
    for label in [c for c in df_pct.columns if c != "AAPL"]:
        pfig.add_trace(go.Scatter(x=df_pct.index, y=df_pct[label], mode="lines",
                                  name=label, connectgaps=True,
                                  hovertemplate="Date: %{x}<br>Change: %{y:.2f}%<extra></extra>"))
    rangebreaks2 = []
    if days_span <= 120:
        rangebreaks2 = [dict(bounds=["sat","mon"]), dict(bounds=[16,9.5], pattern="hour")]
    pfig.update_layout(
        title=f"Since {st.session_state.start_date:%Y-%m-%d} (cumulative % change)",
        xaxis_title="Date", yaxis_title="Change (%)",
        template="plotly_white", hovermode="x unified", height=520,
        yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor="#aaa", ticksuffix="%"),
        xaxis=dict(rangeslider=dict(visible=False), rangebreaks=rangebreaks2),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(pfig, use_container_width=True)

    st.caption("SPX→^GSPC, ‘QQQ (Tech ETF)’→QQQ. All series rebased to 0% on the start date; adjusted closes used.")

# =================================================================
#                 Risk-neutral density (3D) surface — AAPL options
# =================================================================
st.header("Risk-neutral density (3D)")

cA, cB, cC, cD = st.columns([1.4,1.4,1.4,2.2])
rf_pct = cA.number_input("Risk-free rate (annual, %)", value=4.0, step=0.25, min_value=0.0, max_value=15.0)
n_exp = int(cB.slider("Expiries to include", min_value=2, max_value=12, value=6))
n_strikes = int(cC.slider("Strike grid size", min_value=25, max_value=200, value=100))
smooth = cD.checkbox("Light smoothing", value=True, help="3-pt rolling mean before ∂²/∂K² to reduce noise.")

@st.cache_data(ttl=300)
def get_rnd_surface(ticker: str, n_expiries: int, nK: int, r_annual: float):
    """
    Build strike×maturity surface of risk-neutral density from call prices.
    Uses a rolling quadratic fit for C(K) to get a stable second derivative.
    Returns:
        K_grid (nK,), T_arr (nExp,), Z (nExp x nK)  where Z = q(K,T) normalized to [0,1].
    """
    t = yf.Ticker(ticker)
    expiries_raw = getattr(t, "options", [])
    if not expiries_raw:
        return None

    # choose nearest future expiries
    exp_dt = []
    for x in expiries_raw:
        try:
            d = pd.to_datetime(x).date()
            if d >= date.today():
                exp_dt.append(d)
        except Exception:
            continue
    exp_dt = sorted(exp_dt)[:n_expiries]
    if not exp_dt:
        return None

    # spot (for strike window)
    spot_df = t.history(period="5d", interval="1d", auto_adjust=True)
    S0 = float(spot_df["Close"].iloc[-1]) if not spot_df.empty else None

    # ---- collect calls per expiry with robust price fallback
    K_all, chains = [], {}
    for ed in exp_dt:
        try:
            ch = t.option_chain(pd.to_datetime(ed).strftime("%Y-%m-%d"))
        except Exception:
            continue
        if ch is None or ch.calls is None or ch.calls.empty:
            continue
        calls = ch.calls.copy()

        # robust price: last -> mid -> mark -> ask
        price = calls.get("lastPrice")
        if price is None or price.isna().all() or (price <= 0).all():
            bid = calls.get("bid", pd.Series(dtype=float)).fillna(0.0)
            ask = calls.get("ask", pd.Series(dtype=float)).fillna(0.0)
            price = (bid + ask) / 2.0
        if price is None or price.isna().all() or (price <= 0).all():
            mark = calls.get("mark")
            if mark is not None and not mark.isna().all():
                price = mark
        if price is None or price.isna().all() or (price <= 0).all():
            price = calls.get("ask", pd.Series(dtype=float))

        df = pd.DataFrame({
            "strike": pd.to_numeric(calls["strike"], errors="coerce"),
            "callPrice": pd.to_numeric(price, errors="coerce")
        }).dropna()
        df = df[df["callPrice"] > 0].sort_values("strike")
        if df.empty:
            continue

        # remove extreme price outliers (helps fit stability)
        lo, hi = df["callPrice"].quantile([0.01, 0.99])
        df = df[(df["callPrice"] >= lo) & (df["callPrice"] <= hi)]
        if df.empty:
            continue

        chains[ed] = df
        K_all.extend(df["strike"].tolist())

    if not chains:
        return None

    # ---- adaptive, wide strike window (prevents overly tight ranges)
    Kmin = float(np.percentile(K_all, 1))
    Kmax = float(np.percentile(K_all, 99))
    if S0:
        Kmin = max(Kmin, 0.3 * S0)
        Kmax = min(Kmax, 2.0 * S0)
    if Kmax <= Kmin:
        return None
    K_grid = np.linspace(Kmin, Kmax, nK)

    # ---- helper: rolling quadratic fit to get C''(K)
    def quad_second_derivative(K, C, K_eval, win=7):
        """
        For each K_eval[i], fit y = a*K^2 + b*K + c on a local window around K_eval[i].
        Return 2*a at each eval point. Uses least squares; win must be odd.
        """
        n = len(K_eval)
        out = np.zeros(n)
        half = max(3, win//2)  # ensure enough points
        for i in range(n):
            # pick nearest points from original data (not the grid)
            # use indices by distance to K_eval[i]
            idx = np.argsort(np.abs(K - K_eval[i]))[:max(win, 5)]
            Kw = K[idx]; Cw = C[idx]
            # design matrix for quadratic
            A = np.vstack([Kw**2, Kw, np.ones_like(Kw)]).T
            try:
                coeffs, *_ = np.linalg.lstsq(A, Cw, rcond=None)
                a = coeffs[0]
                out[i] = 2.0 * a
            except Exception:
                out[i] = 0.0
        return out

    # ---- build Z rows per expiry
    r = float(r_annual) / 100.0
    T_list, Z_rows = [], []
    for ed, df in chains.items():
        Kraw = df["strike"].to_numpy()
        Craw = df["callPrice"].to_numpy()

        # ensure strictly increasing K
        order = np.argsort(Kraw)
        Kraw = Kraw[order]; Craw = Craw[order]

        # light pre-smooth on raw calls (median filter style)
        if len(Craw) >= 5:
            Craw = pd.Series(Craw).rolling(5, center=True, min_periods=1).median().to_numpy()

        # compute second derivative via rolling quadratic fit (on raw points!)
        Cpp_grid = quad_second_derivative(Kraw, Craw, K_grid, win= nine if len(Kraw)>=9 else 7)  # noqa

        T = max((ed - date.today()).days, 1) / 365.25
        q = np.exp(r * T) * Cpp_grid
        q = np.clip(q, a_min=0.0, a_max=None)  # BL density must be ≥ 0
        Z_rows.append(q)
        T_list.append(T)

    if not Z_rows:
        return None

    Z = np.array(Z_rows)
    T_arr = np.array(T_list)

    # sort by maturity
    order = np.argsort(T_arr)
    T_arr = T_arr[order]; Z = Z[order, :]

    # normalize (avoid flat color when values are tiny)
    zmax = float(np.nanmax(Z))
    if zmax > 0:
        Z = Z / zmax

    # need at least a 2x2 surface
    if Z.shape[0] < 2 or Z.shape[1] < 2:
        return None

    return K_grid, T_arr, Z
