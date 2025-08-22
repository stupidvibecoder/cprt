import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from datetime import date, timedelta

# ============================== Page ==============================
st.set_page_config(page_title="Copart (CPRT) Stock Chart", layout="wide")
st.title("Stock Price Chart")

TICKER = "CPRT"
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

COMPARATORS = {"SPX": "^GSPC", "RBA": "RBA", "S5INDU": "XLI"}

choices = st.multiselect("Compare against (multi-select):", options=list(COMPARATORS.keys()), default=[])

@st.cache_data(ttl=120)
def fetch_close_series(ticker: str, start_d: date, end_d: date, interval: str) -> pd.Series | None:
    df = fetch_stock_data_range(ticker, start_d, end_d, interval)
    if df is None or df.empty: return None
    s = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
        s.index = s.index.tz_convert(None)
    return s

cprt_close = fetch_close_series(TICKER, st.session_state.start_date, st.session_state.end_date, interval)
if cprt_close is None or cprt_close.empty:
    st.warning("No CPRT data for the selected range.")
else:
    base = float(cprt_close.iloc[0])
    df_pct = pd.DataFrame({"CPRT": (cprt_close/base - 1)*100})
    for label in choices:
        sym = COMPARATORS[label]
        s = fetch_close_series(sym, st.session_state.start_date, st.session_state.end_date, interval)
        if s is None or s.empty:
            st.info(f"Could not load {label} data."); continue
        s = s.reindex(df_pct.index).ffill()
        df_pct[label] = (s/float(s.iloc[0]) - 1)*100

    pfig = go.Figure()
    pfig.add_trace(go.Scatter(x=df_pct.index, y=df_pct["CPRT"], mode="lines",
                              name="CPRT", connectgaps=True,
                              hovertemplate="Date: %{x}<br>Change: %{y:.2f}%<extra></extra>"))
    for label in [c for c in df_pct.columns if c != "CPRT"]:
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
    st.caption("Notes: SPX→^GSPC (S&P 500), S5INDU→XLI ETF proxy. Series rebased to 0% on the start date.")

# =================================================================
#                 Risk-neutral density (3D) surface
# =================================================================
st.header("Risk-neutral density (3D)")

cA, cB, cC, cD = st.columns([1.4,1.4,1.4,2.2])
rf_pct = cA.number_input("Risk-free rate (annual, %)", value=4.0, step=0.25, min_value=0.0, max_value=15.0)
n_exp = int(cB.slider("Expiries to include", min_value=2, max_value=8, value=5))
n_strikes = int(cC.slider("Strike grid size", min_value=25, max_value=200, value=80))
smooth = cD.checkbox("Light smoothing", value=True, help="3-pt rolling mean before ∂²/∂K² to reduce noise.")

@st.cache_data(ttl=300)
def get_rnd_surface(ticker: str, n_expiries: int, nK: int, r_annual: float):
    t = yf.Ticker(ticker)
    expiries_raw = getattr(t, "options", [])
    if not expiries_raw: return None

    exp_dt = sorted([pd.to_datetime(x).date() for x in expiries_raw if pd.to_datetime(x).date() >= today])[:n_expiries]
    if not exp_dt: return None

    spot_df = t.history(period="5d", interval="1d", auto_adjust=True)
    S0 = float(spot_df["Close"].iloc[-1]) if not spot_df.empty else None

    # collect per-expiry call data and global strike range
    K_all, chains = [], {}
    for ed in exp_dt:
        try:
            ch = t.option_chain(pd.to_datetime(ed).strftime("%Y-%m-%d"))
        except Exception:
            continue
        if ch is None or ch.calls is None or ch.calls.empty: continue
        calls = ch.calls.copy()

        # choose a robust price: lastPrice; if bad, mid = (bid+ask)/2; finally use 'ask' as last resort
        price = calls.get("lastPrice")
        if price is None or price.isna().all():
            bid = calls.get("bid", pd.Series(dtype=float)).fillna(0.0)
            ask = calls.get("ask", pd.Series(dtype=float)).fillna(0.0)
            mid = (bid + ask) / 2.0
            price = mid
            if price.isna().all() or (price <= 0).all():
                price = calls.get("ask", pd.Series(dtype=float))

        calls = pd.DataFrame({
            "strike": pd.to_numeric(calls["strike"], errors="coerce"),
            "callPrice": pd.to_numeric(price, errors="coerce"),
        }).dropna()
        calls = calls[calls["callPrice"] > 0].sort_values("strike")
        if calls.empty: continue

        chains[ed] = calls
        K_all.extend(calls["strike"].tolist())

    if not chains: return None

    # reasonable strike window
    Kmin = float(np.percentile(K_all, 5))
    Kmax = float(np.percentile(K_all, 95))
    if S0:
        Kmin = max(Kmin, 0.5 * S0)
        Kmax = min(Kmax, 1.6 * S0)
    if Kmax <= Kmin: return None
    K_grid = np.linspace(Kmin, Kmax, nK)

    T_list = []
    Z = []  # density rows per expiry (len = n_expiries, nK columns)
    r = float(r_annual) / 100.0

    for ed, calls in chains.items():
        # interpolate call prices on common K grid
        Ck = np.interp(K_grid, calls["strike"].to_numpy(), calls["callPrice"].to_numpy())
        if smooth and len(Ck) >= 3:
            Ck = pd.Series(Ck).rolling(3, center=True, min_periods=1).mean().to_numpy()

        dK = np.gradient(K_grid)
        dC_dK = np.gradient(Ck, dK)
        d2C_dK2 = np.gradient(dC_dK, dK)

        T = max((ed - today).days, 1) / 365.25
        q = np.exp(r * T) * d2C_dK2
        q = np.clip(q, a_min=0.0, a_max=None)  # clamp negatives
        T_list.append(T)
        Z.append(q)

    if not Z: return None

    Z = np.array(Z)  # shape: (n_expiries, nK)
    T_arr = np.array(T_list)

    # Sort by maturity just in case
    order = np.argsort(T_arr)
    T_arr = T_arr[order]
    Z = Z[order, :]

    return K_grid, T_arr, Z

with st.spinner("Estimating risk-neutral density from the options chain…"):
    rnd = get_rnd_surface(TICKER, n_exp, n_strikes, rf_pct)

if rnd is None:
    st.warning("Could not build the RND surface (no usable option data). Try fewer expiries, a smaller grid, or different market hours.")
else:
    K_grid, T_arr, Z = rnd  # x=strike grid, y=maturity list, z-matrix = density

    # Build mesh for go.Surface: x(K) × y(T) -> z(density)
    X, Y = np.meshgrid(K_grid, T_arr)

    # Color scale: blue (low) → dark red (high)
    colorscale = [
        [0.00, "#2c7bb6"],
        [0.25, "#91bfdb"],
        [0.50, "#ffffbf"],
        [0.75, "#fdae61"],
        [1.00, "#d7191c"],
    ]
    zmin = float(np.nanmin(Z))
    zmax = float(np.nanmax(Z) if np.nanmax(Z) > 0 else 1.0)

    surf = go.Surface(
        x=X, y=Y, z=Z,
        colorscale=colorscale, cmin=zmin, cmax=zmax,
        showscale=True, colorbar=dict(title="Density"),
        contours = dict(z=dict(show=True, usecolormap=True, highlightcolor="black", project_z=True)),
        opacity=0.95,
    )

    fig3d = go.Figure(data=[surf])
    fig3d.update_layout(
        title="Risk-neutral density surface — x: Strike, y: Maturity (years), z: Density",
        scene=dict(
            xaxis_title="Strike (K)",
            yaxis_title="Maturity (years)",
            zaxis_title="Risk-neutral density q(K,T)",
        ),
        height=700, template="plotly_white", margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(fig3d, use_container_width=True)

    st.caption(
        "RND estimated via Breeden–Litzenberger: q(K,T) = e^{rT} * ∂²C/∂K². "
        "We interpolate call prices across a common strike grid per expiry, apply a light smoothing (optional), "
        "and clamp negative densities to zero to reduce numerical artifacts."
    )
