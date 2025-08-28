# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from datetime import date, timedelta

# ============================== Page ==============================
TICKER = "CPRT"
TODAY = date.today()
st.set_page_config(page_title=f"{TICKER} Dashboard", layout="wide")
st.title("Stock Price Chart")

# ====================== Shared helpers ======================
def choose_interval(s: date, e: date) -> str:
    d = (e - s).days or 1
    if d <= 7: return "5m"
    if d <= 60: return "30m"
    if d <= 365: return "1d"
    if d <= 365*5: return "1wk"
    return "1mo"

@st.cache_data(ttl=120)
def fetch_stock_data_range(ticker: str, s: date, e: date, interval: str):
    t = yf.Ticker(ticker)
    df = t.history(
        start=pd.Timestamp(s),
        end=pd.Timestamp(e + timedelta(days=1)),
        interval=interval,
        auto_adjust=True,
        actions=False,
    )
    if df is None or df.empty:
        df = yf.download(
            ticker,
            start=s,
            end=e + timedelta(days=1),
            interval=interval,
            auto_adjust=True,
            progress=False,
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
    try:
        t = yf.Ticker(ticker)
        df = t.get_earnings_dates(limit=60)
        if df is None or df.empty:
            return pd.DataFrame()
        if df.index.name and "date" in str(df.index.name).lower():
            df = df.reset_index()

        rename = {}
        for c in df.columns:
            lc = str(c).lower()
            if "earn" in lc and "date" in lc: rename[c] = "Earnings Date"
            elif "estimate" in lc and "eps" in lc: rename[c] = "EPS Estimate"
            elif ("reported" in lc or "actual" in lc) and "eps" in lc: rename[c] = "Reported EPS"
            elif "surprise" in lc: rename[c] = "Surprise(%)"
        df = df.rename(columns=rename)

        if "Earnings Date" not in df.columns:
            if "Date" in df.columns:
                df = df.rename(columns={"Date": "Earnings Date"})
            else:
                return pd.DataFrame()

        df["Earnings Date"] = pd.to_datetime(df["Earnings Date"], errors="coerce").dt.date
        for c in ("EPS Estimate", "Reported EPS", "Surprise(%)"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["Earnings Date"])
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=120)
def fetch_close_series(sym: str, s: date, e: date, interval: str):
    df = fetch_stock_data_range(sym, s, e, interval)
    if df is None or df.empty:
        return None
    srs = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if isinstance(srs.index, pd.DatetimeIndex) and srs.index.tz is not None:
        srs.index = srs.index.tz_convert(None)
    return srs

# ====================== PRICE CHART (Section 1) ======================
if "start_date" not in st.session_state:
    st.session_state.start_date = TODAY - timedelta(days=30)
if "end_date" not in st.session_state:
    st.session_state.end_date = TODAY

# quick presets + date inputs on one line
row = st.columns([1,1,1,1,1,1,1,1,2.3,2.3])
presets = [("1D",1),("5D",5),("1M",30),("3M",90),("6M",180),("1Y",365),("5Y",365*5),("YTD","ytd")]
clicked = None
for i,(label,span) in enumerate(presets):
    if row[i].button(label): clicked = (label,span)
if clicked:
    _, span = clicked
    if span == "ytd":
        st.session_state.start_date = date(TODAY.year,1,1); st.session_state.end_date = TODAY
    else:
        st.session_state.end_date = TODAY
        st.session_state.start_date = TODAY - timedelta(days=span)

start_input = row[8].date_input("Start date", value=st.session_state.start_date,
                                min_value=date(1990,1,1), max_value=TODAY, label_visibility="collapsed")
end_input = row[9].date_input("End date", value=st.session_state.end_date,
                              min_value=start_input, max_value=TODAY, label_visibility="collapsed")
if start_input > end_input:
    start_input, end_input = end_input, start_input
st.session_state.start_date, st.session_state.end_date = start_input, end_input

interval = choose_interval(start_input, end_input)

with st.spinner(f"Loading {TICKER} {interval} data…"):
    stock_data = fetch_stock_data_range(TICKER, start_input, end_input, interval)
if stock_data is None or stock_data.empty:
    st.error("❌ Unable to fetch stock data for the selected range."); st.stop()

# top metrics
latest = stock_data["Close"].iloc[-1]
prev   = stock_data["Close"].iloc[-2] if len(stock_data)>1 else latest
pct    = ((latest - prev)/prev*100) if prev else 0
c1,c2,c3 = st.columns(3)
with c1: st.metric("Current Price", f"${latest:.2f}", f"{pct:+.2f}%")
with c2: st.metric("Latest Bar Range", f"${stock_data['Low'].iloc[-1]:.2f} – ${stock_data['High'].iloc[-1]:.2f}")
with c3: st.metric("Volume", f"{stock_data['Volume'].iloc[-1]:,.0f}")

# MA + earnings opts
span_days = (end_input - start_input).days or 1
if   span_days <= 7:  default_ma = None
elif span_days <= 90: default_ma = 9
elif span_days <= 365: default_ma = 50
else: default_ma = 100

show_ma = show_earn = False
ma_data = None
if default_ma is not None:
    show_ma = st.checkbox(f"Show {default_ma}-period Moving Average", value=False)
    if show_ma: ma_data = stock_data["Close"].rolling(window=default_ma).mean()
show_earn = st.checkbox("Earnings calendar", value=False)

# price chart
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=stock_data.index, y=stock_data["Close"], mode="lines",
    name="Close (Adj.)", line=dict(width=2), connectgaps=True,
    hovertemplate="Date: %{x}<br>Price: $%{y:.2f}<extra></extra>"
))
if show_ma and ma_data is not None:
    fig.add_trace(go.Scatter(
        x=stock_data.index, y=ma_data, mode="lines",
        name=f"{default_ma}-period MA", line=dict(width=2, dash="dash"),
        connectgaps=True, hovertemplate="Date: %{x}<br>MA: $%{y:.2f}<extra></extra>"
    ))
if show_earn:
    earn_df = fetch_earnings_df(TICKER)
    if not earn_df.empty:
        m = ((earn_df["Earnings Date"]>=start_input) & (earn_df["Earnings Date"]<=end_input))
        ewin = earn_df.loc[m].copy()
        if not ewin.empty:
            daily_close = stock_data["Close"].resample("D").last().ffill()
            xs,ys,txt = [],[],[]
            for _,r in ewin.iterrows():
                d = pd.Timestamp(r["Earnings Date"])
                y = float(daily_close.asof(d)) if not daily_close.empty else float(stock_data["Close"].iloc[-1])
                est,rep,spr = r.get("EPS Estimate"), r.get("Reported EPS"), r.get("Surprise(%)")
                if (spr is None or pd.isna(spr)) and (pd.notna(est) and pd.notna(rep) and est):
                    spr = (rep/est - 1.0)*100.0
                hover = f"Earnings: {d:%Y-%m-%d}<br>Est EPS: {est if pd.notna(est) else '—'}<br>Actual EPS: {rep if pd.notna(rep) else '—'}"
                if spr is not None and pd.notna(spr): hover += f"<br>Surprise: {spr:.2f}%"
                xs.append(d); ys.append(y); txt.append(hover)
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers", name="Earnings",
                marker=dict(size=9, color="crimson", symbol="diamond", line=dict(width=1)),
                hovertemplate="%{text}<extra></extra>", text=txt
            ))
rb = []
if span_days <= 120:
    rb = [dict(bounds=["sat","mon"]), dict(bounds=[16,9.5], pattern="hour")]
fig.update_layout(
    title=f"{TICKER} — {start_input:%Y-%m-%d} → {end_input:%Y-%m-%d}",
    xaxis_title="Date", yaxis_title="Price ($)",
    height=520, template="plotly_white", hovermode="x unified",
    yaxis=dict(zeroline=False, tickprefix="$", separatethousands=True),
    xaxis=dict(rangeslider=dict(visible=span_days>7), rangebreaks=rb),
)
st.plotly_chart(fig, use_container_width=True)

# high/low readout
hi_ts = stock_data["Close"].idxmax(); lo_ts = stock_data["Close"].idxmin()
st.markdown(
    f"<div style='font-size:14px;'><b>High:</b> ${float(stock_data.loc[hi_ts,'Close']):,.2f} "
    f"(<span style='color:#888'>{hi_ts:%Y-%m-%d %H:%M}</span>) &nbsp;|&nbsp; "
    f"<b>Low:</b> ${float(stock_data.loc[lo_ts,'Close']):,.2f} "
    f"(<span style='color:#888'>{lo_ts:%Y-%m-%d %H:%M}</span>)</div>",
    unsafe_allow_html=True
)

# ================== COMPARABLE PERFORMANCE (Section 2) ==================
st.header("Comparable performance")

COMPARATORS = {
    "SPX": "^GSPC", "OPENLANE": "KAR", "Ritchie Bros.": "RBA",
    "UniFirst": "UNF", "CarMax": "KMX", "Carvana": "CVNA", "Industrials ETF": "IYJ",
}
choices = st.multiselect("Compare against (multi-select):", options=list(COMPARATORS.keys()), default=[])

interval2 = choose_interval(st.session_state.start_date, st.session_state.end_date)
base_close = fetch_close_series(TICKER, st.session_state.start_date, st.session_state.end_date, interval2)
if base_close is None or base_close.empty:
    st.warning("No data for the selected range.")
else:
    base_val = float(base_close.iloc[0])
    df_pct = pd.DataFrame({"CPRT": (base_close/base_val - 1)*100})
    for lab in choices:
        sym = COMPARATORS[lab]
        srs = fetch_close_series(sym, st.session_state.start_date, st.session_state.end_date, interval2)
        if srs is None or srs.empty:
            st.info(f"Could not load {lab}."); continue
        srs = srs.reindex(df_pct.index).ffill()
        df_pct[lab] = (srs/float(srs.iloc[0]) - 1)*100

    pfig = go.Figure()
    pfig.add_trace(go.Scatter(
        x=df_pct.index, y=df_pct["CPRT"], mode="lines", name="CPRT", connectgaps=True,
        hovertemplate="Date: %{x}<br>Change: %{y:.2f}%<extra></extra>"
    ))
    for lab in [c for c in df_pct.columns if c!="CPRT"]:
        pfig.add_trace(go.Scatter(
            x=df_pct.index, y=df_pct[lab], mode="lines", name=lab, connectgaps=True,
            hovertemplate="Date: %{x}<br>Change: %{y:.2f}%<extra></extra>"
        ))
    span_days2 = (st.session_state.end_date - st.session_state.start_date).days or 1
    rb2 = []
    if span_days2 <= 120:
        rb2 = [dict(bounds=["sat","mon"]), dict(bounds=[16,9.5], pattern="hour")]
    pfig.update_layout(
        title=f"Since {st.session_state.start_date:%Y-%m-%d} (cumulative % change)",
        xaxis_title="Date", yaxis_title="Change (%)",
        template="plotly_white", hovermode="x unified", height=520,
        yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor="#aaa", ticksuffix="%"),
        xaxis=dict(rangeslider=dict(visible=False), rangebreaks=rb2),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(pfig, use_container_width=True)

# ========== RISK-NEUTRAL DENSITY (3D) + 2D SLICES (Section 3) ==========
st.header("Risk-neutral density (3D)")
cA,cB,cC,cD = st.columns([1.4,1.4,1.4,2.2])
rf_pct     = cA.number_input("Risk-free rate (annual, %)", value=4.0, step=0.25, min_value=0.0, max_value=15.0)
n_exp      = int(cB.slider("Expiries to include", min_value=2, max_value=12, value=6))
n_strikes  = int(cC.slider("Strike grid size", min_value=25, max_value=200, value=120))
smooth     = cD.checkbox("Light smoothing", value=True, help="Pre-smooth raw call quotes before curvature.")

@st.cache_data(ttl=300)
def get_rnd_surface(ticker: str, n_expiries: int, nK: int, r_annual: float):
    """
    Build strike×maturity surface of risk-neutral density from call prices.
    Returns: K_grid, T_days (maturity in days), Z (norm), exp_dates (list of date)
    """
    t = yf.Ticker(ticker)
    expiries_raw = getattr(t, "options", [])
    if not expiries_raw: return None

    # future expiries
    exp_dt = []
    for x in expiries_raw:
        try:
            d = pd.to_datetime(x).date()
            if d >= TODAY: exp_dt.append(d)
        except Exception:
            continue
    exp_dt = sorted(exp_dt)[:n_expiries]
    if not exp_dt: return None

    # spot
    spot_df = t.history(period="5d", interval="1d", auto_adjust=True)
    S0 = float(spot_df["Close"].iloc[-1]) if not spot_df.empty else None

    # collect calls
    K_all, chains = [], {}
    for ed in exp_dt:
        try:
            ch = t.option_chain(pd.to_datetime(ed).strftime("%Y-%m-%d"))
        except Exception:
            continue
        if ch is None or ch.calls is None or ch.calls.empty: continue
        calls = ch.calls.copy()

        # robust call price
        price = calls.get("lastPrice")
        if price is None or price.isna().all() or (price <= 0).all():
            bid = calls.get("bid", pd.Series(dtype=float)).fillna(0.0)
            ask = calls.get("ask", pd.Series(dtype=float)).fillna(0.0)
            price = (bid + ask)/2.0
        if price is None or price.isna().all() or (price <= 0).all():
            mark = calls.get("mark")
            if mark is not None and not mark.isna().all(): price = mark
        if price is None or price.isna().all() or (price <= 0).all():
            price = calls.get("ask", pd.Series(dtype=float))

        df = pd.DataFrame({
            "strike": pd.to_numeric(calls["strike"], errors="coerce"),
            "callPrice": pd.to_numeric(price, errors="coerce")
        }).dropna()
        df = df[df["callPrice"] > 0].sort_values("strike")
        if df.empty or len(df) < 6: continue

        # outlier trim
        lo, hi = df["callPrice"].quantile([0.01, 0.99])
        df = df[(df["callPrice"] >= lo) & (df["callPrice"] <= hi)]
        if df.empty or len(df) < 6: continue

        chains[ed] = df
        K_all.extend(df["strike"].tolist())

    if not chains: return None

    # strike grid
    Kmin = float(np.percentile(K_all, 1))
    Kmax = float(np.percentile(K_all, 99))
    if S0:
        Kmin = max(Kmin, 0.2 * S0)
        Kmax = min(Kmax, 3.0 * S0)
    if Kmax <= Kmin: return None
    K_grid = np.linspace(Kmin, Kmax, nK)

    # local cubic second derivative at K_eval
    def local_cubic_dd(Kraw, Craw, K_eval, win=11):
        win = int(win) if int(win) % 2 == 1 else int(win)+1
        out = np.zeros(len(K_eval))
        for i, Ke in enumerate(K_eval):
            idx = np.argsort(np.abs(Kraw - Ke))[:max(win, 7)]
            Kw, Cw = Kraw[idx], Craw[idx]
            A = np.vstack([Kw**3, Kw**2, Kw, np.ones_like(Kw)]).T
            try:
                a3, a2, _, _ = np.linalg.lstsq(A, Cw, rcond=None)[0]
                out[i] = 6.0 * a3 * Ke + 2.0 * a2
            except Exception:
                out[i] = 0.0
        return out

    r = float(r_annual) / 100.0
    T_days_list, Z_rows, exp_dates = [], [], []
    for ed, df in chains.items():
        Kraw = df["strike"].to_numpy()
        Craw = df["callPrice"].to_numpy()
        order = np.argsort(Kraw); Kraw = Kraw[order]; Craw = Craw[order]

        # optional pre-smooth
        if smooth and len(Craw) >= 5:
            Craw = pd.Series(Craw).rolling(5, center=True, min_periods=1).median().to_numpy()

        Cpp = local_cubic_dd(Kraw, Craw, K_grid, win=11 if len(Kraw)>=11 else 9)
        if np.allclose(Cpp, 0, atol=1e-12):
            Ck = np.interp(K_grid, Kraw, Craw)
            dK = K_grid[1] - K_grid[0]
            Cpp_fd = np.zeros_like(Ck)
            Cpp_fd[1:-1] = (Ck[:-2] - 2*Ck[1:-1] + Ck[2:]) / (dK**2)
            Cpp = Cpp_fd

        T_days = max((ed - TODAY).days, 1)   # maturity in DAYS for axis
        T_years = T_days / 365.25            # years for discounting only
        q = np.exp(r*T_years) * Cpp
        q = np.clip(q, 0.0, None)

        Z_rows.append(q); T_days_list.append(T_days); exp_dates.append(ed)

    if not Z_rows:
        return None

    Z = np.array(Z_rows)
    T_days_arr = np.array(T_days_list)
    exp_dates = np.array(exp_dates, dtype="object")

    order = np.argsort(T_days_arr)
    Z = Z[order, :]; T_days_arr = T_days_arr[order]; exp_dates = exp_dates[order]
    zmax = float(np.nanmax(Z))
    Z = Z / zmax if zmax > 0 else np.zeros_like(Z)
    if Z.shape[0] < 2 or Z.shape[1] < 2:
        return None
    return K_grid, T_days_arr, Z, exp_dates

with st.spinner(f"Estimating risk-neutral density from {TICKER} options…"):
    rnd = get_rnd_surface(TICKER, n_exp, n_strikes, rf_pct)

if rnd is None:
    st.warning("Could not build the RND surface (insufficient option data). Try fewer expiries or a smaller grid.")
else:
    K_grid, T_days, Z, exp_dates = rnd
    X, Y = np.meshgrid(K_grid, T_days)

    colorscale = [
        [0.00, "#2c7bb6"], [0.25, "#91bfdb"],
        [0.50, "#ffffbf"], [0.75, "#fdae61"],
        [1.00, "#d7191c"],
    ]
    surf = go.Surface(
        x=X, y=Y, z=Z, colorscale=colorscale, cmin=0.0, cmax=1.0,
        showscale=True, colorbar=dict(title="Density (norm)"),
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="black", project_z=True)),
        opacity=0.97,
    )
    fig3d = go.Figure(data=[surf])
    fig3d.update_layout(
        title="Risk-neutral density surface — x: Strike, y: Maturity (days), z: Density",
        scene=dict(
            xaxis_title="Strike (K)",
            yaxis_title="Maturity (days)",
            zaxis_title="Risk-neutral density q(K,T) (normalized)",
        ),
        height=700, template="plotly_white", margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(fig3d, use_container_width=True)

    st.caption(
        "RND via Breeden–Litzenberger: q(K,T) = e^{rT} · ∂²C/∂K². "
        "Local cubic fits with butterfly fallback; negatives clipped; densities normalized."
    )

    # -------- Two 2D cross-sections below the surface --------
    st.subheader("RND cross-sections")
    left, right = st.columns(2)

    # Left: maturity (days) vs strike at max density
    with left:
        tf_opts = {
            "All maturities": None, "0–30 days": 30, "0–60 days": 60,
            "0–90 days": 90, "0–180 days": 180, "0–365 days": 365,
        }
        sel_tf = st.selectbox("Timeframe (x-axis filter):", list(tf_opts.keys()), index=0, key="tf_left")
        limit = tf_opts[sel_tf]
        idx_max = np.argmax(Z, axis=1)
        K_star = K_grid[idx_max]
        T_plot = T_days.copy(); K_plot = K_star.copy()
        if limit is not None:
            m = T_plot <= limit
            T_plot, K_plot = T_plot[m], K_plot[m]
        fig_left = go.Figure()
        fig_left.add_trace(go.Scatter(
            x=T_plot, y=K_plot, mode="lines+markers", name="Max-density strike",
            hovertemplate="Maturity: %{x}d<br>Strike: $%{y:.2f}<extra></extra>"
        ))
        fig_left.update_layout(
            title="Option Market Price Estimate",
            xaxis_title="Maturity (days)", yaxis_title="Strike with highest RND (K*)",
            template="plotly_white", height=420
        )
        st.plotly_chart(fig_left, use_container_width=True)

    # Right: strike vs density for a selected expiry
    with right:
        labels = [f"{d} ({int(t)}d)" for d, t in zip(exp_dates, T_days)]
        sel_label = st.selectbox("Expiry to slice:", labels, index=0, key="slice_right")
        sel_idx = labels.index(sel_label)
        z_row = Z[sel_idx, :]
        fig_right = go.Figure()
        fig_right.add_trace(go.Scatter(
            x=K_grid, y=z_row, mode="lines", name=str(exp_dates[sel_idx]),
            hovertemplate="Strike: $%{x:.2f}<br>Density: %{y:.3f}<extra></extra>"
        ))
        i_star = int(np.argmax(z_row))
        fig_right.add_trace(go.Scatter(
            x=[K_grid[i_star]], y=[z_row[i_star]], mode="markers", name="Peak",
            marker=dict(size=9, color="crimson"),
            hovertemplate="Peak<br>Strike: $%{x:.2f}<br>Density: %{y:.3f}<extra></extra>"
        ))
        fig_right.update_layout(
            title=f"RND slice at {exp_dates[sel_idx]}",
            xaxis_title="Strike (K)", yaxis_title="Risk-neutral density (normalized)",
            template="plotly_white", height=420
        )
        st.plotly_chart(fig_right, use_container_width=True)
