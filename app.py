import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from streamlit_plotly_events import plotly_events
from datetime import datetime, timedelta
import numpy as np

# ---------------- Page ----------------
st.set_page_config(page_title="Copart (CPRT) Stock Chart", layout="wide")
st.title("Copart (CPRT) Stock Chart")

# Valid Yahoo period/interval pairs - FIXED
TF = {
    "All Time": ("max", "1mo"),  # Changed from 1wk to 1mo for better reliability
    "5 Years": ("5y", "1wk"),    # Changed from 1d to 1wk
    "1 Year": ("1y", "1d"),
    "6 Months": ("6mo", "1d"),
    "3 Months": ("3mo", "1d"),   # Changed from 1h to 1d
    "1 Month": ("1mo", "1d"),    # Changed from 30m to 1d
    "1 Week": ("5d", "1h"),      # Changed from 7d/5m to 5d/1h
    "1 Day (Intraday)": ("1d", "5m"),
}

label = st.sidebar.selectbox("Select timeframe:", list(TF.keys()))
period, interval = TF[label]
autorefresh = st.sidebar.checkbox("Auto-refresh intraday (30s)", value=False) if "Intraday" in label else False

# ---------------- Data ----------------
@st.cache_data(ttl=60 if "Intraday" in label else 300, show_spinner=False)
def fetch_prices(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Fetch stock prices from Yahoo Finance with better error handling.
    """
    try:
        # Create ticker object
        tkr = yf.Ticker(ticker)
        
        # For debugging
        st.sidebar.write(f"Fetching: {period} / {interval}")
        
        # Try to download data with specific parameters
        if period == "max":
            # For max period, use download with specific date range
            end_date = datetime.now()
            start_date = datetime(2010, 1, 1)  # Reasonable start for most stocks
            df = yf.download(ticker, start=start_date, end=end_date, interval=interval, 
                           auto_adjust=True, progress=False, threads=False)
        else:
            # Use download for better reliability
            df = yf.download(ticker, period=period, interval=interval, 
                           auto_adjust=True, progress=False, threads=False)
        
        if df is None or df.empty:
            st.error(f"No data returned for {ticker} with period={period}, interval={interval}")
            return pd.DataFrame()
        
        # Handle multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Use Close price
        if 'Close' in df.columns:
            price_col = 'Close'
        elif 'close' in df.columns:
            price_col = 'close'
        else:
            st.error(f"No close price column found. Available columns: {list(df.columns)}")
            return pd.DataFrame()
        
        # Create clean dataframe
        result_df = pd.DataFrame()
        result_df['Price'] = df[price_col].astype(float)
        result_df = result_df.dropna()
        
        if result_df.empty:
            st.error("No valid price data after cleaning")
            return pd.DataFrame()
        
        # Ensure index is datetime
        if not isinstance(result_df.index, pd.DatetimeIndex):
            result_df.index = pd.to_datetime(result_df.index)
        
        # Remove timezone if present
        if result_df.index.tz is not None:
            result_df.index = result_df.index.tz_localize(None)
        
        # Add timestamp column
        result_df['ts'] = result_df.index
        
        # Validate prices are reasonable (CPRT typically trades $40-$70)
        price_check = result_df['Price'].describe()
        st.sidebar.markdown("### Price Statistics")
        st.sidebar.write(f"Count: {len(result_df)}")
        st.sidebar.write(f"Mean: ${price_check['mean']:.2f}")
        st.sidebar.write(f"Min: ${price_check['min']:.2f}")
        st.sidebar.write(f"Max: ${price_check['max']:.2f}")
        st.sidebar.write(f"Latest: ${result_df['Price'].iloc[-1]:.2f}")
        
        # Warning if prices seem off
        if price_check['max'] > 200 or price_check['min'] < 10:
            st.sidebar.warning("⚠️ Prices may be incorrect!")
        
        return result_df[['ts', 'Price']]
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        st.sidebar.error(f"Debug: {type(e).__name__}")
        return pd.DataFrame()

# Auto-refresh logic
if autorefresh:
    st.autorefresh(interval=30_000, key="refresh_intraday")

# Fetch data
with st.spinner("Loading CPRT prices…"):
    data = fetch_prices("CPRT", period, interval)

if data.empty:
    st.error("Failed to load data. Please try a different timeframe or refresh the page.")
    
    # Show alternative options
    st.info("💡 Try these timeframes that typically work well:")
    st.write("- 6 Months")
    st.write("- 1 Year") 
    st.write("- 5 Years")
    st.stop()

# ---------------- Plot ----------------
fig = go.Figure()

# Add the price line
fig.add_trace(
    go.Scattergl(
        x=data['ts'],
        y=data['Price'],
        mode='lines',
        name='CPRT Close',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Price: $%{y:.2f}<extra></extra>',
    )
)

# Configure x-axis breaks for intraday charts
rangebreaks = []
if label in ("1 Day (Intraday)", "1 Week"):
    rangebreaks = [
        dict(bounds=["sat", "mon"]),  # Hide weekends
        dict(bounds=[16, 9.5], pattern="hour"),  # Hide non-trading hours
    ]

# Update layout
fig.update_layout(
    title=dict(
        text=f"CPRT - {label}",
        x=0.5,
        xanchor='center'
    ),
    xaxis=dict(
        title="Date/Time",
        rangeslider=dict(visible=False),  # Disable rangeslider for cleaner view
        rangebreaks=rangebreaks,
        type='date',
        showgrid=True,
        gridcolor='rgba(128,128,128,0.2)',
    ),
    yaxis=dict(
        title="Price ($)",
        tickprefix='$',
        tickformat='.2f',
        side='right',  # Move price axis to right side
        showgrid=True,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=False,
    ),
    hovermode='x unified',
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(l=10, r=80, t=80, b=50),
    height=600,
    showlegend=False,
)

# ---------------- Display Chart ----------------
selected = plotly_events(
    fig,
    click_event=False,
    select_event=False,
    hover_event=True,
    override_width="100%",
    override_height=600,
    key="plotly_events"
)

# ---------------- Info Display ----------------
col1, col2, col3, col4 = st.columns(4)

# Current price and change
if len(data) >= 2:
    current = data['Price'].iloc[-1]
    previous = data['Price'].iloc[-2]
    change = current - previous
    pct_change = (change / previous) * 100
    
    with col1:
        st.metric(
            label="Current Price",
            value=f"${current:.2f}",
            delta=f"{pct_change:+.2f}%"
        )
else:
    with col1:
        st.metric(
            label="Current Price",
            value=f"${data['Price'].iloc[-1]:.2f}"
        )

# Period high
with col2:
    high_price = data['Price'].max()
    high_date = data.loc[data['Price'].idxmax(), 'ts']
    st.metric(
        label="Period High",
        value=f"${high_price:.2f}",
        help=f"On {high_date:%Y-%m-%d}"
    )

# Period low
with col3:
    low_price = data['Price'].min()
    low_date = data.loc[data['Price'].idxmin(), 'ts']
    st.metric(
        label="Period Low", 
        value=f"${low_price:.2f}",
        help=f"On {low_date:%Y-%m-%d}"
    )

# Data points
with col4:
    st.metric(
        label="Data Points",
        value=f"{len(data):,}",
        help=f"{interval} intervals"
    )

# Hover info
if selected:
    try:
        hover_date = selected[0]['x']
        hover_price = float(selected[0]['y'])
        st.info(f"**Hover:** {hover_date} — ${hover_price:.2f}")
    except:
        pass

# ---------------- Additional Analysis ----------------
with st.expander("📊 Additional Analysis", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Price Distribution")
        # Simple histogram of prices
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=data['Price'],
            nbinsx=20,
            name='Price Distribution',
            marker_color='lightblue'
        ))
        hist_fig.update_layout(
            xaxis_title="Price ($)",
            yaxis_title="Frequency",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(hist_fig, use_container_width=True)
    
    with col2:
        st.markdown("### Statistics")
        stats_df = pd.DataFrame({
            'Statistic': ['Mean', 'Median', 'Std Dev', 'Range'],
            'Value': [
                f"${data['Price'].mean():.2f}",
                f"${data['Price'].median():.2f}",
                f"${data['Price'].std():.2f}",
                f"${data['Price'].max() - data['Price'].min():.2f}"
            ]
        })
        st.dataframe(stats_df, hide_index=True)

# Footer
st.markdown("---")
st.caption(f"Data from Yahoo Finance • Last updated: {datetime.now():%Y-%m-%d %H:%M:%S}")
st.caption("Note: Prices are adjusted for splits and dividends")
