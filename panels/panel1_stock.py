"""
Fionn/Panel 1.py
================
Panel 1 - AI Stock Performance

Question this panel answers:
    How have AI-related companies performed relative to the wider stock market?

This is ONE self-contained file (data loading + calculations + charts), so it
runs on its own. To try it:

    streamlit run "Fionn/Panel 1.py"        (run from the Group_Project folder)

How the file is organised (read it top to bottom):
    0. Shared data      (company list, price download)
    1. Settings and colours
    2. Pure data functions  (no Streamlit, no charts - just pandas maths)
    3. Chart functions      (turn data into Plotly figures)
    4. render_panel1()      (the function the main app calls)
    5. Local test page      (only runs when you run this file directly)

Integration contract with the main app:
    render_panel1(selected_companies)
    `selected_companies` is a list of TICKERS, e.g. ["NVDA", "MSFT", "PLTR"],
    taken from the single global toggle bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# ===========================================================================
# 0. SHARED DATA  (company list + price download)
# ===========================================================================
# TEAM NOTE: when the dashboard is combined, these pieces should live in ONE
# shared place so every panel uses the same tickers and the same price data.

# ---------------------------------------------------------------------------
# 0a. The company universe (ticker -> display name)
# ---------------------------------------------------------------------------
# These are the 12 proposed companies. app.py builds the global toggle bar
# from this dictionary, so there is exactly ONE place to add/remove a company.
COMPANIES: dict[str, str] = {
    "NVDA": "NVIDIA",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "META": "Meta",
    "AVGO": "Broadcom",
    "AMD": "AMD",
    "TSM": "TSMC",
    "ORCL": "Oracle",
    "PLTR": "Palantir",
    "CRM": "Salesforce",
    "ASML": "ASML",
}

# Market benchmarks (ETFs that track the wider market).
BENCHMARKS: dict[str, str] = {
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq-100 ETF",
}

# How far back we download. 2010 comfortably covers a "10Y" view.
PRICE_HISTORY_START = "2010-01-01"


class DataLoadError(Exception):
    """Raised when price data cannot be downloaded."""


# ---------------------------------------------------------------------------
# 0b. Price download (cached)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60 * 60, show_spinner="Downloading price history from Yahoo Finance...")
def load_prices(tickers: tuple[str, ...], start: str = PRICE_HISTORY_START) -> pd.DataFrame:
    """
    Download daily ADJUSTED closing prices for several tickers.

    Returns a DataFrame:
        index   = trading dates (timezone-naive, sorted, no duplicates)
        columns = tickers
        values  = adjusted close price in US dollars

    What "adjusted" means here
    --------------------------
    We call yfinance with auto_adjust=True. Yahoo then rewrites old prices so
    that STOCK SPLITS and DIVIDENDS are accounted for. Without this, NVIDIA's
    10-for-1 split in 2024 would look like a 90% crash. This is why we can
    compare companies fairly, and it makes returns "total return" style.

    Why this raises an error instead of returning an empty table
    ------------------------------------------------------------
    st.cache_data only caches successful calls. If a download fails and we
    raised nothing, an EMPTY table would be cached for an hour and the
    dashboard would look broken until it expired. Raising avoids that.

    `tickers` must be a tuple (not a list) because Streamlit needs hashable
    arguments to cache a function.
    """
    if not tickers:
        raise DataLoadError("No tickers were requested.")

    try:
        raw = yf.download(
            list(tickers),
            start=start,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as err:  # network down, Yahoo blocked us, etc.
        raise DataLoadError(f"Yahoo Finance request failed: {err}") from err

    if raw is None or raw.empty:
        raise DataLoadError("Yahoo Finance returned no data.")

    # With several tickers, yfinance returns two-level columns:
    #   ("Close", "NVDA"), ("Close", "MSFT"), ("Open", "NVDA"), ...
    # raw["Close"] picks just the closing prices, one column per ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:  # defensive: single-ticker shape
        prices = raw[["Close"]].copy()
        prices.columns = [tickers[0]]

    # Tidy up: plain dates, sorted, no duplicate days, drop dead columns.
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    prices = prices.dropna(axis=1, how="all")

    if prices.empty:
        raise DataLoadError("Downloaded data contained no usable prices.")

    return prices


# ===========================================================================
# 1. SETTINGS AND COLOURS
# ===========================================================================
MAX_COMPANIES = 12
TRADING_DAYS_PER_YEAR = 252          # used to annualise volatility
MAX_3D_POINTS = 350                  # keeps the 3D chart smooth to rotate

PERIOD_OPTIONS = ["1Y", "3Y", "5Y", "10Y", "Since ChatGPT"]
CHATGPT_LAUNCH = pd.Timestamp("2022-11-30")   # public launch of ChatGPT

# Bloomberg-style palette: dark navy with blue and green accents.
NAVY = "#0A1628"
PANEL_BG = "#0F2038"
GRID = "#1E3252"
TEXT = "#E6EDF7"
MUTED = "#8FA3BF"
BLUE = "#3B82F6"
GREEN = "#22C55E"
FONT = "Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif"

# One fixed colour per ticker, so NVDA is the same colour every time you toggle
# companies on and off. Other panels can import this dictionary too.
TICKER_COLOURS = {
    "NVDA": "#4ADE80",
    "MSFT": "#3B82F6",
    "GOOGL": "#F59E0B",
    "AMZN": "#FB923C",
    "META": "#818CF8",
    "AVGO": "#F43F5E",
    "AMD": "#22D3EE",
    "TSM": "#A78BFA",
    "ORCL": "#F87171",
    "PLTR": "#E879F9",
    "CRM": "#2DD4BF",
    "ASML": "#FDE047",
    "SPY": "#F1F5F9",    # benchmarks are white / grey so they stand apart
    "QQQ": "#94A3B8",
}


def _colour(ticker: str) -> str:
    return TICKER_COLOURS.get(ticker, MUTED)


def _display_name(ticker: str) -> str:
    return COMPANIES.get(ticker) or BENCHMARKS.get(ticker) or ticker


# ===========================================================================
# 2. PURE DATA FUNCTIONS (pandas only)
# ===========================================================================
def get_period_start(period: str, last_date: pd.Timestamp) -> pd.Timestamp:
    """Turn a label like '5Y' into the date the window starts."""
    if period == "Since ChatGPT":
        return CHATGPT_LAUNCH
    years = int(period.rstrip("Y"))
    return last_date - pd.DateOffset(years=years)


def rebase_to_100(window: pd.DataFrame) -> pd.DataFrame:
    """
    Index every series to 100 on its FIRST available day in the window.

    Example: a share price that goes 50 -> 75 becomes 100 -> 150.
    This lets a $900 share and a $30 share be compared fairly.

    If a company was listed part-way through the window (e.g. Palantir in
    2020), it starts at 100 on its own first trading day. We never invent
    prices for dates before a company existed.
    """
    series = {}
    for ticker in window.columns:
        s = window[ticker].dropna()
        if s.empty:
            continue
        series[ticker] = s / s.iloc[0] * 100
    return pd.DataFrame(series)


def compute_drawdowns(window: pd.DataFrame) -> pd.DataFrame:
    """
    Drawdown = how far the price is below its highest earlier price.
    0% means "at a record high"; -40% means "40% below the peak so far".
    Returned as fractions (-0.40), not percentages.
    """
    series = {}
    for ticker in window.columns:
        s = window[ticker].dropna()
        if s.empty:
            continue
        series[ticker] = s / s.cummax() - 1
    return pd.DataFrame(series)


def compute_stats(window: pd.DataFrame) -> pd.DataFrame:
    """
    One row per ticker with the key risk / return numbers.

    total return   = last price / first price - 1
    CAGR           = the constant yearly growth rate that gives the same total
                     return (only shown for windows of about a year or more)
    volatility     = standard deviation of daily returns x sqrt(252)
                     (a yearly measure of how bumpy the ride was)
    max drawdown   = the worst peak-to-trough fall inside the window
    """
    rows = {}
    for ticker in window.columns:
        s = window[ticker].dropna()
        if len(s) < 2:
            continue

        total_return = s.iloc[-1] / s.iloc[0] - 1
        years = (s.index[-1] - s.index[0]).days / 365.25
        cagr = (1 + total_return) ** (1 / years) - 1 if years >= 0.9 else np.nan

        daily_returns = s.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        drawdown = s / s.cummax() - 1

        rows[ticker] = {
            "Name": _display_name(ticker),
            "Start date": s.index[0].strftime("%d %b %Y"),
            "Total return (%)": total_return * 100,
            "CAGR (%)": cagr * 100,
            "Volatility (%)": volatility * 100,
            "Max drawdown (%)": drawdown.min() * 100,
            "Current drawdown (%)": drawdown.iloc[-1] * 100,
        }

    stats = pd.DataFrame.from_dict(rows, orient="index")
    stats.index.name = "Ticker"
    return stats


def find_late_starters(window: pd.DataFrame, tickers: list[str], tolerance_days: int = 7) -> dict:
    """Companies whose first price is noticeably later than the window start."""
    window_start = window.index.min()
    late = {}
    for ticker in tickers:
        if ticker not in window.columns:
            continue
        first = window[ticker].first_valid_index()
        if first is not None and (first - window_start).days > tolerance_days:
            late[ticker] = first
    return late


# ===========================================================================
# 3. CHART FUNCTIONS (Plotly)
# ===========================================================================
def _style_2d(fig: go.Figure, height: int) -> go.Figure:
    """Dark terminal styling shared by the 2D charts."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PANEL_BG,
        font=dict(family=FONT, color=TEXT, size=12),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#13294B", bordercolor=BLUE, font=dict(color=TEXT)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, hoverformat="%d %b %Y")
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def build_3d_figure(rebased: pd.DataFrame, order: list[str], log_scale: bool) -> go.Figure:
    """
    3D view:  X = date,  Y = company,  Z = indexed price (start = 100).

    Each company is one line laid out along its own "lane" on the Y axis, so
    the lines never tangle together. A label at the end of each line shows the
    final indexed value, which keeps the chart readable rather than decorative.
    """
    fig = go.Figure()

    # Thin the data so rotating stays smooth. We count back from the LAST row
    # so the most recent day is always included.
    step = max(1, round(len(rebased) / MAX_3D_POINTS))
    positions = np.arange(len(rebased) - 1, -1, -step)[::-1]
    sampled = rebased.iloc[positions]

    for lane, ticker in enumerate(order):
        s = sampled[ticker].dropna()
        if s.empty:
            continue

        is_benchmark = ticker in BENCHMARKS
        date_text = s.index.strftime("%d %b %Y")
        return_text = [f"{value - 100:+.1f}%" for value in s.values]

        fig.add_trace(go.Scatter3d(
            x=s.index,
            y=[lane] * len(s),
            z=s.values,
            mode="lines",
            name=ticker,
            legendgroup=ticker,
            line=dict(color=_colour(ticker), width=4 if is_benchmark else 6),
            customdata=np.column_stack([date_text, return_text]),
            hovertemplate=(
                "<b>" + ticker + "</b> (" + _display_name(ticker) + ")"
                "<br>%{customdata[0]}"
                "<br>Indexed price: %{z:,.1f}"
                "<br>Change since start: %{customdata[1]}"
                "<extra></extra>"
            ),
        ))

        # Label at the end of the line (same legend group, so toggling a
        # company in the legend hides its label too).
        fig.add_trace(go.Scatter3d(
            x=[s.index[-1]],
            y=[lane],
            z=[s.values[-1]],
            mode="markers+text",
            text=[f"{ticker} {s.values[-1]:,.0f}"],
            textposition="top center",
            textfont=dict(color=_colour(ticker), size=11),
            marker=dict(color=_colour(ticker), size=4),
            legendgroup=ticker,
            showlegend=False,
            hoverinfo="skip",
        ))

    axis_style = dict(
        backgroundcolor=PANEL_BG,
        gridcolor=GRID,
        zerolinecolor=GRID,
        showbackground=True,
        color=MUTED,
    )
    fig.update_layout(
        height=540,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=TEXT, size=12),
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0, x=0, font=dict(size=11)),
        hoverlabel=dict(bgcolor="#13294B", bordercolor=BLUE, font=dict(color=TEXT)),
        scene=dict(
            xaxis=dict(title="Date", type="date", **axis_style),
            yaxis=dict(
                title="Company",
                tickmode="array",
                tickvals=list(range(len(order))),
                ticktext=order,
                range=[-0.5, len(order) - 0.5],
                **axis_style,
            ),
            zaxis=dict(
                title="Indexed price (start = 100)",
                type="log" if log_scale else "linear",
                **axis_style,
            ),
            aspectmode="manual",
            aspectratio=dict(x=2.2, y=1.1, z=1.0),
            camera=dict(eye=dict(x=1.7, y=-1.9, z=0.9)),
        ),
    )
    return fig


def build_2d_figure(rebased: pd.DataFrame, order: list[str], log_scale: bool) -> go.Figure:
    """2D view: one line per series. Benchmarks are dashed."""
    fig = go.Figure()
    for ticker in order:
        s = rebased[ticker].dropna()
        if s.empty:
            continue
        is_benchmark = ticker in BENCHMARKS
        fig.add_trace(go.Scatter(
            x=s.index,
            y=s.values,
            mode="lines",
            name=ticker,
            line=dict(
                color=_colour(ticker),
                width=2.6 if is_benchmark else 1.8,
                dash="dash" if is_benchmark else "solid",
            ),
        ))

    # Dotted reference line: "where everyone started".
    fig.add_hline(y=100, line=dict(color=MUTED, width=1, dash="dot"))

    _style_2d(fig, height=480)
    fig.update_yaxes(
        title="Indexed price (start = 100)",
        type="log" if log_scale else "linear",
        hoverformat=",.1f",
    )
    return fig


def build_drawdown_figure(drawdowns: pd.DataFrame, order: list[str]) -> go.Figure:
    """Drawdown ('underwater') chart: how far below its peak each series is."""
    fig = go.Figure()
    for ticker in order:
        if ticker not in drawdowns.columns:
            continue
        s = drawdowns[ticker].dropna()
        is_benchmark = ticker in BENCHMARKS
        fig.add_trace(go.Scatter(
            x=s.index,
            y=s.values,
            mode="lines",
            name=ticker,
            line=dict(
                color=_colour(ticker),
                width=2.4 if is_benchmark else 1.5,
                dash="dash" if is_benchmark else "solid",
            ),
        ))

    _style_2d(fig, height=420)
    fig.update_yaxes(title="Drop from previous peak", tickformat=".0%", hoverformat=".1%")
    return fig


# ===========================================================================
# 4. THE FUNCTION app.py CALLS
# ===========================================================================
def _show_summary_cards(stats: pd.DataFrame, company_tickers: list[str]) -> None:
    """Four headline numbers above the chart."""
    companies = stats.loc[stats.index.isin(company_tickers)]
    if companies.empty:
        return

    returns = companies["Total return (%)"]
    best, worst = returns.idxmax(), returns.idxmin()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best performer", best, f"{returns[best]:+,.0f}%")
    col2.metric("Weakest performer", worst, f"{returns[worst]:+,.0f}%")
    col3.metric("Median selected return", f"{returns.median():+,.0f}%")

    # Compare with a market benchmark, if one is being shown.
    reference = next((b for b in ("SPY", "QQQ") if b in stats.index), None)
    if reference:
        beat = int((returns > stats.loc[reference, "Total return (%)"]).sum())
        col4.metric(f"Beat {reference}", f"{beat} of {len(returns)}")
    else:
        col4.metric("Beat benchmark", "n/a", "add SPY or QQQ", delta_color="off")


def _show_stats_table(stats: pd.DataFrame, company_tickers: list[str]) -> None:
    """Sortable table: companies (best first), then benchmarks."""
    companies = stats.loc[stats.index.isin(company_tickers)].sort_values(
        "Total return (%)", ascending=False
    )
    benchmarks = stats.loc[~stats.index.isin(company_tickers)]
    table = pd.concat([companies, benchmarks])

    pct = st.column_config.NumberColumn(format="%.1f%%")
    st.dataframe(
        table,
        column_config={
            "Total return (%)": pct,
            "CAGR (%)": pct,
            "Volatility (%)": pct,
            "Max drawdown (%)": pct,
            "Current drawdown (%)": pct,
        },
    )


def render_panel1(selected_companies) -> None:
    """
    Draw Panel 1.

    selected_companies : list of tickers from the global toggle bar in app.py.
    """
    # --- Header --------------------------------------------------------------
    st.subheader("Panel 1: AI stock performance")
    st.caption("How have AI-related companies performed relative to the wider stock market?")

    # --- Clean up the incoming selection ---------------------------------------
    requested = list(dict.fromkeys(selected_companies or []))      # drop duplicates, keep order
    unknown = [t for t in requested if t not in COMPANIES]
    tickers = [t for t in requested if t in COMPANIES]

    if unknown:
        st.warning(f"Ignoring tickers that are not in the company list: {', '.join(unknown)}")
    if len(tickers) > MAX_COMPANIES:
        st.warning(f"Showing the first {MAX_COMPANIES} selected companies.")
        tickers = tickers[:MAX_COMPANIES]
    if not tickers:
        st.info("Select at least one company in the toggle bar above to see its performance.")
        return

    # --- Panel-specific controls -----------------------------------------------
    # Keys start with "p1_" so they can never clash with widgets in other panels.
    col_period, col_view, col_bench, col_log = st.columns([3.2, 1.5, 2.2, 1.4])
    period = col_period.radio("Period", PERIOD_OPTIONS, index=2, horizontal=True, key="p1_period")
    view = col_view.radio("View", ["3D", "2D"], horizontal=True, key="p1_view")
    chosen_benchmarks = col_bench.multiselect(
        "Benchmarks", list(BENCHMARKS), default=list(BENCHMARKS), key="p1_benchmarks"
    )
    log_scale = col_log.checkbox("Log scale", value=False, key="p1_log")

    # --- Load data (cached; one download covers every toggle combination) -------
    all_tickers = tuple(list(COMPANIES) + list(BENCHMARKS))
    try:
        prices = load_prices(all_tickers, PRICE_HISTORY_START)
    except DataLoadError as err:
        st.error(
            f"Could not load price data. {err} "
            "Check your internet connection, then reload the page."
        )
        return

    missing = [t for t in tickers + chosen_benchmarks if t not in prices.columns]
    if missing:
        st.warning(f"No price data was returned for: {', '.join(missing)}. They are left out.")

    company_tickers = [t for t in tickers if t in prices.columns]
    benchmark_tickers = [b for b in chosen_benchmarks if b in prices.columns]
    shown = company_tickers + benchmark_tickers
    if not shown:
        st.info("None of the selected series have data available.")
        return

    # --- Slice the chosen period and calculate -----------------------------------
    start = get_period_start(period, prices.index.max())
    window = prices.loc[prices.index >= start, shown]
    rebased = rebase_to_100(window)
    if rebased.empty:
        st.info("No price data is available for the selected period.")
        return

    order = [t for t in shown if t in rebased.columns]
    stats = compute_stats(window)
    drawdowns = compute_drawdowns(window)

    _show_summary_cards(stats, company_tickers)

    # --- Tabs: chart / statistics / drawdowns -------------------------------------
    tab_chart, tab_stats, tab_drawdown = st.tabs(["Performance", "Statistics", "Drawdowns"])

    with tab_chart:
        if view == "3D":
            fig = build_3d_figure(rebased, order, log_scale)
            st.caption(
                "Drag to rotate, scroll to zoom, hover for exact values. "
                "Height shows the indexed price, so every line starts at 100."
            )
        else:
            fig = build_2d_figure(rebased, order, log_scale)
        st.plotly_chart(
            fig,
            theme=None,                      # keep OUR dark styling, not Streamlit's
            key=f"p1_chart_{view}",
            config={"displaylogo": False},
        )

    with tab_stats:
        _show_stats_table(stats, company_tickers)
        st.caption(
            "Volatility is annualised from daily returns. Max drawdown is the worst fall "
            "from a previous peak inside the chosen period. CAGR is shown only for periods "
            "of about a year or more."
        )

    with tab_drawdown:
        st.plotly_chart(
            build_drawdown_figure(drawdowns, order),
            theme=None,
            key="p1_chart_drawdown",
            config={"displaylogo": False},
        )

    # --- Honest footnotes ------------------------------------------------------------
    late = find_late_starters(window, company_tickers)
    if late:
        details = ", ".join(f"{t} (from {d.strftime('%d %b %Y')})" for t, d in late.items())
        st.caption(
            f"Listed after the start of this period, so indexed to 100 on their first trading "
            f"day instead: {details}. Their returns cover a shorter span."
        )

    last_day = prices.index.max().strftime("%d %b %Y")
    st.caption(
        f"Source: Yahoo Finance via yfinance, daily closes up to {last_day}, in US dollars. "
        "Prices are adjusted for stock splits and dividends. TSMC and ASML are US-listed "
        "ADRs. Company share prices reflect the whole business, not only its AI activity. "
        "Past performance does not predict future returns."
    )


# ===========================================================================
# 5. LOCAL TEST PAGE  (only runs when you run THIS file directly)
# ===========================================================================
# When the main app imports render_panel1, this block is skipped.
if __name__ == "__main__":
    st.set_page_config(page_title="Panel 1 - AI stock performance", layout="wide")
    st.title("The AI Boom Dashboard: Panel 1 test")

    # Stand-in for the global toggle bar, which will live once in the main app.
    selected = st.multiselect(
        "AI companies",
        options=list(COMPANIES),
        default=["NVDA", "MSFT", "GOOGL", "AMZN", "META", "AMD", "PLTR"],
        format_func=lambda ticker: f"{COMPANIES[ticker]} ({ticker})",
        max_selections=12,
    )
    render_panel1(selected)
