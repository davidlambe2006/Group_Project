"""AI Boom dashboard — SINGLE FILE EDITION.

Install once:
    python -m pip install dash==4.5.0 plotly==7.1.0 yfinance==1.8.0 pandas==3.1.1 numpy==2.5.4 Flask==3.1.3
Run:
    python dashboard.py
Open: http://127.0.0.1:8050
Another port: python dashboard.py --port 8052

Everything belonging to this dashboard is in this file: Python panel logic,
Dash layouts and callbacks, CSS, navigation JavaScript, and dated real Yahoo
snapshots. Installed Python libraries are still required. No sibling files,
asset folders, external panel servers, source extraction or exec() are used.
Fresh responses are cached in memory until the process exits.

Code map: shared data helpers → original stock functions → panels 1–4 →
embedded styles/data → cache wiring → main dashboard → launch command.
Prefixes stock_, p1_, p2_, p3_, p4_ keep each panel's functions separate.
"""

# ====================================================================
# CACHE: data_cache.py
# ====================================================================

"""Real Yahoo response snapshots; no synthetic data or alternative calculations."""
from datetime import datetime, timezone
from functools import wraps
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import threading
import time
import pandas as pd
from dash import ctx, html

cache__local = threading.local()
cache__locks = {}
cache__guard = threading.Lock()


def cache_reset_notices():
    cache__local.notices = []


def cache_notices():
    return list(dict.fromkeys(getattr(cache__local, "notices", [])))


def cache_notice_component():
    return html.Div(
        [html.P(n) for n in cache_notices()],
        style={"color": "#b9d7e6", "fontSize": "12px", "padding": "0 16px"},
        role="status",
    )


def cache__encode(value):
    if isinstance(value, pd.DataFrame):
        return {
            "type": "frame",
            "value": value.to_json(
                orient="split", date_format="iso", double_precision=15
            ),
            "dtypes": {str(k): str(v) for k, v in value.dtypes.items()},
        }
    if isinstance(value, pd.DatetimeIndex):
        return {"type": "dates", "value": [v.isoformat() for v in value]}
    if isinstance(value, tuple):
        return {"type": "tuple", "value": [cache__encode(v) for v in value]}
    return {"type": "json", "value": value}


def cache__decode(value):
    kind, data = (value["type"], value["value"])
    if kind == "frame":
        frame = pd.read_json(StringIO(data), orient="split")
        if "dtypes" in value:
            frame = frame.astype(value["dtypes"])
        if len(frame.index) and isinstance(frame.index[0], str):
            frame.index = pd.to_datetime(frame.index)
        return frame
    if kind == "dates":
        return pd.DatetimeIndex(pd.to_datetime(data, utc=True))
    if kind == "tuple":
        return tuple((cache__decode(v) for v in data))
    return data


# ====================================================================
# STOCK: stock_logic.py
# ====================================================================

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta

stock_COMPANIES = {
    "NVDA": "NVIDIA",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "META": "Meta",
    "AVGO": "Broadcom",
    "AMD": "Advanced Micro Devices",
    "TSM": "Taiwan Semiconductor",
    "ORCL": "Oracle",
    "PLTR": "Palantir",
    "CRM": "Salesforce",
    "ASML": "ASML",
}
stock_DEFAULT_COMPANIES = ["NVDA", "MSFT", "GOOGL", "AMZN", "META", "AMD"]
stock_BENCHMARKS = {"SPY": "S&P 500 ETF", "QQQ": "Nasdaq-100 ETF"}
stock_DARK_BLUE = "#0A1830"
stock_STOCK_COLOURS = [
    "#41DDB4",
    "#5BA8FF",
    "#FBC56C",
    "#FC8F73",
    "#C29BFF",
    "#6AE6E6",
    "#E879BD",
    "#AFD873",
    "#FF748E",
    "#CED4FF",
    "#E5B77A",
    "#83BBE9",
    "#FFFFFF",
    "#A8B9CF",
]


def stock_get_prices(tickers, start, end):
    """
    Fetches historical stock prices for the given tickers and date range.

    Parameters:
    tickers (list): List of stock ticker symbols.
    start (str): Start date in 'YYYY-MM-DD' format.
    end (str): End date in 'YYYY-MM-DD' format.

    Returns:
    pd.DataFrame: DataFrame containing historical closing stock prices for selected companies.
    """
    stock_data = yf.download(
        tickers=tickers, start=start, end=end, auto_adjust=True, progress=False
    )
    if stock_data.empty:
        return pd.DataFrame()
    if isinstance(stock_data.columns, pd.MultiIndex):
        closing_prices = stock_data["Close"]
    else:
        closing_prices = stock_data[["Close"]]
        closing_prices.columns = [tickers[0]]
    if isinstance(closing_prices, pd.Series):
        closing_prices = closing_prices.to_frame()
    closing_prices = closing_prices.reindex(columns=tickers)
    closing_prices = closing_prices.apply(pd.to_numeric, errors="coerce")
    closing_prices = closing_prices.dropna(how="all")
    closing_prices = closing_prices.sort_index()
    return closing_prices


def stock_index_and_normalise(prices):
    """
    Index historical stock prices to a common starting value of 100.

    Parameters
    ----------
    prices : pd.DataFrame
        Historical adjusted closing prices for selected companies.

    Returns
    -------
    pd.DataFrame
        Indexed stock prices, with each company starting at 100.

    """
    common_prices = prices.where(prices > 0).dropna(how="any")
    if len(common_prices) < 2:
        return pd.DataFrame()
    starting_prices = common_prices.iloc[0]
    indexed_prices = common_prices.div(starting_prices).mul(100)
    return indexed_prices


def stock_calculate_statistics(indexed_prices):
    """
    Calculate historical returns, volatility and maximum drawdown.

    Parameters
    ----------
    indexed_prices : pd.DataFrame
        Historical stock prices indexed to a starting value of 100.

    Returns
    -------
    pd.DataFrame
        A table containing the following statistics for each stock:
        - Total Return (%)
        - Annualised Volatility (%)
        - Maximum Drawdown (%)

    """
    if len(indexed_prices) < 3:
        return pd.DataFrame()
    daily_returns = indexed_prices.pct_change().dropna()
    total_return = (indexed_prices.iloc[-1] / indexed_prices.iloc[0] - 1) * 100
    volatility = daily_returns.std() * np.sqrt(252) * 100
    previous_peaks = indexed_prices.cummax()
    drawdowns = (indexed_prices / previous_peaks - 1) * 100
    max_drawdown = drawdowns.min()
    statistics = pd.DataFrame(
        {
            "Total Return (%)": total_return,
            "Annualised Volatility (%)": volatility,
            "Maximum Drawdown (%)": max_drawdown,
        }
    )
    statistics.index.name = "Ticker"
    return statistics


def stock_build_chart_3d(indexed_prices):
    """
    Create an interactive 3D chart of indexed stock performance.

    Parameters
    ----------
    indexed_prices : pd.DataFrame
        Stock prices indexed to a common starting value of 100.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 3D chart with stock lines and shaded areas
        underneath each line.

    Notes
    -----
    - X-axis: Date
    - Y-axis: Company
    - Z-axis: Indexed stock price
    """
    fig = go.Figure()
    if indexed_prices.empty:
        return fig
    dates = pd.to_datetime(indexed_prices.index)
    days = (dates - dates[0]).days
    baseline = 100
    for number, ticker in enumerate(indexed_prices.columns):
        colour = stock_STOCK_COLOURS[number % len(stock_STOCK_COLOURS)]
        z_values = indexed_prices[ticker].values
        y_level = number
        fig.add_trace(
            go.Scatter3d(
                x=days,
                y=[y_level] * len(days),
                z=z_values,
                mode="lines",
                name=ticker,
                line=dict(color=colour, width=6),
                text=dates.strftime("%d %b %Y"),
                hovertemplate=f"Company: {ticker}<br>Date: %{{text}}<br>Indexed Price: %{{z:.2f}}<extra></extra>",
            )
        )
        x_mesh = np.concatenate([days, days])
        y_mesh = np.concatenate(
            [np.full(len(days), y_level), np.full(len(days), y_level)]
        )
        z_mesh = np.concatenate([z_values, np.full(len(days), baseline)])
        i = []
        j = []
        k = []
        for n in range(len(days) - 1):
            i.append(n)
            j.append(n + 1)
            k.append(len(days) + n)
            i.append(n + 1)
            j.append(len(days) + n + 1)
            k.append(len(days) + n)
        fig.add_trace(
            go.Mesh3d(
                x=x_mesh,
                y=y_mesh,
                z=z_mesh,
                i=i,
                j=j,
                k=k,
                color=colour,
                opacity=0.18,
                hoverinfo="skip",
                showlegend=False,
            )
        )
    tick_positions = np.linspace(0, len(dates) - 1, min(6, len(dates)), dtype=int)
    fig.update_layout(
        title="AI Stock Performance (3D)",
        template="plotly_dark",
        paper_bgcolor=stock_DARK_BLUE,
        font=dict(color="white"),
        height=620,
        margin=dict(l=0, r=0, t=50, b=55),
        scene=dict(
            xaxis=dict(
                title="Date",
                tickvals=days[tick_positions],
                ticktext=dates[tick_positions].strftime("%b %Y"),
                gridcolor="#35506D",
                backgroundcolor=stock_DARK_BLUE,
            ),
            yaxis=dict(
                title="Company",
                tickvals=list(range(len(indexed_prices.columns))),
                ticktext=list(indexed_prices.columns),
                gridcolor="#35506D",
                backgroundcolor=stock_DARK_BLUE,
            ),
            zaxis=dict(
                title="Indexed Price (Base = 100)",
                gridcolor="#35506D",
                backgroundcolor=stock_DARK_BLUE,
            ),
            bgcolor=stock_DARK_BLUE,
            aspectratio=dict(x=2.2, y=1.0, z=1.2),
            camera=dict(eye=dict(x=1.5, y=-1.9, z=1.1)),
        ),
        legend=dict(orientation="v", x=1.02, y=0.95),
    )
    fig.update_layout(
        font=dict(color="white"),
        title_font=dict(color="white", size=18),
        legend=dict(
            font=dict(color="white", size=13),
            bgcolor="rgba(10, 24, 48, 0.8)",
            orientation="v",
            x=0.99,
            xanchor="right",
            y=0.98,
        ),
        height=560,
    )
    fig.update_scenes(xaxis_color="white", yaxis_color="white", zaxis_color="white")
    return fig


# ====================================================================
# P1: panel1_dash.py
# ====================================================================

"""Dash controls for the original Panel 1 calculations and Plotly figure."""
from datetime import date, timedelta
import pandas as pd
from dash import Dash, Input, Output, dcc, html


def p1_create_app(**kwargs):
    app = Dash(__name__, **kwargs, include_assets_files=False)
    dropdown_style = {"color": "#0A1830"}

    def control(label, element):
        return html.Label([html.Div(label, style={"marginBottom": "8px"}), element])

    app.layout = html.Div(
        [
            html.H2("Panel 1 — AI Stock Performance (3D)"),
            html.P(
                "How have AI-related companies performed relative to the wider stock market?"
            ),
            control(
                "AI Companies",
                dcc.Dropdown(
                    id="p1-companies",
                    options=[
                        {"label": f"{n} ({t})", "value": t}
                        for t, n in stock_COMPANIES.items()
                    ],
                    value=stock_DEFAULT_COMPANIES,
                    multi=True,
                    style=dropdown_style,
                ),
            ),
            html.Div(
                [
                    control(
                        "Historical Period",
                        dcc.Dropdown(
                            id="p1-period",
                            options=["1 Year", "3 Years", "5 Years", "10 Years"],
                            value="3 Years",
                            clearable=False,
                            style=dropdown_style,
                        ),
                    ),
                    control(
                        "Market Benchmarks",
                        dcc.Dropdown(
                            id="p1-benchmarks",
                            options=list(stock_BENCHMARKS),
                            value=["SPY", "QQQ"],
                            multi=True,
                            style=dropdown_style,
                        ),
                    ),
                    control(
                        "Chart Detail",
                        dcc.Dropdown(
                            id="p1-detail",
                            options=["Daily", "Every 5 Days", "Every 20 Days"],
                            value="Every 5 Days",
                            clearable=False,
                            style=dropdown_style,
                        ),
                    ),
                ],
                className="p1-controls",
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(3,minmax(0,1fr))",
                    "gap": "20px",
                    "marginTop": "18px",
                },
            ),
            html.Button(
                "Refresh Yahoo data",
                id="p1-refresh",
                n_clicks=0,
                style={"marginTop": "14px"},
            ),
            dcc.Loading(html.Div(id="p1-result")),
        ],
        style={
            "background": stock_DARK_BLUE,
            "color": "white",
            "padding": "24px",
            "fontFamily": "Arial, sans-serif",
            "minHeight": "100vh",
            "boxSizing": "border-box",
        },
    )

    @app.callback(
        Output("p1-result", "children"),
        Input("p1-companies", "value"),
        Input("p1-period", "value"),
        Input("p1-benchmarks", "value"),
        Input("p1-detail", "value"),
        Input("p1-refresh", "n_clicks"),
    )
    def update(companies, period, benchmarks, detail, _refresh):
        cache_reset_notices()
        if not companies:
            return html.P("Please select at least one AI company.", role="status")
        tickers = list(dict.fromkeys(companies + (benchmarks or [])))
        start = date.today() - timedelta(days=round(int(period.split()[0]) * 365.25))
        try:
            prices = stock_get_prices(
                tickers,
                start.isoformat(),
                (date.today() + timedelta(days=1)).isoformat(),
            )
            missing = [t for t in tickers if t not in prices or prices[t].isna().all()]
            prices = prices.drop(columns=missing, errors="ignore")
            if not any((t in prices for t in companies)):
                raise ValueError("No selected AI companies have available data.")
            indexed = stock_index_and_normalise(prices)
            if len(indexed) < 3:
                raise ValueError("Not enough shared historical data to compare.")
            stats = stock_calculate_statistics(indexed)
            step = {"Daily": 1, "Every 5 Days": 5, "Every 20 Days": 20}[detail]
            sampled = indexed.iloc[::step]
            if sampled.index[-1] != indexed.index[-1]:
                sampled = pd.concat([sampled, indexed.iloc[[-1]]])
            cards = [html.H4("Total Return")]
            for i, ticker in enumerate(indexed.columns):
                colour = stock_STOCK_COLOURS[i % len(stock_STOCK_COLOURS)]
                cards.append(
                    html.Div(
                        [
                            html.Span(
                                f"{stock_COMPANIES.get(ticker, stock_BENCHMARKS.get(ticker, ticker))} ({ticker})"
                            ),
                            html.Strong(
                                f"{stats.loc[ticker, 'Total Return (%)']:+,.1f}%",
                                style={"color": colour, "whiteSpace": "nowrap"},
                            ),
                        ],
                        style={
                            "background": "#102A45",
                            "borderLeft": f"4px solid {colour}",
                            "borderRadius": "7px",
                            "padding": "12px",
                            "marginBottom": "8px",
                            "display": "flex",
                            "justifyContent": "space-between",
                            "gap": "10px",
                            "fontSize": "13px",
                        },
                    )
                )
            return [
                cache_notice_component(),
                (
                    html.P("Data unavailable for: " + ", ".join(missing))
                    if missing
                    else None
                ),
                html.Div(
                    [
                        dcc.Graph(
                            figure=stock_build_chart_3d(sampled),
                            config={"responsive": True, "displaylogo": False},
                            style={"minWidth": 0},
                        ),
                        html.Div(cards),
                    ],
                    className="p1-chart-row",
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "minmax(0,3fr) minmax(220px,1fr)",
                        "gap": "20px",
                    },
                ),
                html.Small(
                    f"Source: Yahoo Finance via yfinance | Actual comparison period: {indexed.index[0]:%d %b %Y} to {indexed.index[-1]:%d %b %Y} | Prices adjusted for stock splits and dividends."
                ),
            ]
        except Exception as error:
            return html.P(
                f"Unable to load historical stock data. {error} Use Refresh Yahoo data to retry.",
                role="alert",
            )

    return app


# ====================================================================
# P2: panel2_valuation.py
# ====================================================================

import yfinance as yf
import plotly.express as px
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, dash_table, Input, Output

p2_tickers = [
    "NVDA",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "AVGO",
    "AMD",
    "TSM",
    "ORCL",
    "PLTR",
    "TSLA",
    "ASML",
]
p2_sector_map = {
    "NVDA": "Semiconductors",
    "MSFT": "Software & Cloud",
    "GOOGL": "Software & Cloud",
    "AMZN": "Software & Cloud",
    "META": "Internet Services",
    "AVGO": "Semiconductors",
    "AMD": "Semiconductors",
    "TSM": "Semiconductors",
    "ORCL": "Software & Cloud",
    "PLTR": "Software & Cloud",
    "TSLA": "Consumer & EV",
    "ASML": "Semiconductors",
}
p2_NUMERIC_COLS = ["Market Cap ($T)", "Capital Expenditure ($B)", "Annual Revenue ($B)"]


def p2_load_valuation_data(selected_tickers):
    """Download Yahoo Finance financial metrics for the requested companies.

    A failed ticker is recorded with missing values so other tickers still load.
    """
    rows = []
    failed_tickers = []
    for ticker in selected_tickers:
        name, cap, capex, revenue = (ticker, None, None, None)
        try:
            company = yf.Ticker(ticker)
            info = company.info
            name = info.get("displayName") or info.get("shortName") or ticker
            cap = info.get("marketCap")
            income = company.income_stmt
            cashflow = company.cashflow
            if not income.empty and (not cashflow.empty):
                if (
                    "Total Revenue" in income.index
                    and "Capital Expenditure" in cashflow.index
                ):
                    common_years = income.columns.intersection(cashflow.columns)
                    for year in sorted(common_years, reverse=True):
                        rev = income.loc["Total Revenue", year]
                        expenditure = cashflow.loc["Capital Expenditure", year]
                        if pd.notna(rev) and pd.notna(expenditure):
                            currency = info.get("financialCurrency")
                            market_currency = info.get("currency")
                            if not currency:
                                raise ValueError(
                                    f"Financial currency unavailable for {ticker}"
                                )
                            if cap is not None and (not market_currency):
                                raise ValueError(
                                    f"Market cap currency unavailable for {ticker}"
                                )
                            fx_rate = 1.0
                            if currency != "USD":
                                fx_ticker = f"{currency}USD=X"
                                fx_data = yf.Ticker(fx_ticker).history(
                                    start=year - pd.DateOffset(years=1),
                                    end=year + pd.Timedelta(days=1),
                                )
                                if fx_data.empty:
                                    raise ValueError(
                                        f"No exchange rate available for {currency}"
                                    )
                                fx_rate = fx_data["Close"].dropna().mean()
                            revenue = rev * fx_rate / 1000000000.0
                            capex = abs(expenditure) * fx_rate / 1000000000.0
                            if cap is not None and market_currency != "USD":
                                fx_ticker = f"{market_currency}USD=X"
                                fx_data = yf.Ticker(fx_ticker).history(period="5d")
                                if fx_data.empty or fx_data["Close"].dropna().empty:
                                    raise ValueError(
                                        f"No market cap exchange rate for {market_currency}"
                                    )
                                fx_rate_market = fx_data["Close"].dropna().iloc[-1]
                                cap = cap * fx_rate_market
                            break
        except Exception as error:
            failed_tickers.append(ticker)
            cap, capex, revenue = (None, None, None)
            print(f"Error retrieving data for {ticker}: {error}")
        rows.append(
            {
                "Company": name,
                "Ticker": ticker,
                "Market Cap ($T)": cap / 1000000000000.0 if cap is not None else None,
                "Capital Expenditure ($B)": capex,
                "Annual Revenue ($B)": revenue,
                "Sector": p2_sector_map.get(ticker, "Other"),
            }
        )
    return (
        pd.DataFrame(rows, columns=["Company", "Ticker", *p2_NUMERIC_COLS, "Sector"]),
        failed_tickers,
    )


def p2_prepare_chart_data(df):
    """Remove missing/invalid bubble inputs."""
    cleaned = df.copy()
    cleaned[p2_NUMERIC_COLS] = (
        cleaned[p2_NUMERIC_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    chart_df = cleaned.dropna(subset=p2_NUMERIC_COLS)
    chart_df = chart_df[
        (chart_df["Capital Expenditure ($B)"] > 0)
        & (chart_df["Annual Revenue ($B)"] > 0)
        & (chart_df["Market Cap ($T)"] > 0)
    ]
    return chart_df


def p2_calculate_regression(chart_df):
    """Fit annual revenue against capital expenditure; return fit statistics."""
    x = chart_df["Capital Expenditure ($B)"]
    y = chart_df["Annual Revenue ($B)"]
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return (None, None, None, x)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    ss_total = np.sum((y - y.mean()) ** 2)
    ss_residual = np.sum((y - predicted) ** 2)
    r_squared = 1 - ss_residual / ss_total
    return (slope, intercept, r_squared, x)


def p2_build_valuation_chart(chart_df, regression):
    """Build the interactive bubble chart with its legends and regression toggle."""
    slope, intercept, r_squared, x = regression
    size_max = 75
    fig = px.scatter(
        chart_df,
        x="Capital Expenditure ($B)",
        y="Annual Revenue ($B)",
        size="Market Cap ($T)",
        color="Sector",
        color_discrete_map={
            "Semiconductors": "#00D4FF",
            "Software & Cloud": "#FF6B6B",
            "Internet Services": "#00E09F",
            "Consumer & EV": "#B477FF",
        },
        text="Ticker",
        size_max=size_max,
        hover_name="Company",
        hover_data={
            "Ticker": False,
            "Market Cap ($T)": ":.2f",
            "Capital Expenditure ($B)": ":.2f",
            "Annual Revenue ($B)": ":.2f",
        },
        title="The AI Bubble: Capital Investment vs Revenue",
    )
    if slope is not None:
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        fig.add_scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="Regression Line",
            showlegend=False,
            hoverinfo="skip",
            line=dict(color="#FFD166", width=3, dash="dash"),
        )
    fig.update_traces(
        marker=dict(opacity=1.0, line=dict(width=2, color="rgba(225,255,255,0.95)")),
        textposition="middle center",
        selector=dict(mode="markers+text"),
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#031A2F",
        plot_bgcolor="#031A2F",
        margin=dict(r=250, t=170),
        title=dict(x=0.5, y=0.97, yanchor="top", font=dict(size=24, weight="bold")),
        xaxis=dict(
            title_text="<b>Capital Expenditure ($B)</b>",
            showline=True,
            linecolor="white",
            linewidth=2,
            gridcolor="#243449",
            griddash="dash",
            zeroline=False,
        ),
        yaxis=dict(
            title_text="<b>Annual Revenue ($B)</b>",
            tickformat=".0f",
            showline=True,
            linecolor="white",
            linewidth=2,
            gridcolor="#243449",
            griddash="dash",
            zeroline=False,
            minor=dict(showgrid=False, ticks=""),
        ),
        hoverlabel=dict(bgcolor="#061426", font_color="white", bordercolor="#00D4FF"),
        legend=dict(
            title="<b>Sector</b>",
            x=1.02,
            y=1.08,
            bgcolor="#061426",
            bordercolor="#1E3A5F",
            borderwidth=2,
        ),
    )
    fig.add_annotation(
        x=0.59,
        y=1.2,
        xref="paper",
        yref="paper",
        xanchor="center",
        text="Exploring the Relationship Between Capital Expenditure and Annual Revenue",
        showarrow=False,
        font=dict(size=15, color="white"),
    )
    fig.add_annotation(
        x=0.59,
        y=1.15,
        xref="paper",
        yref="paper",
        xanchor="center",
        text="Bubble Size = Market Capitalisation ($T)",
        showarrow=False,
        font=dict(size=13, color="#00D4FF"),
    )
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=1.02,
        x1=1.2,
        y0=0.32,
        y1=0.75,
        line=dict(color="#1E3A5F", width=2),
        fillcolor="#061426",
    )
    fig.add_annotation(
        x=1.105,
        y=0.725,
        xanchor="center",
        xref="paper",
        yref="paper",
        text="<b>Market Cap ($T)</b>",
        showarrow=False,
        font=dict(size=14, color="white"),
    )
    max_cap = chart_df["Market Cap ($T)"].max()
    for cap, y in [(0.5, 0.63), (1.0, 0.525), (2.0, 0.395)]:
        size = size_max * np.sqrt(cap / max_cap)
        fig.add_shape(
            type="circle",
            xref="paper",
            yref="paper",
            xsizemode="pixel",
            ysizemode="pixel",
            xanchor=1.06,
            yanchor=y,
            x0=-size / 2,
            x1=size / 2,
            y0=-size / 2,
            y1=size / 2,
            line=dict(color="white", width=2),
            fillcolor="#607D9F",
        )
        fig.add_annotation(
            x=1.1,
            y=y,
            xref="paper",
            yref="paper",
            text=f"${cap:.1f}T",
            showarrow=False,
            xanchor="left",
            font=dict(size=12, color="white"),
        )
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=1.02,
        x1=1.2,
        y0=0.01,
        y1=0.32,
        line=dict(color="#1E3A5F", width=2),
        fillcolor="#061426",
    )
    fig.add_annotation(
        x=1.11,
        y=0.285,
        xanchor="center",
        yanchor="middle",
        xref="paper",
        yref="paper",
        text="<b>Regression Analysis</b>",
        showarrow=False,
        font=dict(size=13, color="white"),
    )
    if r_squared is None:
        fig.add_annotation(
            x=1.11,
            y=0.16,
            xref="paper",
            yref="paper",
            text="Regression unavailable",
            showarrow=False,
            font=dict(size=10, color="white"),
        )
    else:
        fit_label = f"{r_squared * 100:.1f}% of revenue variation<br>explained by CapEx"
        fig.add_annotation(
            x=1.11,
            y=0.14,
            xanchor="center",
            xref="paper",
            yref="paper",
            text=fit_label,
            showarrow=False,
            font=dict(size=10, color="white"),
        )
        interpretation_index = len(fig.layout.annotations) - 1
        fig.add_annotation(
            x=1.11,
            y=0.205,
            xanchor="center",
            xref="paper",
            yref="paper",
            text=f"<b>R² = {r_squared:.6f}</b>",
            showarrow=False,
            font=dict(size=15, color="#FFD166"),
        )
        r2_index = len(fig.layout.annotations) - 1
        regression_trace_index = len(fig.data) - 1
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    showactive=True,
                    active=0,
                    x=1.11,
                    y=0.075,
                    xanchor="center",
                    yanchor="middle",
                    bgcolor="#607D9F",
                    bordercolor="#00D4FF",
                    borderwidth=1,
                    font=dict(size=12, color="black"),
                    pad=dict(l=2, r=2, t=2, b=2),
                    buttons=[
                        dict(
                            label="On",
                            method="update",
                            args=[
                                {"visible": [True]},
                                {
                                    f"annotations[{r2_index}].visible": True,
                                    f"annotations[{interpretation_index}].visible": True,
                                },
                                [regression_trace_index],
                            ],
                        ),
                        dict(
                            label="Off",
                            method="update",
                            args=[
                                {"visible": [False]},
                                {
                                    f"annotations[{r2_index}].visible": False,
                                    f"annotations[{interpretation_index}].visible": False,
                                },
                                [regression_trace_index],
                            ],
                        ),
                    ],
                )
            ]
        )
    fig.update_layout(margin=dict(r=40), xaxis=dict(domain=[0, 0.8]))
    fig.layout.legend.x = 0.84
    for annotation in fig.layout.annotations:
        if annotation.xref == "paper" and annotation.x > 1:
            annotation.x = 0.82 + (annotation.x - 1) * 0.8
    for shape in fig.layout.shapes:
        if shape.xref == "paper":
            if (
                shape.xsizemode == "pixel"
                and shape.xanchor is not None
                and (shape.xanchor > 1)
            ):
                shape.xanchor = 0.82 + (shape.xanchor - 1) * 0.8
            elif shape.x0 > 1:
                shape.x0 = 0.82 + (shape.x0 - 1) * 0.8
                shape.x1 = 0.82 + (shape.x1 - 1) * 0.8
    for menu in fig.layout.updatemenus:
        menu.x = 0.82 + (menu.x - 1) * 0.8
    return fig


def p2_create_app(selected_tickers=p2_tickers, **kwargs):
    app = Dash(__name__, **kwargs, include_assets_files=False)
    tab_style = {
        "backgroundColor": "#102D47",
        "color": "white",
        "border": "1px solid #1E3A5F",
        "padding": "12px",
    }
    selected_tab_style = {
        **tab_style,
        "backgroundColor": "#FFFFFF",
        "color": "#031A2F",
        "fontWeight": "bold",
        "borderTop": "2px solid #00D4FF",
    }
    app.layout = html.Div(
        [
            html.Button("Refresh Yahoo data", id="valuation-refresh", n_clicks=0),
            dcc.Tabs(
                id="view-toggle",
                value="graph",
                children=[
                    dcc.Tab(
                        label="Graph",
                        value="graph",
                        style=tab_style,
                        selected_style=selected_tab_style,
                    ),
                    dcc.Tab(
                        label="Table",
                        value="table",
                        style=tab_style,
                        selected_style=selected_tab_style,
                    ),
                ],
                style={"width": "220px", "margin": "0 auto"},
            ),
            dcc.Loading(html.Div(id="view-content")),
        ],
        style={
            "backgroundColor": "#031A2F",
            "minHeight": "100vh",
            "margin": "0",
            "padding": "8px 0",
            "color": "white",
            "fontFamily": "Arial",
        },
    )

    @app.callback(
        Output("view-content", "children"),
        Input("view-toggle", "value"),
        Input("valuation-refresh", "n_clicks"),
    )
    def switch_view(view, _refresh):
        cache_reset_notices()
        df, failed_tickers = p2_load_valuation_data(selected_tickers)
        chart_df = p2_prepare_chart_data(df)
        if chart_df.empty:
            return html.P(
                "Yahoo Finance returned no valid company metrics. Please retry.",
                role="alert",
            )
        omitted = df.loc[~df["Ticker"].isin(chart_df["Ticker"]), "Ticker"].tolist()
        if omitted:
            print("Omitted due to missing/invalid metrics: " + ", ".join(omitted))
        regression = p2_calculate_regression(chart_df)
        if regression[2] is None:
            print("Insufficient valid data or variation to calculate regression.")
        else:
            print(f"R-squared: {regression[2]:.6f}")
        fig = p2_build_valuation_chart(chart_df, regression)
        table_df = chart_df[
            [
                "Ticker",
                "Company",
                "Sector",
                "Capital Expenditure ($B)",
                "Annual Revenue ($B)",
                "Market Cap ($T)",
            ]
        ].sort_values("Ticker")
        table_df = table_df.rename(
            columns={
                "Capital Expenditure ($B)": "CapEx ($B)",
                "Annual Revenue ($B)": "Revenue ($B)",
            }
        )
        numeric_table_cols = ["CapEx ($B)", "Revenue ($B)", "Market Cap ($T)"]
        if view == "graph":
            return html.Div(
                [
                    cache_notice_component(),
                    dcc.Graph(
                        figure=fig,
                        style={"width": "100%", "height": "calc(100vh - 125px)"},
                        config={"responsive": True},
                    ),
                ]
            )
        else:
            return html.Div(
                [
                    cache_notice_component(),
                    html.H2(
                        "The AI Bubble: Company Comparison",
                        style={"textAlign": "center", "padding": "25px"},
                    ),
                    dash_table.DataTable(
                        data=table_df.to_dict("records"),
                        columns=[
                            {
                                "name": col,
                                "id": col,
                                "type": (
                                    "numeric" if col in numeric_table_cols else "text"
                                ),
                                "format": (
                                    {"specifier": ",.2f"}
                                    if col in numeric_table_cols
                                    else {}
                                ),
                            }
                            for col in table_df.columns
                        ],
                        sort_action="native",
                        page_action="none",
                        style_table={
                            "width": "95%",
                            "margin": "auto",
                            "overflowX": "auto",
                        },
                        style_header={
                            "backgroundColor": "#163954",
                            "color": "white",
                            "fontWeight": "bold",
                            "border": "1px solid #2D5474",
                            "padding": "15px",
                        },
                        style_cell={
                            "backgroundColor": "#0B2943",
                            "color": "white",
                            "border": "1px solid #23445E",
                            "padding": "14px",
                            "textAlign": "left",
                            "fontFamily": "Arial",
                            "fontSize": "14px",
                        },
                        style_data_conditional=[
                            {"if": {"row_index": "odd"}, "backgroundColor": "#102F49"}
                        ],
                    ),
                ]
            )

    return app


# ====================================================================
# P3: panel3_dotcom.py
# ====================================================================

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from functools import lru_cache
from pathlib import Path
from dash import Dash, dcc, html, Input, Output, ctx

p3_DEFAULT_AI_COMPANIES = [
    "NVDA",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "AVGO",
    "AMD",
    "TSM",
    "ORCL",
    "PLTR",
    "CRM",
    "ASML",
]
p3_NAVY = "#0B1220"
p3_BLUE = "#38BDF8"
p3_CYAN = "#00E5A8"
p3_GREEN = "#4ADE80"
p3_PURPLE = "#C4B5FD"
p3_PINK = "#FB7185"


@lru_cache(maxsize=1)
def p3_trading_session_closes(session_date):
    try:
        metadata = yf.Ticker("^IXIC").get_history_metadata()
        schedule = metadata["tradingPeriods"]
        closes = pd.DatetimeIndex(pd.to_datetime(schedule["end"], utc=True))
        if closes.empty or closes.isna().any():
            raise ValueError("Missing session closing times.")
        return closes
    except Exception as error:
        raise ValueError(
            "Unable to check completed trading sessions. Please try again shortly."
        ) from error


@lru_cache(maxsize=32)
def p3_download_prices(tickers, start, end=None):
    try:
        data = yf.download(
            list(tickers), start=start, end=end, auto_adjust=True, progress=False
        )
        if data is None or data.empty:
            raise ValueError("No prices returned.")
        prices = data["Close"].copy()
    except Exception as error:
        raise ValueError(
            "Yahoo Finance prices are unavailable. Please try again shortly."
        ) from error
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    prices.columns = [str(col).upper() for col in prices.columns]
    missing = set(tickers) - set(prices.columns)
    if missing:
        raise ValueError(f"Missing price data for: {', '.join(sorted(missing))}.")
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans:
        raise ValueError("Yahoo Finance returned invalid price dates. Please retry.")
    prices.index = prices.index.tz_localize(None)
    prices = prices.sort_index().apply(pd.to_numeric, errors="coerce")
    prices = prices.loc[prices.index >= pd.Timestamp(start)]
    if end is not None:
        prices = prices.loc[prices.index < pd.Timestamp(end)]
    if prices.index.has_duplicates or ((prices <= 0) | np.isinf(prices)).any().any():
        raise ValueError("Yahoo Finance returned invalid prices. Please retry.")
    unavailable = [ticker for ticker in tickers if prices[ticker].dropna().empty]
    if unavailable:
        raise ValueError(
            f"No valid price data for: {', '.join(unavailable)}. Please retry."
        )
    return prices.dropna(how="all")


def p3_calculate_ai_portfolio(selected_companies, start_date="2023-01-01"):
    selected = list(dict.fromkeys(selected_companies))
    if not selected:
        raise ValueError("Select at least one AI company.")
    invalid = set(selected) - set(p3_DEFAULT_AI_COMPANIES)
    if invalid:
        raise ValueError(f"Unknown companies: {invalid}")
    now = pd.Timestamp.now(tz="America/New_York")
    closes = p3_trading_session_closes(now.strftime("%Y-%m-%d"))
    completed = closes[closes <= now]
    if completed.empty:
        raise ValueError(
            "No completed trading session is available. Please retry shortly."
        )
    end_date = (
        completed.max().tz_convert("America/New_York").normalize()
        + pd.DateOffset(days=1)
    ).strftime("%Y-%m-%d")
    prices = p3_download_prices(tuple(sorted(selected)), start_date, end_date)
    prices = prices[selected].dropna()
    if prices.empty:
        raise ValueError("No overlapping price data available.")
    indexed = prices.div(prices.iloc[0]) * 100
    portfolio = indexed.mean(axis=1)
    return portfolio


@lru_cache(maxsize=1)
def p3_calculate_nasdaq():
    prices = p3_download_prices(("^IXIC",), "1995-01-01", "2004-01-01")
    close = prices["^IXIC"]
    return close / close.iloc[0] * 100


def p3_calculate_metrics(series):
    cumulative_return = (series.iloc[-1] / series.iloc[0] - 1) * 100
    running_peak = series.cummax()
    drawdown = (series / running_peak - 1) * 100
    return (cumulative_return, drawdown.min())


def p3_create_forecast(start_value, target_return, months=24):
    t = np.arange(months + 1)
    target_value = start_value * (1 + target_return)
    trend = np.log(target_value / start_value) * t / months
    fluctuation = 0.035 * np.sin(np.pi * t / months) * np.sin(2 * np.pi * t / 7)
    values = start_value * np.exp(trend + fluctuation)
    values[0] = start_value
    values[-1] = target_value
    return values


def p3_monthly_with_latest(series):
    monthly = series.resample("MS").first().dropna()
    latest_date = series.index[-1]
    first_in_last_month = series.loc[series.index >= monthly.index[-1]].index[0]
    if latest_date > first_in_last_month:
        monthly.loc[latest_date] = series.iloc[-1]
    x = (
        (monthly.index.year - series.index[0].year) * 12
        + monthly.index.month
        - series.index[0].month
        + (monthly.index.day - 1) / monthly.index.days_in_month
    ).to_numpy(dtype=float, copy=True)
    if latest_date == first_in_last_month and len(monthly) > 1:
        x[-1] += (latest_date.day - 1) / latest_date.days_in_month
    return (monthly, x)


def p3_render_panel3(selected_companies=None):
    if selected_companies is None:
        selected_companies = p3_DEFAULT_AI_COMPANIES
    nasdaq = p3_calculate_nasdaq()
    ai = p3_calculate_ai_portfolio(selected_companies)
    nasdaq_return, nasdaq_dd = p3_calculate_metrics(nasdaq)
    ai_return, ai_dd = p3_calculate_metrics(ai)
    nasdaq_monthly, nasdaq_x = p3_monthly_with_latest(nasdaq)
    ai_monthly, ai_x = p3_monthly_with_latest(ai)
    nasdaq_monthly = nasdaq_monthly / nasdaq_monthly.iloc[0] * 100
    ai_monthly = ai_monthly / ai_monthly.iloc[0] * 100
    nasdaq_monthly.iloc[-1] = nasdaq.iloc[-1]
    ai_monthly.iloc[-1] = ai.iloc[-1]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=nasdaq_x,
            y=nasdaq_monthly,
            mode="lines",
            name="Dot-Com Era · NASDAQ",
            line=dict(color=p3_BLUE, width=3.5),
            customdata=nasdaq_monthly.to_numpy() - 100,
            hovertemplate="<b>Dot-Com Era · NASDAQ</b><br>Month: %{x}<br>Index (base 100): %{y:.2f}<br>Return since start: %{customdata:+.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ai_x,
            y=ai_monthly,
            mode="lines",
            name="Modern AI · Portfolio",
            line=dict(color=p3_CYAN, width=3.5),
            customdata=ai_monthly.to_numpy() - 100,
            hovertemplate="<b>Modern AI Portfolio</b><br>Month: %{x}<br>Index (base 100): %{y:.2f}<br>Return since start: %{customdata:+.2f}%<extra></extra>",
        )
    )
    last_month = float(ai_x[-1])
    last_value = float(ai_monthly.iloc[-1])
    forecast_x = last_month + np.arange(25)
    month_ends = ai.resample("ME").last().dropna()
    current_month = (
        pd.Timestamp.now(tz="America/New_York")
        .tz_localize(None)
        .normalize()
        .replace(day=1)
    )
    month_ends = month_ends.loc[month_ends.index < current_month]
    monthly_log_returns = np.log(month_ends / month_ends.shift(1)).dropna()
    if len(monthly_log_returns) < 2:
        raise ValueError(
            "At least three completed month-end observations are needed for the volatility-based scenarios."
        )
    sigma = float(monthly_log_returns.std(ddof=1))
    months = 24
    z = 1.645
    spread = z * sigma * np.sqrt(months)
    scenarios = [
        ("Bull", 1.166 * np.exp(spread) - 1, p3_GREEN),
        ("Base", 0.166, p3_PURPLE),
        ("Bear", 1.166 * np.exp(-spread) - 1, p3_PINK),
    ]
    for name, target_return, colour in scenarios:
        forecast_y = p3_create_forecast(last_value, target_return)
        fig.add_trace(
            go.Scatter(
                x=forecast_x,
                y=forecast_y,
                mode="lines",
                name=f"{name} Scenario",
                line=dict(color=colour, width=2.5, dash="dash"),
                customdata=np.column_stack(
                    (np.arange(25), (forecast_y / last_value - 1) * 100)
                ),
                hovertemplate=f"<b>{name} Scenario · Illustrative</b><br>Forecast month: %{{customdata[0]:.0f}} of 24<br>Projected index: %{{y:.2f}}<br>Return from forecast start: %{{customdata[1]:+.2f}}%<br>24-month target: {target_return:+.1%}<extra></extra>",
            )
        )
        fig.add_annotation(
            x=forecast_x[-1],
            y=forecast_y[-1],
            text=f"<b>{name} {target_return:+.1%}</b>",
            showarrow=False,
            xanchor="left",
            xshift=10,
            yshift={"Bull": 18, "Base": 0, "Bear": -18}[name],
            bgcolor=p3_NAVY,
            borderpad=5,
            font=dict(size=13, color=colour),
        )
    fig.add_vline(
        x=last_month, line_color="rgba(203,213,225,0.35)", line_width=1, line_dash="dot"
    )
    fig.add_annotation(
        x=last_month,
        y=0,
        yref="paper",
        yshift=12,
        text="Forecast starts",
        showarrow=False,
        font=dict(size=11, color="#CBD5E1"),
        bgcolor=p3_NAVY,
    )
    fig.update_layout(
        paper_bgcolor=p3_NAVY,
        plot_bgcolor=p3_NAVY,
        font=dict(
            family="Inter, Segoe UI, Arial, sans-serif", color="#CBD5E1", size=13
        ),
        xaxis_title="Months Since Start",
        yaxis_title="Indexed Performance · Base 100",
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#172338",
            bordercolor="#334155",
            font=dict(color="#F8FAFC", size=13),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=15, color="#F1F5F9"),
            itemsizing="constant",
            itemwidth=40,
            tracegroupgap=0,
        ),
        margin=dict(l=60, r=115, t=65, b=65),
        height=740,
        autosize=True,
        hoverdistance=30,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.07)",
        automargin=True,
        title_standoff=18,
        tickfont=dict(size=12, color="#CBD5E1"),
        zeroline=False,
        showline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.07)",
        automargin=True,
        title_standoff=18,
        tickfont=dict(size=12, color="#CBD5E1"),
        zeroline=False,
        showline=False,
    )
    fig.update_layout(
        meta={
            "nasdaq_return": float(nasdaq_return),
            "nasdaq_dd": float(nasdaq_dd),
            "ai_return": float(ai_return),
            "ai_dd": float(ai_dd),
            "company_count": len(selected_companies),
        }
    )
    return fig


def p3_nasdaq_crash_detail():
    nasdaq = p3_calculate_nasdaq()
    drawdown = nasdaq / nasdaq.cummax() - 1
    trough_date = drawdown.idxmin()
    peak_date = nasdaq.loc[:trough_date].idxmax()
    crash = nasdaq.loc[peak_date:trough_date]
    detail = go.Figure(
        go.Scatter(
            x=crash.index,
            y=crash,
            mode="lines",
            line=dict(color=p3_BLUE, width=2.5),
            customdata=(crash / crash.iloc[0] - 1) * 100,
            hovertemplate="<b>NASDAQ · Historical crash</b><br>%{x|%d %b %Y}<br>Index (base 100): %{y:.2f}<br>Return from peak: %{customdata:+.2f}%<extra></extra>",
        )
    )
    for label, date, anchor in [
        ("Peak", peak_date, "left"),
        ("Trough", trough_date, "right"),
    ]:
        detail.add_annotation(
            x=date,
            y=float(nasdaq.loc[date]),
            text=f"<b>{label}</b> · {date:%d %b %Y}",
            showarrow=False,
            xanchor=anchor,
            yanchor="bottom",
            yshift=8,
            font=dict(size=12, color="#E2E8F0"),
            bgcolor=p3_NAVY,
            borderpad=3,
        )
    detail.add_annotation(
        x=0.96,
        y=0.85,
        xref="paper",
        yref="paper",
        text=f"<b>−{abs(drawdown.min()):.1%} Peak-to-Trough</b>",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        font=dict(size=17, color=p3_PINK),
        bgcolor=p3_NAVY,
        borderpad=5,
    )
    padding = float(crash.max() - crash.min()) * 0.15
    detail.update_layout(
        title=dict(
            text=f"NASDAQ crash detail · Peak-to-trough drawdown {drawdown.min():.1%}",
            font=dict(size=14),
            x=0.02,
        ),
        paper_bgcolor=p3_NAVY,
        plot_bgcolor=p3_NAVY,
        font=dict(
            family="Inter, Segoe UI, Arial, sans-serif", color="#CBD5E1", size=12
        ),
        height=350,
        margin=dict(l=60, r=30, t=65, b=45),
        showlegend=False,
        hoverlabel=dict(bgcolor="#172338", bordercolor="#334155", font_color="#F8FAFC"),
    )
    detail.update_xaxes(tickformat="%b %Y", nticks=6, showgrid=False, zeroline=False)
    detail.update_yaxes(
        title="NASDAQ index · Base 100",
        automargin=True,
        range=[float(crash.min()) - padding, float(crash.max()) + padding],
        gridcolor="rgba(148,163,184,0.07)",
        zeroline=False,
    )
    return detail


def p3_metric_card(label, value, colour):
    return html.Div(
        [
            html.Div(
                label,
                style={
                    "color": "#B7C5D8",
                    "fontSize": "12px",
                    "marginBottom": "12px",
                    "letterSpacing": "0.7px",
                    "lineHeight": "1.5",
                },
            ),
            html.Div(
                value,
                style={
                    "color": colour,
                    "fontSize": "28px",
                    "fontWeight": "600",
                    "fontVariantNumeric": "tabular-nums",
                },
            ),
        ],
        style={
            "backgroundColor": "#131F31",
            "padding": "20px",
            "border": "1px solid #263449",
            "borderRadius": "10px",
            "flex": "1 1 180px",
            "minWidth": "0",
        },
    )


def p3_panel3_layout(selected_companies=None):
    if selected_companies is None:
        selected_companies = p3_DEFAULT_AI_COMPANIES
    try:
        fig = p3_render_panel3(selected_companies)
        crash_detail = p3_nasdaq_crash_detail()
    except ValueError as error:
        return html.P(f"Unable to load Panel 3. {error}", style={"color": "white"})
    metrics = fig.layout.meta
    return html.Div(
        [
            html.H2(
                "AI Boom vs. Dot-Com Bubble",
                style={
                    "color": "#F8FAFC",
                    "marginBottom": "6px",
                    "letterSpacing": "-0.5px",
                    "fontSize": "clamp(23px, 3vw, 32px)",
                    "marginTop": "0",
                },
            ),
            html.P(
                "Two technology booms. One historical comparison.",
                style={"color": "#94A3B8", "marginTop": "0", "marginBottom": "22px"},
            ),
            html.Div(
                [
                    p3_metric_card(
                        "NASDAQ PRICE RETURN",
                        f"{metrics['nasdaq_return']:+.1f}%",
                        p3_GREEN if metrics["nasdaq_return"] >= 0 else p3_PINK,
                    ),
                    p3_metric_card(
                        "NASDAQ MAX DRAWDOWN", f"{metrics['nasdaq_dd']:.1f}%", p3_PINK
                    ),
                    p3_metric_card(
                        "AI PORTFOLIO RETURN",
                        f"{metrics['ai_return']:+.1f}%",
                        p3_GREEN if metrics["ai_return"] >= 0 else p3_PINK,
                    ),
                    p3_metric_card(
                        "AI MAX DRAWDOWN", f"{metrics['ai_dd']:.1f}%", p3_PINK
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "12px",
                    "flexWrap": "wrap",
                    "marginBottom": "16px",
                },
            ),
            html.Div(
                "Solid lines · historical observations    /    Dashed lines · illustrative scenarios",
                style={"color": "#B7C5D8", "fontSize": "12px", "lineHeight": "1.7"},
            ),
            dcc.Graph(
                figure=fig,
                config={"displaylogo": False, "responsive": True},
                responsive=True,
                style={"width": "100%", "minWidth": "0", "height": "740px"},
            ),
            dcc.Graph(
                figure=crash_detail,
                config={
                    "displaylogo": False,
                    "displayModeBar": False,
                    "responsive": True,
                },
                responsive=True,
                style={"width": "100%", "minWidth": "0", "height": "350px"},
            ),
            html.P(
                f"AI portfolio: {metrics['company_count']} companies, equally weighted at inception. The 24-month Base return is assumed at +16.6%; Bull/Bear are illustrative volatility-based scenarios using completed monthly log returns (z = 1.645), not statistically validated predictions. Intermediate fluctuations are illustrative, not statistically simulated forecasts.",
                style={
                    "color": "#94A3B8",
                    "fontSize": "12px",
                    "marginTop": "8px",
                    "lineHeight": "1.7",
                },
            ),
        ],
        style={
            "backgroundColor": p3_NAVY,
            "padding": "clamp(14px, 2.5vw, 28px)",
            "minWidth": "0",
            "borderRadius": "14px",
            "fontFamily": "Inter, Segoe UI, Arial, sans-serif",
        },
    )


def p3_create_app(**kwargs):
    app = Dash(__name__, **kwargs, include_assets_files=False)
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Label(
                        "AI Companies",
                        htmlFor="global-company-selector",
                        style={"color": "#CBD5E1"},
                    ),
                    html.Span(
                        id="panel3-selection-count", className="panel3-selection-count"
                    ),
                    html.Div(
                        [
                            html.Button(
                                "Select All", id="panel3-select-all", n_clicks=0
                            ),
                            html.Button(
                                "Deselect All", id="panel3-deselect-all", n_clicks=0
                            ),
                        ],
                        className="panel3-selection-actions",
                    ),
                    dcc.Dropdown(
                        id="global-company-selector",
                        options=[
                            {"label": ticker, "value": ticker}
                            for ticker in p3_DEFAULT_AI_COMPANIES
                        ],
                        value=list(p3_DEFAULT_AI_COMPANIES),
                        multi=True,
                        placeholder="Search companies…",
                        className="panel3-company-select",
                        style={"backgroundColor": p3_NAVY, "color": "#F1F5F9"},
                    ),
                ],
                className="panel3-selector",
                style={
                    "marginBottom": "20px",
                    "width": "100%",
                    "maxWidth": "1000px",
                    "--Dash-Fill-Inverse-Strong": p3_NAVY,
                    "--Dash-Fill-Interactive-Strong": p3_CYAN,
                    "--Dash-Fill-Interactive-Weak": "#153343",
                    "--Dash-Fill-Primary-Hover": "#1E334B",
                    "--Dash-Fill-Primary-Active": "#153343",
                    "--Dash-Fill-Disabled": "#263449",
                    "--Dash-Stroke-Strong": "#334155",
                    "--Dash-Stroke-Weak": "#263449",
                    "--Dash-Text-Primary": "#F1F5F9",
                    "--Dash-Text-Strong": "#F1F5F9",
                    "--Dash-Text-Weak": "#CBD5E1",
                    "--Dash-Text-Disabled": "#94A3B8",
                },
            ),
            html.Button("Retry Yahoo data", id="panel3-retry", n_clicks=0),
            dcc.Loading(html.Div(id="panel3-content")),
        ],
        style={
            "backgroundColor": "#080F1C",
            "minHeight": "100vh",
            "padding": "clamp(12px, 2vw, 25px)",
            "fontFamily": "Inter, Segoe UI, Arial, sans-serif",
        },
    )

    @app.callback(
        Output("global-company-selector", "value"),
        Input("panel3-select-all", "n_clicks"),
        Input("panel3-deselect-all", "n_clicks"),
        prevent_initial_call=True,
    )
    def select_companies(select_clicks, deselect_clicks):
        return (
            list(p3_DEFAULT_AI_COMPANIES)
            if ctx.triggered_id == "panel3-select-all"
            else []
        )

    @app.callback(
        Output("panel3-selection-count", "children"),
        Input("global-company-selector", "value"),
    )
    def selection_count(selected_companies):
        return (
            f"{len(selected_companies or [])} / {len(p3_DEFAULT_AI_COMPANIES)} selected"
        )

    @app.callback(
        Output("panel3-content", "children"),
        Input("global-company-selector", "value"),
        Input("panel3-retry", "n_clicks"),
    )
    def update_panel(selected_companies, _retry):
        cache_reset_notices()
        if _retry and ctx.triggered_id == "panel3-retry":
            p3_calculate_nasdaq.cache_clear()
        if not selected_companies:
            return html.P(
                "Please select at least one company.", style={"color": "white"}
            )
        content = p3_panel3_layout(selected_companies)
        return [cache_notice_component(), content]

    return app


# ====================================================================
# P4: panel4_simulator.py
# ====================================================================

"""Panel 4: self-contained Dash valuation simulator.

Install: python -m pip install dash plotly yfinance
Run:     python panels/panel4_simulator.py
Open:    http://127.0.0.1:8054

Embed in the team's Dash app (no server starts when imported):
    from panels.panel4_simulator import create_layout, register_callbacks
    app.layout = html.Div([other_panels, create_layout()])
    register_callbacks(app)

All data loading, calculations, layout and callbacks are in this file.
"""
from datetime import datetime, timezone
from math import isfinite
import time
import threading
import webbrowser
from urllib.error import URLError
from urllib.request import urlopen
from dash import Dash, Input, Output, State, ctx, dcc, html
import plotly.graph_objects as go
import yfinance as yf

p4_NAVY = "#081423"
p4_CYAN = "#20e3e8"
p4_WHITE = "#f3f5fa"
p4_MUTED = "#aabed0"
p4_LINE = "#354b60"
p4_BUTTON = {
    "backgroundColor": p4_LINE,
    "color": p4_WHITE,
    "border": "none",
    "padding": "10px 14px",
    "cursor": "pointer",
    "margin": "12px 0",
}
p4_SCENARIOS = {
    "Bear": dict(growth=10, margin=20, pe=30, years=5),
    "Base": dict(growth=20, margin=25, pe=30, years=5),
    "Bull": dict(growth=30, margin=30, pe=30, years=5),
}
p4_COMPANIES = {
    "NVIDIA": "NVDA",
    "Microsoft": "MSFT",
    "Alphabet": "GOOGL",
    "Amazon": "AMZN",
    "Meta": "META",
    "Broadcom": "AVGO",
    "AMD": "AMD",
    "TSMC": "TSM",
    "Oracle": "ORCL",
    "Palantir": "PLTR",
    "Salesforce": "CRM",
    "ASML": "ASML",
}
p4__cache = {}


def p4_positive_number(value, label):
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Enter a valid number for {label}.") from error
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{label.capitalize()} must be a positive, finite number.")
    return value


def p4_load_company_data(company, force=False):
    """Return real, dated inputs. Never substitute fictional numbers on failure."""
    if company not in p4_COMPANIES:
        raise ValueError("Choose a company from the list.")
    cached = p4__cache.get(company)
    if cached and (not force) and (time.monotonic() - cached[0] < 900):
        return cached[1].copy()
    ticker = p4_COMPANIES[company]
    stock = yf.Ticker(ticker)
    info = stock.get_info()
    price = p4_positive_number(info.get("regularMarketPrice"), "the share price")
    stamp = p4_positive_number(info.get("regularMarketTime"), "the quote timestamp")
    quote_date = datetime.fromtimestamp(stamp, timezone.utc).strftime(
        "%d %b %Y %H:%M UTC"
    )
    currency = info.get("currency")
    financial_currency = info.get("financialCurrency")
    if currency != "USD" or not financial_currency:
        raise ValueError(
            "The listing/reporting currency could not be verified. Use manual USD inputs."
        )
    statement = stock.income_stmt
    if statement.empty or "Total Revenue" not in statement.index:
        raise ValueError(
            "Annual revenue is unavailable from Yahoo. Try again or enter it manually."
        )
    revenues = statement.loc["Total Revenue"].dropna().sort_index(ascending=False)
    if revenues.empty:
        raise ValueError("Yahoo did not provide an annual revenue figure.")
    revenue_date = revenues.index[0].strftime("%d %b %Y")
    reported_revenue = p4_positive_number(revenues.iloc[0], "annual revenue")
    share_value = info.get("impliedSharesOutstanding") or info.get("sharesOutstanding")
    shares = p4_positive_number(share_value, "total equivalent shares")
    market_cap = p4_positive_number(info.get("marketCap"), "market capitalisation")
    if abs(shares * price / market_cap - 1) > 0.1:
        raise ValueError(
            "Yahoo's price and share units disagree. Refresh or use verified manual inputs."
        )
    fx_rate, fx_note = (1.0, "No currency conversion needed.")
    if financial_currency != currency:
        fx_symbol = f"{financial_currency}{currency}=X"
        history = yf.Ticker(fx_symbol).history(period="5d", auto_adjust=False)
        if history.empty:
            raise ValueError(
                f"Could not load {financial_currency}/USD FX. Use manual USD revenue."
            )
        closes = history["Close"].dropna()
        if closes.empty:
            raise ValueError("The FX series has no valid closing rate.")
        fx_rate = p4_positive_number(closes.iloc[-1], "the currency conversion rate")
        fx_date = closes.index[-1].strftime("%d %b %Y")
        fx_note = f"1 {financial_currency} = {fx_rate:.6f} USD | {fx_symbol} | {fx_date}. FX held constant."
    share_note = (
        "Yahoo total equivalent shares; all share classes included where reported."
    )
    if ticker == "TSM":
        share_note = (
            "Yahoo ADS-equivalent count. One TSM ADS represents five ordinary shares."
        )
    data = {
        "company": company,
        "ticker": ticker,
        "revenue": reported_revenue * fx_rate,
        "reported_revenue": reported_revenue,
        "financial_currency": financial_currency,
        "shares": shares,
        "current_price": price,
        "currency": currency,
        "quote_date": quote_date,
        "revenue_date": revenue_date,
        "retrieved": datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC"),
        "source": "Yahoo Finance via yfinance",
        "mode": "Yahoo Finance",
        "fx_note": fx_note,
        "share_note": share_note,
    }
    p4__cache[company] = (time.monotonic(), data)
    return data.copy()


def p4_percent_to_decimal(percent):
    """Convert a percentage, such as 20, into 0.20."""
    if not isinstance(percent, (int, float)) or not isfinite(percent):
        raise ValueError("Enter a finite percentage.")
    return percent / 100


def p4_calculate_valuation(revenue, shares, current_price, growth, margin, pe, years):
    """Project a future price with constant shares and no dividends."""
    values = (revenue, shares, current_price, growth, margin, pe, years)
    if any(
        (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (not isfinite(value))
            for value in values
        )
    ):
        raise ValueError("All inputs must be finite numbers.")
    if revenue <= 0 or shares <= 0 or current_price <= 0:
        raise ValueError("Revenue, shares and current price must be positive.")
    if growth <= -1:
        raise ValueError("Revenue growth must be greater than -100%.")
    if margin < 0:
        raise ValueError(
            "Negative earnings are unsuitable for conventional P/E valuation."
        )
    if margin > 1:
        raise ValueError("Net margin must be between 0% and 100%.")
    if pe <= 0 or years <= 0 or years != int(years):
        raise ValueError("P/E must be positive; years must be a positive whole number.")
    try:
        future_revenue = revenue * (1 + growth) ** years
        future_net_income = future_revenue * margin
        future_eps = future_net_income / shares
        future_price = future_eps * pe
        annual_return = (future_price / current_price) ** (1 / years) - 1
    except OverflowError as error:
        raise ValueError(
            "These assumptions produce numbers that are too large."
        ) from error
    result = dict(
        revenue=future_revenue,
        net_income=future_net_income,
        eps=future_eps,
        price=future_price,
        annual_return=annual_return,
    )
    if not all((isfinite(value) for value in result.values())):
        raise ValueError("These assumptions produce numbers that are too large.")
    return result


def p4_value_for(data, assumptions):
    """Project price and annual return from constant growth, margin, shares and P/E."""
    return p4_calculate_valuation(
        data["revenue"],
        data["shares"],
        data["current_price"],
        p4_percent_to_decimal(assumptions["growth"]),
        p4_percent_to_decimal(assumptions["margin"]),
        assumptions["pe"],
        assumptions["years"],
    )


def p4_heatmap_data(data, assumptions):
    """Recalculate the selected scenario across growth and margin assumptions."""
    growth = sorted(set([0, 10, 20, 30, 40, 50, assumptions["growth"]]))
    margins = sorted(set([15, 20, 25, 30, 35, 40, assumptions["margin"]]))
    results = [
        [p4_value_for(data, {**assumptions, "growth": g, "margin": m}) for g in growth]
        for m in margins
    ]
    return (growth, margins, results)


def p4_create_layout():
    """Return Panel 4's layout for a standalone or shared Dash app."""

    def number(label, key):
        return html.Label(
            [
                label,
                dcc.Input(
                    id=f"panel4-{key}",
                    type="number",
                    min=0,
                    placeholder="Enter a positive number",
                    debounce=True,
                    style={
                        "width": "100%",
                        "boxSizing": "border-box",
                        "padding": "9px",
                    },
                ),
            ],
            style={"display": "block", "marginBottom": "12px"},
        )

    return html.Div(
        [
            dcc.Store(id="panel4-data", storage_type="memory"),
            html.H2("04 | AI Valuation Simulator", style={"color": p4_CYAN}),
            html.Div(
                [
                    html.Aside(
                        [
                            html.Label("Company", htmlFor="panel4-company"),
                            dcc.Dropdown(
                                id="panel4-company",
                                options=list(p4_COMPANIES),
                                value="NVIDIA",
                                clearable=False,
                                style={"color": p4_NAVY},
                            ),
                            html.Button(
                                "Refresh Yahoo data",
                                id="panel4-refresh",
                                n_clicks=0,
                                style=p4_BUTTON,
                            ),
                            dcc.Loading(
                                html.Div(
                                    id="panel4-status",
                                    role="status",
                                    style={"fontSize": "13px", "marginBottom": "16px"},
                                )
                            ),
                            html.H3("Scenario"),
                            dcc.RadioItems(
                                id="panel4-scenario",
                                options=list(p4_SCENARIOS),
                                value="Base",
                                labelStyle={
                                    "display": "inline-block",
                                    "marginRight": "12px",
                                },
                            ),
                            html.Div(
                                id="panel4-assumptions", style={"margin": "16px 0"}
                            ),
                            html.Details(
                                [
                                    html.Summary("Data & manual inputs"),
                                    html.P(
                                        "Use USD figures for the selected listing. Refresh restores Yahoo data."
                                    ),
                                    number("Share / ADS price (USD)", "price"),
                                    number("Annual revenue (USD billions)", "revenue"),
                                    number(
                                        "Total equivalent shares (billions)", "shares"
                                    ),
                                    html.Button(
                                        "Apply manual inputs",
                                        id="panel4-apply",
                                        n_clicks=0,
                                        style=p4_BUTTON,
                                    ),
                                ],
                                open=True,
                            ),
                        ],
                        style={"flex": "1 1 250px", "minWidth": "230px"},
                    ),
                    html.Main(
                        [
                            dcc.Loading(
                                html.Div(
                                    [
                                        html.Div(
                                            id="panel4-source",
                                            style={
                                                "color": p4_MUTED,
                                                "fontSize": "13px",
                                            },
                                        ),
                                        html.Div(
                                            id="panel4-metrics",
                                            style={
                                                "display": "flex",
                                                "flexWrap": "wrap",
                                                "gap": "12px",
                                                "margin": "18px 0",
                                            },
                                        ),
                                        html.H3("Annual return sensitivity"),
                                        html.P(
                                            "Turquoise = gain | White = 0% | Orange = loss. Solid outline = Base; dashed outline = selected scenario.",
                                            style={
                                                "color": p4_MUTED,
                                                "fontSize": "13px",
                                            },
                                        ),
                                        dcc.Graph(
                                            id="panel4-heatmap",
                                            config={"displayModeBar": False},
                                            style={"height": "480px"},
                                        ),
                                        html.Details(
                                            [
                                                html.Summary("Compare scenarios"),
                                                html.Div(
                                                    id="panel4-comparison",
                                                    style={"overflowX": "auto"},
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ),
                            html.Details(
                                [
                                    html.Summary("How it works"),
                                    html.P(
                                        "Future price = annual revenue Ã— (1 + growth)^years Ã— net margin Ã· equivalent shares Ã— future P/E."
                                    ),
                                    html.P(
                                        "Annual return = (future price Ã· current price)^(1 Ã· years) âˆ’ 1. Growth, margin, shares, P/E and FX stay constant. Presets are illustrative assumptions. Excludes dividends, fees and taxes."
                                    ),
                                ],
                                style={"marginTop": "20px"},
                            ),
                        ],
                        style={"flex": "4 1 600px", "minWidth": "0"},
                    ),
                ],
                style={"display": "flex", "flexWrap": "wrap", "gap": "28px"},
            ),
        ],
        id="panel4",
        style={
            "backgroundColor": p4_NAVY,
            "color": p4_WHITE,
            "fontFamily": "Arial, sans-serif",
            "padding": "24px",
            "minHeight": "100vh",
            "boxSizing": "border-box",
        },
    )


def p4_make_heatmap(data, assumptions):
    """Use the same valuation for chart cells, tooltips and headline metrics."""
    figure = go.Figure()
    figure.update_layout(
        paper_bgcolor=p4_NAVY,
        plot_bgcolor=p4_NAVY,
        font={"color": p4_WHITE},
        margin={"l": 65, "r": 30, "t": 20, "b": 65},
        xaxis_title="Revenue growth (%)",
        yaxis_title="Net profit margin (%)",
    )
    if not data:
        figure.add_annotation(
            text="Load Yahoo data or enter manual inputs",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
        return figure
    growth, margins, results = p4_heatmap_data(data, assumptions)
    returns = [[result["annual_return"] * 100 for result in row] for row in results]
    prices = [[result["price"] for result in row] for row in results]
    limit = max(15, max((abs(value) for row in returns for value in row)))
    figure.add_trace(
        go.Heatmap(
            x=list(range(len(growth))),
            y=list(range(len(margins))),
            z=returns,
            customdata=prices,
            text=[[f"{value:+.1f}%" for value in row] for row in returns],
            hovertext=[[f"Growth {g}% | Margin {m}%" for g in growth] for m in margins],
            texttemplate="%{text}",
            textfont={"color": p4_NAVY, "size": 14},
            colorscale=[[0, "#ff9855"], [0.5, "#edf0ec"], [1, p4_CYAN]],
            zmin=-limit,
            zmax=limit,
            zmid=0,
            xgap=2,
            ygap=2,
            colorbar={"title": "Return %", "ticksuffix": "%"},
            hovertemplate="%{hovertext}<br>Annual return: %{z:.1f}%<br>Future price: $%{customdata:,.2f}<extra></extra>",
        )
    )
    figure.update_xaxes(
        tickmode="array", tickvals=list(range(len(growth))), ticktext=growth
    )
    figure.update_yaxes(
        tickmode="array", tickvals=list(range(len(margins))), ticktext=margins
    )
    for scenario, dash in [(p4_SCENARIOS["Base"], "solid"), (assumptions, "dash")]:
        if dash == "dash" and scenario == p4_SCENARIOS["Base"]:
            continue
        x, y = (growth.index(scenario["growth"]), margins.index(scenario["margin"]))
        figure.add_shape(
            type="rect",
            x0=x - 0.47,
            x1=x + 0.47,
            y0=y - 0.47,
            y1=y + 0.47,
            line={
                "color": p4_WHITE if dash == "solid" else p4_NAVY,
                "width": 3,
                "dash": dash,
            },
        )
    return figure


def p4_update_company_data(company, trigger, price, revenue, shares, previous):
    """Keep each browser's selected/manual inputs in its own dcc.Store."""
    saved = previous if previous and previous.get("company") == company else None
    try:
        if trigger == "panel4-apply":
            if company not in p4_COMPANIES:
                raise ValueError("Choose a company from the list.")
            data = {
                "company": company,
                "ticker": p4_COMPANIES[company],
                "current_price": p4_positive_number(price, "share price"),
                "revenue": p4_positive_number(revenue, "annual revenue") * 1000000000.0,
                "shares": p4_positive_number(shares, "equivalent shares")
                * 1000000000.0,
                "mode": "Manual inputs",
                "source": "User-entered USD inputs",
                "quote_date": "User supplied; date unspecified",
                "revenue_date": "User supplied; date unspecified",
                "retrieved": datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC"),
                "share_note": "User-supplied equivalent share count.",
                "fx_note": "Revenue entered in USD. FX stays constant.",
            }
        else:
            data = p4_load_company_data(company, force=trigger == "panel4-refresh")
        for scenario in p4_SCENARIOS.values():
            p4_heatmap_data(data, scenario)
        return (data, f"{data['mode']} | Retrieved {data['retrieved']}")
    except Exception as error:
        status = f"Could not update data: {error} "
        status += (
            "Showing previous inputs." if saved else "Retry or enter manual USD inputs."
        )
        return (saved, status)


def p4_render_results(data, scenario):
    assumptions = p4_SCENARIOS[scenario]
    description = f"Growth: {assumptions['growth']}% / year | Net margin: {assumptions['margin']}% | P/E: {assumptions['pe']}x | Period: {assumptions['years']} years"
    if not data:
        return (
            [],
            p4_make_heatmap(None, assumptions),
            [],
            "No company figures loaded.",
            description,
        )
    result = p4_value_for(data, assumptions)
    metrics = []
    for title, value in [
        ("Current price (USD)", f"${data['current_price']:,.2f}"),
        (
            f"Est. price in {assumptions['years']} years (USD)",
            f"${result['price']:,.2f}",
        ),
        ("Est. annual return", f"{result['annual_return']:+.1%}"),
    ]:
        metrics.append(
            html.Div(
                [
                    html.Div(title),
                    html.Strong(value, style={"fontSize": "28px", "color": p4_CYAN}),
                ],
                style={
                    "border": f"1px solid {p4_LINE}",
                    "padding": "14px",
                    "flex": "1 1 180px",
                },
            )
        )
    rows = []
    for name, values in p4_SCENARIOS.items():
        outcome = p4_value_for(data, values)
        cells = [
            name,
            f"{values['growth']}%",
            f"{values['margin']}%",
            values["pe"],
            values["years"],
            f"${outcome['price']:,.2f}",
            f"{outcome['annual_return']:+.1%}",
        ]
        rows.append(
            html.Tr([html.Td(value, style={"padding": "10px"}) for value in cells])
        )
    table = html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th(label, style={"padding": "10px"})
                        for label in [
                            "Scenario",
                            "Growth",
                            "Margin",
                            "P/E",
                            "Years",
                            "Future price",
                            "Annual return",
                        ]
                    ]
                )
            ),
            html.Tbody(rows),
        ],
        style={"width": "100%", "textAlign": "left"},
    )
    source = html.Div(
        [
            html.Div(f"{data['company']} ({data['ticker']}) | {data['source']}"),
            html.Div(
                f"Revenue: ${data['revenue'] / 1000000000.0:,.2f}bn | Equivalent shares: {data['shares'] / 1000000000.0:,.3f}bn"
            ),
            html.Div(
                f"Quote: {data['quote_date']} | Annual period ended: {data['revenue_date']}"
            ),
            html.Div(data["share_note"]),
            html.Div(data["fx_note"]),
        ]
    )
    return (metrics, p4_make_heatmap(data, assumptions), table, source, description)


def p4_register_callbacks(app):
    """Call once on the shared app after adding create_layout() to its layout."""

    @app.callback(
        Output("panel4-data", "data"),
        Output("panel4-status", "children"),
        Input("panel4-company", "value"),
        Input("panel4-refresh", "n_clicks"),
        Input("panel4-apply", "n_clicks"),
        State("panel4-price", "value"),
        State("panel4-revenue", "value"),
        State("panel4-shares", "value"),
        State("panel4-data", "data"),
    )
    def update_data(company, _refresh, _apply, price, revenue, shares, previous):
        cache_reset_notices()
        return p4_update_company_data(
            company, ctx.triggered_id, price, revenue, shares, previous
        )

    @app.callback(
        Output("panel4-metrics", "children"),
        Output("panel4-heatmap", "figure"),
        Output("panel4-comparison", "children"),
        Output("panel4-source", "children"),
        Output("panel4-assumptions", "children"),
        Input("panel4-data", "data"),
        Input("panel4-scenario", "value"),
    )
    def update_results(data, scenario):
        return p4_render_results(data, scenario)


def p4_create_app():
    """Build the standalone app without fetching data or starting a server on import."""
    app = Dash(
        __name__, title="Panel 4 - AI Valuation Simulator", include_assets_files=False
    )
    app.layout = p4_create_layout()
    p4_register_callbacks(app)
    return app


def p4_open_when_ready(url, timeout=30):
    """Open the browser after Dash responds, allowing up to 30 seconds to start."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{url}/_dash-layout", timeout=1) as response:
                ready = response.status == 200
        except (URLError, OSError):
            ready = False
        if ready:
            webbrowser.open(url)
            return
        time.sleep(0.2)
    print(
        f"The dashboard did not start within {timeout} seconds. Check the terminal for an error, then run the file again."
    )


# Embedded overview.css
OVERVIEW_CSS = r"""
* { box-sizing: border-box; }
html,body { margin:0; background:#060e1b; color:#edf3fc; font-family:Inter,'Segoe UI',Arial,sans-serif; }
.topbar { height:44px; padding:0 16px; display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid #263449; background:#0b1524; }
.brand { font-size:18px; letter-spacing:2px; font-weight:750; color:#5ce0d3; }
.subtitle,#overview-hint { display:none; }
#grid { height:calc(100dvh - 44px); display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); grid-template-rows:repeat(2,minmax(0,1fr)); gap:1px; background:#35455d; }
.panel-card { position:relative; overflow:hidden; background:#081423; min-width:0; min-height:0; }
.card-title { height:30px; display:flex; align-items:center; gap:10px; padding:0 12px; background:#101e31; font-size:12px; font-weight:600; white-space:nowrap; }
.number { color:#5ce0d3; font-variant-numeric:tabular-nums; }
.open-label { margin-left:auto; color:#aabed0; font-size:11px; }
.preview-stage { position:absolute; inset:30px 0 0; overflow:hidden; }
iframe { position:absolute; inset:0; border:0; width:100%; height:100%; background:#081423; }
.open-panel { position:absolute; inset:0; z-index:2; width:100%; height:100%; border:0; background:transparent; color:transparent; cursor:pointer; }
.open-panel:hover { box-shadow:inset 0 0 0 2px #5ce0d3; background:#5ce0d305; }
.open-panel:focus-visible { outline:3px solid #5ce0d3; outline-offset:-3px; }
#back { border:1px solid #5ce0d3; color:#b9fff2; background:#143137; border-radius:6px; padding:9px 15px; cursor:pointer; font:inherit; font-size:13px; }
#back[hidden] { display:none; }
body.expanded #grid { display:block; background:#081423; }
body.expanded .panel-card { display:none; }
body.expanded .panel-card.active { display:block; height:100%; width:100%; }
body.expanded .active .open-panel,body.expanded .active .card-title { display:none; }
body.expanded .active .preview-stage { inset:0; }
body.expanded .active iframe { width:100%; height:100%; left:0!important; top:0!important; transform:none!important; }
.sr-only { position:absolute; width:1px; height:1px; padding:0; overflow:hidden; clip:rect(0,0,0,0); white-space:nowrap; }
@media(max-width:650px) { .subtitle,#overview-hint { display:none; } .topbar{padding:0 12px;} .card-title{padding:0 6px; gap:5px; font-size:9px;} .open-label{display:none;} }

"""


# Embedded navigation.js
NAVIGATION_JS = r"""
// Keep the same four iframe documents mounted: inputs, zoom and rotations survive.
(() => {
  let selected = null;
  let attached = false;
  function resize() {
    document.querySelectorAll('.panel-card').forEach(card => {
      const frame = card.querySelector('iframe');
      const compact = selected !== card;
      frame.dataset.compact = String(compact);
      if (frame.contentWindow && frame.contentWindow.setGraphOverview) {
        frame.contentWindow.setGraphOverview(compact);
      }
    });
  }
  function open(id, focus=true) {
    const card = id && document.getElementById('card-' + id);
    document.querySelectorAll('.panel-card').forEach(c => c.classList.remove('active'));
    selected = card;
    document.body.classList.toggle('expanded', !!card);
    if (card) card.classList.add('active');
    document.getElementById('back').hidden = !card;
    document.getElementById('overview-hint').hidden = !!card;
    document.querySelectorAll('iframe').forEach(f => f.tabIndex = card && card.contains(f) ? 0 : -1);
    document.getElementById('navigation-status').textContent = card ? card.querySelector('iframe').title + ' opened' : 'Four panel overview';
    resize();
    if (focus && card) document.getElementById('back').focus();
    // Let Plotly remeasure after expanding; the iframe itself never reloads.
    if (card) setTimeout(() => card.querySelector('iframe').contentWindow.dispatchEvent(new Event('resize')), 50);
  }
  function fromHash() { const match = /^#panel-([1-4])$/.exec(location.hash); open(match && match[1], false); }
  function bind() {
    if (attached || !document.getElementById('grid')) return;
    attached = true;
    document.querySelectorAll('.open-panel').forEach(b => b.addEventListener('click', () => { history.pushState(null, '', '#panel-' + b.dataset.panel); open(b.dataset.panel); }));
    document.getElementById('back').addEventListener('click', () => {
      const previous = selected && selected.querySelector('.open-panel');
      history.pushState(null, '', location.pathname); open(null); if (previous) previous.focus();
    });
    window.addEventListener('popstate', fromHash);
    window.addEventListener('hashchange', fromHash);
    document.querySelectorAll('iframe').forEach(frame => frame.addEventListener('load', resize));
    window.addEventListener('resize', resize);
    new ResizeObserver(resize).observe(document.getElementById('grid'));
    fromHash();
  }
  new MutationObserver(bind).observe(document.documentElement,{childList:true,subtree:true});
  bind();
})();

"""


# Overview-only presentation. The same live graph expands with its original
# layout and all controls restored; data, callbacks and user inputs are retained.
GRAPH_OVERVIEW_CSS = r"""
html:has(body.graph-overview),body.graph-overview { height:100%; overflow:hidden; }
body.graph-overview .overview-path {
  display:block!important; position:relative!important; width:100%!important;
  height:100%!important; min-width:0!important; min-height:0!important;
  max-width:none!important; margin:0!important; padding:0!important;
  border:0!important; border-radius:0!important; box-sizing:border-box!important;
}
body.graph-overview.overview-path { margin:0!important; padding:0!important; }
body.graph-overview .overview-path > :not(.overview-path):not(.overview-plot):not(svg),
body.graph-overview.overview-path > :not(.overview-path):not(.overview-plot):not(svg) { display:none!important; }
body.graph-overview .overview-plot {
  position:absolute!important; inset:0!important; width:100%!important;
  height:100%!important; margin:0!important; padding:0!important;
}
body.graph-overview .modebar { display:none!important; }
"""

GRAPH_OVERVIEW_JS = r"""
(() => {
  const originals = new WeakMap();
  let compact = true;
  let scheduled = false;
  let lastGraph = null;
  let lastCompact = null;
  const panel = Number(location.pathname.split('/')[2]);
  const copy = value => JSON.parse(JSON.stringify(value));
  function update() {
    scheduled = false;
    if (!document.body) return;
    document.body.classList.toggle('graph-overview', compact);
    const outer = document.querySelector('.dash-graph');
    const graph = outer && outer.querySelector('.js-plotly-plot');
    if (!graph || !graph._fullLayout || !window.Plotly) return;
    outer.classList.add('overview-plot');
    let ancestor = outer.parentElement;
    while (ancestor) { ancestor.classList.add('overview-path'); ancestor = ancestor.parentElement; }
    if (graph === lastGraph && compact === lastCompact) return;
    if (!originals.has(graph) || (graph === lastGraph && !lastCompact && compact)) {
      const l = graph.layout;
      originals.set(graph, copy({
        margin:l.margin || {}, title:l.title || {}, legend:l.legend || {},
        annotations:l.annotations || [], shapes:l.shapes || [], updatemenus:l.updatemenus || [],
        'xaxis.domain':l.xaxis ? (l.xaxis.domain || [0,1]) : [0,1],
        ...(l.scene ? {'scene.domain':l.scene.domain || {x:[0,1],y:[0,1]}} : {})
      }));
    }
    lastGraph = graph; lastCompact = compact;
    let changes = copy(originals.get(graph));
    if (compact) {
      changes.title = {text:''};
      changes.margin = panel === 1 ? {l:0,r:0,t:4,b:45} : {l:60,r:35,t:35,b:45};
      if (panel === 1) changes['scene.domain'] = {x:[0,.96],y:[.06,1]};
      if (panel === 2) {
        changes.annotations = []; changes.shapes = []; changes.updatemenus = [];
        changes['xaxis.domain'] = [0,1];
        changes.legend = {orientation:'h',x:0,y:1.08,font:{size:10},bgcolor:'rgba(0,0,0,0)'};
      }
      if (panel === 3) {
        changes.margin = {l:55,r:100,t:40,b:45};
        changes.legend = {...changes.legend,orientation:'h',x:0,y:1.09,font:{size:10}};
      }
      if (panel === 4) changes.margin = {l:60,r:65,t:10,b:45};
    }
    window.Plotly.relayout(graph, changes).then(() => window.Plotly.Plots.resize(graph));
  }
  function schedule() {
    if (!scheduled) { scheduled = true; requestAnimationFrame(update); }
  }
  window.setGraphOverview = value => { compact = value; schedule(); };
  window.addEventListener('resize', () => {
    schedule();
    const graph = document.querySelector('.js-plotly-plot');
    if (graph && window.Plotly) window.Plotly.Plots.resize(graph);
  });
  new MutationObserver(schedule).observe(document.documentElement,{childList:true,subtree:true});
  // The containing overview owns the mode; direct panel URLs show the full UI.
  compact = window.frameElement ? window.frameElement.dataset.compact !== 'false' : false;
  schedule();
})();
"""


# Embedded panel_styles.css
PANEL_CSS = r"""
body { margin:0; }
* { box-sizing:border-box; }
button { cursor:pointer; }
#p1-refresh,#valuation-refresh,#panel3-retry {
  background:#163249; color:#e8f4ff; border:1px solid #35506d;
  padding:7px 12px; border-radius:5px; margin:8px 0; font-size:12px;
}
#panel4-scenario { color:#f3f5fa; --Dash-Text-Primary:#f3f5fa; --Dash-Text-Strong:#f3f5fa; --Dash-Text-Weak:#aabed0; }
#panel4-scenario label,#panel4-scenario span { color:#f3f5fa; }
#panel4-scenario input { accent-color:#20e3e8; }
@media(max-width:600px) {
  .p1-controls,.p1-chart-row { grid-template-columns:minmax(0,1fr)!important; }
  #p1-result > div { max-width:100%; }
}

"""


# Embedded panel3.css
PANEL3_CSS = r"""

/* Panel 3 standalone company selector only. */
.panel3-selector { color: #f1f5f9; font-size: 14px; }
.panel3-selector label { font-weight: 600; }
.panel3-selector .dash-dropdown-wrapper { width: 100%; min-width: 0; }

.panel3-selector .dash-dropdown,
.panel3-selector .dash-dropdown-content {
  background: #0b1220;
  color: #f1f5f9;
  border: 1px solid #334155;
  border-radius: 8px;
  font: inherit;
  color-scheme: dark;
}

.panel3-selector .dash-dropdown {
  width: 100%;
  height: 50px;
}

.panel3-selector .dash-dropdown .dash-dropdown-trigger {
  min-height: 48px;
  grid-template-columns: minmax(0, 1fr);
  grid-auto-flow: column;
  grid-auto-columns: auto;
  gap: 8px;
}

.panel3-selector .dash-dropdown-value { min-width: 0; }

.panel3-selector .dash-dropdown-value-item {
  display: inline-block;
  padding: 5px 9px;
  margin-right: 4px;
  border-radius: 5px;
  background: #153343;
  color: #b9f3ed;
  font-size: 14px;
  font-weight: 600;
}

.panel3-selector .dash-dropdown-value-item:not(:first-child)::before {
  content: none;
}

.panel3-selector .dash-dropdown-placeholder {
  color: #94a3b8;
}

.panel3-selector .dash-dropdown-value-count {
  background: #153343;
  color: #b9f3ed;
}

.panel3-selector .dash-dropdown-content {
  z-index: 1000;
  background: #080f1c;
  max-height: min(200px, calc(100vh - 100px)) !important;
}

.panel3-selector .dash-dropdown-search-container {
  background: #131f31;
  border-color: #334155;
}

.panel3-selector .dash-dropdown-search {
  background: transparent;
  color: #f1f5f9;
  caret-color: #38bdf8;
}

.panel3-selector .dash-dropdown-search::placeholder {
  color: #94a3b8;
}

.panel3-selector .dash-dropdown-option {
  color: #e2e8f0;
  accent-color: #00e5a8;
}

.panel3-selector .dash-dropdown-option:hover,
.panel3-selector .dash-dropdown-option:focus-within {
  background: #1e334b;
}

.panel3-selector .dash-dropdown-option:has(input:checked) {
  background: #153343;
  color: #b9f3ed;
}

.panel3-selector .dash-dropdown-action-button {
  color: #cbd5e1;
}

.panel3-selector .dash-dropdown-action-button:hover {
  color: #00e5a8;
}

.panel3-selector .dash-dropdown:focus-visible {
  outline: 2px solid #38bdf8;
  outline-offset: 2px;
}

/* Selection count and buttons. */
.panel3-selection-count {
  margin-left: 12px;
  color: #94a3b8;
  font-variant-numeric: tabular-nums;
}

.panel3-selection-actions {
  display: inline-flex;
  gap: 8px;
  margin: 0 0 10px 16px;
}

.panel3-selection-actions button {
  background: #131f31;
  color: #cbd5e1;
  border: 1px solid #334155;
  border-radius: 6px;
  padding: 7px 12px;
  cursor: pointer;
  font: inherit;
}

.panel3-selection-actions button:hover {
  background: #1e334b;
  border-color: #38bdf8;
}

.panel3-selection-actions button:focus-visible {
  outline: 2px solid #38bdf8;
  outline-offset: 3px;
}

@media (max-width: 540px) {
  .panel3-selection-actions {
    display: flex;
    margin: 10px 0;
  }
}


"""


# Compressed real market data only; each response retains its retrieval date.
import base64
import gzip

SNAPSHOTS = json.loads(
    gzip.decompress(
        base64.b85decode(
            "ABzY8|BLi$0{?uyTdzG^Qr-DgP(5$9GUq*cvv4;r#<XnO4>o?_QY|4FgD^(j;>-"
            "8=jhMMsu6^nsRoHd*S^E+>GiF>zjQF?T{oOzQ%isRvKm74O{qx`d?eG8BA0Nl>|K@1(Xvg~f-"
            "~4U;>%Xb}{N11Y(ZBufU;h3de*eGz<*)zYpY+u^uh!0^pX>XY=XGBF>yLi-"
            "&%ghdfByU5|LgDnPT%mm^ZcX#_>cekN98EvkIw!l<@}@n`!E0Wclxfs`|E%C>-c^D-#`7H@BDxM`LF-s_k8lNfBirHxBtz*{nhXO"
            "_8<Sp|Ne*n`M>_^cYpHV{_1!C?$7`IU;XZj|NXE3G5+B{|M`E2fA}wd_80z#KmAXC?*IL>zxazki;w>N|M<&)7ys~||IfelKm5o4"
            "^6%r{{KbFz)A*kM>p%SEfAT;4<$u<X|9Abv-"
            "~Z3Q`~CkLe^sK_KR(VsKI&f{$DjEBzK`SZ_rE@C$!9(JY$Tsuzy9nzlF!cMvyyz4Jl{E!&sOqT@(d+;hLXrqlCv(!S(oIjOY(d*k"
            "+c5ttcyNQJ}b#*HTi5LpUvd6m3(&n`m<l_SxNLopS9$(o_v-(-"
            "<dpLNzS??XI+wKD9JOFOqO4sbxm}lCOT0Qov4XU)I=w0q7yaIiJIs{O?09rI#CmysEJP0L?>!uUuvQ+HPM%v=u1uXr6&4PQ++APd"
            "#K6t{n{?IM3-"
            "9P$6DgYTB3t3(U+F^s+Rbwmgrea^sFWN(h@(`5<k`wJ!^@cwM1W9;>TLz$6BIiEzz@{*rlG>rJm?gPjsm#y3`Z9)DyeZ6FuvRp7l"
            "h}dSaJ)VwZZNXFbugp6FRm>{3tcQcv`(CwkTsJ?n|>>51(biJpx_&qkszBe6Xr@qZ)Hvytf8Nc3eSc4;JbX(W0!5<MG<?HP&f8Hp"
            "~9M3+XQ6C<%bBe6Xr@o^*3vzgeQndr+*^kpXcG86kU6Z<j~eVK{A%tT*iVqa!rUuI%oW}-"
            "_o(WRN#mzmg?nds6?bZI8KG!xsi65F#9+p`iqTZxZbi7u_g_N>JAtV9P_qJt~ZrIpy8mDrw@=-"
            "^6pa3#9561%h#yR;G=T!{{@M3+`#msVnzR-%JfqJvjrv#!KuU5TDuiJo1FzFdjTx)PgpC3<!xdUhrHawRtFN^I7Z=-"
            "HL%*_G(amDsE+u~}E5XV<Uw>^y#Lm-"
            "wtEpS9$(o_scv&k{L*X_xryOg>AV?@XTWOyvBfUE;Hqe3m?4NuIAHa+c&hl;k~><oQbSd^M5tmv-qqe{GletR$b+<g<}{Hj~d*^4"
            "az4&wj0gNo||YTJl*>K1-hOOrEbKa+c&hl;k~><QYmPOLEpFIqP5RSyK1rvzC0;lg~!-*-"
            "SoLzy9n@<ovZ=O6uNcQujWSx;LL?a$djwtR(NDB+plp=PSusm*lKV@(jPwrRF|YNj|H|XD#`xC!ek4vqa8c*q4^{+*;Cei_b>#*-"
            "So5p5aWMp(L`D<g80_)+IUXl009g6E!*O`pdKarR`}+&n-Ud$!8<^Y$l(r<g@G7pPh-EXH7n9$!E#)oyqf+Bl#>j>ykV}NuHr(vL"
            "t6+lC%D`p4CKOYGPk%qAxYkmzwBHP4uNE_N6BJQWJftiN4fCUut4|YGQk8qDwW=rJCqcO>9q1bg3q~R1;lliN3VNzO=-"
            "TwM5TaqGv79mzLO<miV!j=vhnjtR?!=68q8;AJ-B+Yl)t<L|<BBds^b-"
            "TB2ty(X*D=o|gEyp6FRm^sFa())Twb6T8$CJ?n{{^+eBlVwZYimwKXSJ<+qC=vhzfQcvtsPxP!Ode##?>xu2@i7xe2mwp{DjYMBY"
            "VwXl@mqwyzBhj;w=*vj#(n##mNc3zZdNvY$8HrsQi7t&qmqwx!Be6Xru{|U4aU;>Qnb@A0*q)i_(oFPaCiZ0}_GKpeG8287iN4Il"
            "zRbkF%tT*iqAxS?V>7WYGqEo-(WRN_(oA$|CbnlLwr3``XC->J5<OdqF0I7&ti<-"
            "LL<d)*gDcUcmDrw@*q)W>;7W9GCAzc{yR;I!v=SX$i4LwrmsVnzR$`Y{qJt~Z!7H&#SK{NYM9;27&#pvYuEb_tiOsqaJ-ZS;yApl"
            "55}S1;HtS0C>`L_PO7!JQY}S?7tSiy8E77y-*ZM-"
            "A>r6f?$!9hBtR<h#<g=B0cK!OZGkLx<dA>7|^Ott%*ZsJb^y6C6kMn0Gk+UT4p(O92B+plp=PQ|<$$QB3?3cdr*ZsKotR<iI<g<}"
            "{Hj~fRuRl8zIe%%he%+62Nk6V7{kZrnlk@uZXTP*rzwXDiq#qZbC32SJJ(T1<{94bF+V)uzyHpZgN^0U~QWHOu+V)uzyHpZgN^0U"
            "~QWHOu+V)uzyHpZgDv2&7_3|^Rd!Hq-OG!=qOlso#tR}Mj!uIr}AJ>z9Tu)}%;<K52wtoHDFZ87+^J?)~OFrw#XUQ|1$upGXJWFz"
            "(C3%LDJVVK3NzU^ZHl!y#x1RLedNQvTpS9$(o_scv&*rZ``?bFO(!TVh=hl;+TTgm!@!3i~yMFyyNuHr3&rlLsN^;gEIqQ;~^{@4"
            ";COT0Qov4XU)I=w0q7yaIiJIs{O?09rI#CmysEG}!iN4fCUuvQ+HPM%v=u1uXr6&4P6Md<vzWmz0{IciPlb%~odTu@Gxy5HQ`D`V"
            "hUBCY9*E-k|9c+m%wZxCL#E-"
            "Q^2V0^qE%9S5@nbE~vzF*tOZ25Beyk;atS7qE6J6?wUFwNl>WMD(M3;J^OFgkmJ+VtY(WRc~QcrZLCw8ePcBv<N))PJJiJtYu_Vh"
            "%TdZJ4`(WQ~-"
            "%Sdd`NNmqY^lT)0HWGaqiR~GQ?HP%ljYQ8zqAw${JtOgdBhjUi=)_2D&q!>~NPOH#^lT*hG85Y~6J45#F3rTg%*4LTL|<m2FEi1X"
            "nb?<^*q52;%S`lTCi*fH`!W;zG80{zi7w4Vmu6ynW@3A0s!P9)msX-"
            "@E3rK*u{|rXJuA_{mFVC~bZI5FXC=00B|5kg9bAbnt;ENz#4fEw2UntlE77Hu*rk=&rIqO5N_6l_?9!Fkr7O|1E77wn(U&W+Syy7"
            "Su0+qSM9;27U#`SvU5U-Q5<R;TJ-ZTpxe}XoB{u6y^z2IX>`L|Jmp1FyeYKwS)q2ub>&d)YPx@*-"
            ">8tgmuhx^kT2JQHdeT?xNnfoeeYKwS)p|0o)|0+kPx@*->8tgmuhx@!wVw3VdeT?xNnfoeeYKv<tM#O>)|0+kPx@*-"
            ">8tf*Uacp6wVw3VdeT?xNnfoe^J+cmtM#O>)|0+kPx@*-nPuxqU#%y7wVuqm^<-"
            "YHCw;Y^^woOOSL;b%ttazpJ?X3Uq_5VKzFJTEYCV}(>q%d&Cw;Y^^woOOSL?~VT2K0FJ?X3Uq_5VKd9|MO)q2ub8%bYnB=c$`>8p"
            "*TuQrmt+DPWyMl!E9lAha0dTt}>xs9afHWH4=k@Va~(sLV0&ut|0Y9r~njil!`lAha0dTt|`R~t#sZ6rOnk@Va~(sLWhyxK^5ZX@"
            "Zrjil!`lAha0cpyj8a~ny|Z6rOnk<6=&q~|u0p4&)zZX@Zrjf5j|Bt5s0^xQ_$a~ny|Z6q9#Bk8$~q~|u0p4&)zZX@A{97)e@Bt5"
            "s0^xQ_$a~laq<VbpMBk8$~q~|u0p4&(`B1h758%fV?Bt5s0@Ia2FuQrmt+DQ6pBk8M+gd=h!eYKJF)ke}+8%bYnBpi_=>A8)h=Qf"
            "g_+emtDBjJb~NzZL0J-3nc+(y!K8wp3`NP2D~>A8)h=Qfg_+ekPfN78c}NzZL0J-"
            "3nc+(yC?Ig*~+NP2D~>A8)h=Qa`^$dUBiM$&T|NzZL09FZgGxs9afHj<v(NP2D~;fNeb&ut_<w~_SRM$&T|2}k5edTt}>xs9afHW"
            "H4=k@Va~(sLV0&ut_<w~=r}j-=-"
            "{lAha0dTt}(h#X1JZ6rOnk@Va~(sLULN90I)ZX@Zrjil!`lAha0I3h>Va~ny|Z6rOnk@Va~!Vx)=p4&)zZX@Zrjil!`5{}4`^xQ_"
            "$a~ny|Z6rOnk#Izgq~|u0p4&)zZX@Zrjf5j|Bt5s0^xQ_$a~ny|Z6q9#Bk8$~q~|u0p4&)zZX@A{90^C{NcwRj>Bo(vA2*Vo+emt"
            "DBk8$~q~|sgj>wVp+(y!K8%fV?Bt5s0a72!z=Qfg_+emtDBk8$~gd=h!J-"
            "3nc+(y!K8%fV?Bpi_=>A8)h=Qfg_+emtDBjJb~NzZL0J-3nc+(y!K8wp3`NP2D~>A8)h=Qfg_+ekPfN78c}NzZL0J-"
            "3nc+(yC?Ig*~+NP2D~>A8)h=Qa|K$nghzZtQwu6BVnT{@ZBZ$9Y}X7_FXPtA8)Py2jPkSFh)H8`pV`avs-LFW;lAHI6px_&RI(F1"
            "?qdmNxa1>$^<-"
            "+$!UoUuUV`y^nDo=P}z?(f9WA=(E*!tgo|umobaJyO;B8Uf<)WO@Bzg`D=XFg#c@e{xy$p{f2QB$u$4_7s8IAXQ)FGA?#WCUj4-"
            "Qk|^ItSyw&uD5qqXJm+Yu9TKOG?`!sBm8$>wTJyU!{X<{<YLaz)pW_^3ta+SYrIhcgZ{xo_rv9X)k@R)wPafa3jAQAcOO>pB^7pS"
            "&=hxJ6=;udkD`e%@NY-"
            "=JQ$JY0&!Vr&Y(uiP@jdkz=d7zN4tCC7*3|J930uFXB)fWBwegg7&9=_5t}bEw_pCZdJ>@yR%Be5=arV;JynL{y{z7kuo?5~pPOE"
            "yGlY>Qo<D7aw^ZcIYsrT0V90*&#ug;SktLc2kw;u5pUElFt^p3C6*3os&`d0njYe^&tJHOYQYgMVq$xrrk%&IdjdXlpAB$7ryd@T"
            "tp9l6TV3FvGQb{$f)S)0yaN!9fG^^xR~u<h`FbO>Kl!j)N%so!@YtX?1QUB9|6US>rn4k?>{P)e@9I7=CQ6dm;U(lPd94o;_iSI$"
            "EU(yayOb?6=Sqc_W1+BwHjdcC9t$9Fq*7v`v^G`8zi_e=j)n*FuPche*5N$OG1f-"
            "fD>AyuoFe%d0gZ~a=mVb;WbtxIS0dimqjWhj!V>JK2H6i=rhF_%Rga~-"
            "AWppdY0e9{IVwO(40N=%)1wVocARD9~vS;E!zQsVbW!rt{mM=zyI!eKSlaaTmWq(5=t5%rR&oTjeB_`14YJ@>y>>C@0P*2U~qsw3"
            "+lS#>Aoah=D|iM8^5uC7mqg#5aWZ=HKB`rR#r9YvS1=rwf-"
            "yJT3(#i5-AVXrpF!jkAXn}pRRp5;JIujMy?UNQ|+fBtAy3N<)Z*#mzV$0}LNc$qG}r|)BM1Bd<}YFcYQq*KEdM;Gqm4W9ZwWR-Ey"
            "@0mqkowe&-"
            ">v#4hJ0oKyRg^53j#ftSlrD0xdiTq+N|<x%Ik@w(oyeodyEyZzx1+<5Aw*co^L4G~y&t+QmrPe*ah^Kt&im5Iw~$e9NQ!jGo?WNR"
            "t3=?sEX%0#>RehKF9!9ThpwGOnaH?Wk^LHnzSnk9e?Z!)Q__L@dK`6~$ar3|9j8RBA!C2P?rS`?c2q92%!PEyqE5`gCAT6`kw$-"
            "R=w|!XOJDm5^Qo>`l}njj=HcqP^s;8siB%^5>g$}}1G%t_{qxQ1Fz3raExU--i~eYdH{S~?JS8}9zP-"
            ")WsHlbh#?WKOjnaLuy<N*nI(4IE4tRcE{5Vg&$!oQ^QGLm<=nPsYsh%7MatuVRG7!?5Gv2Olh!jX~m^**H?Grj7cc5$7j}i7s@9~"
            "!Is=T;3RsG;`eYfjnW;$+CN3MtB(lmXqtf%~rEZEn@J1!i!l-"
            "0*7MX9>)*3!aF#UD4E=xN=NmPnyG!X7%(qF>m#{<FTWlpGP?*cy)#2YZZ{$G>D+^#eusPJae}sCS_QmHAzIytqnBw?$@J9*b9@*C"
            "qv%(?4Vc&MLzvdsX#LbU%3cI%KImmVp!1>yx9_AD>7fhp8Aq4yO1j-"
            "9`#GqA#Sja{TyG*{tW=msxy=EPAouQ!lKpLr&B8UT2_ND20@vDYABYic42va2r>XzR3m^j{E$UDbp8KzgNOat8~lzQlz3sxpZ?c8"
            "6WxhlXrZ{_u^!#%)M^94^-"
            "!fPIcU0mqlp^+rD21TDVj79Z~7N$nxorD=LzMl(3Gd<N#$`&VJ&B51pobJwGY2bd$y{tZtE{lhfu}NgEWR$iM4iA*`&7E`eho835"
            "^-97H>buRc!p`tJJ6ve8oO5cWD>5tUB%eCTSIYmlK^{Uz*$h1KoV-66n|7D-"
            "vVPUMnn)!*y7b{C#jp@&{;1WHx|1yIt2rH?JFES_$bZ@OHoUZ-"
            "67VKpcgRu3Nj?~wAyKR2JO4BI>WyYy=?t$LUEjd6ZgnMMx1Nm$)bMOQ~taE5)<$zFO2#U;9DI$8aiDf2Z;voXVOuetPiUq!Mjiai"
            "xGwm4Z`M@3kAi#DzDSVgWuuVjZIJci?6-RpD7r-fzh5yB9f>5-4}j9C<a=u?@CrHi=`R<=?~B$M4OY}08-"
            "Zxt+|v#shAT?c~It_bwd-%ylTEvx*H4CsNAm2p!*z}>?O=*{Zt?Fd{a*~0`77%s&@7vCu1kX7+adugoHA-"
            ";LL0j#2QpVu6I)i$SJI8tOCdLqBrW(T6kPwzp88D2sWxZ<`@S-olf0Uh;;v6FIL>m<h20m&HXt=0=~&-"
            "L;RlI9JBJqa~!L$(t1$H5X<bFi|f3jeBptKRb@>e%;7d4=@yLRj4yT^zlFvaIZ~(E3vUpH97TMDqB`QDn$CS)~YLMK&Q}8(u<>qF"
            "<|{ZSrZ-"
            "i*ld_(|1M@gQHQREFGlON7CwB$2&)nVR;Hac=1aYMUGNG+i^G*+thh7#gNiF`DOjKj<)M}N)u(1x=jV1qK6~X#nY>odDrhz&LTxn"
            "R4rxK@m6&9YPBdZ>|D3+^BoJy8t93pY>e~|X{8v)D;7Kzs6LE?{@nMz(S;mla)}fdDRGiz*X@&okX~@%6(7na$~wq}<GL2<S2!{W"
            "Upm1UzeIW@C!t5tUl=+Q#S)}DvgLZ4ihLyPwO&_D5s_S-ZWRxLwBs48q6v{Y`l#hWb>ljziQFVtOp+=-"
            "If;qnzP`~|#TJvuOFt@s6fctINnAN}<)XzdPU0SVm=IQ<N?g4I>67%T*m_BuCZ0jkHv9c0CE{%LSjrp`0ck(t01U#@)XKU``yQI0"
            "zkOB3lRVi_Y%fu{l#K_KF_R6Dk07r@+9G|^(Jvj65?j6DDn-)c4NH3xU+Es$CbUbMr?6eWS1;SqtD@e4%T-"
            "!%MyQXt{rT=K?)lELB%mzxp?qg3Ug&!8qDwU<8iwL3q|&`7B(*N+o21t*#8omRzpe|WtEj`huo0Z*g{L5=A(K>4;7~<jhdyfL!E!"
            "LKWE(f8V^=gUh1=JRh)g<nHe0Y~AgdTqw(Gu0D8{K{n1kd)>4IWOA^;iq`qSDfE7dVC4z!~Q1c;JmC<w5yCnq=xMGXQ(S&jo|fhy"
            "QH$r@T%MZfy{a^*ENp*-"
            "Tl_30U;CVfB1X*VV{(_bM7QH&)GlR(mFeJP(y0HQaxxUHn=y7b3M$|PB(H1hu~?y_W+VU~s|^tY_iOu6HxxAH|u@-"
            "hirGKGf9b`P6d@^bW42cQ&MHgi1GKpDa*6QFzDhJ|hV*F;y&NtCwB`RI2!POK!h-"
            "N6=VK*78&#X=Z};%se5dn!6{lB7t1#|Y1$w=7kcuo&C^&bTgop`09n?->XC@C6s4j$`zbAXs6mf>}F;)7I-"
            "GMNu}5`NAIPvg;rhiKP_pT;zMF6hukr)cYH(eb@EZV^tZUh!>EROQXnC&dSEC6+JYm$9+`t)SY7`Rg^6$4rOeV-"
            "$CN_3>k3?eW!M=zmTjdPL?uV)>jE54OXBvEGjv_{1I*eS?AQn;QmQgX^5iT<y1&2p!05$dmfHt4y>_Ej9$(|Ta|UpA&Xs!DMHfoX"
            "}8X23^u3+=`YCRP8~UBL3R~MOOI0xPpH%S85|HG=|+|)3F-sn(I3`VzFNvJWie!{>~EEMVxJ2SNm0<J)wx_)ayei!e>;WaS-"
            "T?re)!u;nMR=<-"
            "aD+Uo<qh_f7sE1ZoTYH!IocSk!aH8^vab7S$AD^WoE^hyR3treA<I4Qlp$fW`^04JC&%^W$>`?^m|Z&oi3(1@LJ#9LHFb3D<^dgH"
            "PUX`PMJMj<bhq|kSJns*o}AJlq=|llRU@+V@FQMP?AszC+VfA@*GI0STMZUaOQQ%(-"
            "3azNk1Y2X|POm8>_Asnb&;RSac};eYjF@*;T@hWxGlYqU=ETs}ZQ`*>qg8dNO5PReiB9d>AsxeM_hCg=*T2!!9ILB$OyUwn}b9N<"
            "<=#i=8jKT8IuB!A;#+;=?#y-7f0B{)gZ0RXNl~RkF(8C@<8Z5;n^k^2RJEg#;Ox3tu}(*A<%-m(@y(YOw5@^aepG-"
            "8IGAI4IYD^t^gS1$WioN~&illuz~ITpE&AbR>Kf+ok%hLdk*4Gjf`pv=z^OInGwtGn}bbgp@(n^_K8uy9El8@___{7&F;SLI#pAo"
            "o2^km8f5F>e8rj5&@h(P0A$^dFkEC!JH+O^ATQ2*>rL;RN;$sqlW8&MD56|7k*Jj?=lWJ9QpwZn_c5_Eg7aNJ0-;-x-"
            "O2V=dyz?I7BXfwcFPj%gV`ww}RV0uAN6&ab+&L!Ye6Oi1^e_3IXVbUQUTJ>%uk=q&O8AO9``#b@fZ92o%i?JT%9%sKyoBEqRsB(c"
            "?NBfhC1>sgi{yfnU6gf<*eZ2s1g)!!e4hnuQ2m@s#4A`>sC>lY+MTa;pl}2o7}vbWmLSj;6|pQ!ACIgzlY6fNRN8+sK`{^*Xvm?S"
            "ogqu_}=65tkvUtUrm9%(Kr<A%(<I7$oPk+WT0Y|Iv_<-"
            "wjdA`|41J;*i~ekf^fwuL`?;zRu+sU210=s$5T2RRNYmV0tch+)C3y`IThFSIAhA-"
            "^iCyQpezzqW7kE=lDcQ+T_xtP&m@4wow$rx+<@c-PTnId*9kC@<Y}~Hpr#dEnoUG-RG*{k#f-"
            "0iSnydE4ogSP7a#tp~9Ex+Mk@Iv`m2^Q6cvpy;qW`q`>-0maR-o;_-"
            "WqN`GSdV6X8mU^I$RO2v4W5llTQSJHv&T_PA%$MhEE*mY=<{hLM#u@nIcwysjJ(8b+8TF;27^Mer8?|eXJ=9^GcVS(-"
            "&vT|B74AeSs=98X=mEbd^CLrj9U8gIq>%-A=x3Ho(CrV2WqL5Lq_oDTIjJN}`ZN=|3G@ReB_unj}o=jHW-"
            "9oYyJ*ko}C;Z$g#i@$MaAlOQbj77dBO>Yfx>F?ZRB|cxA>&6@f{A)Z{rz&UPA2OtL{T#2az=bfjYAfA5z6bv6Yv$<gHlU9di7!Bi"
            "mu&gkv%Q=!?7HxRMH1Us}(Kc4J%*OFP4t{lvQV0B5sU0s=k{*l9e*<zEpIrbW(EBjf=~<p7Kbt;+(5wB{c1G#1Gbi*{3k>W7UNto"
            "se^rs7iAy<$XN^mP-MSOb4;PbHC0dk>_$5puQ@YgeCHLCoQg>O&yPND}7$&c={&2vb(+NEK>%xQSqNVn^F(zS<cfjCl|%>5MK(fg"
            "I&^d`+D6m=^%9)PE(Jg2UW;q7pmMwo>wByANO9*qWDdjPIpPz=Jj=ec<qNd<wND_ZNGXsR^@18BC%ciRplADF}6`$mO`7aIe=c&!"
            "*v!`)03LJ-"
            "b}AYjR+beCB+%ER|`H~{^2AssZOGF8EjC8(txXyUxl!46oqHffwuaJs5w}@i&M_KR|~s?U^+zDp~8QhY`cBYqnJ&FS|=+7*e=d`D"
            "pNMFQDcYu6pToGX|Kwd&f&j;^?wxDWWN+(D!P&uA?$os19gqpd{|Uj1v#l5@}a1T{w0#44a?JXE!O4m`ttAH5#ZP9ZiaFNc}?DXw"
            ";h+wk8)eZsVx3O`$BJ1c1J%c2@&><2yn;f#M&viMA++DRk_A9lvbvr-zou}ouIN^A&{cyiz>x*LCw+r;aIFFzNdtbt0FX$H6N~0I"
            "PaG$&~q!=?!H#CRl~s<@CAMQke_V_C;My+xtdX35=K~w*YEn!8C8l$f_my`Pp7~-"
            "LXK9>higfnh<h%vccGPhQE3$aN2yEsi=I<Y3q*koRZ)>NO*d0nw1g*1C{gC)Xs=HD(lxyKY_{S64n3mtd88$?ddGk3nXoDwkvU(("
            "heupUd7VPdWk+xVS<qQK0OdLHA-aJIhIF>LlJ@EhD=MJ!Aq!3+6VqR*&#q-"
            "mK$T^X&aP`(58ft=R)BKt6E_@3yWW+{>!SlFrT<1NrC>6!lAUOiSnPCdlBh)Hh`-S-"
            "tenQ8SV_B*Xs6aj!<#oh+=G+;;Qx_UFGT64e7$?C9JnII<8<6QpT2d{WS6SD>E_@1$f=KystN!UV+$O$ND(e#C`p9U@ZgFN3UG+("
            "$|VI5MTfYGg7i9+W&tJcpsK#81nUKZbCFA}^mLr2f`8GEoj?@g&hI49O*~mr`Kl{4p?~Dm&te5h5b<#Mg+VKr$|=pKh_Uo1CnmDF"
            "iYTwCJ0uC|08D~_8=%7Ba<b8YLt1Vw?o6kkU$m(5SS0-"
            "HBdK2G9YpqehmzK>aK7Ew`p^aHYjo8pkc2K2T0zm0AevWx+Bjh@C^Z0`Pm~L|X7B1w0h@%?RGdQ6MJ=Im+3AmSTjlsMG;Mu`Sx(&"
            "ul5_$+&h}XdS|9PNeybv2MXU-"
            "x9By4ej6f$97u8#Yk_Qo@gq87k*UFmEz!u1r$jIu|Pa>S^LLG4eh!9%#l$D*s$kbgFM(U3S0kQ+HnPcCwXNJ8#>=$Nn5!C5<9K*;"
            "PQ*An2^_4T9SO_D5s3JA`pbd<Z)mFcQj;#~eLC0w*>CwF_2-}qO%Sh<-"
            "9jnlPqyLPshr276v)Q2_3TOi{JJK3m@URKbI_oKuC3RM6bU6W|03pMOjg_>h_@wyUwoRa0D7PztrMBd<ioq7XOyarb7UjWawUs6V"
            "k0KBNo+T$DLwH$Qgv4rd)GJQ(qla<0|6cVMWJymVoMP*G;BagrairWK*)U@m^dtJay9lbIR4A>Bx~Trwc<)l}R%*zG71X&Nu0<&h"
            "lZnt*{j*+8`q#;Al&x(?@D2gE#D@!i4?vIJw%(o|^e8|C5K+d}y=fO>m2Dh^lm|r<{iJM?-"
            "YCexsq+H6rYLoAiwd1!y<N;d7N985-GzX&#ha$kx@c<qeYjSED!v0Otl}eqrH>uuG&-"
            "w|3#P;8Me6V?pgVAQelTjl%9>7s=R;zbcfdlT83tTGjHCl)KcX#6`d-"
            "KZl~4Z0mP*IoWdLG;w^H&q+oQ!sQ(c$qB)y)9OWdiKM2Ch`)vqi>u}3tm5T10CQ1zrjB#-"
            "RKTJl}AHKG)Zm(?pAcwOLq?o7?yBVA7{1NVI@BwAFS!cjetYxXveAfSPAqlepwT%`l@SFKNCQoDKSkiFIsuS9t;<zU>bJQ62%MI|"
            "AmJt$E{=t=`v+P@P(-d=U3o@iC-vMs+Md~^(y!YLs`-"
            "12$2=lAiBw=n&R1n<rhWj}P`#KKG5^ILzf(ulmGnb>qXikKBJS{E9)93n#N?4Z+k8WvmVW-"
            "2M6559~BQG%eGB0u(<087C`fCoZY*D?dY;A{1kol-R=CIC36?UND|xmDf#Zu=w^))~0gdmfYcM&?$Y3Qdsd2lp_h-IftQ4OQXq6#"
            "9JlK6&ioZjXH3C@#`|*4jEoSeJn!R;a?G5#)y8JO848DL=8^2`q*53a+i%_+Kw*uAAT_p$GbCO_(G<2E|BP`cJLgFw7ZfDUR1L%9"
            "p8-9#~f0#*NA|$n$91@;?*U>m%2$7a>Yl;M&6SgS_2sGQz7W!*>yUg;V8mWo~sNe8&i;6gQD26rc#w_ET!nthMC7(jO85{5+-"
            "516Th=EB{>$R-B`#XOR!LP=p6k*kSj(#$MaE6S_cog2)+#0AK-Yk%a{BsgpccTu;!L17JtBw6x81@{-"
            "@uu1DjFX59(C6_s8D@Fha1@}n-"
            "~Rt(^jxrP03Smg<G(W6RfHWaHNEyi2%oDCKQ8PK@0=<9S}Sn<;hb~8A*D<Uai5v&LL`#XoXSDhCHN}M-"
            "ZtB|sEmj~Aqg)M1;PE(pQu=xVvXEFQRKLEdRCzM|~5fkK1sr1}z39!eyKOFHvrBr{rPmlsKzIq0k(n?OLGer&EN1$6vSS5Qj9DyL"
            "^1YXebbHHEUy>=>wCE~*#N$iJ}rIY*-"
            "sVF53*x$Zdb)S^2EO55V%4B_&GdOM0v9g1HoYY5Wv+Tm84gf%1$Oa;%n+58@a8n(K+`8@ux9c`rNA%%!S6tgaL)`g}uN~AM54aH3"
            "NEb2ZvQy~&#*j%|S}*c$x~44@R<?-"
            "tR>hElVW!kEniJQBDUARYtkdK@w=00}7*B~K3y!%a;Y9<0wneHiH~1vPz1jmC&_2ZFODX+!-"
            "S{wBCb??I2c06<l$Zu!0yR?ny7gD8<lE+k$7N|bI^8XD2OMsrRH=9B?k~V2MM0e$T!!?G8<Q@`vO6?5p4CkL+#{tW<Myb<F^_Pzf"
            "GR3Ilxa1Pen|}nsU6QK#nw9`l=^LGAf+7uNV?Skd?4YNqV|E=y1Thx>px<y<{K$%v(^R+wPzALo+AW^ODRn8;I3v53Na{c@xfMr+"
            "Emh3q%L?4O4J-I@Fjrxkq6R;ioAv99pY9XNYZ&9ERo8PZE|z)(W#aJ!J|x?+-"
            "H=v`WxtqxDOD<ILIWZSGYHT8ccj4bwOJM<s@TAWxu&(eDhtwlYCT04qu}dmrT}Ir_1yPl~J9Zv>5%8xW_w`LD{?9<lIPx2hiQ-"
            ">n%HVEs6_vVK!gHT&+ZTlMg8voF1#-M?n#|3fLTcclQI_hb4EB%Jhvkp+JBwt@Mgf1!WHRbIq_xp|sN;;vk+g)ZJ)?k1KYqM|10-"
            "`K)UJI@K-=oU(0k#0E0GEl?t1`zRm~t}9-lC8`J{y6%j6fY^_ClWaz}K|zp{*ozw=b+3Tue@ZK1uX^4Pf%$A1DTMoI!}29%-"
            "POVPbf6I=UM>hw3LjLprPHB9y3g)+leSjpp9=7bi!Vg_N)zd*%qS2jNRw-KpGr4Uzp_z>;B41pG)JKx@ei+TS2*j_IrL|BmZp)yx"
            "=S~Q;COh#i&Se64}f7%6ct?-iwkDzpoUG)#p(MQ{;9aWCw-H|4#>Z0dV@}#j%7UJb2<u|-"
            "%&>_;D}E$%Uy?k0<g*l0PFO7^oWrDGQDgWXxW(q02aa8;wvhDmh#kHDU{N#K)A}dFC!f#rk;20%CMdRx60Vda8HH$NI<JY%r`8P9"
            "z-QPlwG^FPk5r+D#I$TvdhVfWH7<AY*ggQWM%tk8&ZQmz<nI8hpDB#2dMqB3S2T>SJXg|R<=NKRwY>QD2Rjr?{7n^8&M(&lWlrO-"
            "qP2L)-"
            "fg2XvB^3k%Nr~q~&D31H2B2DYDXPtVlPEwZn&3MG{VnKkaO_Gp<cJvdkL#s@pH~UV&Xm1N3si^~(WTTKS9?z`+?=)`by4T4gA^dl"
            "jh6@$_PiY5{6$bl_olBc6kD=o#f=*HcL4Q9WmhGA%DPJ-_kIj!fInBV|qjr96#dGS*SAO7YnzM1LB-OT?vP+yPg81hM@&RydfgIl"
            "=%$latg_QFL<zBg?$xOR1Abf}q<iGpIP4BR#Oig9Fvus78JBnlFQzKmc62ftP&&9o=5QHiOZBB1aEby5)rJ3mD0Im)Q{ZBR?P))k"
            "z%+Eh4AAF-"
            "*Q$dNTx`kq$dsI0GKwgFQgB27tf11zD6lI8rN2g(=rkz7WyYB&r3f{lgW24=km>Q?LF^V*~v8o_}I!B_=97qjk|i!;&rEEUX0P-"
            "cu_!S8|PqlC@=Ni9eZ)pb&4sgxmtJ+jf8~NbYD%H!Njp^{xp)zpfADyG#VCM2mDqY3nildIBwbTQQg6WV9~2k?tSqvVsa=(xp&bk"
            "V@5PF7cy-Uvoquo{^+MqQ#u64p&;!fh@TF##rRVb<$s%0pltW;wvMkRz@hhj?e@hh#n*StcPvh4W#oEL#!jZ0&KS<Z+(O8P<5jKh"
            "+${=<v@$i?hA2%7K!nUu5hb6&Xvc17F>L`43-"
            "f>y8WzlQaUSp^L0I2GF&owyC^n>CX_cEhXZ<iDRcWBX6XcIOWS}$kR!u=Tn5r?II@0}anKSF-ey%Ga%CuRT~=JlrVyH37I6z8T;{"
            "QgVGB2$tD_3Pm2!T}Y0!jwg7GGS4(^_unQj4kY-E*>RQFdO$@}y;QzipX?j@c;4g`e(QOcPra-ItOn5e>U(N;JK9uTrHCDjdv<Ci"
            "*9q<BL$gtCfsK3W?eXk1;EkL<g;N!Xn}xwj#uX0)!>Xbs7q9L(k(=ni#1g<;a|?YI<<?5d3Qm1=Es|DrGnrAw_;k{J&g@w%(U{_l"
            "F?@uW8m6c62hOKZ>w2SX8-"
            "bhz255~&p2mivW3?1^f~uq#b^{lvw1w9Xxbu44c~f)=oVN`x?aba0TBA*D{I{d6A}I6+_$fQlB6GkYu!bY`(!^P~NCeu6|N{f3v-"
            "q$wrUQ(V9pCWXKXhUntxzD{NrloZHTUDVfGGvFxzGGFz?8@%#a)OHe+0IhcOWEtUWp!xY>Q<o=`$J*^e+x&Nk%6*_qpiDy%^&M9a"
            "u*Zk%QN;NaR|ug1A6ra0uz?Tk0cbn!sKH`NQTg*UyD_u_EWj|vR6dK+6|bXO%+JPKQ9w>Z^t6~XIC$NUTD*+fH$jhwh<%!a4yx-"
            "{#S!VNo(5^ax@&=_ks=qNuuSlfSm40Nt=u>-@K&t>7@$E(8KcqBLCPn%ZGyo%rIvC@Tgt~s?n$D$-"
            "qjYy_D9ql6x`@gC$qq=(z*kvM>%oFH05=y1XH?XfUgY_lZKs9M60gKHbX!f{*@3<S~87IS0twI-F+d7;q6__q$Q+R<zKS}e#|-"
            "Z_UDafUq;tb+-"
            "5!(?>7g{0OpJoBrnWFjr%kX(hsGgffABIIoy!;eWEDjE^I|me&s=#CQw{VK@$5i(9c(|LJJ7nbH5S<Ek}T0Zuis){pgKE4n$NwBn"
            "kPN3oM8!jVG|G+*TUj=b$%#f?Vs4NaV-"
            "madhp$$AJTuCF0BZ9X&4hpD6L_LMrxG=8c#S1)$4A{3YLUUKVpctHVhBD7)y~nPk*kGKLa7P=(V0W93SU-uRA^tgwnznnCQRXIZc"
            "^MrlZ63utMbCG;7({)^(49Or42z0QGp5R(N=cN4_jg%&14IB;F&BLw~!boXt3hpUqcrHmHEew_s4{qpz?z~ALw00`L$2=wv^_k+e"
            "Sbya21$J5X1v6TZVj3D@sz;s9@y^f9O4F4f-zz7MC-kz3#xP*)APCpV;%zal^P9wTsG3z`iDN_f5o`WNnU#GglNIex+h9Qe#p{M6"
            "7^kec3Kyh_Dl^E#9Xf;S+;x*Z2kLXrsEIM^2(cS#%FkW=v5kKXXp=r{(3vhzvy*_#|r95%PGmI&g4v#P?9=97XB5}T%`!x2y#0fL"
            "(<YJsr0tX`xWIxL)!PFAv59L81U?s&-"
            "&)uz}_;MQBhrs>a+IEgiyK2Tjw1Nn|^mmrU?}aX<xtXeXWFM>|A~`iz18`k30afXTYp<@;Zs_v)TR|zaoinmJ1cgoJpdS@StKScQ"
            "0t^;N`i@u#59U;`KzOhveLCCHpXJxdEZdw4#TJ#y34BqSYNSG#2T<9D#QPz^Efp}t0TN*H#-n2vm_oBJu60-"
            "X&m5T{)O0Dp=`^?Kb;BQnyF|1KiX-"
            "&#ZTo;`!w&NV=TOzr6`7|?tJVc+D?ra>U4Z?;xoiny8GNT6kjELcg6HK(RR)ZcaCeQum#l*nmE7Xm<H8g<0_}eq!S&Kz+$(rc8oR"
            "eDyc0rh%tQQP6fBKa*S9BAeSkQnCYN&<X$z%HTr5AT46nvio#b`<)bK`!FN}tq!_)<h+r}82B93#T5CBf=wHG{aqdR_QF?0^8^vW"
            "Zy;S)Ze)uFAtdW*^?Cd1X_hSQlD6VihsH7Po{o1>ZRq|nKrXCk{iV|a+`>(Ze;#R-"
            "{Aq}|m$8nmRBD60qGf~Pr8rkkFUk}^d^wRPdqIt+XNPPjb4!BnG*Z%C{+YpiCtisA`#C3<1V4pI@?1!2gfa%}?)WdJ+_>+5Z0pcj"
            "o|xlz`^Gts&~JXiy_!|S@?2G0!*w8Ymlg+&D)m1Fr6g7D*>(SZLz|D6QTHV{xCb!REIWb)&l=(p)*^hP#ko({qpE72X-"
            "e<>d6zEyB!GKwhv39U#6XGmS<>`(9J@zsMmt)B7Phuok3czn3G)8nIloWmk_60CNWD=dUZ50oi{?d1%Yi<p6f)WxYW499&V-"
            "?*wL%&05M|HyAPA_jMMJ1(qw;+8=601^P}5oMYkIMnA&vs<{8W8y<riP0G2m6^)7s|rCI12B$Ve!AfaasqZ2&yOjhA?0XCB%Q55X"
            "aq_Y7(J@Za5NaB;L&;4jW)@^QY4`C<YmfUu`Tj{L^=xBJz+t{=CK<F!FOkq97z$#!hr^#7gQNoX&T{xUV%A$)#`j&d3(pBHK>jD5"
            "W}`b3sjnCw@%Y8+zk)Y4A4H4XJ0#dd`#})DB3$N$5AjN?oOR&E@))~Ib6{v&?BFQ;6xZ1Z8-=*-"
            "0qU23glTexz@S~3y@clKAAv^P!NW&3y=%*i{Gy{_!%Ll-"
            "j~bPJI49p*dATcNjXgsLo;H?@<JRo;IlC1;W{?+O&rrj(N${7un7g<8Uyf@4s@ewy^aVYjbD5ZvAFzO%mZZ(sJY%T1Fh-"
            "{DwNfHz~{%bgx%?*kVsbqSOS5X3lGLU`qMDKZ<+^(O8}rp;7VPvUPhoDmoFM~YKK9d4Ja5!UGJw@f00}w8gSX3i4Ub+h_TupLyDT"
            "(o0p;<(T)W4jK82Z6QR^}@!=#5k6I$OOorU$g%z2a!nF-"
            "g)Cq${m$QwQdabxjMl7i*x4_MJCKjSnCPi0D(pNaP<6TW8Ii}3`KH=j7hDMF+al$B~*(RFak8$JVV1rS8r@|*RCs7bV;FBO%f>7?"
            "H!6s+sDHFM3o5(Y;^970J&{o4$7Z-"
            "&l{=G2fV~JlgX|AvyF|pZqC8R$E{xEREEp0`_#aLMchW_DVXuvo_8Le}~O?Mxv0ce2&z*rOM@rprt;`F=IblCKQ)tO9qhqJi!zVt"
            "OH>a#uzpqDvkYOu@g6Mm39*xPaGB5Ec6oo}D(;eFWC3i=KrmU!SEnmkPdzyJc+qQW2#j8g%O=Y*NqYzBcqy+c=l2Hq947?526x{g"
            "@qjqRZlF>A!eWefbkNg=QitJSdF37~G%E7rJlBwW<ihNDScmF7Ng_oXAI-NS6?FcZ9+KB>ADvY~^OBZfxdG6up&j4sLlP!v7;V@y"
            "f$tg;2Sc_Xnnf4Ypvj2iMDGL9RB0Cu6_@oDcy&53SCjlLd%KO8AWf!Rm^H;+qZ4ZiUKC*l%1Zip`j%#@-"
            "@@h*pGZ$z?3JO&6K<$V^oezfvrBp!5tXoQI|0_+9m{RG#mKU3{ClxXxG9c~PIXAnt;u|gv@4T6KpTHLw_dNA(XNj})fAYDY#W9te"
            "U&+HPy8}a?~gg=XU68J%5Bpu@lmLu#tTDica0y?t5gD-bO_w@KHTtx)IQ0N>l3K86fR0HHcbqvW%b-"
            "93E2^H2I?VzvBCi9WOF}N)UqI=iJeZuX-"
            "47;_~!2UW29`Erb&ty}28sRtStv4nFI~5^j1;{c@TJ}|p<B}4d009x0w!;+;$mD{Xk|B%)rhZZV8ICAFRHE}c0KBv<@Uu!@)(1l*"
            "!`GW3fv|@=ccbL2!v&kAm%BnG9tEo6;af!qF@J?hOj>;x4+ojpY|x-"
            "B*JzJ#zx1_iGx~Ghjufukh><Pu4<qpiCB_7~z9)g)aO)JHJp1xn&kR16@lO%}BPO=GyC6;9;Rs|(vkogl0k|<J67#NTqx3WAB<rJ"
            "u)*;xFm4mHw3*4Wv+WZ(>SPqq3K@Ii>{ip(q0QZw4mx}3JZefrG%hO5)K+jyK8HkPmRXNKrkW8UwGp2wxJ|ZEm0-"
            "ptutf!<uVzo{L*baDG&u!HnENtc_eH^g7Yw3y`b|gW?zoWsI(X><xH}*fOuQ=C!M8n^?tt_(fq^|GKGTB@L&9|&jW;vq+2KN+Iiz"
            "9d?AKXSD^<<n)9$<+-s20xp!F-8X+n`-1MX)jI#K3#W5a`eVOifVZVTToT5OXp5LA6&on-"
            "1IE(Ha!33t(CyxRhTW!yV6^#<^~~A{>5z6wCxri92fBR_&Q7*AJB4^xX#J;z1S)$0bT3Fw?*~sq<qLBQgz{a$vI;j+jHMcX?A+t~"
            "tzG(;QjwMpPy@g9!=83<(lxwHXoLRUWbp%I{zu>$nAepnxjGtM5pn5}}jen4V78qFio6)4;BjfTaVw?naTO7@b=wSk)I@P^C8y6H"
            "1u&X0Na?jAPnmz_4<fUI=OLYTt_7s8J@rvEshhPp46zKzEtY?_gYXGl^Ln?ibDUWp$7-"
            "<6z3QYZ#MCf*E7zQrvvKlCf6LF&(?h(jYHnM#!|uhzzMH{G?>D^QXv=v<0`*A1M6kL?7vfLJ@s=x?a>g>cPBov`Z|q?(ZG$I$gf%"
            "2Dnw~m>|>fr(ZH8_^zAkvN49cU~x0t^dzW0aX=8AoF6lJzjbrt1J79dhu45k#H@5~-"
            "~zmU!FMDt&;UlL3{5}Hzede{Q7Ai^fs>CmkMEZvHvBz3Sp}Dzuwuc0wU&q@lp<~)h`{Y_)RsvUyU{y4w4@EqA4gCE>dVwz6p^51J"
            "2pSrHmJh~yIyS*dkY2P1jO`Y3mAcSf~A;)+48uW)#|%+(9!qBHZ;Af4Fov11Gs44co3h~hE#vMi{obmahOi-Vz4W+3x<6f?Hi~|L"
            "TMKX@Vx9LmQgqIw3NU~;3%u{3J6Mss4H09h;P_n0z9|E>|6CyGe*rmUvi964Vb8oThJ$XsjR2f9kf_a+^Y+?i;>d-"
            "&mP9I&<vD3=5;RCfl7~Id%XqoXf*%G>8oe=a0?XF8yNo%kxhgST<%6;?F>8ixcTM{xOqsoCq5z%s_&T0^wX3JO?bvo&yszIn7LX|"
            "V#LiO8^zc~xA|xhlMO)Yg<ib0m5yJ%bJm77U6CudfFA(AG<TZ)LwmYl+QQZJZFg?`7#t|;BbgrrNT2{g$e6Mayde!T3tVvy9>wxc"
            "9*k=mX>AYe?v^Lf97R^LwnM@Au+r*q<V8$f3lJKpb3EJ+UHDB<_AxzVe(>&grq1Z5SfN*2Y5pG&$3PnX1hW<ck_!|W1@Ew!FI%sU"
            "ML~)CMl`i<&Ixjr5w4fJvOGhHqz@oAydjJwk*m&K*m*#&P0WA34Fc`NZcU&913Zs%STF8LaJFEa285>YA(MQ>*eU`t+|YvHfRlwZ"
            "4^(KI7_M==CSx2bbl^KO3U&>Ku3r#%IRTR4aF^%WvCR{6Q|r_9(qXChq@C@8{$bF^sXkz!l`gXu%iDmRQZZ=mQ1lq(#iF<+RF|oT"
            "^>P}dY7D{y{Bf)RQ9fi29{B(07*j?7)&-"
            "O3c4g1OeYnjZFfr(_!I_#x(*u7qSnihv1}{I*nMj@9kbdd+H<E6sG4jHeW`a#0Fbx9pzRerzv)3(rd3yC7JbKob!q;hZE3xsj5nt"
            "AaLJWXPd(P4x;(`wOf)LsxBXV<1_Qi=7k^SH<PBfS>`cc+lGG&i;NC&XHl&f^pXJl*~!h}2BzBhPrz?B&XPi74V{COup4^meu;E$"
            "=u0d>&}@I$oDk48(IK1(Not}(<}Jm$-=MIougM=F8=1$9K9rEkvC!**9bnwdJrV|J-"
            "V>Q8Wo#7`D)(}D%cw*8?j9uWSk12xRvpqxVHWtw(#fd)Kn`ifyedQ=Y(x{Yu__#b?f1Y}gKHZ75L<cJ_q2M54NUIr{7wi}!3I;CH"
            "Lo9@BpRXlz~wR6I~a_;!%F;8;ZvcKt*X9pzpO5>Tg+MS@{dNp=&vuL|yaWNqGLgr9qE>TR7KZH9Yf?|j0bYw~~K-"
            "QJ0&SH%Le=#BALye)!h*6GnZmoEBV(Z5s4}{OeAHiv2_KFG+^#HrzO368xTB0QF(y?3?YD1Ju?&z5UH}2>cUZ?Az_;t=0tm_8RGL"
            "0fBJVXcQ$`g$1$p##a60$oad*TQTQi*7ycF#;pZ@_kQGIZY#c;^D`0%ZQU#~XYF0PyIQxF64u)fv1A2RE<9?h4SMu9Xn(EUI>2z="
            "c2sEt5{U*yoX7xzjx^b;pj+qg~H@Vwe(d!9&ubEUAYDYUM#_KnC)*H+>$0lAXCwW<eM233HKKP~iJ;-"
            "!<3P3)5iAkBszQaI(pJG9m3Y_q&<JItk_t%x-"
            "v>svq+^b$tv@L(KQ4O@!U)qM1YzY>6i%RcAm~s0URbt(C&Br}@FTY%m<{;lqHm;!-"
            "t}W#*x1akOtT@K1pmWPyma?x{|@@JAJ@9Um5=VLMsHKzYUlfei8Qp&@{DIIhFNqvK^JuM7}2HghjVut{Sd3mi|**=Y}*M2L0l+WO"
            "P3l6OC*{1r~f(;w$q?i$I%Y1H{7SYi%!JyJ}%Eq-qnm(PNf2;T(Gz3)vN#aV^NXJVq4177!Zg)0tb-RrO}Mdg>|+C!l(LOpnj=A$"
            "jPHR9RouKfuD0AS2)&<<poSD~Bf1sd*NoWbenJc!~s2HFLcNH1Q*L7@^iF}NkJFjWT)h*KNA_R}!hvww{KPgB(uAUNTc+sELM;vR"
            "a9%B9PUr3-"
            "^0rLbp=0ae_;w<3Yw1zE8vF*srjwW0gPy#Ph{&PzsQ$U~u{(J&2xVY8<kbo{mo)jmf2h;tfMiYV%Q4jUox(`6V~aYYlYATaaud+P"
            "_~7KW%}T?sx8lqa6dL*5>e-{^14OU~f?>jMVpDN9*<kdRXg)P0`K?W{-9!GmynsCnH`$?~wjA4Q#D-LHG`j!95KB-"
            "76g#3ZzCEq2=w_b>z&GfLD^bxhN9wAEH^^Kmeg1e~%^s{!8;sQkKzF};$>F(Es%l)<z)_rX3e4#-"
            "Jv0pRiON*42pGEAhOJHtN^MC_qw1f*SZ05>ddR(LT4V6tMPa5J@cXF<w43a&7^D@U*pc=(bddUJrnrNf>VfJi%%{81Efr=T(NWMY"
            "kRk3bkf0Q$ge{{eZ}q)*coL79vK2Ps9kxMSAC2q91&n)wjIQxicO6g?gh`6^am%rh1-"
            ")%Rp_4B5BSWH1{a|D2nIOZPqJ;|)RbYNjozn^;!f*lT`@0S3oyR(~3?pu{7Z08>zn*qu8srs_msBopoO&p~A=_|zO|0abW`dm#r="
            "L|<;NW^z5>Ec|bS>c_@b&$j^YR<p?+L~?ZS;#@mJKJJ>>63_xXO__ovc#HrzBSMES8W=8@DzQ1-#*J7EYN!?zvm1pI<b@{Q55lZ-"
            "%lRyfc(2QwYoFf7*fa&f%L^keOp6hU0m^d`ckJdy-yYUC%R4@TgQdET4_3fsC6Y`N?MF^bzv6_fz3#s2<6Pl}I2coiRUzE4ANA;G"
            "M!`)Blq567uAeuGii0BA;iz=ez4EA)s#_GnDL2p{>^D{i7!fn7tVH{2z_@e{Mkwx9fm}c{4`P2>F3(w)W_Usz@OW-"
            "w`svI%x_S>iM9OyH9oEi(pa}`C1V1XN#l>0>#`-*--vZR;=)8=mA?hGl7634%z_Ag7HVe)ky*PI$K8dS3Ja-"
            "2?ANz#o)7F9}daE#GE5j&clg!Zkox2L*o(dQ)56$h(Mj)lGz)G1oaFak*A2A%Gdk0wqin<3Z1D-LNM{?^Pov=sU_;hR+g#xxH-Nx"
            "+c&Op%bif;Mo9U3S>_nj=Ql8PS5gIgO5BrZcnmbO!i^mIDMdjeMo4#wurV~3uj$MLdl&YZF>poi~q_3OPfYX|60l7$Kh{M+z0pl2"
            "_lUkJPuvmKp9CP2Pf9kGj}hzor745nk|C+T??lT;fM_x?T24!x01UjaD4-6<feB@&4Tf*PK;sxv#zjU>6Aq-#H`=lab7`ga)-"
            "c~_m@winva#=2{xwk=nok29&DP!8Ig3K*j>7XD;}{fHt7fiYajTru3pY_Avm;VZmHW^69CKXl_hb2T%j7ZWVD*r$gnq<T2aRU+@-"
            "qD&Qm+}a(@(<WlQm)W?CwpWc*OMJrAGfj8nGO?nYpzr!|#uOKo3`>RO3Ae=05-"
            "Roi@{S)d=XJ&0V@S;M9hd%Ghw6=en=r+OdD#iw8C}e~EsPN^F|--bK)eQ&j8dVsF@0_Aaz^{n*tQ{M69|>&Z6ls-"
            "D?+>Dea7?<u5zzrz;&VhCX1mksb^KF5P)CS-"
            "|4C;0Yd*P%zW*|yHY7g94|JI<{?vdPM%)xX@bek%<%&wXR2X6v+HxWz|SJd`EkU=U?(#v-"
            "gf~#r5ECvx{MkjbQ28Y*I0tbUN0wVk!8vk?Y?F7iKlBKdhHGRqAlVc@rUVY+w=ZxyPCCaPowP|=uxF&edJS#VQY1u;8F~zH$Vf@h"
            "#8Y1?BUzIl3sqMKt+B7J7#((+55J^Bq+RYQD94`syAxY`mN<FGV^kCQ-jSoo(x@#AWs9Fwhli=?GgClM23f(U#8!5jCJGoEinPfC"
            "fH=B>n39XuH)+ZKs(Gqj7nVI{pbZr=&<Y>L^&lWAKbhy23nbR=FK1k&#2zM8|JT>%QqsbeIU>}G$Wg+JT`;!EH6$r=3dAoDbZQt("
            "`!P>^KFW$vt9swqaUyl;2_Z4ioc3aV_IUS49*gZ^!g9xZW(j7UY6(yQ7n2;M%3$>XF`r#mNQ41e#;?puR+j*#IK~N#@<}^RS?P5X"
            "b?`Q8+O5((yQ|LS)q<0m9MeV%w;HwYWl`a0{bz;;5{kVhN15>{s=Eg^<V9*KG`qy`m>CgzMASK3W{}a60lO#wc$yj#zCZjx3G6EY"
            "g0Rs$X_>;06{ww0XHzU|6snEuZ=;QVYJqdV2!rU2KDvn$kZ1Jl6ATRja{Ur{ALKJ7B4{|cMryp&5IaA0r1tWYcnYUh&tJ^$HZ178"
            "Mt_cay;EspOiv+Mp``UrXBvk{eGZQmE({qZZL_tQmDqU6lGV=h#h1aY5<eKQhGCTb!d4>`76|p)k#sYBgQHRz#dx#D{2fewAZ*1s"
            "MN@7I2LB$www>RFFnn7c-AK%>2l-QR!oy`JXkS7P6e6KcaFiYWi-=d0fT|MXlh3FSLR&?l2u~hwi*-"
            "S5cLP2$AUBaxW~qx?v0y6*?0xFm<QKhrVJGEJ4BN%L23O03{Xha-"
            "8JW%<!#gA5f3BpW}W5gU>L5+%knY1hoK18>o^1L3s9&tm~MH25#u(8lcEQYCzAt5Hx=O8tiz8sA+g_srKfXWUqCXZcF&vE*8};=!"
            "Zz1<;Dia#2A3q5Llh}uGKzik#mc?F8+mP|qKRJ2y6yGMH=Ufpz7MMqCl(yGo7D4B@%<?;(<(1ukf)3bMXw;AT%V2D3$M=JQI>cY8"
            "mOnP9Hf7jQ9DD4XDQYeP;m|L3<O7;n>n^Ro=)c1gMN}%aZhS*!~57k@>AkLu(9qDJ8EDweF|H_OeR2OWD_%;US`aaXm7+ZeSpHpS"
            "ma~~*C=IIqd?fxid7KYVpE%33MuPZ?TzPr(+PkK+x&HR`Qh2cGLG-PrF_Z&$w}<Zsy1ZskzT=tv`nsk8R_RE9we5Vn1#Cm{%o0p9"
            "OvB23x|m1rN_9Onp3kGfZKVX7`^dEi$+T+O4F3Qp98R0IFsmSV!`yk$6HL6GxuuZ1<rNR9;SBJY;Z3|LN6OMYU$Tr>V$uwF@3*-"
            "ldz2OoeqUY5OjK3tFNVlh6it|_nz!S-"
            "yY&)nB^nP={=Z0&390+x%yT29FAFO;ku`H70qV4T*0~TK$69wEXFOtg&>=&gQe==gT;^xR@=mcjqY#(NMk3l9;MSl$iG9NU7Y^7Z"
            "E_3J0D@1T??${#4=sc7nFM+kg}sA%vZxM|V6-D}P9E|-"
            "h!CS>>hfBv9>qpbdSAHl6|7&h9ZnF+F~;&i=U1qmT9vlWJ^|n5!>TFg`F_zM*^Dtxs`Hp894BMy%RE-"
            "q63;hb7ncl5<+~kEom5dqW!Y171<?erxD1B914=uJz*dG2dPhjR0ko~TgY=)6`iI2y0dX8>bY8vw>jlaBF=he9*fQxK?Hxp<faPd"
            "E`tg%aWK?z`xhw&GID_Pf6`(<~;aRMAOc0a&%-IL$n92N}^U&*Ap6t`BKsdVDgK?8u-"
            "llA$TFa=i7ibX4QOawVx&J*w3YM_MoFiKaXG9mv07)DWXoMR4ps;w{FKhk~7+8dK=2hAGF?iFS5wE#eqPf;=1fi~qK1=aDDBG|jJ"
            "tTnC5A0Qh00!v$*VxSDa^F2uXv_{{S<#g83Dcl8vcedc`9>r!4~M>26%xNa6HQmD-"
            "$5D0HYT4}_gqLLGq}NUUiSc(7H!tz@V;=57@~1zxuaMVhA$V4$H9|CJr&l)0)xw@k}`n$^ma*VDUVJ-eeLes0zdYtVMw2&LiqJ!g"
            "qe2>&^u%T%-C#?8JBxbYIhb{31t@U>P&w}Lp>nhuKk$3nBEFNIK!PRF)J>29|?(F0;Y0=#GsL53&zgnKM-eEWd1u!_jrenEtuq_Q"
            "0<y8YC*^j-m@Pl#e2*YQ^s`r?qIq;SsW45OEk)tbhvuE=;7+y1t)AM@uMmR==FrV`Y3k^xP8L7p6iU3(H-"
            "51jeCoG{Rb*doX&2z!=boOm?1~tbkjOa$?33}rhvu*E{7p*4MA-"
            "S#Z=wYmZbI<tQ9}S)oU|#{BfN7cpp6gW3k>QqhEHkv41T#lAk>YzJmguc22B)jb8qXk;4ai#qD%2ODfEo!^4v|(J-"
            "DYXESDpv_j_EVUp1qpUf{TmvqVNZeo0lS*#~$Qrd~5m4yr#S1_M@0Ji<6p9!+NJL;lAQys@LTeD}Qd2@a9VaIqLh2_!aA5e$~WXH"
            "@eYr#9(=N9-"
            "0lq>7|b>L(e<@?6XQ{8EF0b*RQTO(XwjiC_)CX`b;E<@lHXaz#{h$#!uR9xkw#~=&*rEaVQ!yPapvy0APcMD)2<1k~_kIz^8?QPU"
            "b|JWP{#xppVDe1v0&2o|zwGoqa!%7hQ>?yKSaP*othv+gT;6{+Jnj^$~H7=FV(x^8l^952KMh#r1k(1|Bg(n|*di)O4o_^zloSy@"
            "JFpCkC(to8<6e&@`OOt~{fwDj!A6w(G5oC~Cd?2?lF)@!76WxuwI_@Y<+I`na#)m>o1F26)C9b<p@~Od3c|pceBt4ul^A+LOG7~)"
            "-3|&%bvHPG>t~`V8xmzJR+YT>7kj(@$^x5>_0&wmU)k_b4Fp3<-"
            "HOCQeBjCj`YY!*y?G_I*Kx*ob(%D!K^>K3DBEmq?yypzWfEOQQ12!k>89f1fN!u7XMhj*&GYwr%DnbpsQh{le0meClyX0pvl^||Y"
            "Ae!5}ew;4dga_;H^qxP2pi;)3S;#W>OJwuFj0F(pr=ZXzVZh6r9^X}}+;@mqjSY`HqIy;ncC|C=ve)oEs1PkUuq=;iHBIQcbFMhv"
            "XLG*HCvP}qD4CJ71?j3mTy!Em%EKxNTES5)J!4C_bb~#{ZfRkhV^d>9X(KwT)peb^c_9yPRROXJ3ug~*iuZ}d7Bt0uP+jf%&iWV#"
            "nQv4o$2;B?XJ{qarK2asFI@o7mEi7W671k(9iL+S5eOrQ8KYVelaJRuqSLsDtcPIxmDlSq$+b>HvYsp@=DFYd0neyDL~G{}Lv&^="
            "t4-"
            "7%#cLakyw@4&VRn(qQ!7w~6@~(q6E|jici2gN8e;a%!Nve=kA8ow_X0pfvb&7Xc#^SGbDPRGz>VLeYY%$R>bwm^@Vrey^n!vSo?Z"
            "|J2gGS4YG#+x3CxR#)2aGm{oT3m)(1|XA|$*=`tG)OR!k7r^2f&JJOfC$h{&WdKRXM0^3>2Tess7oB>mBSrH#JFukmHf&rFs82Vl"
            "0~ve}Te44B>EICtdHPZqbDGu#<5{Kw4c$GR-meuvv}{P+eX6OPF{9uD(1tlrp+^*tU<E3jhDfbaPlHaWy9Atw_wVN3LN4AR-"
            "S%NZ#!K!PBmaOcAdkBQ)#=|WhLW1h7ja9@x$9|lja^)#|4t2E4vzkz#pnPle#!)%-"
            "L7O~Dc)mH$<8Z}R^L|VyyT`YdBKkKD6P<Z!_tm=iLFEDVkEn_-z?HuA6k+0{@bx<U_-"
            "0Z8p2Z^8k_x51`=(xfP7c<Kd9LrqkT<K6AvD28KmI{HveZcoNe2DzD!y-"
            "oJRY6Oze>JwrG;e9sO~Z0W#K`bwr(kB3_ajSNHYAD<dL6Gh*LMm*DEHWZUDzGq(a8<`*STV$=7?Es=9P=#P{wffd!2YU&SiQjrGK"
            "#=1$%tOkOoN)Whm2c1x&U<&AZujKGz5QOw&C3{L!s?Dlfzx@7+;IT<)E-"
            "o_%J!cq{Q|i%Y<D@4Bw;Zw9qEv(LSl`i>q%0Ck`|V}^kI0~p?e_hm(4eAgh&?Hc%#bB{KDobktKBdCm68{m0(`7J_}jTR~abUeZO"
            "j)0im@&4u^8TT$?d0p@Qb5Vw(8}x}X&+{-Wh5`Z;gd?a70NuYqafAis4Ci!D3O-"
            "X|NMzOnWd^eM#R1Jc6|73J$H%NtkhR}Y;{g#}fQ^b9?<}(4HJZxW$-"
            "ULPB5;esNf`l4XAYygIr?S7)c3Fg=rrmQSdUF__uSycnLftasHA&D(xgxjxPiA-AV=6p#9vJuYLZ*he*M)Ww%);}69j>#uH}ef@|"
            "(d%-Ms!8@|OT-"
            "P;p$$JuV*odgks|jMX~FzUN(!zzaOsc!#@rZLf}nL$fgG=_Ilk?Gfe1vSC}@p&C9k*bpkyjc%=t(+iX$MdN7MC@!}e9KiTThXMBE"
            "rj~q7Rp8?mShlmx`)D6tt$j?n4){(%H<&AlpzBhs3sF`HSkYq%1_+0GMVqskljZ<Hx;vMHa?zX3ffnAoaKqFh?^i*jX}_N(e|7+X"
            "km|u8Z3>15`)TV>0+j&^;M~90ETshJu|OtNVaDQ_fSC~hJ6sES*^t4Q78x3^ivC1o#K6ot;}p#M8hA+at$1hN7vwOHA&}0ZKn2bm"
            ")Fz92vV*npG3hv|ZS01mnII~=f83#qaXk58b0GxGY;QQCuFMSPZ04;pXnL3^e4~=T00?&877<C{(%)>O=E?iu7X`HIh4ot{9-"
            "FgK;&nYH9gP4T!Tf7|l9PZ1lER3&NatDs9kNcyeV5HHVD7379nz7H{=MM>J#>N)E~qWywgJDRcM<mCL66XU8RWNW4BC{B^+=rY(="
            "nTAQU44dlojfD?H@E<=OE`f?O<H*UE7V<7$NS_|Du}8>Sd@Aq7c}al!Fqtc$}PY3sB>30_=qn+di<WpcT)OwJq@nPG@|?J`)7ci)"
            "oP`h8fmmb#cZl=iD1q0lrXc@7@H?Eb+Z8w}evMflveIi;ZtlLA1bco^;T1kLfG#)U4pZ?h}4&ZIJDV7RSaFO{RNZb6i)DRlyZoVu"
            "UaV^KC?s_e6mq0$RRzGMbU9!OCbF`eO#21`GPY^6%O<?E;1#Tq8x`=QzCQ3t_82gK?mI?5sxO2WhS+giZ#5&6N|&qofq}!=BP)P1"
            "9bXRl5oQdbM4nAsML$ksC=Wn}}FEQhdO049aJp^$%#o)5pb{9R+r=jmYHDkMR}8C(Ap%m?uPU-rzGVFPMrw$7lu$s-"
            "BZs#%K#sI`i5FyAFI|Dc<S93GX^SPcDgROPUcru=Q|(z)|A0OdnIF>{kZ73_t6(Y#(;w1KlURk&}+0nZvv9XaXIVa~@WGpI}gf{M"
            "pchU_A_H#{$67_LgEx`~q{k2SDGCVc79gV#ZY;Gu*Uc$7E0IKI4x9&QG17E!1M7sdy10aoz-)B%SD{*6dW@_DIu!-"
            "nbhr=K@(>vH1b-L4D_B$%@Ap%mB&#?$!$WH;lq~JJj)vEpb@P;=Fl%*T>KpS=M77U{e2?F~AydVF;-_FnP0&--B#Frp}4queB)}B"
            "eqlkA<e1gF6tEL@9F-?&#nzWBN~3<z^j*0$Kc;h^zjVU#`jtu?(RjJ5!*U1wpn<Om&rO9L-"
            "Kk{r$`V~>~zzlU&qa*qev(=XI{}aCgG5IpU0jM(hmZ&^Tf^iVAi3m51SUl`1e%l`EkO}0TK9S%h(I&sK?JuJy1-"
            "S4%zfxdtgqU*ILBgihq~*%td?NYi@kN=SG)1-8kQ?*M4wOP!`8Yte6l9B;$=L`gy+8cX(-"
            "65GPz*D<ctzFq<Rec5heu5wOMHyBVDz2~6lZB3|o0V7Ax={sDnmu)BLXW}<t~Mte?xn6_+s)b%mR(B_kM6pp?V3gIZKGgWgk7kTr"
            "%OZjyCt}6Wjy9Q1?D3g8X^uot#F_%Z{hl)fu&nSr&ll~N*e8SHN(|a6Q{2Ap40OybXJjuq}W(?O@Jw*dv{_-"
            "l?d#DMsc5_#q0oEXPSXNWlA!B!;+n}5<AdkqJeZr6xz}DY4Ul`8A_VPCZgGzBdz`;cQ=q>w%-`!(4@3sl7z3jDob!PywYvvC-cGY"
            "5oz*dlOM-*@5kk_Je#{JF9(I8*c^?+~3I}8BhUct6Pw#~nyKG;}_5`wOeiASUAU*-"
            "%x*RurWf88+R;M1UuR!G=oR$4H@I9CSw>rtBsBiO$c^nn(%H3Luq!gxnHGwJ~5t|0&1=rmexKH9~i+{;S|HVev@`1q-"
            "_6C5m<_JD?Xj`Ji8H?s_a)@vsvks=SCu)z=Hx~cN+dqH8O$ogdg({Cm^fXj%<E5j)%)0vR>laihi>j3a6FXRBNUe7wG{_zQbfUW3"
            "Ju{VC0Z92`lxB?&O=x%0VA8lu<_Sz>qAK*h4pRmPw7@p<KbKQMg>ezCHErv&ajPnDxg7cGzbD!)Zt&l%_mQyD%03$`CVD5I*D~m6"
            "zG}{#7Zpbsv$Wm<c=PIcCz8AyLFdjxkPIDWRWn7pHuSPd^3#1j}P~}JW6+qiwlNF%Z_CepObm`fSpQav_A@C^8p1N0-"
            ")V)5J%?fps_SUthOHPkdlDak5C{8E^fPN1zlYv%V?*xfyA4k(<uTng2h}Suv$-URF-"
            "$)#%awG>29851yNUDw#j<#467dZH9Q6wtMmcdZ~tHS0HJ^5GnNT=%XVy}yee3^wtd*ql4_nw6aretPDNaO7;x=9ZiAn=szR^TTrs"
            "@D(OfQf5+lT`rVWrV?mct(u13T@*FqLTb3Vlc%PqmE+}Sg{l3t)<u?mH}MB5l4%i5zk2uZY%bnjFSozWP9>ti~J}~BJzioAYC8#k"
            "^u^IuYlj&^sj}_u5z9b_A(-"
            "JS>S^Jj9y~1F`M)lhdCXY(OWdXWFY6>3`{Dc7vr;iauCEgk>zpx%TU|r89%<9uHoO@Kv70wX!8o%1j>(uho`lnMfvWGB@(qmWP|G"
            ")@Hrr{!V#8+670b3G(fEZ@Ug%$#gnZAzH{TzQ!@<b!g?NI)<`PeW&w#G!1YOha7}Y&9C60|J!Fo=WDxsWSt}qcV$`g-"
            "WKEGecA8n_La{ZFf=ub@+m-~6JEvbpuj>di(cWDbFKNeKd*t6DLzDCZn_1*{gf^R*JZ^lS)q-J-Gl;;)@v$aAHu7T2(k_^EH>mDy"
            "Cj)p$pZ<PV)F}Py_RjMUSc(2dJy_y-Z!NOK^HVL(Cr`-"
            ")pVb*hilspY?(@!;IcU3G)*g+t#C7jLfSIJ5FqQv?^H@~tShN5m@K2eUi`eXL!N~S5VS8eM=$JXyYrBspza)0OJcyK0&nQ~2jcfy"
            "t+9+K*=80*{5`ejdoi;b9?3&?pQF~bt2EPkh)FVKnQUnU26yP^cGMebDf6((C?8@<W@NJ483H|AE_<L}Oc2VCegToBb6+!Ecoi2L"
            ";RK(j6?{x>0#r}>XzQ-"
            ";w#T~^6jv%xI^QC~r9DElG0~uP2H38SN4(Qq<_9BjWJSSqE8a*1G3^zfTZMLdtrNXEav<!sqol<!pCT*nXqXXQ?4i1^NfOkuA_gM"
            "RQ@7zY>1)Q-"
            "O@(24yVc9BC=~rwHGply1v2WlHDFd?_?`j%nTTByTvf#r+$a_n)C2b%J<XKkCy)h0IQ)RRc^$Irh!Ggk9WLgtz11iaqgoqa)yDZV"
            "#fz{+a$s4Ol2P2njTeU~k8s&YA6+(`BBU@T*m+qdiG}%6GnQrZhvoj7xifp1CE?oa0eZ~;KT>KaAACnD`^R`@6m6l7a<D+Nffr%d"
            "7GK+jYA+*?tM;glR6F4@#<a(ppAev#$Z#upNdttoOv>}$AA<tXn7({*wm`iYhVAf!}+^fJl;U>Edo8j-"
            "Y{wwnpG@Lx({Z7pQPC?b}^^w~U_cvF$1Ge%=iz1}l+hGE1IiUQT%<3s%+YOYMPoikX*aw^xKGqAa80S<`1bhR5y0?}}+*rbJ#1ep"
            "Y1sLL^_2!&u#BQdsMLyo;hqL!jdUzLNqT?R>kS}N<k7<7R;q}ecA?UEk^OX==WKd6=qft(e$j0n(lBxsHrF%H%j*87$X980z#@j_"
            "-(+jO)#oX5AN57LeemsWK9p=@W6VBtFR1hh=o)-"
            "DBtD0$5{f)YfZB!~1^8^NM9YjpwqkPCFdX^G`9q_)#L4ThBibHl_R#2hKpt;kVAsxuGWcY0!(2GYI6E?h8*>X3pA<a}2LfA&t_O)"
            "sdflrDn!=v;#3-"
            "%BW*p(H=F|T@hk0)CLI+G4md|>$wiK+N_`?GzAkx(*L)?_$<WlT$`#rP!Q@8?H<19RA7PsWThR>V`6D^ot>U=8+wqyxA5;wo@*H2"
            "bT;!ANjn--U4--5i+Ii+Y@~;hEG#*U!60ezTG9&YMU1fTHP)yK^M^2RP6pcGeUMl3tJ{K^bZd*rSPknDz}S2Y8;RH=V>wl$_MAf)"
            "wxc(Xty46M4o}uJ??=`5R{9>%p9aq_aIoq`7h*5hh9BYf$?%DE`=6FT5rKzj~Oq#(pQ1E_SBE;-"
            "wSW`U}?@h`?%48VRAaVRUcLWC{kx7+H}x+F1Q<WiZv5D;GpqKHK%-"
            "J$n0C+(P&X*IUqKQje8_KtOR3#k;d~?P`#H`T(0Ajo3;zG6Ho^O&eb!e-S8~&VC?(Qr-"
            "{cL1<~*6{EGp@z32jKTnyi;iCwsH}<e|lK^1DciJXyppeaM3p?n%jVbcyX!*j8ZkAzd>;pv1DR;K)MClr20Phm+5np9G=<?vFW8%"
            "jMvf&7HBcP$Q0y9Xu9?pX(@nzPQsX1%#JI9Fr#cbK|IN^j|vu*-"
            "97mxVoU=WkQRPSo20h{TtC~Vwwk}?>UYCMf893jzQuaaKv2(z}kU@W(UF2<21mhsWT@3G(>J$NsH=S-"
            "Pv<@rFKOjiMq!@MN=nZwm;+7*^Dh(;oDl#D@uy15Cxmv=EYXf`@X5a<%dTD9|Y27n3!yk+K7^szemKl{gQ>>^0LHviofOQI&%74)"
            "oRZSD!4Et<-"
            "t^Y9Bt?q7qwG*;2FIFG7fjQU@!WnOR%dHQ72=Q1d8PwLh8ETkzs^)uakRa?|2GJ^c~9RMv>ww<vR)?7p};;p;kUG%G%RPa9Cm{p5"
            "e4T(s-eD4BR7Ic`NaBaSm;TheAhrIUIX`QRyF*R=x4DW#5ytxH-7N&tMa{0h3>y!^4?wpZ=nT3e6?q>wwLA|#7K-"
            "grL9$4nhcq=cSira2Xw8P!Cwud4Itmb#H86Xb%ctx<il%MSNbo03`=A5xTm`@5?2Rd({%mR4>FJ&yBGSxh#qdh<5<veXy`-"
            "uH#Q2aBB%=LN&_syCd)UYsFpt_sZ$;ED9UT881ruicq@<<kopgTjkKHTG3R_AaZgfVM)QV-q)bAAox&SNK9@$Q)R-"
            "fT_)yx!c2m2NK7+8bJT#p()2-qr^YR#=)DeF3JXl&hY>2L3c+N6eKDbGL03e$uqfkMRaXUYoUUFRf-"
            "V!Rg^JbKJ50qi)484q0h&C}QI<7Ms+A0Nk^sja-"
            "MC`k*#QUN5uNb{qUXgg>AA|J+<uq%_PX`v?5-{Gc3OUgEUt8nKXtz*_3hFuB>o;5Py45frh6!*_FDN_O-YV2Nu;Eo(WLW3lh~9-"
            "=nAGNCDM$uY!yceqY-&aqXPk@Bt0k1pIfK5Fp`wus#(+@^x`(U^$hb8fqYH6!eSSV%g_97)44D=+la$+XPJD<1iRCLIWTjm-"
            "!g<*sbAe(8KRHXxdU3455_ZmaWif5beVFr|x<7khAO)FXh<_q!VCRXs*@1|=p@ggnJ*mMA(|3|94+bpUTqLvRyx_?+o==y4sZu{o"
            "HY1Pm7nsOm=)0oDt&hndcN&R3Z1U2x@hy<@e@bA{uE%z5P>^LxdU)5Xj)=dt%6xtg(|88c*N_It1AYu@MHZp_Ls5f^~n(`FYw%@$"
            "kh=u!suD3T4MPIW`0!?SJj(`?N4#1LP6d%wRwcG@xi8W)zii`8C(pF|R`WvyUVJ??Si720mo$7ZQ7=q9+4fa2qvQI$R31MqjgTFj"
            "=vWm&yTVwU;*sKo>Cd7d-Q`=}cgc*eRGsMWmtf^lGZ`?+^o8t-"
            "<oafZI8UL<o|W6zj2b66dB2?h5hYe?gGuR~!|KroDdmHvcwcx;-"
            "v2zIibTjpoBKi21958aw$uY&MaX0XczuBUJ)t8QXN1K#AduD$mfvjJ20xw|uLEG_pwcpuc&jm)YMMY5>C9PXU*@zjj^eN*}GSZIB"
            "$2joJ&73DJ7YV2(8F^w_%{RVvlANU7r^N@~4;NFI3U0;OIOkF8ig2r75LWOstZN_{eOu`=hE0V<?{n-"
            "GmdFp>%0gek@9v`J`?%x}SdwDtQ*lbB#FWvwe?%%ZCii(O6Cwe+h80m)Kwoe1FIAqVl9;S&xEn?`vD}c?^MpPGYzVbn0>|LyVZK_"
            "#|T%UnEBKLPq^I}+}y*foP_SCvT17~bd)8m|8U1_(?yN-z&AF}>o-W*=0{qJ+H=R;<Y+jjfT$=BVYM=x4?oN@u65^J-"
            "4`$qqWje_9jjzbqrk((8Gdu{N%yU@z*J9bT$5>fTcu*n8txF4|ad2)bZaHt@US4Cw|;CoR&?cw9y)4?!5lW4Cm4)TM*z>oHkEsYO"
            "I{9>iRFcG%fG|=u(3NnYO-KL{3xtbODd8*k;(E^Ky!Pvn)nKJ2;@*Y!l!P3a{Kt*pr3)DCIp_6@LpyECvP()V3xcofMn$C|6<nv="
            "+)ddbh5&FFz#vJz;h*%4^JX91pc>w4q(+*zX^3|ZGC{6@u@ig!k*XDwoUq(Qk!A)}p_ZaPSn&^XLe)Joy%-"
            "7gO<$eDi=p<ymu2&)JvR|rUnuD|WxL5uJQ7M5E{GKCXqru;`6Ziciu(vg>j_119&jNCpew%R9j|iF@hy^D8SatLqK{{=uL1D50eM"
            "|i8Ay6rO2X_}F9lO$x6{CkvJ9+PsHSV?CLnY>pU@wqOuOtR*IqW&81Asl~thIE^1tkd_F<Yk&7s)}$di7-eXos=oeB%-"
            "#%9zG{%=+{v7bAwpd!pWx4m5ap7*8DXhtIu}ruwZi48`IU6ha9~5FpGDm1Sp<D@=1M_ZAH&H{lE3$l+jV&$=KAaWZEM^6I)3;^Fk"
            ">M=A3H$gOaWZ74?!%E^1d1u;D(3G-6BdB8sz6|E07io7g!V7j?>V@6q{Nl)oDV&OCj!-"
            ")B_aeEKZoiT_@FHR925;JJ}fWv=!6t115U@`lx!fa#-"
            "sFT@lHo+<9XskB#puhq+Wp>QsYo=Ml8M%hP(0iAHqlVO;1;IZ>?oS}GccO79l<zTa@KroQ30H>RTsW6K3Jf6uTcl}*Pl$ce)tjSU"
            "lir-NIX@0}3Yr<BVs+WUt%HbgFJuZXzIR9(+p=^~f{mBO?$6O#cW0Elupz)QfAE)*2tKXb98Y2-WQJd6df?PVgyAFNl;LAyOxT-"
            "z8^G`tbHc>U^dJJVj1O5hNCTBTqo}5kWiPyOIP@I!^f>xny1YexOh#ZT9wnlyc@#04OtDW;!jjd;<}c<V8P|;T8!iXso5Qoi(hQK"
            "vZEs<>4+&F!uVZ;aRpt|&04Dpr2PpQ2lrz?(Au=_Or^xjqxL_jO+j>ml7$9e+Ybl=u5mN9`wz__9_-"
            "J70584RPA!SxbCwA>wXhtxo`MN$D%qF_sfpo{5^>;u7)5Exb(lVb<JOUSRJ=)R-F<<bGd7B~KaaM^0_L{-"
            "<dsWq@{XTD0X!+`na)M9PYm&Wf2Vgca-9<!X3+x&Q7X#|56*be(vSDLToPrMHc-"
            "D>i0o<PbnFs1_eFt@gF?C*{GhZVGOjolb{kV3+)Hd{<YwU+t3t#sH7*oy)rZ%1A;y1D7CKe|Hd&RV}NoFAPQ=f@jE*S1;pxDx_Pn"
            "pl>nmhReBF~KkiS6rsZcv-<;Nk92Y{}Rc6u^kpyLnZ?L4nb{g}jt2yvb4b#=LVc5c3@jwhrEB=^r&uc>9lYFOA-"
            "&?it{7&W|P!*uQH<KPzNbd<Zdtmn(tYc@taOI$s+O4G0La+D(~)sUv}89AL)$1$Ro0utw$plk{#(n-"
            "S=ZabrC3>THoZ920r8=HUWA+L1OJ?DeiNGTR;1ve%6)tHxvRq<LI)%q3<jxr5P*j731eJV8_K0tqy*FV>|3cQMof&_>yz1m(5sow"
            "#~Oob#hPJHZLb<uRtZlcu)wyv>^2!%9GO<4r~A7{lsb;HEQO^GQ+GD4Xy4#u4jA^vmTk@Q#*{G!@oByN{2x2%O=)WzmOveu~<j-"
            "J8qfn30(JzLE9ncgH}(GPlVl^@>_24JRt&m}0uC<BYbu&XntR-TZ9Og)HWI&1-0p*=g$>A6xtwv-6U;O8by-"
            "(IOwu<&PHmkxdi+&$a8BL0ip_8|z~JXey93R+Ce+C@jGAepd<4D#yOlqy}~tw%WaH*F)1S7HxdR5RU?Ks5I>Gf+CeQBfzcuPCTBi"
            "Fp3Xv94rw~js2^H-WHT@R`Ym*nJU+wm)@-~rIN~O*zJp?2g{R#Td%Qr0$b**!Z5)3O0k36hQyeR1^%4(hy<=kFQEkJ$x-"
            "I@jM`D$M|txCdiBCeIDF!uE{JtcTLnsz(=6F8VUZJVrXj2^OUUodmuT>L?kyJFF>uxQdUJRlh(Qimx*Qol*D1jzG@2W0^)9YFnxs"
            "Eub|G=S$|o$uqg{Hs65}{?^8lYw!n)B_iWj;m?S@w<71)tZZ&GhwM~5Xl835>THby>lZW6oVTlZ99uVU!5d&+&zA1b__rHiYZAlm"
            "Sv5>HHDt-{+_W5Jfp&n9gLuEUb6m`IY${R3vsNZvm679%DFB2Z7-ScfZ6UspRxIi92^-u|j9Oo()!^D_jxp3sELy-"
            "4|z9;+x|moB?9Xw6`y&s`|=B9P`hsW>!_^$NjCMhob*zs?hpz-gGH51R?y-"
            "R`jdD9BbX((v|(_Uh|BDncaWk0Mt0Zn#(|5E~C)>x4s_+k-|6P*n5VWFi`eF-d-"
            "7zMK2^TEXL8tx#f=;3e_Msx(_a>zy6#Ib>o$`-=^BYsQmfV15j3U)}j-"
            "Gd}n|iBtn@9_+Sq3Wg_H=C&CzadHgiU3cp@YzAA@gIbnF=(0t+%WaoUX{~L7g`ek4TN-"
            "u(m_T=1y1lCt?oz+wPOQdW?su$UJZ^w<gK!Fid=*0YnEdf1{@lzpk{Qb501=s?*UUG2pT!l!@3Bk5g<hRjiRLsF$*06rfEF_(*KV"
            "YLpo@|CF+-"
            "t*hyv!^T&w=+d6Ncn$e@_+w+p7NwB6;mqb`sRxH}<%nWlOr2ADb5goA~Ld2tkW9^PkJKB`m%RWjfZ{?|WHLSDzSa(J<m_3{99j3d"
            "!FKjua+7DN1+BbGDJ>*t&Di#$)y`l`>O7uT*YH~SLQ=W=3qdIFw_OtK6Y-X6^|PtRV<Fzl@s^G!2-"
            "(kXwp8ZKJaBQPfQ+IxD|y{L+Ktw9h6C3>KWNK_rL;Y?_{E5W#e=gHH;<-"
            "TxGqfi<fW0i9Uf;5k8d=U8=VlpS)X<yY?d^Y+A6o4Q`o|?E|+=NJ8WHf1*!DCGnbTG1gdGg%|PyO|arJf{8ET07%YJ9vR3P8DDB^"
            "}Fr7=S)&FvUOz<mm)&p4rW$fR)A^39ZAihlpv*K4OdYv4sZ@Qr{d0Chs`GwX@37oMeH?^yWJAH-kGRKJTa@+c;xuP4nvXad=3~I}"
            "+%549b3?z+iX6O1eP!B!zSXlOER!vH|x4-#%~k=Y-e=)+A<b4G^q0Dc9kBQ5Ly9MqIlI@o#>LetN6lxG5*R8))|uO_oGvGWL=-"
            "G?-"
            "O&%stsmY*I%F{bIqLpBrubp58HW0Gwj)Q`@Bi8Kh=PNa&;$$JWgzX}H79OoOE5`zCF+JtI~L0^MSLAa_5^$LQkuWPmoV0Pp2dpfF"
            "68jv2&lyD|ObxTHk(Xg-"
            "3<i>~y!7qJr~?S*I52P=%~3@FBcmhVo4dEKi5U)#^yX{74r5%qd`QtBYc@Jw)D0cLO3;YWp#oaTja+2@QR&=`HYWe)#jANdMG)9#"
            "IBEufVqd?Oo+X>1TsY}<=QXon&R?0<FRw{d{5w;pB3m=diH7wmnyA3M>X17R+izeZJky7gmc(E6bExLUwE%5BQ%D7^AKr%6l%u1H"
            "Klhc@=VI;)_P5{&p!CuHYj0}<od)@6ir=?1_Vk_g-0@LIa^1aW)bl37F@Z{Il5=L|v+J--4>DkI~yOvw}D@<C^K=mP3c*Js!WgzY"
            "^zLpZtQF2?s}H^gE8D872Z$L^RHgJU|Is-"
            "T{n#wk2nOVymdaBc<h1iO>pWs@@ooZl9WZ8+<$i+K+LZ5mG+6+=eXtM=+Bi7McVLM9!5kpJH2M<J65;SHW(?F-"
            "`~ot}cv7%Gk+<OyWOyL(=P%XvcX;N;oP|KZ`z51<fl?-kibRx>E5>PZ9ZsDScf!N3*RPJlAeIF-zY*R$*r3!NpV1Z#~`HXAq`NS%"
            "zx%SYb>XJ^1@?n<T+Rc;vD!S((@_dc&@iQCvFP?X+_cy5b+n0-"
            "Zh;j+lu1z{w{mPO555>wKKHOY&!C{^7hz~a3m?Y+#)n45sBqmJ&yeqa;|WQ1l++qm<Os(gFqAqz_w%iN=QKH(<>pVln1nYEa8>1W"
            "{c2E#Spi{#D{b1LmJ!!YoyzN>tS`=-Y)Hbq|SAcQhXO4x%6=-"
            "CYeNB~pnwUec`K4qRysM8_>Td$sr={px~1ZZy!k3ND@jmin<%*xqtmC+5OGRdZ5iOkNwcE2&+CU8x7+5dmK&MfJ2RY}$pN=p-"
            "+kmf(JiqAdJg50O>!?oHcJAMQ~G~T3o5xJbm(k%I(0babVM3&nowbG*7Mvm`3aC#nVVy$L{>pR~Bo#kgB&LOrJA#P%cpyskW+?z;"
            "eYG9_>J<xf)i0K;`B&N=d07-w;Z2RqEm5V!$;_<Py<Z|Hq=i!s~AcI=IQ>K1fw);u9W6dodHQI%A&4!J9RKZpiOJKP9)-"
            "ZF`z4!`RAPi$&Rtu9b;~hexDy!cnyNshGj217Z*4)m<>!IYzHor8aTv%!>2IC8*d`D)Ap`5|!2GZk{CC;BtGCNY7v+Fc>$8>B`(x"
            "de@HIu{qAa(}ffhF(OTqvhSxecM&+Ja;5YR+n2Y>x?o)1Q*szmK10W*Xy(<_N~GQd<Bz?M*^fOC^ntJaXjX4M<TKPo9X&;>06d&G"
            "2}T*z@|04O)o8S4u2ib<rr!YH91pKOD&J;YG1Y{QM|RaWLeQEb|b_1_KvR8D?K_lp-"
            "%{i6u&5I;)aDC_PtcyUG%t^4pqzBp4fhlXg5JCP;BV$y7OJHT0Dq@v_$dPRInbyIzi#2rAhWwM&BBz3wJbWi0a+%1vq9E9^X4sDu"
            "?%`frhtyW^Y&2PilZ>vZS#FrmRue6IP?!}`Br0v8lI4<8at-2-"
            "Ij&7A<)aoJ9u{X{_uh2D6eVxjQXpEv|=s<p>QjRstkF7Ja(GxmjZ0np=q7Fj-3Tfijjk*`P&0}Z<_>fHv3eGX$2LU8?wnhG?s+-"
            "YAub?sTuw?9;cHUSW#f{XT!&)E}PdWD1`r!q}tc;s6XYg<^o2vTq@wC1rv+h~*ZYWrOWv_k}wJhT3KY(U#ybXl!PB?uuOVVG1ypr"
            "QY4p(P%^`_sPYh)B`_oon{>sCd$>5716*Ys$R}$xjOulZ`1n(0*{~Z#)u4^y5rkwr9pio8p0mI?9`a75RxE+1G_MnwJKWw|$%nN*"
            ">_bJ&)|SCLo$9q6KxS0Uq_ATv=Wcv$&%gDUtx;{MoH)L7ufoGiY6}bx+<*+WL<`%!;>o{Kzh}w$#tlgb}V_6S56>WUonU7R2nvT}"
            "lj10hoYB<vZ8Kt?Pb2)vc=FSt8k0W{hJ`)G8N)Y~^fwW*qP55<4yVW^56cCC8mZpT>o`PcVJ8`euV2Dg|98YJA2OH$Dq-"
            "r?lTfwB4leq;3WQ!5c`myH``5Ro1gqP!83aC*?rY0F+Cfz2RX<qHOKWct#N?VtlxKgS4kM0a1sEV}Q8j<qjH%G7!^7+eNxlK>hUu"
            "aCBiMpUa+y8joYa*gYOs>nc8tdH*uF-"
            "8!xRR{5hGYDLeXrSldOEZY55`2|G*5G&mY4?KP&y9M04uO)*U_v~+Ra;J{=#o%3708TP(Sk!+cMXEJ-"
            "I&88`CP&+0(U7(x9^H~?&G@iWFtLmIb$uSVx%XJzYy=`x0-l45%gJp&an=i!Dw)MX-"
            "u<XlTs4?p!*efA!=}dpd4{WTAvC0%asxPG$beE>!KgeB?h%9M-"
            "lclfD|zrreUgxkb9g>WRb%v5s#>C%Jiry2Al&NfuYk*F5e@p3`Cy>QEx5$vED{4oeFX9zXJQ9rc1`^=0BgPPL1q-"
            "tXoKCUQz292<lT~Id!{Z*!l;plyJbbHL%mu(&?Rlh7`^wL;111q-%EbWG+-c97g-"
            "$8MX4WCT}kj(=|hT&^MV|Q#rYs$?!lq$<<YAsbbXXjo~{V`Y>1LzhB$(^D;K3Z^`tUeY%^5vlnHTbf@l$ObXuzNio2E)r7U^=%%w"
            "Gg{>3xU>_k^Zn*@>WVl}u1{T_{%`vMd!j?DEY8jv|M-EFZb?o_nkA7=VX8kk5FIOXs)m^OG|5OGX|dK&vxMi^#H>dhA-Un4Fh!9O"
            "ao6a&eDY2~>EbRXQi=f9vq9{jB6XvbJG1q@SF@f0_P;(02&tD;?#bm{sndL`vj(Ic5DrW8;M_@LV;JI3_|+}W*X<K8kLiELfhw&Z"
            "@0f?~=0WE7M`{LV|`vNVcmz~5hH28<*lX?fkMrnD-"
            "c#+uUJe$%9js$>~G7H(kjoDgl`<#=Dy;;Z}|QlOhP*A1+TvO=P_j2eWVvA0y((P_J6h4en@M#;nhW>p$+Ym92?sGbnib#97C6j9F"
            "q5_vGRh-lMqw;tp58J^pIbvfv1@VnhWDf+}00&kJk#*tB!qoIjFL>JNFnu!{`U3oUKGqF+qZlFr#Y{e=EdKiZ#vIb`7XO9F+J8ot"
            "c(x%Yh9g(23>ln=p>DpA^<#n33X%nLwAPLl5OtnAtk;qPB?Ls?~6vJM{$dt@BmTnhQrOcg#1FJx5)^rV8K8<`?q`}RM_o=zJ-"
            "8TWMa!l*gqAH+zFAtLD>A6eGU^~YE@QXO_A_EcOB2AC@f)pPOpkf4=C|R6|_6@K)kc@aamIJ6}a26YGew&9Cb=~w}+)-"
            "eyj7u?YXQ@z&e~7?5$VWrI?yW+wK;>FaZ|;TG%L*gSY6)=~)Av<GIg$j?U?JqGk!=KWmmuj#(KC`YI5er&#G!!KaEWvJpVt~$MKn"
            "Xma`I42mvnpQRSK>c6Ph5_D4b#eaxatBHAA_<jW>ypLCD2eLsyPb7z}j&^O_rD*ShfEfnl@bil@1|qVdmJkuXtF^Zww_?9kqlxc#"
            "oLaVQL-Ygu|(VDz)zs_ohl6MCrUA{NJ*H2?$O=FTfE_{W}|=8DQS#r-"
            "LDC77KFzNvV;At;bJ#Xdk@71`n5eA0wR*vMAMYuwI)yYu<-0TN=-"
            "h$3D#7U_S?fx_B?c239s^fnTCnrqcYG8z_%k!(I&Y%)?qQb**8a`@g4%PCDsZDvNar9e6x+nRx<V7ZmzeHAUjPx+sXmi8oKF<t4!"
            "GLch`9C@vk1DbL;^yfPJ{AzGe5W<NcUE6#!IlAP5r%_nhlVp`P&AtnHHg*k-"
            "vnwC#WOSAd4cCXvGj6HUc%LIS)53e)o`aAqUS?BUK9xwod^2fej6xIj@~9d4*x5c;rbwgX7(c_quzfr#6XW=J%5&D`JBN-"
            "JF&>5O6xlK;hnO14KhJ-YS8b#KZoASnVz8^egj$$FJ-"
            "H$f;!1>AYk>1y(8B;#*8h_WwZk@Iz)N~CM;V@b+$XRG6PzEg1}n^XZIZ9(XC%IA%1OuHJq4&-"
            "`pQ6hOFACJqDeaz?^PR&Qq;6!5UY&5kB7MpQw}1-"
            "L*cB7>#@&0h8no$pnVgu9rAHPG%xa!PjlaHBzA3L9a{5O0}K5<J^@q!Ovojw))wOGa{;ko7399SIOEPYB#sB<msib|eexmH?M4q9"
            "&)&9V$va>AqK1gnuVhij&@xiu+p>+7Cc^W%9&k7c9XXOkA=~gGu|i9%8HixS?Nl9;j(Z$=Nn(0rUO#PrKw}%Q7#%z7Szcc<v`KOm"
            "jo*LI;Dumh>1zTm)xS8BL(+*AbYA@BdYnxeX+MtaKprcVn0QTER?b#_RX2>m&V{CwZ3J&(v(Z4CsIC+VaID5rWoC>!kO}R;fLu*p"
            "I-hu^w0|N;Hesv+{KMkHY{24#?#g<rqu%=Yr)$lxB+V75P3DLoz%0Qq!DN`y3BfvSUp1P!$aE4PR;{|>@u>T@WdFK^)iBw-"
            "s8<y!Gl~tWNbq|ik6B|tB<te)D&3ErhKt_f#gyq<N8j=~I2SbLDynZT+tCYewlM_Jy`eSfJGHd9Dyz}G?{z<~Ay})QiH=ZD<fBu@"
            "ELG%zi3~t_Ob&?e8rhjtm_P+RpVLy8pU^7FQpTMh$M|Yu(rkz$zv>w7LIom0H}eNab_|c5u7`JId@uUvap2vjPjPu!tyk?d*{IV1"
            "&wC*5D`VAcdVa|6*lAiBpWU{s8ZRE7qW%^?6910%hU+-"
            "=%vJ5UpJ=!|4b*=y3tMaXY7)8a4v;<{7go4FA5AnOQiSq216?x^0P<KB-oz}8qty?m`n#kFD(ntk2PkZ&v}MKXTRy*a91TQ?5%4="
            "Xwz=v($N>)7#?=?HfaJ*8GQu54G<fTUpoI1?EU<c~&^1>c`L4#08=#Hz!OQxCD_6l?J1$_wZx-"
            "z0!G}=*+2J8lePH|>|MFZh&yc3A9S18Y`Nk?tkn@8iZLhD2y~Hr8LZ43;C0-"
            "9HX>cwOGtkIkYEqWO70FK(r94alYBQ4Jee0601joD4g2=%?xWm5K`*;tpOHPHh9m=WmOig&5hS*71OT`5k<IjA-"
            "!MNdR)?ZB}dl!<=vMf1g2J^#ZbjePG-"
            "@jeR((N>mS443kDeRh)$DQ>^%llUdz5v<gTC+EJve!i_d6HDU?UP%z=r(LZ)mZea_dabA;(v?)2pPlu8yG*-"
            "3qXAB$J2g#xoq&3e=;kgt9dUBA@W&UcMBPTP0HdDcj^ar9}=Jd8v(Vsq89oyO4nP&XcZOiE-cpleCkbIGGrg99uz`+Wia`|g{W~B"
            "b)j`*8H;Qk0Sc%k@=~*nxXqf_>Z;WOx%C*FLX1%`2};}qsuJ)Nv0+21utufE5vgb*=n?tZ3ZN|;XS*@`Gkv0P{YaY(7Oaq2npH=}"
            "F>|_6V~)@(N^YKtoi>lW%Qo!2n7qm`*qIN#sz|_-"
            "qKhPKq@_y;O&pDhKpWCATJ$TCM#Xl<W)T2CfiHwcXvC49+Rwav^396J8(8R;6z@c0+Jn=(@A@|Kwj}R_9)&)MIhagG`PdzqBQF-"
            "=ed|ea8kjxHZ~Ye5KUJ6%;OI+XRySo?FW@VK>CDRc02IA~HEXfk7E=Saoa*%~Rriz#aL<NHzYwe0vN#@(N^pRYD-"
            "}?*5LZ(RaRrc!R;3u1=dvEX#;O6=vKHfK84<!imr9VHyf&5rj#j<po{OEVN;ext+p5<V17O=A=4!5hSgS-"
            "@WQ_+;CJCJ1o66#39X9C8(6LCpn_-1^k#-!~;>3YuFNw+Ae64C`>)FAeHyQtoYmyJpy&dBB&{epA3FbgueJk~S6Q-"
            "?u7lCVD=>x><3ZMrd!`NZ&&-_~Ckxu<hQAn%hAbuY&b86gldG1Kbl4Cg!eDou=@7N6n!bz#vvxxUS9}x*>JfFERkl(l}-"
            "GT^;ZE;?1Nzi!JWwByn)rz^si*c3*G7B`Bno`cIfX2tR!~@d%vI95MxRZW~>jMy4`o0PF-"
            "^(cBkG7rW6lKv|oRPdOwXufQ%%H5MpRbB)NMSN*9&|X;Z?sJB^fcJNrCR|kycBj}C;;<l%JS_wu(6NPSOu$q#U0y4o3LcsT1QtG8"
            "Vx>!wPs-pptzZj2rAOWlxssDHbB{+*Rp7&5#pFmw#dy$VRbqp&xIcKsKbu1z|XpHo~Skmnuv&!B<7;`M(x`VUdTn|l=E<jXW^C<o"
            "1iMXO4oZy(B*uE4iG?K=R>2;G|c4TyEp2bDF2^*BU-JDx*PLMHhOb<R`Ym0G>eM0S9@>(e=reTk#{;08x6@_3C^R_5&35ENJ$%wZ"
            "D$Je!qV&C6jnRUWb%`ejD;=vZf!I!CI&M85yfCT9Z+2s?jWz>-"
            "bsAhfkUOpPAZyD2X$pBS#$8H;4m?Su6JT&^5~qV&=<Ag1REnw=Fwgv)LhcSoq5!!gVG8jPmkRHST#L{2cY_zt{!uexhuINYC`yw_"
            "tT@1MZ}JD2GNm*57MOAYjk445JSTcF5b}^09riV^LM(;jBCo?Nh#<Ql|!(t`uv+0#bcT#A|Jrz7)VL8NtE#P?Ij_pP(U&|%X|S<!"
            "+7g&6@<0mE=(X%=X0sCFym>GfwS|fTU<iCRlDL6Lco|vrLmr%QZ9Y5G>=aOLGw9Al-um=nIcFRPc)R^m&85mBqJ6TPXsh?ldEZD8"
            "fvHpbNoupFwik%f?%35-;#vIV^@t%LH48KzfC@=&n}i<q<c!lsC6s=(a}q+`^kQp$b4n06;F9lt#wwVVfjhBao_+nrM`IQQ{4-"
            "Bp+gZTtvKcXZw3k}CN4JYECuJSSF1Uy5x3WB2wbY5c0yCJdixpyGZaK767WXzrF+&`MN6$ustgq=njo<<EQ8fW{_1oVL@1CZU&$V"
            "VF5Mu5y4^#34q^ynr-"
            "H^D*Cz3LNAu#^!yORA+YNd}QsyjicjG5gxR?EIvnrNVCF7=07+vV{3}jR5GNoe>1p?Ya>mH3X=JND`UH8R}$VOPb51fmDjyJX_@3"
            "qXBcKQM_`d30du-"
            "&9lWCH9hRjJGRtkrhUk2^c!bMa8zFbyQKc;S&?pdvzgWpfuQhCaeg&Bj(kM@kBByj8#Cen_sbHOAV@lxZK6aEjx`ky>p+grpL*hT"
            "<W#wb4ozyMy$NsEhn@HHNA+*#X_4fPA!S0_ee(!r@C|0${Ei+zqevPUMZx6rU<J`*W=`N9!9@kEttIBjIf{RVkVQh(l2U&EdE?m`"
            "7zZvVU|~7TXrrL38Sj`YrYE$PoBpt0f4Yb}tko`NkF=Vin`L+)Z1C58Gsq3}a&*N9K4~Q>Z~^st|f0I1AO}l6ui`^?OSp$bp9ydA"
            "$&YAabB`Ti~7HdM9m-"
            "Cz0kE8U(!LB>ECwX`A3CMB;mycI1`Yg&@&^J_QN@cQl)L=<`%$w4UaDzQ1OQlF``V3sq=Ycrv8GwMghU9p5w!@1Bw&c0uEQW;Yox"
            "m)ij3vky<3jd`xRGTP^QbXuOnr0XH=c|Ya{qAd}lJRv1<yYREtWI@86{e$bh$~=n<$eQ?%&)43y$_njUgA#?}M*O!ULoU&k{a`c="
            "hT<|5+fTA!cZ_}&-$d}o!+eLci+^A^RaASB|B^nPE*$o6nY+Y4`ZrB5)Z)C*ZfMp7nd!jL;<tMuUu7<~5Pnk9*)T2W3|Sz|Y-"
            "3LR7R_*o{AFtoRXkWwkHOU^iOXX=bX5nqe|6As$}~l8G0t$r`Li<@CYuYNW(N)Ygpi-REOvz%0LGO=a$~y$GBz!!D(I$Y$=9JgPo"
            "MRdsy(ExACk(7!PlMO-"
            "B>fa5Z^!@Hpr68mPZt`1w@g_H~+MEr*w1e;IdQ=XSL?{O>=P$#7?Y$xB4vbM!HY#*u98JTi*8YIz>?IWh}aJBpeNY6%B|;(SRbnl"
            "%i@dMSdcu+N#&NeuhP+34!{1)Zd0&O|cmVFuN|uHrRAYk>Z@L&z=QBPvf2aox2aa`-"
            "g+Vq|=%y5+D3}2zs0gY5WcPz=Jqi6`%yFdP7g@$30enL}|FBG{7aele0sMS4<Ra;%8)}B{R`aicSas(vL{YOCIN)0f0ryFG4@E<a"
            "u55c02SyRHH8LE~mlcby1y6252ev^=FVXl+QA&*-3CN3j?sf_s@U|@q-"
            "96uQg=abzq2Y;1XSZn?7@==eg*8r<w1*_p&pjx{SjytW_(o&p2wAq+Zx(h$s<RAz;DiC5|HafTs=31@wUudVyfql5uG{?~4bP{}0"
            "~nq0KA5#btl2XQZ4mY1hdiBecdvvMH)Kk5M3cJgCfg-RG8<;e8j!e-"
            "I0P;BWzyks2|c7JvxzJew|=JWluZeWb#F9nBWHoKZ<d#7dTTzy2l;XzLR_RQ{W81=++l3K_7gI=JCWgtu2p9L;A$jC4qaj_F;7fX"
            "}<)C8Ll5#ce|2lUB-00Vj+aZ6gQzd#Ed&9%%m&%=;l);A_MwYbz^*sd!tbxY&*oHOS29xEi-"
            "b;_!WGa3y4I%|d}<No4U!#vJTNi`iQfFa9mM$&c1~)S&;)m$)^%NYjB`Kn=U4th($ADwdzsvc_6++@Mbh{|2hLyyaqZQ)|hJH(eR"
            "99R@v82ML3eL9teejK9RP`s1ni&olw4%O~aIvaZ`<H~(lU7KwKuM)OKWw*PIvG7mpLMi`{S&8ac1YfFB*y_qOPPH*fp(PLg82I;o"
            "1@cqto`MT8_;y@^Am{LjuTeKPz_epSjc9D0f9hdJb7r7xsqT=RS<*nekPyQR8lRvnp4(~OrB2!HG-"
            "z2k0%H)*_0l+@ZvnMr;weD|+6>v3pG2@@|Ya2sT{BE-"
            "`{_2*mtOITTIit~`e*udoEmnAJA{JfYIL8t?S{xW^k5ua^rsqb0v~0ocx)~OPx>(X?`r(FX@R4uSY0w@H#Jddk!J>ziXSApX(nNu"
            "g-l@-j3lNl|E3)SZYYe6a(4A~0MvRD@IXmDg^?B(pymnmIsW;(l7KGK-hy`-{>4jvi|L&^pn4|t^v9GwxEm_Y-"
            "@O%!;c#BOWOKor`?PMI1Z!#xE;mDdS>dx7a4MR2)EiaEP5;DH7pDk%BwOFirw<j9v)MqfA9&QI}Z%~ecd%1=kuWWP!ojg*g*6dSy"
            "r4yYRosco=GT9#R;<Kd=E9X)K5@<oxc7%1YjFH%$yl9zvdo0QOu>*z>o4`bvJY{iOENHv=#7t^|4EqFH>P`78F*B`7Nt#S!iYD9y"
            "59_XJMp`k2$`Kx*(@QQp9^uizjc@5bt9n(-$Ej<n`<!1xGMTQWeddXpW@@%25f;y9)}i5t4o>VLs@$&Z7E-"
            "df0E<m27M@lb@&0@kpgGk%Cp`txz>LwQb)nQhG`?QsZBg)5<Ns^we`?X==)%u>ZZyZ=^TlcpQ{d0GgHSVt8S%91xq4c>{u+}@cLl"
            "lsM<ZS7Dx}cvl}ZX&VI<*A;#_eA)7HxmGYQ=^)wcYvckbcXKJNM5Y6&PVAoAEX<EQ@)z6cJH)0_*R8e21SXj9jSL&OD(1;v&L(<c"
            "`ct0YQ;Fj7F;o3W=i%iG0wl;K7|9G?yQeL8B$tMcJPj31`EqqnsPOtDmGbf1-"
            "{e6iS7@x&wp60ur5uVy9j3k<D6sWQ|1)H@UJ5x<!PA9=Myor}M?^TGhf%8cw#L2EmYWx~PI0NH+yR8<B-"
            "X*h~O!m<ZU^B!Oy48%Q1+K1&xiFdyQ;wSLM@AO?sk<1+76fJr$=4hqPOU?h1J*vI!L+|>Rbxy{TaTgZAw9Ag5IsC<<aiJ++MGY*("
            ";4~s9HAa-CNn&8lml)OQZ5khzsvqP}`FFfQO}qhXaJFJFG&sIl7ww)Uv+S8N=DPh0utv?r$Gmb1eYPgLXlTMy2}B5)I~8}Djjx7$"
            "*xj0tAff}irA_PS<=w6WNiifUb<sdy&9&X=92DTzOaVf<VDiRj5b+a>P$g?C>1ai}J`FPW@g8y<eeQ&JQx<Vy#VxX<T=|T6t$9n|"
            "D#9|pynnw<E{gn&9q{yZ->28z3PyvV-"
            "@ITp*=o*(zU|@y;~UV_r@=iu9*qZQC9fG2Be~oC@0z;+l#e>$)t>1zpq>P<(9fSABwZ>iEV_v;MVp4^Fco7y$Gf&0t)|Tr50;^3s"
            "74jgh~);MYur@k4}{fCU=%<f(^`1{Q&$j`)v{7GO|O#gV*Q<n@KZ2m1$V$zaSsNr5FNeNZ=)ezA?$=7cgc787<@&b@LKK1B_^BDd"
            "e5E8<+K9Z=JF}pa<Mpn+&s}ET2HxgfYNOw7{CldAm$U}qT4_t#{D)V69RR^v|+U8Zq!`bCG9$RvD=dupAE!=p%*ST>JP7`Qn`F3p"
            "3)+Q`|5YZqQ!JenzdMCJT}cm4XnrKv*guh$EN@h##r+>LL9v2zG~oH^Ikwb{S0x^0A0S9>h5sn-GmVSgSMCzj?`isGn3X>Gj^1vS"
            "+KBE>^Fn=hYiaB0Y!Bv?)=R=RI}CjhpWh2QUJ$V_uy;n<(b$!na{l!N)s7ncX}*o1UdYIFzDKuR5@b29Nc`d6Zx##nQM(+oVc~fY"
            ">--Zt47Py3b`&NP|EmV-lBO+B?>C|hgnfN2z+P3cl&AZ$l=%w#IcQucT~!9&NLzId=Z1`8juyoz#)%KE+;0-"
            "c94D(>BvSS8L^ZhWzuoq!o%nT#-?0uxND#F?Js$Q@bh%V=qpju_aSV96U^pG31>XB-"
            "^<n8U|%WoDV{+%fbS9NK;33A@m6~rzzrjGrio&gGA2o?dWvF8Lm80Fptn+!S^-"
            "$35Wh)B6`2yt%6wJ)o8}zK(wmkAZ+7X)IIwXHB&!sAVbN@Iwpd-"
            ">y)}iIh5&ybtcb%Qa(Z&z1&WNbotW`k22km6@kfFc+u>jWkt&mY?eq|WVn4~436^A~y1B8`^1y$R2BT*Je!_Q??rB;Jjaxd%Q>yG"
            "mN-U7)W=3T8D5{NSAww|Q5815!-O&~Kt(7)K0+RbD60_aD_rNtx)bDFj<b{L-EAV{w%aL0C`2V8hCS@wNmvmz+-"
            "x^qu9RnT8&*a>o*9+Ul{Uh|GI#OU9i5236=BAgn;(vllcVNnC2@=nnD67LV=BA&=d*|^hx)2efRB8w<M41Er)}Z5Cezt4s+vRw|b"
            "*e^DplSNjLAvrd0htmsptH@YfbbLaqOmml(M|7p6ik%(KVk)~;-"
            "8$TI9G$RCLNB7tXL(;WG7M3VhLIq@|3=lPO7+kS=&$KF%w4yBWFi3hLtItgqY@*6g|>3eSbCHeWspLp8ILtFgZy_ziwx&$XxDp50"
            "W;7ZQ8^+fd7c!>qKfN)+~K0t3Ox+ySWP2vjm-K=@{IOM3>u0xTW7mH7z?bXAU0i%J6A>01txwgU{;E9bR@`xILxe8b1T-"
            "+r6(R{a#-3QTU-15&xSU2&x`!677g<GD?8X9S6hMd%b`>*r}ZK-"
            "*^F0`IhxpVwh&Cmu|f)xt%|sS)~qVc57_R{gO)N3#_)nQsw>=NiT>q$$ZkgDl#6A>i&DKZ`|so-QxyRP6O-"
            "Xmv6(VOgRBMby<tuDIoojx{<akn)8;kxTWA7w{TYa6_bbJ!|>7$0%~(fTWlav7l-q<-;gTncuH!@4;KW5`1xu4=_un1Q<)%-"
            "KP_(;)?N-"
            "&2u8|1VPweI;quR6rooy)7^Ym5py4>UJrv!sW<c#_Y+dDj(ev4fTP*sz^KI+dpa;}o;FWF7Q%w91Q`x>P+GM()ivpdXoz-"
            ";j_AngfvOQ3u04=K(8T@TpJVvabnHJCDc<Vj^q^~)Ed91b&(qzMJceg3x!^EpEkz^{$>sOLYku(gF>&q2%*WfsNS(l`pvmv3-%-"
            ">b)*ls7&c*e6mLC;hqEbZiwnzneQ5^{oT2vDS63*g-lM~ANLvSa>=_4XVn1#pj16T3T?7Mt5DiG#bQ7o#D#4v9WSDRJR%Z`-"
            "0x3lLmBl}SmZ9v=X|!3h~E&_!YWxxxt@(PggEG9CC>NBA3#12R2sXef}<5|n<3gpQq^Ow;K3$fNp!o^CkyZSO~B(8(ZDW>I8k5fy"
            "3tP)C;lpPl<tKo?{MLT`hKqeBO>nunScDxraye?G`!y1h8pd$N!JpL@3}cyJqwpvvF=TG$(Vnm8yg<5G#{kayT}O7l_+o%(D<hMa"
            "=Ea@d$D!QsX{S)W|0pVdHV7I0+AyX4=4%xO;m^ykdG9i3;dRGxas)QWmfL0Kd*!#?8+RkI1IbfH-"
            "Dyl9_Gl3`ndSo#wE)0)wcl!~F&Jjok9wdiRRm3L5bZaDVbi>W@2N&0iJogT?(CTiCpxQcD_la8?yb^my`?SOpCz-sXrhLoWezsbV"
            "+6`OS#k*n7k`Arn8ecJUErsv|UO7Hi}=~&cE@W+!G^NvD&4l>T9=ejirp!tLN5%ohhn^(70y$hFi8R|-"
            "Df<Sz%Fj0@J8DoT+UQf(OyNUj`P`y5!Pp^l^&7&XICTF?eeU5%Il5bAg?djds@}7KBUU-"
            ">eK`Gp#hZn${<RCWC7j2&;u?ExD<1UFZMuY)#r|JdId*P~y8SlB`$iEVtn@e5?;+HSqGNY83t9%KjpmErkO1Bc28J)lURZ0uU1ZH"
            "8f*a(R2w&WkdBFm0qrr6kkCetP<hDox!m$)=QzMGEFs`mEx<Jl)pu>~9=<+j)I9{Y)V(A<E4eSZ_x`dRk=Cm&)l({f}_D)wKPZ;1"
            "6hLA{LUNo;jJ*7Qnw2Zl~@-"
            "+s*VnSOVW+XV8Ju7h4O;$7WLAg)^^@p1TU1WFzj*CWf1fl6HgJfq00qkCHpmrKyW1&rh%IsdjBh&bV=BLV&Rr>7L<!WwVXE4g@lM"
            "fRls_f|Po+=iOR)tcED;l6>~Pd70!*MAM&p^<#oaGxWA!#aY%%Za1GJx+%>or(G;y4o8}3{aohyv%oD9kREz+e(vo{!p7Y65=(lu"
            "YoPYHK5@5sIy!N19H6?rM%IrDz2psI<07t`(TR%+D*LfsQVj4;is~wXK*y*as}!>rB2;}GQD>-"
            "*7(uLV3zF~NoVZQ8q`)jp2_nyz+y;lN5Y~*Mb6TpFaQ7tNdqF8%^+85Nh{0aK1oY`^?KaDW*j~pcS&c16(n}5^xWGsaY+7tz`^E#"
            "IpQKTlVf>`;OG{~>g&BPGd0we-YXYQR0!BP*HTJDz$MY+Ot=oeIi8@`v84B-"
            ")s3bvRC>(omz>Nlx_CI}>0tZ>f%%&;E24r1(**ZQ!fsC3X;_b$qMz}9{I&LJ?sNP*DL|XR;DQRM0n++{A9I)`ACc?@hXpq_B0?&&"
            "sCjW9=jX+QdJKEYBw>PnoEU0YDh%71nG7J`^&E97*^EuSa&RiUs|Yni_N<KSNNUq3Ub+n^x9achry1xA+KG+<L!>1;zSHySZs&E2"
            "#c}0S{XcaGe_2ZmfgeH?Ri$&wlGU5C!r!Ac#}DIls+!{;mm3YyYu)I$I}ELr1>MgtSUi9t;x?C_repoU8td}?g7i%?R>$gAn7tXV"
            "gIPgD<<vQ9<{Q3f9_vu3cR}e;*L;%8mP~C!#bN<!nilt9A<<`sh=SC1_iU#TKii#vtkY8ZJ(x^40|vg=lj^||gqTA{6tT}pWW`4;"
            "^LjIpiSTFfaF*PYDNA8}DPn)MA!r4zzjEU2T&%W(a*BVh7Gq6KP!=+8CwL#iy|-"
            "A>yt~#zhK=x~S&>{fAY52XM5h_St9ZNV6wsYbRXWN6Myll}1}NalGt&Dx>%BVVnAB7|ZF3z`nBJ{%{N_IIq9(LBW2r-"
            "<x3uXqVUyX7Bs5s79iFXogh6@i>TUIDb%vHhmQ{u06V^mcidjSB1tpx6BS?eP8#uvHGa)B|4uz})5@)eN9x*cA-tC(|y*V@^SVnd"
            "xky0$(VhIA5HyZ`k{H=Xz%?(XU%fd*9bLGhiu&qoT#4{2;>4EM++3*KCMLpD+!B2Us)?o5iCE;?ru}p6ysTkiWuA1Hd6zm3o12$B"
            "~FicI6HE6fU@}?LIuhBAA>U#H{ul+A749(f6ChMB^LfqqiX{b5ei(eakOLSS;Vd6KpF+;tcy;Qw{jlHILq$0ssa#H<Iqvd76IS~W"
            "JX1;#~HU1tr*kxTG=M0tn6uyIj7@^aMGf_QNJ(7xy(*Twgi|tZeF@ZQ8+sU~12{tz`KJHYoa7P)f4)Z}8Jyf`Z=vsQ(y`_5t0`5f"
            "E*Eg?#DSBnvCJ#|0JQ7Q9sU2G4uUVi=rYSpG_#I$*D#D-Ee8|XFK|Q{U7-"
            "hlVbdcA4gty|eJ=3C%%(ERb8Qo|W<clkaJZ2fI_SVQc4obza#k#BX<q>ic-<r2E$;q+L?=i9_`gaVdX?Y}SWD_Don{d04yd-aITv"
            "~FgbiilZU~KJlDl^^2*uqD%DShlTxt8gVO7I^!b}WD)Jh^E>soUmDggdoI+315&&vY&>yj;W+gRz|kOq)0E&@E?a@TA%Av`DqI41"
            "ilo%HY#@A_w9rv04EoU&M@s5GTK{+(@27D{r-ht#=8ErEfZ;cy#-"
            "#Pt_kW{%{dmcAbT~)4p#mcP&d=MKd6bqcP5meP*7d>D!u9#p4whh15N6K8;0Xe@fafXuZE2wtJ$Om<vHa7{wS)$-"
            "_crg#VLuJ>nNr4L0|Ad!y@hU{P<tdgbe-ttPZbj0zhI*3BlP(UX&=AvtEHOwuiotV~~i<f7Vv=#{i~ijCOpk(&?fed`YW+XUb=|D"
            "sBTv$5^+K<ncEI5?n^5fYInY^)e$!j(C^>u+8kNp|SPc6-"
            "*=?hy1|ka|V0EPgvokQHZ)R{2`Ae{_j{(6~pBp1l!#H0GXxax}KNDo3LldTICwpPRN@WX<smN(Xl4xXFrEd_x;&4|SQ#H`KX|&03"
            ";lLOH&7$~Gv8oXe2nuE)R+O&!7&GP=3sW8QaIaoo2$dZ`2J(dln08EMa7<RIR~voL!c@krYDBD_mYTXobm#UccPrI*_hAIUkEs()"
            "DmejJWNp!ZfdnuDxyF=V$f$vbycC!wyR8J#o~%=MUqSkJ9B^)fz3&RZ~@&$5u5K@k*ijSfdku~{jx7M+HSiz+(Fw~Js6H1ziP%-"
            "=*k#?SCyMhkba|6X#+lgKG0Eudvp3<O7-5;OpFpOTz+`nf(Flz&30bRYnLN`1TumjbkNP*5ADt4b>G2TWmDH@1E!&w}6eJRIAIp-"
            "Q~zcL^!>JsB4i51L&2a@{5<-"
            "u*qm(vCm|UB1i|*&j^7g30mdEQc&Qa^qKD#pj37AQvx1I$n&NR7vmTN~MfMr)~<2RSs?6f0Ax|t`R+<Llq%XsQgVxW5X)Q^7{f=#"
            "d5h|^UirJ%@^KyS!K9&zk!{F+%1bi_eU6`P4|{&;A-"
            "XO9&ag}_IPj5l7^+vO|`J6r$_Vywc!4N6_{Z{xH5}(P=|h#DXlqf4<J4U75uZrrfRWbeRwF=@+<j+rNeU=f$H?29daiOapT~SId8"
            "?)MEj~&n+qC}!Y-"
            "z$5zjWxZfWi&VwXTslBFMXQehoi;!fI}5nz@PIA?109i73V0FmBPsCu978#4lV{UxGh=Q7lCbiF3v*LI1?R_<io%y^vD`exBlHbo"
            "kJkHo&m|6C2wizMg$o6b`jp8?0{s~=-)c$%7zECelm^f8mE-"
            "?kyPj0K)%RC#8iTW7h_LHm`2G(h7}Vv?w&?v@AEBUl#m@rpE~NBiz+;9qatG;2nR#}+e!LiHs;Qj|S!{xcS0%gQ_c?3swZ=u-"
            "9E*WBnWdRMt{|3Ipjp1o&H6QR&S2fyKAG^Q1?S9kV5r#z4Bz(F$E0pXUEY3#6XBAXZqdqL|V$_4z`a}@c`CBXd$ND!GQE@ro-"
            "o4ynwUzCjqhi>Jy%^qlR{AQl2U+2dvW6^PH78QVfj&~ZuDSBXc5rIl!uv;z?0J3RkqO}UtlU_@x+`eDVH&2y9I<cQFfAW+!WE=va"
            "d(n;a<85iTV&4UtCHca(F2kXFS<aF@_iSPnosPIvU)9}2^_zwk+f63kAGaSJpC3BZB2b}7f1c<T_poY|E2i??1Gp7rzp>xVZ*Q4>"
            "x`Dw5zAUcNU}|3r%4MEDZ~+_R-rywPVwGnbRkiCCRm6~U0JXeLb*A%NW_j(8>`IQQUQ<wc5+`T5&@bi3_4WrPnO9@j)iVp$-"
            "2N#EpNKWTe^Z-"
            "B74D0PLzepyXsenM0N)uEb@1J9I%XTr4v8%e#N*_<KlhqZ02m|aLR~gW^5q!7HE+4*Jr}ig+j`0;(EomaU43EKA|NFr6ND|itgrJ"
            "M+fAm&f`R1BD|J25yIeBM&9gKhH9B;@zqH{Jwl>@8OQ$^CsvC+Mc)b0}Z0_J@i8`S&WhsbB8A9eWS$adeT-PbHNz}|1ek>~zdPq_"
            "pPZr5KrSo`wEb9N{fmHcJN-^#fmO#*UQ_y<Fq`u7o%tdb#U=-B1LTe&Y@yWyp&OTu=xu?W7UjWKQC?Spx#v_>!qES9m^~!3O%-"
            "!QEo9fG8@GEaE!#Q}9;%C6Tb2=@4k#Yiiq^05}{1u7lO>c*KO7^@jtQ(>L@!{fyN2)OPS5Ecr5Ko`tpQ<4@?^RnSl{vv2$EEynsL"
            "6mV)szS@2s^Mnh&yNVh-R^xtzL84Wrp;YY~+FKk=9C*Q!BQ{k=R_uP&UM+?w?#2xB8by4%)g2F{BJ^PRCP_tw^{D-"
            "(pE5Xb^|vmDNOnZEaWY6(J|#1}wTf7Q>7+Li<O{`q@GpJox1ZNSh`?@B+^@xQ*@)3wk7#K(;Q|Oqwi&h^E63$s%=qa&p}y@}8bcO"
            "ttU|e33eM*yr6B?&VEh_%|*2-G9B-"
            "zPHnf%F|kPRfcW!ylDvNfo~9ntbn3%x6IzMIr^&O6?tFkN5z^s)_Bik)0ubY`J_e2?y>hVApg1kBlI@&;OJ}|WEs6Azoc}6R6&om"
            "WxTLceRhkQDoB&4i_yeTk27}|9K%X(r{9d?S3@-w3flbRMq8{Qe$rG~sz22O66q}-z-"
            "5d0;W{|uMg6%8Xjyx5M8n?LNT31~GpO#kW@e)!FS?f^6?!k^dyXGuM?Y*XV(SgBYzzD-"
            "2_5}YslMfPlh3N8nt#XjLRNY`SSxO9**>|^zU2Pbny_jziy2jyi>s?`eo{AmB#H>F$lo{qQY31=CD*^7vRBR{7X2bk{35bwB+=xY"
            "yWwymHNrJd49q*%T(-|pC0xg=7k^q-"
            "j}r5TUR5yJgaXyjr%y%>Zz2%>2+{JLo;nlj43J#_V19liav4ZPFdnIsfeIWiq^!5*spI`lu`F6$68vD!`?lD!H*zC!{OcTEET<}w"
            "hqMi-3K(&o63(v67xzN+7KkNuDDLOVIW2?X-HMBkgPgpjYbeyfwZ3w~oTrY~d+Q<7bEEzo)S&Ggm!n-"
            "r=eA+iTWqQ>&H{LwNe@(p1KFU_Q(b<{S#VHMUd=R8!sQ{t%q{hAz0}04^bS*_vR*j>=GBu*nU677x_w4!x#I2Dc}g}-"
            "hWio3I`>&C-Z(Z~+G#RxH@vRJ{~__6s97n1;+Q-Fh714lyvuDMEtz6l7{a-"
            "^8vlDsm6v%OQ=JSntL4p`$GH8<7Q*uO+ox2^KZ#Xh2|b`!7~fd*b1)cR1>Pr}lj0v>c$`v@>JA{Kvz~0G(w|0x`>XMes!`H7&d=G"
            "Jt>EWs^8M;TN+euto!j`B5wx*;65}W?mk6n;ki$kJb2X9tuO8E4z%?}wQSNztL7S%0dO78$aZ;M;n7*qMFF*6&gH~~kf3c@Q9xAJ"
            "n4S!$hM{SGllGI#nBVL*|Fn~ipTupr>pG1XqShfm3mo9Q6><!Z`z$C?cRXB6?$VURr8KsrP0Fo*EeV<>YbSbyKr+`$M8|}jr*8J-"
            ")PA2?cxKwQj=E82R4ZJtSY-"
            "*>!RZ8EF*i3OYq<8Qx$6wm%?<ax5ihnt*WKHAXr22ByYn3$!OsodDTJlN^m1ame^cOXrok~9OM*X2mh=VYMQ#HLY`fZ<dFX#D9S!"
            "<>ExYN$Dgtd`DkGzW7#4S*&v1UVadDt=4wmKzz%KOXac(JpWk&%wE=}(G;HSc8!W=-)_zRutH2=M_38-(L1I%wOY?l?5!C-i-"
            "4v%J_*7+D3Qd4d@^_VN-}4VDmCt7k>P#7UVhC|bMUKC1!VQo`ul=dGO~ouFUAid)Q#tDo3Oa_sw)Xl+HlEaQWNk&-"
            "m4sXP(v1ldkbh<OAc2=c;`MdPC6H&YI%i2FAxFIu-dF7B16?JK{EOhLC+7*F=r<DarQ*24w7ok&RKDA7733!q}TThUW{`N{ziEf^"
            ">;Nt>ve1&(_C0yGo%$O#flECw=)8jh3Co@a2N`#uXS&-"
            "rp+J7NPq_N_ue4GHex*Q#cpsnUJ}>lZd$rz@pbpFVI|$PFV1majzG(3QE6b70|a%;V|T{gHF&FKhsIWmX^z!78cteICahq>CnpO%"
            "0lmvx_1+K{yI^djs~xb_hgjW}xveb8U=5qwI+r2#IwVyP7U^8YLlS3;#<j8Gz%z@ukosv5i)s%LHkYQ<AmX)^8TN<g@3jvI(ne84"
            "7iu9*z>32P0z<-V`>Hs>7&|Cj|%l6)3u09I84&hU6j9W641f(DxP>n73OnRZCXJStYUZyLlLkd?Hymi$tON8X~C&`!j3I`$0b-"
            "p<rAOoORsJ>jL%t7^PiIDfY7@@}r~r(=kE&qXFnIi(r0Vas>M3TKA|d1)J}!oJ|gTJ#W0W7wD5(#hL+Dn)#1iBVUg<*6gDxLt7H1"
            "vL&**iquR{UkB0cYz`=Cmy{VJ7x+rQ2uFSiVAXkH9*O<&V<^8x3%Wz`XY<2Rl+YNq0E;rvK9*#*n!y)A+NSk4vLB`s3!))Wg*T_g"
            "MIs83ho{S(^wZ-"
            "Q4;g6uyJiAh{3Xl<bS8*lzrMjg`+PIX8j1Wi+$;WY_Q@L%UVlI_FJD!e)1dMQtdKO~kwHem{BiVA<Ujg~kp=^)-IAD4fNOPO%3-"
            "?g;9p9jM5k-XeJAYHQ#0cRt&SH~I{jW<O2)+^$PK}ijYsqHS2?%2m~4Rr>`O34J-G;C^N|R&^ejh91cMa-"
            "5Knlej><f$ZLi;hwPI?)t{3|^px~PQgLIE&lx!38-Z$E@C=jWSxV$r&-O%T{DP;k3>F2$ZtX8K65R^prwYJjML>n>Q%ev_DE$#a-"
            "o4&W65&HM(lUG;IXd9Yv>3;9d#5n3kX)W?OVhuXsv#dTM4VYrb7u~^E6DA&r$_qwWN3Wttc>>ze=$1hdi>|@G&z`IRrm-ES<zgPW"
            "kAFqTYHp$DdEUWjpdWtUn{4ipiJF=yL>&xXr#!;l7|1XPOrl~e0O5#oHx`r$n3mfyGqBz+Nx~e9?P}IvR0CH3qAAe#gt^O2MTqhz"
            "4|Cm~8N%-"
            "t=lH87Ybxv{@q93fG#iZx{7m0^&6R~Hk3Xq_;7$K1NuCUOt|2k3Md+vfOtV(0tX0afNCRUVhq)aC?I?NfD>0ih(<00I?9DA?lxIu"
            "yuc9LgIk$;)oCl^C>6F6^TyTb>mMPk{O*V0$zp<~1!&0(HEa-GZP*K}L5>*|*qur~n(;84}pLe7t-Cc}Q&**j-"
            "kxqJ*OUY`wzFtTJ&2?IN*DI_JV#FurdSz<K8y*QD35B@MQ8f?r=f@>!g=Sl53D=yn$g8Uz6+i9F<o?<Rs<k}_%)Lt2C!RW)in8d-"
            "B!V{F73e(XX`&`*GuU-Wg)*;*kkU+xQKAeedg)_S38H{0kUsIP{0_}ewSCLs+DhezDtHwFk5ylDmkOis-"
            "#ht$3hT8rwn^_JcCPs|I3vk{x_KdgqR5Q|h$*Hn#j@mGRV2lh+U2TGDeVD6MR-W>;0O~%v>90%Z)0KFi1XWk-0Hr)j;-"
            "yP^^|&I+#}B;Wrm3ap3Nbfys1;L*obt3ln+cS8u!y@q(mxW^<^d)=HFBqbTM5|Ia-"
            "=v{)4{|;l?MHqcxXZdVt>BUPm#li?kuL^8c(Eu_6d3jH$fQjqVaKbJa7kZNH6Xi0*R+N{iwhM%vQa!*jJ?3zQ~8<1e4%+2y{AS(v"
            "NdVLSBz|4p1&e#W_)Samv7qLOet$mXwPDMD|JRU2Iu7Xlt)`}&Cp#XsE=sa*qz__LR7=5(m<ufCPM%7N&sl4>)^AXmLfddjH!H{7"
            "1$cMBLxVQr`4*)4J`Ak5Cm#!+Gb%A=L0dN)3brSG~9b>tlJ>!y~1-id(M#b-"
            "}I3M2}dAF}FyidLvCbc{3Vwc~`y1!mi8O6!sFlzVE$No9B(YSg}^1?d9<!0#Do4!VDX_`&1+NLWHE*Ic$!>dctpFEhVspxtGsP@a"
            "r(5(nZZkI;hf=XwXX@KO(g^Ay(1z5%Fu2xC;yAqN53B;4U1eKlvB{Al8s+u~G1W>h*PRZnO_^ZrTqK{LnRO&w;jhA=40jIZy9ngp"
            "poYZbZZeXzo!_ik55xWgI@o&`nyv)rInwjxw&>j3TaljQ2Eft-"
            "~_^L>FkX9b(wX@>6hpO;f5HEfZ=Mmb((4ToFtG{5|h?71Fz(AlK=nWQcH9r44?bh)H+6mLb9sYDY3r>BT3IUQXX^U4GpD4Mg|=B)"
            "mwMTa!!^JI@o`g2<rB=`=qNx<Jv#_0o-"
            "eeb?HJc;qtS{}QqkvgbDZTl7l6j5z_m~0$uvoPtZaDD?F1&?@KXTrqjGyu_$RHG(6`o??wv2}7ez?I;n(hlB;G>8zthAj#5-"
            "XRb&2JLw@fRq?wnv%^UHL&AtEgJtr481-d;#=a(hQ(RRuYoH_1|v=g^byj=6j)|@-"
            "F9q}=q3DM<ml~zIR|K}CAXvBRStCrGQ1Wsb~^RY9gZieOHebG%#`g;_>R##@x3&iiZx5e)X^WgEl5Fx4ncox_R5jOG-"
            "r*f3RW*YBD{E%8{w8dd87&eCA{c<pm)k4y)&hVgr|}pL`JajKBjC<V_Xq=eFR(KwP^Z$<#B8@6H3_TVx*z-"
            "2>mffbv!LE$~&ngkMs#|l;>;kH#Px0hTCm*(qhpNr?$it(A)+IWtgK6Xv&mKG`dklQ+zc*0awcI16PZGyohBW;5-"
            "63Igf8r9vX=UR~UX4yI%8%%|;bI%617sGCohQup?eCd9Mb`(f=Y~(aaO%Vk+zmWL{p8NZ^3IHV(7a?kj8C`1G{Rb)KWPO)5SP6fH"
            "7EOAR=grX?9aqi0q3sEMWi^BkA}X{yBYYrQ*`{7>tD?7|-ete%R=#jyGuWXPN3!Dw#{k<A%|5Ng3p!2WTgzF-"
            "a?gPwz_JUqZmv92^+KCAAh?IoW4Srh~oRr``)KY8DF_n<sS68)XM$7;~+5l6MkoUhgoDU5Yr&C=_pCD1>unD&T-"
            "^EEvs)&>?91i3_#85^SA)h7G9OHYV&v*-a@YuPP6IL2FLGnuMZ82nbpkpc+*NMJ#b94J5CS4m!A?w1V$Oi1g;J82BMe-"
            "9p=^>{|&Z7JE*fWlB*nrYhn8@WD`hD(2HABm$PxK8%IQt_pXBw+NyZ#EIjEBL(Ig9VUeco!SmdRM_l;vT}f<`84A6~&kg*=TqdI)"
            "$%PP{m`M-Ti!)xS<o#gT@{gRCLP&afA|PA+fo*1XA@_=o}c%V4ywF8j73wx1{k}v65T--"
            "6I*HHxr?E^{3@iGjnuIcEQ-{1@4Lzatf=)Wy2tO`o{h8IK+!_bW={q_TJ~fyYB)j5DMO3b6MU|xEnB%e<dLn2YEr3^$5Hs5DW-Ve"
            "HTz^;K|J65N(N`0sO=vD=#_zv^f5DaR;;d1CtOiFH+$+Eka?x_t^%!ci8QXcvAm4s6Mg>zC_GLl#Bja@<dZN5<nc5SAXt{^MZqzj"
            "HTOD2$?x%COzl`m|@;=r?F*c)`(T}IS~3_>2RGABzXmiAJl=&bVukDZd*<1euKeSbU`skf)43mHPRG-oU>JlSAt8-"
            "6pwVdVZBiOK7!aSv}T`Yp}RY&=Q5sx#j)`=z#PZtc=fw|^UCf;+KweZR^BP8WB16{BbnrJH0YrGJVxX)&od?04-"
            "h4gj8eI@k$FrDE0Hfqs`QNoLl*tB?DuY6Q?kD;!G{p?p#Rt-mB?*=Pv`Khxd_atmrs(+<bLFd0Xs;q4f{|*vR`YT#m|sVe#}{KOv"
            "cQfAt<N(kF232>i1b<!%Jc#2W9&olp=Pv31gMmmc`w;`r!=y9%K_zIC$QMk$J!uFS5i2i5h~K=R`xN8F}<64!YnO9|mjq9eF;QcX"
            "7Y3UB-"
            "3|HC~)`HZr(9Q=W3AsPUmv!H6Em4qmJ~30eg{Vg;$A1NDETc;>5ZzHInhEOCn{N_FiwBu34_QKmdCFLS4^b3rl?&0hA#{L81gHLO"
            ">9hhM|481YYh4kc79d%u6|fsWpIR8(ns?vpX@6ix8S_(`p@&5O2u-2x|8=;U^WBDH9B@pG-"
            ">U+EP}yRD0RlVEHx@0%ti0!I@*dr1$+Yo7SDejMidPjSaney2ZxmW(SDoR{$=g_M2;_8GBcw2bUS!;l4})j*dX2^uwU@qbvtOC8N"
            "TXiL<BIvV>vcc`lFv)0__eHQeh*tG$$LEpTKjSQB7pB*6Nkoe%c)R=sl;}$;?1;r_>?UK#pFyFhz5xd+;ilYo1hVh_essWpdsGT1"
            "JY(EazMCqnv`{hV6+JVex<}g>*ZONmhAy$-"
            "P6#RDmCNTO){|eDI#*Uoh*@ph}nUZ2Y7>{_tb>)w}T`J7j$oK!eR@LQMiJa9S_3Q@h4KAo@<p7er&l+5dR1wL+gXVml5;OYj`kZv"
            "~j2b5$IYy&rrRfk<QVJmW?A}6vZ!{jI9%~?p9p@NT@Bn95BVFR})fV4TtOs=wrJDL)1~#}S0>L@qPz1KR>`c@51(<0~t`t4=MO=-"
            "PI!>I5Yf%V`S^_j0rmV;)lBV@7@s2z_l`fee_iws${nN`mI{){hFptoJ@?(@omIhEPxx=h6I>C~T&{Q7^m@1TN<Q2l!-"
            "&4mo`+xRV7L5M3-+21eJ-"
            "0bo9N%5P>K<ElkimGHu>Iq|jn>Eb+G(~t#x~0*f>h%D=@23}4?EyW9WluxFEM(aX*!_GQ;Hej3rqV}`w)m?>}gsUKjW0fuO#)=2y"
            "Lyoo&&3c45QvdLLrO6(eZjL#8bVw0<ju)plH!F={)7uqDTCsRS$DRIuah8-B{G&iOYOnwz826Nu1LnI@3S36H&e|#NJx-"
            "c6LbMf2Gtd1&bUvrIhGvuB}pK7~uxP4ZC&3rnO=<>3_rVl#Z-"
            "KLfAZYK<c}ut+LdZU4j={^xhx>`yNjo3TAC>^uZ0hcPNRdtPQOQKYKiyks+6K`xM=_Ed6!Xj25i<YrJ|3W8b{wE4For`lYMC;M_&"
            "$`PcpwW#`A>0Yy$e6BUMt5ARV%OQLU+%^>_)^-ff`LBihFokJiW0Y+=gDN0=_*YMmD_1_$k+AV^AgsiI3N?djRl=a-wA;?`-"
            "YW3Yuqw6MrvOv!`su50HTz8CfRCnq$ki(*yf<DBnq4D6yCsKF)DqKGsr2QFGA*g4$djtX^($$$$9W^cErG!N)TO;6%@qgbSp!z;M"
            "<*%(vvzn)wbbm`Hm||^rj&b1pdEns(@g(%>k-G}+tuvXvjDO-MYa$lO5hhX^wIB#?T;F)d#K?`13sSwK>Da1WR0mbYgp3bDd~X$^"
            "nKE9RO>unHN8MiT0~$A!ihSbw>V%-6LXJMPWKiFzEIr<2HpbHr+XcPZ>4@`-"
            "Y{s(29Dk&uhV}v2dp*>6B#TnRig!~vvFjpE=*YP?-"
            "G3mY&o==Tp7SqfRYRWt^srC{V$Abzskx>Z#+v(pX^+f)c!I4*54*?Ik0vc(*W&kgN!Nm)Y9m=o?=e)DzfrALXVbZO4pX@kuyLcNh"
            "WJQIaUrnRmf@jnH0XM^v`XMV)A_MSou|*KJPQm#lOSZgfMj^~g8{errcg~2Quj$j<#TXp<!cp{v>DNk5{JZU-"
            "$|@U7$yH%8Nn2bXZiL*n2Vrh^8p(teKg%zGPxo{L6P*;Crog-"
            "Y9<V@$X~0MEDhDiu5McL)&dwbNh3lEN5GQ%Uu%>cG83epW!dgoeKJ1V5IB%{7PVcvC>YuO(;OXD=24gJ?_768xL;#-"
            "Cgq@+RG4TOOq@&y$oClj3St`zc)q2MonIuGn61G=TJU){Tjul9W{N;e_ldzC{5!jmW5{l%)+i@|>XB`!^n;#sk1v3qfHat%h}eh4"
            "q?qMxUz{lXJykx$$CCRF#Rv4?%Lf~yo`s2*+H4)lXcF^vgmJf+_KWVXcnY`8sN>P#%-"
            "WV+HcWU>q3|^?W~RcPBa5t{Ng}odFQ^4~PvAByo_|L!<(FYLiG!@pv%sPtQfVgG0j+`HGAo?-"
            "EpC$ax~G<UA=^Nu!uv*zUi0bD`&Juu$)iMp9RZLnp7<Ylj9Y3$`Gg~*L{$C6?XGe(w|?cmh_N|TY|DrgJ1UVNWYhXnx#+U8{#2l3"
            "!_cKx7xi#MLkQX2AvYjelLY!ER)EMt;O1=M_a18N^bdbaV#W84$CqruCUFK8`*#T?dC-"
            "ToO!aPzl*P_Xh7s^;tQwmP8R|>qB#&@^m};D?d&EtAd8=M$CfDv4(&GE!$_Qvo-aXai%AKyj&q2Jt`5;2LUc$W2E{1n%)OOSWA{="
            "8*-tuN>)2RHh@Qc)PB|(jvWs?pN7QN68yk5^Dp~0rm7~QmIH{Yhgn|D1UTHj?q@1u8oOFKAwzA3;Gp-jo7p1QAVsmL}GXv@2Sx7`"
            "!rzzrnY6VH`Kf6d-"
            "!n6lfd0_c5w#j;$mFp}7gG841UBP;veyH?F1x_rx12c)2zr;H{HVrQsP)e#&xi0ek7K1D<UZX6q}KBZl98JLPG{8PL~-"
            "URO{y~%o@i{wiELCaZM!0*aYkt`kFe^g24p}a-"
            "T|2%!ohp46I$EK)~wJNt=v(#^kz&Dq>ZpRiSzOxjYsY<t_qlR}RNXy;=As^&o6UkRj)t_6Iz=pl&ns-"
            ">k**4;ZuGdVUzdwUD`_<JzP^jLbdP6Z3J?Qax6)z)z{*OiN>8iY7TrQ&%Z5SS(fHBHrnF*%a5CG-OeV?AoWL!PDoGkk6dp`XDSIh"
            "KA=y*xV{kze?Yct!iI~T6zh8V_11K*5~ASP84k1js2{n1(Tn&vd;l+3$Lvi$Q8KSVn9719wH+W($t&g?GB2A!R*g1rQoh)<H)G=a"
            "fJ@cG#?5c;4}W8f!!rH=ooU2mYgq%4~x3yTO))i6+{?iY?RU8WQ;nzo!$hB?T(0T;-j-"
            "tI#2?y5TXSq`fSQ_@bS${{G?KuLC#oy<O;38qf56u`elmIngu^cBnnh(QP@PE1n}M&*7A)o!=$y*We|AHZasy*Gyodrpc%+Sqb?@"
            "jI9JIq6E=_sI-MlQO6jY3n0Dmmjmjtl&t3{GP(d=Zhtjq%bLN$lJ`MJH!S*9bM*O&!Zgz0Ac$zASUPOd!F|kL=uqb496e${27)|4"
            "-4}9+C2(`C~-gC!ii$3(Y&S{=>lSQcST_f)EvVQ&tcAu)OPiB4+xHuRS(j3?Oz7!h{-"
            "Kv*>&3SN`_;TJ_qtyFJo!MrJ8100414nGGD}4-"
            "foPg(!pbB%GRreu<Vdm6aVfb8<!U;C^@%@R}#6^Wn#$*JeBDV@|2o>ufh6u;NQcVaB6C8l9oEmgIu;jt$b{{(=DtmM)hbC+)W>8n"
            "8DdOli)67*XQRDiHrhwV_<IU`5w^vORXX>eEWc}ZoT+y4jsj6+e0=8hWKVlreMu3a;33_=M|jHt?;`(<4VbM-"
            ")+IIwPrd+I%*@K?6K9NpL@-b;@a5O5YN2<1ZXV>Gku}7cHruBkQ{Kb_}K~oYLAnJgXbl+YY|o2D+xsN$56Yr?}~93Pxc-EbUFs98"
            "_t1%)<x>mMJ9<XE8eDS4Kj$)Tbm6${*9m*7d&z8J7132x#3cAf_OHxB#rq`$B;#nG_HY}95dN4-"
            "W6~c3*U)AB2r&ju1;H^qsS&uc(e0!fgQ$Rsy=&bLTsQhv-"
            "V5^%nkqs5i$NunbE}cyB(WhLCr+5$RCJ1U(!%TIG{uim|yhNrFh%1j(t8wMFD8vRVe5sqU?*>uPXJpP8JXh(KP6;ITblr&WA(L#G"
            "f>SfJpmLqL*;1R~<h#Of)K1+P)0dZ5~VeW2s=Gwv9aei$JA8Fmwsy=QE#HzAoE<l?~pi3@Vl<URts9a)WnZXDoVqU1rC{Aq^VEya"
            "s4!tN}|FJ<9z_n!e1?hJ@$h7^7$>|7nFeDO4vGw_5=nlzVZY;tJdV`5O!sf%A8}!T+rJgzhO7|8Z3%+Lqc<U95e6o{l*4Yww!R<j"
            "k#Vlz=39qE$ygyeR#!Ord)MDT)Yb-F9@^KThzyn}<-$ww+XR$sAd1+?E5r1H5<0<+JMS8vtLEnhKiG?OhHaOTxO_0ffg!1v&M4M-"
            "l&eZ4}G=<gx5dw_F&rzgs1QwtDarF-lZ%c#QpNy|0vaF<~}(2Gg_OVFIZY|IV`LWT#VpR7M2yX^T~pxf?$-"
            "`6|}HhffKx(bO3oX>`Pv@gwZ(buk_0exePsvfMQtp|K;yvN9y9oJ|1;<Fd`bl&7m2Lsh+KP>+`$exY3Ax4|2>cI~x`+QlyY9lsmU"
            "X?<K@XHirK^p!gpD3`!cv+)CW<lcKj5-"
            "ND%*wwzk%59&iUWXA+l~l7^uhRtVVl_ej6Xag&*Jqd~`dX|KAv{6ME>i?7bcPa1bG72AMq?qJa(VYrd$-"
            "u4nSi?^Gen4ruw&eSyox)2=el>SWaJWR7w2$S1daRCGONOM-MOJ@U&XAF*DUrLWXNX^&=t5Zx&k#V6nKh?MTO;2JQj`2t9}m50v)"
            "_tvdMX@pOZ_qJu;BpS9b`#B?NH@`dzYdHNERJK(>AEpm%XePo78(eIrG%`uK<IpkBoS6Uftdv504;(FdMoE>zg!XVi^KH4Y3SQci"
            "W@$Yar{$i<^;(0Rp77u%EshL^wA_)r7%Mox6;mR1G52n6v=ITNz5dFp)gA~J24HlNP)K8B!yw?NWk-Nb*8v>Sz-"
            "K2IChcRGZ9wxO+C1~ikBCna}?4urSS)}aT9_1SF?NUR1%vxNF%g+wkjIQdsa109o*RxA=G+sqVj0ko4(T}OIX*SY@Tlsvmoc$H*)"
            "JnEJTuC$QiA`Fj3i$p|_4S6J=8`l!_*6rLw>fH~rkKSjb$Uee}?2m~nQruH)*~BH4dmw}WntLpw@c)m<fJT{27qV4`)pZR@)dq`?"
            "+cFgEA>VXR5WS5jT+~2L2PNfWMg3u$o4d=^shKyx>nd1w5oK*lE_4rb{`f4>i7k?&masV?O_GFfh|bpH++{yiyqj0MzgRUeevAj="
            "7x~ho(3`duv1aZD)4!C=#}vjUfjQWR7PQQ=+qEB3O6V6+AmQYbDtySI9jq!6rYl>dH0wwaZbJ-"
            "<TS|yXuJI;QXJ`Y=7pj3BW>pe*`Yp+k_<8K<-"
            "Kb_V1z$(&isy$hmx`=|+=Z}hRH;gPQ$}Tn!H|pW$;Uu*a@Brx_zR9XMsrtrhD|#Z7>CCRPYV?3oN|nteX^x{7atIIdH-Vt-PVPLb"
            ">O)f9N>SaNyQ<`EhR|kvsoF!;M>9nxT0T!W)dLEM=L`ap(Cj!X-qwd#MU^-O$Kw&>h3Bp=0_m@2njjW>@-|+fUB-"
            "J@SI}fU$walVC7{6@J~jB_HxI3W*T-"
            "4fN7_doSIQU!{djsRf#3$aQC)!d^Lf^ikvpIo<J}JBbO1F8Aq~X1vF#XJAp_r8RLd_vP}cxpc2TJMgly0aDbwFS#Rskecq*E%Asy"
            "LCEVRHcYt+Y!825^s@m~AQjMjlAudJxWK*@zN#Euog5`*Rq%FS<Snii}{nE1$`pT8Q5bfwUQ~E71hlx@I4=r|6NuSKNlRS4h(lLE"
            "Zg*H8dabnP4;BL>cYpAo>%K$Ff)AQkf5+_1WFn@pW1GBy-CCPO1(oU5=WDQnprLFt7v9n7(a#EIAehQB~D)@*=Sv!6Rq|@TRAM?o"
            "oSdgqXMQ^BR8%4e|8sA#2cx^Bol;*ig{;`agEt}VN0}x^g05}}3QsVLW1oyFOp}6kl!c!5hMb>Ce*}P$xVpxmwr1-NV8Cb8SL-Q-"
            "4u)}#B-wcP>y}=JNBpOf%_^343?Wd3`Q;Q)x*)&h9hsip`t{oOU1V?s8^JwfkK6XZI^{d)xxswigCQRX&6zRHAAzZnxHs{ahpsU-"
            "0383N+NNqmgC8J)lsO?U1nB#>>W%=CcLxiv%>abv0l8mRFe*U(vXR1dgQc6IoG69@QcV56+5_AB2wHgnxYC(>xOf9o1ijS{A*M5O"
            "bTmzj{q?Y}q5v-"
            "bAMWY!9j_0%}cn=R*wJG8=IY(zgI8n=UM6Li!ATXuofXoFj4JMNKRDb;SST%4$enzPRv)4?eWE){n=Z)$cZnd_>-R*-"
            "&;1JA`{MeH9+p~^mA0SR#Q{z$$3zvf==y%faeW>Yq0SE7C5FZ;G5kVnZDS4WV{ON*Z|ICV!{Eir>RKhO@pI8Rr+UQNx?-"
            "CsCaZ;9|?tOE;(WLG>9*<a{W%X(^pL6eJiUko_#(R05Ew?#TMo^zC5|_<lkDhslCJE+CeF$|$$I>TI^oXk;`(&vHtVQc)LqPdY@)"
            "LhJ1^@Tp0<Pj$&^}OajQ$Z|gU7tDrnz6jj%B!QU?{^Mq?zW^&A0k6b8)1>^2Irr**hDevGk=SFZ??8EQSzEmwa0ln)bBbm-Yu}q0"
            "Pji9EL0t*}L+3{?vp`oMkCW8LIgHj6#@fAgJe%hb(n0vL$FK9zagvvlOI;;>9$hMpr40^?c>?ZWP_HRQL^$rKEeWWaKybjJBWrNr"
            "x^@c{*=1XM2CpSw<`IzLW${D%}7tS5zZT)QyKxJrsr5<UNvMlB82y-gfsCyv-~235`oVVG9dT*yGnM1-"
            ")z+&u_53GV|4JVIa3h(3#q{IF3@tP^)9F3**WmaQ~=Jrjp@JXvFj$Bb=IKnt?RvVxsUc@JwFstz=XZNIwrfNbDu0T+)qYr)6sl`7"
            "5yC_{}G!^L3irCzYu+$7-|0Hs)~|OWuOZ$M;fHq!R`$2{%uvT4<&b(RvmAu#x~BBa)(h-p?xXC=ujcxGyV+-"
            "zy^Ju}vlDmG%0Sjz}EdRB@4<sf>BbjPrSEYJ;oMTo`S{Rc`;9@;cejt{*~;Q5<}$DsK0UBU-hYvYQ9^40ViN?_~U^jm#mQ-"
            "}dKz3R82ZbGy1RTj*z}8$Y82m{|1t*<-"
            "QKh<h(@6XtLzPc`c~M|tTV&UHOvD^c&#f3I$_@8Ua6fe?|ksK%F*+V<o~0bzSmFz4xZ(rl`yZi`r{MijAdc{Dgt`c36gSLv4AMLm"
            "(OVgDQ!WrRJJj-Mnu;zg^|MwfC@E~lvmQmIPZbE6N}$oHV(0-m1>N9mo7-"
            "}kJV5ZQ*$vXIn$M&w>9eN1ioQlKYxY&~{*b+jH*&YkbSD)pA4tiR5M)G;IN23phBnLcFn>%pLUJrP}sXe4l-"
            "Y^(Wzk#wdSEIWOf`pStH?>JsJo&P9gu{{12V+@`j9!0S%8Uim@{z!TVcR#GEya)fxrz~gOJ^8Qop!Nv%wh+?pFOs~b;57K2pWUN;"
            "K_g_PdV=Uz@47v)W<<n@S`s~zU(pL4AW8t+B*d#IO928NLVT${vb2A+*n5YYNQ-{UM6n2uU$w-"
            "JP!Mq<vY$brf_*@;T!h#(6@k9<ls^FWlG85=No@LW56|iFKGl`-"
            "EFm6W2R^NJH)X9;v`>o_=A!HQ;_Qhi|5QAL)2jEpDA&e{!1p#yE;)cOck8S+?wm3wCz=TA&!QuS#BO@H#s=R@n_vIzV^)X$rhbT7"
            "RVm*^@vk1WOti<|8{kW9NRBu<nD11b-l=Iy1M6=awK`-Vbf@%T5hrM*Z-o@QhG@I0y#@%n-"
            "*{HKKpeKXq0H1d7Q=03`S_eYxkew7YJh@FHrm$?Dj?MDLf)u%{&WRd_2T1`B2vj>tX9#a6bu=9V4R)X;%M=+3vMR@f>pj#`%4XAh"
            "Cm5QZc1%_M1kvgGYHqMrY&18x}R^#Azwown-JzAS%io5aY}=P^HE_?r;wr`@LY*U8k>ZPJ4FX-teRm#F<#jE1mh<gL}8C0%bQXF<"
            "Fv%;0^GUGZ|Mxb|2jjTO(+PH8JnEYnrCvg)Qflt2O5Cov$7C~Y+R{djB;d0?9)n$gU0;l!A_Ts=BQ1i)uaM#xcZQz9Z7|h_t7<sU"
            "C{a9I%$5nXQIe{RCSQ_<N@W4d@wl%s=DjXD;+EyK3cETg~>Hih>LeyYmHB}JlQxGNtxiUYVu(yrbdd+IDdvE=#)lMm7`KEwx9Z9b"
            "APw_wk7Ik0mGXRGBnK}t4mW0Qotm=)2j5T?lH8(pLD*jV9P6)MHm=Ic->RS6_#-NJN0)T4C{K9-DwtGkUgwCyC)0v-"
            "0<tE)KdhhzU-W<qO86DE<4M`XInT5Wp&R}(xaU<zsRr}7cJ5Sp|43INR|^{CZ5yI-"
            "y7JJ)D%xjbp!v&`s&f@FL06KFSp)Ml@C{B8n4g-RT|+Fh^U?l9{C731H14pXCnHI#egc#2O63-"
            "z;jcv^v_nDxfm)2wr3pulqw~EnyG=HMc?~*(5L8!X+iu;3*rDCVk})>lNuk7!n(QKV||sXufzb}6;rI4vFeAU;C+gMCv8~N*uLrO"
            "=2-"
            "baE&exX$Pa7_g>}CH=zlCLbS@&!AQd2|;G$KcK;QI&jea@r$gy`l;LG9{kRwfsL=Vb`jE3~QQ$vv%!i6wr;+DqTBQBqz517$RKD6"
            "q2$d5zA*(QZWY44gb)K`+}cKj*^>OKeoKGN=c6(AyQ07#GF-Os35i>9;jTJ*sk;OkDCYPx|H5?@IW`<YdspBxwC(w)|#xoh7l$z="
            "DZ5LNU~%^(&n0%gvXRU#IeSt|D>UB=&HQ)ha1t49YklfG<OJt(T0xO0is)0!XCEu^hMbl8##rbb`OMr*#0L1F8AS&MfF!?_Hu9FG"
            "W;Gt(WO?+NNH{Z3xU9Ay!8C-EWG)XFPjP=s^3<<c$T120f-Q?r=NG^3?RMVqyHlN*-xmwh}AU?ZiwAi6YUv&?Lc8VCfqqN7V%-"
            "n6nebb;A>N=Z*~Kw`@*#We75n6EkR9z`%+wQI3ie4i+SIev3{0_AY)7DaCHHegoQY*U(sTKbJ3Du)%~KuVHhv3H|*#r(+z9QS4HF"
            "vATRZU#G&YoIHMi)5N8Nwp$!Jo{+#<4<@X7U_`g^d1(#J+-"
            "{}T690oMQLk?W3#{6kHF@aiq}V1`~{~96425tG(Q_=Bt;m#csevzC!M2Edo~6iwPz|{nXmOAsAdw8wtw4Tk_lcd1ihStwzIOW``Q"
            "tO8L8Gk+2ZHuj*_AFbRTrIh_GwV)3~(A*BTsHhnQkfNZH7`xu5W?Tv!~w&BR6f|0_lHF2}8VAY(XM)T19-"
            "kD^g2drg>5l`P&ueIv2bG<A>vC*dxI3y5^73JMxCQm{1G8w8%^>1of~_T5C~pz-9tD{Mmr+-"
            "{a&B(WiK=E_xXLb28O(_f5ZGh>U~t80*km&JGn%=r3I*3D_vEIg``BQe#8UeM=QwQOrw8>~n*t05PHTnog73TPAvpg`~YT!KDV@+"
            "S;io2SddX+tH!+9>_p%+ywwPwdNTtyNZzQcAuMQGHW_QbXrh__fkHe*PajkTPr)Qq5Noh*GRdj^rqeZtXgNHCL(kC!9VHX%(57Y)"
            "z!b7<LHab+PJKwZYI|kXOV~Bi%yeVDk$J$5hiSez=;WssSJ@&h3|PGcYP|O}#P(yoihmT;jdnav*WA85q>WoS*cRS`R>26!4%&R>"
            "IlPzxko8r47%jA`lb{Eu(fLLyPLRnMGu!*F+~QZgA=<v)7T>M#)3)G+~ZDnf1n4)rM`B+W)=e>$K|Y6nwu9XaS>4&86gY+L#bszt"
            "m32Nr(8XK+B<z3zpDbCeZPP<Vw**eA^0h7L8g{WgtLe*FI$-{n~XFRzEGfOp%UkqwH`n1S?y*nq5|?93VR-"
            "(BQJ@C%P&~WW)nH=V;wN7BIH0j5hiwNop5ZJFp85*HKi99a;qb6F*Fa#WOEsuzU=6Y=L@Ezi~a|>7O+y@#B3GOw|(70Ua2L0(&sM"
            "*{>#4ce5)E?;k)_n@^IIh>GlJk$ARUDJBwMh1(A(9jAltHSg2P`w!!<=g3C+GLcE7Zbp6m{l|1!2ofWe1#aUwV^ceT!l{uc-qRNQ"
            "9N-RVf)@33G#A(~doXZ57ZqcqebM8bKTHU+wK0wWUXO{;WSn&^#uw^|L#iCKsI=6$5#1kKP^YLr1h#4EpOY5s$SV-"
            "N)4x1nG4FC=Z(1y+-h#@HL*YXcAoyNF@a6-4ba~4uDkX_n?t#oW9Dyt|phdTaIgMd+z+%QyGtf0i1QiV@7QGDGFT-"
            "L6+GNgOBw~%_q;`Wf@AmbyR^!KseJq|@-Sd12pPjZMsB4N6l0PwAX-nwdcDWN0zGSU|=e2KeC`^L2Rh38#oHjr4SvwkIOl*7_+2#"
            "Y`I?*8GIcIGW&wMO`kvn}@n9tKA3CinoeAcRo8gS(FesODfQ#FKSl8Uv=Z9^pp4M6kq71375#^-"
            ">5Y>1v~Bok=!s$>49FR2gn5jCoso4+o<m1=$o-"
            "gDX{O?aC&J+8{38&sq5uBZ`j;`2_ocaZ|KpcB`Ky}tU=9_*6li!YXdAR+Jw@;6O?yEDaYNs0aYG%3gQMl)yn*?B;YF~Bi?iJhY%b"
            "k9V`Bfq}E&>^D4cWdAy?UO(_$u=rik+ff95^J;AI^ED3UPA0Ti?H)*t;8{#wVDmGH@F&Nl5Y~uwB~=hy=7;7v(X<wdq@)-"
            "d$ClrmB<1kKaU87X^C@{l0Y1-"
            "%MC_)3AAOpac<J`#Y6_)Ca=G(nC5COxg0#2MMDBXeh#ylbe}tY>FRHKRWu21)+0hTK(L!LUb28hHYp90$*j_jntFlVh5Mfl)E;fM"
            "oN_1u=C{R}7p{l>-2)kx@JlH<S9(m9WvB854U84W&)i9JfU(r#IUyGehQKAISoU}2MjYfOlf)$xs%G6QGJz|ovstGAdqo-"
            "N4jMh$%~TIz{0HSz6hv0DZr|T=a1om1td`$2jg76KVyB<+nMEid0Xcge>R_WpGI*I_GsyUUg1wqdTo7q}lG&aJe-"
            "9PDwM=%{fWSw0I;CsX|84f}1_t!DT4J}AAW=~?ftM9Ds5}g9T4@Su@Kw*YPzH%NR)97tISap{Hqm2IALgj{=kM!hWlw<qX%>v~?="
            "Np8BCwZZolJXb(@{zXYU&H#xhC?ddpY?3(yVDpV*4r(n8K=2J~Z$*-s;{MB-Qi*Qj1~?&5m~HV#PG8`sb?mc?!mZ+p!j;9{m)nJu"
            "qlwOyfSXdK{%$M903Vy-}$O-KW(1n_b1KLErYzS};QF^GRmwCr$+lE5{x9L^6fvyL>rt+Xvq5K(9^F!RMj|;-P%!@!%Oje-aJeSM"
            "oi`pIzC`xK$N0&o+U*3lUxKFRg<^7kSx>ny~WvWE)6?jR3K|&pM@fuN6SX*;SA;`tN*@8d=SPTi<}+m(TZEpRM2+)8y!>GWorgQ9"
            "5C1!cB<Fgc9PDuan`L2+kM--M}-nCLQ5vfyq7R7*fo9=jSsjv9PTqe2?r7hId@S?7TG;-R{-;&_|meS4SHS*EQ=HE-"
            "}IpIgw$}pD}KNkLf*F;7lt0$v<m%`Y`-<npq_19vo|D&Q&%{JmF=V-"
            "ad4@Kfsz+om?ctNYy2Hnw96Y@jMz`ig1y9{{=WKoNU!oI7xS`!bpDCKIn<Y2v7XurKjaZIgo!t(6=}q@QX6&9kx%$Z-"
            "1D3{v<6L@I`n5<}<R)YoUXy-kW%}mRjqSI?CsIHc*EgG@(Re*Q(>9rmzp`Ow@g?$#2dfaeEf&{*&soHTu2}_>{akfA!V_r;1TMTc"
            ")l448qbB^Jj8eMXP;A*4qTk=QF%lg%wG2O1&;vyrHtHHql1Nk-8qAEx(5}l7J8U-H-"
            "R*o!(N^2i8ozWL(7gK#1a}kLUxD)KRylpWkq3dP*DtSHNOTum&dZLKUJ+Ko{}3#OAA<`7&w>Y~9b{_ADvII2#PoMowdQU@0E!zp>"
            "j3lQMqe;~6`FTJkNLC-Wb5EUQ_PJ+fvgImM9%hmvC8-"
            "IZ)gAP?2COkgd|s`@xynY*T&^*sAY_mGf?2l=_;9dq8KVES8emmoq8gBwV~`1%W^#C*McG$23{V2U3pSL<w4I9+bmNiQ(=)L>@|Z"
            "S=+ala&fMUtcss6}DBOWE`)lUdx_`W6wRXTl*zYFPYpLh48cxYvTYH4XhONpC1c^Uie&cReoEjL7KoTLk#%xsnSTL1ki*^cmj2VV"
            "-zGWdwL#+Lk`_Sw6z80I1;MEjP^wj;Ij#2%{>LDs!zutn%U!*HD6Y@2TMr6vp9&gav0fzyW+Jmln_%4OG(hk+~TZky0+E}b;3fDt"
            "tD@BMC_2ZC5z|QYnn^4W<VMv*&pNsjIx|d&{L*dZGNU{qF+zC^LfnxNs%J);Xr6XO3!{-"
            "m>bU&2KMCTo1IMchjOZuF?t_ol;dT2F+@yBy#8d~@qYG<P5iTTCesNoKO+;SSZ&I%)<B=SNpeZe7>HCK65}RUFeg~tmlGT~ckhur"
            "(0haiC;jWFzfz_68p@5NR!NuorD|*@MX;MA2^GOpMfLnD${R*Q4p9zLmNspv?vEnJ_pkVY=M;-"
            "x$o58~VvpgxM$);{N4Mip$K^zm4_`LFu+`d2YJw6Z$Acu~kw$_3lUR_vd5C>cT&~c<HQbVGO7Qq%R)a2rJ;ix-oa&y~ZmUvlhA=S"
            "!@qZL9_+<Wm$X9Rp2ghl`sHE9kPg~{>SirpmI1_JFC*-N^GF#e3Ddbf(CRK!^%>6UTl#c(V%7zE&_H<0;Y55~-"
            ")&1=8T}Q~HDYJwjo6MDID_MM3pfY$u2659;u6Nc%f*HKn$61|EZGTCix`n4v#?L(+HSz0Xx6t<A+xm}vCx9kffB#n7;iD{ohS*q>"
            "SCwE6`2aDXQ9&Yc0<^@<lLrMD`R)0!iL@>g*D1pxwD2_sAYB?ihfPhN<2Q1>rbW<q$FRuxw)}@>o7)okVa=`H6Y6r{4(KxeT~bsS"
            "u2GFqU0ey3lPy$FxllIyo8Lsjgpke1Zhz4U!2!EiitCZJ?YOPI1(*m8=&#4n4xkd&M~upJkz>_z+1-"
            "01$7{pBAcMVBl!<%s9VNE~R+l;e)p*HbyJ&nq2a0LdpQn^X1FeN-yY2OLLA#ZJuQF=-"
            "u==+gthe4jM6`Tebwosw+mH3Z7JuP@sXZ!j5f=R}6OkO*#94~}`6`ap%Te4oHe5bt3PBT57yiILQXGv`jjkHHtk$)swA%rYmw@L+"
            "GIk*f95$+BSI>R<%f?a%P@4A+zecn^m@QwVsqqlJ9tzkqgaTanb7VNP3Ls-NZ0|E%8B&**mnBoTczLh1HbJxcw<-p$@%JR-MXRp6"
            "TZWovzq0a``w-nqjsI9Ri!vHQ%OsBVl~L1tOe3Y$qFM&eLU)wk6jGw;@yrh~;$<^NDq>E7PR|bdk`>}>;nuDLa<-"
            "wMZ~VH#uf_t;og$fX#suSRK{mm0xcI%O%Y=L~+PnpP_GJ21J{HXoLkcFQ=#`q%RtAUiZQ!<TK1by=KUWG8t5+@OQ~eLB``WMWD^2"
            "9QUXK(|(MT)n9HZ?A|C@Bj1Fn*OHMXpcKdKyq*PM3WXLLjWp=F+qte1?Onj{Sj3EGQ9%qz>x(oJiAjxJ?6$fz$r@pvT;tAZ=8?R%"
            "~9xUOQR`j4S1PvP(()+7TE?pZUQj73wnzDf2VLcA-jFhnQnttp*QcdobQhZq@gLgzFnjO(<u&}A$6+i3`LB(Ni0CmSf8Hc{`<fDR"
            "<+i*k6xcyq1Wr>d#DSdc3Cwz}f?WvfKi)wzc1r@AW}VSIVX%y+L=xr%Wq9t(wP)u)b2IYm`3f?`6kd)!4Np=mFt7$PB~aS%;I!No"
            "@`V0n0ebzEIqHBSWu`zgvCWW^>`z9lQA%xTY3UrDhTj_=KG2N=DyLnY&a*e<PMb(@=I3m0jHH1)#GJW++fdA``LG^D>D>|19fg4H"
            "4!avReEySi!9Q{S`OFJMRhgQNVqFf1Fe!oJ)0UgOefh2|$ED0aIGWI}!f*-C2X9gvk+nCb{E-"
            "Bq(#44hU+ovHX(3a8HLkkdy?m>$LymOD|(o|$hK4z2JwTZl!qJlXryoMh?upL~(O*|XrEYP38!6c)&Z$5MWNs=Vh9Kd6Y-"
            "Q+CS`50y=mg!E=%Y$c-CPX2{-GRR77>83?*@AbPQ+qLTJUzX~YArhEnW_nrlw-Gl!`$}~Xp#7M`>@xQ^C781E{4|LG14THvhAVXm"
            "zGru#Jt;_G!d&yPe*;30@>#rrJChgk!2XxRe&rIn(iB3HCouZx;oBOJBl2E$)s)@~px<VJ9KDS&kW<%~RRgY5I^i=gMT}7sNFI3k"
            ">ZC2=@1Sz(Z%A@V)mb$E?uiB30=)86ofd<S6-"
            "CnOw|LPwN2Q$u{kO=Vo?@^eL%PQ%kd!WDPpw%VozW26SF91t6|q))vU=^&xs9`4(%1{n6=U4QKlmwh!n4@z+pbY|Dr}=M2%OIHjm"
            "*24Y2TWoi(>{*C`GO4ve#9gM==Af>8XkB7J~M|AB*0)9_@RzO{T#4J03v_oYtyv&PT(deE1i->@?r5E|HxjIfK-"
            "8ocXp$zXT7|fU~5blRxiecExHx>A*P94N3BdiNcEr$`Yq>thWl}PhEI$Eq8zS#JTeP(bWL$_hrb|JjmDtHZLi*E1#Dy3Vp2~fQ@i"
            "r$4rYS&L)T5V$)z|M(J8Ov}ooh#p<g}(ql6$b8YHnitNix@Y%sK@*>8!Sq*S)aX$>C>meH!Yi7Ih(8Zc%=>^w7X@p`2)^GY9=I8W"
            "0{&ugeKWEkf*l)${*NU<r3na&@Gu8onUPgcRy+zjGO=Uwg55|2wnjDC_A7{h^rW2*EnbB{0BblXG!IMgAtU7|hZCd1?I1RXrE~_2"
            "_byQezOY}0)P><g$_lKz)yu=X%I=jxin!Znxr02SY+oMSm${qv9Os@uCzGto!$p6Bk`{IK3eig%?TQyN?==&w4!<r_ooQnpkLB@m"
            "nRvl=#+~uSh@Vk0Vezi-uOG~|udR=~cTaE_?>sBO}OPmgGk}e$QK2amHiWo|!y<i++;9NE3w&*hv;e=+kmpt?Yx450w#_4lkN))6"
            "cHI)rzx1N0-"
            "jQt9ZL!>*dErY1C;;{yE@dJ9E>_H3EfS#L;f>d2jFeoF&H}c%J8N&4Qx#7glAMTBY_t!fsE)uqa3eUIO*7KL2!b54^eY!o#gCYOJ"
            "5b_2s0~^CAD<wkr#2ClS0`ZC(Rjv_FW<w};PzlCGk|LWIc2ma@cFh975J2iVuel?8(?%YHkN7YxbJKlt+xyvTLK|;cu~j8`kCFXS"
            "`G=bgAjqi}`Q-"
            "JDJ?|n0bu}CnPWW5~>UL*`%<^Q)B0$`g&=9LUY~w^+r9Z)2Wu|Qh!dhu&@7d5^^dO(d1wu$<z?O}7bvtLVrjkljQqAR>B~)LxWlI"
            "zE<AH{yZh=lV2XCn?^354AppIC?E!bHt6!4biv17NqtTybdoELwp!k5&~K31kKBu-IgysBr~!BfgWN9T`0r!~4z&)o35_@pkvy;V"
            "HHL^hR1>TAXpMH~Qpru(?V)E{X^r$uMJ;Q3ccfdQKMaUaXyMAj>zW{MpDR!zE@z?*CGsRZ&}Wmc^ELUN#_b``AzQm&q=E|^$0s)s"
            "sEhJD}V)}~}>Z#6w~<gyKf4#+&A_w&$VjU0-"
            "U;mH?)bQ+w@+l?~$r_}@rq{p7i1`zMBrS^B(^9H5ncja)m`cMh)r9j<nauEXGP&7;pI<4%KDHTcTWIb&x2964lgyrMeKud25GF@s"
            "9QGnR4Kn}ev`DR*ndlPcut_^(3+rDXJMYIRw4~Q~?1V<v6n?baD3!VOIlou9|S#@GPh(;R#QJX!pLt3-"
            "qf+W$!F*R!Lb6b`0aXjK;d&&uNiO&k$+Ys^l^`m*_jecTXRCWV|<e0A|PgxDVxO9CeGuo=O_8I6g$$Jg8ESZ;aGMKL=uR{3Y2FFo"
            "KN_t&*K;1?{eI9t(Yd@_1$TPGrcWE*9Zwsur!lPT2&zO9Iwjt=W#J5EWSaoeD*1MiZ1I^l>001@16QAQwaT<_uQSx7Cc|D3t7v^u"
            ")Odq)6R%6aN(d4wfe+X*mYNRq6LFaZYN3@jF-li{a{O4+-sJVV9?MkPp9nG;TRR(#=HDHN<NZyTBC+Q!*-"
            "Og(Xn5N?|1=b?a*i`rTF=|db9l4l)(R!VvPl6kFIrH!ud?Q7AS<T&-d(8uOR?fBWG-Ix~P^Jt|osWAH{&!lGxMo!s04P!sdH}&US"
            "Bkf^qqh?QeJL8?-Z-j<NJYwhndI172{XrE-9jY6K~xE^x{&8HU+<k-"
            "^m1jh{mtlQVgq#bgl<p(zbN&IGXAF(H6`=H#xs8Xb=sX8ctzlaqIN=hw8atku(AkXGnT`M%z1_)23Cc2N&ZL+{Y1uYgCp)>`4R^l"
            "jp7bq<|YUvfKo~<OX1PEW~GTamd5x}t$8JW$E~oY&A%9?qA5uVpQR)Co+&}*%wj$p>h*-"
            "MZ#bLS4oE`ZjJv(kc6|~Lq3=Odk)CmE@he}3C;if`BfcnSIDIV})}9@jifyZB2$v{l?<Z%!NvvsjUK0tfRtHgLr;xONiYx`NrGD6"
            "6(OutE$&^A2PdkpdU5ieMROV_oWPwacaZ|vS=wh#LOOm7Ua=+<m&-"
            "jXnGGuEG9H{T}Pfq><%4^l{!C`2J4n`ae#^)uq^Hqey{cwXzB|9D#38!heJ*wi-J-QJ_zLIwNDA?f<_bIr&0MQT{FR7}FBlNNeL3"
            "?U%hIQ3j_O-}6co4A<%2~ldB;v6H377EsyTEtds^hcq8Y}5w+3&mfO;VEPirS!FN>(o&2pLnEk`<p&;;**T0qLv<#73Z-<|h-"
            "GmXcUKkI3E;5y|1V*+pI6S9G@Y+kNmqOZGBo_Bc!e(Xwi>{+HJPJ_YgoEtyl~GckZ#F0xZMXbu`$kEAG#5EWiqvPWCb9Ip8o7TD;"
            "y*?S2f8pE&Yfzh+uGSVt#v*QPSa?&QsNYs3dI=scQnPFvDv)FL!^|Ot_v3_KYWW;cyerGCS@n{cXFGHQ|?7jCEQR4dm?z!7LuB;w"
            "o-uR?+J<6Au%5NhDa;mx>$dgSE*Ba_>5q{|5#{&{SDDTPg`s#QD#J<nI&IoEv{WgdaYWnUeR*nmY-a<>ORdo`WP@Eb-"
            "+eEZjsD7C_L>OU?`KE@TV3qL56VBO+K-"
            "=az4|SM61CA<PIV1(Ldq>AKh4FyS>a0fc9Z8k5?SrPrb2~rzzg+gg&pEyg;D2<HTtUS@4G@>Wr)#Fo-"
            "r)0`5$v5FeU3OIGWYNcT^;>OD!h409*1L2ivbqmC~JVauT^jDIvQiFAl^&Mm5Nb&f+2Sy_#)BXWID2Hlm;W+d4yuT3y*aq2Bg45?"
            "6HLXk)xk~Q#FK2O&Y5a2$!xgo~ZG#-cch?vl^~@r0$C9^-a16hAbXc7WP?--YRyI>|@bzGjv?2A=zqjhU}Vy2j%3x($$P*m)(Ws7"
            "IyX<l5(zp3d>6N=s>;?kR7t7iXuh&LQod|nj1|HSVcdeVC#@oj$H4MA1D6-"
            "npTH_+YLfj#ONr!M@@0}_?>BX(WH*@N{Efy2wpB>xvhJ%O*6B3ZQpqfMT0r}rV3?o=$=;<5~im+h+IcG`;WSu-bX7!>Yr{BnvGAq"
            "6zZ!;`l-"
            "|0uKD*)_GYPi)t}{2T$SxTHNh_yB0=2L6kcPm;y<4>w&EckhBKuum3JBJ<PNm2hePUx;hSqFH4vIio8q7$@X2anvOhHkuVXI=Z=~"
            "Grx}>qCb^LrL07~~Rrf!yXSGkzjC%4U}!61=;z4jLe#YFVlW8Hy@q)8Qz{3i|$l=Uv_!PJ<OQ~Fpig-"
            "LJ`%vGX^T);?Vp$|e-W4ZK5v){-=9o6g$)3w@Ck7lQ_6^0>fWh-xK92RA--sEy9zN6GUcJLKsmwSC?!Sf=@rgtta*T^Q8p^zpFQ_"
            "ifUK{sjwzaP1)kd64Cf~+X7ZTdQN#F#clY%Gf3NAe+SFUBbGkLv=qBW%>U3#?$*`*-"
            "<NkbJaxGW+Dlvl>&xq`3I0L3u+So6j%Nt57mx1711w5j7y{(p*a^FZN!Qry5TidKafp^piB$%}U@<KsZ#7BMs1ZTm*31G=uu^5?>"
            "iib{yu^uNAB~emGC}?Lc{zY`039`C%k@+s=&pmC8lTK;J=sVG4EXRqqKAUA;JXmlVhT2Oy{&y(p5GzioM;#dFPIrzwRlcA%vow6r"
            "-dc)Wd!u7UMNZOiUXDfRqxk8Mc^e;MibV!N2Q&dXs-"
            "#HnrO$z>F2yxc3xyLMjDnViOfj}UCx{`Lr_(w5I+k&b?>Xsp=5SwknqVlG(0Vtj64{aALF!K1KBhKwymmF{(-"
            "s1^l2PEYUIe3Zz3L@yqScO8%|FKLHNI!+o87{(fH<$z;NizFEESYm`XoFREhKC^sP$Cqnq)$#KPuX($bYr7P|7dRpOQo42h;||$0"
            ">6*>PD7o2(C{x|hSK93O;<F`-i!+k!xzub+Ga}M=EqfHnMDrR-Bk;LXHIZfG5Rn@73dJ}qv}}FIyJ+!})au<Q%(UQZ_EdA(8(Y=8"
            "y`|JQYI>hHP;(NL5(}*Qvmxj@A{$^+)6Z1GR*8#{lezfasM8b<D|T;!5xR8tBEal3S;mMx#X43u;h^bUJLfMJ-NeRQ(jIiikcvZ)"
            "(g&N*k>0cHb;JF|nE9J3;n(FLTt`N1Yv$^H-"
            "qMwMwbu>lkH@l!Q;(v7SIbo~+mHKHG+q~s9Mi<FFDwnhTNZYbULtd9COl|cP|%VScXP#WA<E)JJR<VZ3wsAWBpdY55#X~}afQ%UX"
            "lB-Cr=+<LxSPb!<L=ymVKNuJPHoWLU0DQ@3(+ZIG@3cJ_d=@U;)Ys;Yr#h18vk4NO7*I!U$3%~W%I|%l?f2tz(~LDoZDYA#jJ)gs"
            "g1rRqck~i0TpE;kHoGn)~|gAR;)r(m%D;EMbG-"
            "M1}<I8yW1f;&Wn2!IuaWLRsGU88@@$H;vtOQHVJ?KkFdAfwPweS1Yf0au0L<2{wd$~Rp0bg4={ks0~D%i00Zr|VOJY7X!zZSK`Jv"
            "vt+QepZToyX_g+gWg(!+a2H`$FjkYY4aI+dHZ9=;j>(9D;=sm=X={#n8izyXjDRh^6TQ^I$Se<__xFzU7zPiP@zqB>M`|$h)*SA}"
            "E$68F!f;gFV9GA4$RTtyd?#dwZmuiqXs(p8_lI6PBr-"
            "4>6o%u)h9%>d1a_*LKVpFnK9bI=pp{^c*Xse7Mw_;mdpv1C#t|iyyrcd|8YVYm^tvj^a&$@0D=f<zlstPpMejZwnYTdWp>piTzPh"
            "E}TEs@M$0uN{+jOGLC>#i<&iV>8<Kq${9q@vKQJr#?lpOoM3*EsTYA9hgh?)7aKaV%qU@u|yWOt@rhZeR_9l?jP5b?FEq8`<j6=I"
            "S(V@dHF7eAiiZxtjcy%4)6bT#mx9Z0&Ho(=hT^>?RH>0Ey%tJJox*?rs;UG`ixw)FOr?XjGh5I&!|Es8rq~WeKj_3M+#31e>;;br"
            "X(M>$4bolsc%D^0>}4ido;Yc@P8LhSasZJBPz#Mg`+#0@#+0k`NoND4BO{x4vygXyM!)y_cO1!MxBFE}5Ty-"
            "$|=owGfoQoZPmWZ)BTTFGw3TdK|3=WO~c_K{K^=zeT4*U6Of6PEDQb79e)<)3z!vn+<y-=!>auEwNmm=%fBqLJM0f?0)(hn+-"
            "v7WB1+z2$1N&me6D|zqUNBQWKO^SG}uu9UZd7=bf$v&5yLr?)Dp5@s@YtsNNhP%+`GR+ZdCGK+bl~gj%&^FrJ>h4JkCM4axa#0PH"
            "&3MO+#i*h6NN3jpe{QL!-"
            "Y9^us*w5>kHXmJcZBsC|~#c|;{nl9G7g@zx1pfDXBS}gx%jrpjwz0hkA`r&u|gjqXuP)8|Y@=8GHwOi5Qa3if#?Y+c*b?m9`O2*b"
            "p5?qgt#pQ?nK8QxPSd`VEp@_vnoCw%_HQ=H@wKm2q&!8e(QfTwJXwEx#svskBKDDu4J@7QDD^|CW>k<~|Xe+Iz-aW%#6~%zJ5Rzv"
            "gs|#?gc2(B&Q-"
            "N(ll+Jo?WG&$Nbc435Rx6=?%XGfqtXTcppSEtT#bpb|3sLi}&|%Bdc>oM(Ni5}P#Ao#2xC<)KkT#=pi`YfYtTbeTye8d>g}3Hbxb"
            "ehYs3^F+!u3fo-"
            "C}r=;VR}xj&VoBl7024=C;|eD};amHn0B!_*ZSA`AuEC6;!dNfhZdq9(hgHOho^;*BqI`m=y0eF|D$EvuW7i8R~TXraoJWTCoynm"
            "wH*G_EtN@Y+wCO@wSMAAWs%8-"
            "TcG5`o*`q2kzuOdSI&cNPc>b!j(pc+B+I%k;sHo(^>@tm2T{*^aMFq7V@5_UV}f9tpBaf0>Ty5|J;>Di!-"
            "K3{w+&v+6%nUTnrM~&IBlu@7~hYq`8p6>;0hxdY9IU=5C6CA|Tf(Hjr#-"
            "Y{gWEZck_pd(Ym3i?J2?Rj{P(=^tKP7W_$62S}doL3S)6YsZpa(RWnm%4!qP6bHqrckAuyImovq>&wicF@ULIS~Y&#Ac7pOB~N<~"
            "dcQ&<5B=SrPy+oi@mfF2Gh+Eq15}7Gfx$`ExY=S**sCkN0Xe6;FB6g{ipkqQS^vniz*P%XCawzuX(#W)XtM9SvsWy|)l;S2R`+N|"
            "f>^yx^;@WBhC&U>Q^S>zEL0(|01a^q0@pju!Mj{|P*<-"
            "58L23%5XM`AL!I~<#$_O01+`(jjs8Lli>>&=d`P*Q0<ILMy@(t|eD86j_=gL1H#4*`>pCIM=h?2vS+6%s;OZ2JxLY&Tfuy$C?_L4"
            "bu5k&aM3%e>rby?_ax=P1HFq8JusS?+2;0J;ILeXKW7X`Q!jwbM7ti;j96X%bgeCV=>_Du^sUw&2Mm7`l4h?+BuJMK~rd2(?TP~&"
            "=BtcpUl}_#Bre#^A-"
            "n2lS*y1(mry4mnIa0=LLCvWSLf+Q4`t`w+S&T=>)}o_O(oEiX0a0!=AV?7KF|${m*4(xgEjzk<76mBIEma8-"
            "UwzXhQEk2XQoplW&>3f^T&(eithuxBy7zO5`Acx=pgHI><@HtSl|o<QuHPJJDT_ubei_4Qf>@mnR>0XWp0nWcx+uNm3NLC&MqTE~"
            "lFh0MWaYDoZgDN0)zijiSVFmCx<UdqPL_&xQ?ASiTeqZJEo;th|N2H3IzFxvOH`6udNm@wI*6reXq0JBcXhNEh>xfDmbIr{_1?|="
            "o^XyM+dbc`@T#J`tYP4$sh4cONGhvBy%;~GL)E=B+)@qXqgbCcr!_$1iwU#!;j(z^s$H!wWL`qq-"
            "k)J&XP5v#QkCX?AbV4sojJr!Ne6-MmUS@zJJ)sfXfQxDt}bJC3}WXM7fq~QA`8FC)27Xh)qi-"
            "~WJAWhTZJVXwf;xHCl$w_p1v!k!o>WRoO1H6R9n3k68bN>!({ic_wI~!4>_UQ5!udjskRi8Wk8*{YoH@U{?<em;L6({p`8aR9dv%"
            "WtAkW<F~HF3DQWd9D`zfSGFwH)p+$2H8GFzJ83B+3wo(HFblY_`k(RSxXzO+D83cYOsgK&crQE_SfXP^L!)|VnU<w`k`y{kXqO!6"
            "$FAqAZ1EP+R++8^I&2`Gn=}OjN{k3!Gbe$zTf3l(n`@RFszlCb>UGhl20J%_a$jhW6l9<J!NZIH^_|^7`*^+@)>muwcY(nd9YTWt"
            "~(<!U6&;Ww_s(4jhDnVPWAbIRs^=f5BYW5(c$#=gu@rwBPT=DXC<9h3#01zZx%F0hc9~>*3u?!}ZJFQ?gG~9Gf_2%wE2qEtPP5Gc"
            "fTklh>?18!h5fb&~tvIdR=gF2WVY0vPG*^&7Pa>dMyZ%&!_zHvOw}#usw2wE{v0E?mTf|m}4Oym@QsX3jeab461z_(!ga+v?#D$G"
            "`>vvsuy5=sSeT7~rXpDxYc2-"
            "<cR&+q;(L8p~B1h<Dh4f7r9>QC2h17naVcJ^?pmzw=K6vk<o3elWB0w6xhqb=8PJCO23^E7WLq^_ny;TvRerC&Fs!&tvPrK7%>_G"
            "4Hde8yQ><-zEYLRsawJbr!j&@CY;-CjGstAyjjHG$%JPGrm^Sof{-"
            "rT!tnCmTpO+7k9k6>P%AVSn{k1pwM+Np84?CK@l{p@j^LcF7``dC!&tM#qkRSHHAoI67Lh(5w{y?LZa`kbpbfJq@BpqSU`Wyblor"
            "q^mPxm^$;aO?dYuCYw@PQ|F*+5^;3oa}e*a5-<|X7u#LxmHXJx>j@PM|*<X-8=#Y1enP4)-"
            "E(|1tl%5nFcZ>$@OAkmA5+gQJEQUtY^Y1y<KtjU5*E**T&bPwA2-(yyC2w{7%a&p-"
            "W#%Z{BJu)iJBa5am~nrsL|F{Ln;T3#})lJXQoHoSubkHTl-mKrPedMdOg~d_A{f$~MgvQS#c4dl2D@Iai2ez3l{y1knre>doS*7b"
            "w*<#t6z1;v&aJA*vzB-g^U1De4LFdM>nKk&mbL-`d07QDEya8O}YFK+3IVuy*4v@c>2x%Y~Zj712zd7OhrfNnC-Gs(n=>h~+Gy;K"
            "&HNWEF_81>M7W=9R+P@(#S-dZnZ3F4QBfJ#IsJlR!Ezd7rm;u*s+g>yMjPg36xNcF8U{f$%I6GM+0|LJ;$*fJ-"
            "w{`EUpn7u~C@!2!s9ez?V(sNuIjjU9*zUZ``;=3ST3<+rXvAlw69YQjW5%jJD7-SLmT49q=!@8D4L&^Nqg&Vg7ch9)ObtTZnzea-"
            "9&{PSsW4U>mRG-U2dS7-q3*1gWQ@UUorn$P5|m64RHHpBfE{Bui*E~aJC+>|(;0}Bq{dM9ejv~C%X(vEdgn$1=BZMKICB-tr{S)*"
            "UDL^wj>>FE``h?CDqifeYeOD@)Bp{P6PjoyKfh?}Ok=F+ztNYK~p(MYt}o*~5=je|DF4~M_5b9=6qSewWB#ePt~MVH#tO)W!l-"
            "kZzF#5;A?^AFLZvRE9}vtv_wQl>G?cD*f$@}B6A_^tZb#k&lx%VB6BUP5KQW#Z@u-NhH#3BV=RZH-"
            "%87SyyGtdi&HtBb?@)}gVBIC8`0lX``Gm+BdfVu2Z?AF4wx*DVA8x_Zz>yvqx3h=gX`WhcXo2Uu!;pe>8-"
            "cF28AGb2D_*HaK#8@?aqZg=cC_Dj+QGytUSt$~Aq&15q2?Za6|zGXzZOMY)t32uHqInK*Y2f>NDhqU30{XCtby=ZG77Gs$uX8iPo"
            "LyPZWDXfs*Sxt+(9WRRT+xX#{oL=JEdY)_~3v%M_OUn3l2u&mLp+J!@-"
            "Q50F1LuO^+~!sn(~wdxYcWL#1gc@URcbA<WIQ(JvPOE%v#Vf4ypG3%9+;oHQWB~svTe;L;XKb3Qgp!V9tb)mJXEydI%q~A&b6ng6"
            "bqq@l^vp^uG@v)AcE3}rOFC<BDA<(siSJlVN<;LT<lo%?wRRY5E)N1E<<v~nqA&5808=Kb(hX&zonx*|LrYP1bO>zS|nxsp#R&7<"
            "s%Cn!c)`ZUB>+BFM8zPt}YXC2A9j|Fza_A8V(riHt>p@PVjFtdk0($Z}pEeA-"
            "5I|5R<(di&SXKJKeA3DeZHvCeOT_yb{eVFjbzW!1!aRY~qq?lML=Nt%iFYJ#X($pYAW|x8Kb?jVrotl?1y90967eUCiWdVHgV^uN"
            "_=ujXHER3w=^RmHpg0stEETqZ@vnx81&T{+2c*yi=V`#|=uEAvX1>4CB^?1oU**d}5*0FTi3{k3B-"
            "u+Y}rvu&`ZUEdfa<y2#ysc}C&QYlfOnWa7ChsqaH%JG}+S820scQ7&eSAF{Y3HtjX#eu1%SzHzOZ9E42Q1T^1y8B~r-"
            "P}pq0R}xDY9e$r%Rxeyx%n8}DWKFk0y^7|_aK)s(yulDII4JK%KU9wZn170TC~s(?@J2{|326IWpx&7;uC)YpZ5RE<Y(xIIt_q$O"
            "<2GnZ3X5g2s+HWkSBIi8%RbwP$F<~5<mX59bV3F7=?#X{yl0fOo11~6#7zXky0D3*1t;e{AyR^SSu>{^!?ISh)Ar*&Xs`(LP8|BZ"
            "WAO?Jr-+zbBQ5<XW0lo4a{9xIzok90-llD#E9254-"
            "E<pniC`TxT0_^R+wJtB?o4l=V~DsLXIJlzvfa8svFOHgt+xe`5Kb8Tjo92h^;*;F+z#hD-"
            "x59E;$)1{Ey!*exYyH{Pbb<Mz>4epnOm=Rxq0_lOAeE#&AU0%dRX>i`Eq7cwmJn*SzA$ym8TMvd()n=W$Xnyc8Uj(>>~3z6|P(Lb"
            "ewcuiwOlC80<Ffs#NbIv%~)6x(p6bjjw#7wa{Y&dx~&`2ZFa;glgTEj2JBPRN8KvF%)Z9*2f-"
            "=#A&Sqd!()_9Q74NG6k51ZmXANe>aGcEmyA~Gu0)hoDu*r_LMOqwvJtM)1Oee-"
            "ozj?-e_P#bS2kL+*uaHIX7w_sla+fwACKa<n2~-nc-<L2#J|Io9FPuY4H49lLwX6lf1iX1@?Se(s<UkXFNSO5xNvRt-ws<3VH3rZ"
            "q{GQ>)6*Rv*%G?;V4Zmmr(1it^+rGucwCUG9**2x(V4@32ErAyV%fR=xuFnp|12Msdv;V6g{_w9V?(ASB)sH>170sAk)-"
            "sD`#UNWWNuf!|WDtvYgU%cQ^8gbem7}PGnN)j&a#J*H^vDZQ7bu%gv{Hr-NcbU((XncD!!xacxKaCQm-"
            "JkjPfuWETRx9u>Rvt<cR8PuRs3bp-hRzA|!g)rA(Et^5)P9_sOWnF_8k$ineHTRLo=g7Q>JBGx77_h40AX8PbwO&2f*D)jU!tHd;"
            "!l=vIoD<)QJl^X6_qgdFE@)0KavNXXJ==CgmT?xz4w@IE`rGWJ*)fcq91syHYhD|Wp;t6Qk-trPD@`xZ}?aOXMCP=sGtx16<bbBT"
            "f2<*J+ZldkZe!PSZ(#YXJGu*tS|1oSd+;CUw1JWi#d3U>Qp6xSNPXUbD60cB!gQkJ#GKQ)3P?T-"
            "#8uV&#ThfEd&A?qhxtC$55~mnW`x1$>Se{YttErx!DsVy2?n%rwy~KG$2u1;^>zdX~M-"
            "6nR0@m~&k$p9b7tb5(VC|(=>SSs`UKJNpwN=#x|L^>^y(^RnYOEBG=QKh8p7QkH0|IePSXsD8yS=kCvUt*!o+g$%0X#tLYAy|V%h"
            ">JJL)XQ4Ll`{7h%Ib$1qL6o-"
            "ireExFb>>_YOrRTnv%lr8v5|5NyG_Cz|(+M@eexUX%m3Z$7PGI*O8(M7C8xTg%40<T;aR6Cr4e?hbHkkRE*>O9)NxTdrHza~Du_o"
            "tmS*MM6GrwFV`3b)Ku_Egl}xC6p(i^F`^%i#3s`C#|%DfpkexbD%l`|7AJwUf%LZlzPZfXVOEa7Mk8qlP;#1ae=IEw@yt=eB8@{O"
            "`j^i(rv>w@^F<&&!tlXAWrw-"
            "`Z%AQWmD)fr5zXD!u1FGd8^kdhArc11aDLknmbwm4j#kqyTLZ%=H=bWEeoiHP+pxoABNP3pKGhzGA)3+6}ngFp~%O}1h7Z%+Lt0$"
            "mQx;KOpEvG!Ru9Vo-"
            "M6jv{(8i3m)t#H6><Tp@Q6rKV9^tzZ8V><i{)y`d+S*>uNO5zP`vKNBz553U_m_yoBtL)uz^Cy{Sd;+LFUtq6cVfG@O{R6wNyqA^"
            "Wa;_f@f8-@2FyF%%3I=hNHn6mz`2o25)F@iFspp!D%p1DJ59D-"
            "BrH3nm6ZH&;(W62bx%3{OQ8PG4;$ibY8TU#fTb<dzTGYD+sA!w%NN9O&fDEY`5grg7D-"
            "$PuAa$f`O=mKUM}O?S`b&N}oc8YHtWPfpJ5{^nmDnM04XbhXNqRdm+8lQ2s3;C?1ctdT7TwFV_=^H6Zj!TIQ8t|pLvC#OrOxJGG8"
            "T@{^%NIZr_pKn6c=&BL31Xn*?Va`nS2l+pyT;!QmNSx@QQ3i^G=FQF-"
            ")Jj_S3u9aZ^VzLt3T+ft@g#^Ed1|TvkS#mJ`Pz9K01Ib5hpZq{qHhFAD<`1k=7%h0399vX7BX5UphVc+A_ObYCDkzRVJ5w!6jusQ"
            "78-pcG)7#E<tgZh5pYAX-"
            "o&Aryd&4nFxwT<daNK^fhFFw4`H#U`X$*}W8|+02ttc>D5T3)vRctfcic@uU8hoamQ`|l^>SbBzl%rGH>GSelV9OSJG<c}1QEp+^"
            "HX5!vUNVfCCK(tLY6kPI?O~PVN_oLv0)6lj>S}Ti~nD55BO>VD^Dl#WP)gQ6<E3aAj&iJ&k=#e&2@7*Ol`ET9sw9OqQP4<4Bbdyu"
            "pm7OEoh4cX??Cy)!tVH(6F9N7Cr19Xh+O+DbaN!{Z{MoVhEWq%HGe$y2gScH$g%_q|jX4Rj=%N^=3T+l{)HW4lwKzrF&6~wWSYQH"
            "qhlaYMX7u*6+fmTM}JA&deMfD)lF(>B1xRYAE`;U7rw4&sO@kXf>mwkbLL4)Ryk1XBist+)nByFbG}@Ww>5xf?=Z%)3qTRy&j%~v"
            "Et3gZyx~`M6LEMLTTB~>JV7!=MpRQ*txMS@Tlajp0Wf}WEd<kB4{sYGlC14UlGUMNzEX#tD4TsQZJtpmtwxkO5AdYAaH#r?G$6a+"
            "bvXT2M**6`qJl%CycdAyc<!$hOCCzcwrz!_o1S~>v9n*RCi!nnW5#y!r*%+1oNPFBLeZ}b|<c3)GeC5-a<`cMdpP<YgH}-A=ga-R"
            "NF+4y(-4JFVQrcYfBV(H1w-"
            "!Yq^GE@f<HyJPEvwY4I=pBJs2!%cR&6PmoWfCbG6WCkU<h+B>_Djowc81T<zZ$H(2UCespEQ|E<*Vqv<~FbpONzO47iF+oD~E%Or"
            "7U&x0V`Y%h@&CYpD__?Tpa1A4Mh%W05gNx#RZ<!W6ttOl)W<*jLwC&}WY(YeQwc4}I2+P<UPoD3gTdtWR?aInnV*Cj4LXq=P_l6O"
            "j-"
            "r~w~4yIL8mFky8wBF3O{5H<=E!iI_YbXsH_w5D2p51N2x$q9TVwKvX$5<ifr?(YTv|hRvvZ|Op2?fzD&$zW%DH<3jkGR0ugRw4Nc"
            "4VF@5~i!`ARrxT(b=A7x@T@8Dg-rP5Th{GW_~xP3I51ttQ&4@2^6XUv6_HqTe_mY6Vc8XLh8vp!RF_DH~AO!E<J}H$hh@2(Jc>T0"
            "kpncli#FGE1GsCOPwft5I|4-pGO}8fzs*`qQ~G_e&nO5o|>_V?Y<UP%juX)iOqbbR_^W!HewXKq*aFq=D#gH+hSYc8qQl9tpn9O-"
            "NQ@oWy0{H{C3)=ETORy-uPM$g>|6OA%#ru5d!O}S^E+{$WW$^yeqZJPka~Z?P{BKy72UZqItOsIXkkUUd4cl=#HAtWQN97qTYa-D"
            "SEHio}0X_I19+?PwlCVI;2x(<KH%Tpq<#-?<l3ViWUVu8Z(Kf$moS+)z%c;S8>O-"
            "PKLi4oP9K~wxI6HN5;5~1GQua#G@u^Z44rMf+o|+@(cRwpH`~~y{oY>YOAQ)Mje{*re^IUYF-"
            "U!fv~%pp~%N2QLGi$t#8{AJ68OArlT8bc%@M3tllJWpm4mo;1V)>Z)F}E_GP9nLv#u`c0SsMkR49j*ZqL0_*vDoVmBp|fv|PLEvt"
            "LOC%rbc>|OyVP`^tWj@zr3XGtl;#i^PLT;kLg=(zP&r{BDEym&tS7S1)o5gof%PHe5agOPj4ywMl6HZqUS6Z#(C%5&Me`J!cyr_U"
            "ISS1)<ESZYUp$~)7RD1j`dTN0xE<eLR_o$rh&tfd9G(Rec<Fk2|(g>cqzGgm{uW=KZ7sxuJMha7+E>*z-"
            "Q;M)oqj^sdy00**XROPpE2Wn~)7^9_vV0|Lhb#~>;Y<1-"
            "wW!9I`kIA&mT|&@?y)hA3#scpQDkV1RYu|1d^?k}&))TJv?NRE3xq^75U(<}J7zp<aWBM@ax|+2thXc3Ja^JPA3>K6rvYHSk6rj;"
            "+0dVM}8gbE{C@J2uJ3wDQx|bMqs@qX*y=^yg1!l09r-zIg9Q5n-"
            "Hh35_^x3>wkR5}N(O#hi?7ar}&8Y$`GL*dR$n~y(=#wDBdZ6KKN)W5Wh2D~4x6lF7MQa_d;bfo&qyOHt;Mu+XE+DE5&c}L;qF@tu"
            "X@&T$d@@XvAKfsG$h2g2HCWb8qQiI1)|WucDTMcNt<IRJph0U_Z*=Dx4#>`5)Ek`SElgvH{;t7bS<}^omEN4#9?nHSQ66{IP6aG@"
            "_cj;FgyRO3F_9SmG1Cy_*ZpGxez;kxUC{#uRcH|Qgm~oi-eG1w;5<Q5U2M=5qh6;sqAY2NHrSv$t`V=3)O#okK*aOz0s?rzqbyIK"
            "1<dr4M>d`t;k(wn0^J-tsAmq3OT(&<yUTpFV{lJPLH`05tgwU34wT3EVjos0o*TWRqlRepE}xF-hONi|(dB`5;2z?|wnZ(TDXM!<"
            "PTgAh&J2Q7^F+}I=p8spx1lme3EA6^2DD{#!MRb4th`Tz%DlTN2E8RLRcqJVz3l{Mdba36r(Jv)KGDy3`3j5~4%BS*PV|Ogv~{TT"
            "qq|hOZ&gs}g)S!uui`$mtV}y->RH+++J;_YB(GH#4bL{R0Pj|mgYVTbtPbUvY75+7-"
            "qfZ!zls}=bp<nWF1z=Pg7=l<Y8{TQr)^w8ea)S8)NHiK?pQKLa8lF+_9gadJWhYwG?+4Py({JH_Pv=eyH4KGh-{;U=-"
            "PY)lyJw{&K2{_mdpyoQ!C8Ss^KlJb}|+}JS|wQE}%=TCKnW5>fiRa6DeBy9@SdNB9$DOZRM(d%A{M;x>?@f#=@JuYPX!3jXL-"
            "m2zBf?Li8cfLH4_>9z`>kKq{m<f7_8+?~>k-p`OT8!!^jX-mCs@^CZHGV)Ug*k5d=TN~$ad&=f@AD^%AM#jSQ@d#aHLcwuoHjlMN"
            "_DbE-&xx04jber(rWwOb|i$p3?jP!yers8imXmkwo7BCh&M%CX6oi?k9Smm?&w-"
            "tH#*Zoge>5`tvKgvvzNzuF+D}s1i%5+8ggiduey%!9-x^UQ1a<YCd8(}fB1sRs}Ze`xkK(?f+;+p=_?f``zIM7-"
            "36i%_+uw78tL<jCX53?N)d|ctK%iU1ZIe<5)U(u9L2W`rFKHGM?x!t@L?qtcjZp*l9|Lu+jPUkL8hX#oevDO2_0$QHfZOSQNupdV"
            "saix3*$6TiZuX{^JA}Dtk>nA5aw^2jPWvcnKdy<_lJ*fW08+$S><2K8q%nLo7M?xJpXK8myDnj*oDpl`YYl%PdB;><Pi+G_6Qbdr"
            "@`ZgFb1Bi99BtdgNHsTlu@pY$@n4P{tl(&Y4s_M!hdu6_?l$nr~-"
            "{!n>zgDLFFVDw<hK=>GZU{2HTAFsnLd#g+6IK~};SyS`H#{mBae;J-"
            "8mG%9eE<i#uLf$CAK_xC8%Y#OO5hM9f@P@~)e``9MY@N+Ej0r)wPkutCxX#6z@7O9>D0~LLJk_X(3)LTqoA5+^;6QhFLlWZn9j@|"
            "SHeM|XXW&&4ZG0hUB$*44PqBUy#6Uw0!OO{+wh_(r<UbuzK3o_NOD_kjddx0u1yJKx*U!1g4wlyW?OP&W`}f}B$OXnigrjc!EJKf"
            "{b4a!6IFlNq0oL1p;^V%D)AnMX0_yZRS51{=_xt!l%6-"
            "B&>(EF;%CS<^orEgdCumvqcGCD)_MoZJEhX9CN~}eF6}G2Xs1iy2QbB3evq1T93W{)mN<PdRxgTd7h9Cft;8fN6;vm7B?2~7Tj;7"
            "=@o+L!>D}qp@G|vrbYrq0qUZD`ZyJ(Kp~^2)19|Iylt0oZW~fJGJww+Eggk$q5}a&-G$?-"
            "Esg6BNKA3t46m>tWo8`);yD2a{`3nb;%NxO5=>Mpl?qf-}p+)&p-bSs~mI#QmNj-4gX7D!6H60<=fnmc@27`krme-iVK`Kvp5#%*"
            "o3van#1n}1V2Zy*`XCG<)8F&+KSO(1j8Ph4PdEe>%LThA!b-"
            "h`Q3=I?c$F;z`mpXFB@V6i{=2#j9#<(N2l3Xek36po?o)_mrY{>;|o2Ak%DP|YqPgA_YuNe{$cdjd&0Yw(cWo^uH4*lby_o4P+`C"
            "S-n86khv>KESBh>4f0UO;6;@(lU!)@o8O6{r?j2@I=#(lXXMT5&zuZn*EtK<k&0yo9vAEd;F&XRHpk-7-3Mi;OH-"
            "St;yxWTSv#^Xf`Wrkd6wb5VTRCL5QfzRU$wn%a^f70(E2>f#jB>1a@4)m!V5-Uq$fSMx#0;a-+)FosXQQQ-"
            "MXi%d;OcmIWxSvRF~cQ#t*86M9WSBkBAcG!nJ-DEA1qoMbU7jAn=lKv^Az`Za=&%H%%T2vA`!IqL_=?%pLSb14sbH}aq2DnLt;XS"
            "mGdT=igH9+Y&nntZ5k5@_*L*6qP2XZ`CPf;9s-fnn#|K=wuM$`ibq;1HMA-yaW_E&=}dAXL_o*jrf1FkeNyjouWN>{6-RTgx$yF-"
            "0~jKMGP3%%V8q@g$mGn_{d`O))Dhc~l#VD;t-"
            "IRCQAHG#j(=v%gkLF5bGQt`i<buoLyC5w5$tFp)iYoVEL{Uo~AVJWz$*)|v}%&?v#xG1A32r`3{yF%vT%abk(1n&YP12Qc2wkkGB"
            "dKOzvt402Cal9dB=@7Uo4_E(c)&;ny)er|+YdVm(-~}#zT}4}7UWmRYxP_uOLWUTIA7R-"
            "k{#iVO({NE{Ooxa`@?y4WHG=Y_aRdkKtD2SbbliSD0Sx6b5Kgf4Sw%8D$?kb;d1Z=@>RnU4pwgl;UTlb-"
            "(@z;MlwAXuQLUgSr<WIF)S!mZt<Q?OJZ)_Y)J?b(`n#Ci?ma`USLzNBM2vq}UvBd*kQcP-"
            "{V{A<l~whN$u8+>=wmyu_Z3g@yxeOAjfd~S?LM{S&)iJeF@5c!njCKg3Tx!n`k~jYwD2iR#$ofega<2%E(_hsoo)%d%frpqj|SqP"
            "%`b?cGW9H6xrMleVDNWi9Qc?|o6TP+Io8u~<zAWD7$F6-"
            "hWbee3H{g*?^eZXWY8Jb1wKvh4bAC_1yhR2y9p8x;2boB!0J^NVFfpCAU3vQJdiU^t95!Yo;dfX8#5F;HYuv=-N(FtRJG$|r(Dgy"
            "--P4@6=P5xa4~4lA=8J$3#%cn2mc(^AD)f#1Xawx(06wg_htkOw6QO**D<|oG-I@JdeRp<2tv8OdGu(DY&7eW=|%4=9JA*Y)X-"
            "l>Zyl)osCS=ixcTV%3a#2|5_PNnwE3mzS3Ds=WYs;ORIa<cHanNKv>Ye49kTt@rgh74Z60+ny*}s$zP>~?#gV*(iWfNQwhNK%y=9"
            "#M9G9QMmH~>JwUlLS%*W3t-"
            "ZuZiq8hkVE{+JIj=h@a?Dqr9Hd;l!*&>>Yko6n_)c4h;@D6>x_Y#VOP_GIerLX`JOifz#7RkG;ZdD)fK3g@#1XBtJU8&pXmSsmnF"
            "{CJ06L3XP1EP51L9q^ROwwEp_^=TX3AlTVg6dm-h}jOSpw``HMb3cqHVzDXOpj@}p@taO-UTy=j_a@?YgC?`{%KtzAhKY$K-"
            "{eh6b_M}X1BmCaR53+9XC6#b+_~mUA)UZ66;ZtS0;q#K`RJXRXIdxon*Bt65omF$MKjvU}zuq2nw3R17nNwwi#Qh1YVDc+}ydkA="
            "+WB#>4`22*xzX*ymN3XJvXiy(X@e4I_yAKZR51PN4wFC`fdoaGPudkQE^L35`NbXmdwvkQauMZfH4bEaaVV>?(T@PqpUIX7)0c(("
            "(m55?b`~0Fh%j(KxBzEEH?z)N$}5N6M~!=n~q{xA7(lUu1(bd%DAz3o>_RFvs#Iq-"
            "a3c({EPQ?nbzitiE>GkI09)Gj@md85}DQvinWfi?)j8=?I{7XMve+O_ilRk2nUjx~zqO5BnvsjFD-"
            "Z$ZE4<piP`PW^)g8Q%rr;7?`z&H0-"
            "hrRv|cUc^PCm<y)k~yN6%NGIC8)8haU<8O57v;x^cyo=M<MTQ0X7h30ARpiDOpoit&L>tZOze&Zf{HM?4Cq#l*G=ecegT}CG{4cE"
            "90tv>B2OuMB!K+4uVmvY@<U&j(w<gWE8c7V@UXeGr46=1+2j8IINVv#{Drl9ZDWCL$~_cyKPIJvY;_Nx&@cW>4WHRc_Ce5@i8=1A"
            "RJ%k0Z08iw@2l_6wWR4qRQO}2`jVXX~Zs;F&Ag2+jB?aOSAHG+s+4OR^$TX4&u#berKR`v?JIqJ<bTPCq0aF~(;W_qs5iIPl{p*#"
            "$y%R*h*0v~A1ejzIc&J1J~Vdxs@IQ`n@r%Zo+ifa7{bg3*mc6lIM^_;aJ-"
            "x#cc#Pgosy=Wxuqm+e)H(RktO<NVQ1)+5+?k*!|{L8YJ6dbrcB*O>fyag&ZFxM5;mCNTe!*{JWywbW3>_X|K>~E>VMoxr0*Tm_UR"
            "TEcm>IUwrcnv_Y$a;Gkt)V^{2eYBxVrN@#8B;~lW6_H5nqR!<QZlvX1I8t)k*zqCQ?-"
            "7TMl_Ul`Px#^OOE$(w1i#_%W#PiAgtY`y*L{S<k~#4Y!&XM-dYk3_Y}3z)mkXos3&f4;;ayaC%~@jJsKo9>H%RDJF=J2dtlcdvJ8"
            "n^RoF~O$e)L{^k(+DsSVWo;$c~X`OfVtrdsNe7N|}aHGav<HW9BwtaMSy#(uuJS=OHng*0g0m~5IWP>q{~Ys&3%*p^Uu%Wzr%x(4"
            "N+5)`5=f#wdAXP>LbQSkCA{hwS{N&q;rd5er$8Mu0^rq^di4-Ox=)TLvKjAgHfLh&iqC}A2=$YJOR!%lA-"
            "U&}<9docwxfcMHQob`Jz?rsF^d`hYXH;FrSAB8Lxs=zKY-yr&`$KGw`(QOeq-"
            "T7s}6vP?rbZ|Qydoeq7pzSp4a0RS`vxUuD@Rz{9AVQ0y-"
            "WrBj02dh#1)SF^T!R9JIxl{CwwbM7rwoD+TCyf^YvD^=JCKQ&oK3XGLiwp%hR1<`X|vXNhG2^mTdKYb8>#ySZ^VizaKqhY<WQRSr"
            "lKTRDp}2enr9EZopUTb7b3Xscv`*jbcz_@YR!av=jzA;#cp|e)EYH`3%=TMiwNO9sF#vbyy`87YUFB7E7T}G!$y+r<|rZUbfaUfr"
            "Nau5CXY6@rRZYncd%n1g5k;g?eT75pch1}`AT3$d{$fx%S8^&Ug(xqYf<dl23^(sCR?bj?l>gW-"
            "9<Jt3N3xhuiSz?zJ#MWzp0}G$w)6}1y=L%Z=J30&{NZ$>=j<<%ehL`AryhEE3dAp$Q-"
            "|`zm|##`n95qyo+pYjeu4X`f~~*qo6@sXZ5l}*R$;&KB8DoAI6BR`=*UqegBo&v@&bc(|Hf!q`8_W7H=AUHD@)3&9mC~>p29iRl*"
            "&hb%C?-v|(<cVHy&3<fh+!)N?Oo$|HvqgJB7ab@fK~q;tJn*ibwXs@^Lj&Ryw@`Q5_cQBlJS5tX8ojlMY7qhX&==-BRq-jr)0t%I"
            "~>mqgi*69}ux3S&5`ad~<J2m`(x@^8$*qDt+jFB->a4v-"
            "%!U&r&9D{5q>^+kgty%k9|Z^*iG>~z>BG2QO7+awDiXPsB+7c`G`cj_s-dKX;5{YNmW-"
            "|e$dD7Kqt(<s<3widDMcl#(L`*lQ@JpN{G6pCZLM!OlG1Ofvkc{sc4oe8BD1^HZWudVFnSWHm0STXY6lL#Unds3zeysMv3T<izsk"
            "6ZGWV`R(dFjw-"
            "{n(z$RykaR{U661)uuOI@e16;HH0bM9by|{js?wt!j?zQQLh@r3>mw3Yt!<pb*Y!1#1HWBc%+tQ0+vde9*pOSHxTU~uMSB9NrMsh"
            "1@{Fzt_tEHl9MOX9tbgX^P3wXKr=(d0<I_VGW<(FD;k<yS&mdFEcmWqhNzLnKtCbCf>$q(9$A#=`PrPd(-"
            "B>D$2^(q?y)0sw=)ET&TVKl6b1%6F2m}S)=>kHa4(BPGxYJG4GcRs4Lsnh9EyQ$<91PcVr&fW*mJ5blQ9U&+GsNt541Dxx`@~A&u"
            "hm8!?oNJbAH)&SbuxJp82A>j7wh^28-"
            "n<?Wwhyz0Ll@OGqD~^r$VryUYHMOLwbYsKwZfTVWwfumqe3|`ZDS69{Ze#nD*!8N`db7gEd(~q9$O9keBf<z{Nn<P%r0g)?!7M%Q"
            "QDFw@HsplQh13v;)S^GC$39po`dLnU8lJ;1lM}7JN1NAJ5~EIL9E|ABP(F`B{>JCRKq{SAEx2PG#LIk%5W3VqL|0vQm<MHK_H~rO"
            "iC7^B3&<$~Z<EDbj#^pP8Lj^ZqoI87XG4LBps_?=Q6yeYT_9GW_>Q9_Y)r%a|!8>c5J0Ni^vt++|mi6?CysYv<QL9X2u$?($V@g}"
            "En;z=)--Cb<OZsu7AjEa66w7&~0L$_je7+7<Lne#6y(s_qB(NO9A~tqOUH-ii*x)*Dq$OVR>aYVFRJ$EeHG^d3z|gEVoso;-"
            "_Iw$duyNClD~b`#n{T(}pC+=rZ5Y3~S(FzseA2670j0LAsHy0w3NG(=3k3UO|yElf7oTMX*~1Tqet{+5gqrg&@7h{U!1&_xg*+Py"
            "36IA*ZUMy}jB!FIpu))KhK#+Rd5j~L7MMd%+~?KRflt)#|mJzTxLq}&>m@S@>z-"
            "z|8oyZJ8m20fel^vuEJVskBXXw8=QN);2@T&fi|xibi$^w0oU0fFu!E}U;o)sStvssbGl5c+a9!OXJ0=29GK-"
            "f^cc0Ti4iV=EJdzlH{Rb;&){A2hYH&OmN6LNx_0ym!-avl&-"
            "B0b`?Z1nx@pYbSPJv*WdQ*E2Esw)SMdATYB5r|E=kX^Iufdb^a@G?|^Y0+;9&o6+d4m<}g;<k)C&Lvri5E?E7o5JTc<a7D`pDJ1t"
            "@wp#;C-"
            "8w57);ArJLH$n9EZ_Iub#RVC`)11+O1Esc_2#=}lU;OYbD)uxyy8#EybL5l?Ij683REpPN&;OQ8LVM9eTmh`(@MpkIvYoZZctB6f"
            "SWN){lso#dfh_)faxgPPW~OAF5^Ry59h9VmXKNudu93g=J$_lkFQ%f^5)ZF*ov{1R&zsu6gmsmWY4+;LPO9QZ4)<wL+6ET5PLGAE"
            "Vp+@+7nXl5r=p20{{d0YI^hO)vGJFh^~p>=+)iV!fb|>Ry<xm0K4g3cVmtp_Zfq}@2Zy%8b`+}yFnQ_0_VdU%r|qz%Ux*D%#(WN1"
            "9P3=h3a6%<#FaMIkXzfY<HFOiiO$Q;D%9eWGCnb{bp~&?0hD+g|%I`O;CqCQ3Y#T<UT*L1Cb6PPjVGDVbBnQocb$>R;?yjeO7eCu"
            "xN3O$B8;3i``xJd2gEAWbcEL(0KFGqxv4TWqDuL<{qU5$k)*ULlp!7so_>>HBCA>D=#y}RtRVeqpJwHk?d`H>bONuEY&LH=%PDIb"
            "qdX4JwnVMN5WPhG_1eWa+v+XnyY)YrF*N<vHC1emtuIGfDY>lDVfVw9rwbmz78G7P<mr>u&cpIomNQf3TK<7BvPrzio7Q;TWKL-"
            "36$j~4lvN6V3T0%&W;YR>VO!DUNGyePJ#T8_h#I!_r+7%SWN`>Z7tlOxTlAb_t5P)6K|b+V{Ss?0Z#Dtb|*t1datR78M`_bOzzv~"
            "P$4mpji??rfDO+XoeFi|ua0fomBICaUHK4gU2!Gf+$0bX`@YAGYVC<+z}_KZ8u~4Hf0#uJ?Tr;ohJA}gql!-"
            "5x0{WJ>o?YGh$V$=6`R(=?)@xqEU4D>s{1-"
            "~>}VuG%cL2Zceh8e{E_!5jv4e>v3&A=XH0A?Exf1gDDS3vbaxuDaF%rh(<RUghi7znLmzAOd+V1Xq=Sn7?Ss*2aPLHdo|cMLYnbk"
            "I@8)oT{1lX&TrO4xcx{bZP>!&xOhTg0dq;V&5jI5qFFmtRHHPhnG|K5s)U1x?x$mHZJdPW0c4tTU{zd7Eos5tZ1p%LG662%no*>R"
            "bmlas9sA7Mwp{TK^p<h8uS|Mg@XV%^+hM2(H84a&AU!T~-eUc?mAh)oZ1k-EkQYcBWP8$Vn2+>!1T?n^P<5FE#tNN9y+dOq<d-"
            "HzTRmGjUv@~!~#_ZK~sA>V$FR^$oN@QNIt;3xhcL(HO)tWr-"
            "I(qv7fb6J84`x}oxBe=P&ATvSvxRCgD35O$gDA{vd9?);^MlasVD)#mN}jFbU>$oQMmHhY)mJFn9Gk{^tY^iGmv-HPD#%uQrDl;?"
            "{GLz7XwGnwqO4{Ud4*2QxEAS9SBlB&vzO3@!*oyH`o3-"
            "*m!9i&HAm~6Wsne|%?p&ZuDgl^RYz~eB23%sFzc|bo_w!i+RG9wY_7W)ZQWe5EyT04HCNo)1=Rfv8m6>mZqe?5<=7G%*W$g?XcY2"
            "+@cKhUV6@ql8K2SCBMA)md(Sx+7a`osB$2?r$$|n<@5`W5WP6V$gq|<+EOy;&oSW+KFDKTdK(`WaWYyV}*6_w4r}kXpYBguxlF@T"
            "A_Cj!^&u_xsc7i^@9cKQXK_WF|FB3-"
            "3;=S2tHNlfqRvW|=#5a&t@wh!UxAT;S+ZSq}_<dF8Ab^2J99uLUxtW`cI^OYpI^bfxTe;34#|fu%du7-"
            "(b&C*UutrX3NVq|w?com$J7)ZwqTQZ?i9q68b&MRTUBC+3`diN}M7iG3)39F2PXeEvr+uk1uK4FJ8w>o@Rj1&M;I>>>h;cudz_72"
            "eBMY@1d;hV4i^y2%({2x9zW0WxV&Zwf29ko`lcJ$!dOM;F$*(J6IRMGEa=gb^eK(Tn={;+|MjP^CLLx8L?PtxBXN|X@$bcM+cg}I"
            "o-Y2<uySP>R*)9o&5wE&yWy3pxIEGOHR-v+$O~9x=-"
            "jnUb6tj?Km}J2e$$c!Bc);Ww=G+{Q_s%|0rThkWaIijqnryQ7y;CgT^k&~M*}I$!ZcQ_cx><<kE~tUx$(B1M&?ULXox}yq?qn@0$"
            "G65amNWfW7Yuvbc^LL~kR166uteB`-ulG)W$x*@^JhP;yR|relU-3NvzJ&w-=aR<91S7+plcX32p=7*?1^2-"
            "rrkps_`Xdd8=)q&3Rxj%LF0g5)xDGTf;P%zCq3ddn`0u)GIjfNZsbt^nCEUdOAXbNFfiNFw+M_};vKkJyE{Fiigi|mpZGqlmYlVX"
            "7O~Xy8l2XHG=W%)zr`=13~V=SIUG&!e>O=X?71z`zFC{!7X~wE>ccTE{cb@wV-2R{_b+Y4ZI>}Yh-lmD3SHRp^jwrikMmrEwmTBQ"
            "e<1^QGaeY~zN1TM;%z#B_-"
            "*uQb3vrdcLo{)WOHbc0B0fuXswR&X<25sx(%{9s4JByON7I6%SLW!r@MIsP#qq7Z>`|jT1C)%8Es?p56l)zxI_2QlL@Jb5L+s%8L"
            "dgFZa&b_%M&Nu<^v@$yS@Mi>G)lcH_s?x8a5H)x529I7by70Q1#gZ7O}_9fJFCAFS}QS!i#k&vfaLWnxSdSSY51dMi>f(Z8OGEho"
            "p?yJx7OarTuS%a+~B9tc~*oPv#EJ+PXTJB`4O87WaGK;uidM3(XjVw$=YyFUHM>FM9>QQ5d@pLEB)BsYP6Szk0-"
            "Gx46AOrC4UZa0#`Rd#|c748)6WP%4`sLn|euJ=&gpb4<&%(~#D@U^|qNOO8RJG+;Z`u)teu7qk!bln75V!-"
            "DLU^L=j?1b=ArZQ#0ZB@0%wwFgmR?lE$Bw<sK3Xsa0Zdwmu^Agsk;M@#qi9<}SCIxxE+bPY~8u1s5ID-"
            "70_;)6yz8`GALqdjOE*!6w)8Ih{CeHJ$UdsXX}>F}1KLG@v~8dzQ5Hs-pYQUjrpcgC>X`<*b2kaCiPHjx?GP1^(4gm8^Vk@L3iaf"
            "RGwz*K>zX+zqP5Nm2|*ipEDvp^jtMYj@^hPS^-;e1BY=3cTlL-"
            "?(l6My4uB_{9XFFk0MLnj3U=9BwY#_m9w6SDQ4sQX#f=h$ZzHJp1`HHG+XG}B1Z*ee_G!gIS8gE84JRP74xCs@waN!$WFIdZaKfj"
            "0LBwZ!jtp>On7%WDDLEzQpA+qVvYOP9#2U67U<DDxbRY%`(VSN5fvU<26Aad$(V)_7<?rJeqH+;YI}<vx22vdd`MvO5?@$S()F!j"
            "fp0Zh6`RJH=aA-"
            "A4gr={NWz(@w6tM#q|RZ<>_|r}j<26>t3Fw*(n;tLohjc54%lLGE733|(najQ_&Ztn3vo@hvC}BhTk&DTEpla82K>>=m@ZP3!6nn"
            "n8qZZbl-"
            "74E{7?2g@rQ?=y50g(kOzVWS7ZX0nk;y${=pPM?{9*RuP%HRQ7cz?<)7)t_&C1;gHRrBaLGs2ZZrV8^y7Fq+N>X!IT&^ctmfporh"
            "M>R<#k{g$=4O;Yx=8~RMaNJOVI#d?DhQUfvCEgWZ7B6wiv7`_>+Yuk<3c5T1MAZ>oT@0#EEx**v)0X{H9#A}OTS-HV>7lh=<-"
            "dXI5!GoX<;PS{@=mF_Kdwb{_xQEoRiQgG2*;23{zY$M2<1vE9DWveWi#DuiXt18c&Py|j_k@~i)B^M{YzeV%7}>qv9mmW*Y=mPrd"
            "!K`X6PjmjegU+huGcqhhmzzK0sv;@_Cos>Zk#(LmMYC2TJBL~$x=4%Q6_6+wPo|*O0=zHhP3{BD8blWzglZM2CpcLis<fJZH+{+-"
            "OoZsnlVO@1=^;zha(^rE3<js{~3Gxp!2}S0!dI(=vNCdhZ4di)&XZTOXvl;?Zy_b+`;rNO~*|M9T%W|YAw@GM;kru6x!q<6MtJCB"
            "gNl0ZDo%!nm4&`yCr8g^<s#`#~yqHQ<2$2XNbccIV$^1Ah857Gi3U02uLBl76{l}!;TB%>q~*(+4iRCgKlO4f*fz*EEw8tN1#-"
            "qhbs_Lano}J%$Di%x^39n`vHP@<v#4P-^bC3Y?~Sz_I8&q&?k9}8~U&8TriG7c_ECad6Po#+jRZoKmO-"
            "$|M{Q4{_xvh{^f6f{+ECLyMO(+fBN%Z|NZa&`j@}xUGgt~{qtYzpD)YP`14<W`};rq@8ACRPyhVafB)N`{`p_^UzQr2E_hIaJi=D"
            ">bN=nGfBUz8`s;uE6~F#lN`Lro|MDMyKxX|PQv3Uy{_ubN_y6)6zU@zc`;&j(|NF0h#c%%K|MsVU{wtpR$DjVk|FWO_??3+b=YRP"
            "h|LdRs=fD5)x4-{~KmPWA{n!8Vzy6m${`QZ*{`lKJ{m;Mr_5bmo>ioCAyVBoX<-cFo-"
            "`oF3zq~*G_G^1q$+JeDb@FU}f0i!tERkoKJR|Ry$T-s=&xAZ9@0ZE@Wirl8)*+L1$mIPpdA~x&S-"
            "!4A*2m;oCeI3a*2%L$o(Xw2zd!qq!A!8E&nkJ=$TRYOiM(GXbDhatXYvl2yhCPV`8wByV4@IA6oQFDFi{953c*Anm?#7jg<zr(Oc"
            "a8NLNHMXCJM1Hh2W(SycB|$Lhw=uUJAuaChJhh`+c`dm0+ooGp>>|t`ZDZg276#RLL1v$r)D(1}ni!CFfWr=U62;s|072;H8pttd"
            "euA5}Z|nvqtPvBX+3~EH#3qMzGX~U24QGHG-"
            "u^u+#{a8nH`_*ri5r)(FlT!C51AsS&%>2+kV8StB@W#P&2|dpf~cCphZ_FP+$)PHaynIO_ywo#3St+tbPU*9p!#!C5DG>BKH|VwX"
            "C>StmH_1TUS~rB3WpC+A-"
            "$7#zed4T7aXurvsk2C+SZ*q%YKGzgXk!O|eMXAs*n2$lxH(jZtG#P$qgdj`SMAXpj%OM}=YA$CcKT@r#bA?KeEED5noLhO<d3<|-"
            "Z5G)C?OG50D5DW^zpb#txu}ebik`N3E!JrT<39(B;?2-"
            "@+PJ+QnY}O<;YZ9DIg0o5RGKtNa#AZ!`vq^9^30@|#S(DhTNpLm^&L+XjBsOajn>7i}=69T>>$_dTvqGL#@~n|(ojfDs{A!o*ERk"
            "pA{StY<M8^5mF5#JwXXO1ddB04?naMh2vJRQNUncKY$T+{+rIfzgB|OXISs~9lc{a#1A<yRbXWubMV&ZsK$+JeDk@rjF{W2M6ChL"
            "&NI%M(=nT>_abtZHD9cLsaj%SrTYvfrc&jxuW-"
            "=8Hi&hK`K#KaSci6;^h$1^t0`TbcY>yXL&W%7QR%ylMnoyj|V!BRJVER$!2JgekcBhQ39BjfyHUpnFHI^pX485!pn`_c(l*9lkG3"
            "0G&&G8tzk>yXJhWb%HQydT3$A?r}SuESTm)CpJT&l-"
            "8w$+JP833)cZKTBkssgP%tJR|Ry$ou7sJR@_R$vb574w;RG%ylMn{T*k8;H41TQwUxP!Al`{DFiQt*q%c0QV3oO!Al`{Da7^^VtW"
            "d~QXyC>1WSe3o<gux2$l-LQXyEX1WT3Jo=VQYN-"
            "$UnUMjIKm7H;v;H(myRf3mF>`NtQTqQWG1ZS1tr4rjy$r)D(&MLuKrFi*%#x;VaMzGWfmKw23jo76|aMlRU8o^m3cBv7&)CkTR!C"
            "50XYs4-!VwW1hStB@W1ZR!do<^|LD3-"
            "qaOP%1Q6Wi0t`PT`~I>A{dc<ID0bz+w~!C5Cb>jW>I*riVFQYSd;1ZSP#r4zf<iCyaC{ObgRgV?1(?9w1O8w5*(*q%Xb&mdSD1WS"
            "WpX%O2pi0v5!OM_r(5G)O1dj_#RgJ5Y8EDeICLG02Xc4-j1Bm`$da3%yxLhO<dyCejILNF)<OG50D5W6G<gF-MU1WQ8fk`TKj1cO"
            "2_C<IGF?2-"
            "_>Bm{#(FgS@_n&kYO1ZR`rY!bXoVzVZ(S(D&w5}Zwfmq~2aBsOajoK1qWN$@g>&6>n!O@gyYa5lf=1^ied&oX&d$g@hG4f0ILv-"
            "$m5BJY>T`z12YuXgD>kJky0*9nhj&oUWjChL&NI%M*GnY>?S<0R|AaQ5|V{LbU~vr3*d@~o3*gFKV(&k`BuSDW>n$Loa0>x9SiXK"
            "bAF`?Ie$>pPFv36JN`$T%}uhfLPtJI+YlJ7r>*GQkpwmnRZ0PbBW0GO<gUV2Q-b6N#5668BD-"
            "*riOclnItdEIpB!c*?{sk$8C`@p64u$XLGEo<VrLL3q4DYUljfAkTz6o8O;(!O|eLbN;N7XN^1~@0ZB?Wirl8)*+L1$mIPpdB4oY"
            "N!B5gbtvThzT+$tEPb_0gYbBR@OXppc!Si=`7`nkDU)a9{StY<B;?ur{w$Ms$mAU|8A~Q}oylBhGS}a6RtP2v!9*dLC<GIQV4@IA"
            "6oQFDFi{953c*An_N5TK6oQvR@KOj~3c*Vucqs%gh2W(S8}gNp8-"
            "$M=gpV79j~k>uZV;|+5Uy?zu5J*n&YzKSem}=5Imar&StU5D1TU4GW0jm^mEf!roK<39Dmlk0Ima5oStB@W#4a^rmm0xYBRFdWXN"
            "}mUM(k1}IBNuFjo_>iyVQtXY6NGE;H(jxHDY@j!BQhwY6MH2;H4A0)QMf{1ZSP#tP{L+VwXCxOP%1X6P$H|mrm?bCw8e5oOOb;PV"
            "mx+UFyUxb#nf7g27I(G>Baq1ZRWbY!KTsi0v5!OM_r(5G)O1dj_#RgJ5Y8EDeICL2S<;wr3D54T7aXur!EW8pJLQily&<mJpl?u}"
            "ebik`TKj1cO2_C<IGF?2-_>Bm{#(Fen5|LhO<dyCejILNF)<OG50D5W6G<gF-MkiCvn+E=_{7NpLm^UM8_wlh~|Da5f3fCc(=jHf"
            "s`_H3`lp!Pz8unZ#yIVzVZ}*(5ld6fa+G)_1OM5Uy?zu5OUpxk0$PLAbg>xVk~Ox<P8^2I1-k;pztA>IUKJ2C1DJgsU5bs~d!?8-"
            "%MHq;_r)u5J*nZV;|+5Uy^J+POivx<R<QLAbg>xVk}V=LX^G2I1-k;pztA>ISKu8-"
            "%MHgsU5bs~d!?8>Fso5Uy?zu5OT;yFqH_2I1-k;pztA>IUKJ2C1DJgsU5bs~d!?8-"
            "%MHq;_r)u5J*nZV;|+5Uy^J+POivx<R<QLAbg>YUc*w>IUKJgm862YUhM-bwap0AzYmhu1-kpoDi-"
            "~2v;YBs}sW23E|^}@Nq);I3awTkoq_we4G$IP6!_-gpU(aA18#56T-&{;p2qxaYE|jgz#}f_&6baoDe=vh+d@-"
            "K28W9Cxnj^QXeOTj}yYj3E|^}@Nq)4EQRoKLiji#e4G$IPKcJJ5I#-_A18#56T-"
            "&{(Xtf6#|h!%gz#}f_&6b2mO}VAA$*(=K28W9Cq&Cq2v;YBs}sW23E}F5Xjuy3>V$B0Lby61dX+->I3awT5I#-"
            "_A16f1QV1U>gpU)##|h!%glJg`;p2qxaYFbwA$*(=ElVMMoDe=v2p=bej}xM0DTI#`!p8~W<Am^WLbNP}@Nq);I3awT5I#<bUZoH"
            "|P6!_-gpU)VWhsP@6T-"
            "&{;p2qxaYD2#h467g_&6baoDe=vh?b=gK28W9Cxnj^!p8~GvJ}F{3E|^}@Nq);I3Ze=Liji#e4G$IPKcJJ5I#-_A18#56T-"
            "&{(Xtf6#|h!%gz#}f_&6b2mO}VAA$*(=K28W9Cq&Cq2p=bej}yYj3E|^}Xjuy3<Am^WLiji#e4G$1OCfxm5I#-"
            "_A18#56QX4)gpU)##|h!%gz#}fv@C`2aYFbwA$*(=K2C_1r4TJkAzYmhu1*M7Cxnj^!p8~W<Am^WLbNP}@Nq);I3awT5I#<bmZcC"
            "rP6!_-"
            "gpU)##|hE06vD>|;p2qxaYFbwAzGF~_&6baoDe=v2p=az%Tfp*Cxnj^!p8~W<Ai8g3gP2~@Nq);I3awT5G_j~e4G$IP6!_-"
            "gpU)VWhsP@6T-"
            "&{;p2qxaYD2#h467g_&6baoDe=vh?b=gK28W9Cxnj^!pHeDGR{oaA(M5;<Q=}_B@??uV&aL!#1n~$f7h}U!p8~W<Am^WLiji#T9!"
            "iiI3awT5I#-_A16f1QV1U>gpU)##|hE06vD>|;p2qxag%6SPQu4c!pBX*$4$b=O`>Hv2_H8JA2$gfHwhm%iC*O-"
            "eB2~_+$4P5Bzl#T@Ntvyag*?Ilkjnq=v7X_$4$b=O~S`b!pBXbS2+nEHwhm%2_H8JA2*36<s^LEBz)W?eB2~@m6Pyslkjnq@Ntvy"
            "ag%6LPQu4c!pBX*$4$b=O`=IT2_H8JA2$gfHwhm%i6-SFeB2~_+$4P5Bz)W?nv|39ag*?Ilkjnq@Ntu9Qcl9hO~S`b!pBY0KW-Av"
            "Z4%CH63%TB&TSG+%1QXRN%*))__#^<xJfiAC*k8J;o~OZ<0j$bCefsvgpZqqkDG*#n}m;>M3Zt7K5h~|ZW2Cj5<YGcP0C65xJmf9"
            "N%*))__#?lDJS9MCgI~I;o~OZ<0jFhoTRthBwXDjT-_vG-6VY6Bz)W?eB2~_+$5Tmlkjnq@Ntvyag*?IlW0;-"
            "!pBX*$4$b=O~S`bqDeUkA2$gfHwhm%2_H9!Cgmi2+$4P5Bz)W?eB2~_+$4P5Bz)W?eB30Ol#}prlkjnq@Ntvyag%6LPQu4c!pBX*"
            "$4$b=O`=IT2_H8JA2$gfHwhm%i6-"
            "SFeB2~_+$4P5Bz)W?nv|39ag*?Ilkjnq@Ntu9Qcl9hO~S`b!pBX*$4#P1ISC&(2_H8JA2$gfH;E?YBz)W?eB2~_+$4P5B)#P(;o~"
            "OZ<0j$bCefsvgpZqqkDG*#n}m;>M3Zt7K5h~|ZW2Cj5<YGcP0C65xJmf9N%*))__#?lDJS9MCgI~I;o~OZ<0jFhoP>{?gpZqqkDG"
            "*#n?#dx5<YGcK5h~|ZW2Cj5>3iU__#^<xJmf9N%*))G$|+H<0j$bCgI~I;o~OJq@0A0n}m;>gpZqqkDEl3auPmn5<YGcK5h~|ZW2"
            "w(N%*))__#@G;3m<eoP>{?gpZqqkDG*#n?#dx5<YGcK5h~|ZW2Cj5>3iU__#^<xJmf9N%*))G$|+H<0j$bCgI~I(WIP&kDG*#n}m"
            ";>-"
            "}tz!|M5+)@_AOtvqqkE@@#y6_Ir4Fo{@L>J=^m<Bkz~U`~9AMd7jPh&oX(3Ox_`rv1BsWnap)2bDhcieaFl1+ok7OCeI3aR>`wLo"
            "(Xw2zd!qqvrKT730{8RE<MjWc}CtZk@w4Ft}~hIOx_`rcgSokU+20IycB|$Lhw?EeJKPlh2W(SycB|$LhMT+cqs%gh2W(SycA-"
            "43b8$fV5tx+6@sNgY)>IrDilkFyx(`5RSA|Vu|1WXag|`O5)4*?rAlm1C1+eE7_0<?m0+n7+f&K;R|y6y!Am8!r;;<S5}Z|nvqtP"
            "vBX+3~EH#3qMzGX~U24QGHG-"
            "u^u+#{a8nH`_*ri6W)CiUu!BQi3sS&%>2+kV8StB@W#P&2|dpf~cCphZ_FP+$)PHaynIO_ywo#3St+tZ2d=>%t;;H(q8bYgotIsZ"
            "DrStmH_1TUS~rB3WpC+A-"
            "$7#zed4T7aXurvsk2C+SZ*q%YKGzgXk!O|eMXAs*n2$lxH(jZtG#P$qgdj`SMAXpj%OM}=YA$CcKT@r#bA?KeEED5noLhO<d3<|-"
            "Z5G)C?OG50D5DW^zpb#txu}ebik`N3E!JrT<39(B;?2-"
            "@+PJ+QnY}O<;YZ9DIg0o5RGKtNa#AZ!`vq^9^30@|#S(DhTNpLm^&L+XjBsOajn>7i}=69TZsgHY}74oc-XN^4T<QW;~SG)9u$9t"
            "ZU_e<pc5*g=LyYz*}d!CW^%jEqs8D}QzkjXk^@_w1TUm@fCYL~w7c+ay;o)z+}lV^iG6Y^|+fA$@NBqp9nO#BOv_dFx-"
            "m&p5NGR{oaA(M5;<Q*~_3z_Ro=K4F%NW46exc3*m%I8@p&jxuW-"
            "=8Hi&hK_96D*N<c_Q)hMB?6G^eXRXnXE%5@0ZE@Wir>9%ylO3@C8fhn_lJftdM7wJZt2ckY{9^U+haFTwNkuUHYa;`8*rs8F`09-"
            "XW8*WHQ&8%ylMnoyq$#OcXNL<?CF3wLOV&b?IAu-1DrHXM;Qw@@#&8mdH3$A<rs#M&2)x_sbV~M&>$`cgW-&G8+q->-"
            "=@D3&BJom?#7jg<zr(Oca8NLNHMXCJMnsA($ux6NO-"
            "+5c^UHUJAiWA$TbSFNNTx5WEzMmrUO8yIraTOO>2sm7HUhV6YMlR)VEU&aq0)u}Uyl30^8W$0|9;D#2MLII9FNm7HUhoMV;XtP-"
            "46f|o|@QX^Pu1WS!zsS&%>h+S#~OO0Tu5iB)gmm0B4jo_>ioHc^8M(k1}cBv7ZHG;E7aMp<JX%tJ}eWFfq)(KuZu|1vGo=$Mq3C="
            "pfODDFclk=|=oOOb;PVmx+UFyUxb%L`_aMlT4I<ZTg*riU+zfLeXh+P`QE)9aEL9jH4?HR=O41%RWurvsk2C+SZ*q%YKGzgXk!O|"
            "eMXAs*n2$lxH(jZtG#4ZhDmxS0QAvhC)Ga*<KVwZ&2B_S9Tf<Ylz5@MHx*d-wt6oNq^SQ28FgxDn^7!-"
            "m*Ay^V(mxS0QAs7^b!AZ`)No>|6IGY4#li+2N^_#?oOkzVOS@%iSeUkN?#D+{_Lnc}G`F-8L)T=$uGI>_Wvr3*d@{Emhet-6L-"
            "M{l=iST3TTfN%ztdnPhJR|Ry$opk7&P>)JlXb}C{W5vKOvYKrI()Sg-}$jb__6e@UhR3-"
            "$+JP833)cZKTBksU(c%V{8%FVSR(vb`c|)YKl^I4zVl=0TP@r3jEp6dxz1#+GkJ&aILpK?Wnz~|j5?7Rbs{n9l!;x+#4eE-"
            "bs{n9L}Jt_6T6g&UCIPYBz~Pp{5oY~mq?5{kr;I%G3xL2YMJn3nebzo@MD?Oq-Da7Wx|hT!jEObk7ZJmmcQvoKF>ONM&2)x_se9i"
            "Gnwm5-XW8B$ZRZRt_zv#?^ycEk7ZJmmI*(W2|tzzKb8qUmPt)oCj3|?{8%RZSSI{f{-"
            "z&!Kg$<+M&2QlcgSQcnap)2bNwA>g<zr(Oca8NLNHMXCJMnsA($ux6NO-+5KI(;i9&2hA$TbSFNNTx5WEzEmqPGT2wn=oOC@-"
            "z<gBXXtf~ZOmEf!ryi{^lRdQBUg0o6+Rta7zIjbr;t17`+B{-"
            "`DFO{5Qm7HUh;H(myRdSA1a*j2Evqo^%2+kU@OO4p2MsU^$&Kkj4BX+3~yVMBI8o^m3IBUc%HDZ?<!C50XYXoPF*q%nP)F_s|`$V"
            "1Kr4zf<iCyXhXPw}z6TEa{mpZXao#3n!oOObiPV7=AcBvDbb%L`_@Y0E0>clQ}a{hIK!9ncOAa-dGoDG7dL2S<;wr3D54T7aXur!"
            "G68N~Jsf~7&QGzgXku|0#>o<XoQ2$lxH(jaze5W6&pT@r#bAvhC)B_Vc6h+PtbK_M6vf+Zn#Nr+t%f<Yk|6oMroc1eg`5`sY?7!-"
            "mfA$CcKT@r#pAsC#*E=_X&O@gyYa5f2ECb3zQ*sMu#HVMuq!OJ8zYZ99^3C<?L*(7+G#AZ!mvnIjWBsiPj@$#LI%Y={1gpbRFkIU"
            "cmBcEqNp3U#i5_!Kw-"
            "Y=1Hezi;A`M6B@xJ>xC{7padc}CVDlXb}C{W5vK%*IL9f#K}y+4!B0%ir`PpJ$Cc>*U!W&*b~FM8^5mW_{=5GU4Mg;p6f*{mAFp{"
            "Qm5#&HB#AWm2z}zx8u_o{@FPWF5ZaEEBtwiCrQw@kC<ciNwTHCUz+kyF_B*iNwSciHWC7>{2FnDHAM_czGi6@|1~P%ET^_n0O*F@"
            "h|$3&$Hj3>q7XrLio5sG$IS(;|k&93gP1l;o}O?h%AJUD};|LgpVtPk1IqYvJgJ55I(LDKCTdr$U^wILio5s__#v&xI#1{3*qAm;"
            "o}P7;|k&93ekuxgpVtPk1K?aD};|LL<h1EKCTcxt`I)15RJ$}__#v&xI*~2Lio5sbRY}i;|k&93gP1l;o}O?fh>fND};|LgpVtPk"
            "1Iq6vJgJ55I(LDKCTcxt`Hr_Lio5s__#v&xI*~2LUbSt;pz(E>I&iN3gPMs(Sa<4t1E=7D}<{ngsUq=2eJ^Zt`M%S5U#F#>*w}7B"
            "Xga}TxaqQnY=?LbDhate?P|>!BQhwYQ!!zVwW1hQX^Pu1WS$BrAF*hBUow#OO0Tu5xdlgU1|hpjo_>ioHb&58o^Q{SZV}Io#3St+"
            "tZ2d=>%t;;H(q8bYgotu|1vOtP`Adf|pKgPbcSJCphZ_XPw}s6T8%jUFzig>jZ<HU}+G$GziWH!Py|TXAs*n2$lxH(jZtG#P$qgd"
            "j`SMAXpj%OM}>+L2S<;SQ-RNgJ5Y8yEKSh8Wc<4eJLS06JnQy*d-"
            "x$NeBjoU{DB_gxDn^c1Z{Zg<wz!mW0?PA$Ca!28Ccy2$qD{B_Vc62nK~<a1y&TiCvlmXOrM;61+@evnH`wli+L;oK1q4No>|6Hfs"
            "``O@gyY@G^<bn#5*Jg0o3*HYr}d+N|$<Tp@g1A$(jRI*^6%afR@4h468O@NtFcKo-Ks6~f0A!p9ZD#}%RjSqL9j2p?AnA6Ez;SBM"
            "T|A$(jRd|V-XTp@g1Av%zS@NtFkafR@4h468O=s*_2#}&fI6~f0A!p9Y&16c?kR|p?h2p?AnA6JM*WFdT9A$(jR^>KyhKo-Ks6~f"
            "0A!p9ZD#}%RjSqL9j2p?AnA6Ez;SBMT|A$(jRd|V-"
            "XTp@g1Av%zS@NtFkafR@4h3G&Q!pD{0<>UTQAO7V}fBVxPfBXA?_=i!;Pm}9P<B}==_)CA#exzPk?$TOYKj>}FQBtk`gZa~^epIe"
            "CRs5jkpKI=a()v$Beco&(W&1(<8D-9S<x=bY8%aN7j@~+c(DP3U-_qsy!O-8(fu-"
            "83)b)d+L%*s>mh~sC&M9NevOe?L|J~;KrR4`Qn<?jO*uSrzIrI;%+QmMxji082nEC=@f6?S;Ui+w8`jN8FpT27NO#O>;?H{3E+Ry"
            "j6!j}xuCGdZ@xqdLa{@@x{N%(=3pE33cmiANYz9#efsaZcXcO8NK#{8*C-!0E{<@FmKP5)X-"
            "!+)R)UH18$<rup3a~$>f3>iN&eC<p>CA^FN;0l1#&uHu0M&2i1e^YuGpw8R~A1*)oUUSHmhpq4&$GqU{!}91$N_eN)fAS7Z{j8sy"
            "wwJ^Y!ZgSw=bwy-MEwSR>upEiPr^7P>H0|cjH&OUf1=aZ=la3@tu}L#^j+o#97z&)rsNz~P;!rZM$(MCN>cyPkDt;?OZIYU=<Dk^"
            "(j0lLZ`pT|BweR4Kl%{`kRK)2eE-JW|Dr~NAvyizzHLOwHSQuE{!c6WY?K^#jsE(Ngm2A<jlsBD7`RSo3g24drop#{-"
            "7Q528WyGW<2S0BjP`DKQF|Iph}3kfiT)sfR@!yUM(sbNtq+T{pr$v6zpZK|`?jm4j<_RrVUMYW2@7wn1}WGdQ9GLL`$vkqS$aKxR"
            "X6Y0S3&<I?R!+lJQi+@V}|qx5kKSD_!#ooY^r_hZbLBQ2nMyO!BXkuN1KXqJ+t7yk&X{lYnQ^xsilhBT+UaS^(W`>gBCZ7qBGjCd"
            "0`W8<Cyt4O?3t03gQQ8-%q-n$Db65GpJt|M>=&ggp+ONE1uc-lVB-"
            "l(e1K9eU=*0IThJq6{b1{0YCbCKVyfw{@=Jebf%*9og&9?=p3goVRJ^?s(W0G{B9}nA#tcWvN%C{^ami%p5jrWo9_4x99Yf2?~0&"
            "LYG<NpwlM`$qCZK|(oQ|B(zV&*DeD__e}@f#-"
            ";4{cIE&M!Ct=#{fDIs=CpPT9f1m!O#u4a?<Y*)9C#8o^hUN8_RAh#eF<qwx)xV$NUF^@ZzKou2J+F^ZtEOg7zfphWKe-"
            "~3>|??X*TG8!hW_MmwQTUmE9)npudJ3KesWQS-"
            "4OJj)Ox;_I)ayCp!KU&e=mj9Rl@IdJ&_dtSb*Yh(RES$KSnf1_RCWtq{A(1cJUQ)1@-"
            "7iI*e9^zQg^cMc2J{Jtxxh6*`%{>E0~<R}%eT{Yi<gg}z`vCsS>Te&0R`83$}wI6Zxi-@y?Ue<!O?-"
            "NOiV33kJRJ@J$Gy{BtZgI8?}sUP~Dgj0=kIR3GDj%Cyh7k03I%Hi_rPbAuS`$>wwQ`Ffm0iJqCI`%~ECp;BIEy4lLgua!2R&C|}o"
            "n!y#93O9?4%+b+x`A4pb^V+kH;{hU6``kx#)8%O9msV3<x<tEEi50GFI(6d0jrBY<Fx4Kv<N-"
            "*H6%`yg0V)J*W+TMKS}ZMw*6Q4^URju*lqb^AFG==B~EYpFj%nVVNvvRj``7Tcfh=TfV$z3_LoR&vnohZ-"
            "2G|_qvh+z$)!I@@mge?^%$Y9PEB#W(!IF4PmY0o?Ar2SZgmol>D4V6`~^p5{8AY5xNWhN^|UL;3#w0h?wyYvSC2cHz1Q(O?Qo>*y"
            "MkkFODS<F%ZHUH$N8qya5zHqiU+su$q&nnJL0}<=7L1`p&c%Weox-"
            "6P5K$*1ZR@NQUc>~&t+WKqERmA4pRd*Yrv*mW8yB%ABRLbPr>$y-"
            "+6u)!Zz`IRl{6LunIEIvs10PL^C`;P_K?)i<7QXdKhmRCvbEf4m+XCd29&%$+*H>=$MbCnEkLxI+@3t;wN!_QYwe#&?!6KOQ-BG="
            "jr;eH+nbu|DQ1(VYD}hs6_Z<R-"
            ">)$<+L|)9~7f}3`5TS4bL>%8+Fg)A&~Siui@NFdYb$ET%RL6F_x|mW1y>ixHvkn5>E_$({p}w#_~Rg`1a_8VRfQi*Z+5Tt0}b)B<"
            "M~K_|dI$oT&Qshl7<qZwPSW!}J0P@r+FfB*mZLl#A1=4puyMlez<OFDE?&dN{9=o-"
            ";mp^+dz1O@F1IQ>XAdU@WJlPr4}aNKEs%bOMX#MCrY&e0*2j`1r1Pv?jqlJB~-"
            "qVA*};q_Bx#%L=Xdra2$6h@=J}M(*?{<#6jWE(~Ej87@qKU&bjH%tfx}zR2|mEZSeiQx3l)VJY*+;^ut%xfpXi;d4FbCo3ePDZua"
            "aaNo%FgmKpMGp?zggXgTOe-RHU{N((dcAU)i7V*2(oW}c)o{NV?!n#LOsmMC~l%%(>cqFB9`uIRpba)jIdZdR>)-7`w2i-"
            "d%=%}Bgj_QGVz5CU;FVblo5S@ycZ>Nb%ij9b8+rrWIDh{Ks2NPFXIJWfK91)IN<>4W=55GzGU_T7nhqt9~8u4>U`nY+{^8~3M5eI"
            "d6kG~opj!^%&IU$@G-"
            "5wlm@!*(p8uQ;Zr|~A2vYZCCrqdZH<8Y|(lLLpk<Bp%?^B$gHZH~{W$88V4bF78#uNk~`yw_nN5%G(=0GGVr&fzDA1A{wjIF1umF"
            "e1F2hj040u3^U3SLAaO>6aeB&L8I~Ol~}GsvKS%26h|)I<N?dgd5m#l*l<z^PCo)uhZb`X&%2lBV-"
            "Z4rqgyTTvqSmFnZ~{RAnq&c=w!-"
            "IJ6ts7LR~4*np!g0tpAlaU>~_jxpD99(L{YF0$Sx&O_mIglB*RN89nH?KDkUjb6NH;oIY#9!Kr5I*O?J@$K>Qsy8Z+#9=azlUjeW"
            "9_7ObV|AjF10)<XuLd^9WAt<8bBgSrjprD1J`H2mO?0>-tsWj@Zl~GF{dA2m&(WaiwK?(@Nmu4vi=Ix$L%%(G%l2L2evx}UJXl-P"
            "aE#_r&+k%09ghJ0%EOt=a>lXreU50aF2i9-"
            "^tI1V)<Y_qFZ@otCE>f`K~_>f&|7e+2}mgL`Qzuz)49#++(tv4&#VOfNk89I&YN#Wc;T2sId`renun`~_ll0<{8&?P>hZhuuhve("
            "^)DhxS1#Tw3b>Jvxr_I%qFXMyo<;XVbax7l#`wwVdLAqHaqN|P+LTfbmQ{b}I-"
            "(c_M_OFVelh@MoEEJh)(~dA;4%}?RuB;l2q@sD!dX&uUmbtd+Hu{|$vod(S2wyc7~%02<L0o6THokEVWy5%QCkwvw4zRWbeu~$ZC"
            "e519A{H)$?=BjfJX<j=s6tk{rE|Aw@W=qlcKIr+!+Oq#>Zd9Qy1%f>{ReX;f%%Jz5G!+?i?kJb9rE@!?9UXJ6sQZPBfoIkC$k4u`"
            "%MUsDPn6HfTA32(xz_t>bija-YY-;U*W|uH5DkKZSumkDwzj!;bL?dVJSm;OqRI12CG2tnWVKn6I?XVI-LJ({nkls_jTTvXk(z^d"
            "J5pE}bL#AXh)GeHhr`J1ANXmqBgZ;masUj;Vt;5r1->-Y7mfuv+Gf94CDK@QJRErNp;KyIk{W>HEhOe5}Y{QD-"
            "(DQ{{uHDp`)G1MqwpUrXAFpVLm0RO@L9%6W^$UmY73H{D~KU*{20PqN31Z&<j^Adb`7D>9G$SuxSUASy5p>rmkkhFC|vPL8(fu0D"
            "+2hZ_KM9^E$m?PKonIq?Qpap^d|Jssw+UgJ0@DpJkj2y;0ARLgmru6j+2*N3VHa@=6Gp76y_B7EzI5wGp@bB?q<{i}YsCAFLh0kx"
            "iN7=CBXiyo=bS*&^}9Y4_x5w~NNGpnM?k<y~K&g-yb?K<7!mQN(MdJ-oMu2lHs*3W1~0|^_h1g)Icj|S8S7pUP)!%V4RI?fsVB<?"
            "VLPIR0aQgXw}H94~e8Upd*@Nep3M(d*gaYV#Au1w*(dN~8%J=bIJHFd9}gWGg<;tIBLBB>!#985m`D(;zv+~WA|{dA_)mLBesPRn"
            "u1b)=5Qwa_ug@9M}~2&irIq~e;o`;o+Il0MerydZVtb_UCepTya~tVY!Jk3+H}{~_Q|Pmm)j)Nx|8`@bUoIym5nfO1CNI<mXNPU{"
            "~LKDCdB6Wryzwss^!M{LdU@jK%<;yZG(!$rSKK4(gC94~5Jt;4aFY764Zjq4=uM>?2c{om<04aP_8aMUAVEqN;cK4Xgf1GNKICRn"
            "1OHanh0Lv2%Zocae#gSCj?)#cp70~g|e(Q%zu|E|}Vg%SjI0s<OF=Ff1Z9((P6))WJAK4Ed99q;O&A1hVPh<h*N#3b^Cqs776MN2"
            "Wyz)1r!8+SNXD%geA?!5uq6FuXe&YV2hm2({F+-><xHS2nTUf+f^{&+@!fREN=dRD%DS#-"
            "5vc;hQ}PzVtq3gLKsboJRIxz)>=LDbb}j{w1-+i`sKV=)`j+2Y{!$UBaza!yB6hw-"
            "{k%s6<Q#nA(8xt^y6LJjc))fbF=w9Yf*tEDq^qG#2yOIkl8<3Mz@uCVRLCnGC1dK7B;qNmmRdB<w)<J8cd6_M1Cju0pFOljAj$Z>"
            "VRtRC~z&r1^a$}vJK9|?PY9EC#qlXiMj`d%?y(@-!Ip_Q&9@Yj*%8Fm4Fa-"
            "0JQ#YC%*&|C1Zal#zCahz91oD9eLi29uH7UMeiAgp5q|4>^HzkPuI4>LBdlgGt*5^q8S5$!OmBcC}}sBDUBI?73?44lBhj*f9A!3"
            "<m?<9B^*SkzfXB%oF+Vs)f+U}LnCryN<$PB@oFm5*;fPpW~chJe+fr%*iK267jIZ#l-96pY`Akr<tt<D5Ym)-"
            "l3)CRzz{t^=Y_A9MUpKIahCEb&fZ^+90{rL-gTYV{&vJc6YCu$x4!L_FmL6$o*?=7|jyMP0|3uft=)d&N5>N?xM%LwQP!LZKYw_#"
            "7lw2j38-8F7U3I#W+)I`4P#!A7I9<d_WvqQm_KRYKA865Ts7Du&OAUWts;n-KVz!TzAoC7vnL`niI-rs?>uGh8edliRJIfYFcZu5"
            "%a>E_xD0duWwO@lEyJ57gDud$sv_<}P|3a;*e-"
            ")Hueg$dEIsa?GRL(EjRhm62c?4E;a`ZFFdnP!N`I=;bW}aUk6znj0DC%U@^2Z0Mri-"
            "!a!q!wtU<GkzCLz=R#{_c){Os1H3%p`1Brn8x_lQcvCnbyjh-wzd54`{}C4JnI=y+!URliA27*IrMif-"
            ")(2|H42^L$uL14gxxmFNzcuCM$aZ_fH2hgvA0ko6s#j^2ciR*k)a!=BU_Pyf5$lOn^i8gd(_cGn&m__N~I2)HBX8awkDolR+AN{e"
            "4KeM@=*hV(%)!n#z&qV?DcVe{LW53W0^Raj`Jg*bC2TK<1C+Nu8>!d2DpO0I;=)MZpx_mJnoP4wAc8qW3%fy6p1k_B#&1SIRp6J>"
            "Lo&aaXvS)an{eYna`*jDi)83T0f&fYH^RK2Tt{Pxsfx@-"
            "s^di^!^c380XGHj3b^K9r04XQuT61Z{+kv5fO>p6iVZdcmvMC=!@1fPCq_!gGQ-`G3<y3Mvod5W^oqM8Ck)F?uf|2!w-"
            "9<_LKgv=_KH*u4UYXt)8jXxY-?dG^ESN$P+36;=LVbU5qo;GoKQ;OUGK)`)IgMX3b_QAZX-sUPd|zd_;=k-"
            "NU@?@U8H74?A<6#5t@=L<v03<A^C|EFb<#ym*Tq#}NVWGY%i3pRsSVd+~EVBHj4p!#5gd{8x}`8N9BU&b>eLnZ7;s=!~~4oI>$7t"
            "AmYyrRPaMzYDzQaW36_la+ABZ*50h+<rTFZTX|ZM)%M{P)3;&Bp<TQ!x~VF&TRJDrhQsxKk~#fO6`Jktxu1Oaz>3wn2vl|^|UTf*"
            "2_4SIU~6-xb_)|lmWikQfk8lrG7X7`Zh-hH6uATn6!+v+pzi63LOVb&euUrWN^m8$mL?0^SZIsW%SQM)(J{)6<RtX`w7yvb>TqzP"
            "FsRv3F_f@?%-Pnr61^zoGqFDy1IJu>+kDvfaiL7o&U3xH65v6HjEIK%LP^7dLZKUeCkR+mOEGEfrr6^4U7v{QJ`P<*J*G!W&a&m-"
            "#9H*7cQW$&X2FcAC2(l>OZUD#CzX=b>RxMeGJlvf;*p)nz4TOpoMkb(=pJJ6scqX1|%%ZFfK6R@pG~uguJY~BLh_8Ztyd*Yf1~hg"
            "Phd30Qp?&eAXjnWNsmHph&4cl}N-men-tuL<o{TjyEkR0T0-"
            "P0~KbsAXQ{HQ96g+JopUecn$iDHfC_YHU3#Sa_V!<hiFa4c^u78%^w?XwqroZB9MSSBm#vvO9o>c{FQ7ceb_Mb{FthCB;2BOfH($"
            "yPp$0%37mkIhN|vxkpK-EzN%Rr$G(6~VO`>WnMayv2JaUQgK+`A-"
            "Ny04EuR)ot^Z*sk@?ia`XgPT$E}UPzkc$#0ctVg@JEQ!M#G7yXdJuqAo~{5?Q>3CJ`&6_8186X8}d5#;ecLG0kI9`H^H)3Ju{Bkx"
            "lSw6az7V9Plz6N8CZ;X_NYhdd!ZpGIF1y_uhAx=9wYt^>}G%y-"
            "Z*ga`pMU6ZCf$vi2XQ$XB%LNB%m;O!+Q<CGmhFS+tJstkMoE1FDAzF8mNkt#rUSV9-"
            "dAe^|)O(i>2?z5CA<)J1Uo=c|<yFJfEN)6gHPSxKKlR5sQ4G@aEW%5ow^T4v5Ghx?}tAfEtPe!Wr8deDeLf4Egc9j8WZjfcs|r<Y"
            "7gh`}%>s4UA?UNNEdW9%n2h4^zt7zY7{!@kli#urNXd%_0hMLq{(1yAonY*FI)M6+^5)LxL$9P;ke`dz~^WIve$zv6+v30wyL!n="
            "@SG<yrP|wPOQzsNjSXxvK*dog`4WJ-nz<PH~azwRSyx{0Oyi9BEA_E_$#?_KC~i`f*9q&za}A$~cTaQYs>3jT#Tj*4vRKoslLJj~"
            "-k)<HT8%DIi2qeXKRgRfD@T#<_dJ;zX+iem|ZRxc^5hFi<QKPHIF1LNq^v1Bg?Idu80M5L6rSuiBCPoxuUbqZ&cch(^=83K)i3Sv"
            "bx`H*0jP5iO0AVOmkaFhJuhg((DWk%D1Ij5q*nAn`odW7|J#zE=cCXY7c@<Y8(p;qMA`-"
            "NMYnvkKrHmc_BZsp2UbcF+VN1ZZy>;WN{rCXf-fj3|`rbo4N*asQ5z;%p#_6(_YLvKj4*d|0k4ox@2-"
            "EKYy&G32xGDqCbT{?#}lzZsY1_@Wb;bYaX>z2{(Z3M3jrG`2t*B#ft6k^7FzVzqet2n&kf!g)|qJ1+PIwWaZMkZB!ewV(_wI6ehe"
            "knrxs^2K)tsvwO%?9zhUrT;yy6~P}UxGsfjm;S($i;KAdyX`Bu76gM-"
            "N;v>tknkG_$ATo$a9|cVS8?hsxEH40l3}*{rTt)ciHoIHY}=&tlgHExNxYsH1cZYVS74dr5`zT_=Ba=UI({O@ZMDpFF5g6eZ#St0"
            "WJNe)_2Sp!O;mq!xG5DXJAuuq5Dg07UavENP*eFB&zWFcz&u5OAps$5+X8{UU{fm!fp_zypgL-o3#1c+&DZPWVZ!uRYcxF-wXhOj"
            "q8)@<)$jRf%X-"
            "=mvXNEWdsgUc>_DlgBCpVtT7QDc3W0@c8c<<@?Tp&ZX@QtuI5R7RSAqjkp~MmPc}2lN{GDum@rBZw<Uc9rrLp23VSkQ9c=i49ZvJ"
            "_=c!jM~aWk)N{jvW}GQC~<%Jo+WbA5b6u!j}7Li>iRNH!n)cNc|1@RzIICby$OFX|s3JKdaA+`7War)K@&3Mw>b!_o~@F7C?>YBN"
            "Nc)UT_@=OO~FezM%-Mr>KH5zmJ-6Fbs-"
            "MK72||A`zZpSauaZ;!h{ocZ>#H4x>;K6}^Jc(3qR$auQ^mn|%gu)mL@!j)g&UQGh<9buh_x8^ydrS+P#9d(0MN&(ffeSNR-npSlg"
            "j$c7;_<p?CoXasT6EFyWlyIcvE9;ZB&gupqqd_S^1STrBAFJS7v9;)|mSH~JZh?@t70(ZBn0R=$GLLfz7p~wB6)a$I;z~z!JXqf<"
            "M@4eMDh2Pp;L;jGss&DXxWE_l&%$`HTH*K_XCAH&a6}_s#>a_fy1;uJp!9wOGYi<tc$gya9WErrav;NLYE{hpm-"
            "p+~%VjKyyY0&rU(vB?XEItrIZUw7#cDLe@1%ZEt_3wtA>dU|xgQL${Uo|NYHp78Ls3sK_66J4kG52fnZxBE<Q^1w_tE7sK0f@8vG"
            "4KpP(TDHT46noVSF%W1w48LC~|g2T>pOLOCUxS_wR+*>cN9RB3%H)EQgnGUlqbh>d_qLzQ|cI>W5<d95wTCzrY!fs|1<Uxx42D88"
            "6{hR&L?f2Wzm!F}AL+egrynm-Rqy)TMIuoCRsrX!`X$w*0Ei+@Z`p7|n#_$^8c~Tk$L~?W17+AY&ExH*Or~D;;-"
            "!{hbjnwMEX@WI?9D<)Ate#tV}B;{~CgI*tV?oYCqbgDN1Vfar~%j8wI_LuKhu*$2j7Hc|T?-"
            "(sH5XFFYS@WjUgE4&UjyPSZVNY#ptZj;m3JhK2QbaFx%r$Wsp1fJEKyCPj!FL*!!8P`!XRr7VQO%=ja!AZi6F(S^kYz^Oph-"
            ")x*&^bP=!#uvqW;xzq#pKoY`{4x978e_tQvqQkA2;)2xd3}SynHbB1-"
            "Y+rwd5+NJ*`JYM6qZ>*!rkZiWjG0T4Ukhf!7j#Vl{W+m_g;>ZW)Ue5QML)rh>Ttr1W$ip6xX6&{2u^rqqv!X1QQQb~qI)8Xw{EK9"
            "BHsK?U^@C>ZJZcDVOFzJ=aK16fgYA^$(_-nG}Z9J>ztD;M?o_66S$1^ytoW5Yot$B>-Zu?(TQ`ZiQ?ca>b#typmU-"
            "!sOT<k}>69ojGw1PGM4Z=H3{<|fzVe2zKhaot#ms7J!#R6d~|X=423D4s-8V4c?d^w`)KyeB16IH-"
            "s4a{TUqBocM(i36X{uN;p~2@1t=2bjg#6UQ__yH0)MbBd%FI$j{tUUkn89rzq&p7z2WIr8^17zdB+bs9!2d-"
            "=JwW9o#muX}Qu?rYzxm%Sc8QF40AnwAM^#Txz0@Dlqt6U}>RixBKBffk8T+4H9TgjWa7x!8YF&y-"
            "pyRJ?oPfpEJmp~rTd&4&zR5ZQ|^Y$HF)LjK@~QI+`k<hR6k&IkJT7YcNtpXHq5kI+?h%PmWqPWRbr=o|7P9qpnj;q8f3R!{c4u3E"
            "_KOB^OSu=nw(O`VD<u<v7+KH|=HparLA>}A=O&zvtq^Suy4`18J<7S9@`ezNDkp>+0Q2_fD)l7tvd`$vjLCcKn%)@;}|zm-bW6uc"
            "E2_>K<{1g^(^2bdz+>kH}BeJVWeN0&lAu_gZ8a$xhqR%#1p92LmWwBMY_wj%>bog4J|NYRsAaW5T_^LDSWk<V=t$KXzU1XBUquh7"
            "ozFUy_TUEkT3x?|3mwAp{Bo}6oR^^QiBenK6gRnJz4e8L7*{M^8tbX0b@G=YQ%w9vN`^j*XGIr*e`VcWz3v`k0&+WHlaX6Vu3$TH"
            "}G=54d<BAo5f>{d9=C!ekExUmPcmD4hIa3~u9_V3L1wU<xcNE-u5HV@QbIZ<O|o!%h|1X^xDU;WN*gb0f~5Q;Ran-"
            "28;paPH_?mr<f+g=*=>ENLX^j&tql&?pY2*1-#%wjI*%@M8SY7fW_y<Z&?Xl_@>wF5&#EaY*@W02Oe^Zo`EB$jT+Fw&vl^-"
            "DTPWJ73Y5tf+B1NB++34$FXz=A)gVg;jY+`c?(T)Xy&Ct`+lH@o^E?VPhmr+?u|>aiIZe|7`mO&mdF-"
            "~@uLr7E~^0;Jz&*YG{<?RG%}QZ@-"
            "Ak^Rw_5)WDzL3?);BXci%O`4IxULt3~aEsBgQ+MKR2~yu~sZVFgp0ufZfnI}S_I<&7*!4&+C$WIE^DYVXNJL3Cb|cMX#s=aK9G>#"
            "75w<pvz+C*+oM71Jy&>?8A+P6?=7@7fHWCedPWS~>^&uNVE$yq<UCUz^?Xc+NH9vl$_T%MpQ#5vD1V8vV=5kWrG0DF(fheK1_iME"
            "<yP--tJFfi6hM|1P-"
            "hfu#`aj!A$;HmVy@!^0$K$WkY1le*)7xJ$g?MyVo}e3bmL8o!{jFcXoh)tjR1WG)K6<C!*?~p=kb~#tZ8Sk9`QqOFlTWHT;SRT*%"
            "Awquklxq|m~%f^Dcspblu9~zp~~qPrGEMigv{;@e}vWUy*fnJo%}vWy+Qmw`+Mh4c9(*z2S;yAIOlm(chcpm`~B#>Ke^#+I9Z_B9"
            "d&f|=aY+%@T=o<-"
            "q=g3JFsn@pq8UQuAaQ+^{vO4{3>Vn*c&4afo6MKXgYb{<;?A!|NUf!W>)TK^dwXD?(l`T)>j(4<9@l*ZS7|l4|}lpciP>XYfH|i?"
            "fY=h+}?e|JjT(_SkER~t+l)3=_s+Omvh%G8tJ!do3~z0>bm0_vPof+I+aoPrkPaUjQD{}2<{w=DTm=OcEmTW8h}Rs&@DZktQq}nX"
            "b>v^dj^R{d~Xn@4LJ0?KY0f_`@J^}(BFnL^%F+sIr#7BsA~1xJNuXRQ>_DGB=_q$yL>iVxU(BAxcuf`;B&Se29B{h95U#OUF&fI("
            "D6BM{c<{mz5A=3P1>@Wo8BP|=CHLPyuID<c0=AmPol+ki`c{2rR+}z0^T%tSo)JQk-fWzc8G6XcBHM`dx_gyM<0ENZC~t3KDa32Q"
            "NZ=e5ADB`PIf9<@k2A_bJ8KT@UD028T_fc^3dG~4EFEL(%8|Hna@cdv;pa0JveDSxv$fIDrb{_KkoxQsw2P%Crc!Ma;*L~8?wEz*"
            "|j$!aUx|8U#*@TKh(=7_d_^;qFJq*y~%Igr~d59ZQ-"
            "0R!`a3|Cf?ZR+8gb7qh;P{723(k!)oG?)_RU^WX9wqnmFlOS$p683i8n$GL_5OdrKhW=3x?hv3u~G7QMgK=1xXTX4&_1B|>}mZ|j"
            "NJht$V6Mxl86t|gu<NcrUEB{Fxnt)-"
            "q`1VJ1p2UC3Gx;p&|_D&st)z6zjuDFBC3Htcpqswnh1^y)Ut|!7mCw6#r>6BJ_p?lvulf`=ik;%As*1>~Uv7CB>-"
            "+7mM7BTkK$y|3ZCBE@JF)@A6pOG_Q=L=*1@y_9u&yiy6&pO^tbSIlEwad=J`3qh5<@{v##<2D_ONZT4{fItJnNJhi=F{6qJ4QIWt"
            "TT4nZ6XVZdpwig205~`^<e#Fd4A_U&gX0?c5&_RGO;ydvmMEu9JK7c+|Mqa5N021I=RK`7~MrE<*o>0JzsMaY|*~-"
            "(KYR(e;e<_PuzbJ&TFK4^D#KeofP2ZB%UOx^Y|X`6wqwR-?^#VoY#>~&NxCN_7h?Q?H+6M&v%bKo?QY6X3F^$-"
            "8mtLxcFlX0AU!rETnwCv9tJ(mvH(TY$x8eirA)2-k5UQUsOAXrpE4%8%4zaZfT-"
            "FCYtLd)MJcAIy7l0Cu0YjkB{*;d2+S)Gt?)A{XwkF*(>jK`8yMOJ!u+PE!|Ces3$1{U;H_3<QyC-KKFesgkk&lA?&Ij!{biXOg{I"
            "6b)8w`eF#)P-"
            "B&Um9CZ0)x8DaUwR%qLE2p92{G($$nLpD&KD>=;dLtKQz2#uZXHxzc#C8tAqE^_=B*(xV!|dwY`EUT=K8wSh?yLIR4_f&*?ea!G$"
            "0-#1Tc$)E)bs9~`^oP7bLz-"
            "+3FBJ`8~Zs(m(=QqiuVUdMA!QGuK7)cob8XAKeE4Xs*uA_AynHHSW>UZ>MTU2fA&wVC$%6EJiB8|BJ^E-"
            "68Ccwq5DJrQaa62YUhM#HW%+IiEYBiPtutlOYtN`C6h#bRO2znq|~=lIC<OjE_V&oPm>YLDbXVG-"
            "({+Yw>dHI^H9=3lEdzc<FA^3dr#rGh1}QfmXkMRSHonWIAokH_WSo~r-F}aO4(#9ybT11Cn-"
            "6l|D9a_NEfGij!>XG^OlG{oU*q9HXXP6n8|Z;OnUs81EsBrJjax#bWS#pZ)2?WJNw9&{5IK#X_|CewPS<>zx6)m!JXs%qVsd+#yR"
            "*~N_JPoy}TL4qr~S4tvjc|oL~lWsvP5lQh5`cx6wNB%%{e;$$#-"
            "Jdqr1nugpYuP8H=r?TUhACWm(x&R90O`U}3xF(AmD8BG?t53}?(q$$+5v3&Jy{z-"
            "D@>gku%!!BCsjb5AX_;HQg0e91N+&wUYBM;xMzxkMF##dK=4p-"
            "`@Q<nUjOuS8a>E)bQ)KBW|_{M0?IV$H*!uh`QIj4cFFGp;Y+neG#`DB0L6nZ*U5Tkzk$4Q(@^ym?yvCCld;MnT4Ujx$79G*crhXt"
            "g1mOdG}?%JEE;UjE!;THnugStJ$lV0APW2A&aJ1EZ2nT@3RKFWx1G>K$m*{y!g?-"
            "b|l@D^D#JOsL*9yh<?o^Z==bJ0aQwNo?Z_dy-!<n~-oA;|(v?jr--GHT!;(&xC<eN@0(NM6E87{|{!@L6n)+><-"
            "}oW1OqPf}+VL60O+Ij<Z?Wt^1crw>G4h@)tI5{QX3eV)R3`AB?njDU#g?G(;jLkXuhU*5`wq_aF|5{UuZj2dxkM}a|IfcsDC`4fN"
            "6M68@au%}V;nWSWb_8>+*c|d$Y1hT)mI#86pkmq%0Qe=K&=Wvz7zq7&nlh-|-0Z5G}KQk*0@t7R)#^V^z`O`d>gI3v}jOb-"
            "%l$es-b%2tT`%ltYMk;S!Fd-"
            "6$w6t@wLv?Qyqb{7_i+GkdNSSd^(4QzL(bON*`XswPc%x5!%(B3K^HtW`i{9=W0LJ8fI!Mm_Sv=3@aB&fj@1^N?+H>N|_A*C}zd+"
            "Kt!!cN9+qEW3dPn((++jQQZ<|@d){fG)VLgxH``h2wle?Qd0lU6oDew3zU(W(cKfxO!$KR9o<aCH9neh1b@f=JfFu?&qIQ30`8<E"
            "D}Ik3k5jDX3Ea<FT@%^L0L4HCg2J^T6Qb5I5WPY0wcoHX!6Iv+CC-"
            ")P|584a=lYmbCkl0Hh?Jk0|Vanii<cgnG^a1xxBx3YD9gJ0w~lwK`wKRL&;^*7pNmf9zo>p7;5-(!#6_}j-"
            "0nVEVU7BCUy{RGvtvnXQ?a(fy7ZA8I4*?Ep{{H$-"
            "^ti6$@GdI3tulZMblBFNuxme#w*H6cq<n0Gw(4AgTPv`JA(%rg=8aci+IplVfVmK38J)aW^2+ZGYS7-"
            "0tzMcBk1RxTD4Ua+**jaPkG08+Ew+CR{ZK=_J|6jiOFaPE9r*D4#{ttip{kMPo@uy$K@|CSZ_{<Q$#@~Le@oWC(&p!R<Z~plGAHM"
            "$l$8Z1e_xLYPF@Qfne>*moK^i{er>}qf>HDw$@b$O&g>St7^p`*W!zV&$Kl%J;A$<DRfBjFt!EgQc+dqD5pZAY{|CPV_fBgL84}9"
            "<s-~RqjU-jg>AOG<8-"
            "~RBAeE66D>%aK*zxZ=L{^R%G{fDnV+0TBhpZtWie$bx;j>qCQiC}Sh^!Gph^vAEi{oBj${O<c7zWw35@4x;1-"
            "+uR}pMLuK!*^QnU;ovw`Po1J-"
            "M2q|{h5E32v>almHqA?|Mu^H`u@AGItiw|o(J29{Kr52_{Xn*iw%Fhp2+x)&o4&*>3{n62mRtV|INqx^N)Y}NnhxjfBWs<{rKajf"
            "2Q4i{qzstfBWelZL44Tdw=-"
            "xx9`8j2fzDIfBVCa_67d($4_%NpT7I?hky9`)6d`k_`|0kzJC4f*WZ45j_3J(`>a3y_{VR5|LN<0{L}Y%lQ^PJc834Mr*D7%d;9m"
            "N?|%R7&p&_t`P275eE0i5;j=#dpMUoiKb~*v3uN&S#^3!We(QhhYWpsI{S7X`cj0#*2&0dcJ!An^F#cCgEtda2Vf^dA`ip<Miz0r"
            "qOQPqnC#a`REJ4H#>%T0GkVLo_^m6Xv_}Bm2E{>U?U<MZH7o|~?Tl0U9IKIy07sT<`U%%ro=5PL+5BC2_arB>X3ZMSTC&5`_kDsu"
            "gPyZKxh=2T(e}jK6U*pTCfA()a{oU8!|CW*q3HjlV9}4WB|N5_fwTSC?KmG~7{OMo*@bwd3E1yy@aqM{b^QYf^{|{e3{rJ;w@eP0"
            "a$06G9^7>u;?%Qukxmv$<zw^KO8o&PLHy_I@^f$dYSN~s<*FXP@U+>~Nez1sPDg<KO^yc|1|7~iMA+c*!71Upl)2zWR|38z{|IgL"
            "tAAkJrKm1BZ{PVAV_iZZvtNS*&@9J-A|Bd@vfBT^sXSZ}|vHSm;x$z$g;_rU?_77iIN&MzN{|#jE_ka4s4?q9SH-"
            "Gl;{zhi|Z@&5bH{bllU;VSc)<6FFzx<c~{D0QJ{`@cg_rJ7f|Knf(`S{7N^}m1hzx-SM@4xz2|LSkP`S<w8_doph>wmIO#Mg?y^8"
            "Qyo{<U*|rvL8?#8kikJWH2n`SPq>p4E?^1$TMoFVDi|*<}ym_FdZLS^xN1{BRH9SI%FarOUH?c~&mZ>g8Gc_?f?aPaiJN;^o<8fB"
            "v#RfBT;O<7eUW9>QgR;j+JQIqPsa>u}k_2eOnPEWdK$@+@ASrOUH?dDbq^F5lC;%QJs@cG=$tI>esU%d^WK{ACZ}@?FB^ti$E3!{"
            "w~QWq&uBh?ldDAD;Dxy2qY{%d>oWRxZ!#<ypHt>mNV!m+$H0<ypEsyX?<j_7~jc+2yRmWe?%9hj9BYm$MF+v;J7l;zeHKMZd(0yu"
            "^#V#EZPdi@d~(eu)=(i5Gc^7kP;nd5IU@6EC_aUSug=WGP-`DPD9>yvS0#$Wpw>Qo6`Xy6Bg5v192XXXzqm=^`)bqF>U*j-"
            "`v7rHh=Ui@c<Zen}S_mo9RaE^?MG@{%sPCtYk@y2x3&$XUARo^-"
            "Ks`66feB4_y`XZfO+@<lJ@i=5?)oaKw0<%?d*7rm4(a+WW0mM?OaFM26o^isaaS-"
            "!|wzQ|d==$?F$rF@m8kNu@`k(YAOOXZ@M%0<q~Mb64aUdlx;m5W{~7da~zIV%@=DHpv|F0xcEvQ#cIQ7*ctTy#&l*tl|$vwG1z^`"
            "d*~MV9JCUg|}^)Qf(p7kQ}{d8rq9sTciHFZ!ik<fUHZrC#h<z37*E(J%EPOZ6g4^&(64qI>E^_tcB-"
            "X%{(b7ddMeS!x&E(=NKFU1YFbWUyUisa<qWyXc;Fk-"
            ">J6!FG|QcF{}iqL<o52HQmj+eMbzMK86BUTPN^Y!?~q7roRkHm+aftY74;U*x4<bXLFUtbUQRevz|&k(YkaS^c83`bEzAMb7#~Ui"
            "w96^^4Bx7dh(}IqM(Gi+3OEB|Zz6XYukZU7pp;vvzsbKYr#f`}3Fm`OEkGP%rUWxjeh<&tLWzF5fd;-"
            "b1*&hj7_nxa=?7zUSpV+~n*-+vxp81}}MUf606ES-Cu`muKzcXa4d%Kh#+t*W=Qq9+xilID2;cp8exzAL^`+>v8E)kF#f&?-"
            "?%dAza?W$8vVbZSz^YJWH2n`SPq>o?X7@$9gGTWa*L<_m`ZwzvQ<0tX`hA%d^Y=!exKq@;$@lJ%r01!etNf@?AdAJ^50P%a?kbJu"
            "8=I_42HJ{OkjHv1ga>@_`PqXZiB%vIl?JL%5u0xSVIW>>*tC5N_Y)a-"
            "JXPkbJ4R<x9;itCkfn&(h^tzC0_JXZ7P}AIr;!`Xyg#ZuwGk%a@v)J!_X|{o`lhvWIZlL%4jGa5?L6IqPsa>yPCuUSuL(WFlT<B3"
            "@)7USuL(WFlT<B3@)7USuL(WFlU4NW92PyvR$u$V<G)OT5TSyvR$u$V<G)OT5a<$NJ^Nnp?iq-"
            "14R7mM=9odsZ*c+T~gQ_}RxYm@YDyF0zy^b}U`&Sh~nyy2wkq*s*l6W9cGi=^|(8A}{G;$I``)<%=xki!9}fUdk7}lrOTBFS3*`v"
            "Xn1+DPQzbzQ|I($Wp$@QoiV=e9=q!B4_y`XZa#$`J#LBMV9hKmhwfG%0*tvMfa48?kN{JD;GH{7kMcc-BT{Qr(EQ$T;!}=<fUA6P"
            "r2B?a*?HSk%@BAJ>{Z%%EiW&i=35<ywr>CsTWzQ7g?$o{ZcRbrC#KvUgV`-"
            "<fUHpOTFlqdXblUk(YXrmwM4J^`c+uMV9JCmg+^8>P7d|i|(mcS^C&tY8N?c7v0k?x~E-"
            "qPrJxqyU1X>$WpuLo_5ha?IMHiB7^NBOYLIg+C?w5iww4l47Q6bwToVA7roRjGT1IM*e`miU-VMH$XUP0S-"
            ";3jzv!%f(OLZ>XZ<2){UR^@qO<x%XZ4Gm^^2VKi@fxU&gvJP)h}|^FLKte^75h1`nax^FLkwisjJzua(PxS&)Vf#|M;1|?9X4m=Z"
            "AXf<GNbD)YbB(uBK<<@;$@lJ%r192$%hZ%l^XcdtTl{xV(p(oPB5;{Uv|>aa}E6>T3B?SId{Wnx1_uOa5b>^>JM-"
            "U+QZ4Qdi5Dezkn5tK~~wEnn(t`BGQQmtMAfsjKBnT`gbw-"
            "14PgEnn(t`BGQQm%3WM)YbB(UoBtiYWY%E%a^)ZzSPz7rC%*y>T3B?SId{WTE5iP@}*xbU+QZ4Qdi5Dx>~;UtK~~wEnn(t<x*EGm"
            "wvT!sjHPsU9DW|YUR@BRxbT&<x+DimzrC-)ZEIY=B8&K=#X-$xs^-"
            "Jtz2qu<<hTKE;YAuskxO)&8=K&ZspRiRxUNSa;dqMOU<oZYHsDyuU0NKw{oeul}pX7TxxFitX`hA%d^WK!etNP@?FB^ti$E3!{w|"
            "$ma}+~iFlETc#(;Ck%@SbiFlETc#(;Ck%@SbiFlETc+ny8A}{eGFYzKT@ggtrA}{eGFYzKT@ggtrDlhT!JwLWpAJ*K;rRHYO%H>("
            "TJZqO{{o`jJ%UQa}OS;&qbg@<GB4_C$XXzp@>0-yy#g3(moTZDLrHj0ziyccBJC-hTmM^lDFM26o^isaaS-"
            "!|wzQ|d==%swoOZg&a`66feB4_!cm-0m~<%^u<i=5?)oaKw|$roA57g@>|St=KKDHq*SF7~fn<g8retX$-"
            "!T=Y`8=%sRzvvQHMa*>yE(M#o`m&!%X%0(v1Mfa48?kN`=S1xi^uJZD+?x`19sux+R7yVK%`lVjvrC#KvUgV`-"
            "^h>?ymwJ(xdXblUk(YYWFZH5d>P437MV9JCmg+_K)Qj$^7g?%TS^C&tY8TzpF1n{(bWgj;V7tg*yU0?z=$>}bJ?$cc?IMHiB1`R}"
            "d)mduwTle4iww4lEVYYXY8SoKE;86IGT1J%)Gs=#UvyT#$Y8(7V87_Be$iR|B4_;~XZ<2C{i3t_MQ8Pkob`*G^^3gpi_Yp7oz*XL"
            ")-Q6_FY?kaI;(%Ivp%l5l}pXdp5@E4a(PxS&)Vf#|M;1|e9sT{(#JKoa;dqMOU=!mUA||yyoYdk58<-"
            "EaM@qDe9v%s58?72!exIS%h@Hj{c+8$TxxFRQgbVpnwvfAA3yV#J-q2Ga!HcKl2nraE^%OQCPlRO{WE9sgf<85efE)KcvE-"
            "1l=j(kWHM(plo%_1H<T}~RPUP4DSq~2?6Gs5JiHvnlR4WS&(r5v@j<P$<OVH02jbxmlgkB9i^rc9&dBJwV%6{^77};2?hBSRj&K)"
            "U_U&_wkz+km%J?C^l<!|!CEtF?F?~+>+nNi9H^msRZo^YoL(Ng7=Gbn<SBqq6En2uId~Wi+2P_=F#Oxh@!#D0XYIJNbJvDq*;FyG"
            "1l8EZDaQou0Ui?yO2-VlnGS=h3)!cH#PqOV3-vg%?UE)SLf(cK03K0v${H4SmdQB}Soo^#AvTxD*&z#TYd@mt17piUa^)zX=I`-"
            "8nPDu-"
            "==3*+vSpE~g2!}t`mhi4@Tx%$y^o;MWLGdZ$40}sDVb3!R1)rydTj2m9e6P<wxGyDe5JAO<Yh9dA3oVwhZZ54QGQ*S7J(H;(FAm>"
            "}`_*KL-0`MU*0R0iihVneRm(Db1FJ%QQmXjXp71nq93s9`tF_0^oFV5Ljw<2A@#0#-"
            "t8uPGEL?26wP&1EBDpSyP2mOLlsPra&KbuP@jdWae6JMa9$V)xM~;HPnfhGyUTwywi=_mfbN#}Jw&M<PvLE)wnUzK>4}r?n;iDkv"
            "wqbk?+`I^>z``7+obYLQbGGBkaSgaN5ibRM3K?=r;%>Z!5Nw!3;H*OJxP=g$UeagH;R>9b7qJa4%=y!?;*WEwcG9?AF=XGfcAOmK"
            "thk~b^K9|aId$F&&5+?#Uk+T*!f}qQ;2Y)u3rJn&;OWmA?$un$oHLSb$F4U>8nzO2vkUGnUUIgR#^(n7I#mM41@awcUO>rJ4yi#V"
            "yVt6mK!9%<a@TfoMk(v)v}#HCXTZg&W4pCsLkaKUGsj+5oMVfmw%4*9ry4m$AZT6pk{uh0i{q$q9JAYx8Hbo^EHT(IV~HS5WXt0I"
            "m&}8;&2~si#<^)(j(V-"
            "ROV{v_BQHvgI02~(h;}Nzd&zUuPzKz$n!J`xxNfaQN(Pt35lNgiVF902peb;b3iVXgOvM%}&V+B*bL1%QV=7wKty-A_#R|j%$~Wj"
            "5t<ubKO&K3tII=^hjP2lB@R!H3Ay<4m9&+I{E<Y}}i~;@(2h(v{Z-"
            "*Q>uNd1+5n`fs<8#TdF4b~%$?OWZn>dk#^GdXC{(@JU?0ZNpLrFuta4)$BNF0QbBQx=(J?GHJ*M?zta5B9%s$8jMj)O6+tqsE_)Y"
            "5CsN$RR`<VB&nwhhA=;8t<=&S%b^th!*h_S^)=$3hEfw1iIjHB#a;N74pt37=J~*@L(kB2EuXJgUIM<A7%<X8fUR&d(iF)dk&1PL"
            "%F;9FkE8B~%?2!sXD8IX%&(fDiwyv4}Wo=oRQC8Q$h!JLbH$%29I#7giP!SJO0o8asx4#1~3hm^KdY90rJ=by}o-"
            "$B^L!lMF*4`xw?ii`X$#Bg{x?)!s0_Jk(Wd<08TORR`hfS)!F3E+y=#k8c;LexX5zSeDk!?&6RrvSZc$dYY)reYF`$Jm>snXuYV}"
            "C)_TS84OFH;1v$=X+7a6yLlSazfCX7JdKyoa`+|3@Mu$A){B$Co~wkF@RmEB9!OSe`de)%+J-"
            "YNZVEvnU0?|l>@R1Q*bS}@Hqz;RbH{a4l(;J9q?i>cT-X)b7!G00ux1btZ0s{fB;pQo0w5${s4SQ(C<t5`TNr-=J6O7@4k#(?-"
            "(@-#92K+NB~69JWeo+|1;5;6Wo=5#XS^(22M^_Ms<Pwa)!++X`7nj>H{fHyIF{i`LZoq&+Og~eG;qV=zP;l}-"
            "xOj6rPqoUhHsT|!RU<f;$*E?=nKhkO<xWorGZRxabTx#;!qUTHVQX`bH;gtV3#?}gu}0^_F1WZagJ1yuy{T_4gvh4FDb)D;abr0c"
            "HAJ?C0qm>r^C;JY~V;ss?r%$(xm*)CMB6;bZ}Xy6*w)pzKlZBw<>tE883(XTs#quqJ#+yH2ZY5IFUEQC`8^(TNrn~P{Md-CF^Ten"
            "AmZ3@ub7J@@w#2@uTsk3Z#I_8c*?A=7_j`!}OjaUDQO?<v3vp3ZPLD4vB*1g-"
            "OSHI%p_MPU{S{N;>h;jst@5$ytu()Cu?lczlp~$N^n>x+SmIK!KvGO_A^LlpqxF=Dd&gavQ8CXAgCA=x~A@{3ee@E=s|-"
            "ua#j+j++!`&Xa<6fg;OIzucF5o9UEzj;*cqqdCSbLMJ?mxxwf`XTiJCX5l$y`%~BL;Ap1d<n#J)1B(=A4(h0Uzb<AiT*kGHJY|h+"
            "?A`05I2IPJEW`(P+z*c(`@|9O(sN<xT-cW;1;Md&Fr6;>JeGZSgpdVt0fk=Zn8>|d^%SS);X`RWp+0f*@vf(Cv1Q@ed+DyEW#K|{"
            "7hNxxP7DkQ3<56-=UJ@>O+y5CBG_A@C-"
            "Rc@R5Upezb}s<ai$#2K|s8aMrZ_sh&za1|IBgN_$+Kb(bVE%bj6B3*nZ<;hOlf845Q3iKn-yAZNj-XtqOf!S_z69uxc7%j}O1FYE"
            "$GxJrT7kE^Qd!Au~99v>#tCq#UoHP9O6)Ww^GuKSl4FGm$IjCdOKa&y1JZY4NKWT`Ze9V2pBtS6#S=)Z|z}Uc)OI<5V&1+=O4>k&"
            "*8V12g?%Y!6of&LDLpg6tAu?S|Ebdn4C9jhmZ!$OSu@b|o`tY&yr~Zc<Yrju|G(w9Cu6<<0GL<{Fo!x9)8e-"
            "3U$~GuUyEDD!OK1!ta}#R%AWby5*>ZdMJ6yBru0ArQ2)aqjhYp<ZCgEB!(Tf2}Jv#WA^TVIXJC<7?pmqAiTGfZqUBUCn^wV?xlEr"
            "nL`CP{fOmAP~P{5!7+coSKN;I5_b|Mi>9W%01`>Q|;mJIOomVErR;QQgCX$4w~ZEc@L9buz%@=NHdqAHS7Qs7QcdV>Sp-NeFzrk+"
            "C!}5foXajCr`o-!HekQ7U8;$fpeyT5H#Zd8bgJ=&KK1^;sm!2lTeGkR#+|)$1_8av|+D>akP-"
            "0W)59u1Rih?ahsb3WCbP@y0V*r!582ez*3quS^_~F4?c_5Cel_hhK`S|lHR~75JY+`Nt0?zphpvfEnF1ez<q?AR`@sa8>uU_FeJ9"
            "(OW_`4;XSjV>Cezt$x=AU#=Z`}nhS^Nq}d&&zU92CHWr3#H6}jf;X0afn~BGTIS_jImS*?MVh0sCBIdw>R=wh!Uja8&7TjVP5nC6"
            "Q384`5y<V|KUOL7<Obc;lDYO-w_T+3`d>-sLV`ZK667Nydblza-"
            "b{gllrU58gjT7@;`(1SwNAuBc!lBB_d%zLr*!DIZ2$(Je6{Q6U)|%L$OH?!Sbv$)^pDzp=rpFGI1xe>r_Fkh5aO~ItLWbIHyFGDm"
            "By%cq=y7Nq@G7?sfuv_PFHA3oAr9wkBf76nNy|ckc`aA?`HR`DyvQ7RiTlLmI>#O|MMzJa8U<DvFWKTD=m=T>IDZI)B~GjTV>F3Z"
            "!F8ysa0kLm$`Kdh%^>u|Z{TY|J>nFyr+=f&QD(-"
            "@=Oun%wuke!odvmg5goS+)=1)1>nqqsP=}ZKq^yP&n<|6<85!OoR_a=~Lb=dDXbvM5=J8ZqgKRryvH(s$q(wU}oVJU*#|@YDhENL"
            "N3iHNGIdf3;5{eb<h{-XWHV!L!8XtPW5HYUH!M;#K%vW_BUWWoPlOP)plM$HVa265fvdj<1!Vro+T@9QxV~dx@sEQtj*~7wJfkB-"
            "kB&kqvF+Ed6Nj4Hvz@eIY#n~yE##atn815l=+Lx)npFJ=8l6;lKjXVl2Qs7We*tIaS%n*2l&jo*_Q|s_SnEWnXSzo%RP&4E8Kms<"
            "88LuO8Gb2S@Ik)#S$IZi2x0pWbXyL&?DRMw57YsNLMtRtK(3N^+q2-g~Y@ibu(o4xbTAqG)z+t;0-"
            "NWV9By*+cgvWPe>I|9}wkj#&!?G}dxq9Xgu&iU)Y6Ehk_6hAD;rZhmXrE4pPj!RXL;(;E(k+^>@_xY~MtG63UL+zRSmq{;#GB0bl"
            "xYq1ieMq0{5TVNh>1P#Vl>UaB<~v&J-"
            "EcdqBM=JeMPoKAvJvZkSI@{nRl2vrl`g~ak`ND5%_LefJU|<1dAg8%yg(<SCyUuwxnUH+^}0CeQXrpqv3KPj`6yx_!clxum?0bh2"
            "c|UUf=Oz(-"
            "h*;`jH7q{Ak3fvQYSF|D`QyT&@JAIun7M5xFD+E5n?OcjTq}IoQ2H@i9!Jb6yyX;CskVw&<)vZH~9bQL%L6NA?vft@&e)qn>*RZ("
            "6ZZy$5+@xL8~m(KtQ`XUn*+&?LiwZ_FEMVSGbac5|edMog~+R5{{yUL6EMx(Ry6OY(|4hD#v7z(Yv{&*Ld|W_s~i%v9jlGan;vz<"
            "Z7+5>m$;1#d}k?RYIt;5D(Pk>sN2=@K|3C0PsZCkHTNVZ38nQ7JxK7-7J0YAs;N)K8|J=pDmBN3`j+-"
            "zCoFvM}rTUeF|6Pw9}%a1+aBCKnk7<E;dZXUE55dvuklQ6Nk3pK_}bc{sDIFtlU|Df}1q&lIfQA+_PSOb9k<VMoGUmw1)w#<AHW#"
            "$=`&%Fsa!p$Y;;i&q|3p=*pc_FL#@dq*d>jG_=Ueap<(@uTCqV|ws7OUC|H62C_Fje!LuS{W+r7}5dLYmzv2%!$JY6_S@|G0-"
            "R@iiFdJTQ%J%h+&DZ^fg4f!fB$NsJ8CvuvX^sAxfE}jwRAq+hR&Ci^Ia^A`|INx)}kRYfbBD8YEsne$eeE_2nc>Q%RMkm`)e(>`S"
            "Hb1>-A^5($=N9tOgV`=Rt0-%8^A<$~Z}S*-"
            "fhailvE!VP`pAz7~)YhpDKen{;~KGd>B*ifY&?sHbfCou`kItPbd_OUEn@x;qz#+F&5t}MA_+Te@Al(lLio0sU0!>#dd8-"
            "9&t4xHTL10<Gaa0L-J^Q>54#wEu0g}xXnD`c2d{8@`q(vN6(EexB}@T&^Bp&Q&1-Qho>Vc?5-"
            "M#8vPt8r+w28hh<u0wAi6vtPk8sW11Q^c<dLFLsz0%6}qP|YYBqTiJ|qgX8!bc(aX_k}A#hY8nZ^J*I7AY3~fiI*)z4B4RRi+l{K"
            "GtUx9h7eoOz-"
            "LNpnWDsRv%)lV@EXiQa$usF0ao6ovEKX2FL&l`S{jMe#A-0FBo$|{g|P?N(ON8vHNs66Ot386YIutfL34<Ef5jV+Dw6}No-uM;qO"
            "VR3geZ}TfgrV!@s8z#jb*XBq!8TVV8|pqux14WxE0LY=sIiRy4oBjNs3fVwuRT6LStl$TQmyIFlKOwtd?l&CKd?&NCtNn9+M5Zx5"
            ")lg{-wmOih-q5{-Nn7-d~4(LLVZ4wKy?Zni{KRlj2IJxiQ~NjjyO1#u4s(%a-GX4#($047-+Fi+em@s3B&-"
            "DQrC7<;!4Q=8&xHEA5DIwVMMz&iC;EWQJhAS!mHpEM#Z+)TJ^wITZrDl$YDr!)6;*o^Z@tLhBnoBjT?Z7hqLNVqyidDn>$Z#-"
            "qpPXG$9i7cnPwwt4;Gl|6<=<MjD%x!6GGXi?dnd4Z;?NAxOc!;m8y8&&Kk(HtdaS)oyyHEn^f)v4CTN!IQ9^vH?}bG1PWGx5G$AH"
            "rhbl}2G6h#$Z!fyCfyQa|Fbm`legGwSO^M8Ns0qioNs*Ppr{#CQn}sCZq2$tLn5Y25`s!Gnes&883aesT2;O+mpzd`WZ#M2+RB&R"
            "P~MD?&)#m&gbmK|MTgN;SoJxgdOnZooM5XXsqkreEvgm!UXSaw~el8827W75dJd_JB?YYb_S-BbugfD7j)L6fQsE20?<=URg$nCJ"
            "g}mSLG}5X4W(?-#-"
            "o;%7i6$gDo_3ccqvc6*CSB7D_h>zg=*6T3We4tXq^y<HC9MoidFy;#kBj^vNJ?TQM7^CG&+^ZNeZ&5bjx1W(g8*2ZSgPAkm47ss`"
            "2R(o~2PBplJ33sauV39m3-Aht#ogS*tug-ziJ=0M%ynMb9P<15B{VuC@LfU3(B;o;Bp9<MOoD57BIr?}x`O9*zXx~sdiTyPu=-"
            "eH=d6-"
            "AowXJjCigYBclpjEvEi1?C<$`CCNRF&zZCEn{o2{38EtSY|VTC{@dR}Kwtt@ArVR&ZBf0deW={X*p6OfZ^MNbGSzarKDoqT$7zQu"
            "WVS!a%Mm{$|EVjOyzaeRev13^;MmO_C9{<Gn<DDO)hJ%132Ql|g~(wY@4OjX7BJdgU=MsgZ_Q*fYx_hl(-bx-RRS2u!j{*mUG@Vg"
            "xPVK<(xR$G~?au;}LFORC%>hgp>cJTA8E;y~i^@e58Th_7mimpZdZ(4NxUvh$AhYN!pVW(HT0bq7yx+DB+Py#8P+f@z|JmmL`tOP"
            "tn1W@lIrCE%^h+5CDK6ek##7MUB4cFuLi*CtRNZ!M4fFJG*;$|a**0tw(eSv^EAb_oG_ahbyH^DXq%55Q-"
            "BdBHgZDTCR@cS?!?sX!sQuqZGia`d2eGb?Mwa<;67goL+N-"
            "q(#^h(v>+N9nQlrMwby)ML!9jp{3*A+EvQfYzsH1F3hx;Y93f1xKVneqc6u-MCB!tkEYgJ~4MhUEmoJB<;7X--"
            "^i=*ojfs#kvq+Z${7-rjv-"
            "+g2^4v%ZHdgnG?)%2y^uCsC9Asn&GABu$Ix7{b{z}mr1~lqpvo#IuE)te~Q3GMjz4<aB;lKsFJKl*I0Fghdff{baR)$gia#9!}OS"
            "I-A{v6DcmIOfNClec^O&2FoDL&@bY2}6KKTbkZ0)v7S|K8ltF2-x}z-{bEb$n6<z4*;>RPHVQP0e-"
            "?BvX4O|jl0P+l#<6uH70cRe53fvNWuFHl@7^NSjG?-{yY+PHj6ndIo4mH;$@F0(fpLjyZc7V1KW{wxh6<2X*0v9Io2iJ*t-"
            "hoVjx`Hc4Wy_CV5E9gstaTzHRbbk-!Y>6H5khUM?P|j#iZSphRY#xK!A_c*8cMlXg#rO0oL0Ty=C)u9-"
            "h?N<RAr_!<kbr{G$kvEiA72H%4RK1QzDub6wyVUbecHEQ8ibHXK)_4o=X82_a3JMH>K;Y@M|%u8^H-B+VE!tD<s-"
            "*cx}<s<uQv74+S0}vCdS`2>x+<T^P_4@2#*H^tnzKvp5#qWeyN;uChj>pQ)#+Zcm`-5wBFt-ZtfaA&H$=p6U@U1{6073#!D>KC1S"
            "D!)|#k$!Dhd5KK((iz&iYFUPfxBBU?;B`cM4;UY*^VP>?LI5jKhM|-"
            "WYyC{%M(!vXL2(O+Oby#mpH}p+=!u;VEVbly_4C#loc!N^$vuMhzP)Tq@zEG0fe656>#s(qg>3hA4*Pb8)h;ow3<CcD1>L`4zDNN"
            "0!_e!8|SNhp@T<HWBN`+N$aJ?AY#%?9PGJR|<++R5|2;G-"
            "7A~mHd2_s(oEy57;gq6?)!r$awRcy~9RBp8LtoW&(0jZzq#atm@H4RXHqq2DvB>w*|#A)j!lO-"
            "y3zVJ6^ahBILcQ~X3jCqr&&c%{I+q|x@h$=*}OwRHnzF=bJpWEC}+tdV%WoF&C#5MWNSjdlBwM7PvX0YQ8k4Oj~S*yh92sn9J8Bn"
            "q^7o~rea;jo1tMYrk^Wrve0u}wpsk^3xD{gj&BY<yh+r<}7LUDJ6e)ungH3k(+@UjgNnN9I6AKeA|^HBb>#@KFJym!8frkIMpFBg"
            "V^9)%t*%p6Ujlb4kVWNo$zAxQ^=`WO}i#*5($eWvXsS>k%KTc*5Uxf8yWu8s%=S%EnWM2XI~8SExjy%W186p=6>3j;XDx;E=b!hF"
            "lNZBh1WMRnWO#F?%+6m1v<8%PrI++lbaa0}t*!DcAuz*<Ni4I)BMF=B@g1c6mrCT($LW$GV4q!VR{;WfOcxnf>~O2wRFtl6G20yo"
            "h|hX8jRnWP-3C+!%4!VK?JV>_}%xOgKcAxCIkm^Meyx0urOBG|nGMPdmJ-"
            "1m~<u?rZ(wBt~rL3gP#T@U3q+|w;X!vr;5A67YaJfUv+F?^SZFHh*I@_|qwFnzcew&TR8G(eA9RXt>9M{7ERXA8rV`v}99JY-"
            "{+3X=-XG^*cV0-"
            "f&qEL=;tpiC9J$h~2C1D*o{2iNtA6AOwoPQhdg!6ET2R8}=^rFrH?;5M|;03m$omY2g94rXPg_B&fr{8S$m_6K$nUxXkdqBIM^r;"
            "fH?H7TM~;AaoNcu^lwZ7E6i#11E}!3*=uc)18Vg5Bcu^cIXTkcpDQSkn@|Sg{VzzcAAp+epgGt6CHUvCGvBFt!0BmV`d*#87BVdp"
            "3oEY-4isF7b{N3>rIBhDz~rsDr^zSk`6li)VdM@t9N1XDtxrw)>V;-"
            "B4q7zyxsI)f&I#J>tDWxEf(zkqsGkNN%(~6c)7@SoEa$ud5k~g>D>9u^dB6n*cL?<(_sI`VY90I2Y`)jkF1aH3+l-"
            "p#+VwJ|!QLK9U;+7^VUn6TB}a1wn!{<qUsjMWiJ!N`wQ{wDqYmu8WW|i`30&#65yW!o4y%YP`(|h3Kua7EoY};m$F^-"
            "H63`z0X8f1Ztv1<u3#gd{y8CScWQ4kJiY@!LemvKjKY1aF$mye56%Pl%ytZ+hTJe)!SbCDN1i-"
            ")&i1g<;|_(Q;VOYwTfl>uEP14dl8r@hf`7wEr88S7c{HMuwvGr)9B^2172x0ibqE!)w-"
            "TQmz_k7Nk$@jur95g!HP#b!YZ0xs!9iyfXbr67(FBe6p7YWevOjltiwS<EV7V>Snjcw%W}J;0{Ue5fXCi4K}gdtC<T$PEQqzPbP{"
            "VZk??7n+ay2Aa2Bv7uosNZq=J_$nWF9N6}hqE264wZ7UkxXg*}C>28o76js;acaZSx22ad&&S#+s85P)_A<_J>iOa=GVGiQGjlsR"
            "-t8}UbgTOlOiI9S}@n&~1+l89w#BSS=R!vflY2<H{03r_-PswAr<c*)OcT~qBUxhe67m#wOyNie)nBEKsgErg@csBRQkdBCjn=-"
            "SjB8LY4!H`&~`9ka%$7D5jtptO{r>NH&}R9eykFC4GPscnqlCP;(1I#T-"
            "$!XRIF_~dH=if2B+qh3gjEmWZDE+GvRmHL|k^+I<uP1Cu(h7eMQpXrP+u(U)3K_^?@-"
            "~83Zmtk!ZmNsQ8WDudYh%pl&qm$FISE>*S4^d>sXQWGTK3PQ*MYTP;fsVF}--"
            "8d9V%+pZC6?I9ct=^M1>?&d(|j~S;et9^+5{rlhBJf_W<3g<!Y%;(Sy2>Uc@<!YM(0f13tt2Pwj?~W5P4-"
            "qPbnCL<r!WfObF}(TLBp>a^0*}09@qCY#H=5<Dz*_E8(%xdK2uRR|p6!Dv=ZTb%EJRT0t-"
            "HVGt_9{e(o*TTR42rcooZ4a2D8iZZCf!jYl$itekP9ZPCA(ZcH+W(xwYy01-0jl9-2x#RMHMYS*k0U0E$>xDFpqx>dIj5iYL;D-"
            "TnNDoSv)#z8eM?=t_`k_3ERUdmtAZE4dGbI`dzhDoLCap_5Spt)u^koATa+?kToi*Z)=vB~~Gb2Jpm7u8D9mDwBUhS*+C5ghP(}Q"
            "6OfygsB+HF=hJw?4%f=s9v1Uo9QHCEBm5yy(dhXO9j(5&y4Ic5UrO$qW`maZrrG0w9p-"
            "DAB<ewKT*GDN@QA(8S~QUxih)=^?0rNn1Uj%ejIys8$D4}tZ9`@F1z&|>(Yk*}c&Wx-"
            "1639ZcD4VZoFS73nz&c*E05lX9$RFeFtHQZ#Q0HSn)_Prgp(%6-#ZkWVMY!a_+p-J|H7wWb0Ll`TjNb~O)-"
            "m42M4YU?P{RAE~s}iRLtW=aCVn$rrg+TTtQ3KsW&}dVOBvw}I6T&LuHO3(`R|&6=(GM*kyuD@(vja9$t!2`Z#w)g`GM4sF5W-"
            "HqHLpQXuPm(uhiK8%2NF&z6O;m7)Zxx_nWDx{)`7dAb;ZZ%Pb`_etsK~-^fX`Qz8Vc`=YV@T`T~;LB-jvM)nTD_;?BQ?{<ExDDb@"
            "&4t&%zuqslqH1@!|?F1J<a^u-"
            "?WTV{LvL9pq}dWzg7tqgB7Pmr3`Neb)X)~(A30IE8vP0$j73aMsGYB0ULx{X{g3{Yt_oOWjKb2mH01vAU3?6vILD!~IY^1&%Wp&3"
            "Df?9iIGXLlr9+kz!G%AU9PbZ+UY1#%fABEa!wtOZ01!z)V!f%h!1qGN-AI^AUfDJigWQ%}7W^`M|UUbF#!+?L2Da+iLS6?xe6z=p"
            "aq`&hK)%I-"
            "CScNGibpRm$80iLm^yTt!4S~;xMCS2Y|H%0D0qpu|t%;YzCQ6nTVQ4AYlaz!JVn52{Sjp{Ue+Bk7*7fhj7YIgBcVY#8GXL{MKeJ+"
            "{!!<ftlM)MG!D!LXXjtW+ixhPw>s}ubt&<>tuGb9|_eAm=9i=dU2WOLW|G;g3X32J$#yU5%m90&-"
            "5$upZzJyFZT+S8lw@Jo6!qzwz!&^@D;B?oKu8CjtoO(5AkyCq2(2y%`C8esPwFCj%{VYD(k+wrxCTn}<NVSlZ4HEZQcyV$Dw?IGE"
            "2b&=Uq^>CFi#9nqB6;4=XGYTlEK87;279$9ktxYJYfCI0<3cta!C2TTP!d^yc#rMKZVh*BDRQZ-"
            "Y%K{~I6%84xR5ne+IjO^$<(Ap>ryOCi0J`?4C#$l>9^V=7Hk)|)HLk@kvzK(Vx-"
            "b@9v<MMRucioE?eO4^BwvO3MW`ayL)0|hnwlVmEK+VCZ?&rxP6^dTyclo1I>DLqSngZ2=zA2qX-"
            "_)4*y#z%%kl~^8}XyjidL^&rILVD>(P8+uWZDY_l0>SWJdk7yiyeC_N6UZe4_gxYbuKtZIwA~ViKSRqgvJ3HA~eM^g1)?N(4O98E"
            "+LfmL6a&2hw#c7@3NjJTA6H&&n}@DC2D59QHxNK>HlSXtIx55kDXC_-"
            "D^N*Wy4ih03fzR}qFuvBjh3TZIrNtKi5BY{R9fwlaA$sx?6*`6N&{u&0(qp_*U~lbAxT%=W+++Q!p?!Mz_y3O2?pK}uwuE)>gk2t"
            "V$bcw(`9EbJ*ktmE5T4Q&05emboM9>Hi};FeX*gYZuziK4M3r;`N$oCp>}=qF>2iDfL<^HYMRKLB;b)|IG2si%7|{$4xvYyDEl2e"
            "rNF&Td@@rY&S`3I5?uS*9z@O-"
            ")w=`wDC}C;W%qd_iR(vjNo1qffnj4$caVY~YBqv4Qc>g4EAcv273@bt^VQ+TpwBXfKt~BmDIU%~|Bl*M%IaF}(-"
            "nJfO@h6tqo%8PpeIWTM=;)f6z3z<vqqQ)E&-F-tNbRJ=JjFbFPlKk8IVlhna-HFVDfxlMDc1PiuN^RVz>U64E=c)wNM6Y{)fcaLd"
            "eY?Uk!RSj5j1c)0?21Zq~246*S4R+dBoGuM2L{|hFVPHR&m&zs$vte|oslv({9Ejv;5J+{sQ#h@HM>hi~QTwyEhwzTxcjra7ee_S"
            "&y6R{y3$28hgJ}W-<Un%-"
            "S}G|+<pgW8EWYiR$(n(@mY<j@X@TlaRg6U!AQE($i7J8ATHG9)R2IG<84Kut+9<=*@}!lnfIFC!>}VlcXN@=G`ZNz<;k*?$>D51h"
            "v+ml`X8~u4twnJ*cu=7ZgQx5xi9;IgNFDhdqP%qzYaR&<{983!A<-"
            "6~h17qxf~hk%e$|``z$hP)J`tU47|?2sDq~@1%~+|~mYFrn_O-"
            "d?zVzu3vbC1q!m6rvR0?bxBTE7)WDY3Q%x)4$)~7i@_*vnD`Yc;Z#W;}8G6%%s2e#ZvH=ZTML~O3XaOxI~sb|iac28Qk)QO)Trj|"
            "KCe66JNHeJqYvN+0Kc(u6MeO#`k-"
            "<_gJ=XJ)sy!uN<gTQt_=xTy=WT^HvTp12K3qr0MgkkR&;%|G>R;e&7jcKV@Le>$HTen=^kRR*=PfA6pbOggkQnsgCRa=D{RG=9MN"
            "g-Jc(cpvNgVqX`FDm$dOBz*%mX#nJ5jXJizr3AnY-xFLH8F<>LxYIOP-Aa~4}Ml5@6XU2MD)UwqBm*2<MRI4N>L}ZlXXe2;TUzLv"
            "2;0RItj}4qyCkL9fClOw1qLGLADTS3-5xJL!+6}F6h^!y+;STF9Qw3GB!3!Z#_oqh?2)hgKhbgku+i)@6~c-"
            "om)#04T>HLif)gKX7SgEB(tG0w55;h2n^D|tPHL1X8XLQtBB4;lpt0*HAz7Ygr=KpxE%t@#A|4mbLqcQ`=cKKb5Tny$Yw-"
            "$c(qNXfx#{yy1fBpf~NjL71Ei2wbZMNL1Y7#e=6+!9i>0MTP1Gj49mMHMtZU&G1yHJ*>g4GT;gEcAVwG)#p}g+<5Pn&kYtGRgD#K"
            "_#UY8<%G<qJ%a&X@_CeiZoi^sGHeQRW`jyB}TxIu7$}{Jg=%$jUB`S|E+-"
            "35r(6bY$;_u0`lcjK1^db3b@XfuDB!)3w>_8x_b>OAsw7_s9B4A^~$P|@c1}mSvQ>J`Jm8KGW3W30O2!3JT@mi(vgcY<nN=Ru#+h"
            "#Htva`(y!IIjDvNc}I@;Q;3kbbgO!-"
            ")~s=|{^B+xK9UjATcKaE(7?7yUR*_9v{)IXm6r%w!vX(SWF^lFK}AXaABD0*?j1<Y!O3#3@w5=qNjun^~&ijSWo@c=!V1<DjrHQ-"
            "qc406R0VG3+BdVYHwlcD4w`8`axR_(hRBVgM$HDp?D<btG@eR#w8YA-TPP`ddo?8!@L{SGNu>urqSFAy!<+Fs?>Sf_IsQSTVE2zH"
            "paCOTnbeVtLehJu2eenj%&ZFCBgS(7GU0RAur<7%vtN2ootyPV9iX5J^pV`0(@to6lmjh@-BBalU|wUJ=3BT9#r%DBKM857P-"
            "}6&c^5KAxOY)xyk5SKk{eNAtrFIF{K%VH(RV{uJ$UpkYw35;p?F?8%fs+bds;6n-1?B^OT5omSx-"
            "OO`%JJp>RKXoaLy<LawXrN07MzT^cdphP4|2o+$;Qwy_ZFzyjsHB~E*hOwDS-FA$ZQU{7(8Em)Zk?FU(YAT#pc3-"
            "%prxllwZ^*6vu2e@R)8443S%H^kaY7UVnU$WFN@$WbJ3);<A&cbfSgjCg)&^68DQEarqK&ZYt8RamL@JR`)ElI}G6bac)hAw-"
            "!U#eNY>5L{VS#(y2Wf3m5r+z+vA?$ogUl6SuDW5GhK0|;nvxi_uyz73y^ll`6Xfi{RRhfEj(SBUB+<1LO*LAFwLTOD;xYQgAfE=}"
            "Ht^3Q>lq@`02U6({sv+0B~vT~BFG|ph}XhvCQj^xZuSj`Rc3i=lxJo5pj`RkJ{noi@=82$$5h)8YBo<xMH2Ky@|b=$qA383PvMx}"
            ")RN?9bO1MCw)l!t0Y6s-ZRW#BqhfR~hHgbj(niY$iO<+CKl%;TrME~9Q-"
            "Yb*Qib<WZ)^6E<_U_$7%k<x);8M*M_l6B?QThVdKn}%gc2(Q$wX%8$f}tEqWu!{M0H2#Mes!EW-W;Y-"
            "kkn$pQ`5{l4Rh9;k!mQfDm79=sj56LcZE!*dlTe;u6Sbt2^S9Nr_UTp^4=*uL$uZc2{+JRx7+S9%L8n$e~~<Kd4q&Oeb|gldDs$0"
            "hi4K-"
            "cEq>wOUysC35j17NhIBcm@o`V0^8S$?L3fbnPK`<69;bw$hhY1cc}o3yZaIWixuu%@WR8>X!kQ882f6TO7Kyf%&=5j<XEnk(jBQG"
            "@TDCIzq`=-"
            "CWt2Xq|zAW7oL$88RP7)ScC13=^UahNqw^Pl#MV4`Bo|Htd`dDM}F72ESg4mD+i8^SCbqop&f@Qb|o<b=qsniM~%|>!RX%a$>L?p"
            "@3qjN209Sgfr2<TUyLT?=pxvtrECyHGP@%pTWAC*8U2vOt3DC%0(+f_b2O{*2+@7mPY9kCVY?1Q%;g%?5BnSa$(!FUMeK>K^~qui"
            "f$F6n51vR0!dpL0uhpZfe&Yqb`XHr;qynEg}8bc`bL9hG1#?Yai@~YB3+KV8CKeV00OcKZ-"
            "{jfzBVII^~C?+SSJ>T&EDAM<c}lrexS&7PveI2aN=o=DY#^WQEP}v%DXxnwZyLw*@+z3GBG_?&+cTpMOv8d56McBX;@gp;zh-"
            "7T|i6q%abs#J*}OwZ_xqb<bh6A*At`t$th~?R1jEjw9$&Y2Qsa9V%=C|%7N}~GJQMCF@e=2PgO^myqdMClxR#+xp{G%(rM1d@&WB"
            "W(HIX4AibJI^i8nBi1@Fg8(e)XhV`xM&4u=q%HrOe#svC+^HWMi_&BZmkSf+@o2AUDDk^)KBzCs&L)&P%n>8c1uPg@p#T6iH)ys_"
            "#-6yP39XLdWVnkC-"
            "o2&*UI9W!<n4MMh33IS)^NOfp>l0g+*$UY7UfQZMST4a;5e_w}DPc_#dGbu)|Aj|B(DHDGMjBfXL+h{|z%|usM`N6oHTT{iq96v_"
            "s7y}Phg(s9tj5GfC5c!NtO%jmXaF;t{eZzNU9xDPAUu;Qj}rWishv8xyx6?ojericCU%lI8_7`Oq@J;02;X=#bhflzB6^4;-"
            "eiuKK%ONvMG(XbtD!9?W2_3(QHCNbRZ)j1LfSY`vI39Vvf80Gf@z9sBOSN1xdb@qHTDxV>qCaXavN@;FL@9HGEd})+z*VfwOyFp7"
            "zUFLGZ3BqP3EQXpqIR&aYGIlSc-W~@e>R9NGeuli$7#e?JC0rv1gNLH#OLhP{R^fg6(+nO^{V?Cf8S0nAx5p?5%8Z-"
            "l+PsTsjH4T1u+Q6a)?!0@JH@hS#V&rlTeS4)K_(Q__>_VfIk4_^`wVDnB*xv$dgBmj)(&FGx!vpl$eY1YYX3kCi*?-"
            "ciIS#F|s41uQU}1^micZ((wIeb0Ov2RFmmbJkv47_9zDHESZ^Lgw&<2Y*snbYMRKX?D!tawCi2`I~rWJmkI?B?R@%&v=BqXXA4FV"
            "x?VcpMk4Ou(Qzhb*5V3g*OPKh=r(q8Mj#H8zCh#AS);TMJ^)pdl(MNhAiVNfCp6v!+~|RglNIQSXpojPav_~VA!Qe2t=<6zdu>j`"
            "^NRpl}YY3uNnz<7tKQ6!)}BR6ltp0GPrfpz}!%C!<eON-O%yH2Fh*le#ePAt!wjWHh!oy9mulVK_>X?B!W@ZamZkuU>~-"
            "B(?FTFB)r>AkT~J3k$~S4X3_kmmWVhu0!u}JqSslTI_9xQK}b1pKe$p{oQc6!+RQq59v}IoO<OW`!r0!xPwH=6f|9DLjWy92vRDG"
            "m;LSi4eiU@m{K2pc%*tlQMiz{)ac?3z7!)`oWib7`QZg`dZibz426$O97@}=#6AnYIL0%I@sO5?8nbIZ3LkkaDT$NXkg*S$ZOeD#"
            "I6cMHH@@_LLPGioGc3jIm#1Y?P5xgX<ZK;gh28nzf3H9t|K8i^77Y#}7qA@HZNThSTxIm=(9qhDBaI?8L)@Xz-BOZc^g)eGQSYop"
            "tl2u)#!M|Yj<vl!6YjhEUEagGc^%8alf_)Ut4P#CkqE~@PxmwGF8*79W@QtgRt_g9aW_6j$Uy6U3b?%6-"
            ")r|Up2)pHL3D@QQm`B7LJUBi_nk=%-"
            "9v7)Gpw~<8rB0NbG_V&ZTQIoG3O#H^ZZ0WGg4NPjrA5UuD$A_bh@S%QO`+kaEz7G_{+(ddg)Ls$IT*$m4Fpp*!h$Bs$l2P;b(RDy"
            "!LS3g#iRncJWRQHjZD$9bix!}S$f@ItOMkMU$pv|b+xRX1&f_fb#?e1We+IWisPoErTy~a!MGQLCq{;4ZV(4sgFiN(2c;&F5037!"
            "0q5I}t5G*bpQsJX#lKFPwXLXGN_xdKE$l53L}#D}>=IkzpYU9<VU8XxqCLouIXdlkv<a^WzN2N9WrLSCNA9`4=)Xnr6sK(-"
            "$h8I9tDGd$ugQ8ysd*g%b)k=oyq+flMHiXwFs9@^y+cmN^6EBJ2ZuMPnbyU5mASHnI^clZfP!j}STrb6v$I}IG1DBClK}?tss+gG"
            "&EOz1V?m)sA*v-z#SzsxS<MepMiORn(CHd>s13feB(YkPG^bU_o8nNf5u2*T1SD*(r)r#VpTtD!sF-"
            "<sxls3(U)IWy^}&2?keH>>zly`HsvXj@bBnGd=A1yRruhgedrbbbu`a@eI+|1&9*t6)<@HH^#mZb0`DyE0`c0Wpt7j*pEna35abf"
            "mURws`{jiqowu)AtyBl1W<S4C0<L@RsZg9pxC+23TH{LT=@OHEJ)AdGe+_Hg^n5qi9&D8rwodm>pTA4}xHS|grUwbS^6t2tRO^oC"
            "rH)st0<=+!zKw`2nm-SuHvq{@QVi<8dApQ}QE#VbY5j^F0%Q2%*jm2?IMTEQRWPF#j~QJR-"
            ">gR*2pF&NVP>2^qC&rC`iCkN}jJ&z$f!MaznhC&4PN4JfI3J;ik7-Y;=4Om*$*v1#6-"
            "u=X+!~wwzW8p>{XSHnl(qT+oaD!1AR;oGXeneQbZkeRa!3Iz%T{|u?=W0|ZT)W|7NB8E4_E~d)kZ8PMG5%FI90kjL208_}(A3s~-"
            "@J~Ij%|L;8Dj*x1W+t$1^BJ~VPM0Oqn&wJV7BZjfa}JO97Vt;q>sY&@)VQXp1h~oXqJi7%3D2{epd4~7$z9AA_tl|RW65mcA|DQ("
            "4^C)B_ElpUgXPO*j@Vuz~Rbip!%E@tJc&Ckz#p{sxp?@Io2qRm5VVnyV2yy8H5jxm7~07^-"
            "}n*&21Hhsj37T$K<vfu6%{HF|q*LyZ}mzrY7_T-_Mu|*ac~Eli*;Y-"
            "o)0#09^!WGtq%&e{3K@b87&_7QazQH}Wu^m)tQe$hu@;f`QE+))7cWYSN$%x!MiXQN<y*szP?325X{HMKEtwJA#oXMX=#jaZv}dO"
            ")IyRJDwVg6rE3wo6Pcr$qBGdzP=SiM>Ud3oMBi>X3kbmD0~^53ruD)%|>#5^D4A5s}gCj840%0U&9B4K`f+LLUbD3#!8=wsK!|j&"
            "IOxN_1F!yEk4ws${jzba4pM;0zRljWNgLxsZCq=c@GA?Xjr&xSx9#K)WVFNVt-"
            "a7mxzVM_Z$sh$93lr3)3AOWP&xanzfXm#L)%fpa&^@rXGYWH|=+|I--rubMAmoTSsW75gIJV&S+HXGY@RYR)dF$|MS`2Fkak*y@|"
            "}5!l2dwyy3J<4^2TG1iic#aSdkJ7}sZ2UrbJqjnTr)N3v?caL1_aDJ_FHow}dbsOUYAk#PmUu!RdTRM5Cpn{Ktghd!FI#T*kg)-"
            "BH2gSveSe%}#z_ZkXI=nX``+-"
            "hU{T*{FIBtHDr1{zr%9)@60>!T}?*bLT0=xD{EIhYraOCF6(q_YyLQWp2@Y$_Agc5r*ak;&jb89{7t(V|O*leeZ&7CMP%9oae8#%"
            "aU18ZD8MYkW(XE^maftddL)ORSvsJ=ZT!xz=!3o0TRE*WzDigl)L2#d=zXfDr-Rtn{f8?2^c5tvFlch|lI}Z|OUR%OY{HX<T))iq"
            "r!5)KQcc%OF14`ohPaIlE=d0cE`*lt8qXw;nkB71WExX?>%t$O40oRKCsCSyNg>9#*#f$+)l?spHIyykiI$ivr`;@US`RWBM^giG"
            "Md6i#3pJ%hG0N6HThkr`nWApoj#^b4XtkM`xV>(<o*(jVZrCO92NsO%es}nT2_)bWp$+vh}R#6{?~sy?9_p)Q$9z?Xw;OOMx3(?K"
            "U%9${eQbgQZNdMz+)%qg;elb)-as$W~v{B}VW87O+`STV_IRo4#Ae9#L80aZg4p-"
            "Vq^(Ca;Q~Jo`q&pq`@_0J#_=Y=WqtUaygJ?fQUE2cvdgo|1njgxPn`qYiPIe#kFMQH`8oV)STrsW;F&t@K?uBf$2&-"
            "Wd`k&ODGYrmhBz!=+jrGaI>Q>R?f`)!}D80h#f}5Q!o}k7973{xC<;D6hT^Kf6wT71hyJWDCik`~<41s~dM#V7}4JWk^B}J_;j04"
            "c~qmWqSHhj2gLtg|`T$FmT6iXNrmKSc^g@QwT@lb-C8;bS?w7RfkosnP0-"
            "tI#JdD*O#SNLltD+J|;V`a9gv}T)4hIn&xqhoSDAI#Vk?7BiSJ8CSg?=D;FknYqusv^69nLtP5V50g;6RBu3500^%|d)GKx$XHMU"
            "WHt1{)%xYT+Rg?iQelxAiEF_u8HLFx4lL;PeAEjtbNUfxA!1;x+&B!eza`Ow@^Tr0BMKICpuu?2KhA}WVP5B~!NyU9-"
            "`(*4Jbu=yW3~QmW!!{L|r=9D>F4$c6r3A&kx-"
            "@|e0x5Ww!0X^q;cc#*u}b`~0lp)rq8iSjDaOx(03UwB3Ysab1llA`A<fBc%cqujW<bYb*&NQxTj0s96$gRZ4BaT#WZMsdAU61U#l"
            "Yj8Tm!ugO@Lr!9c)ur34;TfhhWET;w9zlH9*I}d8Ccd5a^*dJMr-"
            "({7kCV&uQrl#uS!zBO<`c#_BY6V+pDwSlBqmCNI{Jy%K}_mh#27JV`q(nP`ntgN=th8=e)4LCz+IODa4a2<2<xpTor?tcUYrgg^{"
            "sb4W>p)T}-Ry+G7jcvt~PGLe|{;k^-"
            "c$@HlTrbVF|$)(1SGAQ8H&nfIhf5u8`<u%pVW$$ns$;+zN)JeiJdr?<P3)Nn68kx*y_L^U9W<jLa&x@pdks@%iFjM+ywaI{JTOQ5"
            "#ZD7)0;}8b|j&#ml*auT!v`flXDB%*xOk(6iQX@+ioFb(DYkjtQ)R)6%juWdGfqa%<w~UbGmI;kbWWO*bZ{{!ZDBHGewj`|GR*dd"
            "Jy;^Q`dU!QO&AHR`bsPHzimH+9>U4v#sLZ=eK4E7}a0F17>gMzVFKQ^f7B95{$=q@~ECoTxhlV2jiDtJG)Sl(vSN2T{S0)(FtFAq"
            ")6y9)~UDiQ81|zwxf^8a8b6qwyvRXh^>)RZz$0|~PW~noz31dhA|I-"
            "tk0E?=2tZ5#JY&0@JcC_Q3XN&qIU!5d4LBmc$*!VPQEP~P8s@ILp%M@&J7)pPMUDxVYyizJh&jJibH!cL*K?cx*6l*cda}tD^iX}"
            ">?^J+o1GIGapaNV3niBIcdu|bJC)E7qv8N;WQjhl`4jROWiNQe<&QoLmAnb_>c(2%s^O5n==xS;^~%U%QMhTAaQF7kBQW{pe^XP2"
            "`mzA0Azsa0V`zlWssJe!}uH5T@l94dpLHkfyHksz;DE*a_)Ivm$Xpdryg)feclHI18raCuMPXAhB->u_z-"
            "x)5u#yj#I#mA5M58e^_6s?C2mUCLm@YFCBndDg%tnuh5eST38PLe;SXhC6G?*xsN=3S!WI-QdW6(lVx&ih~Q+0$yDh`T3+S_-"
            "+IHk{ZxnBUsdn=SI+}8@#;6!pU%!8D-"
            "W<ovVCE%2>ru`6=9q%4fis>>4FQI~ojc41;^ohBB2G{OoF^I4`lPY@)N&9DS~^qsTNYe#);vPkvMhtYydI^D&-"
            "B6f1Fh3A#t5C=~xG)(tC6xe(uJ?fE(!R)Lp%GHbQunI2vM*^-hQ4R>?hWNj*=Fu(zG!6MZj)c7sRHg5~R;<Dm|-"
            "tCjG5E8$oLO7u@Oww;I+=-W-"
            "Gn5!F=4bM^;zd@N*abY?kEM$|h#2$riXn(le1qXdB#SSZE>gi|ynk|6bU(cCI1<*aJ$WUi+3+1F`8e4;vKi2#Wmn(B*%)+9&cS6O"
            "E%nnRW|f~YRC9!>b1+v3!}q{@wOJkB00}I?n481f1Wh7f81$9eAr@GG6A95{4$E0`GUi_?dxnecnOrwcnq831a(KyXyn&oY8}M{)"
            "t2E^j^%o|5U$o4sp}edi2d2UZ4SMWsZ^bOh^z+Dzu@s%!D3Xr}5w%Uw0M$oXL#N8$Rf7*NfZyBSn&@68rxsJ1bb_rEJM<PdG2mzd"
            "*l@apy6cm6JZH6qAT!+eXmJc*6)QY_o^{Kbd^4Ln7)Fe`$_(0!c0sgMHYcr&c^wp0j;`hu#b!=V+yZ1ub+>uJc9Fx@W5LSHixZCu"
            "1TTM#B$;V8*oZO$7!|(;F^^ZOhlRA^QU=yeUa%7eZA=DE3a4o;llqLSH8AC{RxZYhjUlL+r6gpJwLsfW!yu4NT5n-"
            "tFT8`3$A%Pi%k|~NBj~QtN@9(Uvw%EeQ+nt*8sdWkE?yG=S@K|$so@r|Y+8{!g#++L*;^WVNAhxPXPyutb2uPBCawd?L^O?NA{h+"
            "vLED-6C*fJK*|z6~2>_TgttrlyBXxnJ?aVnCgvoU7i0Y`#9PjDEmpqc7$>JqMgmeBHhs-kjg)Mg;^@fZtB43kXRi&3{^aTV%1z!i"
            "8!lL5Yx#TAi2pc&~OAS&^=xC+`ENUratY@VK3vUFN_>eF3Pz+&c$}+wDSxxFnGrVL90+R0(4$USh94*Ph>i|(P+ttUyEXY@mwqo_"
            "3Z+U@rKo@c19LEvK2c_0oa~~wL9_ld}Etjce*Jr;29FvZ>o~&yJ+QScxuN*NH0~8ZNDD8L|fl^pzNp_{^g8Ch6${mLRUD8Q6n|vH"
            "c19%Z}O%Sb%&E^~&pRpVL2Wf5O0&C$_x)@h9R>oC^^P;?b1&HMH&66dYAq(7$0Vy3~9jTuBIBCM!Cbr84aA~X+v$lK6uBroex{ER"
            "n7W<Rlxmgzu)s%zOx+l-VxQ6pbEp}sOyf9m7bAc3_(Jm{`jD2owaa@qsd5-"
            "$=VY@{{4$+`EP2^_ZSb74vaf?WO!TI0O?60q{$r&P+V9lO`cy)`Gs(f`SADeQ91>xT=MO2b%&J_MHg(`FfNd?kvgs|8p!_57XVun"
            "RBHhg)-"
            "X3O=3SmI*jd8ydPkoOqL&)U5S^V0;8e!&2<Z7BFTP0Wkt9X_7*83)cOdtIYS*7t|fLtEygP2;w<bjLHPO39_!a%^c+HDo~2$G|vt"
            ";X(}NXiDxmK8#o)Zlky=@!tJ4R)XA<OC;RI!Zh;8>WCzToNiVcX$L}HjRH!rfb<NlG9m%C)hjV*X{}d^r!rH8pnZ(Q<Gk24q7j{@"
            "8$5bkxJ|m#FzW#kg3C61>kjM7FgB7xaAhNpqUn#X>}2JjPj-)wLcZU|9A|<TII+rl-^D(rQ6O>Pn3*-"
            "@d@Sp@N~&ZbYM%Ktf_w+$rfbWK(yDw>P(!opp8W;uVG|n?#>8p$y)tkxD8z^!a0v!gmG<n|d!Uv_o;A%wH;0$$+OkTv=6MhIPdaO"
            "Lb~u8vnL`)VxzL`u#$(6q!ya6wRMUO{w+C#BML@f6T6c+BG*z>V@IecgBX|t#u#nyrKgp9R|D|xroP2Ro8f!#i5`xq&N%7KBQY`f"
            "1M>T-"
            "(jMEg81|0uk1b)gO;)7o0fXK7$b5yE^aZR|B5i*gGrQxbGZYe9vO5tdLCr87Zyt+L?KDizYu%<qs)Yh&}`i4y9Li*IQg|{;L)~!p"
            "JN%hQ;eONba4lTi%&=P4(U?f;#sqHrLcFmb@8YZKjvo=sv%0qF;Nyh(az6Ci~aCoj7n=K0`xaSH@{e~MZ8nZBxg17?22CYs!yV$M"
            "eJz4m`(-5Qk&>(DH>a+Kr14(UinKPg;v%F&#WbhmCzp$i8E7Wbsrb~M*fUe2Ysm_>~GiHr!Y&ajUNmx-G2V*3&k;ms}>MoHOOF^-"
            "a4+C0%bR&0-"
            "SbQua2`y`#)J~c#!!o&`0MjNDX#zp4;*&5w_dy*e9*k&NGadADQT2*n8MVH*s5MRKolUkHfkf&EO;{hzObjmYzl~!FNV3J@Ic_xY"
            ";%i32BQz|4p2z~wE@X!D5BDoBPlzw;L@)$SHhB4-"
            "#1tF()*ve;d~23Qu&j_J53YicHPX&R4xhrZaT6?hz>5pwjB9`$?1Q&6j**X*@C$Zj4d;#u18R_UQVonQ9I-"
            "pV_VQ!WbZ1>#i*<#JbisUVdX<D`ELwvUaADT;#Y9%gP|F&A8^p{Ur~_3HCCgH}8vc|_mdOWcgiG==7uZJ$mPYp?0p!ex#eXMIb@&"
            "y@7*t%V&)iD7X4s`~n!F}*l$vbDrMLxxI=pN;iE+*0jT*~vo&DR+ZGC{GTc5I#3>l9<PZKjuw-twolY$-"
            "l9LYdnO)v_jIa$LroaiKa+VJTbZ_;PwOA{?#naZTDlgYG0Fdd@L{@`jkA(N<g+*7#iu}Wskq()R%rhDLWeH%#wr+FRZ$Qi2Tb!U?"
            "XW{zAmyANR~2Sd$};W0v&GpQmu6G9j*E5=wJN)Y-"
            "k2w}2709Zof5X?E)(Cs;4&ihcc+Zw`2B<n`Q_tW;+!)T|DRKskc8cpwHo9UwqEsU19B^g~`BqtQ8)fx~8Inx<7vVq85<82r%Q*mk"
            "`uS@F)CAU;GyLq5QY}T!TJztR*({^Ryad1a(3wwWxMkKOjx#q<T51q`DRp!N%#Y04H4Bkvrc4L6-"
            "8_I*uM5kvzD$=AZUeb0#GY^_-(dLq)E{+L2JZBOWnt<S(4f`Gkgh+toCzq%^?E-"
            "LA19NQF>?iBDT7%3O>*==@MKN@LBn~=Ft^rv1EPA<U!@&l~Pa9_C%qGZM#dPiT(1f1ahR^HDrj@y2_W&NmxfV5ZX21?SmAD@>iLr"
            "9fY~RUW>gH%xZ}qGdS;?k{5@xHV+vd=*d1)a8H?rPYN*nuC${!Xo(;O6f{bN>zUE%OT3ioX|j<N}(#WQQaYnC#zHT*U6f{kLh(1I"
            "+^CO}fWsWu}W2~qMBrz=)^o||DET-+-EWa)WLLDw|9MK|*{%+O4b-"
            "W(Ry!OS5Ix)p>ugecQjvNnohbNFCE(m*XO{v7kD2qL@@kI;c#qZHa}(J-"
            "1^5zsT)s)w;pWY~*jKE!sZz+j}Ejeyn6ojiD<g%8^*4UMngiN!To_`THa^5&B!fl|rcvk<@eXNtC7@R)k;Mn4Ul2>6j2MZuLfs0>"
            "_Z+!PKYX1SaZG`&5NDtEyX@s*zrB6`Cx1WIm%Ii;YdG1^zk6836v##kFZ=hx)X2(UNCRdx4HYJ6VewY-"
            ")%`aO>6i^jv&;t1A<UBYScC)`G4cN_6ibLJ1JsS-hAK?uYRR>GC%o8Xv2&M;bhSgH?>9kA?qvf)uh_|81`+*pmUTwNs0dzn_uUhf"
            "!ZV8=agJ$c7P4m)%nzRqCca7G0{K0$HZhB<_S^!qH}WLw!mbcjD5D~nsga@=%+m&QvX?IASPJVvg(1`Oox81#A0Y~pK{r}pf>FtY"
            ")*8R%uvOK9%F2$Lld?we!Ng~d!OibBQsHs-"
            "~D?wh+QBA_viKztf(tQ!E$F(^or(S_~)95deqIfX6Dl89_zw?q~(8f8HBu)uOfRC5_)u$R+8_i=H#HIx$`V0@-"
            "|aXuP{6d)W#puivtT2tA9EJ_<9X##<<qc>m?D)aT@z@2Dz&_*iaIf%FN3bH`;aLZ~)595_OE*6$$N~{c05$hTumSaq%sct9DXMdG"
            "bw-x+Cg{#kNWtSri>x38xvWg99hRK-pK|Z0b7EEa@j$^bD)*|>L-gR_9vM=3jU|Hz3S^e%iSU-wjC5}Qdxg~MFBVRbkN=O}uJwS="
            "cr|g_VG{&c7cAG?_CJZ)iW7P|hy?bR0Hr*$xWjTH}jEuxkNs01o^UcQvf_8&4k}6a@Hd5na@`iPLkbXv{u@kCJfE=A!jW&JsBym{"
            "b8A?S4=rO2sAS=9qmbhV=;3z)J0j4r0gJ=b+W+QVJ)N&99kAN@O6n%N)Hy@?vYGjeosjGcJlgPQoHhIo?ESfVIIQ3(A?ZiQotB`W"
            "z-"
            "v(iO@wtu3y`OvDz%t}tenRJ@8JvBT2M3)XX|R*{0&x$>AQqA`I+_&fIhu&9JeJ=$rcSgzCZq7ZuB(#85E{gA8A)v1*@h1c<%Bqhw"
            "2kmXu`0bm#?VbP|GZ6DGLD)~-FgnVEc?;%1vzVsy{1;W7@Qf-(0gv+W5sLE_=$IqymHWmY#oO-AM>}UBp%$eVU@S3wn2~*1&W@^="
            "@`_;**0pm_d<2yq((}4kW(_@&$c^wkGNGCTI(4m%Rs%4i7JNMtqUurX0-"
            "^&fcTIFlO+;UiRhU^$^F8PS}HOOHYI8%@$}H(LEA$;1;fc}W`!MwMjo~uDaks2vCb20ksXXHKTY{!ReZ^?I%(qz@g?$k_~Awq5j8"
            "nuCUtaYSi??aptr`mZPEc@lETaJe)N4}S`^LO9MZ^0?B%A8%gP30L3D!+&2&%mzXn#&gC&z=d+Qb!H}PQCmtI`GsU&b_7z6b*P>#"
            "baUOR4V;-"
            "z4hC^rrVo5C1o&N(au6b%wVHG(gcd3E82|7<X@=bY93Hz7k>7dCTfgUQ92Xq_uAG8QILe{*JpjlA16qsX+GhwEo^&g@KFhtINP3%"
            "5~Tp$2p-"
            "s3oQK#=0|S5}O?LIil6*T{wIqc(P|p8?R%|+2J(oIE1k8wO6&p_#zH8^Jefz1{uvmI}&<De(PZ7m^6IJ8{=1Dya<z_{ziknEeB1f"
            "j68KV8~1#{tm0PbJKA{-"
            "$AXgbGr7LOv2??O(%(k2S%eU>VNFF)oejp%Dn}dxvy`n`+$hbJ!d)>mf{kLBobq#ITcpH7yJSIC=g5&XCSbqtcWPo?`eGzZA&tRK"
            "%r0y}K2zX>R>2fp&W8F8lh277OhX-PcrNrfgJ3_Hba3>u3alE^>a#NE>%6lZ5cOYp0o};rW>Z2X)HrMGz2{fd%XCxb;8tI4VT1~8"
            "@;7B6?3gqZqlF;ZThu!>ptpp|8Zt4V(xY)YdiCv64P_nIh_}hfLsm}K$s)ZFoMN-te>v_wtUD<e8=qlj6eD{}hW3rlsm*D^mXvH8"
            "v4U#s*mURkCap|lY8*pD$^R86Q=-iZ*h{S(U1e=;3J_M8jx>SRZYtG9rXhB#vYn+5G8(*L1D@MZId=6{*-"
            "ZJ8g#8JUY^nVf&%l)PoZ#rKin6?Cc4N?!u|6wy?7;IG(ECaH>NvK+tAUq2D8r?zrbTEXt-"
            "o8nj!RXNsps}EZ>iqCLZQS0lg4};p}&|#Hw!!Kq4=mYFq|ZEYZEJFL!_BF6SIJNg_PAVUxo?Os0_b%WKoDM=y%09JYix-K54R}-"
            "~$aqM~r=8r;f88QD(0CNc3j|HoPLw!t~=SS3^ytewSIdF-YAo(;DWJCSmn#MJL*>YMzDKjR%-"
            "jxo46b;y*ajk?XEG;K=OE1BdV$aJR(629~g_sCiz`G|xF(i~VO1n^w)wrN!jkGKvH2$l&Md7+!mm2$37}znK*EN1-"
            "Z<I>9Ca$pwl}v3nLHIi*~+J+U4ZBi5J0PE8Ft?V@f~Pv;5@;@?g>^PuF_nw_*1fDI7q3R%jekbF>-"
            "kfWLS&~$3(z|j>|<hvjWCp);MkzSz!VP_M3r#<f-XqpAu$K*$$ux9zyPm-gBjE%Dol6xX8@eLT9hb0E)>y~&BFPjYWB@C)Kx=?FZ"
            "0ckSxpz~B&P??1^t0^o<3K^`_1-"
            "n*=pqv`B2!#ZdhaXiHaTbq@MI1L9j;#2*RaF>@8ri8YbCgms#>pXIEdLl3S>&l>)R)n(QaN{(V2RdoWvkF}OqJ1EtJ9F9hgW^d_V"
            "z*iVn9mbObv-X@A90~+BM=}X)@NBO?Ktk$O}r?q9=+5Ax&lpz2~(msrhRUwVfKWm>PY!Lan76-x-"
            "R&=~8(PrdIjXvM5T}i877Xc`cc-"
            "w=nscq7s^7A?InFT>BIvvRavXd~xux2aS=eQQwibl{7znfNu%&eZZzx4_zRfoP$P&redU9=A^$0qO4-"
            "Y;VWp*+0EhU12DVf2St`n*pzA>Y-QyL7n@?Q*Sz333dQfC&lQ|vsh%~_AkhF5UBcKI-l`RKCPmAH0gd4#xVT|VG^OTr&L-"
            "`?<)Ycj97oH9z=exc(?csWFE6A{iIZkzHc5}7oQ8~XCg?;~@H)Y<ne=XKpIta-O4F3%Ve17hT*uika-8Kpo0{baMG{-"
            "bK#wpc)x@tk`t~WciM^0|QyjHic<F2_#9HIrhA|UFYU^By(5}HC3dC^uA<zRftKyjrt5(XE4J3wvv}$Z&rwyTEYMx!D0v5F%sLHF"
            "9EN9TJ3*+eK@Hv~-6=sKTnrPk1dDPa{pdl-8Ii-!1B1FVJ`{IdOSv-2eWvXWkufl|e8NUSDfck0=rs5(t8C$kQ{2+MBG-gJ1&w8k"
            "Twdzy?E9zv#AlQEG<6zE%)Axq=&!Wd-8N2>-"
            "N~|L3@s0+!M4(fi37qzMj2?wjn~Z)%h=@VEz_VV0C+!&SiF)h}dcM}NNa?^%3x#3KG`;$4XS?vIQ;3L>h052A0*=ZZCBKxk@X~Nz"
            ")O@WjZ?U9$@pv;8-L27*+KM=!JUeJ7m&J^!ybX}{ZZg-"
            "3QG)XwjF&fPCdJDh*1ne9B~?q(Rwb)8yI**JUp<Ort8h!46;`#?G?~|pN<WXLoi3qT7bezVj7kyse9r{1j{1=P+RJ?=)eWg4l{j;"
            "gWM81i0P9%?v0c@SgsN!-Nywn3C|lHL@uH`v#IF5)1a3UE=<2d_d6{EC@*I)|3c8^pCiWO6)gMXNJT-"
            "&(_juS#7iVF+_n?m(dtU4P%9Kn#7#sB9U#<tcf>~y?sg_A>Z-Xy17(^%2KO+4CR94@xbV{729LMBy#<nZT0q>-"
            "!^%Opce|C)a@?+)3o_jB7T~cW|uotOxoASu)hM4Z6LZ&5!1;tORJc#G46>AM5!`iCL{@84W@!`Yac(XZvhbUQo2XTSe0Bh@+79!k"
            "lV{R%7e*WR_BrbCyN>PS#Os$FHhsXa84tFlhs@%+A1Ql-6NO!tHXazeV#ET-"
            "e#mlS2*Sh+|O{CRZxM8~SnHq$F^GWRQSdr7Z)Ev&jpKc6{=F+JhlmL+U%qpU?TwLnP*Y-"
            "@45dqKwh<D`3og=_hn|M`mwQ!a7zy&PXxW(x3r7@(ATH4K^%o>wkDV@ru-dvt3iYczO=t3W*7&eS{iPTDuwHPbYRX2@eTh<srn*}"
            "BeNN%$iYwM`2QAXXe#nq_8UPmnwf~Qx*NW>M4P;qNv_LQeN9G!{|A}}dh1x={a9|%gkJswY`yXcG+o4#c%TNfZD@6}+Mbsm0%QCh"
            "gr%8vnS;(%mZUl@c@Rpb*>v2n{wPc%X3ldsO1Sg-"
            "2Et1ECMwUlaw?l!QzZMm)LCyDj~R%wtcl4}<dJ#!x%aP#tdxaYxzC&+BveIhyxnRP?!72?@GUH9WHH&Tg?Zcb_U_ROuNp}-"
            "~1P2?q7`K3Y#LQfXZ3nf4V@&W@SzKt1SYa`WsTpZe?tx3f_lUgR&kYelRNH$+Qvjn6>ljQ3ao>b;MCOMh$|EyMoF>9E7o67P*cn0"
            "YVS7yeL06;c-"
            "iI<4vT0U4pfao;B;^lq_p1c$Vs_5|D)0YspGIf#1X|wfsqd3iPn`$H?xt3)N4P%OzOyXyC`*X|;`Qe@6gf(691^DMgq&XVJtD?JR"
            "Dl|`|iN`PKC1Y=+Wl0NjNEx(@L6Es+hB1?)oi@m3qO+GkE%z_ctzKN?k2&MOnWe3UP`BJhFR6eJefA~vxn*8-"
            "!*6iqZRRN(UM&!ew(?>dt9U=#V8_$St3Sqes24B21RMW0=YT7dXs1mMVKsb~*?Fppa2$Pr>W<#h)$C+}wd*4}Ma00`N~{LC*%QN;"
            "4DZCiqH(N?s|=M)(seCdmgaI+?;>GRvNJ)$J6}6y3)9L~=+?xp08OdSWC7FWv!l%dMIw5u%{XG#Qv-"
            "0+rEGJdS#C!pWdr=U@`k%7O^V^hRdaN^U~@`VQ;R!|6bolz9UW5r$}7nf3A}qFwaugMlCgdI$vW%>w^T`U#rZg+6^-"
            "nUOKQfG26%L41QjXBI%}Y*dD^exb6Od$WmZi@^$&PX$s|&JL#ui8I^L89qb*q{#b81@iUIr-Sv-uKN}JNuMI|-"
            "iD68HsF1Hb+rqOi&YK0>$%c3p`b<7W#pDx9kkf9|DMr7p1PHP)%Gh?R_vBbzQ*a-0lAfZXkvl`LVz_L(<$T-"
            "1o(N*&{0%OTea6CPee1i~%#j9Yff#ZyAO4Pt}YFZ#HvqIV`R0qCHu&JHuQ(AHXl|I;$$u#=yCE3WfVjl+sH1X2)dD&xOXF;MdZE^"
            "00$1U{gZlFWrioD+<K9~hpNGG!{d_$yGTd8^I&vJN_o|51bIyR@L4X$ct^Z;22Ad^~rrbBh^>8WO_GiyC-zE-"
            "6nEvSrJ&o*mFW4h^}P#@F*&gzPaLGjF{eP#8sadfZEpX(BD{3l0BYR_wWi&S+5{z|eLf3<E5UkqvBtPMat_uNO=VJ<!4(VSCTI7>"
            "i>9sC3N<;oDoq1M!zwbDtj9b({}Sr(!}S-"
            "uzFa(=NpX{^9NZENBB<*AiM#TxS(cq1vnKE39>SRd=+(N$iG(Gklk8{gaF;c94vWi%Kt{*ru=8Q(Ij%HVgFudA`sHN{J<GBIQ1ni"
            "Ob(Gu-HASE_T(ZL%S7u%CM~hl~lsGj>j>V^}1cEGG=k<RCN1E<<+Z(}UudwHb`xHZ1K-"
            "Wd;7RGl1meroqRMi%}+u<CLMKCdCmuKM{8Z>+Q5LDV;;Sn8Q@+gL~CP6H(-en~iXr<#d^-"
            "ob7m~ut<W?9%ci2t&KdejVg(GO}IIO-BkkhB1x9FvwU{t+~48yr%{($)kYGH(#Qdcdm9$T)C5;1AuD94k`~x9xg@bxq1T%zk6{Yc"
            "yy?{(vD%lc6Ou(P4h3IA7k2&E0n;*aZ5V6+6xSMkwuc>($!oZCGa6ylq520D5v<<%Xe<TIv%}re6+4YtiGzN<sHJ6cXIm0cqv+G*"
            "`c+A2AT>$GMs`FF6jQzJ<lU(P*Ue#9T6yhWB3DYEr52+bS9I#foEA1P`DX@{rX!l^P&>3!hdW)KE+Ra?Sw!N<;o%t{xdd$ONl-"
            "W6ME{fjr)*pU%`&)_D!1M^h-x$t70X&7+i<2J`kSS&QVAD)h6|s-"
            "kRqnZFvm8DK^4ZrrzzMh%Z!0Jqbx(dgF&>x9S35Jao3HFZpqhH-igPXHvf0sFx8iZn>0IBb5D%IWr(#y#Dw^~vC?a$-"
            "J!v)nl=ZDd9w|#vPva0>5~}MsMbY6GjQ|I+!vv$H<y+hW(6WT%>yO|prz3Z#{Djhg0jO)(uJ6Hnm=u0%<x>w_Ut-IgTuyp@F2r4g"
            "f@eH?<VIy3-^`Lq8Gp~$fqB1p!qzDHLOf4Oz^j*;a2iTPRxh{-"
            "#%_~8RQd+TPTXvjae7Q<XZYu?GcV7E6pzDwIx=nNGYFLTrAP732bp5#X3;bc(<GfJci1Ql{5Yl!pr&xSN3#NMrw_T_nCRqbz0=7G"
            "Tq_Q3I^Ib6G%eY=9D;g1mF)BcBN{=%SyI76sJmZDuWU9v$O;?R%aSBY@E(DtMV|9Qzhm_d5MPXb(2ZdKWS_Hu!`);8mL8WdGhZWI"
            "cp`EIOAmb)C&_$CDMNQ_j=25>kA}_BF?Q@j#E7SMx<G+bgP1q&7BAaafB$kmQ;yh5ijwoS(ze)4dBBq!*iIUH5eAGRdik5L5*y)M"
            "QRwM38wDK#xO<VCu+$_JL$DBD^G(KM05vBI+2<Ese-"
            "NLb!`J-`X^7%&Ip&Dp$Oxw$$*VJ)mVNbVAL3xoi~nnIFD&ETE=Zq9Iup{Jc`Yz^q26I!xk;q!+Xqm7SC*QWpi-lHELelW&5QnG1c"
            "lGtj;GDlf}HJ6yhMAR38nm$-"
            "2w(#d!RrRyb^e!DxJJ!!HC&gHcu4B*$57BZ^eoAuh5Um8+{`47UcDFUchy2?PXjoMtcap^=Ihts=7y1ToM{CcRZKhs=+SHA1}9fo"
            "z;gi>9QtgPEVei?Z{G5oDmMr{*wfEDsH7e>Uuu9BGE|R~s>WcbhOWm2&bnC5*&7E4@KdDtC|!+VGrLJR_ihXS<(Z`f8?yjWO<_84"
            "!C;U|=!OhPh16*x|3YG0H3|R$36-"
            "&|!RyqZvpuoHy(6e>Zlny_y`?b^a>@A_x*>M08hGcXjKVEIEb&T?~uShgkYBCWj7Va)y{0$~Wl0XFqH2s@`>E3L$(U=RI@k?dqz1"
            "U3)ER5>tCpDmojRJ0wJqXS%PD#<7(fx==$50IH5wi&-$5ntJAJuuwmp&Q&YtdI!W%S=CLbY9Aum$MG_Ib2O=0H1&V!;i|?{Abi-a"
            "kpw`KlIm7OTVwzuKLo&#j+e;q!!#D>P%j~f9f>J)*`rs?q@9KwG&sacMC8)sa{7$C*VpHRW}mlw+_|s`?Nd9aPBZJ)sEPW7C)q}$"
            "Xm1bgrOW87`CGpAslE1!uMi!=<CYLlcr>C^Ps^w>3d8%L$p!y$YEXC(E%R0G;}Lv4kc_-"
            "z*sSoTFfGjyYh@vJ>IY4}KKL&M_)cwmRF`iwEL0)9)b0%$t+8;iH>ehgC?^3OxM2J1LZ#Cl*FL{mlphjFiWKV2)w;CRxd<`>8DzN"
            "5HGJWgD0Nui2mqSBVu@}k=D92^I}XPWA;JxZhZ*<7$-5F;s3q8+lWawYR8_o0&BygRTss*`2imE1443LYF+Tybi$GGU^B~YD-hA6"
            ">2Baa~H}~Z6Wpa4kYmZ~bp3}8FG52L3#RQ3+#n9O{;yerLhS<FTnnU!YKw@B*#^<NCZmus;G`GJvnCPgQWlAwHAGgRMy0g<<O~Zm"
            "QMTA1eoTBpiD<D43j#Lh%G9^TaN?k-Qx*Acq0iTsS7EC)J&69}_PZUWxC!3!9zv~Wq*5D);uqp1MP2-X6-"
            "QTlp*|YDH0qFz2>XMJiT=Spfdv(obK13Cfl?N++z1{zte{#NEEo}R4`4|lHYOXC2k2?QT%k^B87X?c->oVS<@_S1+&w-"
            "dHn=zMFyn`w_(`(Aw7upI8!$fFbwmd>iQVy;Awna5>TWZ%Tgim=*%^UCbwj^8_7g@sa%3>9LJnZ2G(7?ecH%&ojr%wii8G#N^^>V"
            "?2J<IZOp>RiDc@V7^h3Gn=OiWjT{`9FAig~=F=!&v3vOSh+p;SwYE7f!COqy9ZCXXf)6!|Jsj8ara`9@4rS1`ejjIF^5h>pkZXh|"
            "GUEdT`z#-"
            "37wBU43Lni#~un+a<#ggDQ<P;D?&J3BbM=7$_gD{~CYW;i5CS1ME*@P#0Mz@9p%Z4uUO8sv^%vIBL6dVLXy9L!nW8i=E~V!?W-"
            ")rIbYiV3BSVF&5~7N&Z4(pDxB#Q+s0PTO*&fHLon?kGdH@JV0TfqnL?Z7;+@?aTpMScp7{4R$B__BJkWc;f~L&Y*qHeOocg*eQ4E"
            "A5E1f5FOL}kjyb?DH;s)QB`-+l3hp;7qHhZ+szwwC!V!K4C)Om?Q%+t0mqMWMW95IJ7(QEBE+-"
            "SV@!EyVLJPyh$8)`+M`6(jbMQYBzoZ8Z*j5^-2qiWQmlf-"
            "3nHY}rplJ{$j0=(7G}4)H~x#NiG6An_$@}oy@;ouyg1(_;{6leF}a`Os!{p%-cI~Va|=R>ha@{*?|EDU1^`>%F-"
            "1{zm&~I??xvtVjhd=MJ{ugRF2j1K%7-Dj)HJ|9JE?lf!r<I}LSZ2bp7)Z-=iY*Q#w7p}tp-"
            "=gf`WmSDkAiKyc5O$`y2yzB9d(|R1HQP>CeJUjlzdIiE{*a*JChVCQ6fC=&?-"
            "GCNkmZO74#EN?w8i==Okpi7L&HO>SvXfLn0mii2YL9KfSpdrhj~)TKIJ2{W(bHSU0at%Y5b3Xgeq^>pMH2Q)wG<2PLLq^+0Qwv;3"
            ";P3Qpw035SJ><V?OKXQV*z~OOBoHjlrYi=<2VzGeNpJg%23^HP6dvP(?&BFF-"
            "y=PDr@7ec=YMcG)V=<V!5N{Tc2@P8pRl*ae8D&@=cS=N!dNIkNCivVnHQ2MAJ(X3jt-"
            "}{9d$PB+k9jY}<+{{XiMcGMJp~SymA0$vEC6jBt&8MQ^S#<<Xz1L|6$X4MKS@PdeTXC?f|+j5u`Qa4dg>2{Mul<f+*7gCZSd}o0E"
            "FVaZM@5asf-x}@zfs6Ap+<_`q$15mxDbfge+1%onVT)w_Idpo<o7;RG)GPUq#^-M4)n4_lXv7oLgiPO_-AEO2{~gT(=cd<3wE*r}"
            "E9rVuP@_IRyv;ND?j@#U5zZU|c`c6eyE`(t|yyh|N<Hf$e>*C4!`%V+_UT3?J`AmR7u(amhWwXXFGhalGg?>s6hB;-"
            "5JlZy5(tIR`S<>AGOK2!*5YQ#2v<-~lueHOJSmT)iZ%>x)E!Dh~x|c?tWa!5L)h-"
            "kExbra%rdsSsE`@5Jc(k~x(CTBz|GRfSHCVTM|;Y&Dx%pzR=OE1dcgo$J?yLYY%0$bObHlPh)Uoa_Uo$CpfFg(nEVfc<Qzda3q1g"
            "Qm_FrU-6ZW6`UK3mHr{8#}&cp-"
            "$APMoC+g#JqSNZ}Kx?V=m{2^&0)D3ZZujwyJtHy6@Krcc@M>jph8lhXfxcvdzxQ=K*k%kEaHaI!%IN#pHj1zKYVBb48zYH;zF-pg"
            "AN;B(t|hpMm`rotoqK+e8|@E?hB(3>Im$EaNPk8oSXR#Z)DBsTOo9cddgbj2@r88KqETVJ0jN>|xdIP2ylI3~9ROb{=NtDI0*+0!"
            "fmd7ExS3Rboj?@Zwb9qC>9$M)S+D(-p91b#KUwPB5fU3INb!RyJZlWvi+a#=B6-"
            "k;9aUuJOK_AL3J9Ej6oaVy2)7gsVJ^H)JujI9;%v9*R#7`!9pasc<Eokh_kNkYmyEIZ=A6WyG8kDDo6^ggT8tpcSVdQt!bon|o_}"
            "QpPPtxlX@3H&m+VVD`-"
            "+!|ZWBRw~sE!hMJLoB^ngfaBz<SbQUWWQf+QOYz!Gg+UGG#p;??t*)9@wjWb%(wVN)*P>^Aj43GDW@6c=VtYv~KC~4DyyWSE1EMg"
            "1wFO6>XD=`DSX<B!IV*9l<w<G&61uM)!ueii@|u{h<1J$VH3`vFrM{Sv2;nZqRqUyT=uo@X2Dq(k?>mr}ihk2L3Ii2R^h%zt&GwR"
            "Y^os8y(JQfgs?0Nyr}ma1*`mOvuB^^i9!fxh=-"
            "dXh8@aUvuZ7CcO(Hn?6E+IFk*=Jpb*m&E?e3_*bRt$8%9ZxOM24qEGO8y}`E>EacJqQJ5L+4Fn2w|_R4A^n>Qqo|k&w2L3p+gtfQ"
            ";KY0t$c(=x-"
            "Ea>w!`%Lqu*yV4Mb2a?d4Ig4s8iw$Dv>KG?dnFmgm7)77ivc?bV47Z>EVx%MLYOlOAA<i3<!AK!p}!L~#?{kX=6;i=yoR8+^v{$o"
            "t)bZu8kCk|=RbuxQtS8`X}SXS{lxC&Yn=RCy9RHpO-j^|s#&wD0O8X-"
            "xe$OCj+6m3xyas$#Pcvhg^tg)nm^wd!y)X8NBF1=G?T|V1%Ow$yPOozu=dZj9FhjcK56lU|yRj6%ty~KPRE)Ud3N(WDOGMb0Wuh$"
            "t#Z^N|@IDj2-*H+jy?$SWBx`<72TBCtX)JQ^ISSI%wu>ILos%6u**(KQU@iR3*^DRn<jZZ1A?9K(!>I~R!uM-EwohmI^0uE`4J?f"
            "$=0VJr2SP^Waq6q;_h`Dd4$&UzrCk6IuWn;|Le3gH-"
            "*HwBjV|GGOR)(?A+0657$LgPu0uD5kms{+@Uk=It^CkpCO#qXlG+v8(_=A3yC@WV~)YubWqaJxdzcLvbv?2_W=t9}S@%G1v;D>Nx"
            "7dHkIsE#SJM;5ew^QmZW6zB2vJ(IAI#bx|-CkP*n7TX^vdq(YT)N5w}3K}q=52!Hqu7zV%QekGTYt(amK1G=Q>>Vj-"
            ">6%xFgZrPv0xnyA!#ib$fD3ibp(k4zP|&2UV&z;4u2<&ER!ueG5Wtge_a^D->M{IfrdvldvX)fOW&Y#Fma7!T!uKRjz-"
            "s2|f@tKXA|9t}*=nj}QzQoO{SfWM6YtN2IkS3wmUC@k@wD8dTTKh~TKfGhh(tgE?TZF8drf0E;J1=z93;vk8#ntqXuKLe$xf4NA="
            "}E-eS#n-"
            "WET<;o^0wKldXE@B^01Ba}1;tE8EugXW|77!HdRn)w#BiZuF23yHHH`pXbzxyVnB*@N2M3dE%O|y)fqVT|Lmvagl>s(Q&TOHjHY_"
            "uJIy}tC2rdDya~EkasvQ{!%w<q&|aym&`S9NMCpxPqFc4F;PmGLrlVN)IQi}wAXD;zB$Y#SFc1pu{o75>t?Wt6FDC-zrhomb4nfh"
            "OLA3%yjydTZKU?P%EzD$@V;I)$wVG?QsfA~m%bJ*$SIJ7sur5d|7JL-"
            "0|1foh313(ffTf5A?1$tqia26YpphCg>zKWE@kPiGUsjgP^>6q6PVwSSJ(!G3P{2(tf@b+<uNglQB|IaNjLWs8OkjEmg24a=K9R("
            "d(oRUGroIOo)3>a*i0W2d6f!>jU1F0h2&K#IxcZ8_wsVI2%~u+R}k)NTLZ$b{yYT+sJ)5O&TEXE4+phT)%NjqAo~LwXPZs4tq*nC"
            "h<{SWj!sFLbI%AxB=-fvc_xXT#{`gPA6N)fhs2ApYi07s;CU&qR>Jxzz_(sxZlHk-"
            "u)T6<rC@dWK9RN8I;GQYOj|)YWcN}csG>Sy`As3<pxw4-"
            "CaoD%VS3Hk6@YoQFB$VdphdS_(4$Z^5Fhf&Lk!Ocxj5M;{7hB38@zI}Ctxen-W~(4+~h2N08IK(;E@b>I@ydzr&g2XFCw-YR~aVz"
            "fB`e`Ya!eaHL1M0ZlQ$86a<FIzxxmXd(iX}4rJ^p_3uXxfPpf4`E`;7KyQUOD(bbnymF#%#cY~=IzytU2tu{A8-"
            "j#T<(C&N0VGh$?6FE`4bM=AMlRx=0UK{kyixE`g%$@ZL-2Jw@#RbJcna%1c)S($r7K4>%R-t^t-"
            "xSEFj%FHTlQhic`b$pK}Cwrd|wc%vvN*Yh$x3A^=Hmy%uCtqHg;S6s=*2|iUU{BCHcz1FAusxE2o^amW4Y=6I9nZRL}Qyd<}NBhs"
            "J=Ub&?5>f(U#Y_L4<z2{d|e_MF;nT6m`qEci1o<>-gnLMta+9aj5PoO>x-23-"
            "?b!pN4%Z?+(_;fnRpgDZMuvm!M>Gq{V4nu7gw&d+MIa~uqEydJZ@i8_bfZU8jf?B8w=fe+#qL19F2zq{9N1B_{xThO5n-"
            "QDHDt934x*vFpH=KwB+)<Z5e#ssk5+f+aXcY`jTV_530wqT1(tu9A4)^H+R3e)#Hisz4;V7&02ygiy2oO3y`$rqdu&SRpS(d@PbG"
            "#5ezYSvpU6<DLTE!y79C_8L|{>n>H7m6J*U=Xp-"
            "=mYPgiF!`N{mWp0A14~Vq}&GeOnPE?DN&pfAD}#P&Q&FoHE|LwY)==;oe^}M?2zP^cwlCwcg;B2*&*EBL&%6)M^5~yxdm$$H3TUn"
            "q!>h{t^teW9wEKp#x>S$k9HHp_9&PIA#*JXtNcZF)pxSh0`kQB><(-"
            "P*`FM;GW%_yaEZDQkMkz!2s5b*kgf*FiBk_z8L!}{EF5u)aL3*Bg%v8I8+9rOGg53sRb`qfm3++ztJrsp&^iFoqA&&pzD=IXTJde"
            "6V|NP*k~5V8zy(%v701e=Kfxnc#|C*_ks+|=T*q?+1s!F;jbh+xSLmYeT6e!fH=|20FPFr32}eKCU|qhIvq_FjR%6x{)e)}D`^11"
            "1%{`7aY~N-$Lny4u4cZ%4Lg}iL=>$nO-AsI>F)t#1Zff^r(QIJMCd4K~fFn8HQxA)!3Zk>hi0k3$i)zvfbX<ke;Ib|b3r+%lELjF"
            "V{`S#ZCr7yEpdQE%Mld0SIG~~p7T?S7u-"
            "&RAyn@f4sE~4;6WGTeE<H8MECau)IwocJL*VmxucR@JI*V(1LV4{GpzaWpS`+TNuhVH+0M47NiJO3ClPkl!OOP0vvtf@rM9Aw*31"
            "ZNLb-PHFNt+Dt*cGf)3&?&BTvyChs9RANTo2`pM|0NS$e51}-"
            "vGC1U{#>MvP&?O1bAk%uZ~V>{(3Xak`m!a3!;;lEEc7bXm9RO3pi_<r#f??liOSOAs(-"
            "@Y@YL;Ug=sAK^WyLH29ZrUB~(^DC9lUd^FhT$_m3AzQLSdRNGX9IG{P#F|~!OU~+*Gr#zn@O1SrZl)`D+=KeSV+&uuF8ltmLf*N&"
            "nZF{-"
            "8_bVWPJwgx}_kh7VdgO23{mHN)=(LK<<aMpwU$8|@52#Yy)(0PWbV5c?QKGTT7mX^Ht=vBk!=Nr++w&1(%x6#ytJPK(vm#GxNu-"
            "a7o_=DYN)tgX%B31`cx^BLbH2ps>?&1^m$eH<bLxC81M@(|HYYy7jxOgKW%m)V$#}%*P;)*x$}Y^GxoWud3%r#`NvyKBW&wTbT)r"
            "XoJ35V`(SyF+Pp2gERzNVcx<%$_=9<RSD6Y}??at-"
            "i$o}_GE5SrFPi5Chq`wZKN%j){<TB`_`BphSmCR9)Sr*w41aUo18R@BIP=lQj)L9Re`9t>!iuA4MdJ#!4Fb7niN(nYgn*6y&0o7{"
            "X$Iz1~gn}Y5W_&;@U%mNBt>`Ww=U!rEUHoBTUbH0W)V6C>nlxkYcKRHYv+9>jI=2?q5@_DR8)rQUE$$I0QXMa~j?oq3x7;@duY3;"
            "qYA2~mL+i)|pP~FvRQ5RR<a)6yI$d@$U}YpFL$8*5;K&<WnT-iT_>cokRqF1ePZ5%W_}O%O2Jz12a^q%)LK$_~s++Kls)QCq={&>"
            "=s6^W0&YP%G|I)RC=DnYjBx>al1PV8_;;K^>IX~O*g}2N#6tkRSDob1Mjwq_^sjcdDIC06QoPen}yQxDXQ22$#>6`T{#Vc*fRS1I"
            "Tj69!DMk?2~r0b!XUDeWt(c4{GJV#F=o*ks<Ob})b`R?OS&ammkUt4fSj?#2}6k0h{Vq{Yct%?>BtEvL~KxT#t6KR#HB1#yWk6fq"
            "JYH|AQl8eYj^qc|bGHqqH$MPWojjO6kQ+;04xha$s_Qsi>iP`fzkWX{0hl3BzIfwF7IQ_(}RcZB&4nCDET+2}XbRyTBrZAk#*oG*"
            "1Wh$}-BapV=4AN_tspzw$7AxCT2@P_oE?SaikuogqEL@mGsgAc3B=@m@4V+A?PBEY-&b-+4H2N`68K+oY1_US0GqcJu@_>M-"
            "o!?FtUzBrK=4zq|4GtwY9snIk)=*d<jf<vo5~tsm_jHlkotNo`Ba#Y@S#$@eqX|J7eke5{{AxF@I9oJlHoF=I=Z1*xE`6k>*+S`L"
            "jPm;EMIbI?WpN2u@D2h`q7X5x<^^$P^d~k4W0jL&&Q>rn2%cCM1k)wPE0+Pd6iCx%_qvW&W_qv&nT+ldJP_kdW8{1`@)W`Owot1R"
            "QthMosZaAEpiuR&b>IY_s@DjlLl*-"
            "&c?gc~2?(H4N`PIf(~E5{njS|QuG9g05zY=R#Qm0Q*e4=!Osw@m9_UW+n7U%sr@6X#hMih{6Iu%i;1J&vYeE93YckDs!HJ(Y@rw7"
            "9{T8tvcM*^2Dw$6*V}RB161?s4O)b5+fN35V+MyY8I>qYq<@plt1dBs2UQOaLR6k&zXu0Wj9n!Y8T^qz4+~S+zPA1t~c)=%bV?^o"
            ">B4`(vU#h}la?&X5>8A3jg~c<UG@EpdrMP?ynrOEl(O_2dXg`;t9mJSZiCsJbxr(XAcMz_6+rjRa^9I+^xkLxUtkQ8y-"
            "Ficj5yo{67ME$9J0Y^O>jA{mtJ6ryX|?>w?Cw2IhR{>ZX#naNlSiP=!gUHtn^=`+w?o6}0eEr-"
            "*>r%+U=DrN;#zjOJb#MTerPn&?dYN0QpfRv{^b&!RSQpk;y7Lbdu`EHNZ}+6`7$^@oRlNyP~@0(CWfNxUWy##w-"
            "6(W&I<u^i=dnx!n+-`I<nStswo^QNIqApE-+$v!z^mr_5e-"
            ";;~GHONE%3z!Z>rhN}KHR1RwU7kAT+fWvS~NOxRsR%1*v3eJ!9PL#xggJ}2evGXR0o1#CH(E!fKf#ZxKB^{Q&6?(*86X7-"
            "<<k5Iw5m%)rvQ*(WoYE1*ws3Z(7gbOxs5lTzZJ8^~)e%Nl>=~KF<`dq3Eie<75e1lML*~WnR>ATAZX&7=9PnSAU*%5Nv<s<Cz2eF"
            "#sEwB$7Qec86KNIwMgJrFRgl~#JQQn+mw*1~f@V*an5lad~oq*V<KiYeE#bfk=ceNB2E)BtUuE?@kIF}vH!#<r;r-"
            "UG7k};3W@N#a;!?lCTj2zUhW!|GZ(wvu^&}yeR+}pq#%ZB9fJn1M7aamC&Ux5=00M)Fl>>da(mg>9!a*AMUc5R64b!uF4RAa7mtA"
            "7@tKvQeabNVDY@0fW`QR@wgz(JHmZ$m^$c9>u{lhHf}TXryhWPF}RGme<bMFKS4r-nd)y<u<U9xAVea>hQ0T&6l5T5=q+JKnJ-"
            "t~6JzI*U-"
            "p&#@I2IJ8^ZpV_WMV{)U#9s@;*o4JdVbmaGFd_ryKQz3$!Zj+SCdF&JWvUuKH7V)}Y?#D?*(cXX5e?rGSdcg6Lc_(osD!fV1+)9b"
            "l!LF3MRc`eQwTcJ%Tk+y0)3Es^Ba{)!=_RJAYm6U8NB(-iS(U*$yChN=r~<5;6}aPX3ulm2&l(|ZRCt8WOU-"
            "(!#X=0Wp&uUv9KO{eBEi}^UW(|=&V=`|UKhB|BlSiEo8UNU1JQ}`9K`vm7Ut%YZr|{$mw4PJ;;96^X$O8_3m3aUQnHF9$VU?P)8%"
            "Ei-L*fGE^-C@FbO0jd73)JaK2m*_>%&wS%^A@67^6;GJ%x9A-"
            "1u%5VxRE%C)?!9_pQPk<(i|gkri|2zjyDIU1np1<zzvnHUBN<kX$gH7*ohA%@D3NxbX=T~1$39D*hH3Y>0KDm5VPQ9PvcC}j<E0b"
            "$ywVcDckF$RqXE|7so1y{OIG`-F)tXk33cW@}|GMYU<GHp^#;CfVsRN^U)hBo$r^~@y&XWy9Kh92-"
            "be7OqbChC*7cuA`|c+H8%pUjvOx^W<@=gnnX>ZH(6OuB#P)Avcb7$>INaZ`a1a~YC_l+?UyEPpuYST2vZmInp$>2eTGdY0n@Ce6k"
            "yuY+9G2p+`!w8`!DpN@2|C{OIS2V93>V)g)%qAET+C3<PLWl$V@Unkc)=(G(kcLr;L;oTBr3rvZ<kTI6a79SL}A`_?Hbqu;gn$Zh"
            "$C_MtR;*JhbYb{xB>MWP@6wtL3a%k{?3}D9~_&sGI?cD7#7FRW0@pJ_1wX{feF`r@2wgNkd;qBUup%0GX8F~3w8PW&|6NN71ex1$"
            ";B=jMvMRB+$MevLPgjc4O!3`>ad|evwW+-JdE@4|ScQOG!=OX=Ihez#7N+-#6RZVzk0cjH9UOL_9p7jbDSj_p-"
            "z33mQ?0*EE&>lk{4K9<A!4e1iJQS5;{^nKMs+o^NJ)KqtvzsO#a$)JF2~k2LjhNRS^Qg~qaPydNxU1N`QOkIgM(M1aRFJRWBe<8z"
            "nRS|jQJ6#_2UGEG1|c*wP*B=ui&jHZF815-"
            "LzX8AAQ2EC8V#Za3}s=bQ#oEVs$z;s3)g4XqYKtG$FH!^Z4MO2!pA*La?^~k7u}n8Ru~|^qSQf0G2sebQht~0+%9K%AJ4okLSx<2"
            "x2jmn@;oQD)>hx}=kHF!XAfMYOC-{>L;d~>OXI2+?Ow>?PuaNHo|RbQ#P7gH?G_ABr0uY)4D4ej?t!wR1!<1^zR-NJB{g1XM^LB|"
            "4r(TlK*(#N<4C2Xmxo>5#k_r%2RS44Y_l|Criy2^+haeb$7uqC+aG#v!NT_09&%&4A$dOd%6%d%c$yQV#{@(2?p1OCs8E$m`Vz6w"
            "VG=d3rCq+c$<Zp*9F+Y!F=4(g%-25~$s=DgIJ6(`3tiwYa-"
            "5*06*BP|Xq7YziVSK2+p?%31$!XB=_VZv2^3p)m2#s}VL@hhX6?$LPv*SXvkC*t&Sr>SG5Pcbtp!nvHwxho^X+0~dd2>&)oRZ8r|"
            "t7S2oer;VI(L%UNMtxE=yCLPXs83e@Sih1<0gpB<t2j+o8HbN0W7rEu@A|Z8Qpl(G}o4^BMw3&z;*x|M%44niGn<GVy%~Z|vi6*_"
            "Q}VSJR|M8ytT1YFRJae8&RIQd!h&fy*qQ=$vnoaEE<N%%754DiC!B>oawudz7i|SThS~d!W=@qUr@w_$V@+t3BjkKE%IO(zA0%;L"
            "vw!0Ux#yhuXb0k&K61*ru8=WZlO$__lG2@I5;*K_Qx&%ndH3QB?t9r)fguRU4M`j;aOK7K_%q-"
            ">u;W=>wCf$`{Rq#OvZDFHVD|@cl^ZqRm+cV=PHV+KWzzR|YYzDz<wDSEvzWHauqS+2=8jw|Uzo(1|j@=>iL^K17b>Q?g@#4Etd|L"
            "d2+CB{W%dlbBQLXJna;@ei0Z896+)DpXH<0J2C6()c&r#YYT@R&}qlxh#0uGTWml${$DzWB;@=RNG^&$yO$GX@Y-"
            "N6}6HmHSc%PrGbs)!M;Lbk;sY#%5W`#=8`^+=OFmd!k180PAVUL*&$|cZ<zYWb&P&KjnQYbseE2&M0p(sy}Q)-"
            "V@lsl;TZ^MtQh)|jxl3p<;tW307m>0;~bg&YZAo1oG5Bs;v6WGcG~6iK3t%F(#o_l?lJg{SMK(nop4nvAEL>>nB6(O7T)b;c=L>+"
            "JQc4PttowB7(B)6F^voAmEb%Gng%Vs_)Yd}5cshm%BJ3&(<|E4iqX*C=C9X&rT;X(p)Vs^z;B<8K#nyt5^#O5fVbMh7wlwhtK(P+"
            "H#w_FO=Ml8aI3~gfu?zv0cXq~DakQ0%ToY22_p-XFENagjkgwZkFwZy+|CL4FM?=S5t!*32Aqi^F6+7@jCX5a@|eaS1I-"
            "kjN2vPF>CEwN@6reJGQj}-"
            "(uqV<?zo+<X@!A?gz#|7bID?nz3q;{!x?5g7;#}AieMhIw9N0Z1^?_`TmWS7hsz5WAz^A)gHE=?Ed1dpXzmekyrE)(OG-"
            "QrzyJ!`Uq0xB=1_m)9Uw;b*EAOHIt^zDMlm-XkXYWXRj@07ubXRrls|jmbFoUaOe#fxHM==lN2d3<a`UzWWjAB#RPJH2sDC?UzO4"
            "%o9;y&a(A*_8Gb>KcBzlY7*oQmEzHRq37zEq#><5f$smCVSr0op2f2@o;sBJ+%^@>ZI;xknUR7w4+S!G+DOauObh-Qd*!M>XOgJ!"
            "5LgsFlGcku{cAp#MRmYua-"
            "f?~V8@HLA)zOsE0d+lUlj1(o*brra`#)yn<&EkeB=Jf$d3U9FmbGa5qs&B8M3DB5?x_X2qwlLfgE(fu>OWd>|W$73_I8+{(bS$tj"
            "qXih06r!{Z@ruJ=?xrsGEh2GF2D;;+yj1CY3_`$N7V=DXLj;?X2j!6H)zrd)nh=52$c;hB^$>`!Za3&bAk1({h6rS2%*R)1J23IV"
            "ZoRQEB!5p#A?l>INvA+kxZ5fhz^tb4QR_Nm`@=3FRfBz|`wc4H!K*~Ma0g%xM4BoD#1#ki5U0I91THwn55sptoi;Sg?YY5(9Fd%3"
            "v7hnyILvdmieBweM@8AfCWvCEfP%<6QlSA1{a$LICe8n7uqB2oc*pQ*dR%NP(*cfJNe$PJD$$0zY**e;Miz3Yy40~p`t=TuO<0x!"
            "3*?Lw4rLD0E7{MM?$p#o^C0)iP-Rqb*D{X<7uf9e=G{WoX3b2*d;(RQEILf3-"
            "|lYI0wsm|)S3u(t{QXp*l4EogHyal#sWe)2z(UmSn}d>(zb9_heFs6XMIe581-"
            "_>lM)*!Vo%OVDelm2NKA%wjiTJG3>2vV6GtCvQcW~W%!6wODExyHPy)9HvBjc?dpr<mACi2GWd)-"
            "<MKY*;j6SWq2O$!V7(|ip1qCF|W4p>Eb&rCUSgS-uKCZKLj2C5%t`Vrfp}HPt*^=mj#xI~iQSRr)Y4fQMAcvIv?JkW5cUu_#pa*Y"
            "GpiAe1$=gQ@wYyCF04b2F9b$P4Y!(aC-9C#1c<B;ub}jF5ldulg=xm9BIWbEBaj-WdE!^FiAv4|OC3czM9qK@4&Bgd6OZWx*U;7Y"
            "|BSG&VNRMd(%B4&ioNhh+L_Za1h-"
            "~&<oh!^`>r0*_;?iSas=31|C#lO3=V~X^q0oG*37r_as_+8SwrFLU$q|+x!F75sp6F=f+2jJpDK9-Xxs^d>W-"
            "Cw`eri}JH_jH1VNF68z0@7@j8owa<VZfqClC|iO=9OIF~lzk^A(P$JtS7FcXFvvnmE)=F|%+CQhdVUB!COB=yb`^?gUtw{!lWS6W"
            "=c@WgokJ9L8b~3PYCCG|K31Qr<wXNS)*BrL+{9P0fP8A%M#OMsh{i={BeJ*#VR?JnD$@#uc4Wy+c0nH(=}H0sA>BXRFGRgp=$nJA"
            ";LUd>$D(_}U(QYQ=Ng02xGq0ew<u+LKMPYwl=ZUzf<pk?~qLx5!}htH64M_Ymi2O))zWLG3YBKn{u!vk1h_0KGBDU7Sav9#R?D<n"
            "(q`?Cjp|AxFgYO#=7h^FrKO@*vDP=8IdhXaXkVbm@Tv^m|z7XyM?f?&v}DV!HGfwwRU81Sw(z42`qmZr`WZ<N-"
            "u1*AN<JhF3Z0)dG7nc(jHvUyZh)1$?gDR?#XfBWHE)^u!5+;(#47b#r*;dnsXQ)C5x}Q--E(9mVZBPM3K-"
            "XF=36kFFMAIIt%{?;tY%Yw7+{du%SuOi&4CHB_7_F2Jo(wh^r06Q5C&D4xMl>r4_Aw}x!C2UV2A?(W6J%SQ_rxh^1B_L-"
            "N2kGAkicZ$0E<hg{ex9#C0du;Z}`EuPnTv9(q5+%=Y@{ZN*^l`Tbssmi0O-"
            "4>>2&F;<S2^hklbl`lH<Vw;$+jzG>k4&8Q6u5EFezoy9J<?2y5VjG2F>nq<f@+Z80@WuSM5byz|rc8x?N{z1Eva3ilPC;q8(mmZ~"
            "lwhI+1};{~UmYqToH(Wt$VtJ1|ZK;z5_=X^PURsM77q0nZ8iBZxl7u&od2WaK0fkq-)r4W0SDe6lzz^=x-"
            "!gisZEU(^#b>6Ey+<xDfWWoIlH+X$l115qvKougx)T9EK(TE6JNPPmF7RpuGk=!+(eOq({^>}8bG!asE?8O6I+M2U7iyaJ=}D%EM"
            "(>|AdEg)r+xGNL&dh0dzSlwa4OMkB-vs<GK5`T@NVD9ZMrAr4YxMD4vO7t#!8UjSG3z3eN8IK@%hQYGbQ9YDrZNLNYlez*t4%6wu"
            "LspaXBODbzaHq-6Log|gs+r41w5@je-"
            "!#PExNn%z$)H%9iK_j0VVHQ{}F6JS?|8)hwZB^g@yoMH}SQhh42egv|S&4SxooiiWrMC+SD>R^Ye_Kp*h2>J1b%u}jdTg7sEVK~H"
            "fImCk;tM7kyDoRoNJ!u&sUkt}dB$w6yNP1=!^_0uU|*A|#^bx`WU7KS2f%Lrk~3zL+GNxpb}u>fb1AZzljzQ-"
            "u3GpR6V&@1$BWKSsGKs#LXft*8LY@iS(>D(ls79^5NjQDmqkUE?30^IC{Vj-"
            "u5KUk;yy!KXmEwf^9c)NdR)qYCr;%*o;dTcHC>6Qq)8|}OMNgw9S(WFWxed~-"
            "4k9>)H^5?xggGXnYc8OFP6cyO!Sz_JLODbC^FUrh9el@O&q~KNXb&}Kq6ZaXadYRT>l2B-$k?-"
            "5LPVVbxfY4x~h~44?FuPgaTQq5ShATD2dosgQ+7d00$kkr6<$18<lIVd`~h%=NqbR;)UxnRCnAG#XW;CmGCDu!s($uERT5l7*&Y?"
            "m-*#wh#shL7PFidz*N_GE|ax-UxIpCCl<_&(&61+tXt+Vqu}_D7a4AGfO<)na8Q)RVwe+r%@K-cPF-"
            "=P2gXK}$gC|ZZi}pniX6?^P|hHXjhT`*m>%!$s8-"
            "L4Ne_KgPpA&NIzzS@>WT#3Ihe?7v!vOiOZRl85gl<oP}&72VrOGODPo_i%+7m5hV3Xmq}2V9+Y#fg&U<gChzYMYGPDL+KyIQz^&Y"
            "TR$I)1X=eDp7o?~Jwkw|MU`kLf+>I2?e$Q2b;NmH%+I5aSN2crjE42#AGlc{$QAPECvsmpZ^_HN0*(KB_)16bkuAeFA3NcAYWv|x"
            ";Vig;p7un5^FC&eI>{R5-?gI&um5<T@jh)Bb95H7d|fE`K-"
            "*r8fRTG1aIi0DL7_l>?Gu(CEWPpZ@AG~!g8Hv|y~=E3gXw2nsit`8)Vp7>~OM1_?+0ZCjmVxn)DPZf=VVC!|l!W3VUj|9~iVb6Z1"
            "_$n42Ei+mr%qiLNbI89JdCE`#)55t<E3NM-BMbv*0@$Fg>QY~{(LipLp&LSN7?8;oOdV2KoQ$2gVV?5}0(KFA4Ypww4$nGzP+~=@"
            "PtU1?R%|ouCCV#3Dww{{ginAhP`;5Q&JGWa55!dQ*>5PCX$Y<_ssb2FyLHa|=o~VO!0E1*LKk0{_r*ssXslcwg+WQ;oKN|VxVYaw"
            "!m!6ir@}Pq;e9Yg!7q|}Fevxp41+@v{DDI|W=UL=YlmaabfynD@)Qbd<?0qm&@YWaKV@10vW857&Q{{5K@0jAg&9v?V5{^td-"
            "P{5By?(pTY<|B7j~4>VCr^Bi_KeUSpe>aQpLh%K@oY&sn3QG6k$Vf^>*=4M18PDx0JxO33VaY(jcxoyXIOE-"
            "|JKMy__#@6Y>aCS>V{$keb6N)Q~!!U1aOGPn7SmazCb6Sc6y-#$l(dVXvXR4z6C>9x0Ybr1Rmqbq--"
            "|v8A}%=tw&uWaw2#(~$bThYDi(!4O2x)j?9J_J5}`?7?tUi)_2gQsyURl*1zaUA|t111$zM@6K_P_CWPHJ6WnNS0XmMlpb|?*FRM"
            "s2&2r+1l*J2e=?+)tbNK{tJX6p=fY%(_u%wGV!M`!;7Y^ic+du0PJ3Cm-"
            "j3P4bb&Ln<COH4|8P;ZG+Txx`a8)E@xY0BiwCxA>QsVGTAd1ol$g@|GmL%MMLg<*Z=#TA#v$xp#FHqkc19jA@n)vuB1~Hy^C3FM3"
            "7mRJc+dGvaf8R{^5SAd2!OV-lw(k?L|%r3IW&C`-DqBEZV>mi51vmQ-h{?NZ}r@Q>AvXTC**fm?lhS@C>-"
            "7lfedy(Q0r}|8g*$MrUEY!I~qsg_Sp4O7ZY_b%Mx6G!AZER92Hp_5h^&8`*^x4P}&sCw!qgCXFJj!(|H@B?&=q2|EGTGJWg<)^>m"
            "8wbbjOy>M<dPs~3VFpiWT&_Sw9xmst;W+*9&{PUI26Iv1X6nlJ|RT^ghqQD0$iMy?d73QAfr{`;b505u0L$k9qd7$zq~A6y)0g0E"
            "ngQl}8>-TZ!E9%gKw{JQL)$gpvJwr7|4?^DPIO!5jr0I8(3Jd=FL3&%Kf^G0&2WI?g`Xr<FN8}t*she6&$KbNn4F9WfJ>%-"
            "p{mmf?uG!g~-"
            "^U?2UjG`om55=OlFFhNy3m4H|b0>63@91zYdUdh)bf%Y3%1Q1MMHaojLG8xnP4z?~jWP>JUgI!_;FSk~lrVbsSIyDZuITCk8VRub"
            "7(LcaJxI{=NJa#Afp~%X5@-"
            "k{GQv8fEB14n#T9e+fTq<TZc=u5GCCGcff`pS(~Z%@xCd2hNRB?0@KZv)113R)^HhN&!b#Lf*sbKz#lG_VXIZ<TKDFvYV$|*cuP2"
            "m)>Rgj6MXkznkgG=coRHQ?N+d27F*rR?D~U{|f>HycJTRsdSF-Rk>D4ir+ytbJ2yV3e6LB7qRdP4%^`R|(!7H3-"
            "C{+<&B`)cDYOvG$-h1^q%|37>sR}0%>j%{?Z1NL3fH7FqF^;4at#gq@`5-yGX9m(mN8`{;@3`-"
            "aRw9bw%RK@%I=wO=|4fnv$bSNiC4Z684I(KHBPEc5MW%;${sXI%?j%5g2fx7mO~a|GzNaE3px+*V>*m5&s!cMg;7&p7@tv<(BHR}"
            "c8xKmEpCayqPv8tKyhQZygLS1Sz{%jCmm$qb8!Xy)0hUS4KVD2aFamx*&?W&#vUs4xSVyFZwta4=&?89oD@hXEF{T!7H{m^S2+62"
            "d+2Q%3Fj5IU&|(1VSTHhRmc15wj+%cM;q;nFZ#UIElRbT5^{KTTj|G4_ZGgFBoeWgbNXlT94I7wo5Y$joAHULcdR``y*w6mcc1yN"
            "L1JB9Ro$Cw(+psSyw>$&R@kwzI=QW#OE^K-"
            "X41Lo0LbNJXn9sfhb1j?5k>;h)lVf(ql$oct_Y>V~@+e84iqdfQhv?zt2D{Sv#~Kr4&$s#%^Z+3~_nJo~_HS#=Nu<1~P1W<kAb!b"
            "v%^8>xS)A82v1ulf(XS^m-quGvmq)D5loU8JF=}O}y4a-JS8-"
            "P!HD44p`{P>evVL1%&QpHma;)r^`_eZc?@e1>?{v;1jL4hJgv4VTP*_jA!{D|PU0yjB8e6ct<pGMUqdIh6w`H$Ex~FkrW>#}_Tt9"
            "bLny8;ax1@H;DXjd{bR;)PH9)~*#}8}4CHOklZL2F^tB4j8h^p=Rg@BoqQnhnmjN%`8!}twa@Ls0gxo*kq_B<7!oFhZxuSI>s+sE"
            "Us*)LeZp?A7mTDa$DVc~QlDx-"
            "M{`_p0WWhItW=Q!2dxED{8_2}tM$016hj)!Dh9y`Cn{X_lxsfTZi@Dn~HSs?^ra!;tgc&TNnMly7sY!ogabzexHbAw50R&!Sw%*Z"
            "ieUo`EpU}d>DWr0W^0;oP$7S_s=j?=<q(u0OEDt%PJ(luk{{6(5i&i65*j<HC$f&$OuH-Vxg?_Rt>@Fb3+9i$-"
            "e+%>J2;oQ0Z7qbD0tn#QeJAEDGmV&~CBTs1f2VF}py~Gc9cEa56nM#ae>SJ#NoJnDKTl_(ha`z*~To+sbNSKn0Q{S_(mSKtFPp(e"
            "d;d*N9nE&MDh3!?1BXtfBAn0E5DKWh>UPU}lt~8@A_rx|>`DBoESeH@N^z5{Ja`pdQAbNYw!J?Fw?E#akx$d$QTFUA$G<0<cbFid"
            "pd1sWV&A$>S=FSV2B}Hu69I;+z|G0bJ6gsmR-"
            "T^`~Sw2Lev&9v#^YP!o;Xc~$rL6~MTL#~0?LR>XHIt>lRP&9cDZECYFA^uBwC74O_?{*;@uQn&e^aPJJe7|m_W67soCT)?ig&s>e"
            "(t$It}8U(BxCMLF1aov*_?v%Xk-"
            "!^Wwa$bxvf|6&gXB>$DZd)R)!p6Q5alERu2k$IY7zYOOB{`_3T(eB8YzTj<)PIxcr$g5cpXdeCrGMX%|;dazHsKjEpdOje|l<GH|"
            "dxKPC)Z)B?0`%ke$W#c`ka(#F~ayx7t#bjv&sYSKReP7#CEV}PKPPmqRvnXEj-Mlo+F4bkTs!D;a9&lo#TXJcEkZ#rb@SCa*23$d"
            "L!?xW0~z9s2!&URqjyO&2k7nJ5YJ_e^w8Yke&W=h?%r<hW;`V`<XL6WixG2UJ|0Ns{>lS4h@xe_8$B^l#otbwbv3GUw4PH0wiy!X"
            "YKiGJicqC_frfolQ6s`$O7!LK75GKk`GaEg$H6Wg>qi9{`&A&1C`pUm)TYnYH@Q`IG(ScY+;Gz$(+w4_AmIM+N8ex?k_Y)hQY#-"
            "`+jjm<d(O!SEwYg?n2V`WtCcY491E4p&!4gb17mIc<JWKbA`%DiK389hFjQrjPUFM5Y3RP5P>1^riE86B}_1L34fJp-Dr=T+KRnO"
            "PLg^yw4}_TEx(51DeC)0T)7oPiK(fB3_nKKs+3UcdV6(}y=-"
            "ee>?0e)93{$B*B<fA{exZ?^mE*54doU;ntFSoU=}j?aGj>OVgF^v!o~KYn`i-"
            "S_riJ9ITW_)Z9cp_;T`_2KQO58u4~<J+(72R_^PSHFJ$$5$W&UhT(E%l_(reev_p?5DnZ^XZNMzJL4vEk9|0@b)90{PE4VKkz62@"
            "EL#WC;R@B?f9bXKixL_pZ}r%%a6?u&%gBH{TDvmx4-n^{_?{=XruUcZ$8-"
            ";{p8EzE6dT}Zg1ZnU*BIB+pYcjksaFMuZM5T&fs6(pxS?QgFgN6{kLzI4SGKN`oo*=md`%>kKcdx<@;}c`0m}u-"
            "+%VgfB5}pKY#z-_ix_)+wVVn{rk_p_~y%he)~cH`l~k|{`u{vS3iIA{j2}>U%vS7`ZN2FZ$7>G_SIki>-"
            "TTpef7<!A3nUb5C6(P{MoyAKiFq~Z{Ppj59!Z;`}WIE@9lT~!T$Z5cVE5zSC-vg+u40xU)yzfZQknj_4<Cj+aK=y6c5M`*_ZqE@B"
            "a1|fBQ52o?rj&7yJu>R%3+5PP|=Iq&Hp1uwn_~A9EHg{M|QSzJK@C53K)@FJQm$SKoa3;r++=Uw`t4{O$kz!u~~N^k!KaQ1@ov0+"
            ">Pib*fj0NiF@^%@X#D{^gDB!he1B^Kak(@Kr4Gvv0ruKX3l=t*`MnzxmDI{8Eb?U<NMc(eLv526Ty89{*I$f?eJW@PGFD7uNXh&A"
            "-2Y_xYFazw_Py)&Ksr{fkHoA!0mtoZQa0-"
            "mhh2yxmiuKkL_zukx$6pWeLsyKmoo`uhEc?>_pT{q+~T2WF3LKhQ0M5Q0!kEH&ew;;e0FdYp}g{`TFcw;$fw$@=Zv4}bjT%eUjj{"
            "MCo|Z@!vO^Si(KjjwUS7i@NQJKJkIuXspi1b^)P{_LwizIpfM+pk{fMZfyR+gL+e^b7m{lmvE^mJ!#vUNdSy!iPg!UdE67{x9Bq^"
            "Dl4Sy;?qd^{cP{$*<3Ef2DV@Yd!%AIx$uN+aaa4#Pz=F(O>Vvj_($Y?3L69BB>u)?>8UbeEIEL{r2Dd{m*}?2dH<04utUv(wNT~R"
            "7|h)^(n@T|G%U9cW=IZ^X}6(e&_%0mtXw7R$xjMl+6WBiShP&F{=m@Tx;i#|F$pQe*E@LY{f7AlrTVZfjc|R#|YlKd@*4X^B-"
            "|Wer|r|hwt8gu;2dIzZ)<1x4-(OzefL;O)R3Np!$B<Iv2dVOsm<R5-"
            "zJT+b@e3QPJMG!~e+^o0r3)Vl6+)riL}dP=5UMfe+m#rayRL8D(X^egEdu+4%IwY~+g{`%kRpi=S}2Io8F+JXZbxetiFKG5CM@lR"
            "2S3{XYOm!lKt=1^@"
        )
    )
)


# In-memory cache and data wiring


def cache_cached(name, function, valid=lambda value: True):
    """Use embedded real responses and in-memory refreshes; never write files."""

    @wraps(function)
    def call(*args, **kwargs):
        key = sha256(
            json.dumps([args, kwargs], sort_keys=True, default=str).encode()
        ).hexdigest()[:20]
        cache_key = f"{name}-{key}"
        with cache__guard:
            lock = cache__locks.setdefault(cache_key, threading.Lock())
        with lock:
            saved = SNAPSHOTS.get(cache_key)
            if saved is None and name in (
                "stock-prices",
                "dotcom-prices",
                "trading-sessions",
            ):
                for candidate_key, payload in sorted(
                    SNAPSHOTS.items(),
                    key=lambda item: item[1]["timestamp"],
                    reverse=True,
                ):
                    if not candidate_key.startswith(name + "-"):
                        continue
                    value = cache__decode(payload["data"])
                    if name != "trading-sessions":
                        requested = list(args[0])
                        if not set(requested).issubset(value.columns):
                            continue
                        value = value.reindex(columns=requested)
                        value = value.loc[value.index >= pd.Timestamp(args[1])]
                        if len(args) > 2 and args[2] is not None:
                            value = value.loc[value.index < pd.Timestamp(args[2])]
                        if value.empty:
                            continue
                    saved = {**payload, "timestamp": 0, "data": cache__encode(value)}
                    break
            try:
                trigger = ctx.triggered_id or ""
                force = ("refresh" in trigger or "retry" in trigger) and bool(
                    ctx.triggered[0].get("value")
                )
            except Exception:
                force = False
            if saved and time.time() - saved["timestamp"] < 900 and not force:
                cache__local.notices = getattr(cache__local, "notices", []) + [
                    f'Yahoo Finance snapshot retrieved {saved["retrieved"]}.'
                ]
                return cache__decode(saved["data"])
            try:
                if force and hasattr(function, "cache_clear"):
                    function.cache_clear()
                result = function(*args, **kwargs)
                if not valid(result):
                    if not saved:
                        return result
                    raise ValueError(
                        "Yahoo Finance did not return complete valid data."
                    )
                payload = {
                    "timestamp": time.time(),
                    "retrieved": datetime.now(timezone.utc).strftime(
                        "%d %b %Y %H:%M UTC"
                    ),
                    "data": cache__encode(result),
                }
                SNAPSHOTS[cache_key] = payload
                cache__local.notices = getattr(cache__local, "notices", []) + [
                    f'Yahoo Finance retrieved {payload["retrieved"]}.'
                ]
                return result
            except Exception:
                if not saved:
                    raise
                cache__local.notices = getattr(cache__local, "notices", []) + [
                    f'Yahoo Finance unavailable — showing saved real data retrieved {saved["retrieved"]}.'
                ]
                return cache__decode(saved["data"])

    return call


# Preserve the panel functions; only their data access is cached.
stock_get_prices = cache_cached(
    "stock-prices", stock_get_prices, lambda d: not d.empty and not d.isna().any().any()
)
p2_load_valuation_data = cache_cached(
    "valuation",
    p2_load_valuation_data,
    lambda r: not r[0].empty
    and not r[1]
    and not r[0][p2_NUMERIC_COLS].isna().any().any(),
)
p3_download_prices = cache_cached(
    "dotcom-prices", p3_download_prices, lambda d: not d.empty
)
p3_trading_session_closes = cache_cached("trading-sessions", p3_trading_session_closes)
_original_company_loader = p4_load_company_data


def _refresh_company(company):
    try:
        force = ctx.triggered_id == "panel4-refresh"
    except Exception:
        force = False
    return _original_company_loader(company, force=force)


_cached_company_loader = cache_cached("simulator", _refresh_company)


def p4_load_company_data(company, force=False):
    result = dict(_cached_company_loader(company))
    result["source"] += " | " + "; ".join(cache_notices())
    return result


# Main Dash dashboard and command-line launch

"""One Dash server; four isolated, live Dash documents. Run: python app.py."""
from pathlib import Path
import argparse
from flask import Flask
from dash import Dash, dcc, html

TITLES = [
    "AI Stock Performance",
    "Capital Investment vs Revenue",
    "AI Boom vs. Dot-Com Bubble",
    "AI Valuation Simulator",
]


def create_app():
    server = Flask(__name__)
    children = []
    for i, module in enumerate([p1_create_app, p2_create_app, p3_create_app], 1):
        children.append(
            module(server=server, url_base_pathname=f"/panel/{i}/", title=TITLES[i - 1])
        )
    panel4 = Dash(
        "panel4",
        server=server,
        url_base_pathname="/panel/4/",
        title=TITLES[3],
        include_assets_files=False,
    )
    panel4.layout = p4_create_layout()
    p4_register_callbacks(panel4)
    children.append(panel4)
    panel_style = PANEL_CSS + PANEL3_CSS + GRAPH_OVERVIEW_CSS
    for child in children:
        child.index_string = child.index_string.replace(
            "</head>",
            f"<style>{panel_style}</style><script>{GRAPH_OVERVIEW_JS}</script></head>",
        )
    app = Dash(
        __name__,
        server=server,
        title="AI Boom • Four Panel Dashboard",
        include_assets_files=False,
    )
    app.index_string = app.index_string.replace(
        "</head>",
        "<style>"
        + OVERVIEW_CSS
        + "</style><script>"
        + NAVIGATION_JS
        + "</script></head>",
    )
    cards = []
    for i, title in enumerate(TITLES, 1):
        cards.append(
            html.Section(
                [
                    html.Div(
                        [
                            html.Span(f"0{i}", className="number"),
                            html.Span(title),
                            html.Span("Open panel ↗", className="open-label"),
                        ],
                        className="card-title",
                    ),
                    html.Div(
                        html.Iframe(
                            src=f"/panel/{i}/",
                            title=title,
                            id=f"frame-{i}",
                            tabIndex=-1,
                        ),
                        className="preview-stage",
                    ),
                    html.Button(
                        f"Open {title}",
                        className="open-panel",
                        id=f"open-{i}",
                        **{
                            "data-panel": str(i),
                            "aria-label": f"Open panel {i}: {title}",
                        },
                    ),
                ],
                className="panel-card",
                id=f"card-{i}",
            )
        )
    app.layout = html.Div(
        [
            dcc.Location(id="location", refresh=False),
            html.Header(
                [
                    html.Div(
                        [
                            html.Span("AI BOOM", className="brand"),
                            html.Span(
                                "Four perspectives. One dashboard.",
                                className="subtitle",
                            ),
                        ]
                    ),
                    html.Button("← Return to overview", id="back", hidden=True),
                    html.Span("Select a panel to explore", id="overview-hint"),
                ],
                className="topbar",
            ),
            html.Main(cards, id="grid"),
            html.Div(id="navigation-status", role="status", className="sr-only"),
        ],
        id="dashboard",
    )
    app.panel_apps = children
    return app


app = create_app()
server = app.server
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    print(
        f"Open http://127.0.0.1:{args.port} in your browser. Keep this terminal open."
    )
    app.run(host="127.0.0.1", port=args.port, debug=False)
