"""AI Boom dashboard: ONE PYTHON SOURCE FILE.

Install: python -m pip install dash==4.5.0 plotly==7.1.0 yfinance==1.8.0 pandas==3.1.1 numpy==2.5.4 Flask==3.1.3
Run: python dashboard.py
Open: http://127.0.0.1:8050

All layouts, chart construction, controls and navigation are written in Python.
There are no embedded CSS stylesheets, JavaScript programs, encoded data blobs,
external assets or source files. Dash provides its browser runtime internally.
Appearance uses Dash's native Python style dictionaries. Real market data is
fetched from Yahoo Finance at runtime and cached in memory for 15 minutes.
Internet is required for market data; the simulator also accepts manual inputs.

Code sections: original stock functions; panels 1–4; shared data cache;
graph-focused overview; Python navigation callbacks; launch command.
"""

import argparse
import threading
import time
from datetime import date, datetime, timedelta, timezone
from functools import wraps
from math import isfinite
from urllib.parse import parse_qs, urlencode
import dash
from matplotlib.pyplot import figure
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from flask import Flask
from dash import Dash, Input, Output, State, ctx, dcc, html, dash_table, no_update
from dash.exceptions import PreventUpdate

from panels.panel4_simulator import CYAN
from panels import panel3_dotcom as panel3
from panels.dashboard_shared import COMPANIES, TICKERS, NAVY as SHARED_NAVY, CARD, BORDER, GRID, apply_theme

stock_COMPANIES = dict(COMPANIES)
stock_DEFAULT_COMPANIES = list(TICKERS)
stock_BENCHMARKS = {"SPY": "S&P 500 ETF", "QQQ": "Nasdaq-100 ETF"}
stock_DARK_BLUE = SHARED_NAVY
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


GLOBAL_COMPANIES = list(stock_COMPANIES)
GLOBAL_PANELS = (1, 2, 3)
# Colours follow the ticker rather than its position, so a company keeps its
# colour as the selection changes. The default six and the two benchmarks keep
# the colours they had before the toggle existed.
stock_COLOUR_BY_TICKER = dict(zip(
    ["NVDA", "MSFT", "GOOGL", "AMZN", "META", "AMD", "SPY", "QQQ", "TSM", "ORCL", "PLTR", "ASML"],
    [stock_STOCK_COLOURS[i] for i in [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13]]
))


def stock_colour(ticker, position):
    """Colour for a ticker; falls back to its position in the palette."""
    return stock_COLOUR_BY_TICKER.get(
        ticker, stock_STOCK_COLOURS[position % len(stock_STOCK_COLOURS)]
    )


def global_normalise_selection(values):
    """Valid tickers only, without duplicates, in the canonical 10-company order."""
    chosen = {str(value).strip().upper() for value in (values or [])}
    return [ticker for ticker in GLOBAL_COMPANIES if ticker in chosen]


def global_parse_selection(search):
    """Read '?companies=NVDA,MSFT,...' from a URL query string; None if absent."""
    query = parse_qs((search or "").lstrip("?"), keep_blank_values=True)
    if "companies" not in query:
        return None
    return global_normalise_selection(",".join(query["companies"]).split(","))


def global_panel_url(panel, selection):
    """Address of a full-size panel carrying the header selection."""
    chosen = global_normalise_selection(selection)
    if panel in GLOBAL_PANELS:
        query = urlencode({"companies": ",".join(chosen)}, safe=",")
        return f"/panel/{panel}/?{query}"
    return f"/panel/{panel}/"



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
        colour = stock_colour(ticker, number)
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
                gridcolor=GRID,
                backgroundcolor=stock_DARK_BLUE,
            ),
            yaxis=dict(
                title="Company",
                tickvals=list(range(len(indexed_prices.columns))),
                ticktext=list(indexed_prices.columns),
                gridcolor=GRID,
                backgroundcolor=stock_DARK_BLUE,
            ),
            zaxis=dict(
                title="Indexed Price (Base = 100)",
                gridcolor=GRID,
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


def stock_comparison_data(companies, period="3 Years", benchmarks=None):
    """One real-price/normalisation path for both Panel 1 views.

    Preserve the original lookback, adjusted closes and shared-date indexing.
    """
    tickers = list(dict.fromkeys(list(companies) + list(benchmarks or [])))
    start = date.today() - timedelta(days=round(int(period.split()[0]) * 365.25))
    prices = stock_get_prices(tickers, start.isoformat(), (date.today() + timedelta(days=1)).isoformat())
    missing = [ticker for ticker in tickers if ticker not in prices or prices[ticker].isna().all()]
    prices = prices.drop(columns=missing, errors="ignore")
    if not any(ticker in prices for ticker in companies):
        raise ValueError("No selected AI companies have available data.")
    indexed = stock_index_and_normalise(prices)
    if len(indexed) < 3:
        raise ValueError("Not enough shared historical data to compare.")
    return indexed, missing


def p1_create_app(**kwargs):
    app = Dash(__name__, **kwargs, include_assets_files=False)
    dropdown_style = {"color": "#0B1220"}

    def control(label, element):
        return html.Label([html.Div(label, style={"marginBottom": "8px"}), element])

    app.layout = html.Div(
        [
            dcc.Location(id="p1-url", refresh=False),
            html.H2("Panel 1 — AI Stock Performance (3D)"),
            html.P(
                "How have AI-related companies performed relative to the wider stock market?"
            ),
            dcc.Store(id="p1-companies", data=list(TICKERS)),
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
        Output("p1-companies", "data"),
        Input("p1-url", "search"),
    )
    def use_header_selection(search):
        # Opened from the overview: start from the companies chosen in the header.
        chosen = global_parse_selection(search)
        return list(TICKERS) if chosen is None else chosen

    @app.callback(
        Output("p1-result", "children"),
        Input("p1-companies", "data"),
        Input("p1-period", "value"),
        Input("p1-benchmarks", "value"),
        Input("p1-detail", "value"),
        Input("p1-refresh", "n_clicks"),
    )
    def update(companies, period, benchmarks, detail, _refresh):
        if not companies:
            return html.P("Please select at least one AI company.", role="status")
        try:
            indexed, missing = stock_comparison_data(companies, period, benchmarks)
            stats = stock_calculate_statistics(indexed)
            step = {"Daily": 1, "Every 5 Days": 5, "Every 20 Days": 20}[detail]
            sampled = indexed.iloc[::step]
            if sampled.index[-1] != indexed.index[-1]:
                sampled = pd.concat([sampled, indexed.iloc[[-1]]])
            cards = [html.H4("Total Return")]
            for i, ticker in enumerate(indexed.columns):
                colour = stock_colour(ticker, i)
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
                            "background": CARD,
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
                None,
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

# The same ten-company universe across all panels.
p2_tickers = list(stock_COMPANIES)
p2_sector_map = {
    "NVDA": "Semiconductors",
    "MSFT": "Software & Cloud",
    "GOOGL": "Software & Cloud",
    "AMZN": "Software & Cloud",
    "META": "Internet Services",
    "AMD": "Semiconductors",
    "TSM": "Semiconductors",
    "ORCL": "Software & Cloud",
    "PLTR": "Software & Cloud",
    "ASML": "Semiconductors",
}

p2_sector_map = {
    "NVDA": "Semiconductors",
    "MSFT": "Software & Cloud",
    "GOOGL": "Software & Cloud",
    "AMZN": "Software & Cloud",
    "META": "Internet Services",
    "AMD": "Semiconductors",
    "TSM": "Semiconductors",
    "ORCL": "Software & Cloud",
    "PLTR": "Software & Cloud",
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
                                
                                if fx_data.empty or fx_data["Close"].dropna().empty:
                                    raise ValueError(
                                        f"No exchange rate available for {currency}"
                                    )
                                
                                fx_rate = fx_data["Close"].dropna().mean()
                                
                                if not np.isfinite(fx_rate) or fx_rate <= 0:
                                    raise ValueError(
                                        f"Invalid exchange rate for {currency}"
                                    )


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
                                
                                if not np.isfinite(fx_rate_market) or fx_rate_market <= 0:
                                    raise ValueError(
                                        f"Invalid market cap exchange rate for {market_currency}"
                                    )
                                
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
    cleaned[p2_NUMERIC_COLS] = cleaned[p2_NUMERIC_COLS].apply(
        pd.to_numeric, errors='coerce'
    ).replace([np.inf, -np.inf], np.nan)

    chart_df = cleaned.dropna(subset=p2_NUMERIC_COLS)
    chart_df = chart_df[
        (chart_df['Capital Expenditure ($B)'] > 0) &
        (chart_df['Annual Revenue ($B)'] > 0) &
        (chart_df['Market Cap ($T)'] > 0)
    ]
    return chart_df

def p2_calculate_regression(chart_df):
    """Fit annual revenue against capital expenditure; return fit statistics."""
    # Using the displayed companies keeps the line consistent with the bubbles.
    x = chart_df['Capital Expenditure ($B)']
    y = chart_df['Annual Revenue ($B)']

    # Require at least 3 companies and variation in both variables.
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return None, None, None, x

    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    ss_total = np.sum((y - y.mean()) ** 2)
    ss_residual = np.sum((y - predicted) ** 2)
    r_squared = 1 - ss_residual / ss_total
    return slope, intercept, r_squared, x

# Create the bubble chart
def p2_create_bubble_scatter(chart_df, size_max):
    """Create the bubble chart using company financial metrics."""

    fig = px.scatter(
        chart_df,
        x='Capital Expenditure ($B)',
        y='Annual Revenue ($B)',
        size='Market Cap ($T)',
        color='Sector',
        color_discrete_map={
            'Semiconductors': '#00D4FF',
            'Software & Cloud': '#FF6B6B',
            'Internet Services': '#00E09F',
            'Consumer & EV': '#B477FF'
        },
        text='Ticker',
        size_max=size_max,
        hover_name='Company',
        hover_data={
            'Ticker': False,
            'Market Cap ($T)': ':.2f',
            'Capital Expenditure ($B)': ':.2f',
            'Annual Revenue ($B)': ':.2f'
        },
        title='The AI Bubble: Capital Investment vs Revenue'
    )
     # Improve readability of bubbles and their labels
    fig.update_traces(
        marker=dict(
            opacity=1.0,
            line=dict(width=2, color='rgba(225,255,255,0.95)')
        ),
        textposition='middle center',
        selector=dict(mode='markers+text')
    )
    return fig

# Add regression line
def p2_add_regression_line(fig, slope, intercept, x):
    """Add the fitted regression line when sufficient data exists."""

    if slope is not None:
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept

        fig.add_scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            name='Regression Line',
            showlegend=False,
            hoverinfo='skip',
            line=dict(
                color='#FFD166',
                width=3,
                dash='dash'
            )
        )

    return fig

# Style chart and add titles
def p2_style_valuation_chart(fig):
    """Apply the dark theme, axes, sector legend and subtitles."""

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0B1220',
        plot_bgcolor='#0B1220',
        margin=dict(r=300, t=145, l=80, b=70),
        title=dict(
            x=.5,
            font=dict(size=24, weight='bold')
        ),
        xaxis=dict(
            title_text='<b>Capital Expenditure ($B)</b>',
            showline=True,
            linecolor='white',
            linewidth=2,
            gridcolor=GRID,
            griddash='dash',
            zeroline=False
        ),
        yaxis=dict(
            title_text='<b>Annual Revenue ($B)</b>',
            tickformat='.0f',
            showline=True,
            linecolor='white',
            linewidth=2,
            gridcolor=GRID,
            griddash='dash',
            zeroline=False,
            minor=dict(
                showgrid=False,
                ticks=''
            )
        ),
        hoverlabel=dict(
            bgcolor='#061426',
            font_color='white',
            bordercolor='#00D4FF'
        ),
        showlegend=False
    )

    # Subtitle
    fig.add_annotation(
        x=0.59, y=1.2,
        xref='paper', yref='paper',
        xanchor='center',
        text='Exploring the Relationship Between Capital Expenditure and Annual Revenue',
        showarrow=False,
        font=dict(size=15, color='white')
    )

    # Bubble-size explanation
    fig.add_annotation(
        x=0.59, y=1.15,
        xref='paper', yref='paper',
        xanchor='center',
        text='Bubble Size = Market Capitalisation ($T)',
        showarrow=False,
        font=dict(size=13, color='#00D4FF')
    )

    return fig


# Add custom sector legend
def p2_add_sector_legend(fig, chart_df):
    """Draw a cleaner custom sector legend."""

    # Background box
    fig.add_shape(
        type='rect',
        xref='paper', yref='paper',
        x0=1.02, x1=1.20,
        y0=0.77, y1=1.04,
        line=dict(color='#1E3A5F', width=2),
        fillcolor='#061426'
    )

    # Heading
    fig.add_annotation(
        x=1.11, y=0.995,
        xref='paper', yref='paper',
        xanchor='center',
        yanchor='middle',
        text='<b>Sector</b>',
        showarrow=False,
        font=dict(size=14, color='white')
    )

    sector_colours = {
        'Semiconductors': '#00D4FF',
        'Software & Cloud': '#FF6B6B',
        'Internet Services': '#00E09F',
        'Consumer & EV': '#B477FF'
    }

    # Only show sectors actually present
    sectors = [
        sector for sector in sector_colours
        if sector in chart_df['Sector'].values
    ]

    # Tighter, cleaner spacing
    y_positions = [0.925, 0.865, 0.805, 0.785]

    for sector, y in zip(sectors, y_positions):
        # coloured circle
        fig.add_shape(
            type='circle',
            xref='paper', yref='paper',
            xsizemode='pixel',
            ysizemode='pixel',
            xanchor=1.055,
            yanchor=y,
            x0=-7, x1=7,
            y0=-7, y1=7,
            line=dict(color='white', width=2),
            fillcolor=sector_colours[sector]
        )

        # label
        fig.add_annotation(
            x=1.08, y=y,
            xref='paper', yref='paper',
            xanchor='left',
            yanchor='middle',
            text=sector,
            showarrow=False,
            font=dict(size=11, color='white')
        )

    return fig

# Add market capitalisation legend
def p2_add_market_cap_legend(fig, chart_df, size_max):
    """Draw the custom reference legend for bubble sizes."""

    # Background box
    fig.add_shape(
        type='rect',
        xref='paper', yref='paper',
        x0=1.02, x1=1.20,
        y0=0.34, y1=0.75,
        line=dict(color='#1E3A5F', width=2),
        fillcolor='#061426'
    )

    # Legend heading
    fig.add_annotation(
        x=1.11, y=0.70,
        xanchor='center',
        xref='paper', yref='paper',
        text='<b>Market Cap ($T)</b>',
        showarrow=False,
        font=dict(size=14, color='white')
    )

    # Calculate bubble sizes using Plotly's scaling method
    max_cap = chart_df['Market Cap ($T)'].max()

    for cap, y in [
        (0.5, 0.61),
        (1.0, 0.515),
        (2.0, 0.405)
    ]:
        size = size_max * 0.65 * np.sqrt(cap / max_cap)

        # Draw reference bubble
        fig.add_shape(
            type='circle',
            xref='paper', yref='paper',
            xsizemode='pixel',
            ysizemode='pixel',
            xanchor=1.06,
            yanchor=y,
            x0=-size/2, x1=size/2,
            y0=-size/2, y1=size/2,
            line=dict(color='white', width=2),
            fillcolor='#607D9F'
        )

        # Add market cap labels beside bubbles
        fig.add_annotation(
            x=1.1, y=y,
            xref='paper', yref='paper',
            text=f"${cap:.1f}T",
            showarrow=False,
            xanchor='left',
            font=dict(size=12, color='white')
        )

    return fig

# Add regression information and On/Off toggle
def p2_add_regression_controls(fig, r_squared):
    """Display R-squared and add buttons to show/hide regression."""

    # Regression information background
    fig.add_shape(
        type='rect',
        xref='paper', yref='paper',
        x0=1.02, x1=1.20,
        y0=0.01, y1=0.32,
        line=dict(color='#1E3A5F', width=2),
        fillcolor='#061426'
    )

    # Heading
    fig.add_annotation(
        x=1.11, y=0.285,
        xanchor='center',
        yanchor='middle',
        xref='paper', yref='paper',
        text='<b>Regression Analysis</b>',
        showarrow=False,
        font=dict(size=13, color='white')
    )

    # Handle unavailable regression
    if r_squared is None:
        fig.add_annotation(
            x=1.11, y=0.16,
            xref='paper', yref='paper',
            text='Regression unavailable',
            showarrow=False,
            font=dict(size=10, color='white')
        )

    else:
        # Explanation of R-squared
        fit_label = (
            f"{r_squared * 100:.1f}% of revenue variation<br>"
            "explained by CapEx"
        )

        fig.add_annotation(
            x=1.11, y=0.14,
            xanchor='center',
            xref='paper', yref='paper',
            text=fit_label,
            showarrow=False,
            font=dict(size=10, color='white')
        )

        interpretation_index = len(fig.layout.annotations) - 1

        # R-squared value
        fig.add_annotation(
            x=1.11, y=0.205,
            xanchor='center',
            xref='paper', yref='paper',
            text=f"<b>R² = {r_squared:.6f}</b>",
            showarrow=False,
            font=dict(size=15, color='#FFD166')
        )

        r2_index = len(fig.layout.annotations) - 1

        # Identify regression trace
        regression_trace_index = len(fig.data) - 1

        # On/Off controls
        fig.update_layout(
            updatemenus=[
                dict(
                    type='buttons',
                    direction='right',
                    showactive=True,
                    active=0,
                    x=1.11,
                    y=0.075,
                    xanchor='center',
                    yanchor='middle',
                    bgcolor='#607D9F',
                    bordercolor='#00D4FF',
                    borderwidth=1,
                    font=dict(size=12, color='black'),
                    pad=dict(l=2, r=2, t=2, b=2),
                    buttons=[
                        dict(
                            label='On',
                            method='update',
                            args=[
                                {'visible': [True]},
                                {
                                    f'annotations[{r2_index}].visible': True,
                                    f'annotations[{interpretation_index}].visible': True
                                },
                                [regression_trace_index]
                            ]
                        ),
                        dict(
                            label='Off',
                            method='update',
                            args=[
                                {'visible': [False]},
                                {
                                    f'annotations[{r2_index}].visible': False,
                                    f'annotations[{interpretation_index}].visible': False
                                },
                                [regression_trace_index]
                            ]
                        )
                    ]
                )
            ]
        )

    return fig


def p2_add_small_company_inset(fig, chart_df):
    """Zoomed comparison using the same financial data as the main chart."""

    tickers = ["NVDA", "AMD", "ASML", "PLTR"]
    small = chart_df[chart_df["Ticker"].isin(tickers)].copy()

    if small.empty:
        return fig

    colours = {
        "Semiconductors": "#00D4FF",
        "Software & Cloud": "#FF6B6B",
        "Internet Services": "#00E09F"
    }

    x_col = "Capital Expenditure ($B)"
    y_col = "Annual Revenue ($B)"
    cap_col = "Market Cap ($T)"

    # Inset axes and positioning

    # Inset axes and positioning
    fig.update_layout(
        xaxis2=dict(
            domain=[0.17, 0.45],
            anchor="y2",
            range=[-0.6, max(8, small[x_col].max() * 1.35)],
            title=dict(
                text="CapEx ($B)",
                font=dict(size=10),
                standoff=2
            ),
            tickfont=dict(size=10),
            gridcolor="#1E3A5F",
            showline=True,
            linecolor="white",
            zeroline=False,
            automargin=False
        ),
        yaxis2=dict(
            domain=[0.66, 0.88],
            anchor="x2",
            range=[-25, max(100, small[y_col].max() * 1.6)],
            title=dict(
                text="Revenue ($B)",
                font=dict(size=10),
                standoff=2
            ),
            tickfont=dict(size=10),
            gridcolor="#1E3A5F",
            showline=True,
            linecolor="white",
            zeroline=False,
            automargin=False
        )
    )

    # Background and turquoise border
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0.10,
        x1=0.50,
        y0=0.55,
        y1=0.99,
        fillcolor="#061426",
        line=dict(color="#00D4FF", width=2),
        layer="below"
    )

    # Title inside the inset
    fig.add_annotation(
        x=0.30,
        y=0.955,
        xref="paper",
        yref="paper",
        xanchor="center",
        yanchor="middle",
        align="center",
        text="<b>COMPANIES CLUSTER ZOOMED IN</b>",
        showarrow=False,
        font=dict(size=12, color="white")
    )

    # Scale bubble AREA by market capitalisation
    max_cap = small[cap_col].max()
    sizeref = 2 * max_cap / (46 ** 2)

    label_offsets = {
        "NVDA": (48, 0),
        "AMD": (0, -45),
        "ASML": (48, -22),
        "PLTR": (0, -25)
    }

    for _, row in small.iterrows():
        ticker = row["Ticker"]
        x = float(row[x_col])
        y = float(row[y_col])
        cap = float(row[cap_col])

        # Actual bubble, using proportional market cap
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                xaxis="x2",
                yaxis="y2",
                mode="markers",
                marker=dict(
                    size=[cap],
                    sizemode="area",
                    sizeref=sizeref,
                    sizemin=9,
                    color=colours.get(row["Sector"], "white"),
                    line=dict(color="white", width=2),
                    opacity=1
                ),
                name=ticker,
                showlegend=False,
                customdata=[[cap]],
                hovertemplate=(
                    "<b>" + ticker + "</b><br>"
                    "CapEx: $%{x:.2f}B<br>"
                    "Revenue: $%{y:.2f}B<br>"
                    "Market Cap: $%{customdata[0]:.2f}T"
                    "<extra></extra>"
                )
            )
        )

        # Separate labels with subtle connector lines
        ax, ay = label_offsets.get(ticker, (0, -25))

        fig.add_annotation(
            x=x,
            y=y,
            xref="x2",
            yref="y2",
            text="<b>" + ticker + "</b>",
            showarrow=True,
            ax=ax,
            ay=ay,
            arrowhead=0,
            arrowwidth=1,
            arrowcolor="#8CA5C0",
            bgcolor="#061426",
            borderpad=2,
            font=dict(size=11, color="white")
        )

    return fig


# Main function - combine all chart components
def p2_build_valuation_chart(chart_df, regression, show_inset=False):
    """Build the complete interactive valuation chart."""

    slope, intercept, r_squared, x = regression
    size_max = 75

    # Create bubble chart
    fig = p2_create_bubble_scatter(chart_df, size_max)

    # Add fitted regression line
    fig = p2_add_regression_line(fig, slope, intercept, x)

    # Apply chart styling
    fig = p2_style_valuation_chart(fig)

    # Add custom sector legend
    fig = p2_add_sector_legend(fig, chart_df)

    # Add market cap bubble-size legend
    fig = p2_add_market_cap_legend(fig, chart_df, size_max)

    # Add regression information and toggle
    fig = p2_add_regression_controls(fig, r_squared)

    # Only add zoom inset to the expanded Panel 2
    if show_inset:
        fig = p2_add_small_company_inset(fig, chart_df)

    return fig


def p2_create_app(selected_tickers=p2_tickers, **kwargs):
    app = Dash(__name__, **kwargs, include_assets_files=False)
    tab_style = {
        "backgroundColor": CARD,
        "color": "white",
        "border": "1px solid #1E3A5F",
        "padding": "12px",
    }
    selected_tab_style = {
        **tab_style,
        "backgroundColor": "#FFFFFF",
        "color": "#0B1220",
        "fontWeight": "bold",
        "borderTop": "2px solid #00D4FF",
    }
    app.layout = html.Div(
        [
            dcc.Location(id="p2-url", refresh=False),
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
            "backgroundColor": "#0B1220",
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
        Input("p2-url", "search"),
    )
    def update_view(view, _refresh, search):
        chosen = global_parse_selection(search)
        chosen = list(TICKERS) if chosen is None else chosen
        if not chosen:
            return html.P("Please select at least one company in the universal selector.", role="status")
        df, failed_tickers = p2_load_valuation_data(selected_tickers)
        df = df[df["Ticker"].isin(chosen)]
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
        fig = p2_build_valuation_chart(chart_df, regression, show_inset=True)
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
                    None,
                    dcc.Graph(
                        figure=fig,
                        style={"width": "100%", "height": "calc(100vh - 30px)"},
                        config={
                            "responsive": True,
                            "displayModeBar": False,
                            "displaylogo": False
                        },
                    ),
                ]
            )
        else:
            return html.Div(
                [
                    None,
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
                            "backgroundColor": CARD,
                            "color": "white",
                            "fontWeight": "bold",
                            "border": "1px solid #2D5474",
                            "padding": "15px",
                        },
                        style_cell={
                            "backgroundColor": CARD,
                            "color": "white",
                            "border": "1px solid #23445E",
                            "padding": "14px",
                            "textAlign": "left",
                            "fontFamily": "Arial",
                            "fontSize": "14px",
                        },
                        style_data_conditional=[
                            {"if": {"row_index": "odd"}, "backgroundColor": CARD}
                        ],
                    ),
                ]
            )

    return app


p4_NAVY = SHARED_NAVY
p4_CYAN = "#20e3e8"
p4_WHITE = "#f3f5fa"
p4_MUTED = "#aabed0"
p4_LINE = BORDER
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
    "AMD": "AMD",
    "TSMC": "TSM",
    "Oracle": "ORCL",
    "Palantir": "PLTR",
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

def p4_scenario_options(selected="Base"):
    descriptions = {"Bear": "Cautious", "Base": "Central", "Bull": "Optimistic"}
    colours = {"Bear": "#ffb47d", "Base": p4_CYAN, "Bull": "#8ce6ae"}

    return [
        {
            "value": name,
            "label": html.Div([
                html.Strong(
                    f"{name} - {descriptions[name]}"
                    + (" | SELECTED" if name == selected else "")
                ),
                html.Div(
                    f'{values["growth"]}% growth / {values["margin"]}% margin',
                    style={"fontSize": "13px", "marginTop": "5px"},
                ),
            ], style={
                "padding": "11px 12px",
                "borderRadius": "6px",
                "border": f"2px solid {colours[name]}",
                "backgroundColor": colours[name] if name == selected else p4_NAVY,
                "color": p4_NAVY if name == selected else p4_WHITE,
                "flex": "1",
            }),
        }
        for name, values in p4_SCENARIOS.items()
    ]


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
                                style={
                                    "color": p4_WHITE,
                                    "--Dash-Fill-Inverse-Strong": p4_NAVY,
                                    "--Dash-Text-Strong": p4_WHITE,
                                    "--Dash-Text-Weak": p4_MUTED,
                                    "--Dash-Stroke-Strong": p4_LINE,
                                    "--Dash-Fill-Interactive-Strong": p4_CYAN,
                                },
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
                                options=p4_scenario_options(),
                                value="Base",
                                style={"display": "grid", "gap": "6px"},
                                inputStyle={
                                    "accentColor": p4_CYAN,
                                    "marginRight": "8px",
                                },
                                labelStyle={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "cursor": "pointer",
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
                               style={"display": "none"},
                            ),
                        ],
                    style={"flex": "0 1 290px", "minWidth": "230px"},
                    ),
                    html.Main(
                        [
                            dcc.Loading(
                                html.Div(
                                    [
                                        html.Div(
                                            id="panel4-source",
                                        style={"display": "none"},    
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
                                                                                html.Button(
                                            "Reset view",
                                            id="panel4-reset-view",
                                            n_clicks=0,
                                            style={**p4_BUTTON, "margin": "0 0 8px"},
                                        ),
                                        dcc.Graph(
                                            id="panel4-heatmap",
                                            config={
                                                "displayModeBar": True,
                                                "displaylogo": False,
                                                "modeBarButtons": [
                                                    ["zoomIn2d", "zoomOut2d", "resetScale2d"]
                                                ],
                                                "doubleClick": "reset",
                                            },
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
        xaxis_title="<b>Revenue growth (%)</b>",
        yaxis_title="<b>Net profit margin (%)</b>",
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
    axis_style = dict(
        showline=True,
        linecolor=p4_WHITE,
        linewidth=2,
        ticks="outside",
        tickcolor=p4_WHITE,
        ticklen=6,
        tickfont=dict(size=14, color=p4_WHITE),
        title_font=dict(size=16, color=p4_WHITE),
        title_standoff=14,
        automargin=True,
        showgrid=False,
        zeroline=False,
    )
    figure.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(growth))),
        ticktext=[f"{g}%" for g in growth],
        **axis_style,
    )
    figure.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(margins))),
        ticktext=[f"{m}%" for m in margins],
        **axis_style,
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
            line={"color": p4_NAVY, "width": 4, "dash": dash},
            fillcolor="rgba(0,0,0,0)",
            layer="above",
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
        Output("panel4-scenario", "options"),
        Input("panel4-scenario", "value"),
    )
    def highlight_scenario(scenario):
        return p4_scenario_options(scenario)

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
        Input("panel4-reset-view", "n_clicks"),
        
    )
    def update_results(data, scenario, reset_clicks):
        metrics, figure, table, source, description = p4_render_results(data, scenario)
        if ctx.triggered_id == "panel4-reset-view":
            figure.update_layout(
                uirevision=f"reset-{reset_clicks}",
                xaxis_autorange=True,
                yaxis_autorange=True,
            )
        return metrics, figure, table, source, description


def cached_market_data(function):
    responses = {}
    lock = threading.Lock()

    @wraps(function)
    def wrapped(*args, **kwargs):
        key = repr((args, kwargs))
        try:
            trigger = ctx.triggered_id or ""
            refresh = ("refresh" in trigger or "retry" in trigger) and bool(
                ctx.triggered[0].get("value")
            )
        except Exception:
            refresh = False
        with lock:
            previous = responses.get(key)
            if previous and time.monotonic() - previous[0] < 900 and (not refresh):
                return previous[1]
            if refresh and hasattr(function, "cache_clear"):
                function.cache_clear()
            result = function(*args, **kwargs)
            if isinstance(result, pd.DataFrame) and result.empty:
                return result
            if (
                isinstance(result, tuple)
                and len(result) == 2
                and isinstance(result[0], pd.DataFrame)
                and result[1]
            ):
                return result
            responses[key] = (time.monotonic(), result)
            return result

    return wrapped


stock_get_prices = cached_market_data(stock_get_prices)
p2_load_valuation_data = cached_market_data(p2_load_valuation_data)
TITLES = [
    "AI Stock Performance",
    "Capital Investment vs Revenue",
    "AI Boom vs. Dot-Com Bubble",
    "AI Valuation Simulator",
]
NAVY = SHARED_NAVY
WHITE = "#edf3fc"
CYAN = "#5ce0d3"




def global_chip_style(selected):
    """One company toggle: cyan when picked."""
    style = {
        "padding": "3px 9px",
        "fontSize": "11px",
        "fontWeight": 600,
        "letterSpacing": "0.5px",
        "borderRadius": "12px",
        "border": f"1px solid {CYAN if selected else '#263449'}",
        "backgroundColor": "#143137" if selected else "#0f1b2d",
        "color": CYAN if selected else "#9fb2c8",
        "cursor": "pointer",
        "flex": "0 0 auto",
    }
    return style


def global_status(chosen):
    """Show the number selected from all twelve companies."""
    return html.Span(
        f"{len(chosen)} / {len(GLOBAL_COMPANIES)} selected", style={"color": CYAN}
    )


def compact_figure(figure, panel):
    """Fit each original chart to a responsive overview quadrant."""
    figure = go.Figure(figure)
    if panel == 4:
        # Retain Panel 4's original preview styling and independent data.
        figure.update_layout(autosize=True, height=None, width=None, title=None,
                             margin=dict(l=60, r=65, t=10, b=45), uirevision="overview-4")
        return figure
    figure.update_layout(autosize=True, height=None, width=None, title=None,
        paper_bgcolor=SHARED_NAVY, plot_bgcolor=SHARED_NAVY, font=dict(size=11),
        margin=dict(l=55, r=24, t=38, b=48), uirevision=f"overview-{panel}")
    if panel == 1:
        figure.update_layout(margin=dict(l=0, r=0, t=30, b=8),
            scene=dict(domain=dict(x=[0, 1], y=[0, 0.94]),
            aspectratio=dict(x=2.2, y=1.0, z=0.8),
            camera=dict(eye=dict(x=1.4, y=-2.6, z=1.1))),
            legend=dict(orientation="h", x=0.02, xanchor="left", y=1.02,
                        yanchor="bottom", font=dict(size=9), bgcolor=SHARED_NAVY))
        figure.update_scenes(xaxis_title_font_size=11, yaxis_title_font_size=11, zaxis_title_font_size=11,
                             xaxis_tickfont_size=9, yaxis_tickfont_size=9, zaxis_tickfont_size=9)
    elif panel == 2:
        # Remove expanded-panel annotations and custom legend boxes
        figure.layout.annotations = []
        figure.layout.shapes = []
        figure.layout.updatemenus = []
        
        sectors = set()
        for trace in figure.data:
            if trace.type == "scatter" and trace.marker.sizeref:
                trace.marker.sizeref *= 1.7
                trace.showlegend = trace.name not in sectors
                sectors.add(trace.name)
            else:
                trace.showlegend = False
        figure.update_layout(showlegend=False)
        figure.update_xaxes(domain=[0, 1], autorange=True, title_font_size=11, automargin=True)
        figure.update_yaxes(
             autorange=True,
             tickmode="linear",
             tick0=0,
             dtick=250,
             title_font_size=11,
             tickfont_size=11,
             automargin=True
        )
    elif panel == 3:
        # Compact overview keys only; full-size names and annotations stay intact.
        for trace, label in zip(figure.data, ["NASDAQ · Dot-Com", "AI Portfolio", "Bull", "Base", "Bear"]):
            trace.name = label
        figure.update_layout(margin=dict(l=55, r=110, t=42, b=46),
            legend=dict(orientation="h", x=0, y=1.04, yanchor="bottom", font=dict(size=9)))
        figure.update_xaxes(title_font_size=11, tickfont_size=10)
        figure.update_yaxes(title_font_size=11, tickfont_size=10)
    return figure


def message_figure(message):
    figure = go.Figure()
    figure.update_layout(
        paper_bgcolor=NAVY,
        plot_bgcolor=NAVY,
        font=dict(color=WHITE),
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
            )
        ],
    )
    return figure


def overview_figure(panel, companies=None):
    """Reuse the original panel calculations and figures with their real data.

    Panels 1-3 use the header selection; Panel 4 remains independent.
    """
    chosen = (
        list(stock_DEFAULT_COMPANIES)
        if companies is None else global_normalise_selection(companies)
    )
    if panel in GLOBAL_PANELS and not chosen:
        return message_figure("Please select at least one company in the universal selector.")
    if panel == 1:
        indexed, _missing = stock_comparison_data(chosen, "3 Years", list(stock_BENCHMARKS))
        sampled = indexed.iloc[::5]
        if sampled.index[-1] != indexed.index[-1]:
            sampled = pd.concat([sampled, indexed.iloc[[-1]]])
        figure = stock_build_chart_3d(sampled)
    elif panel == 2:
        data, failed = p2_load_valuation_data(p2_tickers)
        cleaned = p2_prepare_chart_data(data[data["Ticker"].isin(chosen)])
        if cleaned.empty:
            raise ValueError("Yahoo Finance metrics unavailable. Open panel to retry.")
        figure = p2_build_valuation_chart(cleaned, p2_calculate_regression(cleaned))
    elif panel == 3:
        figure = panel3.render_panel3(chosen)
    else:
        data = p4_load_company_data("NVIDIA")
        figure = p4_make_heatmap(data, p4_SCENARIOS["Base"])
    return compact_figure(figure, panel)


def create_app():
    server = Flask(__name__)
    panel_apps = []
    for number, factory in enumerate([p1_create_app, p2_create_app, panel3.create_app], 1):
        panel_apps.append(
            factory(
                server=server,
                url_base_pathname=f"/panel/{number}/",
                title=TITLES[number - 1],
            )
        )
    fourth = Dash(
        "simulator",
        server=server,
        url_base_pathname="/panel/4/",
        title=TITLES[3],
        include_assets_files=False,
    )
    fourth.layout = p4_create_layout()
    p4_register_callbacks(fourth)
    panel_apps.append(fourth)
    for panel_app in panel_apps:
        apply_theme(panel_app)
        panel_app.layout = html.Div(
            panel_app.layout,
            style={
                "position": "absolute",
                "inset": 0,
                "backgroundColor": p4_NAVY if panel_app is fourth else NAVY,
                "minHeight": "100vh",
                "color": WHITE,
            },
        )
    app = Dash(
        __name__,
        server=server,
        title="AI Boom • Four Graph Dashboard",
        include_assets_files=False,
    )
    apply_theme(app)
    cards = []
    for number, title in enumerate(TITLES, 1):
        cards.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                f"0{number}",
                                style={"color": CYAN, "marginRight": "10px"},
                            ),
                            title,
                        ],
                        style={
                            "height": "36px",
                            "lineHeight": "36px",
                            "padding": "0 14px",
                            "fontSize": "16px",
                            "fontWeight": 700,
                            "backgroundColor": CARD,
                        },
                    ),
                    html.Div(
                        dcc.Loading(
                            dcc.Graph(
                                id=f"graph-{number}",
                                figure=message_figure("Loading Yahoo Finance data…"),
                                responsive=True,
                                config={"displayModeBar": False, "displaylogo": False},
                                style={"height": "100%", "width": "100%"},
                            ),
                            parent_style={"height": "100%", "width": "100%"},
                        ),
                        style={
                            "position": "absolute",
                            "top": "36px",
                            "bottom": 0,
                            "left": 0,
                            "right": 0,
                        },
                    ),
                    html.Button(
                        "",
                        id=f"open-{number}",
                        n_clicks=0,
                        title=f"Open {title}",
                        **{"aria-label": f"Open {title}"},
                        style={
                            "position": "absolute",
                            "inset": 0,
                            "width": "100%",
                            "height": "100%",
                            "border": 0,
                            "background": "transparent",
                            "cursor": "pointer",
                            "zIndex": 2,
                        },
                    ),
                ],
                id=f"card-{number}",
                className="overview-card",
                style={
                    "position": "relative",
                    "minWidth": 0,
                    "minHeight": 0,
                    "overflow": "hidden",
                    "backgroundColor": NAVY,
                },
            )
        )
    grid_style = {
        "height": "calc(100vh - 52px)",
        "display": "grid",
        "gridTemplateColumns": "repeat(2,minmax(0,1fr))",
        "gridTemplateRows": "repeat(2,minmax(0,1fr))",
        "gap": "1px",
        "backgroundColor": "#263449",
    }
    chips = [
        html.Button(
            ticker,
            id=f"chip-{ticker}",
            n_clicks=0,
            title=f"{name} ({ticker})",
            disabled=False,
            style=global_chip_style(ticker in stock_DEFAULT_COMPANIES),
        )
        for ticker, name in stock_COMPANIES.items()
    ]
    frames = [
        html.Div(
            html.Iframe(
                id=f"frame-{i}",
                title=TITLES[i - 1],
                style={"border": 0, "width": "100%", "height": "100%"},
            ),
            id=f"full-{i}",
            style={"display": "none", "height": "calc(100vh - 52px)"},
        )
        for i in range(1, 5)
    ]
    app.layout = html.Div(
        [
            dcc.Interval(id="load-data", interval=150, max_intervals=1),
            dcc.Store(id="company-selection", data=list(stock_DEFAULT_COMPANIES)),
            dcc.Store(id="applied-companies", data=list(stock_DEFAULT_COMPANIES)),
            dcc.Store(id="current-panel", data=None),
            html.Div(
                [
                    html.Div(
                        [
                            html.Strong(
                                "AI BOOM",
                                style={
                                    "color": CYAN,
                                    "letterSpacing": "2px",
                                    "flex": "0 0 auto",
                                },
                            ),
                            html.Span(
                                "COMPANIES",
                                title=(
                                    f"Pick any of the "
                                    f"{len(GLOBAL_COMPANIES)} companies. Applies to "
                                    "panels 1-3. Panel 4 has its own company selector."
                                ),
                                style={
                                    "color": "#7f93ad",
                                    "fontSize": "10px",
                                    "letterSpacing": "1.5px",
                                    "flex": "0 0 auto",
                                },
                            ),
                            html.Div(
                                chips,
                                id="company-chips",
                                style={
                                    "display": "flex",
                                    "gap": "6px",
                                    "alignItems": "center",
                                    "minWidth": "140px",
                                    "overflowX": "auto",
                                    "scrollbarWidth": "none",
                                },
                            ),
                            html.Button("Select All", id="select-all", n_clicks=0, style={"whiteSpace": "nowrap"}),
                            html.Button("Deselect All", id="deselect-all", n_clicks=0, style={"whiteSpace": "nowrap"}),
                            html.Span(
                                global_status(list(stock_DEFAULT_COMPANIES)),
                                id="selection-status",
                                style={
                                    "fontSize": "11px",
                                    "whiteSpace": "nowrap",
                                    "flex": "0 0 auto",
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "8px",
                            "minWidth": 0,
                            "overflowX": "auto",
                        },
                    ),
                    html.Button(
                        "← Return to overview",
                        id="back",
                        n_clicks=0,
                        style={"display": "none"},
                    ),
                ],
                style={
                    "height": "52px",
                    "padding": "0 12px",
                    "overflowX": "auto",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                    "gap": "16px",
                    "backgroundColor": "#0B1220",
                },
            ),
            html.Div(cards, id="grid", className="overview-grid", style=grid_style),
            *frames,
        ],
        style={
            "position": "fixed",
            "inset": 0,
            "backgroundColor": NAVY,
            "color": WHITE,
            "fontFamily": "Segoe UI, Arial, sans-serif",
        },
    )

    @app.callback(
        Output("company-selection", "data"),
        *[Input(f"chip-{ticker}", "n_clicks") for ticker in GLOBAL_COMPANIES],
        Input("select-all", "n_clicks"), Input("deselect-all", "n_clicks"),
        State("company-selection", "data"),
        prevent_initial_call=True,
    )
    def toggle_company(*args):
        *_clicks, current = args
        trigger = ctx.triggered_id
        if trigger == "select-all":
            return list(TICKERS)
        if trigger == "deselect-all":
            return []
        if not trigger or not str(trigger).startswith("chip-"):
            raise PreventUpdate
        ticker = str(trigger).split("-", 1)[1]
        chosen = global_normalise_selection(current)
        if ticker in chosen:
            chosen.remove(ticker)
        elif ticker in GLOBAL_COMPANIES:
            chosen.append(ticker)
        else:
            raise PreventUpdate
        return global_normalise_selection(chosen)

    @app.callback(
        *[Output(f"chip-{ticker}", "style") for ticker in GLOBAL_COMPANIES],
        *[Output(f"chip-{ticker}", "disabled") for ticker in GLOBAL_COMPANIES],
        Output("selection-status", "children"),
        Input("company-selection", "data"),
    )
    def show_selection(selection):
        chosen = global_normalise_selection(selection)
        return [
            *[global_chip_style(t in chosen) for t in GLOBAL_COMPANIES],
            *[False for t in GLOBAL_COMPANIES],
            global_status(chosen),
        ]

    @app.callback(
        Output("applied-companies", "data"),
        Input("company-selection", "data"),
        prevent_initial_call=True,
    )
    def apply_selection(selection):
        chosen = global_normalise_selection(selection)
        return chosen

    def register_preview(number):
        uses_selection = number in GLOBAL_PANELS
        inputs = [Input("load-data", "n_intervals")]
        if uses_selection:
            inputs.append(Input("applied-companies", "data"))

        @app.callback(
            Output(f"graph-{number}", "figure"),
            *inputs,
            prevent_initial_call=True,
        )
        def load(_count, companies=None):
            try:
                return overview_figure(number, companies)
            except Exception as error:
                print(f"Panel {number}: {error}")
                return message_figure("Data unavailable — open panel to retry")

    for number in range(1, 5):
        register_preview(number)

    @app.callback(
        Output("grid", "style"),
        Output("back", "style"),
        *[Output(f"full-{i}", "style") for i in range(1, 5)],
        *[Output(f"frame-{i}", "src") for i in range(1, 5)],
        Output("current-panel", "data"),
        *[Input(f"open-{i}", "n_clicks") for i in range(1, 5)],
        Input("back", "n_clicks"),
        Input("applied-companies", "data"),
        State("current-panel", "data"),
        prevent_initial_call=True,
    )
    def navigate(*args):
        *_clicks, applied, open_panel = args
        trigger = ctx.triggered_id
        chosen = global_normalise_selection(applied)
        if trigger == "applied-companies":
            if open_panel not in GLOBAL_PANELS:
                raise PreventUpdate
            selected = open_panel
        else:
            selected = (
                int(trigger.split("-")[1])
                if trigger and trigger.startswith("open-")
                else None
            )
        grid = {**grid_style, "display": "none" if selected else "grid"}
        back = {
            "display": "block" if selected else "none",
            "backgroundColor": "#143137",
            "color": CYAN,
            "border": "1px solid #5ce0d3",
            "borderRadius": "5px",
            "padding": "6px 12px",
            "cursor": "pointer",
        }
        full = [
            {
                "display": "block" if selected == i else "none",
                "height": "calc(100vh - 52px)",
            }
            for i in range(1, 5)
        ]
        sources = [
            global_panel_url(i, chosen) if selected == i else no_update
            for i in range(1, 5)
        ]
        return [grid, back, *full, *sources, selected]
    

    app.panel_apps = panel_apps
    return app


app = create_app()
server = app.server
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-file Python Dash dashboard")
    parser.add_argument("--port", type=int, default=8050)
    arguments = parser.parse_args()
    print(f"Open http://127.0.0.1:{arguments.port} in your browser.")
    app.run(host="127.0.0.1", port=arguments.port, debug=False)

