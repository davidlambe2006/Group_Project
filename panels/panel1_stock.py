#Panel 1: AI Stock Performance

import pandas as pd
import numpy as np

#Library for historical share prices
import yfinance as yf

#Libraries for the dashboard and interactive
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
#Python tools for handling dates
from datetime import date, timedelta


try:
    from .dashboard_shared import COMPANIES as SHARED_COMPANIES, TICKERS, NAVY as SHARED_NAVY, CARD, BORDER, GRID
except ImportError:  # Direct execution of a panel demo.
    from dashboard_shared import COMPANIES as SHARED_COMPANIES, TICKERS, NAVY as SHARED_NAVY, CARD, BORDER, GRID

#Company Information

#Dictionary linking stock ticker symbols to company names
COMPANIES = dict(SHARED_COMPANIES)
DEFAULT_COMPANIES = list(TICKERS)

# Stock market ETFs used as comparison benchmarks
BENCHMARKS = {
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq-100 ETF"
}

#Defining dashboard colours

#Background colour for the dashboard
DARK_BLUE = SHARED_NAVY

# Colours used for different company lines
STOCK_COLOURS = [
    "#41DDB4",  # Green
    "#5BA8FF",  # Blue
    "#FBC56C",  # Yellow
    "#FC8F73",  # Orange
    "#C29BFF",  # Purple
    "#6AE6E6",  # Cyan
    "#E879BD",  # Pink
    "#AFD873",  # Light green
    "#FF748E",  # Red
    "#CED4FF",  # Light purple
    "#E5B77A",  # Gold
    "#83BBE9",  # Light blue
    "#FFFFFF",  # White
    "#A8B9CF"   # Grey
]
CARD_BLUE = CARD
TEXT_MUTED = "#A8B9CF"
# Colours follow the ticker rather than its position, so a company keeps its
# colour as the selection changes.
COLOUR_BY_TICKER = dict(zip(
    ["NVDA", "MSFT", "GOOGL", "AMZN", "META", "AMD", "SPY", "QQQ", "TSM", "ORCL", "PLTR", "ASML"],
    [STOCK_COLOURS[i] for i in [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13]]
))


def p1_ticker_colour(ticker, position):
    """Colour for a ticker; falls back to its position in the palette."""
    return COLOUR_BY_TICKER.get(
        ticker, STOCK_COLOURS[position % len(STOCK_COLOURS)]
    )

PERIOD_OPTIONS = [
    "1 Year",
    "3 Years",
    "5 Years",
    "10 Years"
]

DETAIL_OPTIONS = [
    "Daily",
    "Every 5 Days",
    "Every 20 Days"
]

# Downloading Stock Data with yfinance

def p1_get_prices(tickers, start, end):
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
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    if stock_data.empty:
        return pd.DataFrame()  # Return an empty DataFrame if no data is fetched

     # Select the closing prices from the dataset.
    if isinstance(stock_data.columns, pd.MultiIndex):
        closing_prices = stock_data["Close"]

    else:
        closing_prices = stock_data[["Close"]]
        closing_prices.columns = [tickers[0]]
     # Ensure a DataFrame, even for one company.
    if isinstance(closing_prices, pd.Series):
        closing_prices = closing_prices.to_frame()

    # Keep the company columns in the requested order.
    closing_prices = closing_prices.reindex(columns=tickers)

    # Convert prices to numbers and remove invalid values.
    closing_prices = closing_prices.apply(
        pd.to_numeric, errors="coerce"
    )

    # Remove dates where all prices are missing.
    closing_prices = closing_prices.dropna(how="all")

    # Organise dates from oldest to newest.
    closing_prices = closing_prices.sort_index()

    return closing_prices

#Indexing and Normalising Stock Prices

def p1_index_and_normalise(prices):
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

    # Keep positive prices and dates shared by all companies.
    common_prices = prices.where(prices > 0).dropna(how="any")

    # Check that enough observations exist for comparison.
    if len(common_prices) < 2:
        return pd.DataFrame()

    # Find the first available share price for each company.
    starting_prices = common_prices.iloc[0]

    # Divide by starting prices and multiply by 100.
    indexed_prices = common_prices.div(starting_prices).mul(100)

    return indexed_prices


# Financial Performance Statistics

def p1_calculate_statistics(indexed_prices):
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

    # Check that enough observations are available.
    if len(indexed_prices) < 3:
        return pd.DataFrame()

    # Calculate the daily percentage changes in stock prices.
    daily_returns = indexed_prices.pct_change().dropna()

    # Calculate total percentage return over the selected period.
    total_return = (
        indexed_prices.iloc[-1] / indexed_prices.iloc[0] - 1
    ) * 100

    # Calculate annualised volatility from daily returns.
    volatility = (
        daily_returns.std() * np.sqrt(252) * 100
    )

    # Find the highest previous indexed price for each stock.
    previous_peaks = indexed_prices.cummax()

    # Calculate the percentage decline from previous peaks.
    drawdowns = (
        indexed_prices / previous_peaks - 1
    ) * 100

    # Find the largest historical decline for each company.
    max_drawdown = drawdowns.min()

    # Combine the statistics into one pandas DataFrame.
    statistics = pd.DataFrame({
        "Total Return (%)": total_return,
        "Annualised Volatility (%)": volatility,
        "Maximum Drawdown (%)": max_drawdown
    })

    statistics.index.name = "Ticker"

    return statistics



# One download-and-normalise path, shared by the full panel and the overview

def p1_comparison_data(companies, period="3 Years", benchmarks=None):
    """
    Download, clean and index prices for the companies plus any benchmarks.

    Returns
    -------
    tuple
        (indexed_prices, missing_tickers). Raises ValueError when there is
        not enough data to compare.
    """

    tickers = list(dict.fromkeys(list(companies) + list(benchmarks or [])))

    start = date.today() - timedelta(
        days=round(int(period.split()[0]) * 365.25)
    )
    end = date.today() + timedelta(days=1)

    prices = p1_get_prices(tickers, start.isoformat(), end.isoformat())

    # Identify tickers with no usable prices.
    missing = [
        ticker for ticker in tickers
        if ticker not in prices or prices[ticker].isna().all()
    ]
    prices = prices.drop(columns=missing, errors="ignore")

    if not any(ticker in prices for ticker in companies):
        raise ValueError("No selected AI companies have available data.")

    indexed = p1_index_and_normalise(prices)

    if len(indexed) < 3:
        raise ValueError("Not enough shared historical data to compare.")

    return indexed, missing
# INTERACTIVE 3D STOCK PERFORMANCE CHART



def p1_build_chart_3d(indexed_prices):
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
        colour = p1_ticker_colour(ticker, number)
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
                hovertemplate=(
                    f"Company: {ticker}<br>"
                    "Date: %{text}<br>"
                    "Indexed Price: %{z:.2f}"
                    "<extra></extra>"
                )
            )
        )

    
        x_mesh = np.concatenate([days, days])
        y_mesh = np.concatenate([
            np.full(len(days), y_level),
            np.full(len(days), y_level)
        ])
        z_mesh = np.concatenate([z_values, np.full(len(days), baseline)])

        i = []
        j = []
        k = []

        # Create triangles joining the top line to the baseline
        for n in range(len(days) - 1):
            # Triangle 1
            i.append(n)
            j.append(n + 1)
            k.append(len(days) + n)

            # Triangle 2
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
                showlegend=False
            )
        )

    # Choose readable date ticks
    tick_positions = np.linspace(
        0,
        len(dates) - 1,
        min(6, len(dates)),
        dtype=int
    )

    fig.update_layout(
        title="AI Stock Performance (3D)",
        template="plotly_dark",
        paper_bgcolor=DARK_BLUE,
        font=dict(color="white"),
        height=620,
        margin=dict(l=0, r=0, t=50, b=55),

        scene=dict(
            xaxis=dict(
                title="Date",
                tickvals=days[tick_positions],
                ticktext=dates[tick_positions].strftime("%b %Y"),
                gridcolor=GRID,
                backgroundcolor=DARK_BLUE
            ),
            yaxis=dict(
                title="Company",
                tickvals=list(range(len(indexed_prices.columns))),
                ticktext=list(indexed_prices.columns),
                gridcolor=GRID,
                backgroundcolor=DARK_BLUE
            ),
            zaxis=dict(
                title="Indexed Price (Base = 100)",
                gridcolor=GRID,
                backgroundcolor=DARK_BLUE
            ),
            bgcolor=DARK_BLUE,
            aspectratio=dict(x=2.2, y=1.0, z=1.2),
            camera=dict(
                eye=dict(x=1.5, y=-1.9, z=1.1)
            )
        ),

        legend=dict(
            orientation="v",
            x=1.02,
            y=0.95
        )
    )

    # Improve the readability of the chart text.
    fig.update_layout(
        font=dict(color="white"),
        title_font=dict(color="white", size=18),

        legend=dict(
            font=dict(color="white", size=13),
            bgcolor="rgba(10, 24, 48, 0.8)",
            orientation="v",
            x=0.99,
            xanchor="right",
            y=0.98
        ),

        height=560
    )

    # Make the 3D axis labels white as well.
    fig.update_scenes(
        xaxis_color="white",
        yaxis_color="white",
        zaxis_color="white"
    )
    return fig


# FIGURE FOR THE DASHBOARD OVERVIEW

def p1_render(companies):
    """
    Build the Panel 1 chart for the overview grid.

    Uses the companies chosen in the dashboard header, both benchmarks,
    three years of history and every fifth trading day.
    """

    indexed, _missing = p1_comparison_data(companies, "3 Years", list(BENCHMARKS))

    sampled = indexed.iloc[::5]

    # Include the latest observation
    if sampled.index[-1] != indexed.index[-1]:
        sampled = pd.concat([sampled, indexed.iloc[[-1]]])

    return p1_build_chart_3d(sampled)

# DASH DASHBOARD AND INTERACTIVE CONTROLS

def p1_create_panel1_layout():

    return html.Div([

        # Dashboard heading
        html.H2("Panel 1 — AI Stock Performance (3D)"),

        html.P(
            "How have AI-related companies performed "
            "relative to the wider stock market?"
        ),

        # AI company selector
        html.Label("AI Companies"),

        dcc.Dropdown(
            id="panel1_companies",
            options=[
                {"label": name, "value": ticker}
                for ticker, name in COMPANIES.items()
            ],
            value=DEFAULT_COMPANIES,
            multi=True
        ),

        # Three dashboard controls
        html.Div([

            html.Div([
                html.Label("Historical Period"),

                dcc.Dropdown(
                    id="panel1_period",
                    options=[
                        "1 Year", "3 Years",
                        "5 Years", "10 Years"
                    ],
                    value="3 Years",
                    clearable=False
                )
            ], style={"flex": "1", "minWidth": "200px"}),

            html.Div([
                html.Label("Market Benchmarks"),

                dcc.Dropdown(
                    id="panel1_benchmarks",
                    options=[
                        {"label": name, "value": ticker}
                        for ticker, name in BENCHMARKS.items()
                    ],
                    value=["SPY", "QQQ"],
                    multi=True
                )
            ], style={"flex": "1", "minWidth": "200px"}),

            html.Div([
                html.Label("Chart Detail"),

                dcc.Dropdown(
                    id="panel1_detail",
                    options=[
                        "Daily",
                        "Every 5 Days",
                        "Every 20 Days"
                    ],
                    value="Every 5 Days",
                    clearable=False
                )
            ], style={"flex": "1", "minWidth": "200px"})

        ], style={
            "display": "flex",
            "flexWrap": "wrap",
            "gap": "20px",
            "marginTop": "20px",
            "marginBottom": "25px"
        }),

        # Chart and statistics will appear here
        dcc.Loading(
            html.Div(id="panel1_results"),
            type="circle"
        )

    ], style={
        "backgroundColor": DARK_BLUE,
        "color": "white",
        "padding": "30px",
        "minHeight": "100vh",
        "fontFamily": "Arial"
    })


# CREATE DASH APPLICATION

app = Dash(__name__, title="AI Boom Dashboard")

app.layout = p1_create_panel1_layout()


# DASH CALLBACK

@app.callback(
    Output("panel1_results", "children"),
    Input("panel1_companies", "value"),
    Input("panel1_period", "value"),
    Input("panel1_benchmarks", "value"),
    Input("panel1_detail", "value")
)
def p1_update_panel1(
    selected_companies,
    period,
    selected_benchmarks,
    detail
):

    # Check selected companies
    if not selected_companies:
        return html.P("Please select at least one AI company.")

    selected_benchmarks = selected_benchmarks or []

    # Combine companies and benchmarks
    tickers = list(dict.fromkeys(
        selected_companies + selected_benchmarks
    ))

    # Calculate historical period
    years = int(period.split()[0])

    end_date = date.today() + timedelta(days=1)

    start_date = date.today() - timedelta(
        days=round(years * 365.25)
    )

    # Download stock prices
    try:
        prices = p1_get_prices(
            tickers,
            start_date.isoformat(),
            end_date.isoformat()
        )

    except Exception as error:
        return html.P(
            f"Unable to download stock data: {error}"
        )

    if prices.empty:
        return html.P("No stock data was returned.")

    # Identify missing prices
    missing = [
        ticker for ticker in tickers
        if ticker not in prices.columns
        or prices[ticker].isna().all()
    ]

    prices = prices.drop(
        columns=missing,
        errors="ignore"
    )

    if not any(
        ticker in prices.columns
        for ticker in selected_companies
    ):
        return html.P(
            "No selected AI companies have available data."
        )

    # Normalise prices
    indexed_prices = p1_index_and_normalise(prices)

    if len(indexed_prices) < 3:
        return html.P(
            "Not enough shared historical data to compare."
        )

    # Calculate performance statistics
    statistics = p1_calculate_statistics(indexed_prices)

    # Select chart detail
    if detail == "Daily":
        step = 1
    elif detail == "Every 5 Days":
        step = 5
    else:
        step = 20

    chart_prices = indexed_prices.iloc[::step].copy()

    # Include the latest observation
    if chart_prices.index[-1] != indexed_prices.index[-1]:
        chart_prices = pd.concat([
            chart_prices,
            indexed_prices.iloc[[-1]]
        ])

    # LEFT COLUMN: 3D stock performance graph

    figure = p1_build_chart_3d(chart_prices)

    chart_col = html.Div(
        dcc.Graph(
            id="panel1_3d_chart",
            figure=figure,
            responsive=True,
            style={
                "width": "100%",
                "height": "560px"
            }
        ),
        style={
            "flex": "3",
            "minWidth": "0"
        }
    )

    # RIGHT COLUMN: Performance statistics

    cards = []

    for number, ticker in enumerate(indexed_prices.columns):

        colour = p1_ticker_colour(ticker, number)

        total_return = statistics.loc[
            ticker, "Total Return (%)"
        ]

        company_name = COMPANIES.get(
            ticker, BENCHMARKS.get(ticker, ticker)
        )

        # Create performance card
        card = html.Div([

            html.Span(
                f"{company_name} ({ticker})",
                style={
                    "color": "white",
                    "fontSize": "13px"
                }
            ),

            html.Span(
                f"{total_return:+,.1f}%",
                style={
                    "color": colour,
                    "fontWeight": "bold",
                    "fontSize": "15px"
                }
            )

        ], style={
            "backgroundColor": "#131F31",
            "borderLeft": f"4px solid {colour}",
            "borderRadius": "7px",
            "padding": "12px",
            "marginBottom": "8px",
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "gap": "10px"
        })

        cards.append(card)

    stats_col = html.Div([

        html.H4("Total Return"),

        *cards

    ], style={
        "flex": "1",
        "minWidth": "250px"
    })

    # Combine chart and statistics columns

    result = html.Div(
        [chart_col, stats_col],
        style={
            "display": "flex",
            "flexWrap": "wrap",
            "gap": "24px"
        }
    )

    # Data source information
    caption = html.P(
        f"Source: Yahoo Finance via yfinance | "
        f"Actual comparison period: "
        f"{indexed_prices.index[0]:%d %b %Y} to "
        f"{indexed_prices.index[-1]:%d %b %Y} | "
        "Prices adjusted for stock splits and dividends.",
        style={
            "color": "#A8B9CF",
            "fontSize": "12px",
            "marginTop": "20px"
        }
    )

    output = []

    if missing:
        output.append(
            html.P(
                "Data unavailable for: " + ", ".join(missing),
                style={"color": "#FBC56C"}
            )
        )

    output.extend([result, caption])

    return html.Div(output)


# Integrated dashboard


def p1_comparison_data(companies, period="3 Years", benchmarks=None, *, load_prices=p1_get_prices):
    """One real-price/normalisation path for both Panel 1 views.

    Preserve the original lookback, adjusted closes and shared-date indexing.
    """
    tickers = list(dict.fromkeys(list(companies) + list(benchmarks or [])))
    start = date.today() - timedelta(days=round(int(period.split()[0]) * 365.25))
    prices = load_prices(tickers, start.isoformat(), (date.today() + timedelta(days=1)).isoformat())
    missing = [ticker for ticker in tickers if ticker not in prices or prices[ticker].isna().all()]
    prices = prices.drop(columns=missing, errors="ignore")
    if not any(ticker in prices for ticker in companies):
        raise ValueError("No selected AI companies have available data.")
    indexed = p1_index_and_normalise(prices)
    if len(indexed) < 3:
        raise ValueError("Not enough shared historical data to compare.")
    return indexed, missing


def p1_create_app(*, parse_selection, load_prices=p1_get_prices, **kwargs):
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
                            options=list(BENCHMARKS),
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
            "background": DARK_BLUE,
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
        chosen = parse_selection(search)
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
            indexed, missing = p1_comparison_data(companies, period, benchmarks, load_prices=load_prices)
            stats = p1_calculate_statistics(indexed)
            step = {"Daily": 1, "Every 5 Days": 5, "Every 20 Days": 20}[detail]
            sampled = indexed.iloc[::step]
            if sampled.index[-1] != indexed.index[-1]:
                sampled = pd.concat([sampled, indexed.iloc[[-1]]])
            cards = [html.H4("Total Return")]
            for i, ticker in enumerate(indexed.columns):
                colour = p1_ticker_colour(ticker, i)
                cards.append(
                    html.Div(
                        [
                            html.Span(
                                f"{COMPANIES.get(ticker, BENCHMARKS.get(ticker, ticker))} ({ticker})"
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
                            figure=p1_build_chart_3d(sampled),
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

# RUN DASH APPLICATION

if __name__ == "__main__":
    app.run(debug=True)


