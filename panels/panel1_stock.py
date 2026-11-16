#Panel 1: AI Stock Performance

import pandas as pd
import numpy as np

#Library for historical share prices
import yfinance as yf

#Libraries for the dashboard and interactive graphs
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

def get_prices(tickers, start, end):
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

def index_and_normalise(prices):
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

def calculate_statistics(indexed_prices):
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


# INTERACTIVE 3D STOCK PERFORMANCE CHART


def build_chart_3d(indexed_prices):
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
        colour = STOCK_COLOURS[number % len(STOCK_COLOURS)]
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
        margin=dict(l=0, r=0, t=50, b=0),

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


# DASH DASHBOARD AND INTERACTIVE CONTROLS

def create_panel1_layout():

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

app.layout = create_panel1_layout()


# DASH CALLBACK

@app.callback(
    Output("panel1_results", "children"),
    Input("panel1_companies", "value"),
    Input("panel1_period", "value"),
    Input("panel1_benchmarks", "value"),
    Input("panel1_detail", "value")
)
def update_panel1(
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
        prices = get_prices(
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
    indexed_prices = index_and_normalise(prices)

    if len(indexed_prices) < 3:
        return html.P(
            "Not enough shared historical data to compare."
        )

    # Calculate performance statistics
    statistics = calculate_statistics(indexed_prices)

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

    figure = build_chart_3d(chart_prices)

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

        colour = STOCK_COLOURS[
            number % len(STOCK_COLOURS)
        ]

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


# RUN DASH APPLICATION

if __name__ == "__main__":
    app.run(debug=True)


