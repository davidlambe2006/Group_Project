#Panel 1: AI Stock Performance

#Libraries for handling data and calculations
import pandas as pd
import numpy as np

#Library for historical share prices
import yfinance as yf

#Libraries for the dashboard and interactive graphs
import streamlit as st
import plotly.graph_objects as go

#Python tools for handling dates
from datetime import datetime, timedelta

#Company Information

#Dictionary linking stock ticker symbols to company names
COMPANIES = {
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
    "ASML": "ASML"
}
# Stock market ETFs used as comparison benchmarks
BENCHMARKS = {
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq-100 ETF"
}

#Defining dashboard colours

#Background colour for the dashboard
DARK_BLUE = "#0A1830"

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
        progess=False
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

# Interactive 3D Stock Performance Chart

def build_chart_3d(indexed_prices):
    """
    Create an interactive 3D chart of historical stock performance.

    Parameters
    ----------
    indexed_prices : pd.DataFrame
        Stock prices indexed to a common starting value of 100.
        Each column represents a different company.

    Returns
    -------
    plotly.graph_objects.Figure
        An interactive 3D line chart displaying stock performance.

    Note
    ----
    The X-axis represents historical dates.
    The Y-axis separates companies.
    The Z-axis shows indexed stock performance.
    """

    # Create an empty Plotly figure.
    fig = go.Figure()

    # Check whether the DataFrame contains any data.
    if indexed_prices.empty:
        return fig

    # Convert the DataFrame index into dates.
    dates = pd.to_datetime(indexed_prices.index)

    # Calculate the number of days since the starting date.
    days = (dates - dates[0]).days

    # Create a separate 3D line for every selected company.
    for number, ticker in enumerate(indexed_prices.columns):

        fig.add_trace(
            go.Scatter3d(
                x=days,
                y=[number] * len(indexed_prices),
                z=indexed_prices[ticker],

                mode="lines",
                name=ticker,

                line=dict(
                    color=STOCK_COLOURS[
                        number % len(STOCK_COLOURS)
                    ],
                    width=6
                ),

                text=dates.strftime("%d %b %Y"),

                hovertemplate=(
                    f"Company: {ticker}<br>"
                    "Date: %{text}<br>"
                    "Indexed Price: %{z:.2f}"
                    "<extra></extra>"
                )
            )
        )

    # Select five evenly spaced dates for the X-axis labels.
    tick_positions = np.linspace(
        0,
        len(dates) - 1,
        min(5, len(dates)),
        dtype=int
    )

    # Configure the chart's appearance.
    fig.update_layout(
        title="AI Stock Performance — 3D Comparison",
        template="plotly_dark",
        paper_bgcolor=DARK_BLUE,
        font=dict(color="white"),
        height=580,

        scene=dict(
            xaxis=dict(
                title="Date",
                tickvals=days[tick_positions],
                ticktext=dates[tick_positions].strftime("%b %Y"),
                gridcolor="#35506D"
            ),

            yaxis=dict(
                title="Company",
                tickvals=list(range(len(indexed_prices.columns))),
                ticktext=list(indexed_prices.columns),
                gridcolor="#35506D"
            ),

            zaxis=dict(
                title="Indexed Price (Base = 100)",
                gridcolor="#35506D"
            ),

            bgcolor=DARK_BLUE,

            aspectratio=dict(
                x=2.2,
                y=1.1,
                z=1.1
            ),

            camera=dict(
                eye=dict(x=1.6, y=-1.8, z=1.1)
            )
        )
    )

    return fig

               
            
               
               
        
    







