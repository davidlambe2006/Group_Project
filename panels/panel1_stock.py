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
DARK_BLUE = "0A1830"

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
     # Ensure we have a DataFrame, even for one company.
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









