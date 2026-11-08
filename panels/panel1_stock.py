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


