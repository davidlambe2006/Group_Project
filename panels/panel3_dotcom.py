import pandas as pd
import yfinance as yf

# Download historical NASDAQ Composite prices
nasdaq = yf.download(
    "^IXIC",
    start="1995-01-01",
    end="2004-01-01",
    auto_adjust=False
)

# Check the first five rows
print(nasdaq.head())