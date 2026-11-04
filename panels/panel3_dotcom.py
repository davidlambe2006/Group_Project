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

# Select the NASDAQ closing prices
nasdaq_close = nasdaq["Close"]["^IXIC"]

# Convert prices into an index starting at 100
nasdaq_indexed = nasdaq_close / nasdaq_close.iloc[0] * 100

# Check the first five indexed values
print(nasdaq_indexed.head())

import plotly.graph_objects as go

# Create an interactive NASDAQ chart
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=nasdaq_indexed.index,
    y=nasdaq_indexed,
    mode="lines",
    name="NASDAQ Composite",
    line=dict(color="#00BFFF", width=2)
))

# Add chart titles and styling
fig.update_layout(
    title="The Dot-Com Bubble (1995–2003)",
    xaxis_title="Year",
    yaxis_title="NASDAQ Index (1995 = 100)",
    template="plotly_dark",
    hovermode="x unified"
)

# Open the interactive graph
fig.show()