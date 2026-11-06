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
# Find the highest NASDAQ value during the bubble
peak_value = nasdaq_indexed.max()
peak_date = nasdaq_indexed.idxmax()

# Find the lowest value AFTER the peak
after_peak = nasdaq_indexed.loc[peak_date:]
crash_value = after_peak.min()
crash_date = after_peak.idxmin()

# Calculate the percentage decline
crash_percent = (crash_value / peak_value - 1) * 100

# Display the results
print("Bubble peak:", peak_date.date(), round(peak_value, 2))
print("Crash bottom:", crash_date.date(), round(crash_value, 2))
print("NASDAQ decline:", round(crash_percent, 2), "%")

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
# Mark the highest point of the Dot-Com Bubble
fig.add_trace(go.Scatter(
    x=[peak_date],
    y=[peak_value],
    mode="markers+text",
    name="Bubble Peak",
    marker=dict(color="orange", size=12),
    text=["Bubble Peak"],
    textposition="top center"
))

# Mark the lowest point after the crash
fig.add_trace(go.Scatter(
    x=[crash_date],
    y=[crash_value],
    mode="markers+text",
    name="Crash Bottom",
    marker=dict(color="red", size=12),
    text=["Crash Bottom"],
    textposition="bottom center"
))

# Highlight the Dot-Com crash period
fig.add_vrect(
    x0=peak_date,
    x1=crash_date,
    fillcolor="red",
    opacity=0.05,
    layer="below",
    line_width=0
)

# Show the percentage decline during the crash
fig.add_annotation(
    x=crash_date,
    y=peak_value * 0.75,
    text=f"NASDAQ Crash: {crash_percent:.1f}%",
    showarrow=False,
    font=dict(size=14, color="white"),
    bgcolor="rgba(180, 30, 30, 0.8)",
    borderpad=6
)
# Add chart titles and styling
fig.update_layout(
    title="The Dot-Com Bubble (1995–2003)",
    xaxis_title="Year",
    yaxis_title="NASDAQ Index (1995 = 100)",
    template="plotly_dark",
    hovermode="x unified"
)

# Open the interactive graph
#fig.show()


# Modern AI portfolio — initial data test

selected_companies = ["NVDA", "MSFT", "GOOGL"]

ai_prices = yf.download(
    selected_companies,
    start="2023-01-01",
    auto_adjust=True,
    progress=False
)["Close"]

print("\nModern AI stock prices:")
print(ai_prices.head())


# Index each AI company's stock price to 100
ai_indexed = ai_prices.div(ai_prices.iloc[0]) * 100

# Display the first five rows
print("\nIndexed AI stock prices:")
print(ai_indexed.head())


# Calculate an equally weighted AI portfolio
ai_portfolio = ai_indexed.mean(axis=1)

# Display the first five portfolio values
print("\nAI Portfolio:")
print(ai_portfolio.head())


# Convert daily data into monthly data
nasdaq_monthly = nasdaq_indexed.resample("MS").first()
ai_monthly = ai_portfolio.resample("MS").first()

# Check the results
print("\nMonthly NASDAQ:")
print(nasdaq_monthly.head())

print("\nMonthly AI Portfolio:")
print(ai_monthly.head())


# Create month numbers for each period
nasdaq_month_numbers = list(range(len(nasdaq_monthly)))
ai_month_numbers = list(range(len(ai_monthly)))

# Check the results
print("\nNASDAQ month numbers:")
print(nasdaq_month_numbers[:5])

print("\nAI month numbers:")
print(ai_month_numbers[:5])


# Create the comparison chart
comparison_fig = go.Figure()

# Historical dot-com bubble
comparison_fig.add_trace(go.Scatter(
    x=nasdaq_month_numbers,
    y=nasdaq_monthly,
    mode="lines",
    name="Dot-Com Bubble (1995)",
    line=dict(color="#00BFFF", width=3)
))

# Modern AI portfolio
comparison_fig.add_trace(go.Scatter(
    x=ai_month_numbers,
    y=ai_monthly,
    mode="lines",
    name="Modern AI Boom (2023)",
    line=dict(color="#00FF99", width=3)
))


# Style the comparison chart
comparison_fig.update_layout(
    title="AI Boom vs Dot-Com Bubble",
    xaxis_title="Months Since Start",
    yaxis_title="Indexed Performance (Start = 100)",
    template="plotly_dark",
    hovermode="x unified"
)

# Open the comparison graph
comparison_fig.show()
