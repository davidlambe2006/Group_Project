#Panel 1: AI Stock Performance

#Libraries for handling data and calculations
from py_compile import main

import pandas as pd
import numpy as np

#Library for historical share prices
import yfinance as yf

#Libraries for the dashboard and interactive graphs
import streamlit as st
import plotly.graph_objects as go

#Python tools for handling dates
from datetime import date, timedelta

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
DEFAULT_COMPANIES = [
    "NVDA",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "AMD"
]

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
                gridcolor="#35506D",
                backgroundcolor=DARK_BLUE
            ),
            yaxis=dict(
                title="Company",
                tickvals=list(range(len(indexed_prices.columns))),
                ticktext=list(indexed_prices.columns),
                gridcolor="#35506D",
                backgroundcolor=DARK_BLUE
            ),
            zaxis=dict(
                title="Indexed Price (Base = 100)",
                gridcolor="#35506D",
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




# STREAMLIT DASHBOARD AND INTERACTIVE CONTROLS

def render_panel1(selected_companies):
    """
    Display the interactive AI Stock Performance dashboard.

    Parameters
    ----------
    selected_companies : list
        Stock ticker symbols selected by the user.
        The list will eventually come from the group's
        global company selector in app.py.

    Returns
    -------
        Displays an interactive 3D chart and a table
        containing historical financial statistics.

    Notes
    -----
    The selected companies are compared using adjusted
    closing prices indexed to a common starting value of 100.
    """

    # Display the panel heading and research question.
    st.subheader("Panel 1 — AI Stock Performance (3D)")

    st.write(
        "How have AI-related companies performed "
        "relative to the wider stock market?"
    )

    # Check whether the user has selected any companies.
    if not selected_companies:
        st.warning("Please select at least one AI company.")
        return

    # Create three columns for the dashboard controls.
    col1, col2, col3 = st.columns(3)

    # Control 1: Historical comparison period.
    with col1:
        period = st.selectbox(
            "Historical Period",
            ["1 Year", "3 Years", "5 Years", "10 Years"],
            index=1,
            key="panel1_period"
        )

    # Control 2: Choose market benchmarks.
    with col2:
        selected_benchmarks = st.multiselect(
            "Market Benchmarks",
            options=list(BENCHMARKS.keys()),
            default=["SPY", "QQQ"],
            key="panel1_benchmarks"
        )

    # Control 3: Adjust the amount of chart detail.
    with col3:
        detail = st.selectbox(
            "Chart Detail",
            ["Daily", "Every 5 Days", "Every 20 Days"],
            index=1,
            key="panel1_detail"
        )

    # Combine the selected AI companies and benchmarks.
    tickers = list(dict.fromkeys(
        selected_companies + selected_benchmarks
    ))

    # Convert the chosen period into a number of years.
    years = int(period.split()[0])

    # Calculate the starting and ending dates.
    end_date = date.today() + timedelta(days=1)

    start_date = date.today() - timedelta(
        days=round(years * 365.25)
    )

    # Download the historical stock prices.
    try:
        with st.spinner("Downloading historical stock data..."):
            prices = get_prices(
                tickers,
                start_date.isoformat(),
                end_date.isoformat()
            )

    except Exception as error:
        st.error("Unable to download historical stock data.")
        st.caption(str(error))
        return

    # Check whether the download returned any information.
    if prices.empty:
        st.error("No stock data was returned.")
        return

    # Check for companies with completely missing prices.
    missing = [
        ticker for ticker in tickers
        if ticker not in prices.columns
        or prices[ticker].isna().all()
    ]

    if missing:
        st.warning(
            "Data unavailable for: " + ", ".join(missing)
        )

    # Remove tickers that have no historical prices.
    prices = prices.drop(columns=missing, errors="ignore")

    # Ensure at least one AI company still has valid data.
    if not any(ticker in prices.columns
               for ticker in selected_companies):
        st.warning("No selected AI companies have available data.")
        return

    # Index stock prices to a common starting value of 100.
    indexed_prices = index_and_normalise(prices)

    if len(indexed_prices) < 3:
        st.warning("Not enough shared historical data to compare.")
        return

    # Calculate statistics using the full historical dataset.
    statistics = calculate_statistics(indexed_prices)

    # Reduce chart points to improve 3D performance.
    if detail == "Daily":
        step = 1
    elif detail == "Every 5 Days":
        step = 5
    else:
        step = 20

    chart_prices = indexed_prices.iloc[::step].copy()

    # Include the latest observation if sampling skipped it.
    if chart_prices.index[-1] != indexed_prices.index[-1]:
        chart_prices = pd.concat([
            chart_prices,
            indexed_prices.iloc[[-1]]
        ])


    # Create two columns: 75% for chart, 25% for statistics.
    chart_col, stats_col = st.columns([3, 1], gap="medium")

    # LEFT COLUMN: Interactive 3D stock performance graph.
    with chart_col:
        figure = build_chart_3d(chart_prices)

        st.plotly_chart(
            figure,
            use_container_width=True,
            key="panel1_3d_chart"
        )

    # RIGHT COLUMN: Financial performance statistics.
    with stats_col:
        st.markdown("#### Performance Statistics")

        # Rename the columns to fit the smaller space.
        display_stats = statistics.rename(columns={
            "Total Return (%)": "Return",
            "Annualised Volatility (%)": "Volatility",
            "Maximum Drawdown (%)": "Drawdown"
        })

        # Round the statistics to two decimal places.
        st.dataframe(
            display_stats.round(2),
            use_container_width=True,
            height=560
        )

  

    # Explain the source and methodology.
    st.caption(
        f"Source: Yahoo Finance via yfinance | "
        f"Actual comparison period: "
        f"{indexed_prices.index[0]:%d %b %Y} to "
        f"{indexed_prices.index[-1]:%d %b %Y} | "
        "Prices adjusted for stock splits and dividends."
    )




def main():
    """
    Run Panel 1 independently with six default AI companies.

    Configure the dashboard's appearance and width.
    """

    # Set the dashboard to use the full browser width.
    st.set_page_config(
        page_title="AI Boom Dashboard",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Increase the maximum width of the dashboard.
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1700px;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display Panel 1 using the six default companies.
    render_panel1(DEFAULT_COMPANIES)


if __name__ == "__main__":
    main()




               
            
               
               
        
    







