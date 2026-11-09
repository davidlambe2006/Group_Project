
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

from functools import lru_cache
from dash import Dash, dcc, html, Input, Output


# All 12 companies, selected by default
DEFAULT_AI_COMPANIES = [
    "NVDA", "MSFT", "GOOGL", "AMZN",
    "META", "AVGO", "AMD", "TSM",
    "ORCL", "PLTR", "CRM", "ASML"
]

# Chart colours
NAVY = "#0B1220"
BLUE = "#38BDF8"
CYAN = "#00E5A8"
GREEN = "#4ADE80"
PURPLE = "#C4B5FD"
PINK = "#FB7185"


# ---------------------------------------
# 1. DOWNLOAD DATA
# ---------------------------------------

@lru_cache(maxsize=32)
def download_prices(tickers, start, end=None):
    data = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    prices = data["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    # Ensure consistent column names
    prices.columns = [
        str(col).upper() for col in prices.columns
    ]

    return prices.dropna(how="all")


# ---------------------------------------
# 2. CALCULATE PORTFOLIOS
# ---------------------------------------

def calculate_ai_portfolio(selected_companies, start_date="2023-01-01"):

    selected = list(dict.fromkeys(selected_companies))

    if not selected:
        raise ValueError("Select at least one AI company.")

    invalid = set(selected) - set(DEFAULT_AI_COMPANIES)

    if invalid:
        raise ValueError(f"Unknown companies: {invalid}")

    prices = download_prices(
        tuple(sorted(selected)), start_date
    )

    # All selected stocks must have a valid price
    prices = prices[selected].dropna()

    if prices.empty:
        raise ValueError("No overlapping price data available.")

    # Index every stock to 100
    indexed = prices.div(prices.iloc[0]) * 100

    # Equal initial investment in every stock
    # Weights drift naturally as prices change
    portfolio = indexed.mean(axis=1)

    return portfolio


@lru_cache(maxsize=1)
def calculate_nasdaq():

    prices = download_prices(
        ("^IXIC",), "1995-01-01", "2004-01-01"
    )

    close = prices["^IXIC"]

    return close / close.iloc[0] * 100


def calculate_metrics(series):

    cumulative_return = (
        series.iloc[-1] / series.iloc[0] - 1
    ) * 100

    running_peak = series.cummax()

    drawdown = (
        series / running_peak - 1
    ) * 100

    return cumulative_return, drawdown.min()


# ---------------------------------------
# 3. ILLUSTRATIVE FORECASTS
# ---------------------------------------

def create_forecast(start_value, target_return, months=24):

    t = np.arange(months + 1)

    # Smooth progression to the exact target
    target_value = start_value * (1 + target_return)

    trend = np.log(target_value / start_value) * t / months

    # Small deterministic fluctuations
    # Zero at both endpoints
    fluctuation = (
        0.035
        * np.sin(np.pi * t / months)
        * np.sin(2 * np.pi * t / 7)
    )

    values = start_value * np.exp(trend + fluctuation)

    values[0] = start_value
    values[-1] = target_value

    return values


# ---------------------------------------
# 4. BUILD THE PLOTLY GRAPH
# ---------------------------------------

def render_panel3(selected_companies=None):

    if selected_companies is None:
        selected_companies = DEFAULT_AI_COMPANIES

    nasdaq = calculate_nasdaq()

    ai = calculate_ai_portfolio(selected_companies)

    nasdaq_return, nasdaq_dd = calculate_metrics(nasdaq)
    ai_return, ai_dd = calculate_metrics(ai)

    # Monthly observations
    nasdaq_monthly = nasdaq.resample("MS").first().dropna()
    ai_monthly = ai.resample("MS").first().dropna()

    # Rebase displayed monthly series to 100
    nasdaq_monthly = (
        nasdaq_monthly / nasdaq_monthly.iloc[0] * 100
    )
    ai_monthly = (
        ai_monthly / ai_monthly.iloc[0] * 100
    )

    nasdaq_x = np.arange(len(nasdaq_monthly))
    ai_x = np.arange(len(ai_monthly))

    fig = go.Figure()

    # Historical NASDAQ
    fig.add_trace(go.Scatter(
        x=nasdaq_x,
        y=nasdaq_monthly,
        mode="lines",
        name="Dot-Com Era · NASDAQ",
        line=dict(color=BLUE, width=3),
        hovertemplate=(
            "<b>Dot-Com Era · NASDAQ</b><br>"
            "Month: %{x}<br>"
            "Index: %{y:.2f}<br>"
            "<extra></extra>"
        )
    ))

    # Modern AI portfolio
    fig.add_trace(go.Scatter(
        x=ai_x,
        y=ai_monthly,
        mode="lines",
        name="Modern AI · Portfolio",
        line=dict(color=CYAN, width=3.5),
        customdata=ai_monthly.to_numpy() - 100,
        hovertemplate=(
            "<b>Modern AI Portfolio</b><br>"
            "Month: %{x}<br>"
            "Index: %{y:.2f}<br>"
            "Return since start: %{customdata:+.2f}%"
            "<extra></extra>"

)
    ))

    # -----------------------------------
    # 5. ADD THE THREE FORECASTS
    # -----------------------------------

    last_month = int(ai_x[-1])
    last_value = float(ai_monthly.iloc[-1])

    forecast_x = np.arange(last_month, last_month + 25)

    scenarios = [
        ("Bull", 0.44, GREEN),
        ("Base", 0.166, PURPLE),
        ("Bear", -0.40, PINK)
    ]

    for name, target_return, colour in scenarios:

        forecast_y = create_forecast(
            last_value, target_return
        )

        fig.add_trace(go.Scatter(
            x=forecast_x,
            y=forecast_y,
            mode="lines",
            name=f"{name} Scenario",
            line=dict(
                color=colour,
                width=2.5,
                dash="dash"
            ),
            hovertemplate=(
                f"<b>{name} Scenario · Illustrative</b><br>"
                "Month: %{x}<br>"
                "Projected index: %{y:.2f}<br>"
                f"24-month target: {target_return:+.1%}"
                "<extra></extra>"
            )
        ))

        fig.add_annotation(
            x=forecast_x[-1],
            y=forecast_y[-1],
            text=f"<b>{name} {target_return:+.1%}</b>",
            showarrow=False,
            xanchor="left",
            xshift=8,
            font=dict(size=11, color=colour)
        )

    # Subtle forecast boundary
    fig.add_vline(
        x=last_month,
        line_color="rgba(203,213,225,0.35)",
        line_width=1,
        line_dash="dot"
    )

    # -----------------------------------
    # 6. PROFESSIONAL CHART STYLING
    # -----------------------------------

    fig.update_layout(
        paper_bgcolor=NAVY,
        plot_bgcolor=NAVY,
        font=dict(
            family="Arial",
            color="#CBD5E1",
            size=12
        ),

        xaxis_title="Months Since Start",
        yaxis_title="Indexed Performance · Base 100",

        hovermode="closest",

        hoverlabel=dict(
            bgcolor="#172338",
            bordercolor="#334155",
            font=dict(
                color="#F8FAFC",
                size=12
            )
        ),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0,
            font=dict(size=11)
        ),

        margin=dict(
            l=65,
            r=125,
            t=85,
            b=65
        ),

        height=570,

        hoverdistance=30
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.10)",
        zeroline=False,
        showline=False
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.10)",
        zeroline=False,
        showline=False
    )

    # Store metrics for the Dash header
    fig.update_layout(meta={
        "nasdaq_return": float(nasdaq_return),
        "nasdaq_dd": float(nasdaq_dd),
        "ai_return": float(ai_return),
        "ai_dd": float(ai_dd),
        "company_count": len(selected_companies)
    })

    return fig


# ---------------------------------------
# 7. DASH HEADER AND METRICS
# ---------------------------------------

def metric_card(label, value, colour):

    return html.Div(
        [
            html.Div(
                label,
                style={
                    "color": "#94A3B8",
                    "fontSize": "12px",
                    "marginBottom": "9px"
                }
            ),
            html.Div(
                value,
                style={
                    "color": colour,
                    "fontSize": "25px",
                    "fontWeight": "bold"
                }
            )
        ],
        style={
            "backgroundColor": "#131F31",
            "padding": "18px",
            "border": "1px solid #263449",
            "borderRadius": "10px",
            "flex": "1",
            "minWidth": "150px"
        }
    )


def panel3_layout(selected_companies=None):

    if selected_companies is None:
        selected_companies = DEFAULT_AI_COMPANIES

    fig = render_panel3(selected_companies)

    metrics = fig.layout.meta

    return html.Div(
        [
            html.H2(
                "AI BOOM vs. DOT-COM BUBBLE",
                style={
                    "color": "#F8FAFC",
                    "marginBottom": "6px",
                    "letterSpacing": "0.5px"
                }
            ),

            html.P(
                "Two technology booms. One historical comparison.",
                style={
                    "color": "#94A3B8",
                    "marginTop": "0",
                    "marginBottom": "22px"
                }
            ),

            html.Div(
                [
                    metric_card(
                        "NASDAQ TOTAL RETURN",
                        f"{metrics['nasdaq_return']:+.1f}%",
                        BLUE
                    ),
                    metric_card(
                        "NASDAQ MAX DRAWDOWN",
                        f"{metrics['nasdaq_dd']:.1f}%",
                        PINK
                    ),
                    metric_card(
                        "AI PORTFOLIO RETURN",
                        f"{metrics['ai_return']:+.1f}%",
                        CYAN
                    ),
                    metric_card(
                        "AI MAX DRAWDOWN",
                        f"{metrics['ai_dd']:.1f}%",
                        PINK
                    )
                ],
                style={
                    "display": "flex",
                    "gap": "12px",
                    "flexWrap": "wrap",
                    "marginBottom": "16px"
                }
            ),

            dcc.Graph(
                figure=fig,
                config={
                    "displaylogo": False,
                    "responsive": True
                },
                style={"width": "100%"}
            ),

            html.P(
                f"AI portfolio: {metrics['company_count']} companies, "
                "equally weighted at inception. "
                "Forecasts are illustrative 24-month scenarios, "
                "not statistically validated predictions.",
                style={
                    "color": "#64748B",
                    "fontSize": "12px",
                    "marginTop": "8px"
                }
            )
        ],
        style={
            "backgroundColor": NAVY,
            "padding": "25px",
            "borderRadius": "14px",
            "fontFamily": "Arial"
        }
    )


# ---------------------------------------
# 8. STANDALONE DASH TEST
# ---------------------------------------

if __name__ == "__main__":

    app = Dash(__name__)

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Label(
                        "AI Companies",
                        style={"color": "#CBD5E1"}
                    ),

                    dcc.Dropdown(
                        id="global-company-selector",
                        options=[
                            {"label": ticker, "value": ticker}
                            for ticker in DEFAULT_AI_COMPANIES
                        ],
                        value=list(DEFAULT_AI_COMPANIES),
                        multi=True
                    )
                ],
                style={
                    "marginBottom": "20px",
                    "maxWidth": "750px"
                }
            ),

            html.Div(id="panel3-content")
        ],
        style={
            "backgroundColor": "#080F1C",
            "minHeight": "100vh",
            "padding": "25px"
        }
    )

    @app.callback(
        Output("panel3-content", "children"),
        Input("global-company-selector", "value")
    )
    def update_panel(selected_companies):

        if not selected_companies:
            return html.P(
                "Please select at least one company.",
                style={"color": "white"}
            )

        return panel3_layout(selected_companies)

    app.run(debug=False)
