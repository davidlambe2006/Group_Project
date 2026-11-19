
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

from functools import lru_cache
from urllib.parse import parse_qs
from dash import Dash, dcc, html, Input, Output



try:
    from .dashboard_shared import TICKERS, NAVY as SHARED_NAVY
except ImportError:  # Direct execution of a panel demo.
    from dashboard_shared import TICKERS, NAVY as SHARED_NAVY

# The universal ten-company universe, all selected by default.
DEFAULT_AI_COMPANIES = list(TICKERS)

# Chart colours
NAVY = SHARED_NAVY
BLUE = "#38BDF8"
CYAN = "#00E5A8"
GREEN = "#4ADE80"
PURPLE = "#C4B5FD"
PINK = "#FB7185"


# ---------------------------------------
# 1. DOWNLOAD DATA
# ---------------------------------------

@lru_cache(maxsize=1)
def p3_trading_session_closes(session_date):
    # The date is the cache key: refresh the schedule on each New York day.
    # Yahoo's schedule includes holidays and early closing times.
    try:
        metadata = yf.Ticker("^IXIC").get_history_metadata()
        schedule = metadata["tradingPeriods"]
        closes = pd.DatetimeIndex(pd.to_datetime(schedule["end"], utc=True))
        if closes.empty or closes.isna().any():
            raise ValueError("Missing session closing times.")
        return closes
    except Exception as error:
        raise ValueError(
            "Unable to check completed trading sessions. Please try again shortly."
        ) from error


@lru_cache(maxsize=32)
def p3_download_prices(tickers, start, end=None):
    try:
        data = yf.download(
            list(tickers),
            start=start,
            end=end,
            auto_adjust=True,
            progress=False
        )
        if data is None or data.empty:
            raise ValueError("No prices returned.")
        prices = data["Close"].copy()
    except Exception as error:
        raise ValueError(
            "Yahoo Finance prices are unavailable. Please try again shortly."
        ) from error

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    # Ensure consistent column names
    prices.columns = [
        str(col).upper() for col in prices.columns
    ]

    missing = set(tickers) - set(prices.columns)
    if missing:
        raise ValueError(f"Missing price data for: {', '.join(sorted(missing))}.")

    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans:
        raise ValueError("Yahoo Finance returned invalid price dates. Please retry.")

    prices.index = prices.index.tz_localize(None)
    prices = prices.sort_index().apply(pd.to_numeric, errors="coerce")
    # Enforce the requested range; Yahoo's end date is exclusive.
    prices = prices.loc[prices.index >= pd.Timestamp(start)]
    if end is not None:
        prices = prices.loc[prices.index < pd.Timestamp(end)]

    if prices.index.has_duplicates or ((prices <= 0) | np.isinf(prices)).any().any():
        raise ValueError("Yahoo Finance returned invalid prices. Please retry.")

    unavailable = [ticker for ticker in tickers if prices[ticker].dropna().empty]
    if unavailable:
        raise ValueError(f"No valid price data for: {', '.join(unavailable)}. Please retry.")

    # Failed downloads raise above, so they are not retained in the price cache.
    return prices.dropna(how="all")


# ---------------------------------------
# 2. CALCULATE PORTFOLIOS
# ---------------------------------------

def p3_calculate_ai_portfolio(selected_companies, start_date="2023-01-01"):

    selected = list(dict.fromkeys(selected_companies))

    if not selected:
        raise ValueError("Select at least one AI company.")

    invalid = set(selected) - set(DEFAULT_AI_COMPANIES)

    if invalid:
        raise ValueError(f"Unknown companies: {invalid}")

    now = pd.Timestamp.now(tz="America/New_York")
    closes = p3_trading_session_closes(now.strftime("%Y-%m-%d"))
    completed = closes[closes <= now]
    if completed.empty:
        raise ValueError("No completed trading session is available. Please retry shortly.")
    # Including only dates before the following day excludes unfinished sessions.
    end_date = (
        completed.max().tz_convert("America/New_York").normalize()
        + pd.DateOffset(days=1)
    ).strftime("%Y-%m-%d")
    prices = p3_download_prices(
        tuple(sorted(selected)), start_date, end_date
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
def p3_calculate_nasdaq():

    prices = p3_download_prices(
        ("^IXIC",), "1995-01-01", "2004-01-01"
    )

    close = prices["^IXIC"]

    return close / close.iloc[0] * 100


def p3_calculate_metrics(series):

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

def p3_create_forecast(start_value, target_return, months=24):

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

def p3_monthly_with_latest(series):
    monthly = series.resample("MS").first().dropna()
    # Keep monthly samples, adding the actual latest observation only if new.
    latest_date = series.index[-1]
    first_in_last_month = series.loc[series.index >= monthly.index[-1]].index[0]
    if latest_date > first_in_last_month:
        monthly.loc[latest_date] = series.iloc[-1]

    # Whole calendar months for existing samples; a fraction for the daily end.
    x = (
        (monthly.index.year - series.index[0].year) * 12
        + monthly.index.month - series.index[0].month
        + (monthly.index.day - 1) / monthly.index.days_in_month
    ).to_numpy(dtype=float, copy=True)
    # If the final month has just one observation, position that point by its
    # actual date rather than adding the same observation a second time.
    if latest_date == first_in_last_month and len(monthly) > 1:
        x[-1] += (latest_date.day - 1) / latest_date.days_in_month
    return monthly, x


def p3_render_panel3(selected_companies=None):

    if selected_companies is None:
        selected_companies = DEFAULT_AI_COMPANIES

    nasdaq = p3_calculate_nasdaq()

    ai = p3_calculate_ai_portfolio(selected_companies)

    nasdaq_return, nasdaq_dd = p3_calculate_metrics(nasdaq)
    ai_return, ai_dd = p3_calculate_metrics(ai)

    # Monthly observations
    nasdaq_monthly, nasdaq_x = p3_monthly_with_latest(nasdaq)
    ai_monthly, ai_x = p3_monthly_with_latest(ai)

    # Rebase displayed monthly series to 100
    nasdaq_monthly = (
        nasdaq_monthly / nasdaq_monthly.iloc[0] * 100
    )
    ai_monthly = (
        ai_monthly / ai_monthly.iloc[0] * 100
    )

    # Preserve the exact daily endpoints, avoiding rebasing round-off differences.
    nasdaq_monthly.iloc[-1] = nasdaq.iloc[-1]
    ai_monthly.iloc[-1] = ai.iloc[-1]

    fig = go.Figure()

    # Historical NASDAQ
    fig.add_trace(go.Scatter(
        x=nasdaq_x,
        y=nasdaq_monthly,
        mode="lines",
        name="Dot-Com Era · NASDAQ",
        line=dict(color=BLUE, width=3.5),
        customdata=nasdaq_monthly.to_numpy() - 100,
        hovertemplate=(
            "<b>Dot-Com Era · NASDAQ</b><br>"
            "Month: %{x}<br>"
            "Index (base 100): %{y:.2f}<br>"
            "Return since start: %{customdata:+.2f}%"
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
            "Index (base 100): %{y:.2f}<br>"
            "Return since start: %{customdata:+.2f}%"
            "<extra></extra>"

        )
    ))

    # -----------------------------------
    # 5. ADD THE THREE FORECASTS
    # -----------------------------------

    last_month = float(ai_x[-1])
    last_value = float(ai_monthly.iloc[-1])

    forecast_x = last_month + np.arange(25)

    # Estimate volatility from completed month ends, excluding the partial month.
    month_ends = ai.resample("ME").last().dropna()
    current_month = (
        pd.Timestamp.now(tz="America/New_York")
        .tz_localize(None).normalize().replace(day=1)
    )
    month_ends = month_ends.loc[month_ends.index < current_month]
    monthly_log_returns = np.log(month_ends / month_ends.shift(1)).dropna()
    if len(monthly_log_returns) < 2:
        raise ValueError(
            "At least three completed month-end observations are needed "
            "for the volatility-based scenarios."
        )

    sigma = float(monthly_log_returns.std(ddof=1))
    months = 24
    z = 1.645
    spread = z * sigma * np.sqrt(months)
    # Base is assumed; historical volatility determines the Bull/Bear dispersion.
    scenarios = [
        ("Bull", 1.166 * np.exp(spread) - 1, GREEN),
        ("Base", 0.166, PURPLE),
        ("Bear", 1.166 * np.exp(-spread) - 1, PINK)
    ]

    for name, target_return, colour in scenarios:

        forecast_y = p3_create_forecast(
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
            customdata=np.column_stack((
                np.arange(25),
                (forecast_y / last_value - 1) * 100
            )),
            hovertemplate=(
                f"<b>{name} Scenario · Illustrative</b><br>"
                "Forecast month: %{customdata[0]:.0f} of 24<br>"
                "Projected index: %{y:.2f}<br>"
                "Return from forecast start: %{customdata[1]:+.2f}%<br>"
                f"24-month target: {target_return:+.1%}"
                "<extra></extra>"
            )
        ))

        fig.add_annotation(
            x=forecast_x[-1] + 1.5,
            y=forecast_y[-1],
            text=f"<b>{name} {target_return:+.1%}</b>",
            # Keep labels separate from the data: no coloured connector lines.
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            xshift=8,
            yshift=0,
            bgcolor=NAVY,
            borderpad=5,
            font=dict(size=13, color=colour)
        )

    # Subtle forecast boundary
    fig.add_vline(
        x=last_month,
        line_color="rgba(203,213,225,0.35)",
        line_width=1,
        line_dash="dot"
    )
    fig.add_annotation(
        x=last_month, y=0, yref="paper", yshift=12,
        text="Forecast starts", showarrow=False,
        font=dict(size=11, color="#CBD5E1"), bgcolor=NAVY
    )

    # -----------------------------------
    # 6. PROFESSIONAL CHART STYLING
    # -----------------------------------

    fig.update_layout(
        paper_bgcolor=NAVY,
        plot_bgcolor=NAVY,
        font=dict(
            family="Inter, Segoe UI, Arial, sans-serif",
            color="#CBD5E1",
            size=13
        ),

        xaxis_title="Months Since Start",
        yaxis_title="Indexed Performance · Base 100",

        hovermode="closest",

        hoverlabel=dict(
            bgcolor="#172338",
            bordercolor="#334155",
            font=dict(
                color="#F8FAFC",
                size=13
            )
        ),

        # Reserve the top margin for one horizontal row at desktop widths.
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0,
            font=dict(size=11, color="#F1F5F9"),
            itemsizing="constant",
            itemwidth=30,
            # Natural text widths avoid rounding/wrapping of five equal slots.
            tracegroupgap=0
        ),

        margin=dict(
            l=60,
            r=40,
            t=85,
            b=65
        ),

        height=740,
        autosize=True,

        hoverdistance=30
    )

    fig.update_xaxes(
        # Add label space beyond every observed/forecast point; retain the y scale.
        range=[0, max(float(nasdaq_x[-1]), float(forecast_x[-1])) + 16],
        showgrid=True,
        gridcolor="rgba(148,163,184,0.07)",
        automargin=True,
        title_standoff=18,
        tickfont=dict(size=12, color="#CBD5E1"),
        zeroline=False,
        showline=False
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.07)",
        automargin=True,
        title_standoff=18,
        tickfont=dict(size=12, color="#CBD5E1"),
        zeroline=False,
        showline=False
    )

    # Store metrics for the Dash header
    fig.update_layout(meta={
        "nasdaq_return": float(nasdaq_return),
        "nasdaq_dd": float(nasdaq_dd),
        "ai_return": float(ai_return),
        "ai_dd": float(ai_dd),
        "company_count": len(selected_companies),
        "nasdaq_period": f"{nasdaq.index[0]:%d %b %Y} – {nasdaq.index[-1]:%d %b %Y}",
        "ai_period": f"{ai.index[0]:%d %b %Y} – {ai.index[-1]:%d %b %Y}"
    })

    return fig


# ---------------------------------------
# 7. DASH HEADER AND METRICS
# ---------------------------------------

def p3_nasdaq_crash_detail():
    # Identify the daily peak and trough behind the historical drawdown KPI.
    nasdaq = p3_calculate_nasdaq()
    drawdown = nasdaq / nasdaq.cummax() - 1
    trough_date = drawdown.idxmin()
    peak_date = nasdaq.loc[:trough_date].idxmax()
    crash = nasdaq.loc[peak_date:trough_date]

    detail = go.Figure(go.Scatter(
        x=crash.index, y=crash, mode="lines",
        line=dict(color=BLUE, width=2.5),
        customdata=(crash / crash.iloc[0] - 1) * 100,
        hovertemplate=(
            "<b>NASDAQ · Historical crash</b><br>"
            "%{x|%d %b %Y}<br>Index (base 100): %{y:.2f}<br>"
            "Return from peak: %{customdata:+.2f}%<extra></extra>"
        )
    ))
    for label, date, anchor in [
        ("Peak", peak_date, "left"),
        ("Trough", trough_date, "right")
    ]:
        detail.add_trace(go.Scatter(
            x=[date], y=[float(nasdaq.loc[date])], mode="markers",
            marker=dict(size=8, color=BLUE if label == "Peak" else PINK,
                        line=dict(color=NAVY, width=2)),
            hoverinfo="skip", showlegend=False
        ))
        detail.add_annotation(
            x=date, y=float(nasdaq.loc[date]),
            text=f"<b>{label}</b><br>{date:%d %b %Y}",
            showarrow=False, xanchor=anchor, yanchor="bottom", yshift=8,
            font=dict(size=12, color="#E2E8F0"), bgcolor=NAVY, borderpad=3
        )

    detail.add_annotation(
        x=0.96, y=0.85, xref="paper", yref="paper",
        text=f"<b>−{abs(drawdown.min()):.1%} Peak-to-Trough</b>",
        showarrow=False, xanchor="right", yanchor="top",
        font=dict(size=17, color=PINK), bgcolor=NAVY, borderpad=5
    )

    # A separate, explicitly labelled zoom leaves the main graph's scale intact.
    padding = float(crash.max() - crash.min()) * 0.15
    detail.update_layout(
        title=dict(
            text="NASDAQ · Inside the dot-com crash",
            font=dict(size=16, color="#F1F5F9"), x=0.02
        ),
        paper_bgcolor=NAVY, plot_bgcolor=NAVY,
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", color="#CBD5E1", size=12),
        height=350, margin=dict(l=60, r=30, t=65, b=45), showlegend=False,
        hoverlabel=dict(bgcolor="#172338", bordercolor="#334155", font_color="#F8FAFC")
    )
    detail.update_xaxes(
        tickformat="%b %Y", nticks=6, showgrid=False, zeroline=False
    )
    detail.update_yaxes(
        title="NASDAQ index · Base 100", automargin=True,
        range=[float(crash.min()) - padding, float(crash.max()) + padding],
        gridcolor="rgba(148,163,184,0.07)", zeroline=False
    )
    return detail


def p3_metric_card(label, value, colour, date_range=None):

    return html.Div(
        [
            html.Div(
                label,
                style={
                    "color": "#B7C5D8",
                    "fontSize": "12px",
                    "marginBottom": "12px",
                    "letterSpacing": "0.7px",
                    "lineHeight": "1.5"
                }
            ),
            html.Div(
                value,
                style={
                    "color": colour,
                    "fontSize": "28px",
                    "fontWeight": "600",
                    "fontVariantNumeric": "tabular-nums"
                }
            ),
            html.Div(date_range, style={
                "color": "#94A3B8", "fontSize": "12px",
                "lineHeight": "1.6", "marginTop": "10px"
            }) if date_range else None
        ],
        style={
            "backgroundColor": "#131F31",
            "padding": "clamp(16px, 2vw, 22px)",
            "border": "1px solid #263449",
            "borderRadius": "10px",
            "boxSizing": "border-box",
            "flex": "1 1 210px",
            "minWidth": "0"
        }
    )


def p3_layout(selected_companies=None):

    if selected_companies is None:
        selected_companies = DEFAULT_AI_COMPANIES

    if not selected_companies:
        return html.P(
            "Please select at least one company in the universal selector.",
            role="status", style={"color": "white"}
        )

    try:
        fig = p3_render_panel3(selected_companies)
        crash_detail = p3_nasdaq_crash_detail()
    except ValueError as error:
        return html.P(
            f"Unable to load Panel 3. {error}",
            style={"color": "white"}
        )

    metrics = fig.layout.meta

    return html.Div(
        [
            html.H2(
                "AI Boom vs. Dot-Com Bubble",
                style={
                    "color": "#F8FAFC",
                    "marginBottom": "6px",
                    "letterSpacing": "-0.5px",
                    "fontSize": "clamp(23px, 3vw, 32px)",
                    "marginTop": "0"
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
                    p3_metric_card(
                        "NASDAQ PRICE RETURN",
                        f"{metrics['nasdaq_return']:+.1f}%",
                        GREEN if metrics['nasdaq_return'] >= 0 else PINK,
                        metrics['nasdaq_period']
                    ),
                    p3_metric_card(
                        "NASDAQ MAX DRAWDOWN",
                        f"{metrics['nasdaq_dd']:.1f}%",
                        PINK, metrics['nasdaq_period']
                    ),
                    p3_metric_card(
                        "AI PORTFOLIO RETURN",
                        f"{metrics['ai_return']:+.1f}%",
                        GREEN if metrics['ai_return'] >= 0 else PINK,
                        metrics['ai_period']
                    ),
                    p3_metric_card(
                        "AI MAX DRAWDOWN",
                        f"{metrics['ai_dd']:.1f}%",
                        PINK, metrics['ai_period']
                    )
                ],
                style={
                    "display": "flex",
                    "gap": "16px",
                    "flexWrap": "wrap",
                    "marginBottom": "16px"
                }
            ),

            html.Div(
                "Solid lines · historical observations    /    Dashed lines · illustrative scenarios",
                style={"color": "#B7C5D8", "fontSize": "12px", "lineHeight": "1.7"}
            ),
            dcc.Graph(
                figure=fig,
                config={
                    "displaylogo": False,
                    "responsive": True
                },
                # An explicit container height survives Dash's responsive resizing.
                responsive=True,
                style={"width": "100%", "minWidth": "0", "height": "clamp(580px, 65vw, 740px)"}
            ),

            dcc.Graph(
                figure=crash_detail,
                config={"displaylogo": False, "displayModeBar": False, "responsive": True},
                responsive=True,
                style={"width": "100%", "minWidth": "0", "height": "350px"}
            ),

            html.P(
                f"AI portfolio: {metrics['company_count']} companies, "
                "equally weighted at inception. "
                "The 24-month Base return is assumed at +16.6%; "
                "Bull/Bear are illustrative volatility-based scenarios using "
                "completed monthly log returns (z = 1.645), "
                "not statistically validated predictions. Intermediate fluctuations "
                "are illustrative, not statistically simulated forecasts.",
                style={
                    "color": "#94A3B8",
                    "fontSize": "12px",
                    "marginTop": "8px",
                    "lineHeight": "1.7"
                }
            )
        ],
        style={
            "backgroundColor": NAVY,
            "padding": "clamp(14px, 2.5vw, 28px)",
            "minWidth": "0",
            "borderRadius": "14px",
            "fontFamily": "Inter, Segoe UI, Arial, sans-serif"
        }
    )


# ---------------------------------------
# 8. DASHBOARD INTEGRATION
# ---------------------------------------

def p3_selection_from_search(search):
    """Read the universal selector's iframe URL; preserve an explicit empty list."""
    query = parse_qs((search or "").lstrip("?"), keep_blank_values=True)
    if "companies" not in query:
        return list(DEFAULT_AI_COMPANIES)
    return list(dict.fromkeys(
        ticker.strip().upper()
        for value in query["companies"]
        for ticker in value.split(",")
        if ticker.strip().upper() in TICKERS
    ))


def p3_create_app(**kwargs):
    """Mount under /panel/3/ using the host dashboard's companies query string."""
    app = Dash(__name__, include_assets_files=False, **kwargs)
    app.layout = html.Div(
        [
            dcc.Location(id="panel3-url", refresh=False),
            dcc.Loading(html.Div(id="panel3-content"))
        ],
        style={
            "backgroundColor": NAVY,
            "minHeight": "100vh",
            "padding": "clamp(12px, 2vw, 25px)",
            "fontFamily": "Inter, Segoe UI, Arial, sans-serif"
        }
    )

    @app.callback(
        Output("panel3-content", "children"),
        Input("panel3-url", "search")
    )
    def p3_update_panel(search):
        # Failed Yahoo requests raise ValueError and are not cached. Reloading
        # or changing the universal selection retries through p3_layout.
        return p3_layout(p3_selection_from_search(search))

    return app


if __name__ == "__main__":
    p3_create_app().run(debug=True)
