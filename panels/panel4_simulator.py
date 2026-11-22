# Panel 4: AI Valuation Simulator
# Estimates future share prices using revenue growth, net profit margin
# and a future price-to-earnings (P/E) ratio.
# Compares Bear, Base and Bull scenarios, with a heatmap showing
# how different growth and margin assumptions affect annual returns.
# Runs independently or as part of the four-panel dashboard.

from datetime import datetime, timezone
from math import isfinite
import time
import threading
import webbrowser
from urllib.error import URLError
from urllib.request import urlopen

from dash import Dash, Input, Output, State, ctx, dcc, html
import plotly.graph_objects as go
import yfinance as yf
import sys
sys.dont_write_bytecode = True

try:
    from .dashboard_shared import BORDER, NAVY as SHARED_NAVY
except ImportError:  # Direct execution of the standalone simulator.
    from dashboard_shared import BORDER, NAVY as SHARED_NAVY

# Colours, preset assumptions and available companies.
# Scenario growth and margin are percentages: 20 means 20%.
NAVY = "#081423"
CYAN = "#20e3e8"
WHITE = "#f3f5fa"
MUTED = "#aabed0"
LINE = "#354b60"

SCENARIOS = {
    "Bear": dict(growth=10, margin=20, pe=30, years=5),
    "Base": dict(growth=20, margin=25, pe=30, years=5),
    "Bull": dict(growth=30, margin=30, pe=30, years=5),
}

COMPANIES = {
    "NVIDIA": "NVDA", "Microsoft": "MSFT", "Alphabet": "GOOGL",
    "Amazon": "AMZN", "Meta": "META", "AMD": "AMD",
    "TSMC": "TSM", "Oracle": "ORCL", "Palantir": "PLTR",
    "ASML": "ASML",
}

_cache = {}


def p4_positive_number(value, label):
    """Convert a value to a positive, finite float or raise a labelled error."""
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Enter a valid number for {label}.") from error
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{label.capitalize()} must be a positive, finite number.")
    return value

# Load and validate company data before using it in the simulator.
def p4_load_company_data(company, force=False):
    """Return real, dated inputs. Never substitute fictional numbers on failure."""
    if company not in COMPANIES:
        raise ValueError("Choose a company from the list.")
    cached = _cache.get(company)
    if cached and not force and time.monotonic() - cached[0] < 900:
        return cached[1].copy()

    ticker = COMPANIES[company]
    stock = yf.Ticker(ticker)
    info = stock.get_info()
    price = p4_positive_number(info.get("regularMarketPrice"), "the share price")
    stamp = p4_positive_number(info.get("regularMarketTime"), "the quote timestamp")
    quote_date = datetime.fromtimestamp(stamp, timezone.utc).strftime("%d %b %Y %H:%M UTC")
    currency = info.get("currency")
    financial_currency = info.get("financialCurrency")
    if currency != "USD" or not financial_currency:
        raise ValueError("The listing/reporting currency could not be verified. Refresh to retry.")

    statement = stock.income_stmt
    if statement.empty or "Total Revenue" not in statement.index:
        raise ValueError("Annual revenue is unavailable from Yahoo. Refresh to retry.")
    revenues = statement.loc["Total Revenue"].dropna().sort_index(ascending=False)
    if revenues.empty:
        raise ValueError("Yahoo did not provide an annual revenue figure.")
    revenue_date = revenues.index[0].strftime("%d %b %Y")
    reported_revenue = p4_positive_number(revenues.iloc[0], "annual revenue")

    # Alphabet's class-A shares alone exclude other share classes. The implied
    # count includes all classes. For TSM, Yahoo already gives ADS-equivalent shares.
    share_value = info.get("impliedSharesOutstanding") or info.get("sharesOutstanding")
    shares = p4_positive_number(share_value, "total equivalent shares")
    market_cap = p4_positive_number(info.get("marketCap"), "market capitalisation")
    if abs(shares * price / market_cap - 1) > .10:
        raise ValueError("Yahoo's price and share units disagree. Refresh to retry.")

    fx_rate, fx_note = 1.0, "No currency conversion needed."
    if financial_currency != currency:
        fx_symbol = f"{financial_currency}{currency}=X"
        history = yf.Ticker(fx_symbol).history(period="5d", auto_adjust=False)
        if history.empty:
            raise ValueError(f"Could not load {financial_currency}/USD FX. Refresh to retry.")
        closes = history["Close"].dropna()
        if closes.empty:
            raise ValueError("The FX series has no valid closing rate.")
        fx_rate = p4_positive_number(closes.iloc[-1], "the currency conversion rate")
        fx_date = closes.index[-1].strftime("%d %b %Y")
        fx_note = f"1 {financial_currency} = {fx_rate:.6f} USD | {fx_symbol} | {fx_date}. FX held constant."

    share_note = "Yahoo total equivalent shares; all share classes included where reported."
    if ticker == "TSM":
        share_note = "Yahoo ADS-equivalent count. One TSM ADS represents five ordinary shares."
    data = {
        "company": company, "ticker": ticker, "revenue": reported_revenue * fx_rate,
        "reported_revenue": reported_revenue, "financial_currency": financial_currency,
        "shares": shares, "current_price": price, "currency": currency,
        "quote_date": quote_date, "revenue_date": revenue_date,
        "retrieved": datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC"),
        "source": "Yahoo Finance via yfinance", "mode": "Yahoo Finance",
        "fx_note": fx_note, "share_note": share_note,
    }
    _cache[company] = (time.monotonic(), data)
    return data.copy()
def p4_percent_to_decimal(percent):
    """Convert a percentage, such as 20, into 0.20."""
    if not isinstance(percent, (int, float)) or not isfinite(percent):
        raise ValueError("Enter a finite percentage.")
    return percent / 100

# Valuation calculations.
# Growth and margin use decimals here: 20% must be supplied as 0.20.
def p4_calculate_valuation(revenue, shares, current_price, growth, margin, pe, years):
    """Project a future price with constant shares and no dividends."""
    values = (revenue, shares, current_price, growth, margin, pe, years)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not isfinite(value) for value in values):
        raise ValueError("All inputs must be finite numbers.")
    if revenue <= 0 or shares <= 0 or current_price <= 0:
        raise ValueError("Revenue, shares and current price must be positive.")
    if growth <= -1:
        raise ValueError("Revenue growth must be greater than -100%.")
    if margin < 0:
        raise ValueError("Negative earnings are unsuitable for conventional P/E valuation.")
    if margin > 1:
        raise ValueError("Net margin must be between 0% and 100%.")
    if pe <= 0 or years <= 0 or years != int(years):
        raise ValueError("P/E must be positive; years must be a positive whole number.")
    try:
        # Project revenue, convert it into earnings, then calculate price per share and annual return.
        future_revenue = revenue * (1 + growth) ** years
        future_net_income = future_revenue * margin
        future_eps = future_net_income / shares
         # Calculate the compound annual return over the full investment period.
        future_price = future_eps * pe
        annual_return = (future_price / current_price) ** (1 / years) - 1
    except OverflowError as error:
        raise ValueError("These assumptions produce numbers that are too large.") from error
    result = dict(revenue=future_revenue, net_income=future_net_income,
                  eps=future_eps, price=future_price, annual_return=annual_return)
    if not all(isfinite(value) for value in result.values()):
        raise ValueError("These assumptions produce numbers that are too large.")
    return result


def p4_value_for(data, assumptions):
    """Project price and annual return from constant growth, margin, shares and P/E."""
    return p4_calculate_valuation(
        data["revenue"], data["shares"], data["current_price"],
        p4_percent_to_decimal(assumptions["growth"]),
        p4_percent_to_decimal(assumptions["margin"]), assumptions["pe"], assumptions["years"],
    )
# Sensitivity analysis: vary growth and margin while keeping P/E and years fixed.
def p4_heatmap_data(data, assumptions):
    """Recalculate the selected scenario across growth and margin assumptions."""
    growth = sorted(set([0, 10, 20, 30, 40, 50, assumptions["growth"]]))
    margins = sorted(set([15, 20, 25, 30, 35, 40, assumptions["margin"]]))
    results = [[p4_value_for(data, {**assumptions, "growth": g, "margin": m})
                for g in growth] for m in margins]
    return growth, margins, results

# Public entry points delegate to the integrated implementation below.
# Those functions are defined before these wrappers are called.
def p4_scenario_options(selected="Base"):
    return p4_integrated_scenario_options(selected)


def p4_create_layout():
    return p4_integrated_create_layout()


def p4_make_heatmap(data, assumptions):
    """Use the same valuation for chart cells, tooltips and headline metrics."""
    figure = go.Figure()
    figure.update_layout(
        paper_bgcolor=NAVY, plot_bgcolor=NAVY, font={"color": WHITE},
        margin={"l": 65, "r": 30, "t": 20, "b": 65},
        xaxis_title="<b>Revenue growth (%)</b>", yaxis_title="<b>Net profit margin (%)</b>",
    )
    if not data:
        figure.add_annotation(text="Select a company or refresh Yahoo data", showarrow=False,
                              x=.5, y=.5, xref="paper", yref="paper")
        return figure
    growth, margins, results = p4_heatmap_data(data, assumptions)
       # Each row represents a margin; each column represents a growth rate.
       # Display returns as percentages and retain prices for the hover details.
    returns = [[result["annual_return"] * 100 for result in row] for row in results]
    prices = [[result["price"] for result in row] for row in results]
    limit = max(15, max(abs(value) for row in returns for value in row))
    # Numeric grid positions keep every cell the same size for custom assumptions.
    figure.add_trace(go.Heatmap(
        x=list(range(len(growth))), y=list(range(len(margins))), z=returns,
        customdata=prices, text=[[f"{value:+.1f}%" for value in row] for row in returns],
        hovertext=[[f"Growth {g}% | Margin {m}%" for g in growth] for m in margins],
        texttemplate="%{text}", textfont={"color": NAVY, "size": 14},
        colorscale=[[0, "#ff9855"], [.5, "#edf0ec"], [1, CYAN]],
        zmin=-limit, zmax=limit, zmid=0, xgap=2, ygap=2,
        colorbar={"title": "Return %", "ticksuffix": "%"},
        hovertemplate="%{hovertext}<br>Annual return: %{z:.1f}%<br>Future price: $%{customdata:,.2f}<extra></extra>",
    ))
    axis_style = dict(
        showline=True,
        linecolor=WHITE,
        linewidth=2,
        ticks="outside",
        tickcolor=WHITE,
        ticklen=6,
        tickfont=dict(size=14, color=WHITE),
        title_font=dict(family="Arial", size=16, color=WHITE),
        title_standoff=14,
        automargin=True,
        showgrid=False,
        zeroline=False,
    )
    figure.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(growth))),
        ticktext=[f"{g}%" for g in growth],
        **axis_style,
    )
    figure.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(margins))),
        ticktext=[f"{m}%" for m in margins],
        **axis_style,
    )
    for scenario, dash in [(SCENARIOS["Base"], "solid"), (assumptions, "dash")]:
        if dash == "dash" and scenario == SCENARIOS["Base"]:
            continue
        x, y = growth.index(scenario["growth"]), margins.index(scenario["margin"])
        figure.add_shape(
             type="rect",
            x0=x-.47, x1=x+.47,
            y0=y-.47, y1=y+.47,
            line={"color": NAVY, "width": 4, "dash": dash},
            fillcolor="rgba(0,0,0,0)",
            layer="above",
        )
    return figure



def p4_render_results(data, scenario):
    return p4_integrated_render_results(data, scenario)


def p4_register_callbacks(app):
    return p4_integrated_register_callbacks(app)

# Assemble the standalone Dash app using the layout and callbacks.
def p4_create_app():
    """Build the standalone app without fetching data or starting a server on import."""
    app = Dash(__name__, title="Panel 4 - AI Valuation Simulator")
    app.layout = p4_create_layout()
    p4_register_callbacks(app)
    return app


def p4_open_when_ready(url, timeout=30):
    """Open the browser after Dash responds, allowing up to 30 seconds to start."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{url}/_dash-layout", timeout=1) as response:
                ready = response.status == 200
        except (URLError, OSError):
            ready = False
        if ready:
            webbrowser.open(url)
            return
        time.sleep(.2)
    print(f"The dashboard did not start within {timeout} seconds. "
          "Check the terminal for an error, then run the file again.")


# Shared dashboard styling and implementation.
# The public wrapper functions above also use this implementation in standalone mode.


p4_NAVY = SHARED_NAVY
p4_LINE = BORDER
p4_BUTTON = {
    "backgroundColor": p4_LINE,
    "color": WHITE,
    "border": "none",
    "padding": "10px 14px",
    "cursor": "pointer",
    "margin": "12px 0",
}

def p4_integrated_load_company_data(company, force=False):
    """Use the simulator's cached Yahoo data with guidance for manual inputs."""
    try:
        return p4_load_company_data(company, force=force)
    except ValueError as error:
        message = str(error)
        if message == "The listing/reporting currency could not be verified. Refresh to retry.":
            raise ValueError(
                "The listing/reporting currency could not be verified. Use manual USD inputs."
            ) from error
        if message == "Annual revenue is unavailable from Yahoo. Refresh to retry.":
            raise ValueError(
                "Annual revenue is unavailable from Yahoo. Try again or enter it manually."
            ) from error
        if message == "Yahoo's price and share units disagree. Refresh to retry.":
            raise ValueError(
                "Yahoo's price and share units disagree. Refresh or use verified manual inputs."
            ) from error
        if message.startswith("Could not load ") and message.endswith("/USD FX. Refresh to retry."):
            raise ValueError(
                message.removesuffix("Refresh to retry.") + "Use manual USD revenue."
            ) from error
        raise


def p4_integrated_scenario_options(selected="Base"):
    descriptions = {"Bear": "Cautious", "Base": "Central", "Bull": "Optimistic"}
    colours = {"Bear": "#ffb47d", "Base": CYAN, "Bull": "#8ce6ae"}

    return [
        {
            "value": name,
            "label": html.Div([
                html.Strong(
                    f"{name} - {descriptions[name]}"
                    + (" | SELECTED" if name == selected else "")
                ),
                html.Div(
                    f'{values["growth"]}% growth / {values["margin"]}% margin',
                    style={"fontSize": "13px", "marginTop": "5px"},
                ),
            ], style={
                "padding": "11px 12px",
                "borderRadius": "6px",
                "border": f"2px solid {colours[name]}",
                "backgroundColor": colours[name] if name == selected else p4_NAVY,
                "color": p4_NAVY if name == selected else WHITE,
                "flex": "1",
            }),
        }
        for name, values in SCENARIOS.items()
    ]

# Build the page layout: company controls on the left, results on the right.
def p4_integrated_create_layout():
    """Return Panel 4's layout for a standalone or shared Dash app."""

    def number(label, key):
        return html.Label(
            [
                label,
                dcc.Input(
                    id=f"panel4-{key}",
                    type="number",
                    min=0,
                    placeholder="Enter a positive number",
                    debounce=True,
                    style={
                        "width": "100%",
                        "boxSizing": "border-box",
                        "padding": "9px",
                    },
                ),
            ],
            style={"display": "block", "marginBottom": "12px"},
        )

    return html.Div(
        [   # Store company figures in this browser tab for the callbacks to use.
            dcc.Store(id="panel4-data", storage_type="memory"),
            html.H2("04 | AI Valuation Simulator", style={"color": CYAN}),
            html.Div(
                [
                    html.Aside(
                        [
                            html.Label("Company", htmlFor="panel4-company"),
                            dcc.Dropdown(
                                id="panel4-company",
                                options=list(COMPANIES),
                                value="NVIDIA",
                                clearable=False,
                                style={
                                    "color": WHITE,
                                    "--Dash-Fill-Inverse-Strong": p4_NAVY,
                                    "--Dash-Text-Strong": WHITE,
                                    "--Dash-Text-Weak": MUTED,
                                    "--Dash-Stroke-Strong": p4_LINE,
                                    "--Dash-Fill-Interactive-Strong": CYAN,
                                },
                            ),
                            html.Button(
                                "Refresh Yahoo data",
                                id="panel4-refresh",
                                n_clicks=0,
                                style=p4_BUTTON,
                            ),
                            dcc.Loading(
                                html.Div(
                                    id="panel4-status",
                                    role="status",
                                    style={"fontSize": "13px", "marginBottom": "16px"},
                                )
                            ),
                            html.H3("Scenario"),
                                                        dcc.RadioItems(
                                id="panel4-scenario",
                                options=p4_integrated_scenario_options(),
                                value="Base",
                                style={"display": "grid", "gap": "6px"},
                                inputStyle={
                                    "accentColor": CYAN,
                                    "marginRight": "8px",
                                },
                                labelStyle={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "cursor": "pointer",
                                },
                            ),

                            html.Div(
                                id="panel4-assumptions", style={"margin": "16px 0"}
                            ),
                            html.Details(
                                [
                                    html.Summary("Data & manual inputs"),
                                    html.P(
                                        "Use USD figures for the selected listing. Refresh restores Yahoo data."
                                    ),
                                    number("Share / ADS price (USD)", "price"),
                                    number("Annual revenue (USD billions)", "revenue"),
                                    number(
                                        "Total equivalent shares (billions)", "shares"
                                    ),
                                    html.Button(
                                        "Apply manual inputs",
                                        id="panel4-apply",
                                        n_clicks=0,
                                        style=p4_BUTTON,
                                    ),
                                ],
                               style={"display": "none"},
                            ),
                        ],
                    style={"flex": "0 1 290px", "minWidth": "230px"},
                    ),
                    html.Main(
                        [
                            dcc.Loading(
                                html.Div(
                                    [
                                        html.Div(
                                            id="panel4-source",
                                        style={"display": "none"},
                                        ),
                                        html.Div(
                                            id="panel4-metrics",
                                            style={
                                                "display": "flex",
                                                "flexWrap": "wrap",
                                                "gap": "12px",
                                                "margin": "18px 0",
                                            },
                                        ),
                                        html.H3("Annual return sensitivity"),
                                        html.P(
                                            "Turquoise = gain | White = 0% | Orange = loss. Solid outline = Base; dashed outline = selected scenario.",
                                            style={
                                                "color": MUTED,
                                                "fontSize": "13px",
                                            },
                                        ),
                                                                                html.Button(
                                            "Reset view",
                                            id="panel4-reset-view",
                                            n_clicks=0,
                                            style={**p4_BUTTON, "margin": "0 0 8px"},
                                        ),
                                        dcc.Graph(
                                            id="panel4-heatmap",
                                            config={
                                                "displayModeBar": True,
                                                "displaylogo": False,
                                                "modeBarButtons": [
                                                    ["zoomIn2d", "zoomOut2d", "resetScale2d"]
                                                ],
                                                "doubleClick": "reset",
                                            },
                                            style={"height": "480px"},
                                        ),

                                        html.Details(
                                            [
                                                html.Summary("Compare scenarios"),
                                                html.Div(
                                                    id="panel4-comparison",
                                                    style={"overflowX": "auto"},
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ),
                            html.Details(
                                [
                                    html.Summary("How it works"),
                                    html.P(
                                        "Future price = annual revenue Ã— (1 + growth)^years Ã— net margin Ã· equivalent shares Ã— future P/E."
                                    ),
                                    html.P(
                                        "Annual return = (future price Ã· current price)^(1 Ã· years) âˆ’ 1. Growth, margin, shares, P/E and FX stay constant. Presets are illustrative assumptions. Excludes dividends, fees and taxes."
                                    ),
                                ],
                                style={"marginTop": "20px"},
                            ),
                        ],
                        style={"flex": "4 1 600px", "minWidth": "0"},
                    ),
                ],
                style={"display": "flex", "flexWrap": "wrap", "gap": "28px"},
            ),
        ],
        id="panel4",
        style={
            "backgroundColor": p4_NAVY,
            "color": WHITE,
            "fontFamily": "Arial, sans-serif",
            "padding": "24px",
            "minHeight": "100vh",
            "boxSizing": "border-box",
        },
    )


def p4_integrated_make_heatmap(data, assumptions):
    """Apply the integrated dashboard's colours and manual-input prompt."""
    figure = p4_make_heatmap(data, assumptions)
    figure.update_layout(paper_bgcolor=p4_NAVY, plot_bgcolor=p4_NAVY)
    figure.update_traces(textfont_color=p4_NAVY, selector={"type": "heatmap"})
    figure.update_shapes(line_color=p4_NAVY)
    if not data:
        figure.update_annotations(text="Load Yahoo data or enter manual inputs")
    return figure

# Update company figures from Yahoo or manual inputs.
# If an update fails, keep previous figures only for the same company.
def p4_integrated_update_company_data(company, trigger, price, revenue, shares, previous):
    """Keep each browser's selected/manual inputs in its own dcc.Store."""
    saved = previous if previous and previous.get("company") == company else None
    try:
        if trigger == "panel4-apply":
            if company not in COMPANIES:
                raise ValueError("Choose a company from the list.")
            data = {
                "company": company,
                "ticker": COMPANIES[company],
                "current_price": p4_positive_number(price, "share price"),
                "revenue": p4_positive_number(revenue, "annual revenue") * 1000000000.0,
                "shares": p4_positive_number(shares, "equivalent shares")
                * 1000000000.0,
                "mode": "Manual inputs",
                "source": "User-entered USD inputs",
                "quote_date": "User supplied; date unspecified",
                "revenue_date": "User supplied; date unspecified",
                "retrieved": datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC"),
                "share_note": "User-supplied equivalent share count.",
                "fx_note": "Revenue entered in USD. FX stays constant.",
            }
        else:
            data = p4_integrated_load_company_data(company, force=trigger == "panel4-refresh")
        for scenario in SCENARIOS.values():
            p4_heatmap_data(data, scenario)
        return (data, f"{data['mode']} | Retrieved {data['retrieved']}")
    except Exception as error:
        status = f"Could not update data: {error} "
        status += (
            "Showing previous inputs." if saved else "Retry or enter manual USD inputs."
        )
        return (saved, status)


def p4_integrated_render_results(data, scenario):
    assumptions = SCENARIOS[scenario]
    description = f"Growth: {assumptions['growth']}% / year | Net margin: {assumptions['margin']}% | P/E: {assumptions['pe']}x | Period: {assumptions['years']} years"
    if not data:
        return (
            [],
            p4_integrated_make_heatmap(None, assumptions),
            [],
            "No company figures loaded.",
            description,
        )
    result = p4_value_for(data, assumptions)
    metrics = []
    for title, value in [
        ("Current price (USD)", f"${data['current_price']:,.2f}"),
        (
            f"Est. price in {assumptions['years']} years (USD)",
            f"${result['price']:,.2f}",
        ),
        ("Est. annual return", f"{result['annual_return']:+.1%}"),
    ]:
        metrics.append(
            html.Div(
                [
                    html.Div(title),
                    html.Strong(value, style={"fontSize": "28px", "color": CYAN}),
                ],
                style={
                    "border": f"1px solid {p4_LINE}",
                    "padding": "14px",
                    "flex": "1 1 180px",
                },
            )
        )
    rows = []
        # Calculate each preset using the same company data for a fair comparison.
    for name, values in SCENARIOS.items():
        outcome = p4_value_for(data, values)
        cells = [
            name,
            f"{values['growth']}%",
            f"{values['margin']}%",
            values["pe"],
            values["years"],
            f"${outcome['price']:,.2f}",
            f"{outcome['annual_return']:+.1%}",
        ]
        rows.append(
            html.Tr([html.Td(value, style={"padding": "10px"}) for value in cells])
        )
    table = html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th(label, style={"padding": "10px"})
                        for label in [
                            "Scenario",
                            "Growth",
                            "Margin",
                            "P/E",
                            "Years",
                            "Future price",
                            "Annual return",
                        ]
                    ]
                )
            ),
            html.Tbody(rows),
        ],
        style={"width": "100%", "textAlign": "left"},
    )
    source = html.Div(
        [
            html.Div(f"{data['company']} ({data['ticker']}) | {data['source']}"),
            html.Div(
                f"Revenue: ${data['revenue'] / 1000000000.0:,.2f}bn | Equivalent shares: {data['shares'] / 1000000000.0:,.3f}bn"
            ),
            html.Div(
                f"Quote: {data['quote_date']} | Annual period ended: {data['revenue_date']}"
            ),
            html.Div(data["share_note"]),
            html.Div(data["fx_note"]),
        ]
    )
    return (metrics, p4_integrated_make_heatmap(data, assumptions), table, source, description)


def p4_integrated_register_callbacks(app):
    """Call once on the shared app after adding create_layout() to its layout."""
        # Callbacks connect user actions to updates on the page.
        # Inputs trigger updates; State supplies values without triggering them.
    @app.callback(
        Output("panel4-scenario", "options"),
        Input("panel4-scenario", "value"),
    )
    def highlight_scenario(scenario):
        return p4_integrated_scenario_options(scenario)

    @app.callback(
        Output("panel4-data", "data"),
        Output("panel4-status", "children"),
        Input("panel4-company", "value"),
        Input("panel4-refresh", "n_clicks"),
        Input("panel4-apply", "n_clicks"),
        State("panel4-price", "value"),
        State("panel4-revenue", "value"),
        State("panel4-shares", "value"),
        State("panel4-data", "data"),
    )
    def update_data(company, _refresh, _apply, price, revenue, shares, previous):
        return p4_integrated_update_company_data(
            company, ctx.triggered_id, price, revenue, shares, previous
        )

    @app.callback(
        Output("panel4-metrics", "children"),
        Output("panel4-heatmap", "figure"),
        Output("panel4-comparison", "children"),
        Output("panel4-source", "children"),
        Output("panel4-assumptions", "children"),
        Input("panel4-data", "data"),
        Input("panel4-scenario", "value"),
        Input("panel4-reset-view", "n_clicks"),

    )
    def update_results(data, scenario, reset_clicks):
        metrics, figure, table, source, description = p4_integrated_render_results(data, scenario)
        if ctx.triggered_id == "panel4-reset-view":
            figure.update_layout(
                uirevision=f"reset-{reset_clicks}",
                xaxis_autorange=True,
                yaxis_autorange=True,
            )
        return metrics, figure, table, source, description

# Run the simulator only when this file is executed directly.
# Importing it into the main dashboard does not start another server.
if __name__ == "__main__":
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app = p4_create_app()
    print("Starting Panel 4. Your browser will open when Dash is ready. "
          "Keep this terminal running.")
    threading.Thread(target=p4_open_when_ready, args=("http://127.0.0.1:8056",),
                     daemon=True).start()
    app.run(debug=False, host="127.0.0.1", port=8056)
