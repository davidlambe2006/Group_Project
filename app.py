"""AI Boom dashboard controller.

Install: python -m pip install dash==4.5.0 plotly==7.1.0 yfinance==1.8.0 pandas==3.1.1 numpy==2.5.4 Flask==3.1.3
Run: python app.py
Open: http://127.0.0.1:8050

The controller mounts the panels, handles navigation and the universal company
selector, and reuses analysis and chart functions from the panel modules.
Appearance uses Dash's native Python style dictionaries. Real market data is
fetched from Yahoo Finance at runtime and cached in memory for 15 minutes.
Internet is required for market data; the simulator also accepts manual inputs.

"""

import sys
sys.dont_write_bytecode = True

import argparse
import threading
import time
from functools import partial, wraps
from urllib.parse import parse_qs, urlencode
import pandas as pd
import plotly.graph_objects as go
from flask import Flask
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from panels.panel1_stock import (
    BENCHMARKS as stock_BENCHMARKS,
    p1_build_chart_3d as stock_build_chart_3d,
    p1_comparison_data,
    p1_get_prices as stock_get_prices,
    p1_create_app,
)
from panels.panel2_valuation import (
    p2_build_valuation_chart,
    p2_calculate_regression,
    p2_create_app,
    p2_load_valuation_data,
    p2_prepare_chart_data,
    tickers as p2_tickers,
)
from panels.panel3_dotcom import p3_create_app, p3_render_panel3
from panels.panel4_simulator import (
    SCENARIOS as p4_SCENARIOS,
    p4_integrated_create_layout,
    p4_integrated_load_company_data as p4_load_company_data,
    p4_integrated_make_heatmap as p4_make_heatmap,
    p4_integrated_register_callbacks,
)
from panels.dashboard_shared import COMPANIES, TICKERS, NAVY as SHARED_NAVY, CARD, apply_theme

stock_COMPANIES = dict(COMPANIES)
stock_DEFAULT_COMPANIES = list(TICKERS)
GLOBAL_COMPANIES = list(stock_COMPANIES)
GLOBAL_PANELS = (1, 2, 3)


# Universal company selection


def global_normalise_selection(values):
    """Valid tickers only, without duplicates, in the canonical 10-company order."""
    chosen = {str(value).strip().upper() for value in (values or [])}
    return [ticker for ticker in GLOBAL_COMPANIES if ticker in chosen]


def global_parse_selection(search):
    """Read '?companies=NVDA,MSFT,...' from a URL query string; None if absent."""
    query = parse_qs((search or "").lstrip("?"), keep_blank_values=True)
    if "companies" not in query:
        return None
    return global_normalise_selection(",".join(query["companies"]).split(","))


def global_panel_url(panel, selection):
    """Address of a full-size panel carrying the header selection."""
    chosen = global_normalise_selection(selection)
    if panel in GLOBAL_PANELS:
        query = urlencode({"companies": ",".join(chosen)}, safe=",")
        return f"/panel/{panel}/?{query}"
    return f"/panel/{panel}/"

# Shared overview data cache
def cached_market_data(function):
    responses = {}
    lock = threading.Lock()

    @wraps(function)
    def wrapped(*args, **kwargs):
        key = repr((args, kwargs))
        try:
            trigger = ctx.triggered_id or ""
            refresh = ("refresh" in trigger or "retry" in trigger) and bool(
                ctx.triggered[0].get("value")
            )
        except Exception:
            refresh = False
        with lock:
            previous = responses.get(key)
            if previous and time.monotonic() - previous[0] < 900 and (not refresh):
                return previous[1]
            if refresh and hasattr(function, "cache_clear"):
                function.cache_clear()
            result = function(*args, **kwargs)
            if isinstance(result, pd.DataFrame) and result.empty:
                return result
            if (
                isinstance(result, tuple)
                and len(result) == 2
                and isinstance(result[0], pd.DataFrame)
                and result[1]
            ):
                return result
            responses[key] = (time.monotonic(), result)
            return result

    return wrapped


stock_get_prices = cached_market_data(stock_get_prices)
p2_load_valuation_data = cached_market_data(p2_load_valuation_data)
TITLES = [
    "AI Stock Performance",
    "Capital Investment vs Revenue",
    "AI Boom vs. Dot-Com Bubble",
    "AI Valuation Simulator",
]
NAVY = SHARED_NAVY
WHITE = "#edf3fc"
CYAN = "#5ce0d3"




def global_chip_style(selected):
    """One company toggle: cyan when picked."""
    style = {
        "padding": "3px 9px",
        "fontSize": "11px",
        "fontWeight": 600,
        "letterSpacing": "0.5px",
        "borderRadius": "12px",
        "border": f"1px solid {CYAN if selected else '#263449'}",
        "backgroundColor": "#143137" if selected else "#0f1b2d",
        "color": CYAN if selected else "#9fb2c8",
        "cursor": "pointer",
        "flex": "0 0 auto",
    }
    return style


def global_status(chosen):
    """Show the number selected from all twelve companies."""
    return html.Span(
        f"{len(chosen)} / {len(GLOBAL_COMPANIES)} selected", style={"color": CYAN}
    )


# Overview graph integration


def compact_figure(figure, panel):
    """Fit each original chart to a responsive overview quadrant."""
    figure = go.Figure(figure)
    # Consistent axis-title formatting across all overview panels
    axis_title_font = dict(
        family="Arial",
        size=13,
        color="white"
    )

    if panel == 1:
        figure.update_scenes(
            xaxis_title_font=axis_title_font,
            yaxis_title_font=axis_title_font,
            zaxis_title_font=axis_title_font
        )
    else:
        figure.update_xaxes(title_font=axis_title_font)
        figure.update_yaxes(title_font=axis_title_font)

    if panel == 4:
        # Retain Panel 4's original preview styling and independent data.
        figure.update_layout(autosize=True, height=None, width=None, title=None,
                             margin=dict(l=60, r=65, t=10, b=45), uirevision="overview-4")
        return figure
    figure.update_layout(autosize=True, height=None, width=None, title=None,
        paper_bgcolor=SHARED_NAVY, plot_bgcolor=SHARED_NAVY, font=dict(size=11),
        margin=dict(l=55, r=24, t=38, b=48), uirevision=f"overview-{panel}")
    if panel == 1:
        zoom = 1.0  # above 1 zooms out, below 1 zooms in
        figure.update_layout(margin=dict(l=0, r=0, t=56, b=0),
            scene=dict(domain=dict(x=[0, 1], y=[0, 1]),
            aspectratio=dict(x=2.9, y=0.9, z=0.9),
            camera=dict(eye=dict(x=0.35 * zoom, y=-1.99 * zoom, z=0.99 * zoom),
            center=dict(x=0, y=0, z=-0.3))),
            legend=dict(orientation="h", x=0.5, xanchor="center", y=1.0,
                        yanchor="bottom", font=dict(size=13, color="white"), bgcolor=SHARED_NAVY))
        figure.update_scenes(
            xaxis_tickfont_size=11,
            zaxis_tickfont_size=11,
            yaxis_showticklabels=False
        )
    elif panel == 2:
        # Remove expanded-panel annotations and custom legend boxes
        figure.layout.annotations = []
        figure.layout.shapes = []
        figure.layout.updatemenus = []
        
        sectors = set()
        for trace in figure.data:
            if trace.type == "scatter" and trace.marker.sizeref:
                trace.marker.sizeref *= 1.7
                trace.showlegend = trace.name not in sectors
                sectors.add(trace.name)
            else:
                trace.showlegend = False
        figure.update_layout(showlegend=False)
        figure.update_xaxes(domain=[0, 1], autorange=True, automargin=True)
        figure.update_yaxes(
             autorange=True,
             tickmode="linear",
             tick0=0,
             dtick=250,
             tickfont_size=11,
             automargin=True
        )
    elif panel == 3:
        # Compact overview keys only; full-size names and annotations stay intact.
        for trace, label in zip(figure.data, ["NASDAQ · Dot-Com", "AI Portfolio", "Bull", "Base", "Bear"]):
            trace.name = label
        figure.update_layout(margin=dict(l=55, r=110, t=42, b=46),
            legend=dict(orientation="h", x=0, y=1.04, yanchor="bottom", font=dict(size=9)))
        figure.update_xaxes(tickfont_size=10)
        figure.update_yaxes(tickfont_size=10)
        figure.update_xaxes(title_text="<b>Months Since Start</b>")
        figure.update_yaxes(title_text="<b>Indexed Performance · Base 100</b>")
    return figure


def message_figure(message):
    figure = go.Figure()
    figure.update_layout(
        paper_bgcolor=NAVY,
        plot_bgcolor=NAVY,
        font=dict(color=WHITE),
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
            )
        ],
    )
    return figure


def overview_figure(panel, companies=None):
    """Reuse the original panel calculations and figures with their real data.

    Panels 1-3 use the header selection; Panel 4 remains independent.
    """
    chosen = (
        list(stock_DEFAULT_COMPANIES)
        if companies is None else global_normalise_selection(companies)
    )
    if panel in GLOBAL_PANELS and not chosen:
        return message_figure("Please select at least one company in the universal selector.")
    if panel == 1:
        indexed, _missing = p1_comparison_data(
            chosen, "3 Years", list(stock_BENCHMARKS), load_prices=stock_get_prices
        )
        sampled = indexed.iloc[::5]
        if sampled.index[-1] != indexed.index[-1]:
            sampled = pd.concat([sampled, indexed.iloc[[-1]]])
        figure = stock_build_chart_3d(sampled)
    elif panel == 2:
        data, failed = p2_load_valuation_data(p2_tickers)
        cleaned = p2_prepare_chart_data(data[data["Ticker"].isin(chosen)])
        if cleaned.empty:
            raise ValueError("Yahoo Finance metrics unavailable. Open panel to retry.")
        figure = p2_build_valuation_chart(cleaned, p2_calculate_regression(cleaned))
    elif panel == 3:
        figure = p3_render_panel3(chosen)
    else:
        data = p4_load_company_data("NVIDIA")
        figure = p4_make_heatmap(data, p4_SCENARIOS["Base"])
    return compact_figure(figure, panel)


# Main dashboard layout and shared callbacks


def create_app():
    server = Flask(__name__)
    panel_apps = []
    factories = (
        partial(p1_create_app, parse_selection=global_parse_selection, load_prices=stock_get_prices),
        partial(p2_create_app, parse_selection=global_parse_selection, load_valuations=p2_load_valuation_data),
        p3_create_app,
    )
    for number, factory in enumerate(factories, 1):
        panel_apps.append(
            factory(
                server=server,
                url_base_pathname=f"/panel/{number}/",
                title=TITLES[number - 1],
            )
        )
    fourth = Dash(
        "simulator",
        server=server,
        url_base_pathname="/panel/4/",
        title=TITLES[3],
        include_assets_files=False,
    )
    fourth.layout = p4_integrated_create_layout()
    p4_integrated_register_callbacks(fourth)
    panel_apps.append(fourth)
    for panel_app in panel_apps:
        apply_theme(panel_app)
        panel_app.layout = html.Div(
            panel_app.layout,
            style={
                "position": "absolute",
                "inset": 0,
                "backgroundColor": NAVY,
                "minHeight": "100vh",
                "color": WHITE,
            },
        )
    app = Dash(
        __name__,
        server=server,
        title="AI Boom • Four Graph Dashboard",
        include_assets_files=False,
    )
    apply_theme(app)
    cards = []
    for number, title in enumerate(TITLES, 1):
        cards.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                f"0{number}",
                                style={"color": CYAN, "marginRight": "10px"},
                            ),
                            title,
                        ],
                        style={
                            "height": "36px",
                            "lineHeight": "36px",
                            "padding": "0 14px",
                            "fontSize": "16px",
                            "fontWeight": 700,
                            "backgroundColor": CARD,
                        },
                    ),
                    html.Div(
                        dcc.Loading(
                            dcc.Graph(
                                id=f"graph-{number}",
                                figure=message_figure("Loading Yahoo Finance data…"),
                                responsive=True,
                                config={"displayModeBar": False, "displaylogo": False},
                                style={"height": "100%", "width": "100%"},
                            ),
                            parent_style={"height": "100%", "width": "100%"},
                        ),
                        style={
                            "position": "absolute",
                            "top": "36px",
                            "bottom": 0,
                            "left": 0,
                            "right": 0,
                        },
                    ),
                    html.Button(
                        "",
                        id=f"open-{number}",
                        n_clicks=0,
                        title=f"Open {title}",
                        **{"aria-label": f"Open {title}"},
                        style={
                            "position": "absolute",
                            "inset": 0,
                            "width": "100%",
                            "height": "100%",
                            "border": 0,
                            "background": "transparent",
                            "cursor": "pointer",
                            "zIndex": 2,
                        },
                    ),
                ],
                id=f"card-{number}",
                className="overview-card",
                style={
                    "position": "relative",
                    "minWidth": 0,
                    "minHeight": 0,
                    "overflow": "hidden",
                    "backgroundColor": NAVY,
                },
            )
        )
    grid_style = {
        "height": "calc(100vh - 52px)",
        "display": "grid",
        "gridTemplateColumns": "repeat(2,minmax(0,1fr))",
        "gridTemplateRows": "repeat(2,minmax(0,1fr))",
        "gap": "1px",
        "backgroundColor": "#263449",
    }
    chips = [
        html.Button(
            ticker,
            id=f"chip-{ticker}",
            n_clicks=0,
            title=f"{name} ({ticker})",
            disabled=False,
            style=global_chip_style(ticker in stock_DEFAULT_COMPANIES),
        )
        for ticker, name in stock_COMPANIES.items()
    ]
    frames = [
        html.Div(
            html.Iframe(
                id=f"frame-{i}",
                title=TITLES[i - 1],
                style={"border": 0, "width": "100%", "height": "100%"},
            ),
            id=f"full-{i}",
            style={"display": "none", "height": "calc(100vh - 52px)"},
        )
        for i in range(1, 5)
    ]
    app.layout = html.Div(
        [
            dcc.Interval(id="load-data", interval=150, max_intervals=1),
            dcc.Store(id="company-selection", data=list(stock_DEFAULT_COMPANIES)),
            dcc.Store(id="applied-companies", data=list(stock_DEFAULT_COMPANIES)),
            dcc.Store(id="current-panel", data=None),
            html.Div(
                [
                    html.Div(
                        [
                            html.Strong(
                                "AI BOOM",
                                style={
                                    "color": CYAN,
                                    "letterSpacing": "2px",
                                    "flex": "0 0 auto",
                                },
                            ),
                            html.Span(
                                "COMPANIES",
                                title=(
                                    f"Pick any of the "
                                    f"{len(GLOBAL_COMPANIES)} companies. Applies to "
                                    "panels 1-3. Panel 4 has its own company selector."
                                ),
                                style={
                                    "color": "#7f93ad",
                                    "fontSize": "10px",
                                    "letterSpacing": "1.5px",
                                    "flex": "0 0 auto",
                                },
                            ),
                            html.Div(
                                chips,
                                id="company-chips",
                                style={
                                    "display": "flex",
                                    "gap": "6px",
                                    "alignItems": "center",
                                    "minWidth": "140px",
                                    "overflowX": "auto",
                                    "scrollbarWidth": "none",
                                },
                            ),
                            html.Button("Select All", id="select-all", n_clicks=0, style={"whiteSpace": "nowrap"}),
                            html.Button("Deselect All", id="deselect-all", n_clicks=0, style={"whiteSpace": "nowrap"}),
                            html.Span(
                                global_status(list(stock_DEFAULT_COMPANIES)),
                                id="selection-status",
                                style={
                                    "fontSize": "11px",
                                    "whiteSpace": "nowrap",
                                    "flex": "0 0 auto",
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "8px",
                            "minWidth": 0,
                            "overflowX": "auto",
                        },
                    ),
                    html.Button(
                        "← Return to overview",
                        id="back",
                        n_clicks=0,
                        style={"display": "none"},
                    ),
                ],
                style={
                    "height": "52px",
                    "padding": "0 12px",
                    "overflowX": "auto",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                    "gap": "16px",
                    "backgroundColor": "#0B1220",
                },
            ),
            html.Div(cards, id="grid", className="overview-grid", style=grid_style),
            *frames,
        ],
        style={
            "position": "fixed",
            "inset": 0,
            "backgroundColor": NAVY,
            "color": WHITE,
            "fontFamily": "Segoe UI, Arial, sans-serif",
        },
    )

    @app.callback(
        Output("company-selection", "data"),
        *[Input(f"chip-{ticker}", "n_clicks") for ticker in GLOBAL_COMPANIES],
        Input("select-all", "n_clicks"), Input("deselect-all", "n_clicks"),
        State("company-selection", "data"),
        prevent_initial_call=True,
    )
    def toggle_company(*args):
        *_clicks, current = args
        trigger = ctx.triggered_id
        if trigger == "select-all":
            return list(TICKERS)
        if trigger == "deselect-all":
            return []
        if not trigger or not str(trigger).startswith("chip-"):
            raise PreventUpdate
        ticker = str(trigger).split("-", 1)[1]
        chosen = global_normalise_selection(current)
        if ticker in chosen:
            chosen.remove(ticker)
        elif ticker in GLOBAL_COMPANIES:
            chosen.append(ticker)
        else:
            raise PreventUpdate
        return global_normalise_selection(chosen)

    @app.callback(
        *[Output(f"chip-{ticker}", "style") for ticker in GLOBAL_COMPANIES],
        *[Output(f"chip-{ticker}", "disabled") for ticker in GLOBAL_COMPANIES],
        Output("selection-status", "children"),
        Input("company-selection", "data"),
    )
    def show_selection(selection):
        chosen = global_normalise_selection(selection)
        return [
            *[global_chip_style(t in chosen) for t in GLOBAL_COMPANIES],
            *[False for t in GLOBAL_COMPANIES],
            global_status(chosen),
        ]

    @app.callback(
        Output("applied-companies", "data"),
        Input("company-selection", "data"),
        prevent_initial_call=True,
    )
    def apply_selection(selection):
        chosen = global_normalise_selection(selection)
        return chosen

    def register_preview(number):
        uses_selection = number in GLOBAL_PANELS
        inputs = [Input("load-data", "n_intervals")]
        if uses_selection:
            inputs.append(Input("applied-companies", "data"))

        @app.callback(
            Output(f"graph-{number}", "figure"),
            *inputs,
            prevent_initial_call=True,
        )
        def load(_count, companies=None):
            try:
                return overview_figure(number, companies)
            except Exception as error:
                print(f"Panel {number}: {error}")
                return message_figure("Data unavailable — open panel to retry")

    for number in range(1, 5):
        register_preview(number)

    @app.callback(
        Output("grid", "style"),
        Output("back", "style"),
        *[Output(f"full-{i}", "style") for i in range(1, 5)],
        *[Output(f"frame-{i}", "src") for i in range(1, 5)],
        Output("current-panel", "data"),
        *[Input(f"open-{i}", "n_clicks") for i in range(1, 5)],
        Input("back", "n_clicks"),
        Input("applied-companies", "data"),
        State("current-panel", "data"),
        prevent_initial_call=True,
    )
    def navigate(*args):
        *_clicks, applied, open_panel = args
        trigger = ctx.triggered_id
        chosen = global_normalise_selection(applied)
        if trigger == "applied-companies":
            if open_panel not in GLOBAL_PANELS:
                raise PreventUpdate
            selected = open_panel
        else:
            selected = (
                int(trigger.split("-")[1])
                if trigger and trigger.startswith("open-")
                else None
            )
        grid = {**grid_style, "display": "none" if selected else "grid"}
        back = {
            "display": "block" if selected else "none",
            "backgroundColor": "#143137",
            "color": CYAN,
            "border": "1px solid #5ce0d3",
            "borderRadius": "5px",
            "padding": "6px 12px",
            "cursor": "pointer",
        }
        full = [
            {
                "display": "block" if selected == i else "none",
                "height": "calc(100vh - 52px)",
            }
            for i in range(1, 5)
        ]
        sources = [
            global_panel_url(i, chosen) if selected == i else no_update
            for i in range(1, 5)
        ]
        return [grid, back, *full, *sources, selected]
    

    app.panel_apps = panel_apps
    return app


# Application startup

app = create_app()
server = app.server
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Boom Dash dashboard")
    parser.add_argument("--port", type=int, default=8050)
    arguments = parser.parse_args()
    print(f"Open http://127.0.0.1:{arguments.port} in your browser.")
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app.run(host="127.0.0.1", port=arguments.port, debug=False)

