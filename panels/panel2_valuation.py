# Panel 2: The AI Bubble - Capital Investment vs Revenue
# Compares major technology companies using Yahoo Finance data.
# Bubble position shows CapEx and annual revenue, while bubble size
# represents market capitalisation. Includes regression analysis,
# a company comparison table and an interactive zoomed-in chart.
# Runs independently or as part of the four-panel dashboard.


import yfinance as yf
import plotly.express as px
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, dash_table, Input, Output
import plotly.graph_objects as go
import sys
sys.dont_write_bytecode = True

try:
    from .dashboard_shared import TICKERS, CARD, GRID
except ImportError:
    from dashboard_shared import TICKERS, CARD, GRID

# Companies and sectors used for bubble colours.
tickers = list(TICKERS)

sector_map = {
    "NVDA": "Semiconductors",
    "MSFT": "Software & Cloud",
    "GOOGL": "Software & Cloud",
    "AMZN": "Software & Cloud",
    "META": "Internet Services",
    "AMD": "Semiconductors",
    "TSM": "Semiconductors",
    "ORCL": "Software & Cloud",
    "PLTR": "Software & Cloud",
    "ASML": "Semiconductors"
}

# Financial columns that must contain valid numbers for the anlaysis
NUMERIC_COLS = ['Market Cap ($T)', 'Capital Expenditure ($B)', 'Annual Revenue ($B)']

def p2_load_valuation_data(selected_tickers):
    """Download Yahoo Finance financial metrics for the requested companies.

    A failed ticker is recorded with missing values so other tickers still load.
    """
    # Store each companies financial data and track failed downlaods separately
    rows = []
    failed_tickers = []

    # Retrieving financial information for each selected company
    for ticker in selected_tickers:
        name, cap, capex, revenue = ticker, None, None, None
        try:
            company = yf.Ticker(ticker)
            info = company.info
            name = info.get('displayName') or info.get('shortName') or ticker
            # Retrieving market capitalisation, revenue and cash flow data
            cap = info.get('marketCap')
            income = company.income_stmt
            cashflow = company.cashflow

            # Only continue if both financial statements contain data
            if not income.empty and not cashflow.empty:
                if 'Total Revenue' in income.index and 'Capital Expenditure' in cashflow.index:
                    # Use the latest year with valid figures to ensure a fair comparison.
                    common_years = income.columns.intersection(cashflow.columns)
                    for year in sorted(common_years, reverse=True):
                        rev = income.loc['Total Revenue', year]
                        expenditure = cashflow.loc['Capital Expenditure', year]
                        # Only use a year with both Revenue and Capital Expenditure figures are available
                        if pd.notna(rev) and pd.notna(expenditure):
                             # Identify currencies for financial statements and market capitalisation
                            currency = info.get('financialCurrency')
                            market_currency = info.get('currency')
                            
                            if not currency:
                                 raise ValueError(f"Financial currency unavailable for {ticker}")
                            
                            if cap is not None and not market_currency:
                                 raise ValueError(f"Market cap currency unavailable for {ticker}")
                            
                            # Convert revenue and CapEx into USD using a historical FX average.
                            # Both figures are flows measured over a financial reporting period,
                            # rather than values measured at one point in time.
                            fx_rate = 1.0
                            if currency != 'USD':
                                 # Construct the Yahoo Finance Ticker for conversion into USD
                                 fx_ticker = f"{currency}USD=X"
                                 # Download exchange rates over precednig annual reporting years
                                 fx_data = yf.Ticker(fx_ticker).history(
                                      start=year - pd.DateOffset(years=1),
                                      end=year + pd.Timedelta(days=1)
                                )
                                 
                                 if fx_data.empty or fx_data['Close'].dropna().empty:
                                     raise ValueError(f"No exchange rate available for {currency}")

                                 # Average available daily closing rates to estimate the annual conversion rate
                                 fx_rate = fx_data['Close'].dropna().mean()

                                 if not np.isfinite(fx_rate) or fx_rate <= 0:
                                     raise ValueError(f"Invalid exchange rate for {currency}")
                                 
                            revenue = rev * fx_rate / 1e9 # Convert revenue to USD billions
                            capex = abs(expenditure) * fx_rate / 1e9 # Express Capital Expenditure as positive in USD billions
                            
                            # Market capitalisation is a current valuation, unlike annual revenue
                            # and CapEx. Convert it using the latest available FX closing price.
                            if cap is not None and market_currency != 'USD':
                                fx_ticker = f"{market_currency}USD=X"
                                fx_data = yf.Ticker(fx_ticker).history(period='5d')
                                
                                if fx_data.empty or fx_data['Close'].dropna().empty:
                                    raise ValueError(f"No market cap exchange rate for {market_currency}")

                                # Select the latest avialable closing exchange rate
                                fx_rate_market = fx_data['Close'].dropna().iloc[-1]
                                if not np.isfinite(fx_rate_market) or fx_rate_market <= 0:
                                    raise ValueError(f"Invalid market cap exchange rate for {market_currency}")
                                
                                # Converting market capitalisation to USD
                                cap = cap * fx_rate_market
                                
                            break
                        
        except Exception as error:
            failed_tickers.append(ticker)
            cap, capex, revenue = None, None, None
            print(f"Error retrieving data for {ticker}: {error}")

        # One dictionary per ticker avoids misaligned separate data lists.
        rows.append({
            'Company': name,
            'Ticker': ticker,
            'Market Cap ($T)': cap / 1e12 if cap is not None else None, # Convert dollars to trillions
            'Capital Expenditure ($B)': capex,
            'Annual Revenue ($B)': revenue,
            'Sector': sector_map.get(ticker, 'Other'),
        })

    return pd.DataFrame(rows, columns=[
        'Company', 'Ticker', *NUMERIC_COLS, 'Sector'
    ]), failed_tickers

def p2_prepare_chart_data(df):
    """Clean financial data before plotting or regression.
    
    Remove companies with missing, infinite or non-positive metrics
    so invalid values cannot distort the visualisation or analysis.
    """

    cleaned = df.copy()
    # Convert finanical columns to numbers, replacing invalid entries with NaN
    cleaned[NUMERIC_COLS] = cleaned[NUMERIC_COLS].apply(
        pd.to_numeric, errors='coerce'
    ).replace([np.inf, -np.inf], np.nan)

    # Keep only positive values to ensure meaningful comaprison
    chart_df = cleaned.dropna(subset=NUMERIC_COLS)
    chart_df = chart_df[
        (chart_df['Capital Expenditure ($B)'] > 0) &
        (chart_df['Annual Revenue ($B)'] > 0) &
        (chart_df['Market Cap ($T)'] > 0)
    ]
    return chart_df

def p2_calculate_regression(chart_df):
    """Calculate a simple linear regression of revenue against CapEx.
    
    NumPy fits the least-squares regression line. R-squared measurES
    the proportion of variation in company revenue explained by the
    fitted line within the selected company sample.
    
    Returns the slope, intercept, R-squared and CapEx values.
    """

    # Using the displayed companies keeps the line consistent with the bubbles.
    x = chart_df['Capital Expenditure ($B)']
    y = chart_df['Annual Revenue ($B)']

    # Require at least 3 companies and variation in both variables.
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return None, None, None, x

    slope, intercept = np.polyfit(x, y, 1)
    # Calculate predicted revenue from regression
    predicted = intercept + slope * x
    # Measure total revenue variation around its mean
    ss_total = np.sum((y - y.mean()) ** 2)
    ss_residual = np.sum((y - predicted) ** 2)
    # Calculate R-squared to assess how closely line fits the data
    r_squared = 1 - ss_residual / ss_total
    return slope, intercept, r_squared, x

# Create the bubble chart
def p2_create_bubble_scatter(chart_df, size_max):
    """Create the bubble chart using company financial metrics."""

    fig = px.scatter(
        chart_df,
        x='Capital Expenditure ($B)',
        y='Annual Revenue ($B)',
        size='Market Cap ($T)',
        color='Sector',
        color_discrete_map={
            'Semiconductors': '#00D4FF',
            'Software & Cloud': '#FF6B6B',
            'Internet Services': '#00E09F',
        },
        text='Ticker',
        size_max=size_max,
        hover_name='Company',
        hover_data={
            'Ticker': False,
            'Market Cap ($T)': ':.2f',
            'Capital Expenditure ($B)': ':.2f',
            'Annual Revenue ($B)': ':.2f'
        },
        title='The AI Bubble: Capital Investment vs Revenue'
    )

    # Improve readability of bubbles and their labels
    fig.update_traces(
        marker=dict(
            opacity=1.0,
            line=dict(width=2, color='rgba(225,255,255,0.95)')
        ),
        textposition='middle center',
        selector=dict(mode='markers+text')
    )

    return fig

# Add regression line
def p2_add_regression_line(fig, slope, intercept, x):
    """Add the fitted regression line when sufficient data exists."""

    if slope is not None:
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept

        fig.add_scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            name='Regression Line',
            showlegend=False,
            hoverinfo='skip',
            line=dict(
                color='#FFD166',
                width=3,
                dash='dash'
            )
        )

    return fig

# Style chart and add titles
def p2_style_valuation_chart(fig):
    """Apply the dark theme, axes, sector legend and subtitles."""

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0B1220',
        plot_bgcolor='#0B1220',
        margin=dict(r=300, t=145, l=80, b=70),
        title=dict(
            x=.5,
            font=dict(size=24, weight='bold')
        ),
        xaxis=dict(
            title_text='<b>Capital Expenditure ($B)</b>',
            title_font=dict(family="Arial", size=16, color="white"),
            showline=True,
            linecolor='white',
            linewidth=2,
            gridcolor=GRID,
            griddash='dash',
            zeroline=False
        ),
        yaxis=dict(
            title_text='<b>Annual Revenue ($B)</b>',
            title_font=dict(family="Arial", size=16, color="white"),
            tickformat='.0f',
            showline=True,
            linecolor='white',
            linewidth=2,
            gridcolor=GRID,
            griddash='dash',
            zeroline=False,
            minor=dict(
                showgrid=False,
                ticks=''
            )
        ),
        hoverlabel=dict(
            bgcolor='#061426',
            font_color='white',
            bordercolor='#00D4FF'
        ),
        showlegend=False
    )

    # Subtitle
    fig.add_annotation(
        x=0.59, y=1.2,
        xref='paper', yref='paper',
        xanchor='center',
        text='Exploring the Relationship Between Capital Expenditure and Annual Revenue',
        showarrow=False,
        font=dict(size=15, color='white')
    )

    # Bubble-size explanation
    fig.add_annotation(
        x=0.59, y=1.15,
        xref='paper', yref='paper',
        xanchor='center',
        text='Bubble Size = Market Capitalisation ($T)',
        showarrow=False,
        font=dict(size=13, color='#00D4FF')
    )

    return fig

# Add custom sector legend
def p2_add_sector_legend(fig, chart_df):
    """Draw a cleaner custom sector legend."""

    # Background box
    fig.add_shape(
        type='rect',
        xref='paper', yref='paper',
        x0=1.02, x1=1.20,
        y0=0.77, y1=1.04,
        line=dict(color='#1E3A5F', width=2),
        fillcolor='#061426'
    )

    # Heading
    fig.add_annotation(
        x=1.11, y=0.995,
        xref='paper', yref='paper',
        xanchor='center',
        yanchor='middle',
        text='<b>Sector</b>',
        showarrow=False,
        font=dict(size=14, color='white')
    )

    sector_colours = {
        'Semiconductors': '#00D4FF',
        'Software & Cloud': '#FF6B6B',
        'Internet Services': '#00E09F',
    }

    # Only show sectors actually present
    sectors = [
        sector for sector in sector_colours
        if sector in chart_df['Sector'].values
    ]

    # Tighter, cleaner spacing
    y_positions = [0.925, 0.865, 0.805]

    for sector, y in zip(sectors, y_positions):
        # coloured circle
        fig.add_shape(
            type='circle',
            xref='paper', yref='paper',
            xsizemode='pixel',
            ysizemode='pixel',
            xanchor=1.055,
            yanchor=y,
            x0=-7, x1=7,
            y0=-7, y1=7,
            line=dict(color='white', width=2),
            fillcolor=sector_colours[sector]
        )
        # label
        fig.add_annotation(
            x=1.08, y=y,
            xref='paper', yref='paper',
            xanchor='left',
            yanchor='middle',
            text=sector,
            showarrow=False,
            font=dict(size=11, color='white')
        )

    return fig

# Add market capitalisation legend
def p2_add_market_cap_legend(fig, chart_df, size_max):
    """Draw the custom reference legend for bubble sizes."""

    # Background box
    fig.add_shape(
        type='rect',
        xref='paper', yref='paper',
        x0=1.02, x1=1.20,
        y0=0.34, y1=0.75,
        line=dict(color='#1E3A5F', width=2),
        fillcolor='#061426'
    )

    # Legend heading
    fig.add_annotation(
        x=1.11, y=0.70,
        xanchor='center',
        xref='paper', yref='paper',
        text='<b>Market Cap ($T)</b>',
        showarrow=False,
        font=dict(size=14, color='white')
    )

    # Calculate bubble sizes using Plotly's scaling method
    max_cap = chart_df['Market Cap ($T)'].max()

    for cap, y in [
        (0.5, 0.61),
        (1.0, 0.515),
        (2.0, 0.405)
    ]:
        size = size_max * 0.65 * np.sqrt(cap / max_cap)

        # Draw reference bubble
        fig.add_shape(
            type='circle',
            xref='paper', yref='paper',
            xsizemode='pixel',
            ysizemode='pixel',
            xanchor=1.06,
            yanchor=y,
            x0=-size/2, x1=size/2,
            y0=-size/2, y1=size/2,
            line=dict(color='white', width=2),
            fillcolor='#607D9F'
        )

        # Add market cap labels beside bubbles
        fig.add_annotation(
            x=1.1, y=y,
            xref='paper', yref='paper',
            text=f"${cap:.1f}T",
            showarrow=False,
            xanchor='left',
            font=dict(size=12, color='white')
        )

    return fig

# Add regression information and On/Off toggle
def p2_add_regression_controls(fig, r_squared):
    """Display R-squared and add buttons to show/hide regression."""

    # Regression information background
    fig.add_shape(
        type='rect',
        xref='paper', yref='paper',
        x0=1.02, x1=1.20,
        y0=0.01, y1=0.32,
        line=dict(color='#1E3A5F', width=2),
        fillcolor='#061426'
    )

    # Heading
    fig.add_annotation(
        x=1.11, y=0.285,
        xanchor='center',
        yanchor='middle',
        xref='paper', yref='paper',
        text='<b>Regression Analysis</b>',
        showarrow=False,
        font=dict(size=13, color='white')
    )

    # Handle unavailable regression
    if r_squared is None:
        fig.add_annotation(
            x=1.11, y=0.16,
            xref='paper', yref='paper',
            text='Regression unavailable',
            showarrow=False,
            font=dict(size=10, color='white')
        )

    else:
        # Explanation of R-squared
        fit_label = (
            f"{r_squared * 100:.1f}% of revenue variation<br>"
            "explained by CapEx"
        )

        fig.add_annotation(
            x=1.11, y=0.14,
            xanchor='center',
            xref='paper', yref='paper',
            text=fit_label,
            showarrow=False,
            font=dict(size=10, color='white')
        )

        interpretation_index = len(fig.layout.annotations) - 1

        # R-squared value
        fig.add_annotation(
            x=1.11, y=0.205,
            xanchor='center',
            xref='paper', yref='paper',
            text=f"<b>R² = {r_squared:.6f}</b>",
            showarrow=False,
            font=dict(size=15, color='#FFD166')
        )

        r2_index = len(fig.layout.annotations) - 1

        # Identify regression trace
        regression_trace_index = len(fig.data) - 1

        # On/Off controls
        fig.update_layout(
            updatemenus=[
                dict(
                    type='buttons',
                    direction='right',
                    showactive=True,
                    active=0,
                    x=1.11,
                    y=0.075,
                    xanchor='center',
                    yanchor='middle',
                    bgcolor='#607D9F',
                    bordercolor='#00D4FF',
                    borderwidth=1,
                    font=dict(size=12, color='black'),
                    pad=dict(l=2, r=2, t=2, b=2),
                    buttons=[
                        dict(
                            label='On',
                            method='update',
                            args=[
                                {'visible': [True]},
                                {
                                    f'annotations[{r2_index}].visible': True,
                                    f'annotations[{interpretation_index}].visible': True
                                },
                                [regression_trace_index]
                            ]
                        ),
                        dict(
                            label='Off',
                            method='update',
                            args=[
                                {'visible': [False]},
                                {
                                    f'annotations[{r2_index}].visible': False,
                                    f'annotations[{interpretation_index}].visible': False
                                },
                                [regression_trace_index]
                            ]
                        )
                    ]
                )
            ]
        )

    return fig

def p2_add_small_company_inset(fig, chart_df):
    """Zoomed comparison using the same financial data as the main chart."""

    tickers = ["NVDA", "AMD", "ASML", "PLTR"]
    small = chart_df[chart_df["Ticker"].isin(tickers)].copy()

    if small.empty:
        return fig

    colours = {
        "Semiconductors": "#00D4FF",
        "Software & Cloud": "#FF6B6B",
    }

    x_col = "Capital Expenditure ($B)"
    y_col = "Annual Revenue ($B)"
    cap_col = "Market Cap ($T)"

    # Inset axes and positioning
    fig.update_layout(
        xaxis2=dict(
            domain=[0.17, 0.45],
            anchor="y2",
            range=[-0.6, max(8, small[x_col].max() * 1.35)],
            title=dict(
                text="CapEx ($B)",
                font=dict(size=10),
                standoff=2
            ),
            tickfont=dict(size=10),
            gridcolor="#1E3A5F",
            showline=True,
            linecolor="white",
            zeroline=False,
            automargin=False
        ),
        yaxis2=dict(
            domain=[0.66, 0.88],
            anchor="x2",
            range=[-25, max(100, small[y_col].max() * 1.6)],
            title=dict(
                text="Revenue ($B)",
                font=dict(size=10),
                standoff=2
            ),
            tickfont=dict(size=10),
            gridcolor="#1E3A5F",
            showline=True,
            linecolor="white",
            zeroline=False,
            automargin=False
        )
    )

    # Background and turquoise border
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0.10,
        x1=0.50,
        y0=0.55,
        y1=0.99,
        fillcolor="#061426",
        line=dict(color="#00D4FF", width=2),
        layer="below"
    )

    # Title inside the inset
    fig.add_annotation(
        x=0.30,
        y=0.955,
        xref="paper",
        yref="paper",
        xanchor="center",
        yanchor="middle",
        align="center",
        text="<b>COMPANIES CLUSTER ZOOMED IN</b>",
        showarrow=False,
        font=dict(size=12, color="white")
    )

    # Scale bubble AREA by market capitalisation
    max_cap = small[cap_col].max()
    sizeref = 2 * max_cap / (46 ** 2)

    label_offsets = {
        "NVDA": (48, 0),
        "AMD": (0, -45),
        "ASML": (48, -22),
        "PLTR": (0, -25)
    }

    for _, row in small.iterrows():
        ticker = row["Ticker"]
        x = float(row[x_col])
        y = float(row[y_col])
        cap = float(row[cap_col])

        # Actual bubble, using proportional market cap
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                xaxis="x2",
                yaxis="y2",
                mode="markers",
                marker=dict(
                    size=[cap],
                    sizemode="area",
                    sizeref=sizeref,
                    sizemin=9,
                    color=colours.get(row["Sector"], "white"),
                    line=dict(color="white", width=2),
                    opacity=1
                ),
                name=ticker,
                showlegend=False,
                customdata=[[cap]],
                hovertemplate=(
                    "<b>" + ticker + "</b><br>"
                    "CapEx: $%{x:.2f}B<br>"
                    "Revenue: $%{y:.2f}B<br>"
                    "Market Cap: $%{customdata[0]:.2f}T"
                    "<extra></extra>"
                )
            )
        )

        # Separate labels with subtle connector lines
        ax, ay = label_offsets.get(ticker, (0, -25))

        fig.add_annotation(
            x=x,
            y=y,
            xref="x2",
            yref="y2",
            text="<b>" + ticker + "</b>",
            showarrow=True,
            ax=ax,
            ay=ay,
            arrowhead=0,
            arrowwidth=1,
            arrowcolor="#8CA5C0",
            bgcolor="#061426",
            borderpad=2,
            font=dict(size=11, color="white")
        )
        
        # Arrow from inset towards the cluster at the origin
        fig.add_annotation(
            x=0.18,
            y=0.40,
            xref="paper",
            yref="paper",
            ax=65,
            ay=-65,
            text="",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=2.5,
            arrowcolor="#00D4FF"
        )

    return fig

# Main function - combine all chart components
def p2_build_valuation_chart(chart_df, regression, show_inset=False):
    """Build the complete interactive valuation chart."""

    slope, intercept, r_squared, x = regression
    size_max = 75

    # Create bubble chart
    fig = p2_create_bubble_scatter(chart_df, size_max)

    # Add fitted regression line
    fig = p2_add_regression_line(fig, slope, intercept, x)

    # Apply chart styling
    fig = p2_style_valuation_chart(fig)
    
    # Add custom sector legend
    fig = p2_add_sector_legend(fig, chart_df)

    # Add market cap bubble-size legend
    fig = p2_add_market_cap_legend(fig, chart_df, size_max)

    # Add regression information and toggle
    fig = p2_add_regression_controls(fig, r_squared)

    # Only add zoom inset when requested
    if show_inset:
        fig = p2_add_small_company_inset(fig, chart_df)

    return fig

def p2_standalone_selection(search):
    """Use all companies when Panel 2 runs outside the main dashboard."""
    return None

# Integrated dashboard
def p2_create_app(selected_tickers=tickers, *, parse_selection, load_valuations=p2_load_valuation_data, **kwargs):
    app = Dash(__name__, **kwargs, include_assets_files=False)
    tab_style = {
        "backgroundColor": CARD,
        "color": "white",
        "border": "1px solid #1E3A5F",
        "padding": "12px",
    }
    selected_tab_style = {
        **tab_style,
        "backgroundColor": "#FFFFFF",
        "color": "#0B1220",
        "fontWeight": "bold",
        "borderTop": "2px solid #00D4FF",
    }
    app.layout = html.Div(
        [
            dcc.Location(id="p2-url", refresh=False),
            html.Button("Refresh Yahoo data", id="valuation-refresh", n_clicks=0),
            dcc.Tabs(
                id="view-toggle",
                value="graph",
                children=[
                    dcc.Tab(
                        label="Graph",
                        value="graph",
                        style=tab_style,
                        selected_style=selected_tab_style,
                    ),
                    dcc.Tab(
                        label="Table",
                        value="table",
                        style=tab_style,
                        selected_style=selected_tab_style,
                    ),
                ],
                style={"width": "220px", "margin": "0 auto"},
            ),
            dcc.Loading(html.Div(id="view-content")),
        ],
        style={
            "backgroundColor": "#0B1220",
            "minHeight": "100vh",
            "margin": "0",
            "padding": "8px 0",
            "color": "white",
            "fontFamily": "Arial",
        },
    )

    # Update Panel 2 when the user switches views, refreshes Yahoo
    # data or changes the universal company selection.
    @app.callback(
        Output("view-content", "children"),
        Input("view-toggle", "value"),
        Input("valuation-refresh", "n_clicks"),
        Input("p2-url", "search"),
    )
    def update_view(view, _refresh, search):
        chosen = parse_selection(search)
        chosen = list(TICKERS) if chosen is None else chosen
        if not chosen:
            return html.P("Please select at least one company in the universal selector.", role="status")
        df, failed_tickers = load_valuations(selected_tickers)
        df = df[df["Ticker"].isin(chosen)]
        chart_df = p2_prepare_chart_data(df)
        if chart_df.empty:
            return html.P(
                "Yahoo Finance returned no valid company metrics. Please retry.",
                role="alert",
            )
        omitted = df.loc[~df["Ticker"].isin(chart_df["Ticker"]), "Ticker"].tolist()
        if omitted:
            print("Omitted due to missing/invalid metrics: " + ", ".join(omitted))
        regression = p2_calculate_regression(chart_df)
        if regression[2] is None:
            print("Insufficient valid data or variation to calculate regression.")
        else:
            print(f"R-squared: {regression[2]:.6f}")
        fig = p2_build_valuation_chart(chart_df, regression, show_inset=True)
        table_df = chart_df[
            [
                "Ticker",
                "Company",
                "Sector",
                "Capital Expenditure ($B)",
                "Annual Revenue ($B)",
                "Market Cap ($T)",
            ]
        ].sort_values("Ticker")
        table_df = table_df.rename(
            columns={
                "Capital Expenditure ($B)": "CapEx ($B)",
                "Annual Revenue ($B)": "Revenue ($B)",
            }
        )
        numeric_table_cols = ["CapEx ($B)", "Revenue ($B)", "Market Cap ($T)"]
        if view == "graph":
            return html.Div(
                [
                    dcc.Graph(
                        figure=fig,
                        style={"width": "100%", "height": "calc(100vh - 30px)"},
                        config={
                            "responsive": True,
                            "displayModeBar": False,
                        },
                    ),
                ]
            )
        else:
            return html.Div(
                [
                    html.H2(
                        "The AI Bubble: Company Comparison",
                        style={"textAlign": "center", "padding": "25px"},
                    ),
                    dash_table.DataTable(
                        data=table_df.to_dict("records"),
                        columns=[
                            {
                                "name": col,
                                "id": col,
                                "type": (
                                    "numeric" if col in numeric_table_cols else "text"
                                ),
                                "format": (
                                    {"specifier": ",.2f"}
                                    if col in numeric_table_cols
                                    else {}
                                ),
                            }
                            for col in table_df.columns
                        ],
                        sort_action="native",
                        page_action="none",
                        style_table={
                            "width": "95%",
                            "margin": "auto",
                            "overflowX": "auto",
                        },
                        style_header={
                            "backgroundColor": CARD,
                            "color": "white",
                            "fontWeight": "bold",
                            "border": "1px solid #2D5474",
                            "padding": "15px",
                        },
                        style_cell={
                            "backgroundColor": CARD,
                            "color": "white",
                            "border": "1px solid #23445E",
                            "padding": "14px",
                            "textAlign": "left",
                            "fontFamily": "Arial",
                            "fontSize": "14px",
                        },
                    ),
                ]
            )

    return app

if __name__ == "__main__":
    app = p2_create_app(
        parse_selection=p2_standalone_selection
    )
    app.run(debug=False)
