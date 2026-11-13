import yfinance as yf
import plotly.express as px
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, dash_table, Input, Output


# Companies and sectors used for bubble colours.
tickers = [
    'NVDA', 'MSFT', 'GOOGL', 'AMZN',
    'META', 'AVGO', 'AMD', 'TSM',
    'ORCL', 'PLTR', 'TSLA', 'ASML'
]

sector_map = {
    "NVDA": "Semiconductors",
    "MSFT": "Software & Cloud",
    "GOOGL": "Software & Cloud",
    "AMZN": "Software & Cloud",
    "META": "Internet Services",
    "AVGO": "Semiconductors",
    "AMD": "Semiconductors",
    "TSM": "Semiconductors",
    "ORCL": "Software & Cloud",
    "PLTR": "Software & Cloud",
    "TSLA": "Consumer & EV",
    "ASML": "Semiconductors"
}


NUMERIC_COLS = ['Market Cap ($T)', 'Capital Expenditure ($B)', 'Annual Revenue ($B)']


def load_valuation_data(selected_tickers):
    """Download Yahoo Finance financial metrics for the requested companies.

    A failed ticker is recorded with missing values so other tickers still load.
    """
    rows = []
    failed_tickers = []

    for ticker in selected_tickers:
        name, cap, capex, revenue = ticker, None, None, None
        try:
            company = yf.Ticker(ticker)
            info = company.info
            name = info.get('displayName') or info.get('shortName') or ticker
            cap = info.get('marketCap')
            income = company.income_stmt
            cashflow = company.cashflow

            if not income.empty and not cashflow.empty:
                if 'Total Revenue' in income.index and 'Capital Expenditure' in cashflow.index:
                    common_years = income.columns.intersection(cashflow.columns)
                    for year in sorted(common_years, reverse=True):
                        rev = income.loc['Total Revenue', year]
                        expenditure = cashflow.loc['Capital Expenditure', year]
                        if pd.notna(rev) and pd.notna(expenditure):
                             # Get currencies for financial statements and market cap
                            currency = info.get('financialCurrency')
                            market_currency = info.get('currency')
                            
                            if not currency:
                                 raise ValueError(f"Financial currency unavailable for {ticker}")
                            
                            if cap is not None and not market_currency:
                                 raise ValueError(f"Market cap currency unavailable for {ticker}")
                            
                            # Convert revenue and CapEx using historical exchange rates
                            fx_rate = 1.0
                            if currency != 'USD':
                                 fx_ticker = f"{currency}USD=X"
                                 fx_data = yf.Ticker(fx_ticker).history(
                                      start=year - pd.DateOffset(years=1),
                                      end=year + pd.Timedelta(days=1)
                                )
                                 
                                 if fx_data.empty or fx_data['Close'].dropna().empty:
                                     raise ValueError(f"No exchange rate available for {currency}")
                                 
                                 fx_rate = fx_data['Close'].dropna().mean()

                                 if not np.isfinite(fx_rate) or fx_rate <= 0:
                                     raise ValueError(f"Invalid exchange rate for {currency}")
                                 
                            revenue = rev * fx_rate / 1e9
                            capex = abs(expenditure) * fx_rate / 1e9
                            
                            # Convert market cap using a recent exchange rate
                            if cap is not None and market_currency != 'USD':
                                fx_ticker = f"{market_currency}USD=X"
                                fx_data = yf.Ticker(fx_ticker).history(period='5d')
                                
                                if fx_data.empty or fx_data['Close'].dropna().empty:
                                    raise ValueError(f"No market cap exchange rate for {market_currency}")
                                
                                fx_rate_market = fx_data['Close'].dropna().iloc[-1]
                                if not np.isfinite(fx_rate_market) or fx_rate_market <= 0:
                                    raise ValueError(f"Invalid market cap exchange rate for {market_currency}")

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

def prepare_chart_data(df):
    """Remove missing/invalid bubble inputs."""

    cleaned = df.copy()
    cleaned[NUMERIC_COLS] = cleaned[NUMERIC_COLS].apply(
        pd.to_numeric, errors='coerce'
    ).replace([np.inf, -np.inf], np.nan)

    chart_df = cleaned.dropna(subset=NUMERIC_COLS)
    chart_df = chart_df[
        (chart_df['Capital Expenditure ($B)'] > 0) &
        (chart_df['Annual Revenue ($B)'] > 0) &
        (chart_df['Market Cap ($T)'] > 0)
    ]
    return chart_df


def calculate_regression(chart_df):
    """Fit annual revenue against capital expenditure; return fit statistics."""
    # Using the displayed companies keeps the line consistent with the bubbles.
    x = chart_df['Capital Expenditure ($B)']
    y = chart_df['Annual Revenue ($B)']

    # Require at least 3 companies and variation in both variables.
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return None, None, None, x

    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    ss_total = np.sum((y - y.mean()) ** 2)
    ss_residual = np.sum((y - predicted) ** 2)
    r_squared = 1 - ss_residual / ss_total
    return slope, intercept, r_squared, x

# Create the bubble chart
def create_bubble_scatter(chart_df, size_max):
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
            'Consumer & EV': '#B477FF'
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
def add_regression_line(fig, slope, intercept, x):
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
def style_valuation_chart(fig):
    """Apply the dark theme, axes, sector legend and subtitles."""

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#031A2F',
        plot_bgcolor='#031A2F',
        margin=dict(r=250, t=170),
        title=dict(
            x=.5,
            font=dict(size=24, weight='bold')
        ),
        xaxis=dict(
            title_text='<b>Capital Expenditure ($B)</b>',
            showline=True,
            linecolor='white',
            linewidth=2,
            gridcolor='#243449',
            griddash='dash',
            zeroline=False
        ),
        yaxis=dict(
            title_text='<b>Annual Revenue ($B)</b>',
            tickformat='.0f',
            showline=True,
            linecolor='white',
            linewidth=2,
            gridcolor='#243449',
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
        legend=dict(
            title='<b>Sector</b>',
            x=1.02,
            y=1.08,
            bgcolor='#061426',
            bordercolor='#1E3A5F',
            borderwidth=2
        )
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

# Add market capitalisation legend
def add_market_cap_legend(fig, chart_df, size_max):
    """Draw the custom reference legend for bubble sizes."""

    # Background box
    fig.add_shape(
        type='rect',
        xref='paper', yref='paper',
        x0=1.02, x1=1.20,
        y0=0.32, y1=0.75,
        line=dict(color='#1E3A5F', width=2),
        fillcolor='#061426'
    )

    # Legend heading
    fig.add_annotation(
        x=1.105, y=0.725,
        xanchor='center',
        xref='paper', yref='paper',
        text='<b>Market Cap ($T)</b>',
        showarrow=False,
        font=dict(size=14, color='white')
    )

    # Calculate bubble sizes using Plotly's scaling method
    max_cap = chart_df['Market Cap ($T)'].max()

    for cap, y in [
        (0.5, 0.63),
        (1.0, 0.525),
        (2.0, 0.395)
    ]:
        size = size_max * np.sqrt(cap / max_cap)

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
def add_regression_controls(fig, r_squared):
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

# Main function - combine all chart components
def build_valuation_chart(chart_df, regression):
    """Build the complete interactive valuation chart."""

    slope, intercept, r_squared, x = regression
    size_max = 75

    # Create bubble chart
    fig = create_bubble_scatter(chart_df, size_max)

    # Add fitted regression line
    fig = add_regression_line(fig, slope, intercept, x)

    # Apply chart styling
    fig = style_valuation_chart(fig)

    # Add market cap bubble-size legend
    fig = add_market_cap_legend(fig, chart_df, size_max)

    # Add regression information and toggle
    fig = add_regression_controls(fig, r_squared)

    return fig


    
def show_panel2(selected_tickers=tickers):
    """Run the valuation panel as a standalone Plotly chart."""
    df, failed_tickers = load_valuation_data(selected_tickers)
    chart_df = prepare_chart_data(df)

    if chart_df.empty:
        print('No valid company data is available for the bubble chart.')
        if failed_tickers:
            print('Downloads failed for: ' + ', '.join(failed_tickers))
        return

    omitted = df.loc[~df['Ticker'].isin(chart_df['Ticker']), 'Ticker'].tolist()
    if omitted:
        print('Omitted due to missing/invalid metrics: ' + ', '.join(omitted))

    regression = calculate_regression(chart_df)
    if regression[2] is None:
        print('Insufficient valid data or variation to calculate regression.')
    else:
        print(f'R-squared: {regression[2]:.6f}')
    
    # Both views use the same selected companies.
    fig = build_valuation_chart(chart_df, regression)

    table_df = chart_df[[
        'Ticker',
        'Company',
        'Sector',
        'Capital Expenditure ($B)',
        'Annual Revenue ($B)',
        'Market Cap ($T)'
    ]].sort_values('Ticker')

    # Format the financial figures
    table_df = table_df.rename(columns={
        'Capital Expenditure ($B)': 'CapEx ($B)',
        'Annual Revenue ($B)': 'Revenue ($B)'
    })
    numeric_table_cols = [
        'CapEx ($B)',
        'Revenue ($B)',
        'Market Cap ($T)'
    ]

    # Create the Dash app
    app = Dash(__name__)

    tab_style = {
        'backgroundColor': '#102D47',
        'color': 'white',
        'border': '1px solid #1E3A5F',
        'padding': '12px'
    }

    selected_tab_style = {
        **tab_style,
        'backgroundColor': '#FFFFFF',
        'color': '#031A2F',
        'fontWeight': 'bold',
        'borderTop': '2px solid #00D4FF'
    }

    app.layout = html.Div(
        [
            # Graph / Table toggle
            dcc.Tabs(
                id='view-toggle',
                value='graph',
                children=[
                    dcc.Tab(
                        label='Graph',
                        value='graph',
                        style=tab_style,
                        selected_style=selected_tab_style
                    ),
                    dcc.Tab(
                        label='Table',
                        value='table',
                        style=tab_style,
                        selected_style=selected_tab_style
                    )
                ],
                style={
                    'width': '220px',
                    'margin': '0 auto'
                }
            ),

            # This area changes when the toggle is clicked
            html.Div(id='view-content')
        ],
        style={
            'backgroundColor': '#031A2F',
            'minHeight': '100vh',
            'margin': '0',
            'padding': '8px 0',
            'color': 'white',
            'fontFamily': 'Arial'
        }
    )

    # Switch between the original graph and the table
    @app.callback(
        Output('view-content', 'children'),
        Input('view-toggle', 'value')
    )
    def switch_view(view):

        if view == 'graph':
            return dcc.Graph(
                figure=fig,
                style={
                    'width': '100%',
                    'height': 'calc(100vh - 65px)'
                },
                config={'responsive': True}
            )

        else:
            return html.Div(
                [
                    html.H2(
                        'The AI Bubble: Company Comparison',
                        style={
                            'textAlign': 'center',
                            'padding': '25px'
                        }
                    ),

                    dash_table.DataTable(
                        data=table_df.to_dict('records'),
                        columns=[
                            {
                                'name': col,
                                'id': col,
                                'type': 'numeric' if col in numeric_table_cols else 'text',
                                'format': {'specifier': ',.2f'} if col in numeric_table_cols else {}
                            }
                            for col in table_df.columns
                        ],
                    
                        sort_action='native',
                        page_action='none',

                        style_table={
                            'width': '95%',
                            'margin': 'auto',
                            'overflowX': 'auto'
                        },

                        style_header={
                            'backgroundColor': '#163954',
                            'color': 'white',
                            'fontWeight': 'bold',
                            'border': '1px solid #2D5474',
                            'padding': '15px'
                        },

                        style_cell={
                            'backgroundColor': '#0B2943',
                            'color': 'white',
                            'border': '1px solid #23445E',
                            'padding': '14px',
                            'textAlign': 'left',
                            'fontFamily': 'Arial',
                            'fontSize': '14px'
                        },

                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': '#102F49'
                            }
                        ]
                    )
                ]
            )

    # Start the dashboard
    app.run(debug=False)


if __name__ == '__main__':
    show_panel2()