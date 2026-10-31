import yfinance as yf
import plotly.express as px
import pandas as pd
import numpy as np

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


NUMERIC_COLS = ['Market Cap ($T)', 'Forward P/E', 'Expected EPS Growth (%)']


def load_valuation_data(selected_tickers):
    """Download Yahoo Finance valuation metrics for the requested companies.

    A failed ticker is recorded with missing values so other tickers still load.
    """
    rows = []
    failed_tickers = []

    for ticker in selected_tickers:
        name, cap, pe, growth_pct = ticker, None, None, None
        try:
            company = yf.Ticker(ticker)
            info = company.info
            name = info.get('displayName') or info.get('shortName') or ticker
            cap = info.get('marketCap')
            pe = info.get('forwardPE')

            # Earnings-estimate growth is a fraction; convert to percentage.
            estimates = company.earnings_estimate
            if estimates is not None and '+1y' in estimates.index:
                growth = estimates.loc['+1y', 'growth']
                if pd.notna(growth):
                    growth_pct = growth * 100
        except Exception as error:
            failed_tickers.append(ticker)
            print(f"Error retrieving data for {ticker}: {error}")

        # One dictionary per ticker avoids misaligned separate data lists.
        rows.append({
            'Company': name,
            'Ticker': ticker,
            'Market Cap ($T)': cap / 1e12 if cap is not None else None,
            'Forward P/E': pe,
            'Expected EPS Growth (%)': growth_pct,
            'Sector': sector_map.get(ticker, 'Other'),
        })

    return pd.DataFrame(rows, columns=[
        'Company', 'Ticker', *NUMERIC_COLS, 'Sector'
    ]), failed_tickers


def prepare_chart_data(df):
    """Remove missing/invalid bubble inputs; negative EPS growth is allowed."""
    cleaned = df.copy()
    cleaned[NUMERIC_COLS] = cleaned[NUMERIC_COLS].apply(
        pd.to_numeric, errors='coerce'
    ).replace([np.inf, -np.inf], np.nan)

    chart_df = cleaned.dropna(subset=NUMERIC_COLS).copy()
    chart_df = chart_df[
        (chart_df['Forward P/E'] > 0) &
        (chart_df['Market Cap ($T)'] > 0)
    ].copy()
    return chart_df


def calculate_regression(chart_df):
    """Fit log(forward P/E) against expected EPS growth; return fit statistics."""
    # Using the displayed companies keeps the line consistent with the bubbles.
    x = chart_df['Expected EPS Growth (%)']
    y = np.log(chart_df['Forward P/E'])

    # A slope needs at least two distinct X-values; R² needs variation in Y.
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return None, None, None, x

    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    ss_total = np.sum((y - y.mean()) ** 2)
    ss_residual = np.sum((y - predicted) ** 2)
    r_squared = 1 - ss_residual / ss_total
    return slope, intercept, r_squared, x


def build_valuation_chart(chart_df, regression):
    """Build the interactive bubble chart with its legends and regression toggle."""
    slope, intercept, r_squared, x = regression

    #Create scatter plot using company financial metrics 
    fig=px.scatter(
        chart_df,
        x='Expected EPS Growth (%)',
        y='Forward P/E',
        size='Market Cap ($T)',
        color='Sector',
        color_discrete_map={
            'Semiconductors':'#00D4FF',
            'Software & Cloud':'#FF6B6B',
            'Internet Services':'#00E09F',
            'Consumer & EV':'#B477FF'
        },
        text='Ticker',
        size_max=75,
        hover_name='Company',
        hover_data={
            'Ticker':False,
            'Market Cap ($T)':':.2f',
            'Expected EPS Growth (%)':':.2f',
            'Forward P/E':':.2f'    
        },
       
        title='AI Companies: Valuation vs Earnings Growth'
    )

    #Calculate values for regression line
    if slope is not None:
        x_line=np.linspace(x.min(), x.max(), 100)
        y_line=np.exp(slope*x_line + intercept)
        
        #Add regression line to the scatter plot
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


# Improve readability of bubbles and their ticker labels.
    fig.update_traces(
        marker=dict(
            opacity=0.75,
            line=dict(width=2, color='rgba(225,255,255,0.95)')
        ),
        textposition="middle center",
        selector=dict(mode='markers+text')
        )

    #Apply dark theme and customize layout, axes, hover labels, and legend
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#031A2F',
        plot_bgcolor='#031A2F',
        margin=dict(r=250, t=170),
        title=dict(
            x=.5,
            font=dict(size=24, weight='bold'),
        ),    
        xaxis=dict(
            title_text='<b>Expected EPS Growth (%)</b>',
            showline=True,
            linecolor='white',
            linewidth=2,
            gridcolor='#243449',
            griddash='dash',
            zeroline=False
        ),
        #Use a logarithmic scale to display a wide range of P/E ratios
        yaxis=dict(
            title_text='<b>Forward P/E</b>',
            type='log',
            dtick='D2',
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
            y=1.00,
            bgcolor='#061426',
            bordercolor='#1E3A5F',
            borderwidth=2,
        )
    )
    # Adding Subtitle
    fig.add_annotation(
        x=0.59, y=1.2,
        xref='paper', yref='paper',
        xanchor='center',
        text='Exploring the Relationship Between Expected Earnings Growth and Equity Valuations',
        showarrow=False,
        font=dict(size=15, color='white')
        )
    # Clarify what bubble size and the logarithmic axis represent.
    fig.add_annotation(
        x=0.59, y=1.15,
        xref='paper', yref='paper',
        xanchor='center',
        text='Bubble Size = Market Capitalisation ($T) | Forward P/E on Logarithmic Scale',
        showarrow=False,
        font=dict(size=13, color='#00D4FF')
        )
    
    # Draw a custom reference legend to explain market-cap bubble sizes.
    fig.add_shape(
        type='rect',
        xref='paper', yref='paper',
        x0=1.02, x1=1.20,
        y0=0.31, y1=0.70,
        line=dict(color='#1E3A5F', width=2),
        fillcolor='#061426'
    )

    # Market Cap legend title
    fig.add_annotation(
        x=1.105, y=0.685,
        xanchor='center',
        xref='paper', yref='paper',
        text='<b>Market Cap ($T)</b>',
        showarrow=False,
        font=dict(size=14, color='white')
    )


  # Show three illustrative bubble sizes in the market-cap legend.
    for cap, size, y in [
        (0.5, 20, 0.61),
        (1.0, 35, 0.51),
        (2.0, 50, 0.385)
    ]:
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

        #Add market cap labels beside each bubble
        fig.add_annotation(
            x=1.1, y=y,
            xref='paper', yref='paper',
            text=f"${cap:.1f}T",
            showarrow=False,
            xanchor='left',
            font=dict(size=12, color='white')
        )

    # Reserve an information panel for regression fit and its visibility toggle.
    fig.add_shape(
        type='rect',
        xref='paper', yref='paper',
        x0=1.02, x1=1.20,
        y0=0.01, y1=0.29,
        line=dict(color='#1E3A5F', width=2),
        fillcolor='#061426'
    )

    # Label the regression information panel.
    fig.add_annotation(
        x=1.11, y=0.255,
        xanchor='center',
        yanchor='middle',
        xref='paper', yref='paper',
        text='<b>Regression Analysis</b>',
        showarrow=False,
        font=dict(size=13, color='white')
    )

    # When there is no valid regression, show an explanation rather than 
    # trying to format None as an R² number or toggle a missing line.
    if r_squared is None:
        fig.add_annotation(
            x=1.11, y=0.16, xref='paper', yref='paper',
            text='Regression unavailable', showarrow=False,
            font=dict(size=11, color='white')
        )
    else:
        # Interpret fit strength using R² (descriptive only; not causation).
        if r_squared < 0.1:
            fit_label = 'Very weak fit'
        elif r_squared < 0.3:
            fit_label = 'Weak fit'
        elif r_squared < 0.5:
            fit_label = 'Moderate fit'
        else:
            fit_label = 'Stronger fit'

        fig.add_annotation(
            x=1.11, y=0.14, xanchor='center',
            xref='paper', yref='paper', text=fit_label,
            showarrow=False, font=dict(size=11, color='white')
        )
        interpretation_index = len(fig.layout.annotations) - 1

        fig.add_annotation(
            x=1.11, y=0.185, xanchor='center',
            xref='paper', yref='paper',
            text=f"<b>R² = {r_squared:.6f}</b>",
            showarrow=False, font=dict(size=15, color='#FFD166')
        )
        r2_index = len(fig.layout.annotations) - 1

        # Toggle only the regression trace and its two annotations, not bubbles.
        regression_trace_index = len(fig.data) - 1
        fig.update_layout(
            updatemenus=[dict(
            type='buttons', 
            direction='right', 
            showactive=True, 
            active=0,
            x=1.11, y=0.09, 
            xanchor='center', yanchor='middle',
            bgcolor='#607D9F', bordercolor='#00D4FF', borderwidth=1,
            font=dict(size=12, color='#061426'),
            pad=dict(l=2, r=2, t=2, b=2),
            buttons=[
                dict(label='On', method='update', args=[
                    {'visible': [True]},
                    {f'annotations[{r2_index}].visible': True,
                     f'annotations[{interpretation_index}].visible': True},
                    [regression_trace_index]
                ]),
                dict(label='Off', method='update', args=[
                    {'visible': [False]},
                    {f'annotations[{r2_index}].visible': False,
                     f'annotations[{interpretation_index}].visible': False},
                    [regression_trace_index]
                ])
            ]
        )])

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

    fig = build_valuation_chart(chart_df, regression)
    fig.show()


if __name__ == '__main__':
    show_panel2()
