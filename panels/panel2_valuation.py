import yfinance as yf
import plotly.express as px
import pandas as pd
import numpy as np

# Define the list of AI-related company tickers and their corresponding sectors
tickers=[
    'NVDA','MSFT','GOOGL','AMZN',
    'META','AVGO','AMD','TSM',
    'ORCL','PLTR','TSLA','ASML'
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

# Initialize lists to store company data
companies=[]
market_caps=[]
forward_pe=[]
eps_growth=[]
sectors=[]

#Retrieve data for each ticker using yfinance
for ticker in tickers:
    company=yf.Ticker(ticker)
    info=company.info

    #Use a fallback for company name if displayName is not available
    companies.append(
        info.get('displayName') or info.get('shortName') or ticker
        )
    market_caps.append(info.get('marketCap'))
    forward_pe.append(info.get('forwardPE'))
    sectors.append(sector_map[ticker])

    #Retrieved next years estimated EPS growth
    estimates = company.earnings_estimate

    if estimates is not None and '+1y' in estimates.index:
        growth=estimates.loc['+1y','growth']

        #Convert growth to a percetage if data is available
        if pd.notna(growth):
            eps_growth.append(growth*100)
        else:
            eps_growth.append(None)
    else:
        eps_growth.append(None)

# Create a DataFrame to hold the collected data
df=pd.DataFrame({
    'Company': companies,
    'Ticker': tickers,
    'Market Cap ($T)': [
        #Convert Market Cap from dollars to trillions 
        round(cap/1e12,2) if cap is not None else None
        for cap in market_caps
    ],    
    'Forward P/E': forward_pe,
    'Expected EPS Growth (%)': eps_growth,
    'Sector': sectors
})

#Calculate regression line between EPS growth and log forward P/E
regression_df=df.dropna(
    subset=['Expected EPS Growth (%)','Forward P/E']
)
regression_df=regression_df[
    regression_df['Forward P/E']>0
]
x=regression_df['Expected EPS Growth (%)']
y=np.log(regression_df['Forward P/E'])

slope, intercept=np.polyfit(x,y,1)

#Calculate R-squared
y_pred=intercept + slope*x

ss_res=np.sum((y - y_pred)**2)
ss_tot=np.sum((y - np.mean(y))**2)
r_squared=1 - (ss_res/ss_tot)
print(f"R-squared: {r_squared:.6f}")

def show_panel2():
    """
    Creates an interactive scatter plot comparing AI companies
    based on expected EPS growth and forward P/E ratios.

    Bubble sizes represent market capitalisation, whilecolors 
    indicate sectors. The P/E axis uses a logarithmic scale.
    """
    #Create scatter plot using company financial metrics 
    fig=px.scatter(
        df,
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

    #Adjust buuble transparency, borders and ticker positions
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
    # Subtitle
    fig.add_annotation(
        x=0.59, y=1.2,
        xref='paper', yref='paper',
        xanchor='center',
        text='Exploring the Relationship Between Expected Earnings Growth and Equity Valuations',
        showarrow=False,
        font=dict(size=15, color='white')
        )
    # Additional information
    fig.add_annotation(
        x=0.59, y=1.15,
        xref='paper', yref='paper',
        xanchor='center',
        text='Bubble Size = Market Capitalisation ($T) | Forward P/E on Logarithmic Scale',
        showarrow=False,
        font=dict(size=13, color='#00D4FF')
        )
    # Market Cap legend box
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

    # Create a custom legend showing market cap through bubble sizes
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

    #Regression analysis box
    fig.add_shape(
        type='rect',
        xref='paper', yref='paper',
        x0=1.02, x1=1.20,
        y0=0.01, y1=0.29,
        line=dict(color='#1E3A5F', width=2),
        fillcolor='#061426'
    )

    #Regression box title
    fig.add_annotation(
        x=1.11, y=0.255,
        xanchor='center',
        yanchor='middle',
        xref='paper', yref='paper',
        text='<b>Regression Analysis</b>',
        showarrow=False,
        font=dict(size=13, color='white')
    )

    #Interpret the regression fit
    fig.add_annotation(
        x=1.11, y=0.14,
        xanchor='center',
        xref='paper', yref='paper',
        text='Very Weak Relationship',
        showarrow=False,
        font=dict(size=11, color='white')
    )

    #Interpreation annotion
    interpretation_index=len(fig.layout.annotations)-1

    #Store the position of the R-squared annotation
    r2_index=len(fig.layout.annotations)  
    
    # Store the index of the R-squared annotation
    fig.add_annotation(
        x=1.11, y=0.185,
        xanchor='center',
        xref='paper', yref='paper',
        text=f"<b>R² = {r_squared:.6f}</b>",
        showarrow=False,
        font=dict(size=15, color='#FFD166')
    )
    #Identify the regression line
    regression_trace_index=len(fig.data)-1

    #Toggle regression analysis on and off
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                direction='right',
                showactive=True,
                active=0,
                x=1.11,
                y=0.09,
                xanchor='center',
                yanchor='middle',
                bgcolor='#607D9F',
                bordercolor='#00D4FF',
                borderwidth=1,
                font=dict(size=12, color='#061426'),
                pad=dict(l=2,r=2,t=2,b=2),
                buttons=[
                    dict(
                        label='On',
                        method='update',
                        args=[
                            {'visible':True},
                            {f'annotations[{r2_index}].visible':True,
                             f'annotations[{interpretation_index}].visible':True},
                            [regression_trace_index]
                        ]
                    ),    
                    dict(
                        label='Off',
                        method='update',
                        args=[
                            {'visible':False},
                            {f'annotations[{r2_index}].visible':False,
                             f'annotations[{interpretation_index}].visible':False},
                            [regression_trace_index]
                        ]
                )
            ]
        )
    ]
)

    #Display completed interactive chart    
    fig.show()

show_panel2()