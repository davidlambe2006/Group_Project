import yfinance as yf
import plotly.express as px
import pandas as pd

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

companies=[]
market_caps=[]
forward_pe=[]
eps_growth=[]
sectors=[]

for ticker in tickers:
    company=yf.Ticker(ticker)
    info=company.info

    companies.append(info.get('shortName'))
    market_caps.append(info.get('marketCap'))
    forward_pe.append(info.get('forwardPE'))
    sectors.append(sector_map[ticker])
    estimates = company.earnings_estimate

    if estimates is not None and '+1y' in estimates.index:
        growth=estimates.loc['+1y','growth']

        if pd.notna(growth):
            eps_growth.append(growth*100)
        else:
            eps_growth.append(None)
    else:
        eps_growth.append(None)


df=pd.DataFrame({
    'Company': companies,
    'Ticker': tickers,
    'Market Cap ($T)': [
        round(cap/1e12,2) if cap is not None else None
        for cap in market_caps
    ],    
    'Forward P/E': forward_pe,
    'Expected EPS Growth (%)': eps_growth,
    'Sector': sectors
})


def show_panel2():
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
    fig.update_traces(
        marker=dict(
            opacity=0.75,
            line=dict(width=2, color='rgba(225,255,255,0.95)')
        ),
        textposition="middle center",
        )
        

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
            title_text="<b>Expected EPS Growth (%)</b>",
            showline=True,
            linecolor='white',
            linewidth=2,
            gridcolor='#243449',
            griddash='dash',
            zeroline=False
        ),
        yaxis=dict(
            title_text="<b>Forward P/E</b>",
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
            title="Sector",
            x=1.02,
            y=1,
            bgcolor="#061426",
            bordercolor="#1E3A5F",
            borderwidth=2,
        )
    )
    # Subtitle
    fig.add_annotation(
        x=0.59, y=1.2,
        xref="paper", yref="paper",
        xanchor="center",
        text="Assessing Whether Projected Earnings Growth Supports Current Market Valuations",
        showarrow=False,
        font=dict(size=15, color="white")
        )
    # Additional information
    fig.add_annotation(
        x=0.59, y=1.15,
        xref="paper", yref="paper",
        xanchor="center",
        text="Bubble Size = Market Capitalisation ($T) | Forward P/E on Logarithmic Scale",
        showarrow=False,
        font=dict(size=13, color="#00D4FF")
        )
    # Market Cap legend box
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=1.02, x1=1.20,
        y0=0.30, y1=0.70,
        line=dict(color="#1E3A5F", width=2),
        fillcolor="#061426"
    )

    # Market Cap legend title
    fig.add_annotation(
        x=1.105, y=0.67,
        xanchor="center",
        xref="paper", yref="paper",
        text="<b>Market Cap ($T)</b>",
        showarrow=False,
        font=dict(size=14, color="white")
    )

    # Three different-sized circles
    for cap, size, y in [
        (0.5, 20, 0.59),
        (1.0, 35, 0.49),
        (2.0, 50, 0.36)
    ]:
        fig.add_shape(
            type="circle",
            xref="paper", yref="paper",
            xsizemode="pixel",
            ysizemode="pixel",
            xanchor=1.05,
            yanchor=y,
            x0=-size/2, x1=size/2,
            y0=-size/2, y1=size/2,
            line=dict(color="white", width=2),
            fillcolor="#607D9F"
        )

        fig.add_annotation(
            x=1.1, y=y,
            xref="paper", yref="paper",
            text=f"${cap:.1f}T",
            showarrow=False,
            xanchor="left",
            font=dict(size=12, color="white")
        )
    fig.show()

show_panel2()