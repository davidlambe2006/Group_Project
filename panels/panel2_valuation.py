import plotly.express as px
import yfinance as yf
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
    print(ticker, info.get("marketCap"))

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


print(companies)
print(market_caps)
print(forward_pe)
print(eps_growth)

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
            opacity=0.92,
            line=dict(width=2, color='rgba(225,255,255,0.95)')
        ),
        textposition="middle center",
        textfont=dict(
            size=11, 
            color="white"
            )
    )
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#031A2F',
        plot_bgcolor='#031A2F',
        font=dict(
            color='white',
            family='Arial'),
        title=dict(
            x=.5,
            xanchor='center',
            font=dict(size=24, weight='bold'),
        ),    
        xaxis=dict(
            title_text="<b>Expected EPS Growth (%)</b>",
            showline=True,
            linecolor='white',
            linewidth=2,
            gridcolor='#243449'
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
            minor=dict(
                showgrid=False,
                ticks=''
            )
        ),
        hoverlabel=dict(
            bgcolor='#061426',
            font_color='white',
            bordercolor='#00D4FF'
        )
    )    

    fig.update_layout(legend_title_text='Sector')
    fig.show()

show_panel2()