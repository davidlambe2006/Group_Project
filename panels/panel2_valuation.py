import plotly.express as px
import yfinance as yf
import pandas as pd

tickers=[
    'NVDA','MSFT','GOOGL','AMZN',
    'META','AVGO','AMD','TSM',
    'ORCL','PLTR','TSLA','ASML'
]

for ticker in tickers:
    company=yf.Ticker(ticker)
    info=company.info

    print('Company:', info.get('shortName'))
    print('Market Cap:', info.get('marketCap'))
    print('Forward P/E:', info.get('forwardPE'))
    print('------------------------')

sectors=[
    'Semiconductors',
    'Software & Cloud',
    'Software & Cloud',
    'Software & Cloud',
    'Consumer & EV'
]

def show_panel2():
    fig=px.scatter(
        x=eps_growth,
        y=forward_pe,
        size=market_cap,
        color=sectors,
        hover_name=companies,
        hover_data={
            'Market Cap ($bn)': market_cap,
            'Sector': sectors
        },
        labels={
            'x':'Expected EPS Growth (%)',
            'y':'Forward P/E Ratio'
        }, 
        title='AI Companies: Valuation vs Earnings Growth'
    )
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#061426',
        plot_bgcolor='#061426',
        font=dict(color='white'),
        title_font=dict(size=20),
        xaxis=dict(
            showgrid=True,
            gridcolor='#243449',
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#243449'
            )
    )
    fig.show()

show_panel2()