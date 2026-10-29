import plotly.express as px

companies=['NVIDIA', 'Microsoft', 'Alphabet', 'Amazon','Meta']

eps_growth=[50,15,20,25,30]

forward_pe=[40,30,22,35,25]

market_cap=[4500, 3800, 3200, 2500, 1800]

def show_panel2():
    fig=px.scatter(
        x=eps_growth,
        y=forward_pe,
        size=market_cap,
        hover_name=companies,
        labels={
            'x':'Expected EPS Growth (%)',
            'y':'Forward P/E Ratio'
        }, 
        title='AI Companies: Valuation vs Earnings Growth'
    )
    fig.show()

show_panel2()