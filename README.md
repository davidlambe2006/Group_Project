# AI Boom Dashboard

An interactive Python dashboard exploring AI-related stock performance, corporate investment, historical market comparisons, and hypothetical future valuations. It runs on your computer and opens in your browser.

| Panel | Question explored | Main output |
|---|---|---|
| 1 — AI Stock Performance | How have selected companies performed against market benchmarks? | 3D performance chart and risk statistics |
| 2 — Capital Investment vs Revenue | How does company investment relate to revenue? | Bubble chart, regression, and financial table |
| 3 — AI Boom vs Dot-Com Bubble | How does the AI portfolio compare with the dot-com period? | Historical comparison and illustrative scenarios |
| 4 — AI Valuation Simulator | What future price follows from selected financial assumptions? | Price estimates, scenario comparison, and sensitivity heatmap |

**Dashboard overview:** [Insert a screenshot showing the four panels and company selector. Record the capture date.]

## 1. Opening the dashboard

### What you need

- Python installed. The original project documentation identifies **Python 3.14** as the development version.
- An internet connection to retrieve financial data.
- A modern browser, such as Chrome, Edge, or Firefox.
- The extracted project folder. VS Code is optional.

If Python is not installed, download it from [python.org](https://www.python.org/downloads/). When installing on Windows, select **Add Python to PATH** if offered.

### First launch — Windows

1. Right-click the ZIP and select **Extract All**.
2. Open the extracted folders until you reach **Group_Project**. You should see `app.py` and the `panels` folder.
3. Right-click an empty space inside that folder and select **Open in Terminal**. Alternatively, open the folder in VS Code and choose **Terminal → New Terminal**.
4. Run each command below separately. Wait for each to finish before entering the next.

Check that Python is available:

```powershell
python --version
```

Install the packages used by the supplied application:

```powershell
python -m pip install dash plotly yfinance pandas numpy flask
```

Start the dashboard:

```powershell
python app.py
```

5. Open **[http://127.0.0.1:8050](http://127.0.0.1:8050)** in your browser.

The application prints the local address in the terminal. Allow time for the charts to retrieve financial data. **Keep the terminal open while using the dashboard.**

**Startup screenshot:** [Insert the terminal showing successful startup and the local address.]

### Stopping and opening again

- **Stop:** click the terminal and press **Ctrl+C**.
- **Open again:** open a terminal in `Group_Project`, run `python app.py`, and visit the same address. You do not need to repeat installation.

## 2. First walkthrough

1. **Choose companies:** click the company buttons at the top. Highlighted companies are included. **Select All** and **Deselect All** change the whole selection.
2. **Open a panel:** click one of the four overview charts.
3. **Explore:** hover over the chart for values and use the expanded panel's controls.
4. **Return:** use the back control to return to the overview.

The top company selector controls **Panels 1–3**. **Panel 4 has its own company selector.** Selecting no companies displays a prompt in Panels 1–3.

| Company | Ticker | Company | Ticker |
|---|---|---|---|
| NVIDIA | NVDA | AMD | AMD |
| Microsoft | MSFT | TSMC | TSM |
| Alphabet | GOOGL | Oracle | ORCL |
| Amazon | AMZN | Palantir | PLTR |
| Meta | META | ASML | ASML |

## 3. Panel guides

### Panel 1 — AI Stock Performance

Select companies in the header, then choose a **1, 3, 5, or 10-year** period. Add **SPY** (S&P 500 ETF) or **QQQ** (Nasdaq-100 ETF) as benchmarks. Chart detail options are **Daily**, **Every 5 Days**, and **Every 20 Days**.

Drag the 3D chart to rotate it and hover over lines for values.

- **Indexed price:** starts at 100, allowing comparisons across different share prices. A value of 150 represents a 50% increase from the starting value.
- **Total return:** percentage change over the analysed period.
- **Annualised volatility:** variation in daily returns, expressed on an annual basis.
- **Maximum drawdown:** the largest percentage fall from an earlier peak during the period.

**Screenshot:** [Show the controls, chart, and statistics. State the selected companies, period, and benchmarks.]

### Panel 2 — Capital Investment vs Revenue

Select companies, hover over bubbles for details, and switch to the table to inspect the underlying figures.

- **X-axis:** capital expenditure—spending on long-term assets, in USD billions.
- **Y-axis:** annual revenue—sales before expenses, in USD billions.
- **Bubble size:** market capitalisation—the market value of the company's equity.
- **Regression line:** the fitted linear relationship between spending and revenue.
- **R²:** the proportion of variation in revenue explained by that fit within the selected sample.

The relationship does not establish that spending caused the revenue. Company reporting periods and business models may differ.

**Screenshots:** [Show the bubble chart with readable axes and regression information, followed by the financial table.]

### Panel 3 — AI Boom vs Dot-Com Bubble

Use the shared company selector to compare the AI portfolio **from 2023 onwards** with the **NASDAQ Composite during 1995–2003**. Inspect indexed performance, returns, and maximum drawdowns.

The Bull, Base, and Bear paths are illustrative scenarios, not validated predictions. Similar historical patterns do not establish that the same outcomes will repeat.

**Screenshot:** [Show the historical comparison, metrics, and scenario labels. State the companies selected.]

### Panel 4 — AI Valuation Simulator

Estimate a hypothetical future share price from assumptions about growth, profitability, and valuation.

#### How to use it

1. Choose a company using this panel's selector.
2. Review the financial data and source dates.
3. Select **Bear**, **Base**, or **Bull**.
4. Compare the estimated future price and annualised return.
5. Hover over the heatmap to explore growth and profit-margin combinations.
6. Expand **Compare scenarios** to view the comparison table.

| Scenario | Annual revenue growth | Net profit margin | Future P/E | Years |
|---|---:|---:|---:|---:|
| Bear | 10% | 20% | 30× | 5 |
| Base | 20% | 25% | 30× | 5 |
| Bull | 30% | 30% | 30× | 5 |

These presets are illustrative assumptions, not company-specific forecasts.

#### Financial basis

The simulator uses an **earnings-multiple model**: grow revenue, estimate profit, calculate earnings per share (EPS), then apply a future price-to-earnings ratio (P/E).

```text
Future revenue = current annual revenue × (1 + growth)^years
Future EPS = future revenue × net profit margin ÷ equivalent shares
Future price = future EPS × future P/E
Annualised return = (future price ÷ current price)^(1 ÷ years) − 1
```

Growth and margin are decimals in these formulas: 20% is `0.20`.

A **25% net profit margin** means $0.25 profit per $1 of revenue. A **30× P/E** means a price equal to 30 times annual earnings per share.

**Example:** $50 billion revenue growing at 20% annually for five years becomes approximately $124.4 billion. At a 25% margin, 5 billion equivalent shares, and a 30× P/E, the estimated future price is **$186.62**.

Annualised return expresses the constant yearly price growth needed to reach the estimate. The heatmap repeats the valuation across growth and margin combinations, keeping the selected scenario's P/E and horizon.

**Limits:** this estimates a future price under assumptions; it does not discount future cash flows to estimate today's intrinsic value. Growth, margin, shares, P/E, and FX are held constant. Dividends, fees, and taxes are excluded.

For American Depositary Share (ADS) listings, the price and equivalent share count must use consistent listing units. **Refresh Yahoo data** restores retrieved figures.

**Screenshots:** [Show company, scenario, assumptions, results, and heatmap. Add the expanded manual-input controls with units visible.]

## 4. Data and implementation

Financial data is retrieved from **Yahoo Finance through yfinance**. The application combines historical prices, financial statements, portfolio calculations, linear regression, and valuation modelling.

- Internet access is required for retrieval; market data is not bundled with the application.
- Missing source data can prevent charts or calculations from loading.
- Selected data requests are cached in memory for **15 minutes**.
- Financial values change between runs; screenshots are dated examples.
- Historical data and hypothetical scenarios should be interpreted separately.

**Architecture:** `app.py` combines the panel modules and manages navigation. Dash updates outputs when controls change; Plotly draws the charts; pandas and NumPy process data; Flask provides the local web server.

```text
Group_Project/
├── app.py
├── README.md
├── requirements.txt
└── panels/
    ├── dashboard_shared.py
    ├── panel1_stock.py
    ├── panel2_valuation.py
    ├── panel3_dotcom.py
    └── panel4_simulator.py
```

## 5. Troubleshooting

| Problem | What to do |
|---|---|
| `python` is not recognised | Try `py --version`. If it works, replace `python` with `py` in the commands. Otherwise install Python and reopen the terminal. |
| `app.py` cannot be found | Open the terminal in `Group_Project`, where `app.py` is visible. |
| A package is missing | Repeat the installation command in Section 1 using the same Python installation used to launch the app. |
| `panels` cannot be imported | Keep `app.py` and the complete `panels/` folder together. |
| Browser cannot connect | Confirm the app is still running and use the address displayed in the terminal. |
| Port 8050 is occupied | Run `python app.py --port 8051`, then open `http://127.0.0.1:8051`. |
| Financial data is unavailable | Check internet access and refresh where available. |

## 6. References and attribution

- [Yahoo Finance](https://finance.yahoo.com/) — financial data.
- [yfinance](https://github.com/ranaroussi/yfinance) — data retrieval.
- [Python](https://docs.python.org/3/), [Dash](https://dash.plotly.com/), [Plotly](https://plotly.com/python/), [pandas](https://pandas.pydata.org/docs/), [NumPy](https://numpy.org/doc/), and [Flask](https://flask.palletsprojects.com/) — application libraries and documentation.
- **External code, resources, and assistance:** [identify sources, their use, and any required declarations under module rules].
