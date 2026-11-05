# Panel 4

Open `Conan/Panel 4.py` and press **Run Python File**. Internet access is needed
for Yahoo Finance. Select a company, adjust sliders and hover over the heatmap.
Select Bear/Base/Bull before editing that scenario. Compare scenarios opens a table.

## Files

- `../Panel 4.py`: desktop interface and Matplotlib chart.
- `data_loader.py`: real Yahoo Finance data through yfinance; 15-minute memory cache.
- `valuation.py`: shared calculations for the headline, scenarios and heatmap.
- `requirements.txt`: Matplotlib and yfinance.
- `cache/`: ignored local Yahoo cookie/timezone cache. Do not share it.

Install if needed, from Group_Project:

```powershell
Conan/.venv/Scripts/python.exe -m pip install -r "Conan/Panel 4 data/requirements.txt"
```

## Data choices

- Price: Yahoo regular-market quote and its UTC timestamp; quotes can be delayed.
- Revenue: latest available completed annual income statement, with period-end date.
- Shares: Yahoo's total implied share count where available, including Alphabet's
  share classes. Counts are as supplied when retrieved, not the revenue-period average.
- TSM: the returned share count is ADS-equivalent; do not divide it by five again.
- ASML/TSM revenue is converted from its reporting currency into USD using Yahoo's
  latest available FX close. The rate/date is shown in Data & assumptions. That FX
  rate is held constant; the conversion is not historical-period average FX.
- Price × shares is cross-checked against Yahoo market capitalisation. Inconsistent
  or missing inputs show an error and allow manual entry, never fictional substitutes.
- A failed refresh retains previously loaded figures for the same company, visibly
  labelled Saved data. Switching company clears the previous company's figures.
- Manual inputs use USD and billions; dates are labelled unspecified. Refresh
  restores Yahoo inputs. Future shares remain constant; dividends are excluded.
- Scenario presets are illustrative assumptions, not analyst forecasts.

Sources: [yfinance API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.html),
[TSMC ADS ratio](https://www.sec.gov/Archives/edgar/data/1046179/000119312524099840/d592628dex2a1.htm),
[ASML shares](https://www.asml.com/en/investors/shares).
