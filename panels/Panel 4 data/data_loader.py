"""Yahoo Finance loading and currency checks, separate from the interface."""

from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
import time

import yfinance as yf


COMPANIES = {
    "NVIDIA": "NVDA", "Microsoft": "MSFT", "Alphabet": "GOOGL",
    "Amazon": "AMZN", "Meta": "META", "Broadcom": "AVGO", "AMD": "AMD",
    "TSMC": "TSM", "Oracle": "ORCL", "Palantir": "PLTR",
    "Salesforce": "CRM", "ASML": "ASML",
}

# Keep Yahoo's cookie/timezone cache inside this panel's own folder.
yf.set_tz_cache_location(str(Path(__file__).parent / "cache"))
_cache = {}


def positive_number(value, label):
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Yahoo did not provide {label}. Try again or enter it manually.") from error
    if not isfinite(value) or value <= 0:
        raise ValueError(f"Yahoo returned invalid {label}. Enter it manually.")
    return value


def load_company_data(company, force=False):
    """Return real, dated inputs. Never substitute fictional numbers on failure."""
    if company not in COMPANIES:
        raise ValueError("Choose a company from the list.")
    cached = _cache.get(company)
    if cached and not force and time.monotonic() - cached[0] < 900:
        return cached[1].copy()

    ticker = COMPANIES[company]
    stock = yf.Ticker(ticker)
    info = stock.get_info()
    price = positive_number(info.get("regularMarketPrice"), "the share price")
    stamp = positive_number(info.get("regularMarketTime"), "the quote timestamp")
    quote_date = datetime.fromtimestamp(stamp, timezone.utc).strftime("%d %b %Y %H:%M UTC")
    currency = info.get("currency")
    financial_currency = info.get("financialCurrency")
    if currency != "USD" or not financial_currency:
        raise ValueError("The listing/reporting currency could not be verified. Use manual USD inputs.")

    statement = stock.income_stmt
    if statement.empty or "Total Revenue" not in statement.index:
        raise ValueError("Annual revenue is unavailable from Yahoo. Try again or enter it manually.")
    revenues = statement.loc["Total Revenue"].dropna().sort_index(ascending=False)
    if revenues.empty:
        raise ValueError("Yahoo did not provide an annual revenue figure.")
    revenue_date = revenues.index[0].strftime("%d %b %Y")
    reported_revenue = positive_number(revenues.iloc[0], "annual revenue")

    # Alphabet's class-A shares alone exclude other share classes. The implied
    # count includes all classes. For TSM, Yahoo already gives ADS-equivalent shares.
    share_value = info.get("impliedSharesOutstanding") or info.get("sharesOutstanding")
    shares = positive_number(share_value, "total equivalent shares")
    market_cap = positive_number(info.get("marketCap"), "market capitalisation")
    if abs(shares * price / market_cap - 1) > .10:
        raise ValueError("Yahoo's price and share units disagree. Refresh or use verified manual inputs.")

    fx_rate, fx_note = 1.0, "No currency conversion needed."
    if financial_currency != currency:
        fx_symbol = f"{financial_currency}{currency}=X"
        history = yf.Ticker(fx_symbol).history(period="5d", auto_adjust=False)
        if history.empty:
            raise ValueError(f"Could not load {financial_currency}/USD FX. Use manual USD revenue.")
        closes = history["Close"].dropna()
        if closes.empty:
            raise ValueError("The FX series has no valid closing rate.")
        fx_rate = positive_number(closes.iloc[-1], "the currency conversion rate")
        fx_date = closes.index[-1].strftime("%d %b %Y")
        fx_note = f"1 {financial_currency} = {fx_rate:.6f} USD · {fx_symbol} · {fx_date}. FX held constant."

    share_note = "Yahoo total equivalent shares; all share classes included where reported."
    if ticker == "TSM":
        share_note = "Yahoo ADS-equivalent count. One TSM ADS represents five ordinary shares."
    data = {
        "company": company, "ticker": ticker, "revenue": reported_revenue * fx_rate,
        "reported_revenue": reported_revenue, "financial_currency": financial_currency,
        "shares": shares, "current_price": price, "currency": currency,
        "quote_date": quote_date, "revenue_date": revenue_date,
        "retrieved": datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC"),
        "source": "Yahoo Finance via yfinance", "mode": "Yahoo Finance",
        "fx_note": fx_note, "share_note": share_note,
    }
    _cache[company] = (time.monotonic(), data)
    return data.copy()
