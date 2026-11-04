"""Pure valuation calculations. Money and shares use full units, rates use decimals."""

from math import isfinite


def percent_to_decimal(percent):
    """Convert a slider percentage, such as 20, into 0.20."""
    if not isinstance(percent, (int, float)) or not isfinite(percent):
        raise ValueError("Enter a finite percentage.")
    return percent / 100


def calculate_valuation(revenue, shares, current_price, growth, margin, pe, years):
    """Project a future price with constant shares and no dividends."""
    values = (revenue, shares, current_price, growth, margin, pe, years)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not isfinite(value) for value in values):
        raise ValueError("All inputs must be finite numbers.")
    if revenue <= 0 or shares <= 0 or current_price <= 0:
        raise ValueError("Revenue, shares and current price must be positive.")
    if growth <= -1:
        raise ValueError("Revenue growth must be greater than -100%.")
    if margin < 0:
        raise ValueError("Negative earnings are unsuitable for conventional P/E valuation.")
    if margin > 1:
        raise ValueError("Net margin must be between 0% and 100%.")
    if pe <= 0 or years <= 0 or years != int(years):
        raise ValueError("P/E must be positive; years must be a positive whole number.")
    try:
        future_revenue = revenue * (1 + growth) ** years
        future_net_income = future_revenue * margin
        future_eps = future_net_income / shares
        future_price = future_eps * pe
        annual_return = (future_price / current_price) ** (1 / years) - 1
    except OverflowError as error:
        raise ValueError("These assumptions produce numbers that are too large.") from error
    result = dict(revenue=future_revenue, net_income=future_net_income,
                  eps=future_eps, price=future_price, annual_return=annual_return)
    if not all(isfinite(value) for value in result.values()):
        raise ValueError("These assumptions produce numbers that are too large.")
    return result
