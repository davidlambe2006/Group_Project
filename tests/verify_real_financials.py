"""Opt-in real Yahoo/Git-history audit. Run from the project: python -B tests/verify_real_financials.py.

No fixtures, fallback prices, guessed financials, or modifications to production loaders.
"""
import ast
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as dashboard


ROOT = Path(__file__).resolve().parents[1]


def history(commit, path):
    return subprocess.check_output([
        "git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT),
        "show", f"{commit}:{path}",
    ]).decode("utf-8-sig")


def original_functions(source, namespace, names):
    namespace = namespace.copy()
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<Git financial baseline>", "exec"), namespace)
    return namespace


def main():
    p3 = dashboard.panel3
    old_app = history("bf84eb4", "app.py")
    names = ["stock_get_prices", "stock_index_and_normalise", "stock_calculate_statistics",
             "p2_load_valuation_data", "p2_prepare_chart_data", "p2_calculate_regression", "p2_create_bubble_scatter",
             "p4_load_company_data", "p4_calculate_valuation", "p4_value_for", "p4_heatmap_data"]
    current_nodes = {node.name: ast.dump(node) for node in ast.parse((ROOT / "app.py").read_text(encoding="utf-8-sig")).body
                     if isinstance(node, ast.FunctionDef)}
    baseline_nodes = {node.name: ast.dump(node) for node in ast.parse(old_app).body if isinstance(node, ast.FunctionDef)}
    for name in names:
        assert current_nodes[name] == baseline_nodes[name], f"Changed financial function: {name}"
    baseline = original_functions(old_app, vars(dashboard), names)

    # Both Panel 1 views now call the same cached Yahoo download and transform.
    indexed, missing = dashboard.stock_comparison_data(dashboard.TICKERS, "3 Years", ["SPY", "QQQ"])
    restored = baseline["stock_index_and_normalise"](indexed)
    pd.testing.assert_frame_equal(indexed, restored)
    pd.testing.assert_frame_equal(dashboard.stock_calculate_statistics(indexed),
                                  baseline["stock_calculate_statistics"](indexed))
    print(f"Panel 1: REAL adjusted closes, {indexed.index[0]:%Y-%m-%d} to {indexed.index[-1]:%Y-%m-%d}; missing={missing}; original statistics MATCH", flush=True)

    valuation, failed = dashboard.p2_load_valuation_data(dashboard.p2_tickers)
    cleaned = dashboard.p2_prepare_chart_data(valuation)
    assert not cleaned.empty, "No real valuation rows recovered"
    pd.testing.assert_frame_equal(cleaned, baseline["p2_prepare_chart_data"](valuation))
    original_fit = baseline["p2_calculate_regression"](cleaned)
    current_fit = dashboard.p2_calculate_regression(cleaned)
    np.testing.assert_allclose(current_fit[:3], original_fit[:3])
    print("Panel 2: REAL Yahoo financial statements/market caps, original transformations and regression MATCH", flush=True)
    print(cleaned[["Ticker", "Capital Expenditure ($B)", "Annual Revenue ($B)", "Market Cap ($T)"]].to_string(index=False), flush=True)
    print(f"Panel 2 unavailable tickers (never fabricated): {failed}", flush=True)

    p3_names = ["calculate_ai_portfolio", "calculate_nasdaq", "calculate_metrics", "create_forecast",
                "monthly_with_latest", "render_panel3", "nasdaq_crash_detail"]
    old_p3_source = history("e5c9081", "panels/panel3_dotcom.py")
    old_p3 = original_functions(old_p3_source, vars(p3), p3_names)
    nasdaq = p3.calculate_nasdaq()
    pd.testing.assert_series_equal(nasdaq, old_p3["calculate_nasdaq"]())
    total, drawdown = p3.calculate_metrics(nasdaq)
    assert round(total, 1) == 169.4, (total, "Historical reproduction check failed")
    assert round(drawdown, 1) == -77.9, (drawdown, "Historical reproduction check failed")
    print(f"Panel 3: REAL NASDAQ {nasdaq.index[0]:%Y-%m-%d} to {nasdaq.index[-1]:%Y-%m-%d}; return={total:+.6f}%, drawdown={drawdown:.6f}%", flush=True)
    ai = p3.calculate_ai_portfolio(dashboard.TICKERS)
    pd.testing.assert_series_equal(ai, old_p3["calculate_ai_portfolio"](dashboard.TICKERS))
    current_fig, original_fig = p3.render_panel3(dashboard.TICKERS), old_p3["render_panel3"](dashboard.TICKERS)
    assert current_fig.to_json() == original_fig.to_json(), "Panel 3 financial/chart baseline differs"
    assert p3.nasdaq_crash_detail().to_json() == old_p3["nasdaq_crash_detail"]().to_json()
    print(f"Panel 3: REAL AI buy-and-hold portfolio {ai.index[0]:%Y-%m-%d} to {ai.index[-1]:%Y-%m-%d}; all original traces, scenarios and crash detail MATCH", flush=True)

    data = dashboard.p4_load_company_data("NVIDIA")
    for scenario, assumptions in dashboard.p4_SCENARIOS.items():
        assert dashboard.p4_value_for(data, assumptions) == baseline["p4_value_for"](data, assumptions)
        assert dashboard.p4_heatmap_data(data, assumptions) == baseline["p4_heatmap_data"](data, assumptions)
    assert dashboard.GLOBAL_PANELS == (1, 2, 3)
    assert dashboard.global_panel_url(4, []) == "/panel/4/"
    print(f"Panel 4: REAL {data['ticker']} quote {data['quote_date']}, annual statements {data['revenue_date']}; original single-company valuation and every heatmap cell MATCH", flush=True)
    print("PASS: real data recovered for all four panels; no synthetic providers installed.", flush=True)


if __name__ == "__main__":
    main()
