"""Deterministic integrated callback tests; no Yahoo/network or bytecode writes.

Run: python -B -m unittest discover -s tests -v
Fixtures are confined to unittest patch contexts. This module cannot serve a dashboard.
For visual verification with real Yahoo data, run: python -B app.py
"""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as dashboard


def prices(tickers, start, end=None):
    historical = "^IXIC" in tickers
    dates = pd.date_range(start, "2003-12-31" if historical else "2026-10-30", freq="B")
    t = np.arange(len(dates))
    return pd.DataFrame({ticker: 100 * np.exp(
        (0.0001 if ticker == "^IXIC" else 0.0003 + list(dashboard.COMPANIES).index(ticker) * 0.00008
         if ticker in dashboard.COMPANIES else 0.0002) * t
        + (0.03 if ticker == "NVDA" else 0.08) * np.sin(t / 35)
    ) for ticker in tickers}, index=dates)


def valuations(tickers):
    return pd.DataFrame([{
        "Ticker": ticker, "Company": dashboard.COMPANIES[ticker],
        "Sector": dashboard.p2_sector_map[ticker],
        "Market Cap ($T)": 0.2 + i * 0.3,
        "Capital Expenditure ($B)": 4 + i * 9,
        "Annual Revenue ($B)": 15 + i * i * 8,
    } for i, ticker in enumerate(tickers)]), []


def company_data(company, force=False):
    ticker = dashboard.p4_COMPANIES[company]
    i = dashboard.TICKERS.index(ticker)
    return dict(company=company, ticker=ticker, current_price=100 + 10*i,
                revenue=(80 + 7*i)*1e9, shares=(4 + i/2)*1e9,
                mode="Synthetic QA fixture", source="Synthetic QA fixture",
                quote_date="30 Oct 2026", revenue_date="31 Dec 2025",
                retrieved="Synthetic QA fixture", share_note="Synthetic QA fixture",
                fx_note="Synthetic USD fixture")


def fixtures():
    return [patch.object(dashboard, "stock_get_prices", prices),
            patch.object(dashboard, "p2_load_valuation_data", valuations),
            patch.object(dashboard, "p4_load_company_data", company_data),
            patch.object(dashboard.panel3, "download_prices", prices),
            patch.object(dashboard.panel3, "trading_session_closes",
                         lambda day: pd.DatetimeIndex(["2026-10-30 20:00:00+00:00"]))]


class DashboardIntegration(unittest.TestCase):
    def setUp(self):
        for fixture in fixtures():
            fixture.start()
            self.addCleanup(fixture.stop)
        self.app = dashboard.app
        self.client = self.app.server.test_client()

    def post(self, app, key, values, states=None, changed=None):
        callback = app.callback_map[key]
        outputs = callback["output"]
        spec = lambda o: dict(id=o.component_id, property=o.component_property)
        inputs = [dict(i, value=values[(i["id"], i["property"])]) for i in callback["inputs"]]
        state = [dict(i, value=states[(i["id"], i["property"])]) for i in callback["state"]]
        result = self.client.post(app.config.requests_pathname_prefix + "_dash-update-component", json=dict(
            output=key, outputs=[spec(o) for o in outputs] if isinstance(outputs, list) else spec(outputs),
            inputs=inputs, state=state,
            changedPropIds=[changed or inputs[0]["id"] + "." + inputs[0]["property"]]))
        self.assertEqual(result.status_code, 200, result.data)
        return result.get_json()["response"]

    def test_toggle_and_defaults(self):
        self.assertEqual(len(dashboard.TICKERS), 10)
        self.assertEqual(dashboard.stock_DEFAULT_COMPANIES, dashboard.TICKERS)
        layout = self.client.get("/_dash-layout").get_data(as_text=True)
        self.assertNotIn("AVGO", layout)
        self.assertNotIn("CRM", layout)
        values = {("chip-"+t, "n_clicks"): 1 for t in dashboard.TICKERS}
        values.update({("select-all", "n_clicks"): 1, ("deselect-all", "n_clicks"): 1})
        current = []
        for ticker in dashboard.TICKERS + list(reversed(dashboard.TICKERS)):
            current = self.post(self.app, "company-selection.data", values,
                                {("company-selection", "data"): current}, "chip-"+ticker+".n_clicks")["company-selection"]["data"]
        self.assertEqual(current, [])
        for button, expected in [("select-all", dashboard.TICKERS), ("deselect-all", [])]:
            current = self.post(self.app, "company-selection.data", values,
                                {("company-selection", "data"): current}, button+".n_clicks")["company-selection"]["data"]
            self.assertEqual(current, expected)

    def test_all_selection_sizes_and_views(self):
        nav = next(k for k in self.app.callback_map if "frame-3.src" in k)
        for count in [10, 5, 1, 0]:
            with self.subTest(count=count):
                selected = dashboard.TICKERS[:count]
                applied = self.post(self.app, "applied-companies.data", {("company-selection", "data"): selected})
                self.assertEqual(applied["applied-companies"]["data"], selected)
                for number, panel in enumerate(self.app.panel_apps[:3], 1):
                    values = {("open-"+str(i), "n_clicks"): 0 for i in range(1, 5)}
                    values.update({("back", "n_clicks"): 0, ("applied-companies", "data"): selected})
                    routed = self.post(self.app, nav, values, {("current-panel", "data"): number}, "applied-companies.data")
                    url = routed["frame-"+str(number)]["src"]
                    search = "?"+url.split("?", 1)[1]
                    self.assertEqual(dashboard.global_parse_selection(search), selected)
                    figure = self.post(self.app, f"graph-{number}.figure", {
                        ("load-data", "n_intervals"): 1, ("applied-companies", "data"): selected})[f"graph-{number}"]["figure"]
                    self.assertEqual(figure["layout"]["paper_bgcolor"], dashboard.panel3.NAVY)
                    if count == 0:
                        self.assertEqual(figure["data"], [])
                        self.assertIn("select at least one", json.dumps(figure))
                    elif number == 3:
                        self.assertEqual(figure["layout"]["meta"]["company_count"], count)
                        self.assertEqual(len(figure["data"]), 5)
                    if number == 1:
                        chosen = self.post(panel, "p1-companies.data", {("p1-url", "search"): search})
                        self.assertEqual(chosen["p1-companies"]["data"], selected)
                        content = self.post(panel, "p1-result.children", {
                            ("p1-companies", "data"): selected, ("p1-period", "value"): "3 Years",
                            ("p1-benchmarks", "value"): ["SPY", "QQQ"], ("p1-detail", "value"): "Every 5 Days",
                            ("p1-refresh", "n_clicks"): 0})
                    elif number == 2:
                        content = self.post(panel, "view-content.children", {("view-toggle", "value"): "graph",
                            ("valuation-refresh", "n_clicks"): 0, ("p2-url", "search"): search})
                    elif number == 3:
                        content = self.post(panel, "panel3-content.children", {("panel3-url", "search"): search})
                        if count:
                            self.assertIn("not statistically validated predictions", json.dumps(content))
                            self.assertIn("NASDAQ PRICE RETURN", json.dumps(content))
                    self.assertNotIn("Traceback", json.dumps(content))
                    if count == 0:
                        self.assertIn("select", json.dumps(content).lower())

    def test_panel4_remains_independent(self):
        self.assertEqual(dashboard.GLOBAL_PANELS, (1, 2, 3))
        self.assertEqual(dashboard.global_panel_url(4, []), "/panel/4/")
        self.assertEqual(self.app.callback_map["graph-4.figure"]["inputs"],
                         [{"id": "load-data", "property": "n_intervals"}])
        figures = [dashboard.overview_figure(4, dashboard.TICKERS[:n]).to_json()
                   for n in [0, 1, 5, 10]]
        self.assertTrue(all(figure == figures[0] for figure in figures))
        panel = self.app.panel_apps[3]
        key = next(k for k in panel.callback_map if "panel4-data.data" in k)
        self.assertIn({"id": "panel4-company", "property": "value"}, panel.callback_map[key]["inputs"])
        self.assertNotIn({"id": "p4-url", "property": "search"}, panel.callback_map[key]["inputs"])

    def test_no_duplicate_controls_and_shared_server(self):
        for i, panel in enumerate(self.app.panel_apps, 1):
            self.assertIs(panel.server, self.app.server)
            response = self.client.get(f"/panel/{i}/_dash-layout")
            self.assertEqual(response.status_code, 200)
            layout = response.get_data(as_text=True)
            if i == 4:
                self.assertIn("panel4-company", layout)
                continue
            for obsolete in [ '"id":"global-company-selector"', "panel3-select-all", "panel3-retry"]:
                self.assertNotIn(obsolete, layout)
        self.assertEqual(len(self.app.panel_apps[2].callback_map), 1)


if __name__ == "__main__":
    unittest.main()
