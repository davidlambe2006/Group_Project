"""Company universe and neutral theme shared by the dashboard and panel demos."""

COMPANIES = {
    "NVDA": "NVIDIA", "MSFT": "Microsoft", "GOOGL": "Alphabet",
    "AMZN": "Amazon", "META": "Meta", "AMD": "AMD", "TSM": "TSMC",
    "ORCL": "Oracle", "PLTR": "Palantir", "ASML": "ASML",
}
TICKERS = list(COMPANIES)
NAVY = "#0B1220"
CARD = "#131F31"
BORDER = "#263449"
GRID = "rgba(148,163,184,0.07)"


def apply_theme(app):
    """Use the same CSS variables in every Dash document, including iframes."""
    css = f"""
    :root {{ --dashboard-bg: {NAVY}; --dashboard-card: {CARD};
      --dashboard-border: {BORDER}; --dashboard-grid: {GRID};
      --Dash-Fill-Inverse-Strong: {NAVY}; --Dash-Text-Primary: #F1F5F9;
      --Dash-Fill-Primary-Hover: {CARD}; --Dash-Stroke-Strong: {BORDER}; }}
    html, body {{ margin: 0; background: var(--dashboard-bg); color: #CBD5E1;
      font-family: Inter, 'Segoe UI', Arial, sans-serif; }}
    * {{ box-sizing: border-box; }}
    button {{ background: var(--dashboard-card); color: #CBD5E1;
      border: 1px solid var(--dashboard-border); border-radius: 6px;
      padding: 5px 9px; cursor: pointer; }}
    button:hover {{ border-color: #5ce0d3; }}
    input {{ background: var(--dashboard-card); color: #F1F5F9;
      border: 1px solid var(--dashboard-border); border-radius: 4px; }}
    .dashboard-header {{ min-height: 52px; }}
    .overview-grid {{ padding: 8px; gap: 8px !important; }}
    .overview-card {{ border: 1px solid var(--dashboard-border); border-radius: 8px; }}
    .company-results {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(min(100%,460px),1fr)); gap: 18px; }}
    .company-result {{ background: var(--dashboard-card); border: 1px solid var(--dashboard-border);
      border-radius: 10px; padding: 16px; min-width: 0; }}
    @media(max-width: 850px) {{ .dashboard-header {{ overflow-x: auto; }} }}
    """
    app.index_string = app.index_string.replace("{%css%}", "{%css%}<style>" + css + "</style>")
    return app
