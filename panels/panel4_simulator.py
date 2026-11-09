"""Panel 4: a standalone Matplotlib dashboard. Press Run to open the window."""

from pathlib import Path
import importlib.util
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from datetime import datetime, timezone

# VS Code may select the main Python installation instead of Conan's environment.
# Restart using our local environment if Matplotlib is missing from that Python.
if __name__ == "__main__" and any(importlib.util.find_spec(name) is None for name in ("matplotlib", "yfinance")):
    local_python = Path(__file__).parent / ".venv" / "Scripts" / "python.exe"
    if local_python.exists() and Path(sys.executable).resolve() != local_python.resolve():
        raise SystemExit(subprocess.call([str(local_python), str(Path(__file__).resolve())]))
    raise SystemExit('Install dependencies: python -m pip install -r "Conan/Panel 4 data/requirements.txt"')

import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from panel4_data_loader import COMPANIES, load_company_data
from valuation import calculate_valuation, percent_to_decimal

NAVY = "#081423"
CYAN = "#20e3e8"
WHITE = "#f3f5fa"
MUTED = "#aabed0"
LINE = "#354b60"

def value_for(data, assumptions):
    """Estimate the future share price and compound annual investment return.

    Use annual revenue in USD, equivalent shares and today's USD price. Convert
    percentage assumptions to decimals, compound revenue over the forecast period,
    multiply by net margin to estimate profit, divide by shares for earnings per
    share, and multiply by the assumed future P/E to estimate a future price.
    Return = (future price / current price) ** (1 / years) - 1.

    The headline, scenarios and grid all use this same calculation. Growth, margin,
    shares, P/E and FX are held constant; dividends, fees and taxes are excluded.
    This is an illustrative earnings-multiple valuation, not a guaranteed forecast
    or a discounted present value of the business.
    """
    return calculate_valuation(
        data["revenue"], data["shares"], data["current_price"],
        percent_to_decimal(assumptions["growth"]),
        percent_to_decimal(assumptions["margin"]), assumptions["pe"], assumptions["years"],
    )

def heatmap_data(data, assumptions):
    """Measure sensitivity to annual sales growth and future net profit margin.

    Recalculate the model for every growth/margin pair using the same company
    revenue, share count and current price. P/E and forecast years stay at the
    selected scenario's values. Each cell contains a future price and compound
    annual return, not a probability. Include the selected values in the grid so
    its outline marks an exact calculation. Model limitations match value_for.
    """
    growth = sorted(set([0, 10, 20, 30, 40, 50, assumptions["growth"]]))
    margins = sorted(set([15, 20, 25, 30, 35, 40, assumptions["margin"]]))
    results = [[value_for(data, {**assumptions, "growth": g, "margin": m})
                for g in growth] for m in margins]
    return growth, margins, results

class ValuationPanel:
    """Tkinter supplies controls; Matplotlib draws the chart inside the window."""

    def __init__(self, root):
        self.root = root
        root.title("Panel 4 - AI Valuation Simulator")
        root.geometry("1400x900")
        root.minsize(1100, 800)
        root.configure(bg=NAVY)
        self.data = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.messages = Queue()
        self.request_id = 0
        self.scenarios = {
            "Bear": dict(growth=10, margin=20, pe=30, years=5),
            "Base": dict(growth=20, margin=25, pe=30, years=5),
            "Bull": dict(growth=30, margin=30, pe=30, years=5),
        }
        self.active = "Base"
        self.variables = {key: tk.IntVar(value=value)
                          for key, value in self.scenarios["Base"].items()}
        self.configure_styles()
        self.build_layout()
        self.change_company()
        self.poll_job = root.after(100, self.poll_data)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def configure_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground="#102235", background=LINE,
                        foreground=WHITE, arrowcolor=WHITE, padding=8)
        style.map("TCombobox", fieldbackground=[("readonly", "#102235")],
                  foreground=[("readonly", WHITE)])
        style.configure("Treeview", background=NAVY, fieldbackground=NAVY,
                        foreground=WHITE, rowheight=30, font=("Arial", 10))
        style.configure("Treeview.Heading", background=LINE, foreground=WHITE)
        style.map("Treeview", background=[("selected", "#176466")])

    def label(self, parent, text="", size=12, color=WHITE, **kwargs):
        return tk.Label(parent, text=text, bg=NAVY, fg=color,
                        font=("Arial", size), **kwargs)

    def button(self, parent, text, command):
        return tk.Button(parent, text=text, command=command, bg=NAVY, fg=WHITE,
                         activebackground=CYAN, activeforeground=NAVY,
                         relief="flat", highlightthickness=1, highlightbackground=LINE,
                         font=("Arial", 13), cursor="hand2", padx=14, pady=8)

    def wrapped_label(self, parent, text="", size=10, color=MUTED):
        """Wrap explanatory copy to the available space when the window resizes."""
        label = self.label(parent, text, size, color, justify="left", anchor="w")
        label.pack(fill="x")
        label.bind("<Configure>", lambda event: label.configure(wraplength=max(120, event.width - 8)))
        return label

    def sidebar_section(self, parent, title):
        tk.Frame(parent, bg=LINE, height=1).pack(fill="x", pady=(16, 12))
        self.label(parent, title, 11, CYAN, anchor="w").pack(fill="x", pady=(0, 8))

    def build_layout(self):
        header = tk.Frame(self.root, bg=NAVY, padx=24, pady=14)
        header.pack(fill="x")
        self.label(header, "04", 23, CYAN).pack(side="left", padx=(0, 24))
        self.label(header, "AI Valuation Simulator", 24).pack(side="left")
        self.source_badge = self.label(header, "YAHOO FINANCE", 10, CYAN)
        self.source_badge.pack(side="right")
        tk.Frame(self.root, bg=LINE, height=1).pack(fill="x")
        body = tk.Frame(self.root, bg=NAVY)
        body.pack(fill="both", expand=True)
        sidebar = tk.Frame(body, bg=NAVY, width=240, padx=16, pady=16)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        tk.Frame(body, bg=LINE, width=1).pack(side="left", fill="y")
        main = tk.Frame(body, bg=NAVY, padx=24, pady=16)
        main.pack(side="left", fill="both", expand=True)

        self.label(sidebar, "Company", 11, CYAN, anchor="w").pack(fill="x", pady=(0, 8))
        self.company = tk.StringVar(value="NVIDIA")
        menu = ttk.Combobox(sidebar, textvariable=self.company,
                            values=list(COMPANIES), state="readonly", font=("Arial", 12))
        menu.pack(fill="x")
        menu.bind("<<ComboboxSelected>>", self.change_company)
        self.sidebar_section(sidebar, "Financials")
        self.source_summary = self.wrapped_label(sidebar)
        self.sidebar_section(sidebar, "Assumptions")
        self.sidebar_assumptions = self.wrapped_label(sidebar)
        self.button(sidebar, "Data & assumptions", self.show_data).pack(fill="x", pady=(10, 0))
        self.sidebar_section(sidebar, "Source & update")
        self.source_details = self.wrapped_label(sidebar, "Loading reported financial data...", 9)
        self.reload_button = self.button(sidebar, "Refresh data", lambda: self.change_company(force=True))
        self.reload_button.pack(fill="x", pady=(10, 0))

        headline = tk.Frame(main, bg=NAVY)
        headline.pack(fill="x", pady=(0, 12))
        self.metric_labels = {}
        for column, (key, title) in enumerate([
            ("current", "Current price (USD)"),
            ("future", "Est. price in 5 years (USD)"),
            ("return", "Est. annual return"),
        ]):
            headline.columnconfigure(column, weight=1, uniform="metrics")
            card = tk.Frame(headline, bg=NAVY, highlightbackground=LINE,
                            highlightthickness=1, padx=12, pady=10)
            card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 0))
            self.wrapped_label(card, title, 11, WHITE)
            value = self.label(card, "--", 25, CYAN, anchor="w")
            value.pack(fill="x", pady=(5, 3))
            self.metric_labels[key] = value

        scenario_row = tk.Frame(main, bg=NAVY)
        scenario_row.pack(fill="x", pady=(0, 10))
        self.scenario_buttons = {}
        for column, name in enumerate(self.scenarios):
            scenario_row.columnconfigure(column, weight=1, uniform="scenarios")
            cell = tk.Frame(scenario_row, bg=NAVY)
            cell.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 0))
            button = self.button(cell, name, lambda selected=name: self.select_scenario(selected))
            button.pack(fill="x")
            self.scenario_buttons[name] = button

        self.wrapped_label(main, "Annual return sensitivity", 16, WHITE)
        self.wrapped_label(main, "Colour key: turquoise = gain | white = 0% | orange = loss", 9)
        self.assumptions_label = self.wrapped_label(main, size=9)

        # Pack the footer first so the expanding heatmap cannot hide its controls.
        bottom = tk.Frame(main, bg=NAVY)
        bottom.pack(side="bottom", fill="x")
        footer = tk.Frame(bottom, bg=NAVY)
        footer.pack(side="bottom", fill="x", pady=(6, 0))
        self.button(footer, "Compare scenarios", self.show_comparison).pack(side="left")
        self.status = self.label(footer, "", 9, MUTED, wraplength=440, justify="right")
        self.status.pack(side="right")
        self.wrapped_label(bottom, "Illustrative estimates | Excludes dividends", 9)
        self.calculation_button = self.button(footer, "+ How it works", self.toggle_calculation)
        self.calculation_button.pack(side="left", padx=(8, 0))
        self.calculation_details = tk.Frame(bottom, bg=NAVY, pady=6)
        self.wrapped_label(self.calculation_details,
            "Revenue Growth: how quickly annual sales increase; sales compound at this rate each year.\n"
            "Net Profit Margin: the share of future sales retained as profit after expenses.\n"
            "P/E Ratio: the multiple investors are assumed to pay for future annual earnings.\n"
            "Forecast Period: the number of years projected (five years for all three scenarios).\n"
            "Estimated Return: (future price / current price) raised to 1 / years, minus 1.\n"
            "Future price = current annual revenue * (1 + growth)^years * margin / shares * P/E. "
            "Shares, growth, margin, P/E and FX stay constant. This is a scenario estimate, not a guaranteed return.", 9)
        self.figure = Figure(figsize=(9, 5), dpi=100, facecolor=NAVY)
        self.canvas = FigureCanvasTkAgg(self.figure, master=main)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.canvas.mpl_connect("figure_leave_event", self.on_hover)

    def toggle_calculation(self):
        if self.calculation_details.winfo_manager():
            self.calculation_details.pack_forget()
            self.calculation_button.configure(text="+ How it works")
        else:
            self.calculation_details.pack(fill="x")
            self.calculation_button.configure(text="- How it works")

    def assumptions(self):
        return {key: variable.get() for key, variable in self.variables.items()}

    def select_scenario(self, name):
        self.active = name
        for key, value in self.scenarios[name].items():
            self.variables[key].set(value)
        self.refresh()

    def change_company(self, _event=None, force=False):
        """Download off the UI thread, so the window stays responsive."""
        company = self.company.get()
        self.request_id += 1
        request_id = self.request_id
        if not self.data or self.data["company"] != company:
            self.data = None
        self.source_badge.configure(text="LOADING YAHOO DATA", fg=MUTED)
        self.source_summary.configure(text=f'{COMPANIES[company]} · USD listing\nLoading reported financial data…')
        self.source_details.configure(text="Yahoo Finance via yfinance\nFetching latest available data...")
        self.status.configure(text="Fetching Yahoo Finance…")
        self.refresh()

        def download():
            try:
                self.messages.put((request_id, load_company_data(company, force), None))
            except Exception as error:
                self.messages.put((request_id, None, str(error)))

        self.executor.submit(download)

    def poll_data(self):
        """Only the main Tk thread updates labels and redraws the figure."""
        try:
            while True:
                request_id, data, error = self.messages.get_nowait()
                if request_id != self.request_id:
                    continue  # Ignore a late response for a previously selected company.
                if error:
                    self.source_badge.configure(text="SAVED DATA" if self.data else "DATA UNAVAILABLE", fg="#ffb080")
                    self.status.configure(text="Could not refresh Yahoo. Try Refresh data or use manual inputs.")
                    self.last_error = error
                    if self.data:
                        self.update_source_summary()
                        self.source_details.configure(text=self.source_details.cget("text") + "\nRefresh failed; showing saved figures.")
                    else:
                        self.source_summary.configure(text=f'{COMPANIES[self.company.get()]} · USD\nNo figures loaded. Use Data & assumptions.')
                        self.source_details.configure(text="Yahoo Finance unavailable.\nRetry refresh or enter manual USD inputs.")
                else:
                    self.data = data
                    self.last_error = None
                    self.source_badge.configure(text="YAHOO FINANCE", fg=CYAN)
                    self.status.configure(text=f'Quote may be delayed · Retrieved {data["retrieved"]}')
                    self.update_source_summary()
                self.refresh()
        except Empty:
            pass
        self.poll_job = self.root.after(100, self.poll_data)

    def update_source_summary(self):
        """Label the model's starting inputs and their provenance separately.

        Revenue is the latest available annual sales figure converted to USD;
        equivalent shares match the quoted share/ADS units. Dates distinguish the
        financial year, price timestamp and download time, which may differ.
        Manual inputs retain their user-supplied provenance rather than implying
        a live quote or a verified financial period.
        """
        data = self.data
        self.source_summary.configure(text=(
            f'Revenue: ${data["revenue"] / 1e9:,.2f}bn\n'
            f'Shares: {data["shares"] / 1e9:,.3f}bn'))
        self.source_details.configure(text=(
            f'{data["mode"]}\n{data["retrieved"]}'))

    def show_data(self):
        window = tk.Toplevel(self.root)
        window.title("Source data & assumptions")
        window.configure(bg=NAVY)
        window.transient(self.root)
        window.grab_set()
        content = tk.Frame(window, bg=NAVY, padx=24, pady=20)
        content.pack(fill="both", expand=True)
        data = self.data
        if data:
            details = (f'{data["company"]} ({data["ticker"]}) · {data["source"]}\n'
                       f'Quote: {data["quote_date"]}\nAnnual period ended: {data["revenue_date"]}\n'
                       f'Shares retrieved: {data["retrieved"]}\n{data["share_note"]}\n{data["fx_note"]}')
            if data["mode"] != "Manual inputs":
                details += f'\nReported revenue: {data["reported_revenue"] / 1e9:,.3f}bn {data["financial_currency"]}'
        else:
            details = "No Yahoo data loaded. Enter verified USD inputs below."
            if getattr(self, "last_error", None):
                details += f"\nDownload error: {self.last_error[:220]}"
        self.label(content, details, 11, MUTED, wraplength=610, justify="left").pack(anchor="w", pady=(0, 16))
        entries = {}
        for key, title, value in [
            ("price", "Share / ADS price (USD)", data["current_price"] if data else ""),
            ("revenue", "Annual revenue (USD billions)", data["revenue"] / 1e9 if data else ""),
            ("shares", "Total equivalent shares (billions)", data["shares"] / 1e9 if data else ""),
        ]:
            row = tk.Frame(content, bg=NAVY)
            row.pack(fill="x", pady=6)
            self.label(row, title, 11).pack(side="left", padx=(0, 18))
            variable = tk.StringVar(value=f"{value:.10g}" if isinstance(value, (int, float)) else value)
            tk.Entry(row, textvariable=variable, width=20, font=("Arial", 11), bg="#102235",
                     fg=WHITE, insertbackground=WHITE).pack(side="right")
            entries[key] = variable
        self.label(content, "Manual changes are labelled. Refresh restores Yahoo data.\n"
                   "Future shares and FX stay constant. Returns exclude dividends.\n"
                   "Scenario presets are illustrative assumptions, not analyst forecasts.",
                   10, MUTED, justify="left").pack(anchor="w", pady=14)

        def apply():
            try:
                updated = {"company": self.company.get(), "ticker": COMPANIES[self.company.get()],
                           "current_price": float(entries["price"].get()),
                           "revenue": float(entries["revenue"].get()) * 1e9,
                           "shares": float(entries["shares"].get()) * 1e9,
                           "source": "User-entered USD inputs", "mode": "Manual inputs", "currency": "USD",
                           "quote_date": "User supplied; date unspecified", "revenue_date": "User supplied; date unspecified",
                           "retrieved": datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC"),
                           "share_note": "User-supplied total equivalent share count.",
                           "fx_note": "Revenue entered in USD. FX stays constant."}
                value_for(updated, self.assumptions())
                # Check the full grid as well, before accepting potentially huge inputs.
                heatmap_data(updated, self.assumptions())
            except ValueError as error:
                messagebox.showerror("Check your inputs", str(error), parent=window)
                return
            self.request_id += 1
            self.data = updated
            self.source_badge.configure(text="MANUAL INPUTS", fg="#ffb080")
            self.status.configure(text="Manual inputs · Refresh data to restore Yahoo figures")
            self.update_source_summary()
            self.refresh()
            window.destroy()

        self.button(content, "Apply manual inputs", apply).pack(anchor="e")

    def close(self):
        self.root.after_cancel(self.poll_job)
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()

    def refresh(self):
        """Present the same scenario valuation in metrics and the heatmap.

        Compare projected and current prices to explain the sign of the compound
        annual return. This is a conditional estimate, not a price forecast or a
        recommendation; scenario assumptions are illustrative and shares stay fixed.
        """
        assumptions = self.assumptions()
        self.sidebar_assumptions.configure(text=(
            f'Growth: {assumptions["growth"]}% / year\n'
            f'Net margin: {assumptions["margin"]}%\n'
            f'P/E: {assumptions["pe"]}x\n'
            f'Period: {assumptions["years"]} years'))
        self.assumptions_label.configure(text=(
            'White outline: base | Dashed: selected'))
        for name, button in self.scenario_buttons.items():
            result = value_for(self.data, self.scenarios[name]) if self.data else None
            annual = f'{result["annual_return"]:+.1%} / yr' if result else '--'
            button.configure(text=f'{name}  |  {annual}',
                             bg=CYAN if name == self.active else NAVY,
                             fg=NAVY if name == self.active else WHITE,
                             font=("Arial", 11))
        if self.data is None:
            for label in self.metric_labels.values():
                label.configure(text="--", fg=MUTED)
            self.figure.clear()
            self.axes = self.figure.add_subplot(111)
            self.axes.set_facecolor(NAVY)
            self.axes.axis("off")
            self.axes.text(.5, .5, "Load company data",
                           transform=self.axes.transAxes, color=MUTED, ha="center", fontsize=12)
            self.canvas.draw_idle()
            return
        outcome = value_for(self.data, assumptions)
        self.metric_labels["current"].configure(text=f'${self.data["current_price"]:,.2f}', fg=WHITE)
        self.metric_labels["future"].configure(text=f'${outcome["price"]:,.2f}', fg=WHITE)
        self.metric_labels["return"].configure(text=f'{outcome["annual_return"]:+.1%}',
            fg=CYAN if outcome["annual_return"] >= 0 else "#ffb080")
        self.draw_heatmap(assumptions)

    def draw_heatmap(self, assumptions):
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=.08, right=.94, bottom=.12, top=.98)
        self.growth_values, self.margin_values, self.grid_results = heatmap_data(self.data, assumptions)
        returns = [[result["annual_return"] * 100 for result in row] for row in self.grid_results]
        limit = max(15, max(abs(value) for row in returns for value in row))
        colours = LinearSegmentedColormap.from_list("returns", ["#ff9855", "#edf0ec", "#20e3e8"])
        image = self.axes.imshow(returns, origin="lower", aspect="auto", cmap=colours,
                                 norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
        self.axes.set_xticks(range(len(self.growth_values)), self.growth_values)
        self.axes.set_yticks(range(len(self.margin_values)), self.margin_values)
        self.axes.set_xticks([i - .5 for i in range(1, len(self.growth_values))], minor=True)
        self.axes.set_yticks([i - .5 for i in range(1, len(self.margin_values))], minor=True)
        self.axes.grid(which="minor", color=NAVY, linewidth=.6, alpha=.3)
        self.axes.tick_params(which="minor", length=0)
        self.axes.set_xlabel("Revenue growth (%)", color=WHITE, fontsize=13, labelpad=12)
        self.axes.set_ylabel("Net profit margin (%)", color=WHITE, fontsize=12, labelpad=10)
        self.axes.tick_params(colors=WHITE, labelsize=12)
        for spine in self.axes.spines.values():
            spine.set_visible(False)
        for row, values in enumerate(returns):
            for column, value in enumerate(values):
                self.axes.text(column, row, f"{value:.1f}%", ha="center", va="center",
                               fontsize=15, fontweight="bold", color=NAVY)
        # A dark outer border keeps the white base-case outline visible near zero.
        base = self.scenarios["Base"]
        x = self.growth_values.index(base["growth"])
        y = self.margin_values.index(base["margin"])
        for colour, width in [(NAVY, 5), (WHITE, 2.5)]:
            self.axes.add_patch(Rectangle((x - .47, y - .47), .94, .94, fill=False,
                                          edgecolor=colour, linewidth=width))
        if self.active != "Base":
            x = self.growth_values.index(assumptions["growth"])
            y = self.margin_values.index(assumptions["margin"])
            self.axes.add_patch(Rectangle((x - .47, y - .47), .94, .94, fill=False,
                                          edgecolor=NAVY, linewidth=2, linestyle="--"))
        colourbar = self.figure.colorbar(image, ax=self.axes, fraction=.035, pad=.04, format="%.0f%%")
        colourbar.ax.tick_params(colors=WHITE, labelsize=10)
        colourbar.outline.set_edgecolor(LINE)
        colourbar.set_ticks([-limit, 0, limit])
        colourbar.set_label("Estimated annual return (%)", color=MUTED, fontsize=9)
        self.tooltip = self.axes.annotate("", xy=(0, 0), xytext=(12, 12),
                                         textcoords="offset points", fontsize=10, color=WHITE,
                                         bbox=dict(boxstyle="round,pad=.6", fc=NAVY, ec=CYAN),
                                         zorder=10, annotation_clip=False)
        self.tooltip.set_visible(False)
        self.canvas.draw_idle()

    def on_hover(self, event):
        if self.data is None or not hasattr(self, "tooltip"):
            return
        if event.inaxes != self.axes or event.xdata is None or event.ydata is None:
            if self.tooltip.get_visible():
                self.tooltip.set_visible(False)
                self.canvas.draw_idle()
            return
        column, row = int(event.xdata + .5), int(event.ydata + .5)
        if not (0 <= row < len(self.grid_results) and 0 <= column < len(self.growth_values)):
            return
        result = self.grid_results[row][column]
        assumptions = self.assumptions()
        self.tooltip.xy = (column, row)
        self.tooltip.set_text(
            f'Growth {self.growth_values[column]}% · Margin {self.margin_values[row]}%\n'
            f'Future price ${result["price"]:,.2f} · Return {result["annual_return"]:+.1%}\n'
            f'P/E {assumptions["pe"]}× · {assumptions["years"]} years')
        right_half = column >= len(self.growth_values) / 2
        top_half = row >= len(self.margin_values) / 2
        self.tooltip.set_position((-12 if right_half else 12, -12 if top_half else 12))
        self.tooltip.set_ha("right" if right_half else "left")
        self.tooltip.set_va("top" if top_half else "bottom")
        self.tooltip.set_visible(True)
        self.canvas.draw_idle()

    def show_comparison(self):
        if self.data is None:
            messagebox.showinfo("Load data first", "Refresh Yahoo data or enter manual inputs first.", parent=self.root)
            return
        window = tk.Toplevel(self.root)
        window.title("Illustrative scenarios")
        window.configure(bg=NAVY)
        self.label(window, "Compare the Bear, Base and Bull scenario presets.", 11, MUTED).pack(padx=18, pady=12)
        columns = ("Scenario", "Growth", "Margin", "P/E", "Years", "Future price", "Annual return")
        table = ttk.Treeview(window, columns=columns, show="headings", height=3)
        for column in columns:
            table.heading(column, text=column)
            table.column(column, width=110, anchor="center")
        for name, scenario in self.scenarios.items():
            outcome = value_for(self.data, scenario)
            table.insert("", "end", values=(name, f'{scenario["growth"]}%', f'{scenario["margin"]}%',
                         scenario["pe"], scenario["years"], f'${outcome["price"]:,.2f}',
                         f'{outcome["annual_return"]:+.1%}'))
        table.pack(padx=18, pady=(0, 18))
        # Keep the comparison visible until it is dismissed.
        window.transient(self.root)
        window.grab_set()

if __name__ == "__main__":
    root = tk.Tk()
    panel = ValuationPanel(root)
    root.mainloop()
