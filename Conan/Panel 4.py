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

DATA_FOLDER = Path(__file__).parent / "Panel 4 data"
sys.path.insert(0, str(DATA_FOLDER))

import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from data_loader import COMPANIES, load_company_data
from valuation import calculate_valuation, percent_to_decimal


NAVY = "#081423"
CYAN = "#20e3e8"
WHITE = "#f3f5fa"
MUTED = "#aabed0"
LINE = "#354b60"


def value_for(data, assumptions):
    """One shared calculation for the headline, scenarios and every heatmap cell."""
    return calculate_valuation(
        data["revenue"], data["shares"], data["current_price"],
        percent_to_decimal(assumptions["growth"]),
        percent_to_decimal(assumptions["margin"]), assumptions["pe"], assumptions["years"],
    )


def heatmap_data(data, assumptions):
    """Include selected values so the white outline always marks an exact result."""
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
        self.redraw_job = None
        self.scenarios = {
            "Bear": dict(growth=10, margin=20, pe=30, years=5),
            "Base": dict(growth=20, margin=25, pe=30, years=5),
            "Bull": dict(growth=30, margin=30, pe=30, years=5),
        }
        self.active = "Base"
        self.loading = False
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
        style.configure("Sharp.Horizontal.TScale", background=CYAN, troughcolor=LINE,
                        bordercolor=LINE, lightcolor=CYAN, darkcolor=CYAN, sliderlength=18)

    def label(self, parent, text="", size=12, color=WHITE, **kwargs):
        return tk.Label(parent, text=text, bg=NAVY, fg=color,
                        font=("Arial", size), **kwargs)

    def button(self, parent, text, command):
        return tk.Button(parent, text=text, command=command, bg=NAVY, fg=WHITE,
                         activebackground=CYAN, activeforeground=NAVY,
                         relief="flat", highlightthickness=1, highlightbackground=LINE,
                         font=("Arial", 13), cursor="hand2", padx=14, pady=8)

    def build_layout(self):
        header = tk.Frame(self.root, bg=NAVY, padx=24, pady=16)
        header.pack(fill="x")
        self.label(header, "04", 23, CYAN).pack(side="left", padx=(0, 24))
        self.label(header, "AI Valuation Simulator", 25).pack(side="left")
        self.source_badge = self.label(header, "YAHOO FINANCE", 11, CYAN)
        self.source_badge.pack(side="right")
        tk.Frame(self.root, bg=LINE, height=1).pack(fill="x")
        body = tk.Frame(self.root, bg=NAVY)
        body.pack(fill="both", expand=True)
        sidebar = tk.Frame(body, bg=NAVY, width=285, padx=20, pady=20)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        tk.Frame(body, bg=LINE, width=1).pack(side="left", fill="y")
        main = tk.Frame(body, bg=NAVY, padx=24, pady=18)
        main.pack(side="left", fill="both", expand=True)

        self.company = tk.StringVar(value="NVIDIA")
        menu = ttk.Combobox(sidebar, textvariable=self.company,
                            values=list(COMPANIES), state="readonly", font=("Arial", 13))
        menu.pack(fill="x")
        menu.bind("<<ComboboxSelected>>", self.change_company)
        self.source_summary = self.label(sidebar, "", 9, MUTED, justify="left", anchor="w", wraplength=240)
        self.source_summary.pack(fill="x", pady=(10, 12))
        self.reload_button = self.button(sidebar, "↻ Refresh data", lambda: self.change_company(force=True))
        self.reload_button.pack(fill="x", pady=(0, 10))
        self.slider_labels = {}
        for key, title, minimum, maximum, step in [
            ("growth", "Growth", -20, 60, 5), ("margin", "Margin", 0, 50, 5),
            ("pe", "P/E", 5, 60, 1), ("years", "Years", 1, 10, 1),
        ]:
            row = tk.Frame(sidebar, bg=NAVY)
            row.pack(fill="x", pady=(8, 0))
            self.label(row, title, 15).pack(side="left")
            self.slider_labels[key] = self.label(row, size=15)
            self.slider_labels[key].pack(side="right")
            # A themed slider keeps the thumb clearly visible against the dark panel.
            slider = ttk.Scale(sidebar, from_=minimum, to=maximum, orient="horizontal",
                               style="Sharp.Horizontal.TScale")
            slider.set(self.variables[key].get())
            slider.configure(command=lambda value, k=key, s=step: self.move_slider(k, value, s))
            slider.pack(fill="x", pady=(12, 20))
            self.sliders = getattr(self, "sliders", {})
            self.sliders[key] = slider

        self.button(sidebar, "Data & assumptions", self.show_data).pack(fill="x", pady=(12, 8))

        headline = tk.Frame(main, bg=NAVY)
        headline.pack(fill="x", pady=(0, 16))
        self.price_label = self.label(headline, size=20)
        self.price_label.pack(side="left")
        self.return_label = self.label(headline, size=20)
        self.return_label.configure(bg=CYAN, fg=NAVY, padx=12, pady=8)
        self.return_label.pack(side="right")
        tk.Frame(main, bg=LINE, height=1).pack(fill="x")
        scenario_row = tk.Frame(main, bg=NAVY)
        scenario_row.pack(fill="x", pady=14)
        self.scenario_buttons = {}
        for name in self.scenarios:
            button = self.button(scenario_row, name, lambda selected=name: self.select_scenario(selected))
            button.pack(side="left", fill="x", expand=True, padx=(0, 8))
            self.scenario_buttons[name] = button
        chart_title = tk.Frame(main, bg=NAVY)
        chart_title.pack(fill="x")
        self.label(chart_title, "Annualised return", 21).pack(side="left")
        self.assumptions_label = self.label(chart_title, size=13, color=MUTED)
        self.assumptions_label.pack(side="right")
        self.figure = Figure(figsize=(9, 5), dpi=100, facecolor=NAVY)
        self.canvas = FigureCanvasTkAgg(self.figure, master=main)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.canvas.mpl_connect("figure_leave_event", self.on_hover)
        self.label(main, "Hypothetical, not guaranteed · Shares unchanged · Excludes dividends",
                   10, MUTED).pack(anchor="w", pady=(0, 8))
        footer = tk.Frame(main, bg=NAVY)
        footer.pack(fill="x")
        self.button(footer, "Compare scenarios", self.show_comparison).pack(side="left")
        self.status = self.label(footer, "", 9, MUTED, wraplength=600, justify="right")
        self.status.pack(side="right")

    def assumptions(self):
        return {key: variable.get() for key, variable in self.variables.items()}

    def move_slider(self, key, value, step):
        if self.loading:
            return
        self.variables[key].set(int(round(float(value) / step) * step))
        if self.redraw_job:
            self.root.after_cancel(self.redraw_job)
        self.redraw_job = self.root.after(60, self.refresh)

    def select_scenario(self, name):
        self.active = name
        self.loading = True
        for key, value in self.scenarios[name].items():
            self.variables[key].set(value)
            self.sliders[key].set(value)
        self.loading = False
        self.refresh()

    def change_company(self, _event=None, force=False):
        """Download off the UI thread, so sliders and the window stay responsive."""
        company = self.company.get()
        self.request_id += 1
        request_id = self.request_id
        if not self.data or self.data["company"] != company:
            self.data = None
        self.source_badge.configure(text="LOADING YAHOO DATA", fg=MUTED)
        self.source_summary.configure(text=f'{COMPANIES[company]} · USD listing\nLoading reported financial data…')
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
                    else:
                        self.source_summary.configure(text=f'{COMPANIES[self.company.get()]} · USD\nNo figures loaded. Use Data & assumptions.')
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
        data = self.data
        self.source_summary.configure(text=(
            f'{data["ticker"]} · USD\nQuote: {data["quote_date"]}\n'
            f'Annual revenue: ${data["revenue"] / 1e9:,.2f}bn\nFY ended: {data["revenue_date"]}\n'
            f'Equivalent shares: {data["shares"] / 1e9:,.3f}bn'))

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
        if self.redraw_job:
            self.root.after_cancel(self.redraw_job)
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()

    def refresh(self):
        if self.redraw_job:
            self.root.after_cancel(self.redraw_job)
            self.redraw_job = None
        assumptions = self.assumptions()
        self.scenarios[self.active] = assumptions.copy()
        for key, label in self.slider_labels.items():
            suffix = "%" if key in ("growth", "margin") else "×" if key == "pe" else ""
            label.configure(text=f"{assumptions[key]}{suffix}")
        if self.data is None:
            self.price_label.configure(text="Quote —  →  Future —")
            self.return_label.configure(text="— / year", bg=LINE, fg=WHITE)
            for name, button in self.scenario_buttons.items():
                button.configure(text=name, bg=CYAN if name == self.active else NAVY,
                                 fg=NAVY if name == self.active else WHITE)
            self.figure.clear()
            self.axes = self.figure.add_subplot(111)
            self.axes.set_facecolor(NAVY)
            self.axes.axis("off")
            self.axes.text(.5, .5, "Choose a company · Load data to explore", transform=self.axes.transAxes,
                           color=MUTED, ha="center", fontsize=16)
            self.canvas.draw_idle()
            return
        outcome = value_for(self.data, assumptions)
        self.price_label.configure(text=f'Quote ${self.data["current_price"]:,.2f}  →  Future ${outcome["price"]:,.2f}')
        self.return_label.configure(text=f'{outcome["annual_return"]:+.1%} / year', fg=NAVY,
                                    bg=CYAN if outcome["annual_return"] >= 0 else "#ffb080")
        self.assumptions_label.configure(text=f'P/E {assumptions["pe"]}× · {assumptions["years"]} years')
        for name, button in self.scenario_buttons.items():
            result = value_for(self.data, self.scenarios[name])
            button.configure(text=f'{name} {result["annual_return"]:+.1%}',
                             bg=CYAN if name == self.active else NAVY,
                             fg=NAVY if name == self.active else WHITE)
        self.draw_heatmap(assumptions)

    def draw_heatmap(self, assumptions):
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=.10, right=.92, bottom=.15, top=.97)
        self.growth_values, self.margin_values, self.grid_results = heatmap_data(self.data, assumptions)
        returns = [[result["annual_return"] * 100 for result in row] for row in self.grid_results]
        limit = max(15, max(abs(value) for row in returns for value in row))
        colours = LinearSegmentedColormap.from_list("returns", ["#ff9855", "#edf0ec", "#007d75"])
        image = self.axes.imshow(returns, origin="lower", aspect="auto", cmap=colours,
                                 norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
        self.axes.set_xticks(range(len(self.growth_values)), self.growth_values)
        self.axes.set_yticks(range(len(self.margin_values)), self.margin_values)
        self.axes.set_xticks([i - .5 for i in range(1, len(self.growth_values))], minor=True)
        self.axes.set_yticks([i - .5 for i in range(1, len(self.margin_values))], minor=True)
        self.axes.grid(which="minor", color=NAVY, linewidth=.6, alpha=.3)
        self.axes.tick_params(which="minor", length=0)
        self.axes.set_xlabel("Revenue growth (%)", color=WHITE, fontsize=13, labelpad=12)
        self.axes.set_ylabel("Net margin (%)", color=WHITE, fontsize=13, labelpad=12)
        self.axes.tick_params(colors=WHITE, labelsize=12)
        for spine in self.axes.spines.values():
            spine.set_visible(False)
        for row, values in enumerate(returns):
            for column, value in enumerate(values):
                self.axes.text(column, row, f"{value:.1f}%", ha="center", va="center",
                               fontsize=15, fontweight="bold", color=WHITE if value > limit * .65 else NAVY)
        x = self.growth_values.index(assumptions["growth"])
        y = self.margin_values.index(assumptions["margin"])
        self.axes.add_patch(Rectangle((x - .49, y - .49), .98, .98, fill=False,
                                      edgecolor="white", linewidth=3))
        colourbar = self.figure.colorbar(image, ax=self.axes, fraction=.035, pad=.04, format="%g%%")
        colourbar.ax.tick_params(colors=WHITE, labelsize=10)
        colourbar.outline.set_edgecolor(LINE)
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
        self.label(window, "Select a scenario, then use the sliders to edit it.", 11, MUTED).pack(padx=18, pady=12)
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
        # A modal snapshot cannot become stale while sliders are being edited.
        window.transient(self.root)
        window.grab_set()


if __name__ == "__main__":
    root = tk.Tk()
    panel = ValuationPanel(root)
    root.mainloop()
