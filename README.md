# AI Boom Dashboard
## Project Overview

The **AI Boom Dashboard** is an interactive, four-panel financial analysis application built using Python. It investigates the performance, investment activity, historical comparisons and potential valuations of major companies associated with the artificial intelligence (AI) industry.

The dashboard combines historical stock market data, company financial statements and financial modelling techniques to help users explore the relationship between AI investment, corporate performance and market valuation.

The application is developed using **Python, Dash and Plotly**, with financial data obtained from **Yahoo Finance** through the `yfinance` library.

### Project Objectives

The dashboard aims to:

- Analyse the historical stock performance of major AI-related companies.
- Investigate the relationship between capital expenditure, revenue and market capitalisation.
- Compare the recent AI investment boom with the historical dot-com bubble.
- Evaluate hypothetical future stock valuations under different financial assumptions.
- Present financial information through accessible, interactive visualisations.

### Dashboard Features

| Panel | Title | Main Function |
|---|---|---|
| **1** | AI Stock Performance | Compares historical stock returns using interactive 3D charts. |
| **2** | Capital Investment vs Revenue | Examines relationships between company spending, revenue and market value. |
| **3** | AI Boom vs Dot-Com Bubble | Compares historical market performance across two technology investment periods. |
| **4** | AI Valuation Simulator | Estimates hypothetical share prices using financial assumptions and scenario analysis. |

The dashboard includes a shared company selection interface, individual panel controls, interactive charts and separate expanded views for detailed analysis.

---

# 1. System Requirements

Before running the dashboard, ensure that the following software and resources are available.

| Requirement | Description |
|---|---|
| **Python 3** | Required to execute the dashboard application. Python 3.14 was used during development. |
| **Internet Connection** | Required to retrieve financial data from Yahoo Finance. |
| **Web Browser** | Google Chrome, Microsoft Edge, Firefox or another modern browser. |
| **Terminal** | Required to install Python libraries and launch the application. |
| **Visual Studio Code** | Recommended development environment, although not compulsory. |

> **Note:** The dashboard runs locally on the user's computer. It does not require a separate public hosting service to operate.

## 1.1 Python Installation

If Python is not already installed:

1. Visit the official Python website: https://www.python.org/downloads/
2. Download a compatible Python 3 version.
3. Run the installation file.
4. Complete the installation instructions.
5. Open a new terminal to verify that Python has been installed correctly.

On Windows, it may be necessary to enable the Python PATH option during installation.

### Verify Python Installation

Open **Windows PowerShell**, **Command Prompt**, or the terminal in VS Code.

Enter:

```powershell
python --version
```

A successful installation should return a Python version, for example:

```text
Python 3.14.3
```

If the command is not recognised on Windows, try:

```powershell
py --version
```

For macOS or Linux, the equivalent command may be:

```bash
python3 --version
```

If the terminal displays a valid Python 3 version, proceed to the next section.

---

# 2. Project Structure

The dashboard is organised into a main Python application, separate panel modules and supporting files.

The main application is contained in the **`Group_Project`** directory.

## 2.1 Directory Structure

```text
Group_Project/
│
├── app.py
│
├── README.md
│
├── requirements.txt
│
├── panels/
│   │
│   ├── __init__.py
│   ├── dashboard_shared.py
│   ├── panel1_stock.py
│   ├── panel2_valuation.py
│   ├── panel3_dotcom.py
│   ├── panel4_simulator.py
│   │
│   └── panel3_assets/
│       └── panel3.css
│
└── tests/
    │
    ├── test_dashboard_integration.py
    └── verify_real_financials.py
```

## 2.2 Explanation of Project Files

| File / Folder | Purpose |
|---|---|
| `app.py` | Main application file responsible for launching and integrating the dashboard. |
| `README.md` | Project documentation and installation instructions. |
| `requirements.txt` | Lists the external Python libraries required by the application. |
| `panels/` | Contains the Python modules responsible for individual dashboard components. |
| `dashboard_shared.py` | Stores shared configuration and styling information used across panels. |
| `panel1_stock.py` | Contains functionality related to historical stock performance. |
| `panel2_valuation.py` | Contains the capital investment and revenue analysis functionality. |
| `panel3_dotcom.py` | Contains the AI boom and dot-com bubble comparison. |
| `panel4_simulator.py` | Contains the interactive financial valuation simulator. |
| `panel3_assets/` | Contains styling resources associated with Panel 3. |
| `tests/` | Contains scripts used for testing and verification. |
---

# 3. Installation and Setup
## Step 1 — Extract the Project ZIP

1. Download and save the project ZIP file to your computer.
2. Locate the downloaded ZIP file.
3. Right-click the file and select **Extract All** (Windows), or double-click the file (macOS).
4. Open the extracted project folder.
5. Locate the `Group_Project` directory containing `app.py` and `panels/`.

> **Important:** Do not move `app.py` out of the `Group_Project` directory or separate it from the `panels/` folder. The application imports code from this directory.

## Step 2 — Open the Project in VS Code

**Visual Studio Code is recommended for running the dashboard.**

1. Launch **Visual Studio Code**.
2. Select **File → Open Folder**.
3. Navigate to the extracted `Group_Project` directory.
4. Select **Open Folder**.
5. In the menu bar, select **Terminal → New Terminal**.
6. Confirm that the terminal is located inside `Group_Project`.

### Verify Your Working Directory

Enter the following command:

```powershell
pwd
```

This displays your current working directory.

Next, enter:

```powershell
ls
```

You should see files and folders including:

- `app.py` — Main application entry point.
- `panels/` — Dashboard modules.
- `tests/` — Project testing files.
- `requirements.txt` — Required dependencies.
- `README.md` — Project documentation.

If these files are not visible, navigate to the correct directory using the `cd` command.

### Alternative: Running Without VS Code

The application can also be launched using:

- Windows PowerShell
- Windows Command Prompt
- macOS Terminal
- Linux Terminal

Navigate to the directory containing `app.py` before continuing.

---
## Step 3 — Create a Virtual Environment

Creating a Python virtual environment is recommended.

A virtual environment isolates the project's Python packages from other applications installed on the same computer.

This reduces the likelihood of dependency conflicts.

### Windows (PowerShell)

From inside the `Group_Project` directory, enter:

```powershell
python -m venv .venv
```

This creates a new virtual environment called `.venv`.

Activate it using:

```powershell
.\.venv\Scripts\Activate.ps1
```

When activated successfully, the terminal prompt should usually display:

```text
(.venv)
```

This indicates that the virtual environment is active.

### macOS / Linux

Create the environment:

```bash
python3 -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

### If Windows Blocks Activation

Some Windows installations prevent PowerShell scripts from running.

If this occurs, the virtual environment can be used without activating it.

For example:

```powershell
.\.venv\Scripts\python.exe -m pip --version
```

Subsequent installation and launch commands can also be run using this executable.

---

## Step 4 — Install the Required Python Libraries

The dashboard depends on several external Python packages.

These libraries must be installed before the application can run.

### Required Libraries

| Library | Purpose |
|---|---|
| **Dash** | Creates the interactive web application and user controls. |
| **Plotly** | Generates interactive financial charts and visualisations. |
| **yfinance** | Retrieves financial data from Yahoo Finance. |
| **pandas** | Organises, cleans and analyses financial datasets. |
| **NumPy** | Performs numerical and statistical calculations. |
| **Flask** | Provides the underlying web server used by Dash. |
| **Matplotlib** | Provides additional Python plotting functionality imported by the application. |

### Recommended Installation Method
The complete dependency list is:

```text
dash
plotly
yfinance
pandas
numpy
flask
matplotlib
```
### Installing Dependencies

All required Python libraries are listed in the `requirements.txt` file.

With the virtual environment activated, run the following command from the `Group_Project` directory:

```powershell
python -m pip install -r requirements.txt
```

This automatically installs the packages necessary to run the dashboard.

To confirm that the dependencies are installed correctly, run:

```powershell
python -c "import dash, plotly, yfinance, pandas, numpy, flask, matplotlib; print('All dependencies installed successfully')"
```

If successful, the terminal will display:

```text
All dependencies installed successfully
```

The application is now ready to launch.

---

# 4. Launching the Dashboard

## Step 5 — Start the Application

Ensure that the terminal is open in the `Group_Project` directory.

Run the following command:

```powershell
python app.py
```

This executes the main Python application and starts the local Dash web server.

The dashboard runs on **port 8050** by default.

## Step 6 — Open the Dashboard

Open a web browser such as Google Chrome, Microsoft Edge or Firefox.

Enter the following address:

**http://127.0.0.1:8050**

The **AI Boom Dashboard** should open, displaying the four interactive financial analysis panels.

An internet connection is required to retrieve financial information from Yahoo Finance. Some charts may take longer to load initially while the data is being retrieved and processed.


## Step 7 — Verify the Dashboard

Once the application has loaded, confirm that:

1. All four dashboard panels are visible.
2. The company selection controls respond correctly.
3. Charts update when their respective controls are changed.
4. Individual panels can be opened in expanded views.
5. Financial data and calculations are displayed where the underlying information is available.

If these features operate correctly, the dashboard is ready to use.

## Step 8 — Stop the Application

To stop the dashboard, return to the terminal and press:

```text
Ctrl + C
```

This terminates the local web server.

To restart the application, run:

```powershell
python app.py
```

### Running on a Different Port

If port 8050 is already in use, the dashboard can be launched using another port.

For example:

```powershell
python app.py --port 8051
```

The dashboard can then be accessed at:

**http://127.0.0.1:8051**

---

# 5. Using the Dashboard

The dashboard provides an interactive environment for analysing AI-related companies, historical market performance and financial valuation scenarios.

The main interface displays four panels, each examining a different aspect of the AI investment boom.

## 5.1 Dashboard Navigation

Users can access all four panels from the main dashboard overview.

Each panel can also be opened in an expanded view, providing additional analytical controls and more detailed financial visualisations.

The dashboard consists of:

| Panel | Analysis | Primary Visualisation |
|---|---|---|
| **Panel 1** | AI Stock Performance | Interactive 3D stock chart |
| **Panel 2** | Capital Investment vs Revenue | Bubble chart and financial data table |
| **Panel 3** | AI Boom vs Dot-Com Bubble | Historical market comparison |
| **Panel 4** | AI Valuation Simulator | Valuation scenarios and sensitivity heatmap |

## 5.2 Company Selection

The shared dashboard header allows users to select the AI-related companies included in the analysis.

The application supports ten companies:

| Company | Stock Ticker |
|---|---|
| NVIDIA | `NVDA` |
| Microsoft | `MSFT` |
| Alphabet | `GOOGL` |
| Amazon | `AMZN` |
| Meta Platforms | `META` |
| Advanced Micro Devices | `AMD` |
| Taiwan Semiconductor Manufacturing Company | `TSM` |
| Oracle | `ORCL` |
| Palantir Technologies | `PLTR` |
| ASML Holding | `ASML` |

### How to Select Companies

1. Locate the company selector in the dashboard header.
2. Select the companies to include in the analysis.
3. Deselect companies that are not required.
4. Allow the relevant charts to update.

The shared selection applies to **Panels 1, 2 and 3**.

**Panel 4 operates independently**, with its own company selection and valuation controls.

---
# 6. Dashboard Panels

Each of the four panels examines a different aspect of the AI investment boom. Users can open individual panels for expanded visualisations and additional controls.

## 6.1 Panel 1 — AI Stock Performance

**Purpose:** Compare the historical stock performance and risk characteristics of major AI-related companies against wider market benchmarks.

**Features:**
- Interactive 3D stock performance chart.
- Historical periods of 1, 3, 5 or 10 years.
- Optional S&P 500 (`SPY`) and Nasdaq-100 (`QQQ`) benchmarks.
- Adjustable chart detail (Daily, Every 5 Days or Every 20 Days).
- Total return, annualised volatility and maximum drawdown statistics.

**How to use:**
1. Select companies using the shared dashboard header.
2. Choose a historical period and optional market benchmarks.
3. Adjust the chart detail and explore the interactive 3D visualisation.
4. Examine the accompanying performance and risk statistics.

Stock prices are indexed to a common starting value of **100**, allowing meaningful percentage performance comparisons between companies with different share prices.

---
## 6.2 Panel 2 — Capital Investment vs Revenue

**Purpose:** Examine the relationship between capital expenditure, annual revenue and market capitalisation among AI-related companies.

**Features:**
- Interactive bubble chart and financial data table.
- Capital expenditure (X-axis), revenue (Y-axis) and market capitalisation (bubble size).
- Linear regression and R² analysis.
- Company-level financial information from Yahoo Finance.

**How to use:**
1. Select the companies to analyse.
2. Examine the bubble chart and hover over companies for financial details.
3. Review the regression line and R² statistic.
4. Switch to the table view to inspect the underlying financial values.

**Interpretation:** The regression illustrates the statistical relationship between capital expenditure and revenue. R² measures the strength of the fitted linear relationship within the selected sample, but does not establish causation.

---
## 6.3 Panel 3 — AI Boom vs Dot-Com Bubble

**Purpose:** Compare the recent AI investment boom with the historical dot-com market cycle.

**Features:**
- Historical comparison of the NASDAQ Composite (1995–2003) and selected AI-related equities (2023 onwards).
- Indexed price performance, returns and maximum drawdowns.
- Interactive historical charts.
- Illustrative Bull, Base and Bear scenarios.

**How to use:**
1. Select the AI companies to include in the comparison.
2. Examine the indexed performance of both historical periods.
3. Compare returns, drawdowns and market behaviour.
4. Explore the hypothetical future scenario paths.

**Interpretation:** The comparison highlights similarities and differences between two technology-driven investment periods. The future scenarios are illustrative assumptions, not predictions of actual market performance.

---

## 6.4 Panel 4 — AI Valuation Simulator

**Purpose:** Estimate hypothetical future share prices and investment returns under different financial assumptions.

**Features:**
- Company-specific valuation modelling.
- Bear, Base and Bull scenarios.
- Revenue growth, net profit margin and P/E multiple assumptions.
- Estimated future share prices and annualised returns.
- Interactive valuation sensitivity heatmap.

**How to use:**
1. Select a company from the panel's dropdown.
2. Review its financial information.
3. Choose a Bear, Base or Bull scenario.
4. Compare estimated future share prices and annualised returns.
5. Explore how changes in revenue growth and profit margins affect the valuation.

The simulator uses a simplified earnings-multiple valuation model. All outputs depend on the selected assumptions and should not be treated as guaranteed investment outcomes.

---
# 7. Technical Implementation

## 7.1 Application Architecture

The dashboard is built using **Python, Dash and Plotly**, with individual modules for the four financial panels.

- **`app.py`** integrates the panels, manages navigation and starts the local web server.
- **`panels/`** contains the individual analytical modules.
- **`dashboard_shared.py`** provides shared company information, styling and configuration.
- **Dash callbacks** update charts and financial outputs when users change controls.
- **pandas and NumPy** process financial data and perform numerical calculations.
- **Plotly** generates the interactive visualisations.

## 7.2 Data Retrieval

Financial information is retrieved from **Yahoo Finance** using the `yfinance` library.

The application uses historical share prices, market capitalisation, financial statements and other available company data. Data is processed before being used in calculations and visualisations.

An in-memory cache stores selected financial data for **15 minutes**, reducing repeated requests and improving performance.

## 7.3 Financial Calculations

The principal calculations used across the dashboard include:

| Calculation | Formula / Method |
|---|---|
| Indexed Price | `(Current Price / Initial Price) × 100` |
| Total Return | `(Final Price / Initial Price − 1) × 100` |
| Annualised Volatility | `Daily Return Standard Deviation × √252 × 100` |
| Maximum Drawdown | Largest percentage decline from a previous peak |
| Linear Regression | `Revenue = Intercept + Slope × Capital Expenditure` |
| Future Revenue | `Current Revenue × (1 + Growth Rate)^Years` |
| Future Net Income | `Future Revenue × Net Profit Margin` |
| Future Share Price | `(Future Net Income / Shares) × P/E Multiple` |
| Annualised Return | `(Future Price / Current Price)^(1 / Years) − 1` |

These calculations support historical analysis, financial comparisons and hypothetical valuation scenarios.

The dashboard is intended for **educational and analytical purposes**, not investment advice. Results depend on external data quality and, where applicable, model assumptions.

---
# 8. Troubleshooting and Testing

## 8.1 Common Issues

| Problem | Solution |
|---|---|
| Python is not recognised | Verify the Python installation using `python --version` or `py --version` on Windows. |
| `ModuleNotFoundError` | Activate the virtual environment and install dependencies using `python -m pip install -r requirements.txt`. |
| `No module named 'panels'` | Ensure `app.py` and `panels/` remain in the same directory. |
| `app.py` cannot be found | Open the terminal in the correct `Group_Project` directory. |
| Dashboard does not load | Confirm the server is running and open `http://127.0.0.1:8050`. |
| Port 8050 is occupied | Run `python app.py --port 8051` and open `http://127.0.0.1:8051`. |
| Financial data is unavailable | Check the internet connection and retry. Some Yahoo Finance data may be temporarily unavailable. |
| Charts are slow to load | Allow time for financial data retrieval and processing; check the terminal for errors. |

## 8.2 Automated Tests

The project includes integration tests in the `tests/` directory.

To execute the automated tests, open a terminal in `Group_Project` and run:

```powershell
python -B -m unittest discover -s tests -p "test_*.py" -v
```

To run the additional financial data verification script:

```powershell
python -B tests/verify_real_financials.py
```

The integration tests help verify application behaviour, while the financial verification script checks aspects of the data-processing functionality.

## 8.3 Manual Verification

After launching the dashboard, confirm that:

- [ ] All four panels display successfully.
- [ ] Company selection updates the relevant panels.
- [ ] Historical periods and chart controls work.
- [ ] The financial data table displays correctly.
- [ ] The historical comparison and valuation scenarios respond to user inputs.
- [ ] No unexpected errors appear in the terminal.
---

# 9. References

The dashboard uses the following data sources, libraries and documentation:

| Resource | Reference |
|---|---|
| Python | [Python Documentation](https://docs.python.org/3/) |
| Dash | [Dash Documentation](https://dash.plotly.com/) |
| Plotly | [Plotly Python Documentation](https://plotly.com/python/) |
| Yahoo Finance | [Yahoo Finance](https://finance.yahoo.com/) |
| yfinance | [yfinance GitHub Repository](https://github.com/ranaroussi/yfinance) |
| pandas | [pandas Documentation](https://pandas.pydata.org/docs/) |
| NumPy | [NumPy Documentation](https://numpy.org/doc/) |
| Flask | [Flask Documentation](https://flask.palletsprojects.com/) |
| Matplotlib | [Matplotlib Documentation](https://matplotlib.org/) |

---