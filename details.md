# AUTOPSY — Market Dislocation Fingerprint Intelligence System
### Full Project Blueprint & Vibe-Coding Implementation Guide
**Hackathon: TechEx Intelligent Enterprise Solutions | Track 4: Data & Intelligence**
**Author: Arka Sarkar**

---

## 0. What You Are Building (Read This First)

AUTOPSY is a real-time market stress intelligence system. It answers one question:

> **"Which historical crisis does the current market structure most resemble — and what happened next?"**

It does this by:
1. Pulling ~40 live market indicators across 5 structural dimensions (liquidity, correlation, volatility, credit, positioning)
2. Computing a "fingerprint vector" that describes the current structural state of markets
3. Comparing that vector against pre-computed fingerprints of 10 historical crises using embedding similarity
4. Running an AI agent (Claude via Anthropic API) that reads the fingerprint match and writes a structured risk narrative
5. Displaying everything on a live dashboard with radar charts, similarity scores, and a "rewind" feature that lets you replay any historical period

This is NOT a price prediction tool. It is a market structure stress detector. That distinction is important for your pitch.

---

## 1. Project Structure (Create Exactly This)

```
autopsy/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── data/
│   ├── pipeline.py          # Fetches all live + historical data
│   ├── indicators.py        # Computes all 40 indicators from raw data
│   └── crisis_library.py   # Defines the 10 historical crisis windows
│
├── fingerprint/
│   ├── engine.py            # Rolls features, normalizes, builds vectors
│   └── embedding.py         # UMAP embedding + similarity search
│
├── agent/
│   └── analyst.py           # Claude API call, prompt, structured output
│
├── dashboard/
│   ├── app.py               # Main Streamlit app entry point
│   ├── charts.py            # All Plotly chart functions
│   └── state.py             # Session state management
│
└── tests/
    └── test_pipeline.py     # Basic smoke tests
```

---

## 2. Environment Setup

### 2.1 Python Version
Use Python 3.11+

### 2.2 requirements.txt (Copy This Exactly)
```
streamlit==1.35.0
plotly==5.22.0
pandas==2.2.2
numpy==1.26.4
scipy==1.13.0
scikit-learn==1.5.0
umap-learn==0.5.6
yfinance==0.2.40
fredapi==0.5.2
anthropic==0.28.0
python-dotenv==1.0.1
requests==2.32.3
```

### 2.3 .env.example (Copy This)
```
FRED_API_KEY=your_fred_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### 2.4 How to Get API Keys
- **FRED API Key**: Go to https://fred.stlouisfed.org/docs/api/api_key.html — free, instant
- **Anthropic API Key**: Go to https://console.anthropic.com — free tier available

### 2.5 .gitignore
```
.env
__pycache__/
*.pyc
.DS_Store
venv/
*.egg-info/
dist/
build/
```

---

## 3. Data Layer — `data/pipeline.py`

### 3.1 What This File Does
Fetches all raw market data from two sources:
- **FRED** (Federal Reserve Economic Data) — for credit spreads, funding markets, Treasury data
- **yfinance** — for equity prices, VIX term structure, FX rates, commodities

### 3.2 FRED Series to Fetch
These are the exact FRED series IDs. Fetch daily data.

| Variable Name | FRED Series ID | What It Measures |
|---|---|---|
| ted_spread | TEDRATE | TED spread (3M LIBOR - 3M T-Bill) — funding stress |
| ois_spread | T10Y2Y | 10Y-2Y Treasury spread — yield curve shape |
| investment_grade_spread | BAMLC0A0CM | IG corporate credit spread |
| high_yield_spread | BAMLH0A0HYM2 | HY corporate credit spread |
| commercial_paper_spread | CPFF | Commercial paper funding spread |
| vix | VIXCLS | VIX (implied volatility) |
| move_index | (use yfinance proxy) | Bond market volatility |

### 3.3 yfinance Tickers to Fetch
Fetch daily OHLCV data for all of these:

| Variable Name | Ticker | What It Measures |
|---|---|---|
| spy | SPY | S&P 500 equity |
| qqq | QQQ | Nasdaq 100 |
| ief | IEF | 7-10Y Treasury bonds |
| lqd | LQD | Investment grade bonds |
| hyg | HYG | High yield bonds |
| gld | GLD | Gold |
| uso | USO | Oil |
| uup | UUP | US Dollar index |
| eem | EEM | Emerging markets equity |
| xlf | XLF | Financials sector |
| xle | XLE | Energy sector |
| xlu | XLU | Utilities sector |
| vix_front | ^VIX | VIX (front month) |
| vix_3m | ^VIX3M | VIX 3-month |
| vix_9d | ^VIX9D | VIX 9-day |
| eur_usd | EURUSD=X | EUR/USD exchange rate |
| jpy_usd | JPY=X | Japanese Yen |
| chf_usd | CHF=X | Swiss Franc (safe haven) |

### 3.4 pipeline.py Full Implementation

```python
# data/pipeline.py

import os
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY")

# ── FRED Series IDs ──────────────────────────────────────────────────────────
FRED_SERIES = {
    "ted_spread":              "TEDRATE",
    "yield_curve_10y2y":       "T10Y2Y",
    "yield_curve_10y3m":       "T10Y3M",
    "ig_credit_spread":        "BAMLC0A0CM",
    "hy_credit_spread":        "BAMLH0A0HYM2",
    "ig_hy_ratio":             None,   # computed: hy / ig
    "vix_fred":                "VIXCLS",
}

# ── yfinance Tickers ──────────────────────────────────────────────────────────
YF_TICKERS = {
    "spy":      "SPY",
    "qqq":      "QQQ",
    "ief":      "IEF",
    "lqd":      "LQD",
    "hyg":      "HYG",
    "gld":      "GLD",
    "uso":      "USO",
    "uup":      "UUP",
    "eem":      "EEM",
    "xlf":      "XLF",
    "xle":      "XLE",
    "xlu":      "XLU",
    "vix":      "^VIX",
    "vix_3m":   "^VIX3M",
    "vix_9d":   "^VIX9D",
    "eurusd":   "EURUSD=X",
    "jpyusd":   "JPY=X",
    "chfusd":   "CHF=X",
}


def fetch_fred_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches all FRED series between start_date and end_date.
    Returns a single DataFrame with dates as index, series names as columns.
    Missing values forward-filled then backward-filled (FRED data has gaps on weekends).
    """
    fred = Fred(api_key=FRED_API_KEY)
    frames = {}

    for name, series_id in FRED_SERIES.items():
        if series_id is None:
            continue
        try:
            s = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
            frames[name] = s
        except Exception as e:
            print(f"[FRED] Failed to fetch {name} ({series_id}): {e}")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df = df.resample("B").last()  # business days only
    df = df.ffill().bfill()

    # Compute derived series
    if "ig_credit_spread" in df.columns and "hy_credit_spread" in df.columns:
        df["ig_hy_ratio"] = df["hy_credit_spread"] / df["ig_credit_spread"].replace(0, np.nan)

    return df


def fetch_yfinance_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches closing prices for all yfinance tickers.
    Returns DataFrame: dates as index, ticker names as columns (using our friendly names).
    """
    tickers = list(YF_TICKERS.values())
    raw = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)

    # Extract close prices only
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]]

    # Rename columns to friendly names
    reverse_map = {v: k for k, v in YF_TICKERS.items()}
    close.columns = [reverse_map.get(col, col) for col in close.columns]
    close = close.ffill().bfill()

    return close


def fetch_all_data(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Master fetch function. Returns a single combined DataFrame.
    Default: last 5 years of data.
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.today() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")

    print(f"[Pipeline] Fetching data from {start_date} to {end_date}...")
    fred_df = fetch_fred_data(start_date, end_date)
    yf_df = fetch_yfinance_data(start_date, end_date)

    # Align on business day index
    combined = pd.concat([fred_df, yf_df], axis=1)
    combined = combined.resample("B").last().ffill().bfill()

    print(f"[Pipeline] Fetched {len(combined)} rows, {len(combined.columns)} columns.")
    return combined


def fetch_live_snapshot() -> pd.DataFrame:
    """
    Fetches the most recent 252 trading days (1 year) of data.
    Used by the dashboard for the live fingerprint.
    Returns the same format as fetch_all_data().
    """
    return fetch_all_data(
        start_date=(datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d"),
        end_date=datetime.today().strftime("%Y-%m-%d")
    )
```

---

## 4. Indicator Engine — `data/indicators.py`

### 4.1 What This File Does
Takes the raw price/spread DataFrame from pipeline.py and computes the 40 structured indicators across 5 dimensions. Each indicator is a rolling statistic (velocity, z-score, or ratio) that captures stress in a specific dimension.

### 4.2 The 5 Dimensions and Their Indicators

**DIMENSION 1: LIQUIDITY (8 indicators)**
- `ted_spread_level` — raw TED spread value
- `ted_spread_velocity` — 10-day rate of change of TED spread
- `hy_ig_spread_ratio` — HY spread / IG spread (credit quality bifurcation)
- `hy_spread_velocity` — 10-day rate of change of HY spread
- `ig_spread_velocity` — 10-day rate of change of IG spread
- `yield_curve_level` — 10Y-2Y spread level
- `yield_curve_velocity` — 10-day rate of change of yield curve
- `bond_equity_correlation` — 21-day rolling correlation between IEF and SPY returns

**DIMENSION 2: VOLATILITY (8 indicators)**
- `vix_level` — raw VIX value
- `vix_velocity` — 5-day rate of change of VIX
- `vix_term_structure` — VIX9D / VIX3M ratio (inversion = front-end stress)
- `vix_z_score` — VIX vs its own 252-day rolling mean/std
- `realized_vol_spy` — 21-day realized volatility of SPY
- `implied_realized_vol_gap` — VIX - realized_vol_spy (fear premium)
- `vol_of_vol` — 21-day rolling std of daily VIX changes
- `cross_asset_vol_spike` — max(VIX velocity, bond vol velocity) normalized

**DIMENSION 3: CORRELATION (8 indicators)**
- `equity_bond_corr` — 21-day rolling corr(SPY returns, IEF returns)
- `equity_gold_corr` — 21-day rolling corr(SPY returns, GLD returns)
- `equity_hy_corr` — 21-day rolling corr(SPY returns, HYG returns)
- `safe_haven_demand` — 21-day cumulative return of GLD + JPY + CHF vs SPY
- `cross_sector_correlation` — avg pairwise corr among XLF, XLE, XLU, QQQ over 21 days
- `em_dm_divergence` — 21-day return spread: SPY - EEM
- `dollar_stress_signal` — UUP 10-day momentum (dollar surges in crises)
- `correlation_breakdown_score` — std of pairwise sector correlations (breakdown = high std)

**DIMENSION 4: CREDIT (8 indicators)**
- `hy_spread_level` — raw HY OAS spread
- `ig_spread_level` — raw IG OAS spread
- `hy_ig_divergence_velocity` — 10-day change in (HY - IG) spread gap
- `lqd_hyg_return_spread` — 21-day return: LQD - HYG (flight to quality)
- `xlf_spy_relative` — XLF vs SPY 21-day relative performance (financials leading risk)
- `credit_equity_dislocation` — HYG 21-day return vs SPY 21-day return (should co-move)
- `ig_spread_z_score` — IG spread vs its 252-day rolling mean/std
- `hy_spread_z_score` — HY spread vs its 252-day rolling mean/std

**DIMENSION 5: POSITIONING / FLOWS (8 indicators)**
- `gold_flow_signal` — GLD 10-day momentum (safe haven buying)
- `jpy_flow_signal` — JPY 10-day momentum (yen strengthening = risk-off)
- `chf_flow_signal` — CHF 10-day momentum (CHF strengthening = risk-off)
- `utilities_relative` — XLU vs SPY 21-day relative (defensive rotation)
- `energy_relative` — XLE vs SPY 21-day relative (commodity stress proxy)
- `oil_velocity` — USO 10-day momentum
- `em_outflow_signal` — EEM 10-day momentum (EM outflows in crises)
- `risk_off_composite` — equal-weight composite of gold + yen + chf + utilities signals

### 4.3 indicators.py Full Implementation

```python
# data/indicators.py

import pandas as pd
import numpy as np


def compute_velocity(series: pd.Series, window: int = 10) -> pd.Series:
    """Rate of change over window days, as percentage."""
    return series.pct_change(periods=window) * 100


def compute_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """Rolling z-score against a trailing window."""
    mean = series.rolling(window=window, min_periods=window // 2).mean()
    std = series.rolling(window=window, min_periods=window // 2).std()
    return (series - mean) / std.replace(0, np.nan)


def compute_rolling_correlation(s1: pd.Series, s2: pd.Series, window: int = 21) -> pd.Series:
    """Rolling correlation between two return series."""
    r1 = s1.pct_change()
    r2 = s2.pct_change()
    return r1.rolling(window=window, min_periods=window // 2).corr(r2)


def compute_realized_vol(series: pd.Series, window: int = 21) -> pd.Series:
    """Annualized realized volatility."""
    log_returns = np.log(series / series.shift(1))
    return log_returns.rolling(window=window, min_periods=window // 2).std() * np.sqrt(252) * 100


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw combined DataFrame from pipeline.py.
    Returns a new DataFrame with all 40 indicator columns.
    Index is the same as input (business days).
    NaN-filled with forward fill, then 0 for any remaining gaps.
    """
    ind = pd.DataFrame(index=df.index)

    # ── DIMENSION 1: LIQUIDITY ────────────────────────────────────────────────
    if "ted_spread" in df.columns:
        ind["ted_spread_level"] = df["ted_spread"]
        ind["ted_spread_velocity"] = compute_velocity(df["ted_spread"], 10)
    else:
        ind["ted_spread_level"] = 0.0
        ind["ted_spread_velocity"] = 0.0

    if "hy_credit_spread" in df.columns and "ig_credit_spread" in df.columns:
        ind["hy_ig_spread_ratio"] = df["hy_credit_spread"] / df["ig_credit_spread"].replace(0, np.nan)
        ind["hy_spread_velocity"] = compute_velocity(df["hy_credit_spread"], 10)
        ind["ig_spread_velocity"] = compute_velocity(df["ig_credit_spread"], 10)
        ind["hy_spread_level"] = df["hy_credit_spread"]
        ind["ig_spread_level"] = df["ig_credit_spread"]
        ind["hy_spread_z_score"] = compute_zscore(df["hy_credit_spread"])
        ind["ig_spread_z_score"] = compute_zscore(df["ig_credit_spread"])
    else:
        for col in ["hy_ig_spread_ratio", "hy_spread_velocity", "ig_spread_velocity",
                    "hy_spread_level", "ig_spread_level", "hy_spread_z_score", "ig_spread_z_score"]:
            ind[col] = 0.0

    if "yield_curve_10y2y" in df.columns:
        ind["yield_curve_level"] = df["yield_curve_10y2y"]
        ind["yield_curve_velocity"] = compute_velocity(df["yield_curve_10y2y"], 10)
    else:
        ind["yield_curve_level"] = 0.0
        ind["yield_curve_velocity"] = 0.0

    if "ief" in df.columns and "spy" in df.columns:
        ind["bond_equity_correlation"] = compute_rolling_correlation(df["ief"], df["spy"], 21)
    else:
        ind["bond_equity_correlation"] = 0.0

    # ── DIMENSION 2: VOLATILITY ───────────────────────────────────────────────
    vix_col = "vix" if "vix" in df.columns else "vix_fred"
    if vix_col in df.columns:
        ind["vix_level"] = df[vix_col]
        ind["vix_velocity"] = compute_velocity(df[vix_col], 5)
        ind["vix_z_score"] = compute_zscore(df[vix_col])
        ind["vol_of_vol"] = df[vix_col].diff().rolling(21).std()
    else:
        for col in ["vix_level", "vix_velocity", "vix_z_score", "vol_of_vol"]:
            ind[col] = 0.0

    if "vix_9d" in df.columns and "vix_3m" in df.columns:
        ind["vix_term_structure"] = df["vix_9d"] / df["vix_3m"].replace(0, np.nan)
    else:
        ind["vix_term_structure"] = 1.0

    if "spy" in df.columns:
        ind["realized_vol_spy"] = compute_realized_vol(df["spy"], 21)
    else:
        ind["realized_vol_spy"] = 0.0

    if "vix_level" in ind.columns and "realized_vol_spy" in ind.columns:
        ind["implied_realized_vol_gap"] = ind["vix_level"] - ind["realized_vol_spy"]
    else:
        ind["implied_realized_vol_gap"] = 0.0

    ind["cross_asset_vol_spike"] = np.abs(ind.get("vix_velocity", 0))

    # ── DIMENSION 3: CORRELATION ──────────────────────────────────────────────
    if "spy" in df.columns and "ief" in df.columns:
        ind["equity_bond_corr"] = compute_rolling_correlation(df["spy"], df["ief"], 21)
    else:
        ind["equity_bond_corr"] = 0.0

    if "spy" in df.columns and "gld" in df.columns:
        ind["equity_gold_corr"] = compute_rolling_correlation(df["spy"], df["gld"], 21)
    else:
        ind["equity_gold_corr"] = 0.0

    if "spy" in df.columns and "hyg" in df.columns:
        ind["equity_hy_corr"] = compute_rolling_correlation(df["spy"], df["hyg"], 21)
    else:
        ind["equity_hy_corr"] = 0.0

    # Safe haven demand: cumulative 21-day return of GLD + JPY + CHF vs SPY
    safe_haven_components = []
    for ticker in ["gld", "jpyusd", "chfusd"]:
        if ticker in df.columns:
            safe_haven_components.append(df[ticker].pct_change(21))
    if safe_haven_components and "spy" in df.columns:
        avg_safe = pd.concat(safe_haven_components, axis=1).mean(axis=1)
        ind["safe_haven_demand"] = avg_safe - df["spy"].pct_change(21)
    else:
        ind["safe_haven_demand"] = 0.0

    # Cross-sector correlation
    sector_tickers = [t for t in ["xlf", "xle", "xlu", "qqq"] if t in df.columns]
    if len(sector_tickers) >= 2:
        sector_returns = df[sector_tickers].pct_change()
        rolling_corrs = []
        for i in range(len(sector_tickers)):
            for j in range(i+1, len(sector_tickers)):
                c = sector_returns[sector_tickers[i]].rolling(21).corr(sector_returns[sector_tickers[j]])
                rolling_corrs.append(c)
        ind["cross_sector_correlation"] = pd.concat(rolling_corrs, axis=1).mean(axis=1)
        ind["correlation_breakdown_score"] = pd.concat(rolling_corrs, axis=1).std(axis=1)
    else:
        ind["cross_sector_correlation"] = 0.0
        ind["correlation_breakdown_score"] = 0.0

    if "spy" in df.columns and "eem" in df.columns:
        ind["em_dm_divergence"] = df["spy"].pct_change(21) - df["eem"].pct_change(21)
    else:
        ind["em_dm_divergence"] = 0.0

    if "uup" in df.columns:
        ind["dollar_stress_signal"] = compute_velocity(df["uup"], 10)
    else:
        ind["dollar_stress_signal"] = 0.0

    # ── DIMENSION 4: CREDIT ───────────────────────────────────────────────────
    if "hy_credit_spread" in df.columns and "ig_credit_spread" in df.columns:
        hy_ig_gap = df["hy_credit_spread"] - df["ig_credit_spread"]
        ind["hy_ig_divergence_velocity"] = compute_velocity(hy_ig_gap, 10)
    else:
        ind["hy_ig_divergence_velocity"] = 0.0

    if "lqd" in df.columns and "hyg" in df.columns:
        ind["lqd_hyg_return_spread"] = df["lqd"].pct_change(21) - df["hyg"].pct_change(21)
    else:
        ind["lqd_hyg_return_spread"] = 0.0

    if "xlf" in df.columns and "spy" in df.columns:
        ind["xlf_spy_relative"] = df["xlf"].pct_change(21) - df["spy"].pct_change(21)
    else:
        ind["xlf_spy_relative"] = 0.0

    if "hyg" in df.columns and "spy" in df.columns:
        ind["credit_equity_dislocation"] = df["hyg"].pct_change(21) - df["spy"].pct_change(21)
    else:
        ind["credit_equity_dislocation"] = 0.0

    # ── DIMENSION 5: POSITIONING / FLOWS ─────────────────────────────────────
    flow_signals = {}
    for ticker, col_name in [("gld", "gold_flow_signal"), ("jpyusd", "jpy_flow_signal"),
                              ("chfusd", "chf_flow_signal")]:
        if ticker in df.columns:
            ind[col_name] = compute_velocity(df[ticker], 10)
            flow_signals[col_name] = ind[col_name]
        else:
            ind[col_name] = 0.0

    if "xlu" in df.columns and "spy" in df.columns:
        ind["utilities_relative"] = df["xlu"].pct_change(21) - df["spy"].pct_change(21)
    else:
        ind["utilities_relative"] = 0.0

    if "xle" in df.columns and "spy" in df.columns:
        ind["energy_relative"] = df["xle"].pct_change(21) - df["spy"].pct_change(21)
    else:
        ind["energy_relative"] = 0.0

    if "uso" in df.columns:
        ind["oil_velocity"] = compute_velocity(df["uso"], 10)
    else:
        ind["oil_velocity"] = 0.0

    if "eem" in df.columns:
        ind["em_outflow_signal"] = compute_velocity(df["eem"], 10) * -1  # invert: negative = outflow
    else:
        ind["em_outflow_signal"] = 0.0

    # Risk-off composite
    risk_off_cols = [c for c in ["gold_flow_signal", "jpy_flow_signal", "chf_flow_signal", "utilities_relative"]
                     if c in ind.columns]
    if risk_off_cols:
        ind["risk_off_composite"] = ind[risk_off_cols].mean(axis=1)
    else:
        ind["risk_off_composite"] = 0.0

    # ── FINAL CLEANUP ─────────────────────────────────────────────────────────
    ind = ind.ffill().fillna(0.0)
    ind = ind.replace([np.inf, -np.inf], 0.0)

    return ind


# ── The 40 canonical indicator names, grouped by dimension ──────────────────
INDICATOR_GROUPS = {
    "Liquidity": [
        "ted_spread_level", "ted_spread_velocity", "hy_ig_spread_ratio",
        "hy_spread_velocity", "ig_spread_velocity", "yield_curve_level",
        "yield_curve_velocity", "bond_equity_correlation"
    ],
    "Volatility": [
        "vix_level", "vix_velocity", "vix_term_structure", "vix_z_score",
        "realized_vol_spy", "implied_realized_vol_gap", "vol_of_vol",
        "cross_asset_vol_spike"
    ],
    "Correlation": [
        "equity_bond_corr", "equity_gold_corr", "equity_hy_corr",
        "safe_haven_demand", "cross_sector_correlation", "em_dm_divergence",
        "dollar_stress_signal", "correlation_breakdown_score"
    ],
    "Credit": [
        "hy_spread_level", "ig_spread_level", "hy_ig_divergence_velocity",
        "lqd_hyg_return_spread", "xlf_spy_relative", "credit_equity_dislocation",
        "ig_spread_z_score", "hy_spread_z_score"
    ],
    "Positioning": [
        "gold_flow_signal", "jpy_flow_signal", "chf_flow_signal",
        "utilities_relative", "energy_relative", "oil_velocity",
        "em_outflow_signal", "risk_off_composite"
    ]
}

ALL_INDICATORS = [ind for group in INDICATOR_GROUPS.values() for ind in group]
```

---

## 5. Crisis Library — `data/crisis_library.py`

### 5.1 What This File Does
Defines the 10 historical crisis windows. Each crisis has:
- A name and short description
- A "stress window": the 60-day period BEFORE the main blowup (this is where the fingerprints form)
- A "peak date": the day the crisis peaked/was most visible

The fingerprint engine will extract indicator vectors from the stress window to build the crisis embedding library.

```python
# data/crisis_library.py

CRISIS_LIBRARY = {
    "LTCM_1998": {
        "name": "LTCM Collapse",
        "short": "LTCM 1998",
        "description": (
            "Long-Term Capital Management, a highly leveraged hedge fund, collapsed after "
            "Russian debt default. Required a Fed-orchestrated bailout. Key fingerprint: "
            "basis blowout in theoretically related instruments, liquidity withdrawal from "
            "obscure spreads before equity markets noticed."
        ),
        "stress_start": "1998-06-01",
        "stress_end":   "1998-09-01",
        "peak_date":    "1998-10-08",
        "color":        "#E74C3C",
        "key_signature": "Basis blowouts + liquidity fragmentation"
    },
    "DOTCOM_2000": {
        "name": "Dot-Com Bust",
        "short": "Dot-Com 2000",
        "description": (
            "Technology bubble collapse. NASDAQ fell 78% peak to trough. Key fingerprint: "
            "intra-equity sector divergence (tech vs value), VIX term structure flattening, "
            "without the broad credit market stress seen in GFC."
        ),
        "stress_start": "2000-01-01",
        "stress_end":   "2000-03-10",
        "peak_date":    "2000-03-10",
        "color":        "#E67E22",
        "key_signature": "Intra-equity sector divergence + vol term structure"
    },
    "GFC_2008": {
        "name": "Global Financial Crisis",
        "short": "GFC 2008",
        "description": (
            "Subprime mortgage collapse triggered global banking crisis. Key fingerprint: "
            "TED spread explosion, commercial paper market seizure, everything correlated to 1, "
            "funding markets frozen before equity indices reflected the stress."
        ),
        "stress_start": "2007-08-01",
        "stress_end":   "2008-09-15",
        "peak_date":    "2008-10-10",
        "color":        "#8E44AD",
        "key_signature": "Funding market seizure + universal correlation spike"
    },
    "FLASH_CRASH_2010": {
        "name": "Flash Crash",
        "short": "Flash Crash 2010",
        "description": (
            "May 6, 2010: Dow Jones fell nearly 1000 points in minutes then recovered. "
            "Key fingerprint: pure liquidity microstructure stress without credit signal, "
            "very short duration. Demonstrates positioning-driven stress."
        ),
        "stress_start": "2010-04-15",
        "stress_end":   "2010-05-06",
        "peak_date":    "2010-05-06",
        "color":        "#27AE60",
        "key_signature": "Microstructure-only liquidity collapse, no credit signal"
    },
    "EUROZONE_2011": {
        "name": "Eurozone Debt Crisis",
        "short": "Eurozone 2011",
        "description": (
            "Greek, Italian, Spanish sovereign debt crisis threatened euro breakup. "
            "Key fingerprint: geographic contagion pattern in sovereign credit, EUR/USD stress, "
            "financials leading equity drawdown."
        ),
        "stress_start": "2011-06-01",
        "stress_end":   "2011-08-08",
        "peak_date":    "2011-09-23",
        "color":        "#2980B9",
        "key_signature": "Sovereign credit + financials leading + EUR stress"
    },
    "TAPER_TANTRUM_2013": {
        "name": "Taper Tantrum",
        "short": "Taper Tantrum 2013",
        "description": (
            "Fed signaled QE tapering; bond markets sold off violently. "
            "Key fingerprint: rate-driven cross-asset stress, EM currency selloff, "
            "bond-equity correlation flip, yield curve steepening velocity."
        ),
        "stress_start": "2013-05-01",
        "stress_end":   "2013-06-25",
        "peak_date":    "2013-06-25",
        "color":        "#16A085",
        "key_signature": "Duration selloff + EM outflows + yield curve velocity"
    },
    "CHINA_OIL_2015": {
        "name": "China/Oil Shock",
        "short": "China/Oil 2015",
        "description": (
            "Chinese growth fears + oil collapse triggered global equity selloff. "
            "Key fingerprint: commodity-financial system coupling, EM stress, "
            "energy sector leading, oil-dollar relationship breaking down."
        ),
        "stress_start": "2015-06-01",
        "stress_end":   "2015-08-25",
        "peak_date":    "2015-08-25",
        "color":        "#D35400",
        "key_signature": "Commodity-financial coupling + EM selloff"
    },
    "VOL_SHOCK_2018": {
        "name": "Volmageddon",
        "short": "Volmageddon 2018",
        "description": (
            "February 2018: Short volatility strategies (VIX ETPs) exploded, "
            "triggering forced unwinds. Key fingerprint: VIX futures basis blowout, "
            "positioning-driven without fundamental credit stress, very fast recovery."
        ),
        "stress_start": "2018-01-15",
        "stress_end":   "2018-02-05",
        "peak_date":    "2018-02-05",
        "color":        "#C0392B",
        "key_signature": "VIX futures basis + short-vol unwind + no credit"
    },
    "COVID_2020": {
        "name": "COVID Market Crash",
        "short": "COVID 2020",
        "description": (
            "March 2020: Fastest 30% market crash in history. Key fingerprint: "
            "everything selling simultaneously including gold (margin calls), "
            "correlations going to 1 across ALL assets, dollar surging, "
            "funding markets seizing within days."
        ),
        "stress_start": "2020-02-01",
        "stress_end":   "2020-03-23",
        "peak_date":    "2020-03-23",
        "color":        "#1ABC9C",
        "key_signature": "Universal correlation spike + margin call fingerprint"
    },
    "SVB_2023": {
        "name": "SVB Banking Crisis",
        "short": "SVB 2023",
        "description": (
            "Silicon Valley Bank collapse triggered regional banking crisis. "
            "Key fingerprint: regional bank CDS spike, financials sector stress, "
            "rate sensitivity stress without broad credit market contagion. "
            "Funding-specific rather than credit-market-wide."
        ),
        "stress_start": "2023-02-01",
        "stress_end":   "2023-03-10",
        "peak_date":    "2023-03-10",
        "color":        "#9B59B6",
        "key_signature": "Financials-specific + rate sensitivity + contained credit"
    }
}
```

---

## 6. Fingerprint Engine — `fingerprint/engine.py`

### 6.1 What This File Does
For each crisis window, extracts a fingerprint vector (mean values of each indicator during the stress window, normalized). For the live market, extracts the most recent row as a fingerprint vector. All vectors are normalized to the same scale for comparison.

```python
# fingerprint/engine.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from data.indicators import ALL_INDICATORS, INDICATOR_GROUPS, compute_all_indicators
from data.crisis_library import CRISIS_LIBRARY
from data.pipeline import fetch_all_data


def extract_crisis_fingerprints(
    indicator_df: pd.DataFrame,
    scaler: RobustScaler = None
) -> tuple[dict, RobustScaler]:
    """
    For each crisis in CRISIS_LIBRARY, extracts the mean indicator vector
    during the stress window.

    Returns:
        crisis_fingerprints: dict mapping crisis_key -> np.array of shape (n_indicators,)
        scaler: fitted RobustScaler (use this to normalize the live fingerprint too)
    """
    # Only use indicators that exist in our DataFrame
    available_indicators = [ind for ind in ALL_INDICATORS if ind in indicator_df.columns]

    # Build crisis fingerprint vectors
    crisis_fingerprints = {}
    for crisis_key, crisis_info in CRISIS_LIBRARY.items():
        start = crisis_info["stress_start"]
        end = crisis_info["stress_end"]
        window = indicator_df.loc[start:end, available_indicators]
        if len(window) == 0:
            continue
        # Use mean of the stress window as the fingerprint
        crisis_fingerprints[crisis_key] = window.mean().values

    if not crisis_fingerprints:
        raise ValueError("No crisis fingerprints could be computed. Check data range.")

    # Fit scaler on all crisis fingerprints combined
    all_vectors = np.array(list(crisis_fingerprints.values()))
    if scaler is None:
        scaler = RobustScaler()
        scaler.fit(all_vectors)

    # Normalize all crisis fingerprints
    for key in crisis_fingerprints:
        crisis_fingerprints[key] = scaler.transform(
            crisis_fingerprints[key].reshape(1, -1)
        ).flatten()

    return crisis_fingerprints, scaler, available_indicators


def extract_live_fingerprint(
    indicator_df: pd.DataFrame,
    scaler: RobustScaler,
    available_indicators: list
) -> np.ndarray:
    """
    Extracts the most recent row as the live fingerprint vector.
    Uses the same scaler as crisis fingerprints.
    """
    latest = indicator_df[available_indicators].dropna().iloc[-1].values
    return scaler.transform(latest.reshape(1, -1)).flatten()


def extract_historical_fingerprint(
    indicator_df: pd.DataFrame,
    scaler: RobustScaler,
    available_indicators: list,
    date: str
) -> np.ndarray:
    """
    For the rewind feature: extract fingerprint as of a specific historical date.
    Uses 21-day lookback window ending on the given date.
    """
    end_dt = pd.to_datetime(date)
    start_dt = end_dt - pd.Timedelta(days=30)
    window = indicator_df.loc[start_dt:end_dt, available_indicators]
    if len(window) == 0:
        return None
    vec = window.mean().values
    return scaler.transform(vec.reshape(1, -1)).flatten()


def compute_similarity_scores(
    query_vector: np.ndarray,
    crisis_fingerprints: dict
) -> list[dict]:
    """
    Computes cosine similarity between the query vector and all crisis fingerprints.
    Returns sorted list of dicts: [{ crisis_key, name, similarity, ... }, ...]
    Similarity is in [0, 100] range.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    results = []
    for crisis_key, crisis_vector in crisis_fingerprints.items():
        crisis_info = CRISIS_LIBRARY[crisis_key]
        sim = cosine_similarity(
            query_vector.reshape(1, -1),
            crisis_vector.reshape(1, -1)
        )[0][0]
        # Map from [-1, 1] to [0, 100]
        similarity_pct = max(0, (sim + 1) / 2 * 100)
        results.append({
            "crisis_key": crisis_key,
            "name": crisis_info["name"],
            "short": crisis_info["short"],
            "similarity": round(similarity_pct, 1),
            "color": crisis_info["color"],
            "key_signature": crisis_info["key_signature"],
            "description": crisis_info["description"],
            "peak_date": crisis_info["peak_date"],
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results


def compute_dimension_scores(
    query_vector: np.ndarray,
    available_indicators: list
) -> dict:
    """
    Computes per-dimension stress scores from the normalized fingerprint vector.
    Returns dict: dimension_name -> stress_score (0-100).
    Higher = more stressed.
    """
    scores = {}
    for dim_name, dim_indicators in INDICATOR_GROUPS.items():
        # Find indices of this dimension's indicators in available_indicators
        indices = [i for i, ind in enumerate(available_indicators) if ind in dim_indicators]
        if not indices:
            scores[dim_name] = 0.0
            continue
        dim_vec = query_vector[indices]
        # Use absolute value and normalize to 0-100
        score = np.abs(dim_vec).mean()
        # Cap at reasonable max (robust scaler output rarely exceeds 3)
        score = min(score / 3.0 * 100, 100)
        scores[dim_name] = round(score, 1)
    return scores
```

---

## 7. Embedding Space — `fingerprint/embedding.py`

### 7.1 What This File Does
Projects crisis fingerprints and the live fingerprint into 2D using UMAP for the scatter plot visualization. Also computes the dimension-level radar chart values.

```python
# fingerprint/embedding.py

import numpy as np
import pandas as pd


def build_umap_embedding(crisis_fingerprints: dict, live_vector: np.ndarray = None):
    """
    Fits UMAP on crisis fingerprints and optionally projects the live vector.

    Returns:
        crisis_coords: dict mapping crisis_key -> (x, y) 2D coordinates
        live_coords: (x, y) tuple for live market, or None
    """
    try:
        import umap
    except ImportError:
        # Fallback to PCA if umap not available
        from sklearn.decomposition import PCA
        keys = list(crisis_fingerprints.keys())
        vectors = np.array([crisis_fingerprints[k] for k in keys])
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(vectors)
        crisis_coords = {keys[i]: tuple(coords_2d[i]) for i in range(len(keys))}
        live_coords = None
        if live_vector is not None:
            live_2d = pca.transform(live_vector.reshape(1, -1))[0]
            live_coords = tuple(live_2d)
        return crisis_coords, live_coords

    keys = list(crisis_fingerprints.keys())
    vectors = np.array([crisis_fingerprints[k] for k in keys])

    all_vectors = vectors
    if live_vector is not None:
        all_vectors = np.vstack([vectors, live_vector.reshape(1, -1)])

    reducer = umap.UMAP(n_components=2, n_neighbors=min(5, len(keys)-1), random_state=42)
    coords_2d = reducer.fit_transform(all_vectors)

    crisis_coords = {keys[i]: tuple(coords_2d[i]) for i in range(len(keys))}
    live_coords = None
    if live_vector is not None:
        live_coords = tuple(coords_2d[-1])

    return crisis_coords, live_coords
```

---

## 8. AI Analyst Agent — `agent/analyst.py`

### 8.1 What This File Does
Calls the Anthropic API with a structured prompt that includes the current fingerprint data and top crisis analogues. Returns a structured risk narrative.

### 8.2 The Prompt Design
The prompt gives Claude:
- The top 3 crisis analogues and their similarity scores
- The per-dimension stress scores
- The top stressed indicators by name
- Instructions to produce exactly structured output

```python
# agent/analyst.py

import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def build_prompt(
    similarity_results: list,
    dimension_scores: dict,
    available_indicators: list,
    live_vector,
    query_date: str = "today"
) -> str:
    """
    Builds the structured prompt for the Claude analyst agent.
    """
    top_3 = similarity_results[:3]

    # Find the top 5 most stressed indicators (highest absolute normalized values)
    import numpy as np
    indicator_stress = [(available_indicators[i], abs(live_vector[i]))
                        for i in range(len(available_indicators))]
    indicator_stress.sort(key=lambda x: x[1], reverse=True)
    top_indicators = indicator_stress[:5]

    top_analogue_text = "\n".join([
        f"  {i+1}. {r['name']} ({r['short']}): {r['similarity']:.1f}% similarity\n"
        f"     Key signature: {r['key_signature']}\n"
        f"     Peak date: {r['peak_date']}"
        for i, r in enumerate(top_3)
    ])

    dimension_text = "\n".join([
        f"  {dim}: {score:.1f}/100 stress"
        for dim, score in sorted(dimension_scores.items(), key=lambda x: x[1], reverse=True)
    ])

    indicator_text = "\n".join([
        f"  {ind.replace('_', ' ')}: {val:.2f}σ deviation"
        for ind, val in top_indicators
    ])

    prompt = f"""You are AUTOPSY, a quantitative market risk analyst system. 
Your job is to analyze current market structure and produce a concise, precise risk narrative.

## Current Market Snapshot (as of {query_date})

### Top Crisis Structural Analogues:
{top_analogue_text}

### Stress by Dimension (0-100 scale):
{dimension_text}

### Most Stressed Indicators:
{indicator_text}

## Your Task

Write a structured risk narrative with EXACTLY these four sections:

**STRUCTURAL ASSESSMENT** (2-3 sentences)
Describe what the current market structure fingerprint reveals. Be specific about which dimensions are most elevated and why that matters structurally.

**HISTORICAL ANALOGUES** (3-4 sentences)
Explain what the top 1-2 analogues share with the current fingerprint. Be specific about which dimensions match and which do NOT match — the differences are as important as the similarities. Avoid saying "this means a crash is coming."

**KEY DIVERGENCES** (2-3 sentences)
What aspects of the current fingerprint explicitly differ from the top analogue? What does that tell us about the nature of the current stress?

**RISK POSTURE** (2-3 sentences)
What should a risk-aware institutional investor monitor closely given this fingerprint? Frame as monitoring priorities, not as trading advice.

Keep the total response under 350 words. Be precise. Do not use hedging language like "it seems" or "perhaps". Write as a senior quant risk officer would brief a CIO."""

    return prompt


def run_analyst(
    similarity_results: list,
    dimension_scores: dict,
    available_indicators: list,
    live_vector,
    query_date: str = "today"
) -> str:
    """
    Calls Claude API and returns the structured narrative string.
    On API failure, returns a fallback message.
    """
    prompt = build_prompt(
        similarity_results, dimension_scores, available_indicators, live_vector, query_date
    )

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"[Analyst unavailable: {str(e)}]\n\nTop analogue: {similarity_results[0]['name']} ({similarity_results[0]['similarity']:.1f}% similarity)"
```

---

## 9. Dashboard Charts — `dashboard/charts.py`

### 9.1 What This File Does
All Plotly chart functions. Each function takes data and returns a Plotly figure object. No Streamlit calls inside this file.

```python
# dashboard/charts.py

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from data.indicators import INDICATOR_GROUPS
from data.crisis_library import CRISIS_LIBRARY


BACKGROUND_COLOR = "#0E1117"
CARD_COLOR = "#1A1D23"
TEXT_COLOR = "#FAFAFA"
ACCENT_COLOR = "#E84393"
GRID_COLOR = "#2D3139"


def make_radar_chart(dimension_scores: dict, title: str = "Current Market Stress") -> go.Figure:
    """
    Radar chart showing stress level across 5 dimensions.
    dimension_scores: dict of {dimension_name: score_0_to_100}
    """
    dims = list(dimension_scores.keys())
    scores = list(dimension_scores.values())

    # Close the polygon
    dims_plot = dims + [dims[0]]
    scores_plot = scores + [scores[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=scores_plot,
        theta=dims_plot,
        fill='toself',
        fillcolor='rgba(232, 67, 147, 0.2)',
        line=dict(color=ACCENT_COLOR, width=2),
        name="Current"
    ))

    fig.update_layout(
        polar=dict(
            bgcolor=CARD_COLOR,
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color=TEXT_COLOR, size=10),
                gridcolor=GRID_COLOR,
            ),
            angularaxis=dict(
                tickfont=dict(color=TEXT_COLOR, size=12),
                gridcolor=GRID_COLOR,
            ),
        ),
        showlegend=False,
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        title=dict(text=title, font=dict(color=TEXT_COLOR, size=14)),
        margin=dict(l=60, r=60, t=60, b=60),
        height=380,
    )
    return fig


def make_analogue_bar_chart(similarity_results: list) -> go.Figure:
    """
    Horizontal bar chart of crisis analogue similarity scores.
    """
    top_n = similarity_results[:10]
    names = [r["short"] for r in reversed(top_n)]
    scores = [r["similarity"] for r in reversed(top_n)]
    colors = [r["color"] for r in reversed(top_n)]

    fig = go.Figure(go.Bar(
        x=scores,
        y=names,
        orientation='h',
        marker=dict(color=colors, opacity=0.85),
        text=[f"{s:.1f}%" for s in scores],
        textposition='outside',
        textfont=dict(color=TEXT_COLOR),
    ))

    fig.update_layout(
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        xaxis=dict(
            range=[0, 110],
            gridcolor=GRID_COLOR,
            tickfont=dict(color=TEXT_COLOR),
            title=dict(text="Structural Similarity (%)", font=dict(color=TEXT_COLOR)),
        ),
        yaxis=dict(tickfont=dict(color=TEXT_COLOR)),
        margin=dict(l=20, r=80, t=20, b=40),
        height=360,
    )
    return fig


def make_embedding_scatter(crisis_coords: dict, live_coords: tuple = None) -> go.Figure:
    """
    2D UMAP scatter plot of crisis fingerprints + current market.
    """
    fig = go.Figure()

    # Plot each historical crisis
    for crisis_key, (x, y) in crisis_coords.items():
        info = CRISIS_LIBRARY.get(crisis_key, {})
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(
                size=16,
                color=info.get("color", "#888888"),
                opacity=0.85,
                line=dict(width=1, color='white')
            ),
            text=[info.get("short", crisis_key)],
            textposition="top center",
            textfont=dict(color=TEXT_COLOR, size=10),
            name=info.get("short", crisis_key),
            showlegend=False,
            hovertemplate=f"<b>{info.get('name', crisis_key)}</b><br>{info.get('key_signature', '')}<extra></extra>"
        ))

    # Plot live market
    if live_coords is not None:
        fig.add_trace(go.Scatter(
            x=[live_coords[0]], y=[live_coords[1]],
            mode='markers+text',
            marker=dict(
                size=22,
                color=ACCENT_COLOR,
                symbol='star',
                line=dict(width=2, color='white')
            ),
            text=["NOW"],
            textposition="top center",
            textfont=dict(color=ACCENT_COLOR, size=12, family="Arial Black"),
            name="Current Market",
            showlegend=False,
            hovertemplate="<b>Current Market</b><extra></extra>"
        ))

    fig.update_layout(
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=CARD_COLOR,
        xaxis=dict(showticklabels=False, gridcolor=GRID_COLOR, zeroline=False),
        yaxis=dict(showticklabels=False, gridcolor=GRID_COLOR, zeroline=False),
        margin=dict(l=20, r=20, t=20, b=20),
        height=400,
        title=dict(text="Crisis Fingerprint Embedding Space", font=dict(color=TEXT_COLOR, size=13)),
    )
    return fig


def make_indicator_heatmap(indicator_df: pd.DataFrame, available_indicators: list, days: int = 60) -> go.Figure:
    """
    Heatmap of indicator z-scores over the last N days.
    Shows which indicators are most stressed and for how long.
    """
    recent = indicator_df[available_indicators].iloc[-days:].copy()

    # Z-score each column over the display window for visualization
    z_scored = (recent - recent.mean()) / (recent.std() + 1e-8)
    z_scored = z_scored.clip(-3, 3)

    # Shorten indicator names for display
    short_names = [ind.replace('_', ' ')[:20] for ind in available_indicators]

    fig = go.Figure(go.Heatmap(
        z=z_scored.T.values,
        x=z_scored.index.strftime("%Y-%m-%d"),
        y=short_names,
        colorscale=[
            [0.0, "#1a4f72"],    # low (negative z)
            [0.5, CARD_COLOR],   # neutral
            [1.0, "#8B0000"],    # high (positive z = stress)
        ],
        zmid=0,
        colorbar=dict(
            title="Z-Score",
            tickfont=dict(color=TEXT_COLOR),
            titlefont=dict(color=TEXT_COLOR),
        ),
        hoverongaps=False,
    ))

    fig.update_layout(
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        xaxis=dict(
            tickangle=45,
            tickfont=dict(color=TEXT_COLOR, size=9),
            title=dict(text="Date", font=dict(color=TEXT_COLOR)),
        ),
        yaxis=dict(tickfont=dict(color=TEXT_COLOR, size=9)),
        margin=dict(l=160, r=40, t=20, b=80),
        height=500,
        title=dict(text=f"Indicator Stress Heatmap (Last {days} Days)", font=dict(color=TEXT_COLOR, size=13)),
    )
    return fig


def make_dimension_timeseries(indicator_df: pd.DataFrame, available_indicators: list, days: int = 252) -> go.Figure:
    """
    Line chart showing per-dimension composite stress over time.
    """
    recent = indicator_df[available_indicators].iloc[-days:].copy()

    dimension_colors = {
        "Liquidity": "#E74C3C",
        "Volatility": "#E67E22",
        "Correlation": "#F1C40F",
        "Credit": "#9B59B6",
        "Positioning": "#3498DB",
    }

    fig = go.Figure()

    for dim_name, dim_indicators in INDICATOR_GROUPS.items():
        dim_cols = [ind for ind in dim_indicators if ind in recent.columns]
        if not dim_cols:
            continue
        dim_data = recent[dim_cols]
        # Normalize each column then take mean absolute value
        normalized = (dim_data - dim_data.mean()) / (dim_data.std() + 1e-8)
        composite = normalized.abs().mean(axis=1)
        # Smooth with 10-day rolling mean
        composite_smooth = composite.rolling(10, min_periods=1).mean()
        # Scale to 0-100
        composite_scaled = (composite_smooth / composite_smooth.quantile(0.99).clip(0.01)) * 100
        composite_scaled = composite_scaled.clip(0, 100)

        fig.add_trace(go.Scatter(
            x=recent.index,
            y=composite_scaled,
            name=dim_name,
            line=dict(color=dimension_colors.get(dim_name, "#888888"), width=2),
            mode='lines',
        ))

    fig.update_layout(
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=CARD_COLOR,
        xaxis=dict(
            gridcolor=GRID_COLOR,
            tickfont=dict(color=TEXT_COLOR),
            title=dict(text="Date", font=dict(color=TEXT_COLOR)),
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            tickfont=dict(color=TEXT_COLOR),
            title=dict(text="Stress (0-100)", font=dict(color=TEXT_COLOR)),
            range=[0, 105],
        ),
        legend=dict(
            font=dict(color=TEXT_COLOR),
            bgcolor=CARD_COLOR,
        ),
        margin=dict(l=60, r=20, t=20, b=40),
        height=350,
        title=dict(text="Stress Dimension History", font=dict(color=TEXT_COLOR, size=13)),
    )
    return fig
```

---

## 10. Dashboard State — `dashboard/state.py`

```python
# dashboard/state.py

import streamlit as st
import pandas as pd
from datetime import datetime


def initialize_session_state():
    """Initialize all session state variables if not already set."""
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None
    if "indicator_df" not in st.session_state:
        st.session_state.indicator_df = None
    if "crisis_fingerprints" not in st.session_state:
        st.session_state.crisis_fingerprints = None
    if "scaler" not in st.session_state:
        st.session_state.scaler = None
    if "available_indicators" not in st.session_state:
        st.session_state.available_indicators = None
    if "live_vector" not in st.session_state:
        st.session_state.live_vector = None
    if "similarity_results" not in st.session_state:
        st.session_state.similarity_results = None
    if "dimension_scores" not in st.session_state:
        st.session_state.dimension_scores = None
    if "analyst_narrative" not in st.session_state:
        st.session_state.analyst_narrative = None
    if "rewind_date" not in st.session_state:
        st.session_state.rewind_date = None
    if "mode" not in st.session_state:
        st.session_state.mode = "live"  # "live" or "rewind"
    if "embedding_coords" not in st.session_state:
        st.session_state.embedding_coords = None
    if "live_embedding_coords" not in st.session_state:
        st.session_state.live_embedding_coords = None
```

---

## 11. Main Dashboard App — `dashboard/app.py`

### 11.1 Structure
The app has three views:
1. **Live Mode** — shows current market fingerprint, analogues, AI narrative
2. **Rewind Mode** — pick any historical date, see what AUTOPSY would have shown
3. **Heatmap Mode** — full indicator heatmap over last 60/90/252 days

```python
# dashboard/app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Ensure imports work from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.pipeline import fetch_all_data
from data.indicators import compute_all_indicators
from fingerprint.engine import (
    extract_crisis_fingerprints, extract_live_fingerprint,
    extract_historical_fingerprint, compute_similarity_scores, compute_dimension_scores
)
from fingerprint.embedding import build_umap_embedding
from agent.analyst import run_analyst
from dashboard.charts import (
    make_radar_chart, make_analogue_bar_chart, make_embedding_scatter,
    make_indicator_heatmap, make_dimension_timeseries
)
from dashboard.state import initialize_session_state

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_name="AUTOPSY",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .main-title {
        font-size: 2.8rem;
        font-weight: 900;
        color: #E84393;
        letter-spacing: -1px;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.0rem;
        color: #888;
        margin-top: -8px;
        margin-bottom: 24px;
    }
    .metric-card {
        background: #1A1D23;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #2D3139;
    }
    .analogue-card {
        background: #1A1D23;
        border-radius: 8px;
        padding: 14px 16px;
        border-left: 4px solid #E84393;
        margin-bottom: 10px;
    }
    .narrative-box {
        background: #1A1D23;
        border-radius: 10px;
        padding: 20px 24px;
        border: 1px solid #2D3139;
        font-size: 0.92rem;
        line-height: 1.7;
        color: #D0D0D0;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #FAFAFA;
        margin: 20px 0 8px 0;
        border-bottom: 1px solid #2D3139;
        padding-bottom: 4px;
    }
    .stress-badge-high { color: #E74C3C; font-weight: bold; }
    .stress-badge-med  { color: #E67E22; font-weight: bold; }
    .stress-badge-low  { color: #27AE60; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

initialize_session_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 AUTOPSY")
    st.markdown("*Market Dislocation Fingerprint Intelligence*")
    st.divider()

    mode = st.radio(
        "Mode",
        options=["🔴 Live Market", "⏪ Rewind", "🗺 Heatmap"],
        index=0
    )

    st.divider()

    if mode == "⏪ Rewind":
        rewind_date = st.date_input(
            "Select Historical Date",
            value=pd.to_datetime("2020-03-01"),
            min_value=pd.to_datetime("2005-01-01"),
            max_value=pd.to_datetime("today")
        )
        st.caption("See what AUTOPSY would have shown on this date.")

    if mode == "🗺 Heatmap":
        heatmap_days = st.selectbox("Lookback Window", [30, 60, 90, 252], index=1)

    st.divider()

    load_button = st.button("🔄 Load / Refresh Data", use_container_width=True)

    if st.session_state.data_loaded:
        st.success("✅ Data loaded")
        if st.session_state.raw_df is not None:
            st.caption(f"Latest: {st.session_state.indicator_df.index[-1].strftime('%Y-%m-%d')}")
    else:
        st.warning("⚠️ Click to load data")

    st.divider()
    st.caption("Data: FRED + Yahoo Finance\nAI: Claude (Anthropic)")


# ── Data Loading ──────────────────────────────────────────────────────────────
if load_button or not st.session_state.data_loaded:
    with st.spinner("Fetching market data (this takes ~30 seconds)..."):
        try:
            # Fetch data from 2000 to today to cover all crisis windows
            raw_df = fetch_all_data(start_date="2000-01-01")
            st.session_state.raw_df = raw_df

            with st.spinner("Computing indicators..."):
                indicator_df = compute_all_indicators(raw_df)
                st.session_state.indicator_df = indicator_df

            with st.spinner("Building crisis fingerprint library..."):
                crisis_fingerprints, scaler, available_indicators = extract_crisis_fingerprints(indicator_df)
                st.session_state.crisis_fingerprints = crisis_fingerprints
                st.session_state.scaler = scaler
                st.session_state.available_indicators = available_indicators

            with st.spinner("Computing live fingerprint..."):
                live_vector = extract_live_fingerprint(indicator_df, scaler, available_indicators)
                st.session_state.live_vector = live_vector
                similarity_results = compute_similarity_scores(live_vector, crisis_fingerprints)
                st.session_state.similarity_results = similarity_results
                dimension_scores = compute_dimension_scores(live_vector, available_indicators)
                st.session_state.dimension_scores = dimension_scores

            with st.spinner("Building embedding space..."):
                crisis_coords, live_coords = build_umap_embedding(crisis_fingerprints, live_vector)
                st.session_state.embedding_coords = crisis_coords
                st.session_state.live_embedding_coords = live_coords

            with st.spinner("Running AI analyst..."):
                narrative = run_analyst(
                    similarity_results, dimension_scores, available_indicators,
                    live_vector, query_date=datetime.today().strftime("%B %d, %Y")
                )
                st.session_state.analyst_narrative = narrative

            st.session_state.data_loaded = True
            st.rerun()

        except Exception as e:
            st.error(f"Data loading failed: {str(e)}")
            st.exception(e)


# ── Main Content ──────────────────────────────────────────────────────────────
if not st.session_state.data_loaded:
    st.markdown('<div class="main-title">AUTOPSY</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Market Dislocation Fingerprint Intelligence System</div>', unsafe_allow_html=True)
    st.info("👈 Click **Load / Refresh Data** in the sidebar to begin.")

    st.markdown("### What is AUTOPSY?")
    st.markdown("""
    AUTOPSY analyzes the **structural fingerprint** of current market conditions across five dimensions:
    **Liquidity · Volatility · Correlation · Credit · Positioning**

    It compares this fingerprint against pre-crisis signatures from 10 historical dislocations,
    identifies the closest structural analogues, and generates an AI-powered risk narrative.

    This is not a price prediction tool. It is a **market structure stress detector**.
    """)
    st.stop()


# ── Determine what vector/results to display ──────────────────────────────────
if mode == "⏪ Rewind" and 'rewind_date' in dir():
    rewind_str = rewind_date.strftime("%Y-%m-%d")
    display_vector = extract_historical_fingerprint(
        st.session_state.indicator_df,
        st.session_state.scaler,
        st.session_state.available_indicators,
        rewind_str
    )
    if display_vector is None:
        st.error(f"No data available for {rewind_str}")
        st.stop()
    display_similarity = compute_similarity_scores(display_vector, st.session_state.crisis_fingerprints)
    display_dimension = compute_dimension_scores(display_vector, st.session_state.available_indicators)
    display_date_label = rewind_date.strftime("%B %d, %Y")
    rewind_narrative = run_analyst(
        display_similarity, display_dimension, st.session_state.available_indicators,
        display_vector, query_date=display_date_label
    )
    display_narrative = rewind_narrative
    rewind_coords, rewind_live_coords = build_umap_embedding(
        st.session_state.crisis_fingerprints, display_vector
    )
    display_embedding_coords = rewind_coords
    display_live_coords = rewind_live_coords
    is_rewind = True
else:
    display_vector = st.session_state.live_vector
    display_similarity = st.session_state.similarity_results
    display_dimension = st.session_state.dimension_scores
    display_date_label = datetime.today().strftime("%B %d, %Y")
    display_narrative = st.session_state.analyst_narrative
    display_embedding_coords = st.session_state.embedding_coords
    display_live_coords = st.session_state.live_embedding_coords
    is_rewind = False


# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_date = st.columns([3, 1])
with col_title:
    label = f"⏪ REWIND: {display_date_label}" if is_rewind else "🔴 LIVE"
    st.markdown(f'<div class="main-title">AUTOPSY</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">Market Dislocation Fingerprint Intelligence · {label}</div>',
                unsafe_allow_html=True)
with col_date:
    top_analogue = display_similarity[0]
    st.metric(
        label="Top Analogue",
        value=top_analogue["short"],
        delta=f"{top_analogue['similarity']:.1f}% match"
    )


# ── Heatmap Mode ──────────────────────────────────────────────────────────────
if mode == "🗺 Heatmap":
    st.markdown('<div class="section-header">Indicator Stress Heatmap</div>', unsafe_allow_html=True)
    fig_heatmap = make_indicator_heatmap(
        st.session_state.indicator_df,
        st.session_state.available_indicators,
        days=heatmap_days
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown('<div class="section-header">Dimension Stress History</div>', unsafe_allow_html=True)
    fig_ts = make_dimension_timeseries(
        st.session_state.indicator_df,
        st.session_state.available_indicators,
        days=heatmap_days
    )
    st.plotly_chart(fig_ts, use_container_width=True)
    st.stop()


# ── Main Dashboard Layout (Live + Rewind) ─────────────────────────────────────
col_left, col_right = st.columns([1.1, 0.9])

with col_left:

    # Radar chart
    st.markdown('<div class="section-header">Structural Stress Profile</div>', unsafe_allow_html=True)
    fig_radar = make_radar_chart(display_dimension)
    st.plotly_chart(fig_radar, use_container_width=True)

    # Analogue bar chart
    st.markdown('<div class="section-header">Crisis Analogue Similarity</div>', unsafe_allow_html=True)
    fig_bar = make_analogue_bar_chart(display_similarity)
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:

    # Top 3 analogue cards
    st.markdown('<div class="section-header">Top Structural Analogues</div>', unsafe_allow_html=True)
    for r in display_similarity[:3]:
        border_color = r["color"]
        st.markdown(f"""
        <div style="background:#1A1D23; border-radius:8px; padding:14px 16px;
                    border-left:4px solid {border_color}; margin-bottom:10px;">
            <div style="font-weight:700; color:#FAFAFA; font-size:1.0rem;">
                {r['name']}
                <span style="float:right; color:{border_color}; font-size:1.1rem;">{r['similarity']:.1f}%</span>
            </div>
            <div style="color:#888; font-size:0.82rem; margin-top:4px;">{r['key_signature']}</div>
            <div style="color:#666; font-size:0.78rem; margin-top:2px;">Peak: {r['peak_date']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Embedding scatter
    st.markdown('<div class="section-header">Fingerprint Space</div>', unsafe_allow_html=True)
    fig_embed = make_embedding_scatter(display_embedding_coords, display_live_coords)
    st.plotly_chart(fig_embed, use_container_width=True)


# ── AI Analyst Narrative ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">AI Risk Analyst</div>', unsafe_allow_html=True)
if display_narrative:
    st.markdown(f'<div class="narrative-box">{display_narrative.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True)
else:
    st.info("AI narrative will appear after data is loaded.")


# ── Dimension Timeseries ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">Stress Dimension History (1 Year)</div>', unsafe_allow_html=True)
fig_ts = make_dimension_timeseries(
    st.session_state.indicator_df,
    st.session_state.available_indicators,
    days=252
)
st.plotly_chart(fig_ts, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "AUTOPSY | Market Dislocation Fingerprint Intelligence | "
    "TechEx Intelligent Enterprise Solutions Hackathon 2026 | "
    "Data: FRED, Yahoo Finance | AI: Claude (Anthropic) | "
    "Not investment advice."
)
```

---

## 12. Entry Point — Run From Root

Run the app from the `autopsy/` root directory:

```bash
streamlit run dashboard/app.py
```

---

## 13. Build Order (Follow This Exactly)

### Step 1 — Setup
```bash
mkdir autopsy && cd autopsy
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Fill in FRED_API_KEY and ANTHROPIC_API_KEY in .env
```

### Step 2 — Build Files In This Order
Build each file completely before moving to the next. Do not skip steps.

1. `data/crisis_library.py` — no dependencies, pure data
2. `data/pipeline.py` — data fetching
3. `data/indicators.py` — depends on nothing external
4. `fingerprint/engine.py` — depends on data layer
5. `fingerprint/embedding.py` — standalone
6. `agent/analyst.py` — standalone
7. `dashboard/state.py` — standalone
8. `dashboard/charts.py` — depends on crisis_library and indicator groups
9. `dashboard/app.py` — depends on everything

### Step 3 — Test Data Pipeline First
Before building the dashboard, verify data fetches:
```bash
python -c "
from data.pipeline import fetch_all_data
from data.indicators import compute_all_indicators
df = fetch_all_data(start_date='2019-01-01')
print('Raw columns:', df.columns.tolist())
ind = compute_all_indicators(df)
print('Indicator columns:', ind.columns.tolist())
print('Last row:', ind.iloc[-1])
"
```

### Step 4 — Test Fingerprint Engine
```bash
python -c "
from data.pipeline import fetch_all_data
from data.indicators import compute_all_indicators
from fingerprint.engine import extract_crisis_fingerprints, extract_live_fingerprint, compute_similarity_scores
df = fetch_all_data(start_date='1998-01-01')
ind = compute_all_indicators(df)
cfp, scaler, avail = extract_crisis_fingerprints(ind)
live = extract_live_fingerprint(ind, scaler, avail)
results = compute_similarity_scores(live, cfp)
for r in results[:3]:
    print(r['name'], r['similarity'])
"
```

### Step 5 — Test Agent
```bash
python -c "
from agent.analyst import run_analyst
# Dummy data
dummy_sim = [{'name': 'GFC 2008', 'short': 'GFC 2008', 'similarity': 67.0, 'key_signature': 'test', 'peak_date': '2008-10-10', 'description': 'test'}]
dummy_dim = {'Liquidity': 72, 'Volatility': 45, 'Correlation': 60, 'Credit': 80, 'Positioning': 30}
import numpy as np
dummy_vec = np.zeros(40)
result = run_analyst(dummy_sim, dummy_dim, [f'ind_{i}' for i in range(40)], dummy_vec, 'Test')
print(result[:200])
"
```

### Step 6 — Run Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 14. Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `KeyError` in FRED fetch | Series ID changed or unavailable | Check FRED website for current series ID |
| `yfinance` returns empty | Ticker delisted or market closed | Use `period='5y'` fallback in yf.download |
| `UMAP` import fails | umap-learn not installed | `pip install umap-learn` or it falls back to PCA automatically |
| Anthropic `AuthenticationError` | Wrong API key | Check `.env` file, ensure no whitespace around key |
| `No crisis fingerprints` error | Data range too short | Ensure `start_date='1998-01-01'` in fetch_all_data |
| Dashboard blank on load | Data not loaded | Click "Load / Refresh Data" button |
| Slow load (~60s) | Normal — fetching 25 years of FRED + 18 tickers | Add `@st.cache_data` decorator to heavy functions |

### Adding `@st.cache_data` to Speed Up Reloads
Wrap the heavy fetch functions in app.py:

```python
@st.cache_data(ttl=3600)  # cache for 1 hour
def load_all_data():
    raw_df = fetch_all_data(start_date="1998-01-01")
    indicator_df = compute_all_indicators(raw_df)
    return raw_df, indicator_df
```

---

## 15. Hackathon Pitch Notes

### One-Line Pitch
"AUTOPSY tells you which historical market crisis today's structure most resembles — before the price moves confirm it."

### Track Fit
Track 4: Data & Intelligence — multi-source financial data pipeline, AI-powered analytics agent, anomaly/structural detection, enterprise risk use case.

### The Demo Flow (5 minutes)
1. **Open live dashboard** — show today's radar chart and top analogue (30 seconds)
2. **Rewind to Feb 2020** — show the fingerprint forming before COVID crash (60 seconds)
3. **Rewind to Aug 2007** — show GFC fingerprint building before Lehman (60 seconds)
4. **Read AI narrative** — show structured risk brief (30 seconds)
5. **Explain the research** — formal taxonomy, embedding space, what makes this original vs existing stress indices (60 seconds)
6. **Product roadmap** — arXiv preprint, institutional data subscription, real-time alerts (30 seconds)

### Key Differentiator vs Existing Tools
- **Not a stress index** (single number) — multi-dimensional fingerprint with structural decomposition
- **Not scenario-based** — empirically derived from actual pre-crisis data
- **Not price prediction** — structural pattern matching, avoids the "market timing" criticism
- **Historical analogue mapping** — tells you not just "stress is high" but "which crisis this resembles and why"

---

## 16. arXiv Preprint Structure (Post-Hackathon)

```
Title: AUTOPSY: A Market Dislocation Fingerprint Framework for 
       Real-Time Systemic Stress Detection

Abstract: We present AUTOPSY, a framework for detecting pre-dislocation 
          structural fingerprints in financial markets...

1. Introduction
2. Related Work (Stress indices, regime detection, systemic risk)
3. The Fingerprint Taxonomy (5 dimensions, 40 indicators)
4. Crisis Library Construction
5. Embedding Methodology (UMAP on normalized vectors)
6. Empirical Validation
   6.1 In-sample: Do crisis fingerprints cluster meaningfully?
   6.2 Lead time analysis: How early does each dimension activate?
   6.3 Out-of-sample: Train on pre-2020, evaluate on COVID and SVB
7. The AUTOPSY System
8. Conclusion

Target: arXiv q-fin.RM or q-fin.ST
```

---

*AUTOPSY | Arka Sarkar | TechEx Intelligent Enterprise Solutions Hackathon 2026*