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

FRED_SERIES = {
    "ted_spread":              "TEDRATE",
    "yield_curve_10y2y":       "T10Y2Y",
    "yield_curve_10y3m":       "T10Y3M",
    "ig_credit_spread":        "BAMLC0A0CM",
    "hy_credit_spread":        "BAMLH0A0HYM2",
    "ig_hy_ratio":             None,
    "vix_fred":                "VIXCLS",
}

YF_TICKERS = {
    "spy": "SPY", "qqq": "QQQ", "ief": "IEF", "lqd": "LQD",
    "hyg": "HYG", "gld": "GLD", "uso": "USO", "uup": "UUP",
    "eem": "EEM", "xlf": "XLF", "xle": "XLE", "xlu": "XLU",
    "vix": "^VIX", "vix_3m": "^VIX3M", "vix_9d": "^VIX9D",
    "eurusd": "EURUSD=X", "jpyusd": "JPY=X", "chfusd": "CHF=X",
}


def fetch_fred_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches all FRED series. Returns DataFrame with dates as index."""
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
    df = df.resample("B").last()
    df = df.ffill().bfill()

    if "ig_credit_spread" in df.columns and "hy_credit_spread" in df.columns:
        df["ig_hy_ratio"] = df["hy_credit_spread"] / df["ig_credit_spread"].replace(0, np.nan)
    return df


def fetch_yfinance_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches closing prices for all yfinance tickers.
    FIX #4: Handles both MultiIndex and flat column formats.
    FIX #7: Case-insensitive column matching.
    """
    tickers = list(YF_TICKERS.values())
    raw = yf.download(tickers, start=start_date, end=end_date, progress=False)

    # FIX #4: Defensive column extraction
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"].copy()
        elif "Adj Close" in raw.columns.get_level_values(0):
            close = raw["Adj Close"].copy()
        else:
            first_level = raw.columns.get_level_values(0).unique()[0]
            close = raw[first_level].copy()
    else:
        close = raw.copy()

    # FIX #7: Case-insensitive reverse map
    reverse_map = {v: k for k, v in YF_TICKERS.items()}
    reverse_map_lower = {v.lower(): k for k, v in YF_TICKERS.items()}
    new_columns = []
    for col in close.columns:
        col_str = str(col)
        if col_str in reverse_map:
            new_columns.append(reverse_map[col_str])
        elif col_str.lower() in reverse_map_lower:
            new_columns.append(reverse_map_lower[col_str.lower()])
        else:
            new_columns.append(col_str)
    close.columns = new_columns
    close = close.ffill().bfill()
    return close


def fetch_all_data(start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    """Master fetch function. Returns a single combined DataFrame."""
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.today() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")

    print(f"[Pipeline] Fetching data from {start_date} to {end_date}...")
    fred_df = fetch_fred_data(start_date, end_date)
    yf_df = fetch_yfinance_data(start_date, end_date)

    combined = pd.concat([fred_df, yf_df], axis=1)
    combined = combined.resample("B").last().ffill().bfill()
    print(f"[Pipeline] Fetched {len(combined)} rows, {len(combined.columns)} columns.")
    return combined


def fetch_live_snapshot() -> pd.DataFrame:
    """Fetches the most recent ~1 year of data for live fingerprinting."""
    return fetch_all_data(
        start_date=(datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d"),
        end_date=datetime.today().strftime("%Y-%m-%d")
    )
