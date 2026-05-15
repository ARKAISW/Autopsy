# data/indicators.py

import pandas as pd
import numpy as np


def compute_velocity(series: pd.Series, window: int = 10) -> pd.Series:
    """Rate of change over window days, as percentage."""
    return series.pct_change(periods=window) * 100


def compute_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """
    Rolling z-score against a trailing window.
    FIX #9: Clamp std to avoid near-zero division blowups.
    """
    mean = series.rolling(window=window, min_periods=window // 2).mean()
    std = series.rolling(window=window, min_periods=window // 2).std()
    # FIX #9: prevent near-zero std from producing huge z-scores
    std = std.clip(lower=1e-8)
    return (series - mean) / std


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

    # FIX #5: cross_asset_vol_spike — use Series, not scalar fallback
    vix_vel = ind["vix_velocity"] if "vix_velocity" in ind.columns else pd.Series(0.0, index=ind.index)
    ind["cross_asset_vol_spike"] = np.abs(vix_vel)

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

    # Safe haven demand
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
    for ticker, col_name in [("gld", "gold_flow_signal"), ("jpyusd", "jpy_flow_signal"),
                              ("chfusd", "chf_flow_signal")]:
        if ticker in df.columns:
            ind[col_name] = compute_velocity(df[ticker], 10)
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
        ind["em_outflow_signal"] = compute_velocity(df["eem"], 10) * -1
    else:
        ind["em_outflow_signal"] = 0.0

    # Risk-off composite
    risk_off_cols = [c for c in ["gold_flow_signal", "jpy_flow_signal",
                     "chf_flow_signal", "utilities_relative"] if c in ind.columns]
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
