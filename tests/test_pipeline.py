# tests/test_pipeline.py

"""
Smoke tests for AUTOPSY.
These tests verify imports, data structures, and basic computation
without requiring API keys or network access.
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

# Ensure imports work from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_crisis_library_structure():
    """Verify crisis library has expected structure."""
    from data.crisis_library import CRISIS_LIBRARY
    assert len(CRISIS_LIBRARY) == 10
    required_keys = {"name", "short", "description", "stress_start", "stress_end",
                     "peak_date", "color", "key_signature"}
    for crisis_key, crisis_info in CRISIS_LIBRARY.items():
        assert required_keys.issubset(crisis_info.keys()), f"{crisis_key} missing keys"
        # Dates should be parseable
        pd.to_datetime(crisis_info["stress_start"])
        pd.to_datetime(crisis_info["stress_end"])
        pd.to_datetime(crisis_info["peak_date"])


def test_indicator_groups_complete():
    """Verify all 40 indicators are defined."""
    from data.indicators import ALL_INDICATORS, INDICATOR_GROUPS
    assert len(ALL_INDICATORS) == 40, f"Expected 40 indicators, got {len(ALL_INDICATORS)}"
    assert len(INDICATOR_GROUPS) == 5, f"Expected 5 dimensions, got {len(INDICATOR_GROUPS)}"
    for dim_name, indicators in INDICATOR_GROUPS.items():
        assert len(indicators) == 8, f"{dim_name} should have 8 indicators, got {len(indicators)}"


def test_compute_indicators_with_synthetic_data():
    """Test indicator computation with synthetic price data."""
    from data.indicators import compute_all_indicators, ALL_INDICATORS

    # Create 300 days of synthetic data
    dates = pd.bdate_range("2023-01-01", periods=300)
    np.random.seed(42)
    df = pd.DataFrame({
        "spy": 400 + np.cumsum(np.random.randn(300) * 2),
        "ief": 100 + np.cumsum(np.random.randn(300) * 0.5),
        "gld": 180 + np.cumsum(np.random.randn(300) * 1),
        "hyg": 80 + np.cumsum(np.random.randn(300) * 0.3),
        "lqd": 110 + np.cumsum(np.random.randn(300) * 0.2),
        "qqq": 350 + np.cumsum(np.random.randn(300) * 3),
        "uso": 70 + np.cumsum(np.random.randn(300) * 1.5),
        "uup": 28 + np.cumsum(np.random.randn(300) * 0.1),
        "eem": 40 + np.cumsum(np.random.randn(300) * 1),
        "xlf": 35 + np.cumsum(np.random.randn(300) * 0.8),
        "xle": 80 + np.cumsum(np.random.randn(300) * 1.2),
        "xlu": 65 + np.cumsum(np.random.randn(300) * 0.4),
        "vix": 20 + np.abs(np.random.randn(300) * 5),
        "vix_3m": 22 + np.abs(np.random.randn(300) * 3),
        "vix_9d": 18 + np.abs(np.random.randn(300) * 6),
        "eurusd": 1.1 + np.cumsum(np.random.randn(300) * 0.005),
        "jpyusd": 130 + np.cumsum(np.random.randn(300) * 0.5),
        "chfusd": 0.9 + np.cumsum(np.random.randn(300) * 0.003),
        "ted_spread": 0.3 + np.abs(np.random.randn(300) * 0.1),
        "yield_curve_10y2y": 0.5 + np.random.randn(300) * 0.3,
        "ig_credit_spread": 1.2 + np.abs(np.random.randn(300) * 0.2),
        "hy_credit_spread": 4.0 + np.abs(np.random.randn(300) * 0.5),
    }, index=dates)

    ind = compute_all_indicators(df)

    # Check all 40 indicators were produced
    assert len(ind.columns) == 40, f"Expected 40 columns, got {len(ind.columns)}"
    # Check no infinities
    assert not np.isinf(ind.values).any(), "Found infinity values in indicators"
    # Check shape preserved
    assert len(ind) == 300


def test_fingerprint_engine_with_synthetic():
    """Test fingerprint extraction and similarity scoring."""
    from data.indicators import compute_all_indicators, ALL_INDICATORS
    from fingerprint.engine import (
        extract_crisis_fingerprints, extract_live_fingerprint,
        compute_similarity_scores, compute_dimension_scores
    )

    # Build a long synthetic dataset covering crisis windows
    dates = pd.bdate_range("1998-01-01", "2024-01-01")
    np.random.seed(42)
    n = len(dates)
    df = pd.DataFrame({
        "spy": 100 + np.cumsum(np.random.randn(n) * 0.5),
        "ief": 50 + np.cumsum(np.random.randn(n) * 0.2),
        "gld": 80 + np.cumsum(np.random.randn(n) * 0.3),
        "hyg": 60 + np.cumsum(np.random.randn(n) * 0.15),
        "lqd": 70 + np.cumsum(np.random.randn(n) * 0.1),
        "qqq": 90 + np.cumsum(np.random.randn(n) * 0.8),
        "uso": 40 + np.cumsum(np.random.randn(n) * 0.4),
        "uup": 25 + np.cumsum(np.random.randn(n) * 0.05),
        "eem": 35 + np.cumsum(np.random.randn(n) * 0.3),
        "xlf": 30 + np.cumsum(np.random.randn(n) * 0.2),
        "xle": 50 + np.cumsum(np.random.randn(n) * 0.3),
        "xlu": 40 + np.cumsum(np.random.randn(n) * 0.1),
        "vix": 20 + np.abs(np.random.randn(n) * 5),
        "vix_3m": 22 + np.abs(np.random.randn(n) * 3),
        "vix_9d": 18 + np.abs(np.random.randn(n) * 6),
        "eurusd": 1.1 + np.cumsum(np.random.randn(n) * 0.002),
        "jpyusd": 110 + np.cumsum(np.random.randn(n) * 0.2),
        "chfusd": 0.95 + np.cumsum(np.random.randn(n) * 0.001),
        "ted_spread": 0.3 + np.abs(np.random.randn(n) * 0.1),
        "yield_curve_10y2y": 0.5 + np.random.randn(n) * 0.3,
        "ig_credit_spread": 1.2 + np.abs(np.random.randn(n) * 0.2),
        "hy_credit_spread": 4.0 + np.abs(np.random.randn(n) * 0.5),
    }, index=dates)

    ind = compute_all_indicators(df)

    # Extract crisis fingerprints
    crisis_fp, scaler, avail = extract_crisis_fingerprints(ind)
    assert len(crisis_fp) > 0, "No crisis fingerprints extracted"
    assert len(avail) == 40

    # Extract live fingerprint
    live_vec = extract_live_fingerprint(ind, scaler, avail)
    assert live_vec.shape == (40,)

    # Compute similarity
    results = compute_similarity_scores(live_vec, crisis_fp)
    assert len(results) == len(crisis_fp)
    assert all(0 <= r["similarity"] <= 100 for r in results)

    # Compute dimension scores
    dim_scores = compute_dimension_scores(live_vec, avail)
    assert len(dim_scores) == 5
    assert all(0 <= s <= 100 for s in dim_scores.values())


def test_embedding():
    """Test UMAP/PCA embedding."""
    from fingerprint.embedding import build_umap_embedding

    # Fake crisis fingerprints
    np.random.seed(42)
    crisis_fp = {f"crisis_{i}": np.random.randn(40) for i in range(5)}
    live_vec = np.random.randn(40)

    crisis_coords, live_coords = build_umap_embedding(crisis_fp, live_vec)
    assert len(crisis_coords) == 5
    assert live_coords is not None
    assert len(live_coords) == 2
    for key, (x, y) in crisis_coords.items():
        assert isinstance(x, (float, np.floating))
        assert isinstance(y, (float, np.floating))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
