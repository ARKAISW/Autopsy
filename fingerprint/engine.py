# fingerprint/engine.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from data.indicators import ALL_INDICATORS, INDICATOR_GROUPS, compute_all_indicators
from data.crisis_library import CRISIS_LIBRARY
from data.pipeline import fetch_all_data


def extract_crisis_fingerprints(
    indicator_df: pd.DataFrame,
    scaler: RobustScaler | None = None
) -> tuple[dict, RobustScaler, list]:
    """
    For each crisis in CRISIS_LIBRARY, extracts the mean indicator vector
    during the stress window.

    FIX #3: Return type corrected to tuple[dict, RobustScaler, list]
    (blueprint said tuple[dict, RobustScaler] but returned 3 values).

    Returns:
        crisis_fingerprints: dict mapping crisis_key -> np.array of shape (n_indicators,)
        scaler: fitted RobustScaler
        available_indicators: list of indicator names actually present
    """
    available_indicators = [ind for ind in ALL_INDICATORS if ind in indicator_df.columns]

    crisis_fingerprints = {}
    for crisis_key, crisis_info in CRISIS_LIBRARY.items():
        start = crisis_info["stress_start"]
        end = crisis_info["stress_end"]
        window = indicator_df.loc[start:end, available_indicators]
        if len(window) == 0:
            continue
        crisis_fingerprints[crisis_key] = window.mean().values

    if not crisis_fingerprints:
        raise ValueError("No crisis fingerprints could be computed. Check data range.")

    all_vectors = np.array(list(crisis_fingerprints.values()))
    if scaler is None:
        scaler = RobustScaler()
        scaler.fit(all_vectors)

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
    """Extracts the most recent row as the live fingerprint vector."""
    latest = indicator_df[available_indicators].dropna().iloc[-1].values
    return scaler.transform(latest.reshape(1, -1)).flatten()


def extract_historical_fingerprint(
    indicator_df: pd.DataFrame,
    scaler: RobustScaler,
    available_indicators: list,
    date: str
) -> np.ndarray | None:
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
    Returns sorted list of dicts. Similarity is in [0, 100] range.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    results = []
    for crisis_key, crisis_vector in crisis_fingerprints.items():
        crisis_info = CRISIS_LIBRARY[crisis_key]
        sim = cosine_similarity(
            query_vector.reshape(1, -1),
            crisis_vector.reshape(1, -1)
        )[0][0]
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
    Returns dict: dimension_name -> stress_score (0-100). Higher = more stressed.
    """
    scores = {}
    for dim_name, dim_indicators in INDICATOR_GROUPS.items():
        indices = [i for i, ind in enumerate(available_indicators) if ind in dim_indicators]
        if not indices:
            scores[dim_name] = 0.0
            continue
        dim_vec = query_vector[indices]
        score = np.abs(dim_vec).mean()
        score = min(score / 3.0 * 100, 100)
        scores[dim_name] = round(score, 1)
    return scores
