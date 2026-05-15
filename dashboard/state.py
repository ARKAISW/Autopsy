# dashboard/state.py

import streamlit as st
import pandas as pd
from datetime import datetime


def initialize_session_state():
    """Initialize all session state variables if not already set."""
    defaults = {
        "data_loaded": False,
        "raw_df": None,
        "indicator_df": None,
        "crisis_fingerprints": None,
        "scaler": None,
        "available_indicators": None,
        "live_vector": None,
        "similarity_results": None,
        "dimension_scores": None,
        "analyst_narrative": None,
        "rewind_date": None,
        "mode": "live",
        "embedding_coords": None,
        "live_embedding_coords": None,
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
