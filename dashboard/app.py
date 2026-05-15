# dashboard/app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import re

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
# FIX #1: page_name -> page_title (page_name is not a valid Streamlit arg)
st.set_page_config(
    page_title="AUTOPSY",
    page_icon="\U0001f52c",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ── FIX #14: Cache heavy data fetching ────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_all_data():
    """Cached wrapper around the heavy fetch + compute pipeline."""
    raw_df = fetch_all_data(start_date="2000-01-01")
    indicator_df = compute_all_indicators(raw_df)
    return raw_df, indicator_df


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .main-title {
        font-size: 2.8rem; font-weight: 900; color: #E84393;
        letter-spacing: -1px; margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.0rem; color: #888;
        margin-top: -8px; margin-bottom: 24px;
    }
    .metric-card {
        background: #1A1D23; border-radius: 10px;
        padding: 16px 20px; border: 1px solid #2D3139;
    }
    .analogue-card {
        background: #1A1D23; border-radius: 8px;
        padding: 14px 16px; border-left: 4px solid #E84393;
        margin-bottom: 10px;
    }
    .narrative-box {
        background: #1A1D23; border-radius: 10px;
        padding: 20px 24px; border: 1px solid #2D3139;
        font-size: 0.92rem; line-height: 1.7; color: #D0D0D0;
    }
    .section-header {
        font-size: 1.1rem; font-weight: 700; color: #FAFAFA;
        margin: 20px 0 8px 0; border-bottom: 1px solid #2D3139;
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
    st.markdown("## \U0001f52c AUTOPSY")
    st.markdown("*Market Dislocation Fingerprint Intelligence*")

    st.divider()

    mode = st.radio(
        "Mode",
        options=["\U0001f534 Live Market", "\u23ea Rewind", "\U0001f5fa Heatmap"],
        index=0
    )

    st.divider()

    rewind_date = None
    heatmap_days = 60

    if mode == "\u23ea Rewind":
        rewind_date = st.date_input(
            "Select Historical Date",
            value=pd.to_datetime("2020-03-01"),
            min_value=pd.to_datetime("2005-01-01"),
            max_value=pd.to_datetime("today")
        )
        st.caption("See what AUTOPSY would have shown on this date.")

    if mode == "\U0001f5fa Heatmap":
        heatmap_days = st.selectbox("Lookback Window", [30, 60, 90, 252], index=1)

    st.divider()

    load_button = st.button("\U0001f504 Load / Refresh Data", use_container_width=True)

    if st.session_state.data_loaded:
        st.success("\u2705 Data loaded")
        if st.session_state.indicator_df is not None:
            st.caption(f"Latest: {st.session_state.indicator_df.index[-1].strftime('%Y-%m-%d')}")
    else:
        st.warning("\u26a0\ufe0f Click to load data")

    st.divider()
    st.caption("Data: FRED + Yahoo Finance\nAI: Claude (Anthropic)")


# ── Data Loading ──────────────────────────────────────────────────────────────
# FIX #11: Only trigger on explicit button press, not on first load
if load_button:
    with st.spinner("Fetching market data (this takes ~30 seconds)..."):
        try:
            raw_df, indicator_df = load_all_data()
            st.session_state.raw_df = raw_df
            st.session_state.indicator_df = indicator_df

            with st.spinner("Building crisis fingerprint library..."):
                crisis_fp, scaler, avail_ind = extract_crisis_fingerprints(indicator_df)
                st.session_state.crisis_fingerprints = crisis_fp
                st.session_state.scaler = scaler
                st.session_state.available_indicators = avail_ind

            with st.spinner("Computing live fingerprint..."):
                live_vector = extract_live_fingerprint(indicator_df, scaler, avail_ind)
                st.session_state.live_vector = live_vector
                sim_results = compute_similarity_scores(live_vector, crisis_fp)
                st.session_state.similarity_results = sim_results
                dim_scores = compute_dimension_scores(live_vector, avail_ind)
                st.session_state.dimension_scores = dim_scores

            with st.spinner("Building embedding space..."):
                crisis_coords, live_coords = build_umap_embedding(crisis_fp, live_vector)
                st.session_state.embedding_coords = crisis_coords
                st.session_state.live_embedding_coords = live_coords

            with st.spinner("Running AI analyst..."):
                narrative = run_analyst(
                    sim_results, dim_scores, avail_ind,
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
    st.markdown('<div class="subtitle">Market Dislocation Fingerprint Intelligence System</div>',
                unsafe_allow_html=True)
    st.info("\U0001f448 Click **Load / Refresh Data** in the sidebar to begin.")

    st.markdown("### What is AUTOPSY?")
    st.markdown("""
    AUTOPSY analyzes the **structural fingerprint** of current market conditions across five dimensions:
    **Liquidity \u00b7 Volatility \u00b7 Correlation \u00b7 Credit \u00b7 Positioning**

    It compares this fingerprint against pre-crisis signatures from 10 historical dislocations,
    identifies the closest structural analogues, and generates an AI-powered risk narrative.

    This is not a price prediction tool. It is a **market structure stress detector**.
    """)
    st.stop()


# ── Determine what vector/results to display ──────────────────────────────────
# FIX #6: replaced 'rewind_date' in dir() with proper variable check
if mode == "\u23ea Rewind" and rewind_date is not None:
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
    label = f"\u23ea REWIND: {display_date_label}" if is_rewind else "\U0001f534 LIVE"
    st.markdown('<div class="main-title">AUTOPSY</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">Market Dislocation Fingerprint Intelligence \u00b7 {label}</div>',
                unsafe_allow_html=True)
with col_date:
    top_analogue = display_similarity[0]
    st.metric(
        label="Top Analogue",
        value=top_analogue["short"],
        delta=f"{top_analogue['similarity']:.1f}% match"
    )


# ── Heatmap Mode ──────────────────────────────────────────────────────────────
if mode == "\U0001f5fa Heatmap":
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


# ── AI Analyst Narrative ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">AI Risk Analyst</div>', unsafe_allow_html=True)
if display_narrative:
    styled_narrative = re.sub(r'\*\*(.*?)\*\*', r'<span style="color: #E84393; font-weight: 700; font-size: 1.0rem;">\1</span>', display_narrative)
    st.markdown(f'<div class="narrative-box">{styled_narrative.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True)
else:
    st.info("AI narrative will appear after data is loaded.")


# ── Main Dashboard Layout (Live + Rewind) ─────────────────────────────────────
col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.markdown('<div class="section-header">Structural Stress Profile</div>', unsafe_allow_html=True)
    fig_radar = make_radar_chart(display_dimension)
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown('<div class="section-header">Crisis Analogue Similarity</div>', unsafe_allow_html=True)
    fig_bar = make_analogue_bar_chart(display_similarity)
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
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

    st.markdown('<div class="section-header">Fingerprint Space</div>', unsafe_allow_html=True)
    fig_embed = make_embedding_scatter(display_embedding_coords, display_live_coords)
    st.plotly_chart(fig_embed, use_container_width=True)


# ── Dimension Timeseries ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">Stress Dimension History (1 Year)</div>', unsafe_allow_html=True)
fig_ts = make_dimension_timeseries(
    st.session_state.indicator_df,
    st.session_state.available_indicators,
    days=252
)
st.plotly_chart(fig_ts, use_container_width=True)


# ── Data Sources ──────────────────────────────────────────────────────────────
with st.expander("📚 Data Sources & Methodology"):
    from data.pipeline import FRED_SERIES, YF_TICKERS

    st.markdown("### FRED Data Series (Federal Reserve Economic Data)")
    fred_cols = st.columns(2)
    fred_items = [(name, series_id) for name, series_id in FRED_SERIES.items() if series_id is not None]
    for i, (name, series_id) in enumerate(fred_items):
        url = f"https://fred.stlouisfed.org/series/{series_id}"
        fred_cols[i % 2].markdown(f"- [{name.replace('_', ' ').title()}]({url}) — `{series_id}`")

    st.markdown("### Yahoo Finance Tickers")
    yf_cols = st.columns(3)
    for i, (name, ticker) in enumerate(YF_TICKERS.items()):
        clean_ticker = ticker.replace("^", "%5E")
        url = f"https://finance.yahoo.com/quote/{clean_ticker}"
        yf_cols[i % 2].markdown(f"- [{ticker}]({url}) — {name.replace('_', ' ').title()}")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "AUTOPSY | Market Dislocation Fingerprint Intelligence | "
    "TechEx Intelligent Enterprise Solutions Hackathon 2026 | "
    "Data: FRED, Yahoo Finance | AI: Claude (Anthropic) | "
    "Not investment advice."
)
