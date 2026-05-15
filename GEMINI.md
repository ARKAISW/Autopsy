# AUTOPSY — Market Dislocation Fingerprint Intelligence System

AUTOPSY is a real-time market stress intelligence system designed to identify historical crisis analogues by analyzing the "structural fingerprint" of current market conditions.

## Project Overview

- **Purpose:** Answers "Which historical crisis does the current market structure most resemble — and what happened next?"
- **Core Methodology:**
    1. Fetches ~40 live market indicators across 5 dimensions (Liquidity, Volatility, Correlation, Credit, Positioning).
    2. Computes a "fingerprint vector" representing the current structural state.
    3. Compares the vector against fingerprints of 10 historical crises using cosine similarity.
    4. Projects fingerprints into 2D space using UMAP for visualization.
    5. Generates an AI-powered risk narrative using Claude (Anthropic API).
- **Tech Stack:**
    - **Language:** Python 3.11+ (Current environment: 3.14)
    - **Frontend:** Streamlit, Plotly
    - **Data/Math:** Pandas, NumPy, Scipy, Scikit-learn, UMAP-learn
    - **APIs:** `yfinance` (market data), `fredapi` (macro data), `anthropic` (AI agent)

## Directory Structure

- `agent/`: Contains `analyst.py` for interacting with the Anthropic API.
- `dashboard/`: Streamlit UI implementation.
    - `app.py`: Main entry point.
    - `charts.py`: Plotly visualization logic.
    - `state.py`: Session state management.
- `data/`: Data layer.
    - `pipeline.py`: Fetches raw data from FRED and Yahoo Finance.
    - `indicators.py`: Computes 40 structural indicators from raw data.
    - `crisis_library.py`: Metadata for 10 historical crisis events.
- `fingerprint/`: Core analytical engine.
    - `engine.py`: Vector normalization and similarity computation.
    - `embedding.py`: UMAP/PCA dimensionality reduction.
- `tests/`: Project tests (e.g., `test_pipeline.py`).

## Getting Started

### Environment Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration:**
   Create a `.env` file in the root directory with your API keys:
   ```env
   FRED_API_KEY=your_fred_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

### Running the Application

Launch the Streamlit dashboard:
```bash
streamlit run dashboard/app.py
```

### Running Tests

Execute tests using `pytest`:
```bash
pytest
```

## Development Conventions

- **Data Alignment:** All financial data is resampled to Business Days (`B`) and missing values are forward-filled.
- **Normalization:** Fingerprint vectors are normalized using `RobustScaler` to handle outliers in market data.
- **Type Hinting:** Use type hints for function signatures to maintain clarity in the analytical pipeline.
- **Defensive Data Fetching:** `data/pipeline.py` includes logic to handle both MultiIndex and flat column formats from `yfinance`.
- **Caching:** Heavy data operations (fetching and indicator computation) in the dashboard are cached using `@st.cache_data`.

## Key Indicators (5 Dimensions)

1. **Liquidity:** TED spread, HY/IG ratios, Yield Curve velocity.
2. **Volatility:** VIX levels, Term structure (VIX9D/VIX3M), Vol-of-Vol.
3. **Correlation:** Equity-Bond correlation, Cross-sector dispersion.
4. **Credit:** HY/IG Oasis spreads, Credit-Equity dislocations.
5. **Positioning:** Safe haven flows (Gold, JPY, CHF), Defensive sector rotations.
