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
    """Radar chart showing stress level across 5 dimensions."""
    dims = list(dimension_scores.keys())
    scores = list(dimension_scores.values())
    dims_plot = dims + [dims[0]]
    scores_plot = scores + [scores[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores_plot, theta=dims_plot, fill='toself',
        fillcolor='rgba(232, 67, 147, 0.2)',
        line=dict(color=ACCENT_COLOR, width=2), name="Current Stress"
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=CARD_COLOR,
            radialaxis=dict(visible=True, range=[0, 100],
                          tickfont=dict(color=TEXT_COLOR, size=10), gridcolor=GRID_COLOR,
                          ticksuffix="%"),
            angularaxis=dict(tickfont=dict(color=TEXT_COLOR, size=12), gridcolor=GRID_COLOR),
        ),
        showlegend=True, paper_bgcolor=BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR,
        legend=dict(font=dict(color=TEXT_COLOR, size=11), bgcolor=CARD_COLOR,
                   bordercolor=GRID_COLOR, borderwidth=1, x=0.02, y=-0.05),
        title=dict(text=title, font=dict(color=TEXT_COLOR, size=14)),
        margin=dict(l=60, r=60, t=60, b=80), height=400,
        annotations=[dict(text="0 = No stress · 100 = Extreme stress",
                         xref="paper", yref="paper", x=0.5, y=-0.08,
                         showarrow=False, font=dict(color="#666", size=10))],
    )
    return fig


def make_analogue_bar_chart(similarity_results: list) -> go.Figure:
    """Horizontal bar chart of crisis analogue similarity scores."""
    top_n = similarity_results[:10]
    names = [r["short"] for r in reversed(top_n)]
    scores = [r["similarity"] for r in reversed(top_n)]
    colors = [r["color"] for r in reversed(top_n)]

    fig = go.Figure(go.Bar(
        x=scores, y=names, orientation='h',
        marker=dict(color=colors, opacity=0.85),
        text=[f"{s:.1f}%" for s in scores], textposition='outside',
        textfont=dict(color=TEXT_COLOR),
        name="Cosine Similarity",
    ))
    fig.update_layout(
        paper_bgcolor=BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR,
        xaxis=dict(range=[0, 110], gridcolor=GRID_COLOR,
                  tickfont=dict(color=TEXT_COLOR), ticksuffix="%",
                  title=dict(text="Structural Similarity (Cosine, 0–100%)", font=dict(color=TEXT_COLOR))),
        yaxis=dict(tickfont=dict(color=TEXT_COLOR),
                  title=dict(text="Historical Crisis", font=dict(color=TEXT_COLOR))),
        showlegend=False,
        margin=dict(l=20, r=80, t=30, b=50), height=380,
        annotations=[dict(text="Higher % = current market structure more closely resembles this crisis",
                         xref="paper", yref="paper", x=0.5, y=-0.12,
                         showarrow=False, font=dict(color="#666", size=10))],
    )
    return fig


def make_embedding_scatter(crisis_coords: dict, live_coords: tuple = None) -> go.Figure:
    """2D PCA scatter plot of crisis fingerprints + current market."""
    fig = go.Figure()

    for crisis_key, (x, y) in crisis_coords.items():
        info = CRISIS_LIBRARY.get(crisis_key, {})
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers',
            marker=dict(size=14, color=info.get("color", "#888888"),
                       opacity=0.9, line=dict(width=1.5, color='white')),
            name=info.get("short", crisis_key), showlegend=True,
            hovertemplate=f"<b>{info.get('name', crisis_key)}</b><br>{info.get('key_signature', '')}<extra></extra>"
        ))

    if live_coords is not None:
        fig.add_trace(go.Scatter(
            x=[live_coords[0]], y=[live_coords[1]], mode='markers+text',
            marker=dict(size=20, color=ACCENT_COLOR, symbol='star',
                       line=dict(width=2, color='white')),
            text=["NOW"], textposition="top center",
            textfont=dict(color=ACCENT_COLOR, size=12, family="Arial Black"),
            name="Current Market", showlegend=True,
            hovertemplate="<b>Current Market</b><extra></extra>"
        ))

    fig.update_layout(
        paper_bgcolor=BACKGROUND_COLOR, plot_bgcolor=CARD_COLOR,
        xaxis=dict(showticklabels=False, gridcolor=GRID_COLOR, zeroline=False,
                  title=dict(text="PC1 — Crisis Intensity →", font=dict(color="#888", size=11))),
        yaxis=dict(showticklabels=False, gridcolor=GRID_COLOR, zeroline=False,
                  title=dict(text="PC2 — Crisis Character →", font=dict(color="#888", size=11))),
        margin=dict(l=50, r=20, t=40, b=50), height=500,
        title=dict(text="Crisis Fingerprint Embedding Space (PCA)", font=dict(color=TEXT_COLOR, size=13)),
        legend=dict(font=dict(color=TEXT_COLOR, size=10), bgcolor=CARD_COLOR,
                   bordercolor=GRID_COLOR, borderwidth=1,
                   orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5),
        annotations=[
            dict(text="Closer points = more structurally similar",
                 xref="paper", yref="paper", x=0.5, y=-0.22,
                 showarrow=False, font=dict(color="#666", size=9)),
        ],
    )
    return fig


def make_indicator_heatmap(indicator_df: pd.DataFrame, available_indicators: list, days: int = 60) -> go.Figure:
    """Heatmap of indicator z-scores over the last N days."""
    recent = indicator_df[available_indicators].iloc[-days:].copy()
    z_scored = (recent - recent.mean()) / (recent.std() + 1e-8)
    z_scored = z_scored.clip(-3, 3)
    short_names = [ind.replace('_', ' ')[:20] for ind in available_indicators]

    fig = go.Figure(go.Heatmap(
        z=z_scored.T.values, x=z_scored.index.strftime("%Y-%m-%d"), y=short_names,
        colorscale=[[0.0, "#1a4f72"], [0.5, CARD_COLOR], [1.0, "#8B0000"]],
        zmid=0,
        colorbar=dict(title=dict(text="Z-Score", font=dict(color=TEXT_COLOR)), tickfont=dict(color=TEXT_COLOR)),
        hoverongaps=False,
    ))
    fig.update_layout(
        paper_bgcolor=BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR,
        xaxis=dict(tickangle=45, tickfont=dict(color=TEXT_COLOR, size=9),
                  title=dict(text="Date", font=dict(color=TEXT_COLOR))),
        yaxis=dict(tickfont=dict(color=TEXT_COLOR, size=9),
                  title=dict(text="Market Indicator", font=dict(color=TEXT_COLOR))),
        margin=dict(l=160, r=40, t=30, b=90), height=520,
        title=dict(text=f"Indicator Stress Heatmap (Last {days} Days)",
                  font=dict(color=TEXT_COLOR, size=13)),
        annotations=[dict(text="Blue = below average · Red = above average · Values are z-scored (std deviations from mean)",
                         xref="paper", yref="paper", x=0.5, y=-0.13,
                         showarrow=False, font=dict(color="#666", size=10))],
    )
    return fig


def make_dimension_timeseries(indicator_df: pd.DataFrame, available_indicators: list, days: int = 252) -> go.Figure:
    """Line chart showing per-dimension composite stress over time."""
    recent = indicator_df[available_indicators].iloc[-days:].copy()
    dimension_colors = {
        "Liquidity": "#E74C3C", "Volatility": "#E67E22", "Correlation": "#F1C40F",
        "Credit": "#9B59B6", "Positioning": "#3498DB",
    }

    fig = go.Figure()
    for dim_name, dim_indicators in INDICATOR_GROUPS.items():
        dim_cols = [ind for ind in dim_indicators if ind in recent.columns]
        if not dim_cols:
            continue
        dim_data = recent[dim_cols]
        normalized = (dim_data - dim_data.mean()) / (dim_data.std() + 1e-8)
        composite = normalized.abs().mean(axis=1)
        composite_smooth = composite.rolling(5, min_periods=1).mean()
        q99 = composite_smooth.quantile(0.95)
        composite_scaled = (composite_smooth / max(q99, 0.01)) * 100
        composite_scaled = composite_scaled.clip(0, 100)

        fig.add_trace(go.Scatter(
            x=recent.index, y=composite_scaled, name=dim_name,
            line=dict(color=dimension_colors.get(dim_name, "#888888"), width=2),
            mode='lines',
        ))

    fig.update_layout(
        paper_bgcolor=BACKGROUND_COLOR, plot_bgcolor=CARD_COLOR,
        xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TEXT_COLOR),
                  title=dict(text="Date", font=dict(color=TEXT_COLOR))),
        yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TEXT_COLOR),
                  title=dict(text="Composite Stress Level (0–100)", font=dict(color=TEXT_COLOR)),
                  range=[0, 105], ticksuffix="%"),
        legend=dict(font=dict(color=TEXT_COLOR, size=11), bgcolor=CARD_COLOR,
                   bordercolor=GRID_COLOR, borderwidth=1,
                   title=dict(text="Stress Dimensions", font=dict(color=TEXT_COLOR, size=11))),
        margin=dict(l=60, r=20, t=30, b=60), height=370,
        title=dict(text="Stress Dimension History", font=dict(color=TEXT_COLOR, size=13)),
        annotations=[dict(text="Each line = average absolute z-score of indicators in that dimension, scaled 0–100",
                         xref="paper", yref="paper", x=0.5, y=-0.14,
                         showarrow=False, font=dict(color="#666", size=10))],
    )
    return fig
