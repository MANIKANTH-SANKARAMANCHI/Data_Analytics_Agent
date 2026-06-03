import io
from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

from utils import get_numeric_columns, get_categorical_columns


def plot_histogram(df: pd.DataFrame, column: str):
    """
    Create an interactive histogram for a numeric column using Plotly.
    """
    fig = px.histogram(df, x=column, nbins=30, title=f"Histogram of {column}")
    fig.update_layout(bargap=0.1)
    return fig


def plot_box(df: pd.DataFrame, column: str):
    """Create a box plot for a numeric column using Plotly."""
    fig = px.box(df, y=column, title=f"Box Plot of {column}")
    return fig


def plot_distribution(df: pd.DataFrame, column: str):
    """Create a distribution (KDE-style) plot for a numeric column using Plotly."""
    fig = px.histogram(
        df, x=column, nbins=40, histnorm="probability density",
        title=f"Distribution of {column}"
    )
    fig.update_traces(opacity=0.7)
    return fig


def plot_bar_categorical(df: pd.DataFrame, column: str, top_n: int = 20):
    """Bar chart of category counts for a categorical column."""
    counts = df[column].value_counts().head(top_n)
    fig = px.bar(
        x=counts.index.astype(str),
        y=counts.values,
        title=f"Bar Chart: {column}",
        labels={"x": column, "y": "Count"},
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def plot_frequency_categorical(df: pd.DataFrame, column: str, top_n: int = 20):
    """Frequency (pie or bar) chart for a categorical column."""
    counts = df[column].value_counts().head(top_n)
    fig = px.pie(
        values=counts.values,
        names=counts.index.astype(str),
        title=f"Frequency: {column}",
    )
    return fig


def plot_correlation_heatmap_matplotlib(df: pd.DataFrame) -> Optional[plt.Figure]:
    """
    Create a correlation heatmap for numeric columns using Seaborn.
    Excludes identifier columns. Returns a Matplotlib figure or None.
    """
    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) < 2:
        return None

    numeric_df = df[numeric_cols]
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Create an interactive correlation heatmap using Plotly.
    Excludes identifier columns. Returns a Plotly figure or None.
    """
    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) < 2:
        return None

    numeric_df = df[numeric_cols]
    corr = numeric_df.corr()
    fig = px.imshow(
        corr,
        x=corr.columns,
        y=corr.columns,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Heatmap",
    )
    fig.update_layout(height=500)
    return fig


def plot_pairplot(df: pd.DataFrame, max_vars: int = 5) -> Optional[plt.Figure]:
    """
    Create a Seaborn pairplot (scatterplot matrix) for up to `max_vars` numeric columns.
    """
    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) < 2:
        return None

    cols_to_plot: List[str] = numeric_cols[:max_vars]
    plot_df = df[cols_to_plot]
    if plot_df.shape[0] > 500:
        plot_df = plot_df.sample(500, random_state=42)

    sns.set(style="ticks")
    pairgrid = sns.pairplot(plot_df)
    pairgrid.fig.suptitle("Pairplot (sample of numeric features)", y=1.02)
    return pairgrid.fig


def plot_feature_importance(feature_importances: pd.DataFrame):
    """
    Plot feature importances using Plotly bar chart.
    Expects a DataFrame with columns: 'feature' and 'importance'.
    """
    df_sorted = feature_importances.sort_values("importance", ascending=True)
    fig = px.bar(
        df_sorted,
        x="importance",
        y="feature",
        orientation="h",
        title="Feature Importance",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


def figure_to_png_bytes(fig: plt.Figure, dpi: int = 110) -> bytes:
    """Serialize a Matplotlib figure to PNG bytes and close the figure."""
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    buf.seek(0)
    return buf.read()


def _plot_histogram_matplotlib(df: pd.DataFrame, column: str) -> Optional[plt.Figure]:
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if len(series) == 0:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(series.values, bins=30, edgecolor="black", alpha=0.75)
    ax.set_title(f"Histogram: {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def _plot_bar_categorical_matplotlib(df: pd.DataFrame, column: str, top_n: int = 15) -> Optional[plt.Figure]:
    counts = df[column].dropna().astype(str).value_counts().head(top_n)
    if counts.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(counts)), counts.values, color="steelblue", alpha=0.85)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=45, ha="right")
    ax.set_title(f"Top categories: {column}")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def _plot_feature_importance_matplotlib(feature_importances: pd.DataFrame) -> Optional[plt.Figure]:
    if feature_importances.empty:
        return None
    df_sorted = feature_importances.sort_values("importance", ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(8, max(4.0, len(df_sorted) * 0.22)))
    ax.barh(df_sorted["feature"].astype(str), df_sorted["importance"], color="teal", alpha=0.85)
    ax.set_title("Feature importance (top 20)")
    fig.tight_layout()
    return fig


def build_report_chart_pngs(
    df: pd.DataFrame,
    feature_importances: Optional[pd.DataFrame] = None,
) -> list[tuple[str, bytes]]:
    """
    Build a small set of static charts for embedding in PDF/HTML exports.
    Uses Matplotlib only (no Kaleido) for reliable headless rendering.
    """
    images: list[tuple[str, bytes]] = []

    fig = plot_correlation_heatmap_matplotlib(df)
    if fig is not None:
        images.append(("Correlation heatmap", figure_to_png_bytes(fig)))

    numeric_cols = get_numeric_columns(df)
    if numeric_cols:
        fig = _plot_histogram_matplotlib(df, numeric_cols[0])
        if fig is not None:
            images.append((f"Histogram: {numeric_cols[0]}", figure_to_png_bytes(fig)))

    categorical_cols = get_categorical_columns(df)
    if categorical_cols:
        fig = _plot_bar_categorical_matplotlib(df, categorical_cols[0])
        if fig is not None:
            images.append((f"Category counts: {categorical_cols[0]}", figure_to_png_bytes(fig)))

    if feature_importances is not None and not feature_importances.empty:
        fig = _plot_feature_importance_matplotlib(feature_importances)
        if fig is not None:
            images.append(("Feature importance", figure_to_png_bytes(fig)))

    return images