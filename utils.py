import re
from typing import List, Optional, Dict, Any

import pandas as pd

from data_cleaning import normalize_column_name

# Exact names (normalized/lowercase) that are clearly identifiers
ID_LIKE_NAMES = {
    "s.no",
    "index",
    "passengerid",
    "customerid",
    "userid",
    "orderid",
    "transactionid",
    "productid",
    "rowid",
    "uuid",
}


def _column_name_tokens(col: str) -> List[str]:
    """Split a column name into lowercase tokens (handles snake_case and camelCase)."""
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(col))
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


def _is_id_like_column(col: str) -> bool:
    """
    True if the column name indicates an identifier.

    IMPORTANT: Do not use substring ('id' in name) — that wrongly matches 'age',
    'idea', etc. We require a dedicated id token or explicit patterns.
    """
    c = col.lower().strip()
    if c in ID_LIKE_NAMES:
        return True
    if c == "id" or c.endswith("_id"):
        return True
    tokens = _column_name_tokens(col)
    if "id" in tokens:
        return True
    return False


def get_id_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect identifier columns to exclude from visualizations and ML.
    Uses whole-token matching (e.g. ``customer_id``, ``PassengerId``), not naive substring search.
    """
    return [col for col in df.columns if _is_id_like_column(col)]


def get_numeric_columns(df: pd.DataFrame, exclude_id_columns: bool = True) -> List[str]:
    """
    Return a list of numeric column names in the DataFrame.
    By default excludes identifier columns (e.g. PassengerId, S.no, ID, index).
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if exclude_id_columns:
        id_columns = get_id_columns(df)
        numeric_cols = [col for col in numeric_cols if col not in id_columns]
    return numeric_cols


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """
    Return a list of non-numeric (categorical) column names in the DataFrame.
    """
    return df.select_dtypes(exclude=["number"]).columns.tolist()


def resolve_target_in_cleaned(
    cleaned_df: pd.DataFrame,
    target_original: Optional[str],
    cleaning_report: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Map the user's selected column name (from the raw upload) to the column name
    present in cleaned_df after normalization.
    """
    if target_original is None or cleaned_df is None or cleaned_df.empty:
        return None
    if target_original in cleaned_df.columns:
        return target_original
    mapping: Dict[str, str] = {}
    if cleaning_report and cleaning_report.get("normalized_columns"):
        for item in cleaning_report["normalized_columns"]:
            o, n = item.get("original"), item.get("normalized")
            if o is not None and n is not None:
                mapping[str(o)] = str(n)
    resolved = mapping.get(str(target_original))
    if resolved is not None and resolved in cleaned_df.columns:
        return resolved
    norm = normalize_column_name(str(target_original))
    if norm in cleaned_df.columns:
        return norm
    suffix_matches = [c for c in cleaned_df.columns if c.startswith(f"{norm}_")]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    return None


def detect_problem_type(df: pd.DataFrame, target_column: str) -> str:
    """
    Detect the problem type (classification or regression) based on the target column.
    This is a convenience wrapper that uses similar logic to model_training._infer_problem_type.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    y = df[target_column]

    if y.dtype.kind in "Ocb" or str(y.dtype) == "category":
        return "classification"
    if y.dtype.kind in "biufc":
        n = len(y)
        if n == 0:
            return "classification"
        unique = y.nunique(dropna=True)
        if unique <= max(10, int(0.05 * n)):
            return "classification"
        return "regression"
    return "classification"