import re
import pandas as pd
from typing import Tuple, Dict, Any, List


def normalize_column_name(name: str) -> str:
    """Normalize a column name: lowercase, replace spaces/special chars with underscore."""
    if not isinstance(name, str):
        name = str(name)
    s = name.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed"


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform basic data cleaning steps:
    - Normalize column names (lowercase, underscores)
    - Remove duplicate rows
    - Handle missing values (numeric -> median, categorical -> mode)
    - Attempt to convert object columns to numeric or datetime where possible

    Returns a tuple of:
    - cleaned DataFrame
    - cleaning report (dictionary with simple explanations)
    """
    report: Dict[str, Any] = {}

    original_shape = df.shape
    report["original_shape"] = {"rows": original_shape[0], "columns": original_shape[1]}

    # Normalize column names
    original_columns = list(df.columns)
    new_columns = []
    name_counts: Dict[str, int] = {}
    for col in original_columns:
        base = normalize_column_name(col)
        if base in name_counts:
            name_counts[base] += 1
            base = f"{base}_{name_counts[base]}"
        else:
            name_counts[base] = 1
        new_columns.append(base)
    df_clean = df.copy()
    df_clean.columns = new_columns
    report["normalized_columns"] = [
        {"original": o, "normalized": n} for o, n in zip(original_columns, new_columns)
    ]

    # Remove duplicates
    rows_before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    report["duplicates_removed"] = rows_before - len(df_clean)

    # Track missing values before handling
    report["missing_values_before"] = df_clean.isnull().sum().to_dict()

    # Handle missing values column-wise
    for col in df_clean.columns:
        if df_clean[col].dtype.kind in "biufc":  # numeric-like
            median_value = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_value)
        else:
            mode_series = df_clean[col].mode()
            if not mode_series.empty:
                df_clean[col] = df_clean[col].fillna(mode_series[0])
            else:
                df_clean[col] = df_clean[col].fillna("Unknown")

    report["missing_values_after"] = df_clean.isnull().sum().to_dict()

    # Infer better dtypes for object columns (avoid slow full-column datetime parsing on text fields)
    converted_columns = []
    object_cols = list(df_clean.select_dtypes(include=["object"]).columns)
    for col in object_cols:
        s = df_clean[col]
        as_num = pd.to_numeric(s, errors="coerce")
        num_ratio = float(as_num.notna().mean()) if len(s) else 0.0
        if num_ratio >= 0.95:
            df_clean[col] = as_num
            converted_columns.append((col, "numeric"))
            continue

        # Datetime only when name suggests it or cardinality is very low (skip city/name-like columns)
        nu = s.nunique(dropna=True)
        name_hint = any(
            h in col.lower()
            for h in ("date", "time", "timestamp", "dob", "birth", "created", "updated")
        )
        if name_hint or (nu <= 48 and nu > 0):
            dt = pd.to_datetime(s, errors="coerce", utc=False)
            dt_ratio = float(dt.notna().mean()) if len(s) else 0.0
            if dt_ratio >= 0.8:
                df_clean[col] = dt
                converted_columns.append((col, "datetime"))
                continue

        converted_columns.append((col, "object (unchanged)"))

    report["type_conversions"] = [
        {"column": c, "new_type": t} for c, t in converted_columns
    ]

    report["final_shape"] = {
        "rows": df_clean.shape[0],
        "columns": df_clean.shape[1],
    }

    return df_clean, report