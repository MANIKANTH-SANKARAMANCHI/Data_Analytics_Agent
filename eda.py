import pandas as pd
from typing import Dict, Any, Optional

from utils import get_id_columns, get_numeric_columns


def generate_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a high-level summary of the dataset,
    including shape, column names, and data types.
    """
    summary: Dict[str, Any] = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }

    # Simple info about numeric and categorical columns (exclude id columns from numeric list)
    numeric_cols = get_numeric_columns(df)
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    summary["numeric_columns"] = numeric_cols
    summary["categorical_columns"] = categorical_cols

    return summary


def describe_data(df: pd.DataFrame, max_rows: int = 50_000) -> pd.DataFrame:
    """
    Return the pandas describe table (including non-numeric columns),
    transposed so that each row represents a column.
    Large frames are sampled so unique-count stats stay responsive.
    """
    work = df
    if len(df) > max_rows:
        work = df.sample(n=max_rows, random_state=42)
    desc = work.describe(include="all").transpose()
    return desc


def compute_correlation(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Compute Pearson correlation for numeric columns only.
    Excludes identifier columns (e.g. PassengerId, S.no, ID, index).
    """
    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) < 2:
        return None

    numeric_df = df[numeric_cols]
    corr = numeric_df.corr()
    return corr