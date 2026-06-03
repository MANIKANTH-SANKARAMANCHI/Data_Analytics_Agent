"""Data quality scoring and narrative insights from ``cleaning_report``."""

from typing import Any, Dict, List

import pandas as pd


def build_data_quality_insights(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Derive score, comparison table, issues, ML impact, and recommendations from
    ``cleaning_report`` (output of ``data_cleaning.clean_data``).
    """
    before_mv = report.get("missing_values_before") or {}
    after_mv = report.get("missing_values_after") or {}
    missing_before = int(
        sum(v for v in before_mv.values() if isinstance(v, (int, float)))
    )
    missing_after = int(
        sum(v for v in after_mv.values() if isinstance(v, (int, float)))
    )

    dup_removed = int(report.get("duplicates_removed") or 0)
    duplicates_before = dup_removed
    duplicates_after = 0

    final = report.get("final_shape") or {}
    orig = report.get("original_shape") or {}
    n_rows = int(final.get("rows") or orig.get("rows") or 0)
    n_cols = int(final.get("columns") or orig.get("columns") or 0)
    total_cells = max(1, n_rows * n_cols)
    pct_missing = 100.0 * missing_before / total_cells

    conversions = report.get("type_conversions") or []
    type_fixes = sum(
        1
        for c in conversions
        if c.get("new_type") in ("numeric", "datetime")
    )
    if type_fixes > 0:
        type_penalty = min(5, 2 + max(0, type_fixes - 1))
    else:
        type_penalty = 0

    score = 100.0
    score -= min(100.0, pct_missing)
    if dup_removed > 0:
        score -= 2
    score -= type_penalty
    score_int = int(max(0, min(100, round(score))))

    comparison_df = pd.DataFrame(
        {
            "Metric": ["Missing Values", "Duplicate Rows"],
            "Before": [missing_before, duplicates_before],
            "After": [missing_after, duplicates_after],
        }
    )

    issues: List[str] = []
    if missing_before > 0:
        issues.append(f"{missing_before} missing values detected")
    if duplicates_before > 0:
        issues.append(f"{duplicates_before} duplicate rows detected")

    impact: List[str] = []
    if missing_before > 0:
        impact.append("Missing values could reduce model accuracy")
    if duplicates_before > 0:
        impact.append("Duplicate rows may cause overfitting")
    if not impact:
        impact.append("Data is clean and suitable for reliable ML models")

    recommendations: List[str] = []
    if missing_before > 0:
        recommendations.append(
            "Consider better data collection to reduce missing values"
        )
    if duplicates_before > 0:
        recommendations.append(
            "Ensure duplicate records are avoided during data entry"
        )
    recommendations.append("Monitor incoming data for consistency")

    return {
        "score": score_int,
        "comparison_df": comparison_df,
        "issues": issues,
        "impact": impact,
        "recommendations": recommendations,
    }
