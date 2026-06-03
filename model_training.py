from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils import get_id_columns

# Cap rows used for ML so large uploads stay interactive (full data still used for EDA/cleaning).
ML_MAX_TRAIN_ROWS = 35_000


def infer_problem_type(y: pd.Series) -> str:
    """
    Classification if target is non-numeric (object/category/bool) or numeric with
    relatively few distinct values; otherwise regression.
    """
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


def preprocess_data(X: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categoricals, coerce datetimes and numerics, impute missing values.
    """
    if X.empty:
        return X
    X_work = X.copy()
    X_encoded = pd.get_dummies(X_work, drop_first=True)

    datetime_cols = X_encoded.select_dtypes(include=["datetime", "datetimetz"]).columns
    for col in datetime_cols:
        dt_series = pd.to_datetime(X_encoded[col], errors="coerce")
        X_encoded[col] = dt_series.astype("int64")

    iNaT = np.iinfo(np.int64).min
    X_encoded = X_encoded.replace(iNaT, np.nan)
    X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce")
    medians = X_encoded.median(numeric_only=True)
    X_encoded = X_encoded.fillna(medians).fillna(0)
    return X_encoded


def split_features_target(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """X = predictors, y = target; drops target and ID-like columns from X."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    y = df[target_column]
    id_columns = get_id_columns(df)
    drop_cols = list({target_column} | {c for c in id_columns if c in df.columns})
    X = df.drop(columns=drop_cols)

    if X.shape[1] == 0:
        raise ValueError(
            "No usable feature columns available after excluding target/ID columns. "
            "Please upload a dataset with additional predictor columns."
        )
    return X, y


def prepare_xy(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Full feature matrix and target, ready for sklearn."""
    X, y = split_features_target(df, target_column)
    X_encoded = preprocess_data(X)
    if X_encoded.shape[1] == 0:
        raise ValueError(
            "Feature encoding produced no usable columns. "
            "Please choose a different target or upload richer input data."
        )
    return X_encoded, y


def evaluate_predictions(y_true: Any, y_pred: Any, problem_type: str) -> Dict[str, float]:
    """Return metric dict given true labels and predictions."""
    if problem_type == "classification":
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2_score": float(r2_score(y_true, y_pred)),
    }


def evaluate_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    *,
    label_encoder: Optional[LabelEncoder] = None,
) -> Dict[str, float]:
    """
    Evaluate a fitted estimator on a feature matrix and target.
    For classification trained on encoded labels, pass the same LabelEncoder used in fit.
    """
    y_pred = model.predict(X)
    if problem_type == "classification":
        if label_encoder is not None:
            y_pred = label_encoder.inverse_transform(np.asarray(y_pred, dtype=int))
        return evaluate_predictions(y.astype(str), np.asarray(y_pred).astype(str), problem_type)
    y_true_num = pd.to_numeric(y, errors="coerce")
    fill = float(y_true_num.median())
    y_true_num = y_true_num.fillna(fill)
    y_pred_num = np.asarray(y_pred, dtype=float)
    return evaluate_predictions(y_true_num, y_pred_num, "regression")


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    random_state: int = 42,
) -> Tuple[BaseEstimator, str]:
    """
    Train a single strong default model (Random Forest) for the given problem type.
    Classification uses encoded integer labels internally for consistency.
    """
    if problem_type == "classification":
        le = LabelEncoder()
        y_fit = le.fit_transform(y.astype(str))
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        model.fit(X, y_fit)
        model._label_encoder = le  # type: ignore[attr-defined]
        return model, problem_type

    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    y_num = pd.to_numeric(y, errors="coerce")
    if y_num.isna().all():
        raise ValueError("Target column could not be converted to numeric for regression.")
    y_num = y_num.fillna(y_num.median())
    model.fit(X, y_num)
    return model, problem_type


def generate_simple_ml_insights(
    target_column: str,
    problem_type: str,
    model_results: Dict[str, Dict],
    feature_importances: Optional[pd.DataFrame],
    y: Optional[pd.Series] = None,
) -> List[str]:
    """Short rule-based bullets after training."""
    lines: List[str] = []
    if feature_importances is not None and not feature_importances.empty:
        top = feature_importances.sort_values("importance", ascending=False).iloc[0]
        lines.append(
            f"Top feature influencing prediction is **{top['feature']}** "
            f"(importance {float(top['importance']):.4f})."
        )
    if y is not None and problem_type == "classification":
        vc = y.astype(str).value_counts(normalize=True)
        if len(vc) > 1 and float(vc.max()) > 0.7:
            lines.append(
                "Target variable appears **imbalanced** — one class dominates a large share of rows."
            )
    if model_results:
        if problem_type == "classification":
            best_acc = max(
                (m.get("accuracy") or 0) for m in model_results.values() if "accuracy" in m
            )
            if best_acc >= 0.85:
                lines.append("Model performance suggests **strong** predictability on the test split.")
            elif best_acc >= 0.65:
                lines.append("Model performance suggests **moderate** predictability.")
            else:
                lines.append("Model performance is **limited** — target may be hard to predict from features.")
        else:
            best_r2 = max((m.get("r2_score") or -999) for m in model_results.values() if "r2_score" in m)
            if best_r2 >= 0.6:
                lines.append("R² indicates a **reasonably good** fit on the test split.")
            elif best_r2 >= 0.25:
                lines.append("R² suggests **moderate** explanatory power.")
            else:
                lines.append("R² is **low** — linear/forest models explain little variance here.")
    if not lines:
        lines.append("Run training with more diverse features to deepen insight.")
    return lines


def _subsample_for_ml(
    X: pd.DataFrame, y: pd.Series, problem_type: str
) -> Tuple[pd.DataFrame, pd.Series, Optional[str]]:
    """Reduce row count for fitting; stratify classification when possible."""
    n = len(X)
    if n <= ML_MAX_TRAIN_ROWS:
        return X, y, None
    y_str = y.astype(str)
    try:
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=ML_MAX_TRAIN_ROWS, random_state=42
        )
        train_idx, _ = next(sss.split(np.zeros((n, 1)), y_str))
    except ValueError:
        rng = np.random.RandomState(42)
        train_idx = rng.choice(n, size=ML_MAX_TRAIN_ROWS, replace=False)
    X_s = X.iloc[train_idx].reset_index(drop=True)
    y_s = y.iloc[train_idx].reset_index(drop=True)
    note = (
        f"Training used **{ML_MAX_TRAIN_ROWS:,}** rows (stratified sample of **{n:,}**) "
        "so models finish quickly; cleaning and EDA still use the full dataset."
    )
    return X_s, y_s, note


def train_models(
    df: pd.DataFrame, target_column: str
) -> Tuple[Dict[str, Dict], Optional[pd.DataFrame], str, Optional[str]]:
    """
    Train baseline models, infer problem type from y, return per-model metrics and
    feature importances from Random Forest.

    Returns a fourth value: optional user-facing note (e.g. row subsampling for speed).
    """
    X, y = prepare_xy(df, target_column)
    problem_type = infer_problem_type(y)
    X, y, subsample_note = _subsample_for_ml(X, y, problem_type)

    n_train_after_split = int(len(X) * 0.8)
    n_features = max(X.shape[1], 1)
    # Fewer trees on wide/large matrices keeps the UI responsive.
    rf_trees = min(
        100,
        max(30, 2_000_000 // max(n_train_after_split * n_features // 50, 1)),
    )
    log_max_iter = 400 if n_train_after_split * n_features > 2_000_000 else 1000

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_results: Dict[str, Dict] = {}
    feature_importances_df: Optional[pd.DataFrame] = None

    if problem_type == "classification":
        le = LabelEncoder()
        y_train_e = le.fit_transform(y_train.astype(str))
        y_test_e = le.transform(y_test.astype(str))

        models: Dict[str, Any] = {
            "Logistic Regression": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=log_max_iter,
                            random_state=42,
                            solver="lbfgs",
                        ),
                    ),
                ]
            ),
            "Random Forest Classifier": RandomForestClassifier(
                n_estimators=rf_trees,
                random_state=42,
                n_jobs=-1,
            ),
        }

        for name, model in models.items():
            model.fit(X_train, y_train_e)
            y_pred_e = model.predict(X_test)
            y_pred_labels = le.inverse_transform(np.asarray(y_pred_e, dtype=int))

            metrics = evaluate_predictions(y_test.astype(str), y_pred_labels, "classification")
            model_results[name] = metrics

            if isinstance(model, RandomForestClassifier):
                importances = model.feature_importances_
                feature_importances_df = pd.DataFrame(
                    {"feature": X.columns, "importance": importances}
                )

    else:
        y_train_num = pd.to_numeric(y_train, errors="coerce")
        y_test_num = pd.to_numeric(y_test, errors="coerce")
        fill = y_train_num.median()
        y_train_num = y_train_num.fillna(fill)
        y_test_num = y_test_num.fillna(fill)

        models = {
            "Linear Regression": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("reg", LinearRegression()),
                ]
            ),
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=rf_trees,
                random_state=42,
                n_jobs=-1,
            ),
        }

        for name, model in models.items():
            model.fit(X_train, y_train_num)
            y_pred = model.predict(X_test)
            metrics = evaluate_predictions(y_test_num, y_pred, "regression")
            model_results[name] = metrics

            if isinstance(model, RandomForestRegressor):
                importances = model.feature_importances_
                feature_importances_df = pd.DataFrame(
                    {"feature": X.columns, "importance": importances}
                )

    return model_results, feature_importances_df, problem_type, subsample_note
