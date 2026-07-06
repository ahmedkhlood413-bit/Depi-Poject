"""
============================================================================
 SOCIAL MEDIA IMPACT ON MENTAL HEALTH AND DEPRESSION
 End-to-End Machine Learning Pipeline (Medallion Gold Layer -> ML Ready)
============================================================================
This script consumes the cleaned (Silver-layer) star-schema CSV exports
of the project, assembles them into a single analytical (Gold-layer)
table, validates the data, engineers the target variable, trains and
compares several classification models, and persists the best model
together with its preprocessing pipeline for production use.

Author : Senior Machine Learning Engineer
============================================================================
"""

# ============================================================================
# 1. IMPORTS
# ============================================================================
import os
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# XGBoost is an optional dependency. The pipeline degrades gracefully
# (and keeps running) if it is not installed in the target environment.
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ============================================================================
# 2. CONFIGURATION
# ============================================================================
# All paths and tunables live here so the rest of the pipeline never
# needs to be touched when the environment / data location changes.

DATA_DIR = Path("./data")          # folder containing the source CSVs
OUTPUT_DIR = Path(".")             # folder where artifacts are written

FACT_FILE = DATA_DIR / "fact_digital_lifestyle.csv"
DIM_PERSON_FILE = DATA_DIR / "dim_person.csv"
DIM_DEVICE_FILE = DATA_DIR / "dim_device.csv"
DIM_HEALTH_FILE = DATA_DIR / "dim_health_profile.csv"
DIM_GEOGRAPHY_FILE = DATA_DIR / "dim_geography.csv"

MODEL_OUTPUT_PATH = OUTPUT_DIR / "best_mental_health_model.pkl"
PREPROCESSOR_OUTPUT_PATH = OUTPUT_DIR / "preprocessing_pipeline.pkl"
RESULTS_OUTPUT_PATH = OUTPUT_DIR / "model_results.csv"

TARGET_COLUMN = "high_risk_flag"
DEPRESSION_COLUMN = "depression_score"
ANXIETY_COLUMN = "anxiety_score"

# Risk-rule thresholds (used ONLY if the target column is missing and
# must be engineered from raw clinical scores).
DEPRESSION_THRESHOLD = 7
ANXIETY_THRESHOLD = 7

# Columns that are direct components / proxies of the mental-health
# outcome itself. They are excluded from the feature set even though
# they are not row identifiers, because using them as predictors would
# leak the target (the goal is to predict risk FROM digital-lifestyle
# behaviour, not from other mental-health symptom scores). The list is
# intentionally short and explicit so the reasoning is auditable; any
# column not present in the data is simply ignored.
LEAKAGE_PRONE_COLUMNS = [
    DEPRESSION_COLUMN,
    ANXIETY_COLUMN,
    "stress_level",
    "happiness_score",
    "sleep_quality",
    "mental_health_risk",
]

# Pattern used to auto-detect surrogate/identifier keys so that no
# table-specific column names need to be hardcoded (e.g. fact_id,
# original_id, person_key, device_key, health_key, geo_key, ...).
ID_COLUMN_PATTERN = re.compile(r"(_id|_key)$", flags=re.IGNORECASE)

TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_N_FEATURES = 10


# ============================================================================
# 3. DATA LOADING
# ============================================================================
def load_dataset() -> pd.DataFrame:
    """
    Load the fact table and all dimension tables and assemble them into
    a single, denormalized analytical table (the Gold-layer view used
    for modelling). Falls back with a clear error if a file is missing.
    """
    print("=" * 70)
    print("STEP 1: DATA LOADING")
    print("=" * 70)

    required_files = {
        "fact": FACT_FILE,
        "person": DIM_PERSON_FILE,
        "device": DIM_DEVICE_FILE,
        "health": DIM_HEALTH_FILE,
        "geography": DIM_GEOGRAPHY_FILE,
    }
    missing = [str(p) for p in required_files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required source file(s): {missing}. "
            f"Update DATA_DIR / file constants in the CONFIGURATION section."
        )

    fact = pd.read_csv(required_files["fact"])
    dim_person = pd.read_csv(required_files["person"])
    dim_device = pd.read_csv(required_files["device"])
    dim_health = pd.read_csv(required_files["health"])
    dim_geo = pd.read_csv(required_files["geography"])

    # Join dimension tables onto the fact table on their natural keys.
    # Keys present on both sides are detected dynamically instead of
    # being hardcoded, so the join survives minor schema changes.
    df = fact.copy()
    for dim_name, dim_df in [
        ("dim_person", dim_person),
        ("dim_device", dim_device),
        ("dim_health_profile", dim_health),
        ("dim_geography", dim_geo),
    ]:
        join_key = [c for c in dim_df.columns if c in df.columns]
        if len(join_key) != 1:
            raise ValueError(
                f"Could not uniquely determine the join key between the "
                f"fact table and {dim_name} (candidates found: {join_key})."
            )
        df = df.merge(dim_df, on=join_key[0], how="left")

    print(f"Loaded and merged dataset shape: {df.shape}")
    return df


# ============================================================================
# 4. DATA VALIDATION
# ============================================================================
def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run final validation checks expected before data is fed into an ML
    pipeline: missing values, duplicate rows, data types and a basic
    outlier summary (IQR rule). Lightweight, non-destructive fixes
    (duplicate removal) are applied where it is unambiguously safe.
    """
    print("\n" + "=" * 70)
    print("STEP 2: DATA VALIDATION")
    print("=" * 70)

    # --- Missing values -----------------------------------------------
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print("\n[Missing Values]")
    print(missing if not missing.empty else "No missing values detected.")

    # --- Duplicate rows --------------------------------------------------
    n_dupes = df.duplicated().sum()
    print(f"\n[Duplicate Rows] {n_dupes} duplicate row(s) detected.")
    if n_dupes > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"  -> Duplicates removed. New shape: {df.shape}")

    # --- Data types --------------------------------------------------
    print("\n[Data Types]")
    print(df.dtypes)

    # --- Outlier summary (IQR rule, numeric columns only) -------------
    print("\n[Outlier Summary - IQR rule]")
    numeric_cols = df.select_dtypes(include=np.number).columns
    numeric_cols = [c for c in numeric_cols if not ID_COLUMN_PATTERN.search(c)]
    outlier_summary = {}
    for col in numeric_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_summary[col] = int(((df[col] < lower) | (df[col] > upper)).sum())
    outlier_df = pd.Series(outlier_summary, name="outlier_count").sort_values(
        ascending=False
    )
    print(outlier_df)

    return df


# ============================================================================
# 5. TARGET VARIABLE CREATION
# ============================================================================
def get_or_create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use the target column if it already exists in the dataset; otherwise
    engineer it from clinical scores using the project's risk rule:
        high_risk_flag = 1 if depression_score >= 7 OR anxiety_score >= 7
        high_risk_flag = 0 otherwise
    """
    print("\n" + "=" * 70)
    print("STEP 3: TARGET VARIABLE")
    print("=" * 70)

    if TARGET_COLUMN in df.columns:
        print(f"'{TARGET_COLUMN}' already present in the dataset - using it as-is.")
    else:
        print(f"'{TARGET_COLUMN}' not found - engineering it from the risk rule.")
        for required_col in (DEPRESSION_COLUMN, ANXIETY_COLUMN):
            if required_col not in df.columns:
                raise KeyError(
                    f"Cannot engineer '{TARGET_COLUMN}': required column "
                    f"'{required_col}' is missing from the dataset."
                )
        df[TARGET_COLUMN] = (
            (df[DEPRESSION_COLUMN] >= DEPRESSION_THRESHOLD)
            | (df[ANXIETY_COLUMN] >= ANXIETY_THRESHOLD)
        ).astype(int)

    print(f"Target distribution:\n{df[TARGET_COLUMN].value_counts(normalize=True)}")
    return df


# ============================================================================
# 6. FEATURE SELECTION
# ============================================================================
def select_features(df: pd.DataFrame):
    """
    Automatically determine the input feature set by excluding:
      - surrogate identifier / key columns (pattern-detected)
      - the target column
      - columns that are direct components/proxies of the mental-health
        outcome itself (leakage prevention)
    Remaining columns are then split into numerical vs. categorical
    based on their pandas dtype (no column names hardcoded).
    """
    print("\n" + "=" * 70)
    print("STEP 4: FEATURE SELECTION")
    print("=" * 70)

    id_columns = [c for c in df.columns if ID_COLUMN_PATTERN.search(c)]
    excluded_leakage = [c for c in LEAKAGE_PRONE_COLUMNS if c in df.columns]
    excluded = set(id_columns) | set(excluded_leakage) | {TARGET_COLUMN}

    feature_columns = [c for c in df.columns if c not in excluded]

    numeric_features = (
        df[feature_columns].select_dtypes(include=np.number).columns.tolist()
    )
    categorical_features = [
        c for c in feature_columns if c not in numeric_features
    ]

    print(f"Excluded identifier columns : {id_columns}")
    print(f"Excluded leakage-prone columns: {excluded_leakage}")
    print(f"Excluded target column      : {TARGET_COLUMN}")
    print(f"\nSelected {len(feature_columns)} input features:")
    print(f"  Numerical   ({len(numeric_features)}): {numeric_features}")
    print(f"  Categorical ({len(categorical_features)}): {categorical_features}")

    return feature_columns, numeric_features, categorical_features


# ============================================================================
# 7. PREPROCESSING PIPELINE
# ============================================================================
def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    """
    Build a ColumnTransformer that imputes, scales numerical features
    and one-hot encodes categorical features. Wrapping each branch in
    its own small Pipeline keeps the transformer robust to missing
    values even though none were found in validation.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor


# ============================================================================
# 8. MODEL TRAINING & EVALUATION
# ============================================================================
def get_candidate_models(y_train: pd.Series) -> dict:
    """Instantiate the candidate classifiers to be compared."""
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
        )
    else:
        print(
            "[WARNING] xgboost is not installed - skipping XGBoost. "
            "Run `pip install xgboost` to include it in the comparison."
        )

    return models


def train_and_evaluate(models, preprocessor, X_train, X_test, y_train, y_test):
    """
    Fit a full Pipeline (preprocessing + classifier) for every candidate
    model and compute the evaluation metrics on the held-out test set.
    """
    print("\n" + "=" * 70)
    print("STEP 6: MODEL TRAINING & EVALUATION")
    print("=" * 70)

    fitted_pipelines = {}
    results = []

    for name, model in models.items():
        print(f"\nTraining: {name}")
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", model)]
        )
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1_Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, y_proba),
            "True_Negatives": tn,
            "False_Positives": fp,
            "False_Negatives": fn,
            "True_Positives": tp,
        }
        results.append(metrics)
        fitted_pipelines[name] = pipeline

        print(
            f"  Accuracy={metrics['Accuracy']:.4f} | "
            f"Precision={metrics['Precision']:.4f} | "
            f"Recall={metrics['Recall']:.4f} | "
            f"F1={metrics['F1_Score']:.4f} | "
            f"ROC_AUC={metrics['ROC_AUC']:.4f}"
        )
        print(f"  Confusion Matrix [[TN={tn} FP={fp}] [FN={fn} TP={tp}]]")

    results_df = pd.DataFrame(results).sort_values("F1_Score", ascending=False)
    return fitted_pipelines, results_df


# ============================================================================
# 9. FEATURE IMPORTANCE
# ============================================================================
def get_feature_importance(best_pipeline: Pipeline) -> pd.DataFrame:
    """
    Extract feature importance from the best model, mapping the
    (post-one-hot-encoding) transformed feature names back to
    human-readable labels.
    """
    preprocessor = best_pipeline.named_steps["preprocessor"]
    classifier = best_pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_).ravel()
    else:
        raise AttributeError(
            f"Model {type(classifier).__name__} exposes neither "
            f"'feature_importances_' nor 'coef_'."
        )

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    return importance_df


# ============================================================================
# 10. MAIN PIPELINE ORCHESTRATION
# ============================================================================
def main():
    # ---- Load & validate -------------------------------------------
    df = load_dataset()
    df = validate_dataset(df)
    df = get_or_create_target(df)

    # ---- Feature selection -------------------------------------------
    feature_columns, numeric_features, categorical_features = select_features(df)

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    # ---- Train / test split -------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: TRAIN-TEST SPLIT")
    print("=" * 70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    # ---- Preprocessing -------------------------------------------
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # ---- Train & evaluate candidate models -------------------------------------------
    models = get_candidate_models(y_train)
    fitted_pipelines, results_df = train_and_evaluate(
        models, preprocessor, X_train, X_test, y_train, y_test
    )

    # ---- Best model selection -------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 7: BEST MODEL SELECTION")
    print("=" * 70)
    best_model_name = results_df.iloc[0]["Model"]
    best_f1 = results_df.iloc[0]["F1_Score"]
    best_pipeline = fitted_pipelines[best_model_name]
    print(f"Best model (by F1 Score): {best_model_name} (F1 = {best_f1:.4f})")

    # ---- Feature importance -------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 8: FEATURE IMPORTANCE")
    print("=" * 70)
    importance_df = get_feature_importance(best_pipeline)
    print(importance_df.head(TOP_N_FEATURES).to_string(index=False))

    # ---- Persist artifacts -------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 9: SAVING ARTIFACTS")
    print("=" * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, MODEL_OUTPUT_PATH)
    print(f"Best model saved to        : {MODEL_OUTPUT_PATH}")

    # Re-fit a standalone copy of the preprocessor on the training data
    # so it can be persisted/reused independently of any one classifier.
    standalone_preprocessor = build_preprocessor(numeric_features, categorical_features)
    standalone_preprocessor.fit(X_train)
    joblib.dump(standalone_preprocessor, PREPROCESSOR_OUTPUT_PATH)
    print(f"Preprocessing pipeline saved to: {PREPROCESSOR_OUTPUT_PATH}")

    results_df.to_csv(RESULTS_OUTPUT_PATH, index=False)
    print(f"Model evaluation results saved to: {RESULTS_OUTPUT_PATH}")

    # ---- Final summary -------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Best Model       : {best_model_name}")
    print(f"Best F1 Score    : {best_f1:.4f}")
    print(f"\nTop {TOP_N_FEATURES} Important Features:")
    print(importance_df.head(TOP_N_FEATURES).to_string(index=False))


if __name__ == "__main__":
    main()
