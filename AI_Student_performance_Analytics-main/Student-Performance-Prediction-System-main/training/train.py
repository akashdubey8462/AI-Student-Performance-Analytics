"""ScholarIQ Model Training and Evaluation Pipeline.
Implements leak-free preprocessing, train/validation/test separation,
cross-validation-only model selection, held-out test evaluation,
and complete feature provenance documentation.
"""
import os
import sys
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TRAIN_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import json
import argparse
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from src.config import DATASET_DIR, MODEL_DIR, FEATURE_COLUMNS


FEATURE_PROVENANCE = {
    "dataset_source": "UCI Machine Learning Repository - Student Performance Dataset (Cortez and Silva, 2008)",
    "original_context": "Secondary education (High School) Portuguese language course at two Portuguese schools",
    "intended_purpose": "Research benchmark baseline only. NOT a validated university semester result predictor.",
    "features": {
        "Gender": {
            "source_column": "sex",
            "type": "Recorded",
            "transformation": "Binary encoded: M -> 1, F -> 0",
            "college_availability": "Available upon enrollment"
        },
        "Age": {
            "source_column": "age",
            "type": "Recorded",
            "transformation": "Integer pass-through (15-22)",
            "college_availability": "Available upon enrollment"
        },
        "Self_Study_Hours": {
            "source_column": "studytime",
            "type": "Constructed Proxy",
            "transformation": "Categorical studytime mapped to continuous weekly hours: 1->1.5h, 2->3.0h, 3->5.0h, 4->7.0h",
            "college_availability": "Self-reported survey metric"
        },
        "Previous_Failures": {
            "source_column": "failures",
            "type": "Recorded",
            "transformation": "Integer count of previous class failures (0-3)",
            "college_availability": "Available from institutional academic history (backlogs)"
        },
        "Extra_Classes": {
            "source_column": "schoolsup",
            "type": "Recorded",
            "transformation": "Extra educational school support: yes -> 1, no -> 0",
            "college_availability": "Recorded remedial course attendance"
        },
        "Attendance_Percentage": {
            "source_column": "absences",
            "type": "Constructed Proxy",
            "transformation": "Calculated as (100.0 - absences), clipped between 20.0% and 100.0%",
            "college_availability": "Recorded biometric / LMS attendance percentage"
        },
        "Assignments_Submitted": {
            "source_column": "studytime, higher, internet, activities",
            "type": "Synthetic Composite Proxy",
            "transformation": "(studytime*1.5 + higher*2.0 + internet*1.5 + activities*1.0) clipped [1, 10]",
            "college_availability": "Recorded LMS assignment count"
        },
        "Internal_Marks": {
            "source_column": "G1",
            "type": "Domain-Transformed Proxy",
            "transformation": "First period Portuguese grade (0-20 scale) divided by 2.0 (0-10 scale). G2 and G3 excluded to prevent target leakage.",
            "college_availability": "Mid-semester examination average"
        }
    },
    "target": {
        "training_target": "G3 * 5 (0-100 scale, final secondary school Portuguese grade)",
        "intended_college_target": "End-of-semester composite score based on Mid-Semesters, Activities, External Practical (30), Term Work (20), PBL (50), and Internship (50)",
        "validity_note": "Predictions from this model reflect high-school language performance correlations and must be explicitly labelled as a Research Baseline rather than a validated college engineering result."
    }
}


def build_clean_dataset(raw_path: str = None) -> pd.DataFrame:
    """Preprocess UCI raw dataset into leak-free benchmark format."""
    path = raw_path or (DATASET_DIR / "student_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw dataset not found at {path}")

    df_raw = pd.read_csv(path)
    clean_df = pd.DataFrame()
    clean_df["Student_Name"] = [f"Student {i+1}" for i in range(len(df_raw))]
    clean_df["Gender"] = (df_raw["sex"] == "M").astype(int)
    clean_df["Age"] = df_raw["age"].astype(int)

    study_map = {1: 1.5, 2: 3.0, 3: 5.0, 4: 7.0}
    clean_df["Self_Study_Hours"] = df_raw["studytime"].map(study_map)
    clean_df["Previous_Failures"] = df_raw["failures"].astype(int)
    clean_df["Extra_Classes"] = (df_raw["schoolsup"] == "yes").astype(int)
    clean_df["Attendance_Percentage"] = (100.0 - df_raw["absences"]).clip(lower=20.0, upper=100.0).round(1)

    assign_score = (
        (df_raw["studytime"] * 1.5) +
        ((df_raw["higher"] == "yes").astype(int) * 2.0) +
        ((df_raw["internet"] == "yes").astype(int) * 1.5) +
        ((df_raw["activities"] == "yes").astype(int) * 1.0)
    ).clip(lower=1.0, upper=10.0).round()
    clean_df["Assignments_Submitted"] = assign_score.astype(int)

    # Strictly G1 divided by 2 (0-10 scale). G2 is excluded to prevent leakage.
    clean_df["Internal_Marks"] = (df_raw["G1"] / 2.0).round(1)
    clean_df["Final_Score"] = (df_raw["G3"] * 5).astype(int)

    clean_csv_path = DATASET_DIR / "student_performance_v2_clean.csv"
    clean_df.to_csv(clean_csv_path, index=False)
    return clean_df


def run_training_pipeline(test_size: float = 0.20, random_state: int = 42) -> dict:
    """Train candidate models, select best model strictly via cross-validation on training data,
    and evaluate selected model once on held-out test data.
    """
    clean_df = build_clean_dataset()
    X = clean_df[FEATURE_COLUMNS]
    y = clean_df["Final_Score"]

    # 1. Train / Test Split (Held-out test set remains completely untouched during model selection)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    candidate_models = {
        "Baseline (Dummy Mean)": DummyRegressor(strategy="mean"),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest (Tuned)": RandomForestRegressor(
            n_estimators=150, max_depth=6, min_samples_split=5, min_samples_leaf=2, random_state=random_state
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=100, max_depth=4, min_samples_leaf=5, random_state=random_state
        )
    }

    # 2. 5-Fold Cross-Validation strictly on Training Data
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_benchmarks = []

    for name, model in candidate_models.items():
        scoring = ["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"]
        cv_res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False)

        cv_mae_mean = -cv_res["test_neg_mean_absolute_error"].mean()
        cv_mae_std = cv_res["test_neg_mean_absolute_error"].std()
        cv_rmse_mean = -cv_res["test_neg_root_mean_squared_error"].mean()
        cv_rmse_std = cv_res["test_neg_root_mean_squared_error"].std()
        cv_r2_mean = cv_res["test_r2"].mean()
        cv_r2_std = cv_res["test_r2"].std()

        cv_benchmarks.append({
            "model_name": name,
            "cv_mae_mean": round(cv_mae_mean, 2),
            "cv_mae_std": round(cv_mae_std, 2),
            "cv_rmse_mean": round(cv_rmse_mean, 2),
            "cv_rmse_std": round(cv_rmse_std, 2),
            "cv_r2_mean": round(cv_r2_mean, 3),
            "cv_r2_std": round(cv_r2_std, 3),
            "instance": model
        })

    # 3. Model Selection: Select best model STRICTLY by Training CV R² (No test set peeking!)
    best_candidate = sorted(cv_benchmarks, key=lambda x: x["cv_r2_mean"], reverse=True)[0]
    best_name = best_candidate["model_name"]
    best_estimator = best_candidate["instance"]

    print(f"Selected Best Model via 5-Fold CV on Training Split: {best_name} (CV R² = {best_candidate['cv_r2_mean']} ± {best_candidate['cv_r2_std']})")

    # 4. Final Training: Fit selected estimator on the entire training set
    best_estimator.fit(X_train, y_train)

    # 5. One-Time Evaluation on Held-Out Test Set
    test_preds = best_estimator.predict(X_test)
    test_mae = round(float(mean_absolute_error(y_test, test_preds)), 2)
    test_rmse = round(float(np.sqrt(mean_squared_error(y_test, test_preds))), 2)
    test_r2 = round(float(r2_score(y_test, test_preds)), 3)

    # 6. Held-Out Error & Subgroup Analysis
    residuals = y_test.values - test_preds
    empirical_margin_80 = round(float(np.percentile(np.abs(residuals), 80)), 2)

    subgroups = {}
    # By Gender
    for g_val, g_label in [(1, "Male"), (0, "Female")]:
        mask = (X_test["Gender"] == g_val)
        if mask.sum() > 0:
            subgroups[f"gender_{g_label}"] = {
                "sample_size": int(mask.sum()),
                "mae": round(float(mean_absolute_error(y_test[mask], test_preds[mask])), 2),
                "r2": round(float(r2_score(y_test[mask], test_preds[mask])), 3)
            }

    # By Attendance (>75% vs <=75%)
    for att_high, att_label in [(True, "attendance_gte_75"), (False, "attendance_lt_75")]:
        mask = (X_test["Attendance_Percentage"] >= 75.0) if att_high else (X_test["Attendance_Percentage"] < 75.0)
        if mask.sum() > 0:
            subgroups[att_label] = {
                "sample_size": int(mask.sum()),
                "mae": round(float(mean_absolute_error(y_test[mask], test_preds[mask])), 2)
            }

    # By Previous Failures (0 vs >=1)
    for fail_zero, fail_label in [(True, "no_previous_failures"), (False, "has_previous_failures")]:
        mask = (X_test["Previous_Failures"] == 0) if fail_zero else (X_test["Previous_Failures"] > 0)
        if mask.sum() > 0:
            subgroups[fail_label] = {
                "sample_size": int(mask.sum()),
                "mae": round(float(mean_absolute_error(y_test[mask], test_preds[mask])), 2)
            }

    # 7. Save Model Artifact
    model_artifact_path = MODEL_DIR / "student_model_v2_clean.pkl"
    joblib.dump(best_estimator, model_artifact_path)
    print(f"Saved primary production model to {model_artifact_path}")

    # 8. Feature Importances
    feat_imps = {}
    if hasattr(best_estimator, "feature_importances_"):
        for f, imp in sorted(zip(FEATURE_COLUMNS, best_estimator.feature_importances_), key=lambda x: x[1], reverse=True):
            feat_imps[f] = round(float(imp), 4)

    # 9. Comprehensive Metadata
    metadata = {
        "model_version": "2.0-clean",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_provenance": "RESEARCH_BASELINE_UCI_PORTUGUESE",
        "provenance_details": FEATURE_PROVENANCE,
        "selection_protocol": "5-Fold Cross-Validation on Training Split Only (No Test Peeking)",
        "algorithm": best_estimator.__class__.__name__,
        "parameters": best_estimator.get_params(),
        "feature_names": FEATURE_COLUMNS,
        "feature_importances": feat_imps,
        "metrics": {
            "cv_mae": best_candidate["cv_mae_mean"],
            "cv_mae_std": best_candidate["cv_mae_std"],
            "cv_rmse": best_candidate["cv_rmse_mean"],
            "cv_rmse_std": best_candidate["cv_rmse_std"],
            "cv_r2": best_candidate["cv_r2_mean"],
            "cv_r2_std": best_candidate["cv_r2_std"],
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "empirical_margin_80pct": empirical_margin_80
        },
        "subgroup_error_analysis": subgroups,
        "all_cv_benchmarks": [
            {k: v for k, v in b.items() if k != "instance"} for b in cv_benchmarks
        ]
    }

    metadata_path = MODEL_DIR / "student_model_v2_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Exported model metadata to {metadata_path}")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ScholarIQ prediction models.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    run_training_pipeline(random_state=args.seed)
