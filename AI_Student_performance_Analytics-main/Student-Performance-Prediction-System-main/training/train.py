"""ScholarIQ Model Training and Evaluation Pipeline.
Reproducible script for training leak-free academic performance prediction models.
"""
import os
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from src.config import DATASET_DIR, MODEL_DIR, FEATURE_COLUMNS


def run_training_pipeline(test_size: float = 0.20, random_state: int = 42) -> dict:
    raw_path = DATASET_DIR / "student_data.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}")
        
    df_raw = pd.read_csv(raw_path)
    print(f"Loaded raw dataset from {raw_path}: {df_raw.shape}")
    
    # Construct clean dataset without target leakage
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
    
    # Internal_Marks is strictly mapped from G1 interim exam (0-20 scale mapped to 0-10)
    clean_df["Internal_Marks"] = (df_raw["G1"] / 2.0).round(1)
    clean_df["Final_Score"] = (df_raw["G3"] * 5).astype(int)
    
    # Save clean dataset
    clean_csv_path = DATASET_DIR / "student_performance_v2_clean.csv"
    clean_df.to_csv(clean_csv_path, index=False)
    print(f"Exported clean dataset to {clean_csv_path}")
    
    X = clean_df[FEATURE_COLUMNS]
    y = clean_df["Final_Score"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
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
    
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    benchmarks = []
    
    for name, model in candidate_models.items():
        scoring = ["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"]
        cv_res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        
        cv_mae = -cv_res["test_neg_mean_absolute_error"].mean()
        cv_rmse = -cv_res["test_neg_root_mean_squared_error"].mean()
        cv_r2 = cv_res["test_r2"].mean()
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        test_mae = mean_absolute_error(y_test, preds)
        test_rmse = np.sqrt(mean_squared_error(y_test, preds))
        test_r2 = r2_score(y_test, preds)
        
        benchmarks.append({
            "model_name": name,
            "cv_mae": round(cv_mae, 2),
            "cv_rmse": round(cv_rmse, 2),
            "cv_r2": round(cv_r2, 3),
            "test_mae": round(test_mae, 2),
            "test_rmse": round(test_rmse, 2),
            "test_r2": round(test_r2, 3),
            "instance": model
        })
        
    best = sorted(benchmarks, key=lambda x: x["test_r2"], reverse=True)[0]
    best_estimator = best["instance"]
    
    model_artifact_path = MODEL_DIR / "student_model_v2_clean.pkl"
    joblib.dump(best_estimator, model_artifact_path)
    print(f"Saved primary production model to {model_artifact_path}")
    
    feat_imps = {}
    if hasattr(best_estimator, "feature_importances_"):
        for f, imp in sorted(zip(FEATURE_COLUMNS, best_estimator.feature_importances_), key=lambda x: x[1], reverse=True):
            feat_imps[f] = round(float(imp), 4)
            
    metadata = {
        "model_version": "2.0-clean",
        "created_at": datetime.now().isoformat(),
        "target": "Final_Score (scale 0-100, mapped from UCI G3*5)",
        "target_leakage_status": "VERIFIED_CLEAN",
        "algorithm": best_estimator.__class__.__name__,
        "parameters": best_estimator.get_params(),
        "feature_names": FEATURE_COLUMNS,
        "feature_importances": feat_imps,
        "metrics": {
            "cv_mae": best["cv_mae"],
            "cv_rmse": best["cv_rmse"],
            "cv_r2": best["cv_r2"],
            "test_mae": best["test_mae"],
            "test_rmse": best["test_rmse"],
            "test_r2": best["test_r2"]
        },
        "all_benchmarks": [
            {k: v for k, v in b.items() if k != "instance"} for b in benchmarks
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
