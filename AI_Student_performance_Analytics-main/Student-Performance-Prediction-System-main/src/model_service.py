"""ScholarIQ Model Service Module.
Handles versioned artifact loading, inference, uncertainty bounds, and local feature contributions.
"""
from typing import Dict, Any, Tuple, Optional, List
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from .config import (
    DEFAULT_MODEL_V2,
    DEFAULT_METADATA_V2,
    LEGACY_MODEL_V1,
    FEATURE_COLUMNS,
    GRADE_SCALE,
    RISK_LEVELS
)


class ModelService:
    def __init__(self, model_version: str = "v2_clean"):
        self.version = model_version
        self.model = None
        self.metadata = None
        self.load_error = None
        self._load_model()
        
    def _load_model(self):
        """Safely load model artifact and accompanying metadata."""
        if self.version == "v2_clean":
            model_path = DEFAULT_MODEL_V2
            metadata_path = DEFAULT_METADATA_V2
        else:
            model_path = LEGACY_MODEL_V1
            metadata_path = None
            
        if not model_path.exists():
            self.load_error = f"Model artifact not found at {model_path}"
            return
            
        try:
            self.model = joblib.load(model_path)
            if metadata_path and metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {
                    "model_version": self.version,
                    "algorithm": self.model.__class__.__name__,
                    "target_leakage_status": "LEGACY (Susceptible to leakage)" if self.version != "v2_clean" else "CLEAN"
                }
        except Exception as e:
            self.load_error = f"Failed to deserialize model: {str(e)}"
            self.model = None

    def predict_single(self, input_df: pd.DataFrame) -> Dict[str, Any]:
        """Run single-student inference with prediction, grade, uncertainty, and feature contribution."""
        if self.model is None:
            raise RuntimeError(f"Model service is unavailable: {self.load_error}")
            
        # Ensure correct column order
        X = input_df[FEATURE_COLUMNS]
        
        # Raw prediction
        raw_pred = float(self.model.predict(X)[0])
        score = max(0.0, min(100.0, raw_pred))
        
        # Tree variance uncertainty estimation if available (Random Forest)
        uncertainty = self._estimate_uncertainty(X)
        
        # Grade and category
        grade_info = self.grade_for_score(score)
        risk_info = self.risk_for_score(score)
        
        # Local feature attribution (tree contribution proxy)
        contributions = self._compute_feature_contributions(X, score)
        
        return {
            "predicted_score": round(score, 1),
            "uncertainty_margin": round(uncertainty, 1),
            "score_lower": max(0.0, round(score - uncertainty, 1)),
            "score_upper": min(100.0, round(score + uncertainty, 1)),
            "grade": grade_info["grade"],
            "performance": grade_info["performance"],
            "badge_color": grade_info["badge"],
            "risk_level": risk_info["risk_level"],
            "risk_badge": risk_info["badge"],
            "risk_desc": risk_info["desc"],
            "model_version": self.version,
            "feature_contributions": contributions
        }

    def predict_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """Run batch predictions on an entire cohort DataFrame."""
        if self.model is None:
            raise RuntimeError(f"Model service is unavailable: {self.load_error}")
            
        X = batch_df[FEATURE_COLUMNS]
        raw_preds = self.model.predict(X)
        clamped_preds = np.clip(raw_preds, 0.0, 100.0).round(1)
        
        out_df = batch_df.copy()
        out_df["Predicted_Score"] = clamped_preds
        out_df["Grade"] = [self.grade_for_score(s)["grade"] for s in clamped_preds]
        out_df["Performance"] = [self.grade_for_score(s)["performance"] for s in clamped_preds]
        out_df["Risk_Level"] = [self.risk_for_score(s)["risk_level"] for s in clamped_preds]
        out_df["Passed"] = (clamped_preds >= 50.0).astype(int)
        
        return out_df

    def _estimate_uncertainty(self, X: pd.DataFrame) -> float:
        """Calculate standard error across ensemble decision trees."""
        if hasattr(self.model, "estimators_"):
            # Sample of tree predictions
            predictions = [tree.predict(X.values)[0] for tree in self.model.estimators_]
            std_err = float(np.std(predictions))
            return min(15.0, max(2.5, std_err))
        return 5.0

    def _compute_feature_contributions(self, X: pd.DataFrame, score: float) -> List[Dict[str, Any]]:
        """Compute relative factor contributions based on normalized feature values and tree importances."""
        row = X.iloc[0]
        contributions = []
        
        # Approximate baseline score (mean of training distribution is ~52.1)
        baseline = 52.1
        
        # Reference feature values (means/medians from clean training set)
        refs = {
            "Internal_Marks": 5.45,
            "Attendance_Percentage": 94.3,
            "Assignments_Submitted": 6.0,
            "Self_Study_Hours": 3.1,
            "Previous_Failures": 0.33,
            "Extra_Classes": 0.13,
            "Age": 16.7,
            "Gender": 0.5
        }
        
        weights = getattr(self.model, "feature_importances_", None)
        if weights is None:
            weights = [1.0 / len(FEATURE_COLUMNS)] * len(FEATURE_COLUMNS)
            
        for feat, weight in zip(FEATURE_COLUMNS, weights):
            val = float(row[feat])
            ref = refs.get(feat, val)
            
            # Directional impact
            if feat == "Previous_Failures":
                diff = ref - val  # fewer failures is positive
            else:
                diff = val - ref
                
            impact_score = diff * weight * 15.0
            
            contributions.append({
                "feature": feat,
                "value": val,
                "importance_weight": round(float(weight), 4),
                "impact": round(impact_score, 1),
                "is_positive": impact_score >= 0
            })
            
        contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return contributions

    @staticmethod
    def grade_for_score(score: float) -> Dict[str, str]:
        for lower, upper, grade, perf, badge in GRADE_SCALE:
            if lower <= score <= upper:
                return {"grade": grade, "performance": perf, "badge": badge}
        return {"grade": "F", "performance": "Needs Improvement", "badge": "low"}

    @staticmethod
    def risk_for_score(score: float) -> Dict[str, str]:
        for level, cfg in RISK_LEVELS.items():
            if score >= cfg["min"]:
                return {"risk_level": level, "badge": cfg["badge"], "desc": cfg["desc"]}
        return {"risk_level": "High Academic Risk", "badge": "low", "desc": "Urgent academic intervention required."}
