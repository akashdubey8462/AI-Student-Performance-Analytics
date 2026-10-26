"""ScholarIQ What-If Scenario Simulator Service.
Enables side-by-side comparative simulation of academic interventions and performance adjustments.
Maintains strict separation between deterministic marking differences and ML forecast shifts.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
from .marking_service import calculate_academic_marks, DEFAULT_ACADEMIC_RULES
from .rules_service import get_academic_rules
from ..model_service import ModelService
from ..config import FEATURE_COLUMNS


def simulate_what_if_scenario(
    baseline_inputs: Dict[str, Any],
    scenario_inputs: Dict[str, Any],
    model_version: str = "v2_clean",
    custom_rules: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute a what-if simulation comparing baseline vs alternative academic scenario.
    
    Evaluates both:
      1. Deterministic institutional marking differences (provisional subtotal, attendance bonus, mid-sem average)
      2. Supervised machine learning prediction shifts (score %, grade, risk level, CGPA)
    """
    rules = custom_rules or get_academic_rules()
    
    # 1. Deterministic Marking Calculations
    baseline_marks = calculate_academic_marks(baseline_inputs, rules)
    scenario_marks = calculate_academic_marks(scenario_inputs, rules)
    
    subtotal_delta = round(scenario_marks["provisional_subtotal"] - baseline_marks["provisional_subtotal"], 2)
    midsem_delta = round(scenario_marks["midsem_avg"] - baseline_marks["midsem_avg"], 2)
    bonus_delta = scenario_marks["attendance_bonus"] - baseline_marks["attendance_bonus"]
    activity_delta = round(scenario_marks["counted_activity"] - baseline_marks["counted_activity"], 2)
    
    # 2. Machine Learning Predictions
    model_key = "v2_clean" if "v2" in model_version else "legacy_v1"
    model_service = ModelService(model_version=model_key)
    
    # Map inputs to ML feature vector
    # Internal_Marks is derived from midsem_avg / 2.0 (scale 0-10) to match Cortez & Silva G1/2
    base_feat = _build_feature_dict(baseline_inputs, baseline_marks["midsem_avg"])
    scen_feat = _build_feature_dict(scenario_inputs, scenario_marks["midsem_avg"])
    
    base_df = pd.DataFrame([base_feat])[FEATURE_COLUMNS]
    scen_df = pd.DataFrame([scen_feat])[FEATURE_COLUMNS]
    
    baseline_ml = model_service.predict_single(base_df)
    scenario_ml = model_service.predict_single(scen_df)
    
    score_delta = round(scenario_ml["predicted_score"] - baseline_ml["predicted_score"], 2)
    cgpa_baseline = round(baseline_ml["predicted_score"] / 10.0, 2)
    cgpa_scenario = round(scenario_ml["predicted_score"] / 10.0, 2)
    cgpa_delta = round(cgpa_scenario - cgpa_baseline, 2)
    
    # 3. Generate Evidence-Based Comparative Insights
    insights = []
    
    if bonus_delta > 0:
        insights.append(
            f"Attendance increase triggered +{bonus_delta} attendance bonus mark(s) in the official mid-semester component."
        )
    elif bonus_delta < 0:
        insights.append(
            f"Attendance drop resulted in losing {abs(bonus_delta)} bonus mark(s) in the official mid-semester component."
        )
        
    if midsem_delta != 0:
        verb = "boosts" if midsem_delta > 0 else "reduces"
        insights.append(
            f"Altering mid-semester performance {verb} the selected examination average by {abs(midsem_delta):.1f} marks."
        )
        
    if score_delta != 0:
        verb = "higher" if score_delta > 0 else "lower"
        insights.append(
            f"The ML model projects a final semester score {abs(score_delta):.1f} points {verb} "
            f"({baseline_ml['predicted_score']:.1f}% -> {scenario_ml['predicted_score']:.1f}%)."
        )
        
    if scenario_ml["risk_level"] != baseline_ml["risk_level"]:
        insights.append(
            f"Academic risk stratification transitions from '{baseline_ml['risk_level']}' to '{scenario_ml['risk_level']}'."
        )
        
    if scenario_ml["grade"] != baseline_ml["grade"]:
        insights.append(
            f"Predicted letter grade shifts from '{baseline_ml['grade']}' to '{scenario_ml['grade']}'."
        )
        
    if not insights:
        insights.append("No significant change detected between the baseline and alternative scenario inputs.")

    return {
        "baseline_marking": baseline_marks,
        "scenario_marking": scenario_marks,
        "subtotal_delta": subtotal_delta,
        "midsem_delta": midsem_delta,
        "bonus_delta": bonus_delta,
        "activity_delta": activity_delta,
        "baseline_ml": baseline_ml,
        "scenario_ml": scenario_ml,
        "score_delta": score_delta,
        "cgpa_baseline": cgpa_baseline,
        "cgpa_scenario": cgpa_scenario,
        "cgpa_delta": cgpa_delta,
        "risk_changed": baseline_ml["risk_level"] != scenario_ml["risk_level"],
        "grade_changed": baseline_ml["grade"] != scenario_ml["grade"],
        "insights": insights,
        "limitations": (
            "What-If projections are statistical estimates based on historical correlation patterns. "
            "They illustrate potential trajectory shifts rather than guaranteed examination results. "
            "Actual degree outcomes depend on final unallocated terminal examinations and continuous coursework."
        )
    }


def _build_feature_dict(inputs: Dict[str, Any], midsem_avg: float) -> Dict[str, Any]:
    """Helper to assemble model feature vector with internal mark derivation."""
    # Scale midsem average (0-20) to continuous assessment (0-10)
    internal_marks = min(10.0, max(0.0, float(midsem_avg) / 2.0))
    
    return {
        "Gender": int(inputs.get("Gender", 1 if inputs.get("gender") == "Male" else 0)),
        "Age": int(inputs.get("Age", inputs.get("age", 18))),
        "Assignments_Submitted": int(inputs.get("Assignments_Submitted", inputs.get("assignments_submitted", 8))),
        "Internal_Marks": internal_marks,
        "Attendance_Percentage": float(inputs.get("Attendance_Percentage", inputs.get("attendance_percentage", 85.0))),
        "Self_Study_Hours": float(inputs.get("Self_Study_Hours", inputs.get("self_study_hours", 3.0))),
        "Previous_Failures": int(inputs.get("Previous_Failures", inputs.get("previous_failures", 0))),
        "Extra_Classes": int(inputs.get("Extra_Classes", 1 if inputs.get("extra_classes") in ["Yes", 1] else 0))
    }
