"""ScholarIQ Configuration Module.
Defines system paths, feature schemas, grade boundaries, and chart tokens.
"""
from pathlib import Path

# Paths
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
DATASET_DIR = PROJECT_DIR / "Dataset"
MODEL_DIR = PROJECT_DIR / "Model"
APP_DIR = PROJECT_DIR / "Streamlit_App"

# Default Model Artifacts
DEFAULT_MODEL_V2 = MODEL_DIR / "student_model_v2_clean.pkl"
DEFAULT_METADATA_V2 = MODEL_DIR / "student_model_v2_metadata.json"
LEGACY_MODEL_V1 = MODEL_DIR / "student_model.pkl"

# Feature Schema
FEATURE_COLUMNS = [
    "Gender",
    "Age",
    "Assignments_Submitted",
    "Internal_Marks",
    "Attendance_Percentage",
    "Self_Study_Hours",
    "Previous_Failures",
    "Extra_Classes"
]

FEATURE_RANGES = {
    "Gender": {"min": 0, "max": 1, "type": int, "labels": {0: "Female", 1: "Male"}},
    "Age": {"min": 14, "max": 25, "type": int},
    "Assignments_Submitted": {"min": 0, "max": 10, "type": int},
    "Internal_Marks": {"min": 0.0, "max": 10.0, "type": float},
    "Attendance_Percentage": {"min": 0.0, "max": 100.0, "type": float},
    "Self_Study_Hours": {"min": 0.0, "max": 12.0, "type": float},
    "Previous_Failures": {"min": 0, "max": 5, "type": int},
    "Extra_Classes": {"min": 0, "max": 1, "type": int, "labels": {0: "No", 1: "Yes"}}
}

# Grading Boundaries (0 to 100 Scale)
GRADE_SCALE = [
    (90.0, 100.0, "A+", "Outstanding", "good"),
    (80.0, 89.99, "A", "Excellent", "good"),
    (70.0, 79.99, "B", "Very Good", "good"),
    (60.0, 69.99, "C", "Good", "mid"),
    (50.0, 59.99, "D", "Average (Passing)", "mid"),
    (0.0, 49.99, "F", "High Academic Risk (Failing)", "low")
]

# Risk Levels
RISK_LEVELS = {
    "Low Risk": {"min": 70.0, "badge": "good", "desc": "Solid academic standing; on track for strong results."},
    "Moderate Risk": {"min": 50.0, "badge": "mid", "desc": "Passing border; minor intervention recommended."},
    "High Academic Risk": {"min": 0.0, "badge": "low", "desc": "At risk of academic failure or course backlog."}
}

# Design Tokens (Maroon, Black, Warm Gold Accent)
THEME_TOKENS = {
    "bg": "#0A0708",
    "ink": "#F2ECEC",
    "ink_soft": "#AC9CA0",
    "accent": "#D4A24E",
    "maroon": "#8C1C3B",
    "success": "#3DDC97",
    "amber": "#F5A524",
    "red": "#F5646B",
    "grid": "rgba(255, 255, 255, 0.08)"
}
