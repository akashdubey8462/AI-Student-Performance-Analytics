"""ScholarIQ College-Specific Academic ML Data Contract.
Defines versioned, auditable Pydantic schemas for genuine institutional academic records,
temporal checkpoint boundaries, explicit missingness tracking, and anti-leakage validation.
"""
from datetime import date, datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field, model_validator
import pandas as pd


class CollegeAcademicRecord(BaseModel):
    """Data contract representing a single subject-level academic assessment record.
    Preserves None for pending/unconducted assessments without silent zero-imputation.
    """
    schema_version: str = Field(default="1.0.0", description="Contract specification version")
    student_id: str = Field(..., min_length=3, max_length=50, description="Pseudonymous student identifier")
    semester_code: str = Field(..., min_length=2, max_length=30, description="Academic semester identifier (e.g. SEM-IV)")
    subject_code: str = Field(..., min_length=2, max_length=30, description="Curricular subject code (e.g. CS401)")
    checkpoint_name: str = Field(
        ...,
        description="Assessment checkpoint: 'EARLY_SEMESTER', 'MID_SEMESTER', 'PRE_EXAM', or 'FINAL_VERIFIED'"
    )
    checkpoint_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when assessment features were recorded"
    )

    # 1. Mid-Semester Examinations (Max 20 each)
    midsem_1: Optional[float] = Field(None, ge=0.0, le=20.0, description="Mid-Semester Test 1 (0-20)")
    midsem_2: Optional[float] = Field(None, ge=0.0, le=20.0, description="Mid-Semester Test 2 (0-20)")
    midsem_3: Optional[float] = Field(None, ge=0.0, le=20.0, description="Mid-Semester Test 3 (0-20)")

    # 2. Academic Activities (Max 10 each)
    activity_1: Optional[float] = Field(None, ge=0.0, le=10.0, description="Continuous Activity 1 (0-10)")
    activity_2: Optional[float] = Field(None, ge=0.0, le=10.0, description="Continuous Activity 2 (0-10)")

    # 3. Attendance Signals
    attendance_percentage: Optional[float] = Field(None, ge=0.0, le=100.0, description="Verified attendance percentage (0-100)")
    classes_attended: Optional[int] = Field(None, ge=0, description="Classes attended count")
    classes_conducted: Optional[int] = Field(None, ge=0, description="Classes conducted count")

    # 4. Practical & Project Components
    practical_external: Optional[float] = Field(None, ge=0.0, le=30.0, description="Practical External Exam (0-30)")
    term_work: Optional[float] = Field(None, ge=0.0, le=20.0, description="Term Work / Sessional (0-20)")
    pbl: Optional[float] = Field(None, ge=0.0, le=50.0, description="Project-Based Learning (0-50)")
    internship: Optional[float] = Field(None, ge=0.0, le=50.0, description="Industry Internship Evaluation (0-50)")

    # 5. Deterministic Marking Subtotal (Separate from ML)
    calculated_marking_subtotal: Optional[float] = Field(None, description="Deterministic rule-calculated subtotal")

    # 6. Target Variable (Only available at FINAL_VERIFIED checkpoint)
    actual_final_result: Optional[float] = Field(None, ge=0.0, le=100.0, description="Verified final semester score (0-100)")
    actual_grade: Optional[str] = Field(None, max_length=5, description="Institutional final grade")

    @model_validator(mode="after")
    def validate_temporal_anti_leakage(self):
        """Prevent future assessments or final terminal outcomes from leaking into earlier checkpoints."""
        if self.checkpoint_name in ["EARLY_SEMESTER", "MID_SEMESTER"]:
            if self.actual_final_result is not None:
                raise ValueError(
                    f"Temporal leakage violation: 'actual_final_result' cannot be populated at '{self.checkpoint_name}' checkpoint."
                )
            if self.actual_grade is not None:
                raise ValueError(
                    f"Temporal leakage violation: 'actual_grade' cannot be populated at '{self.checkpoint_name}' checkpoint."
                )

        if self.checkpoint_name == "EARLY_SEMESTER":
            # Early semester cannot possess late-stage project or internship outcomes
            if self.internship is not None:
                raise ValueError("Temporal leakage: 'internship' score cannot be recorded at EARLY_SEMESTER checkpoint.")

        # Attendance consistency check
        if self.classes_attended is not None and self.classes_conducted is not None:
            if self.classes_conducted > 0:
                if self.classes_attended > self.classes_conducted:
                    raise ValueError(f"Attended classes ({self.classes_attended}) cannot exceed conducted classes ({self.classes_conducted}).")
                derived_pct = round((self.classes_attended / self.classes_conducted) * 100.0, 1)
                if self.attendance_percentage is None:
                    self.attendance_percentage = derived_pct

        return self

    def missing_assessment_fields(self) -> List[str]:
        """Return explicit list of unrecorded/pending continuous assessment fields."""
        components = {
            "midsem_1": self.midsem_1,
            "midsem_2": self.midsem_2,
            "midsem_3": self.midsem_3,
            "activity_1": self.activity_1,
            "activity_2": self.activity_2,
            "attendance_percentage": self.attendance_percentage,
            "practical_external": self.practical_external,
            "term_work": self.term_work,
            "pbl": self.pbl,
            "internship": self.internship
        }
        return [k for k, v in components.items() if v is None]

    def has_sufficient_college_data_for_ml(self) -> Tuple[bool, str]:
        """Audit whether this record contains genuine college features or requires fallback."""
        recorded_midsems = sum(1 for m in [self.midsem_1, self.midsem_2, self.midsem_3] if m is not None)
        has_att = self.attendance_percentage is not None
        has_act = (self.activity_1 is not None) or (self.activity_2 is not None)

        if recorded_midsems >= 2 and has_att and has_act:
            return True, "Sufficient college continuous assessment signals available."
        return False, "Insufficient historical college features; college-specific ML model pending dataset maturity."


def validate_college_dataset_csv(file_or_path: Any) -> Tuple[bool, List[str], Optional[pd.DataFrame], Dict[str, Any]]:
    """Validate a batch CSV of college academic records against the CollegeAcademicRecord contract."""
    errors: List[str] = []
    summary = {
        "total_rows": 0,
        "valid_rows": 0,
        "rejected_rows": 0,
        "missingness_stats": {}
    }

    try:
        if isinstance(file_or_path, str):
            df = pd.read_csv(file_or_path)
        elif hasattr(file_or_path, "read"):
            df = pd.read_csv(file_or_path)
        else:
            return False, ["Invalid file source provided."], None, summary
    except Exception as e:
        return False, [f"Failed to parse CSV file: {str(e)}"], None, summary

    summary["total_rows"] = len(df)
    if len(df) == 0:
        return False, ["CSV file contains no data rows."], None, summary

    required_cols = {"student_id", "semester_code", "subject_code", "checkpoint_name"}
    missing_required = required_cols - set(df.columns)
    if missing_required:
        return False, [f"Missing required contract columns: {list(missing_required)}"], None, summary

    valid_records = []
    for idx, row in df.iterrows():
        row_dict = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
        try:
            record = CollegeAcademicRecord(**row_dict)
            valid_records.append(record.model_dump())
            summary["valid_rows"] += 1
        except Exception as e:
            summary["rejected_rows"] += 1
            errors.append(f"Row {idx + 1}: {str(e)}")

    valid_df = pd.DataFrame(valid_records) if valid_records else None
    is_valid = len(errors) == 0 and summary["valid_rows"] > 0
    return is_valid, errors, valid_df, summary
