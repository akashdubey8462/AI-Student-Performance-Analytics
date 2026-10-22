"""ScholarIQ Batch Import Service.
Handles transactional CSV roster imports, data quality checks, and automated inference.
"""
import io
import json
from typing import Dict, Any, Tuple, Optional
import pandas as pd
from datetime import datetime, timezone
from ..database.connection import get_db_session
from ..database.models import Student, Course, Enrollment, ModelVersion, Prediction, ImportJob, ImportErrorRecord, AuditLog
from ..data_validation import validate_batch_csv
from ..model_service import ModelService
from ..config import FEATURE_COLUMNS


def import_roster_transactional(
    file_or_content: Any,
    filename: str = "roster.csv",
    course_code: str = "MATH201",
    user_id: Optional[int] = None
) -> Tuple[bool, Dict[str, Any]]:
    """Execute a validated transactional batch import of a student roster."""
    is_valid, validation_errors, clean_df, summary = validate_batch_csv(file_or_content)
    
    with get_db_session() as session:
        # Create import job record
        status_val = "FAILED" if not is_valid else ("PARTIAL" if summary["rejected_rows"] > 0 else "SUCCESS")
        job = ImportJob(
            filename=filename,
            total_rows=summary["total_rows"],
            valid_rows=summary["valid_rows"],
            rejected_rows=summary["rejected_rows"],
            status=status_val,
            timestamp=datetime.now(timezone.utc)
        )
        session.add(job)
        session.flush()

        # Log row errors if any
        for r_err in summary.get("row_errors", []):
            try:
                row_idx = int(r_err.split(":")[0].replace("Row", "").strip())
            except Exception:
                row_idx = 0
            session.add(ImportErrorRecord(
                import_job_id=job.id,
                row_number=row_idx,
                error_message=r_err
            ))

        if not is_valid or clean_df is None or len(clean_df) == 0:
            return False, {
                "job_id": job.id,
                "status": "FAILED",
                "errors": validation_errors,
                "summary": summary
            }

        # Find or create target course
        course = session.query(Course).filter_by(course_code=course_code).first()
        if not course:
            course = Course(course_code=course_code, course_name=f"Course {course_code}")
            session.add(course)
            session.flush()

        # Active ML model for automated scoring
        active_model = session.query(ModelVersion).filter_by(is_active=True).first()
        model_service = ModelService(model_version="v2_clean")

        created_students = 0
        updated_enrollments = 0

        # Run model on valid records
        scores_df = model_service.predict_batch(clean_df)

        for idx, row in scores_df.iterrows():
            s_name = str(row.get("Student_Name", f"Student {idx+1}")).strip()
            
            # Match existing student or create new
            student = session.query(Student).filter_by(name=s_name).first()
            if not student:
                count = session.query(Student).count()
                student = Student(
                    student_id=f"STU-2026-{count+1:04d}",
                    name=s_name,
                    gender=int(row["Gender"]),
                    age=int(row["Age"])
                )
                session.add(student)
                session.flush()
                created_students += 1

            # Match or create enrollment
            enrollment = session.query(Enrollment).filter_by(student_id=student.id, course_id=course.id).first()
            if not enrollment:
                enrollment = Enrollment(student_id=student.id, course_id=course.id)
                session.add(enrollment)

            enrollment.attendance_percentage = float(row["Attendance_Percentage"])
            enrollment.assignments_submitted = int(row["Assignments_Submitted"])
            enrollment.internal_marks = float(row["Internal_Marks"])
            enrollment.self_study_hours = float(row["Self_Study_Hours"])
            enrollment.previous_failures = int(row["Previous_Failures"])
            enrollment.extra_classes = int(row["Extra_Classes"])
            updated_enrollments += 1

            # Record Prediction
            pred_score = float(row["Predicted_Score"])
            snapshot = {col: row[col] for col in FEATURE_COLUMNS}
            margin = 4.5
            pred_record = Prediction(
                student_id=student.id,
                model_version_id=active_model.id if active_model else 1,
                predicted_score=pred_score,
                confidence_margin=margin,
                score_lower=round(max(0.0, pred_score - margin), 1),
                score_upper=round(min(100.0, pred_score + margin), 1),
                grade=str(row["Grade"]),
                risk_level=str(row["Risk_Level"]),
                feature_snapshot_json=json.dumps(snapshot),
                created_at=datetime.now(timezone.utc)
            )
            session.add(pred_record)

        session.add(AuditLog(
            user_id=user_id,
            action="BATCH_ROSTER_IMPORTED",
            entity_type="IMPORT_JOB",
            entity_id=job.id,
            details_json=json.dumps({
                "filename": filename,
                "course": course_code,
                "valid": summary["valid_rows"],
                "rejected": summary["rejected_rows"]
            })
        ))

        return True, {
            "job_id": job.id,
            "status": status_val,
            "total_rows": summary["total_rows"],
            "valid_rows": summary["valid_rows"],
            "rejected_rows": summary["rejected_rows"],
            "created_students": created_students,
            "updated_enrollments": updated_enrollments,
            "row_errors": summary["row_errors"]
        }
