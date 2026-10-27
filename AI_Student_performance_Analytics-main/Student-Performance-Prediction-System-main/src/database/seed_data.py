"""ScholarIQ Database Seeding Script.
Initializes tables, registers models, seeds roles, and loads initial academic records.
Respects production security boundaries and prevents automatic seeding in production.
"""
import os
import json
import pandas as pd
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

from .connection import init_db, get_db_session
from .models import (
    User, Student, Term, Course, Enrollment,
    ModelVersion, Prediction, Intervention, AuditLog
)
from ..config import DATASET_DIR, MODEL_DIR, FEATURE_COLUMNS
from ..services.auth_service import hash_password


def seed_database(force: bool = False):
    """Run full schema creation and seed baseline records.
    
    In production mode (SCHOLARIQ_ENV=production), automatic seeding is bypassed
    unless explicitly forced to prevent insecure demo credential injection.
    """
    is_production = os.getenv("SCHOLARIQ_ENV", "").lower() == "production"
    if is_production and not force:
        print("[Security] SCHOLARIQ_ENV=production detected. Automatic demo seeding bypassed.")
        init_db()
        return

    print("Initializing database schema...")
    init_db()

    with get_db_session() as session:
        # 1. Register Machine Learning Models (Must precede any predictions)
        if session.query(ModelVersion).count() == 0:
            print("Registering machine learning models...")
            v2_meta_path = MODEL_DIR / "student_model_v2_metadata.json"
            v2_cv_r2 = 0.714
            v2_test_mae = 8.31
            if v2_meta_path.exists():
                try:
                    with open(v2_meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        v2_test_mae = meta.get("metrics", {}).get("test_mae", 8.31)
                        v2_cv_r2 = meta.get("metrics", {}).get("cv_r2", 0.714)
                except Exception:
                    pass

            models = [
                ModelVersion(
                    version_tag="v2.0-clean",
                    algorithm="RandomForestRegressor (Tuned, max_depth=6)",
                    artifact_path=str(MODEL_DIR / "student_model_v2_clean.pkl"),
                    dataset_hash="uci_portuguese_g1_clean",
                    cv_r2=v2_cv_r2,
                    test_mae=v2_test_mae,
                    is_active=True
                ),
                ModelVersion(
                    version_tag="v1.0-legacy",
                    algorithm="RandomForestRegressor (Historical Baseline, Leaked G3/2)",
                    artifact_path=str(MODEL_DIR / "student_model.pkl"),
                    dataset_hash="synthetic_leaked_v1",
                    cv_r2=0.325,
                    test_mae=2.02,
                    is_active=False
                )
            ]
            session.add_all(models)
            session.flush()

        active_model = session.query(ModelVersion).filter_by(version_tag="v2.0-clean").first()

        # 2. Seed System Users if not present
        if session.query(User).count() == 0:
            print("Seeding baseline administrative & academic users...")
            users = [
                User(
                    username="admin",
                    password_hash=hash_password("Admin@123"),
                    full_name="Dr. Elena Vance (Dean of Academics)",
                    role="admin",
                    email="dean.vance@scholariq.edu"
                ),
                User(
                    username="faculty",
                    password_hash=hash_password("Faculty@123"),
                    full_name="Prof. Marcus Chen (Mathematics Faculty)",
                    role="faculty",
                    email="m.chen@scholariq.edu"
                ),
                User(
                    username="advisor",
                    password_hash=hash_password("Advisor@123"),
                    full_name="Sarah Jenkins (Senior Academic Advisor)",
                    role="advisor",
                    email="s.jenkins@scholariq.edu"
                ),
                User(
                    username="student",
                    password_hash=hash_password("Student@123"),
                    full_name="Aditi Sharma (Student)",
                    role="student",
                    email="aditi.sharma@student.scholariq.edu"
                )
            ]
            session.add_all(users)
            session.flush()

        advisor_user = session.query(User).filter_by(role="advisor").first()
        admin_user = session.query(User).filter_by(role="admin").first()

        # 3. Seed Terms and Courses
        if session.query(Term).count() == 0:
            print("Seeding terms and courses...")
            fall_term = Term(
                term_code="2026-FALL",
                name="Fall Semester 2026",
                start_date=date(2026, 9, 1),
                end_date=date(2026, 12, 20)
            )
            session.add(fall_term)
            session.flush()

            math_course = Course(
                course_code="MATH201",
                course_name="Applied Linear Algebra & Statistics",
                term_id=fall_term.id
            )
            session.add(math_course)
            session.flush()

        math_course = session.query(Course).filter_by(course_code="MATH201").first()

        # 4. Seed Student Roster and Enrollments
        if session.query(Student).count() == 0:
            print("Seeding student records and academic enrollments...")
            csv_path = DATASET_DIR / "sample_cohort.csv"
            if not csv_path.exists():
                csv_path = DATASET_DIR / "student_performance_v2_clean.csv"

            df = pd.read_csv(csv_path).head(25)

            # Initialize ModelService to produce genuine model predictions rather than formula approximations
            from ..model_service import ModelService
            ml_service = None
            try:
                ml_service = ModelService(model_version="v2_clean")
            except Exception as e:
                print(f"[Warning] Could not initialize ModelService for seeding: {e}")

            for idx, row in df.iterrows():
                std_code = f"STU-2026-{idx+1:04d}"
                s_name = str(row.get("Student_Name", f"Student {idx+1}"))
                student = Student(
                    student_id=std_code,
                    name=s_name,
                    gender=int(row.get("Gender", 0)),
                    age=int(row.get("Age", 17))
                )
                session.add(student)
                session.flush()

                # Academic enrollment
                att = float(row.get("Attendance_Percentage", 90.0))
                assign = int(row.get("Assignments_Submitted", 7))
                interim = float(row.get("Internal_Marks", 6.0))
                study = float(row.get("Self_Study_Hours", 3.0))
                fails = int(row.get("Previous_Failures", 0))
                extra = int(row.get("Extra_Classes", 0))
                actual_final = float(row.get("Final_Score", 0.0)) if "Final_Score" in row else None

                enrollment = Enrollment(
                    student_id=student.id,
                    course_id=math_course.id if math_course else 1,
                    attendance_percentage=att,
                    assignments_submitted=assign,
                    internal_marks=interim,
                    self_study_hours=study,
                    previous_failures=fails,
                    extra_classes=extra,
                    final_score=actual_final
                )
                session.add(enrollment)
                session.flush()

                snapshot = {
                    "Gender": int(row.get("Gender", 0)),
                    "Age": int(row.get("Age", 17)),
                    "Assignments_Submitted": assign,
                    "Internal_Marks": interim,
                    "Attendance_Percentage": att,
                    "Self_Study_Hours": study,
                    "Previous_Failures": fails,
                    "Extra_Classes": extra,
                    "is_demo_seed": True,
                    "checkpoint_name": "Cohort Baseline Intake"
                }

                # Generate genuine ML prediction via ModelService if available
                if ml_service is not None and active_model:
                    try:
                        feature_df = pd.DataFrame([{col: snapshot[col] for col in FEATURE_COLUMNS}])
                        pred_output = ml_service.predict_single(feature_df)
                        predicted = pred_output["predicted_score"]
                        margin = pred_output["uncertainty_margin"]
                        grade = pred_output["grade"]
                        risk = pred_output["risk_level"]
                    except Exception:
                        predicted = 65.0
                        margin = 5.0
                        grade = "C"
                        risk = "Moderate Risk"
                else:
                    predicted = 65.0
                    margin = 5.0
                    grade = "C"
                    risk = "Moderate Risk"

                if active_model:
                    pred_record = Prediction(
                        student_id=student.id,
                        model_version_id=active_model.id,
                        predicted_score=round(predicted, 1),
                        confidence_margin=round(margin, 1),
                        score_lower=round(max(0.0, predicted - margin), 1),
                        score_upper=round(min(100.0, predicted + margin), 1),
                        grade=grade,
                        risk_level=risk,
                        feature_snapshot_json=json.dumps(snapshot),
                        created_at=datetime.now(timezone.utc) - timedelta(days=(25 - idx))
                    )
                    session.add(pred_record)
                    session.flush()

                # Add interventions for students with academic risk or backlogs
                if fails > 0 or att < 75.0 or predicted < 55.0:
                    status = "Completed" if idx % 3 == 0 else ("In Progress" if idx % 2 == 0 else "Open")
                    due = date.today() + timedelta(days=(7 if status != "Completed" else -5))
                    intervention = Intervention(
                        student_id=student.id,
                        assigned_advisor_id=advisor_user.id if advisor_user else None,
                        priority="CRITICAL" if fails > 1 else "HIGH",
                        category="Curriculum & Attendance Support",
                        issue_summary=f"Attendance at {att:.1f}% with {fails} prior failure(s). Predicted score {predicted:.1f}/100.",
                        student_action="Attend 1-on-1 remedial office hours and submit outstanding problem sets.",
                        educator_action="Review foundational coursework prerequisites and conduct weekly progress check-in.",
                        status=status,
                        follow_up_date=due,
                        notes=f"Initial counseling session scheduled for {student.name}.",
                        created_at=datetime.now(timezone.utc) - timedelta(days=10)
                    )
                    session.add(intervention)

            # Audit log
            session.add(AuditLog(
                user_id=admin_user.id if admin_user else None,
                action="INITIAL_DATABASE_SEED",
                entity_type="SYSTEM",
                entity_id=0,
                details_json=json.dumps({"seeded_students": len(df), "course": "MATH201", "is_demo": True})
            ))

    print("Database seeding completed successfully!")


if __name__ == "__main__":
    seed_database()
