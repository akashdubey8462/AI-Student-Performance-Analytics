"""ScholarIQ Prediction, Analytics and Semester Progress Tracking Service.
Handles prediction persistence, model version routing, longitudinal checkpoints, and executive KPIs.
"""
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from ..database.connection import get_db_session
from ..database.models import Student, ModelVersion, Prediction, Enrollment, Intervention, AuditLog
from ..model_service import ModelService
from ..config import FEATURE_COLUMNS


def save_prediction(
    student_db_id: int,
    features: Dict[str, Any],
    model_version_tag: str = "v2.0-clean",
    checkpoint_name: str = "Standard Assessment",
    user_id: Optional[int] = None
) -> Dict[str, Any]:
    """Execute model inference and persist the prediction record with an immutable feature snapshot and checkpoint."""
    # Choose engine key
    engine_key = "v2_clean" if "v2" in model_version_tag else "legacy_v1"
    service = ModelService(model_version=engine_key)

    input_df = pd.DataFrame([{col: features[col] for col in FEATURE_COLUMNS}])[FEATURE_COLUMNS]
    res = service.predict_single(input_df)

    snapshot_payload = dict(features)
    snapshot_payload["checkpoint_name"] = checkpoint_name

    with get_db_session() as session:
        model_ver = session.query(ModelVersion).filter_by(version_tag=model_version_tag).first()
        if not model_ver:
            model_ver = session.query(ModelVersion).filter_by(is_active=True).first()

        pred = Prediction(
            student_id=student_db_id,
            model_version_id=model_ver.id,
            predicted_score=res["predicted_score"],
            confidence_margin=res["uncertainty_margin"],
            score_lower=res["score_lower"],
            score_upper=res["score_upper"],
            grade=res["grade"],
            risk_level=res["risk_level"],
            feature_snapshot_json=json.dumps(snapshot_payload),
            created_at=datetime.now(timezone.utc)
        )
        session.add(pred)
        session.flush()

        # Update enrollment if exists
        student = session.query(Student).filter_by(id=student_db_id).first()
        if student and student.enrollments:
            enrollment = student.enrollments[0]
            enrollment.attendance_percentage = float(features.get("Attendance_Percentage", enrollment.attendance_percentage))
            enrollment.assignments_submitted = int(features.get("Assignments_Submitted", enrollment.assignments_submitted))
            enrollment.internal_marks = float(features.get("Internal_Marks", enrollment.internal_marks))
            enrollment.self_study_hours = float(features.get("Self_Study_Hours", enrollment.self_study_hours))
            enrollment.previous_failures = int(features.get("Previous_Failures", enrollment.previous_failures))
            enrollment.extra_classes = int(features.get("Extra_Classes", enrollment.extra_classes))

        # Log prediction event
        session.add(AuditLog(
            user_id=user_id,
            action="PREDICTION_GENERATED",
            entity_type="PREDICTION",
            entity_id=pred.id,
            details_json=json.dumps({
                "student_id": student_db_id,
                "score": res["predicted_score"],
                "model": model_version_tag,
                "checkpoint": checkpoint_name
            })
        ))

        res["prediction_db_id"] = pred.id
        res["checkpoint_name"] = checkpoint_name
        return res


def list_predictions(
    limit: int = 100,
    risk_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieve historical prediction audit logs."""
    with get_db_session() as session:
        query = session.query(Prediction).order_by(Prediction.created_at.desc())

        if risk_filter and risk_filter != "All":
            query = query.filter(Prediction.risk_level == risk_filter)

        predictions = query.limit(limit).all()
        results = []

        for p in predictions:
            student = session.query(Student).filter_by(id=p.student_id).first()
            model = session.query(ModelVersion).filter_by(id=p.model_version_id).first()
            snapshot = json.loads(p.feature_snapshot_json) if p.feature_snapshot_json else {}

            results.append({
                "id": p.id,
                "student_id": student.student_id if student else "—",
                "student_name": student.name if student else "Unknown",
                "predicted_score": p.predicted_score,
                "confidence_margin": p.confidence_margin,
                "band": f"[{p.score_lower:.1f} – {p.score_upper:.1f}]",
                "grade": p.grade,
                "risk_level": p.risk_level,
                "model_version": model.version_tag if model else "v2.0-clean",
                "checkpoint": snapshot.get("checkpoint_name", "Standard Checkpoint"),
                "attendance": snapshot.get("Attendance_Percentage", "—"),
                "internal_marks": snapshot.get("Internal_Marks", "—"),
                "failures": snapshot.get("Previous_Failures", "—"),
                "predicted_at": p.created_at.strftime("%Y-%m-%d %H:%M")
            })

        return results


def get_student_prediction_timeline(student_db_id: int) -> List[Dict[str, Any]]:
    """Retrieve chronological prediction checkpoints for a student to track semester progress.
    Compares successive forecasts and tracks error against actual final outcome if recorded.
    """
    with get_db_session() as session:
        student = session.query(Student).filter_by(id=student_db_id).first()
        if not student:
            return []

        actual_outcome = None
        if student.enrollments and student.enrollments[0].final_score is not None:
            actual_outcome = student.enrollments[0].final_score

        preds = sorted(student.predictions, key=lambda x: x.created_at)
        timeline = []
        prev_score = None

        for idx, p in enumerate(preds, 1):
            snapshot = json.loads(p.feature_snapshot_json) if p.feature_snapshot_json else {}
            chk_name = snapshot.get("checkpoint_name", f"Checkpoint {idx}")
            
            delta = round(p.predicted_score - prev_score, 1) if prev_score is not None else 0.0
            prev_score = p.predicted_score
            
            error = None
            if actual_outcome is not None:
                error = round(abs(p.predicted_score - actual_outcome), 1)

            timeline.append({
                "checkpoint_index": idx,
                "checkpoint_name": chk_name,
                "predicted_score": p.predicted_score,
                "score_lower": p.score_lower,
                "score_upper": p.score_upper,
                "confidence_margin": p.confidence_margin,
                "grade": p.grade,
                "risk_level": p.risk_level,
                "delta_from_previous": delta,
                "actual_outcome": actual_outcome,
                "prediction_error": error,
                "created_at": p.created_at.strftime("%Y-%m-%d %H:%M")
            })

        return timeline


def record_actual_outcome(
    student_db_id: int,
    actual_score: float,
    user_id: Optional[int] = None
) -> Dict[str, Any]:
    """Record verified final semester score in enrollment for accuracy tracking."""
    if actual_score < 0.0 or actual_score > 100.0:
        raise ValueError(f"Actual score must be between 0.0 and 100.0, received {actual_score}")

    with get_db_session() as session:
        student = session.query(Student).filter_by(id=student_db_id).first()
        if not student or not student.enrollments:
            raise ValueError(f"Student #{student_db_id} has no enrolled course record")

        enrollment = student.enrollments[0]
        enrollment.final_score = float(actual_score)

        session.add(AuditLog(
            user_id=user_id,
            action="RECORD_ACTUAL_OUTCOME",
            entity_type="ENROLLMENT",
            entity_id=enrollment.id,
            details_json=json.dumps({
                "student_id": student.student_id,
                "actual_score": actual_score
            })
        ))

        return {
            "student_id": student.student_id,
            "name": student.name,
            "actual_final_score": actual_score
        }


def get_overview_kpis() -> Dict[str, Any]:
    """Calculate institutional overview KPIs from live database tables."""
    with get_db_session() as session:
        total_students = session.query(Student).count()
        total_enrollments = session.query(Enrollment).count()
        
        # Latest prediction per student
        latest_preds = []
        students = session.query(Student).all()
        for s in students:
            if s.predictions:
                latest = sorted(s.predictions, key=lambda x: x.created_at, reverse=True)[0]
                latest_preds.append(latest)

        avg_score = (
            sum(p.predicted_score for p in latest_preds) / len(latest_preds)
            if latest_preds else 0.0
        )
        at_risk = sum(1 for p in latest_preds if p.risk_level == "High Academic Risk")
        passing_cnt = sum(1 for p in latest_preds if p.predicted_score >= 40.0)
        pass_rate = (passing_cnt / len(latest_preds) * 100.0) if latest_preds else 0.0

        # Interventions
        open_interventions = session.query(Intervention).filter(
            Intervention.status.in_(["Open", "In Progress"])
        ).count()
        
        today = datetime.now(timezone.utc).date()
        overdue_interventions = session.query(Intervention).filter(
            Intervention.status.in_(["Open", "In Progress"]),
            Intervention.follow_up_date < today
        ).count()

        active_model = session.query(ModelVersion).filter_by(is_active=True).first()

        return {
            "total_students": total_students,
            "total_enrollments": total_enrollments,
            "evaluated_students": len(latest_preds),
            "average_predicted_score": round(avg_score, 1),
            "projected_pass_rate": round(pass_rate, 1),
            "at_risk_count": at_risk,
            "open_interventions": open_interventions,
            "overdue_interventions": overdue_interventions,
            "active_model_tag": active_model.version_tag if active_model else "v2.0-clean",
            "active_model_r2": active_model.cv_r2 if active_model else 0.714
        }
