"""ScholarIQ Student Management Service.
Handles student profile queries, academic directory search, and enrollment records.
"""
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import joinedload
from ..database.connection import get_db_session
from ..database.models import Student, Enrollment, Course, Prediction, Intervention, AuditLog


def list_students(
    search_query: str = "",
    risk_filter: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """Retrieve students with their latest enrollment signals and latest prediction."""
    with get_db_session() as session:
        query = session.query(Student).options(
            joinedload(Student.enrollments),
            joinedload(Student.predictions)
        )

        if search_query:
            term = f"%{search_query.strip()}%"
            query = query.filter(
                (Student.name.ilike(term)) | (Student.student_id.ilike(term))
            )

        students = query.order_by(Student.id.asc()).limit(limit).all()
        results = []

        for s in students:
            latest_pred = sorted(s.predictions, key=lambda p: p.created_at, reverse=True)[0] if s.predictions else None
            enrollment = s.enrollments[0] if s.enrollments else None

            pred_score = latest_pred.predicted_score if latest_pred else None
            risk = latest_pred.risk_level if latest_pred else "Not Evaluated"
            grade = latest_pred.grade if latest_pred else "—"

            if risk_filter and risk_filter != "All" and risk != risk_filter:
                continue

            results.append({
                "id": s.id,
                "student_id": s.student_id,
                "name": s.name,
                "gender": "Male" if s.gender == 1 else "Female",
                "age": s.age,
                "attendance_percentage": enrollment.attendance_percentage if enrollment else 0.0,
                "internal_marks": enrollment.internal_marks if enrollment else 0.0,
                "assignments_submitted": enrollment.assignments_submitted if enrollment else 0,
                "previous_failures": enrollment.previous_failures if enrollment else 0,
                "latest_predicted_score": pred_score,
                "latest_grade": grade,
                "risk_level": risk,
                "last_predicted_at": latest_pred.created_at.strftime("%Y-%m-%d %H:%M") if latest_pred else "Never"
            })

        return results


def get_student_by_id(student_db_id: int) -> Optional[Dict[str, Any]]:
    """Get full longitudinal student detail including all enrollments, predictions, and interventions."""
    with get_db_session() as session:
        s = session.query(Student).filter_by(id=student_db_id).first()
        if not s:
            return None

        enrollments = []
        for e in s.enrollments:
            course = session.query(Course).filter_by(id=e.course_id).first()
            enrollments.append({
                "course_code": course.course_code if course else "—",
                "course_name": course.course_name if course else "—",
                "attendance": e.attendance_percentage,
                "assignments": e.assignments_submitted,
                "internal_marks": e.internal_marks,
                "study_hours": e.self_study_hours,
                "failures": e.previous_failures,
                "extra_classes": "Yes" if e.extra_classes == 1 else "No",
                "actual_final_score": e.final_score
            })

        predictions = []
        for p in sorted(s.predictions, key=lambda x: x.created_at, reverse=True):
            predictions.append({
                "id": p.id,
                "predicted_score": p.predicted_score,
                "confidence_margin": p.confidence_margin,
                "grade": p.grade,
                "risk_level": p.risk_level,
                "model_version": p.model_version.version_tag if p.model_version else "Unknown",
                "created_at": p.created_at.strftime("%Y-%m-%d %H:%M")
            })

        interventions = []
        for i in sorted(s.interventions, key=lambda x: x.created_at, reverse=True):
            interventions.append({
                "id": i.id,
                "priority": i.priority,
                "category": i.category,
                "issue_summary": i.issue_summary,
                "student_action": i.student_action,
                "educator_action": i.educator_action,
                "status": i.status,
                "follow_up_date": i.follow_up_date.strftime("%Y-%m-%d") if i.follow_up_date else "None",
                "notes": i.notes,
                "advisor": i.advisor.full_name if i.advisor else "Unassigned",
                "created_at": i.created_at.strftime("%Y-%m-%d")
            })

        return {
            "id": s.id,
            "student_id": s.student_id,
            "name": s.name,
            "gender": "Male" if s.gender == 1 else "Female",
            "age": s.age,
            "created_at": s.created_at.strftime("%Y-%m-%d"),
            "enrollments": enrollments,
            "predictions": predictions,
            "interventions": interventions
        }


def create_student(
    name: str,
    gender: int,
    age: int,
    student_id: Optional[str] = None,
    user_id: Optional[int] = None
) -> Dict[str, Any]:
    """Create a new student record with audit logging."""
    with get_db_session() as session:
        if not student_id:
            count = session.query(Student).count()
            student_id = f"STU-2026-{count+1:04d}"

        student = Student(
            student_id=student_id.strip(),
            name=name.strip(),
            gender=gender,
            age=age
        )
        session.add(student)
        session.flush()

        session.add(AuditLog(
            user_id=user_id,
            action="CREATE_STUDENT",
            entity_type="STUDENT",
            entity_id=student.id,
            details_json=f'{{"student_id": "{student.student_id}", "name": "{student.name}"}}'
        ))

        return {"id": student.id, "student_id": student.student_id, "name": student.name}
