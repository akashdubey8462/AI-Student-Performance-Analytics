"""ScholarIQ Student Management Service.
Handles student profile queries, academic directory search, and enrollment records.
"""
from typing import List, Dict, Any, Optional, Tuple
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


def check_duplicate_student_id(student_id: str, exclude_id: Optional[int] = None) -> bool:
    """Check if a student_id already exists in the database."""
    with get_db_session() as session:
        query = session.query(Student).filter_by(student_id=student_id.strip())
        if exclude_id is not None:
            query = query.filter(Student.id != exclude_id)
        return query.first() is not None


def update_student_profile(
    student_db_id: int,
    name: str,
    gender: int,
    age: int,
    student_id: Optional[str] = None,
    user_id: Optional[int] = None
) -> Tuple[bool, str]:
    """Update student demographic fields with duplicate checking and audit logging."""
    with get_db_session() as session:
        student = session.query(Student).filter_by(id=student_db_id).first()
        if not student:
            return False, f"Student with ID {student_db_id} not found."

        if student_id and student_id.strip() != student.student_id:
            if check_duplicate_student_id(student_id, exclude_id=student_db_id):
                return False, f"Student ID '{student_id.strip()}' is already in use by another student."
            student.student_id = student_id.strip()

        student.name = name.strip()
        student.gender = gender
        student.age = age

        session.add(AuditLog(
            user_id=user_id,
            action="UPDATE_STUDENT_PROFILE",
            entity_type="STUDENT",
            entity_id=student.id,
            details_json=f'{{"student_id": "{student.student_id}", "name": "{student.name}"}}'
        ))

        return True, "Student profile updated successfully."


def update_student_enrollment(
    student_db_id: int,
    attendance: float,
    internal_marks: float,
    assignments: int,
    study_hours: float,
    failures: int,
    extra_classes: int,
    actual_final_score: Optional[float] = None,
    user_id: Optional[int] = None
) -> Tuple[bool, str]:
    """Update student continuous assessment signals and recorded actual final score."""
    with get_db_session() as session:
        student = session.query(Student).filter_by(id=student_db_id).first()
        if not student or not student.enrollments:
            return False, "Enrollment record not found for student."

        enrollment = student.enrollments[0]
        enrollment.attendance_percentage = float(attendance)
        enrollment.internal_marks = float(internal_marks)
        enrollment.assignments_submitted = int(assignments)
        enrollment.self_study_hours = float(study_hours)
        enrollment.previous_failures = int(failures)
        enrollment.extra_classes = int(extra_classes)
        if actual_final_score is not None:
            enrollment.final_score = float(actual_final_score)

        session.add(AuditLog(
            user_id=user_id,
            action="UPDATE_ENROLLMENT_ASSESSMENT",
            entity_type="ENROLLMENT",
            entity_id=enrollment.id,
            details_json=f'{{"student_id": "{student.student_id}", "actual_score": {actual_final_score}}}'
        ))

        return True, "Assessment record updated successfully."


