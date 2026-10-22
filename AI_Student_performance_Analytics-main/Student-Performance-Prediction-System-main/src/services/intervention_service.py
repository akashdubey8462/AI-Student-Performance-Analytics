"""ScholarIQ Intervention Management Service.
Handles academic intervention tracking, advisor assignments, and status lifecycles.
"""
from typing import List, Dict, Any, Optional
from datetime import date, datetime, timezone
from ..database.connection import get_db_session
from ..database.models import Intervention, Student, User, AuditLog


def list_interventions(
    status_filter: Optional[str] = None,
    priority_filter: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """Retrieve filtered intervention plans."""
    with get_db_session() as session:
        query = session.query(Intervention).order_by(Intervention.created_at.desc())

        if status_filter and status_filter != "All":
            query = query.filter(Intervention.status == status_filter)

        if priority_filter and priority_filter != "All":
            query = query.filter(Intervention.priority == priority_filter)

        interventions = query.limit(limit).all()
        today = date.today()
        results = []

        for i in interventions:
            student = session.query(Student).filter_by(id=i.student_id).first()
            advisor = session.query(User).filter_by(id=i.assigned_advisor_id).first() if i.assigned_advisor_id else None

            is_overdue = (
                i.status in ["Open", "In Progress"] and
                i.follow_up_date is not None and
                i.follow_up_date < today
            )

            results.append({
                "id": i.id,
                "student_id": student.student_id if student else "—",
                "student_name": student.name if student else "Unknown",
                "priority": i.priority,
                "category": i.category,
                "issue_summary": i.issue_summary,
                "student_action": i.student_action,
                "educator_action": i.educator_action,
                "status": i.status,
                "follow_up_date": i.follow_up_date.strftime("%Y-%m-%d") if i.follow_up_date else "None",
                "is_overdue": is_overdue,
                "assigned_advisor": advisor.full_name if advisor else "Unassigned",
                "notes": i.notes or "",
                "created_at": i.created_at.strftime("%Y-%m-%d")
            })

        return results


def create_intervention(
    student_db_id: int,
    priority: str,
    category: str,
    issue_summary: str,
    student_action: str,
    educator_action: str,
    follow_up_date: Optional[date] = None,
    notes: Optional[str] = None,
    advisor_id: Optional[int] = None,
    user_id: Optional[int] = None
) -> Dict[str, Any]:
    """Create a new formal academic intervention record."""
    with get_db_session() as session:
        intervention = Intervention(
            student_id=student_db_id,
            assigned_advisor_id=advisor_id,
            priority=priority,
            category=category,
            issue_summary=issue_summary,
            student_action=student_action,
            educator_action=educator_action,
            status="Open",
            follow_up_date=follow_up_date,
            notes=notes,
            created_at=datetime.now(timezone.utc)
        )
        session.add(intervention)
        session.flush()

        session.add(AuditLog(
            user_id=user_id,
            action="CREATE_INTERVENTION",
            entity_type="INTERVENTION",
            entity_id=intervention.id,
            details_json=f'{{"student_id": {student_db_id}, "priority": "{priority}"}}'
        ))

        return {"id": intervention.id, "status": "Open"}


def update_intervention_status(
    intervention_id: int,
    new_status: str,
    notes: Optional[str] = None,
    user_id: Optional[int] = None
) -> bool:
    """Update intervention status and notes with audit log."""
    with get_db_session() as session:
        it = session.query(Intervention).filter_by(id=intervention_id).first()
        if not it:
            return False

        old_status = it.status
        it.status = new_status
        if notes:
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            it.notes = f"{it.notes or ''}\n[{timestamp_str}] {notes}".strip()
        it.updated_at = datetime.now(timezone.utc)

        session.add(AuditLog(
            user_id=user_id,
            action="UPDATE_INTERVENTION_STATUS",
            entity_type="INTERVENTION",
            entity_id=it.id,
            details_json=f'{{"old_status": "{old_status}", "new_status": "{new_status}"}}'
        ))

        return True
