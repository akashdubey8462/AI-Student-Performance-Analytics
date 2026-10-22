"""ScholarIQ Database Package.
Persistence layer handling relational database models, connections, and repositories.
"""
from .connection import get_db_session, init_db, engine
from .models import (
    Base, User, Student, Term, Course, Enrollment,
    ModelVersion, Prediction, Intervention, AuditLog,
    ImportJob, ImportErrorRecord
)
