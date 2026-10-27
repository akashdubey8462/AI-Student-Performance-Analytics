"""ScholarIQ SQLAlchemy Entity Models.
Defines normalized tables for academic records, predictions, interventions, and audit trails.
"""
from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Date,
    ForeignKey, Text, Boolean, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship
from .connection import Base


def utc_now():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=False)
    role = Column(String(30), nullable=False, default="faculty")  # admin, faculty, advisor, student
    email = Column(String(120), unique=True, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utc_now)

    audit_logs = relationship("AuditLog", back_populates="user")
    assigned_interventions = relationship("Intervention", back_populates="advisor")


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    gender = Column(Integer, nullable=False)  # 0=Female, 1=Male
    age = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=utc_now)

    enrollments = relationship("Enrollment", back_populates="student", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="student", cascade="all, delete-orphan")
    interventions = relationship("Intervention", back_populates="student", cascade="all, delete-orphan")


class Term(Base):
    __tablename__ = "terms"

    id = Column(Integer, primary_key=True, index=True)
    term_code = Column(String(30), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)

    courses = relationship("Course", back_populates="term")


class Course(Base):
    __tablename__ = "courses"

    id = Column(Integer, primary_key=True, index=True)
    course_code = Column(String(30), unique=True, nullable=False, index=True)
    course_name = Column(String(150), nullable=False)
    term_id = Column(Integer, ForeignKey("terms.id"), nullable=True)

    term = relationship("Term", back_populates="courses")
    enrollments = relationship("Enrollment", back_populates="course")


class Enrollment(Base):
    __tablename__ = "enrollments"
    __table_args__ = (
        UniqueConstraint("student_id", "course_id", name="uq_student_course"),
    )

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False, index=True)
    
    # Academic & behavioral indicators for inference
    attendance_percentage = Column(Float, nullable=False, default=90.0)
    assignments_submitted = Column(Integer, nullable=False, default=7)
    internal_marks = Column(Float, nullable=False, default=6.0)
    self_study_hours = Column(Float, nullable=False, default=3.0)
    previous_failures = Column(Integer, nullable=False, default=0)
    extra_classes = Column(Integer, nullable=False, default=0)
    
    # Continuous assessment specific scores
    midsem_1 = Column(Float, nullable=True)
    midsem_2 = Column(Float, nullable=True)
    midsem_3 = Column(Float, nullable=True)
    activity_1 = Column(Float, nullable=True)
    activity_2 = Column(Float, nullable=True)
    practical_score = Column(Float, nullable=True)
    term_work_score = Column(Float, nullable=True)
    pbl_score = Column(Float, nullable=True)
    internship_score = Column(Float, nullable=True)

    # Observed outcome (populated once final exam is graded)
    final_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=utc_now)

    student = relationship("Student", back_populates="enrollments")
    course = relationship("Course", back_populates="enrollments")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    version_tag = Column(String(50), unique=True, nullable=False, index=True)
    algorithm = Column(String(100), nullable=False)
    artifact_path = Column(String(255), nullable=False)
    dataset_hash = Column(String(64), nullable=True)
    cv_r2 = Column(Float, nullable=True)
    test_mae = Column(Float, nullable=True)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utc_now)

    predictions = relationship("Prediction", back_populates="model_version")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False, index=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=False, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=True, index=True)
    term_id = Column(Integer, ForeignKey("terms.id"), nullable=True, index=True)
    checkpoint_name = Column(String(100), nullable=True, default="Standard Assessment")
    
    predicted_score = Column(Float, nullable=False)
    confidence_margin = Column(Float, nullable=False, default=5.0)
    score_lower = Column(Float, nullable=False)
    score_upper = Column(Float, nullable=False)
    grade = Column(String(10), nullable=False)
    risk_level = Column(String(30), nullable=False)
    
    # Uncertainty methodology & outcome verification
    uncertainty_method = Column(String(100), nullable=False, default="Heuristic Empirical Interval (±5.0)")
    actual_final_score = Column(Float, nullable=True)
    actual_grade = Column(String(10), nullable=True)
    error_residual = Column(Float, nullable=True)
    
    # Auditable snapshot of input features at prediction time
    feature_snapshot_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utc_now, index=True)

    student = relationship("Student", back_populates="predictions")
    model_version = relationship("ModelVersion", back_populates="predictions")
    course = relationship("Course")
    term = relationship("Term")


class Intervention(Base):
    __tablename__ = "interventions"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False, index=True)
    assigned_advisor_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    
    priority = Column(String(20), nullable=False)  # CRITICAL, HIGH, MEDIUM, STRENGTH
    category = Column(String(50), nullable=False)
    issue_summary = Column(Text, nullable=False)
    student_action = Column(Text, nullable=False)
    educator_action = Column(Text, nullable=False)
    status = Column(String(30), nullable=False, default="Open")  # Open, In Progress, Completed, Cancelled
    follow_up_date = Column(Date, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    student = relationship("Student", back_populates="interventions")
    advisor = relationship("User", back_populates="assigned_interventions")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    action = Column(String(100), nullable=False, index=True)
    entity_type = Column(String(50), nullable=False)
    entity_id = Column(Integer, nullable=True)
    details_json = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=utc_now, index=True)

    user = relationship("User", back_populates="audit_logs")


class ImportJob(Base):
    __tablename__ = "import_jobs"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    total_rows = Column(Integer, nullable=False, default=0)
    valid_rows = Column(Integer, nullable=False, default=0)
    rejected_rows = Column(Integer, nullable=False, default=0)
    status = Column(String(30), nullable=False, default="SUCCESS")  # SUCCESS, PARTIAL, FAILED
    timestamp = Column(DateTime, default=utc_now, index=True)

    errors = relationship("ImportErrorRecord", back_populates="job", cascade="all, delete-orphan")


class ImportErrorRecord(Base):
    __tablename__ = "import_errors"

    id = Column(Integer, primary_key=True, index=True)
    import_job_id = Column(Integer, ForeignKey("import_jobs.id"), nullable=False, index=True)
    row_number = Column(Integer, nullable=False)
    error_message = Column(Text, nullable=False)

    job = relationship("ImportJob", back_populates="errors")
