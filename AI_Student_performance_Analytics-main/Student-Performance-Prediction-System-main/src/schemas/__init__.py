"""ScholarIQ Data Contracts and Schema Validation.
"""
from .college_contract import CollegeAcademicRecord, validate_college_dataset_csv

__all__ = ["CollegeAcademicRecord", "validate_college_dataset_csv"]
