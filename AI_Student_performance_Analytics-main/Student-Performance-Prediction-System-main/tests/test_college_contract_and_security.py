"""Tests for College ML Data Contract, Anti-Leakage Validation, and Security Guards.
"""
import unittest
import io
from datetime import datetime, timezone
from pydantic import ValidationError

from src.schemas.college_contract import CollegeAcademicRecord, validate_college_dataset_csv
from src.services.auth_service import (
    hash_password,
    verify_password,
    has_permission,
    require_permission,
    LEGACY_STATIC_SALT
)
import hashlib


class TestCollegeContractAndSecurity(unittest.TestCase):
    """Test suite for college academic data contracts and security hardening."""

    def test_valid_college_academic_record(self):
        """A properly formatted academic record must validate cleanly."""
        record = CollegeAcademicRecord(
            student_id="STU-2026-0042",
            semester_code="SEM-IV",
            subject_code="CS401",
            checkpoint_name="MID_SEMESTER",
            midsem_1=18.5,
            midsem_2=19.0,
            activity_1=9.0,
            attendance_percentage=88.5
        )
        self.assertEqual(record.student_id, "STU-2026-0042")
        self.assertEqual(record.midsem_1, 18.5)
        # Unrecorded components must remain None (not coerced to 0.0)
        self.assertIsNone(record.midsem_3)
        self.assertIsNone(record.practical_external)
        self.assertIsNone(record.pbl)
        self.assertIsNone(record.actual_final_result)

    def test_temporal_anti_leakage_prevent_final_in_early_checkpoint(self):
        """Final verified result or grade in EARLY or MID checkpoints must be rejected to prevent target leakage."""
        with self.assertRaises(ValidationError) as ctx:
            CollegeAcademicRecord(
                student_id="STU-2026-0042",
                semester_code="SEM-IV",
                subject_code="CS401",
                checkpoint_name="EARLY_SEMESTER",
                midsem_1=16.0,
                actual_final_result=82.0  # LEAKAGE VIOLATION
            )
        self.assertIn("Temporal leakage violation", str(ctx.exception))

        with self.assertRaises(ValidationError) as ctx:
            CollegeAcademicRecord(
                student_id="STU-2026-0042",
                semester_code="SEM-IV",
                subject_code="CS401",
                checkpoint_name="MID_SEMESTER",
                midsem_1=16.0,
                actual_grade="A"  # LEAKAGE VIOLATION
            )
        self.assertIn("Temporal leakage violation", str(ctx.exception))

    def test_out_of_bounds_academic_marks(self):
        """Marks exceeding official component maxima must raise ValidationError."""
        # Midsem > 20
        with self.assertRaises(ValidationError):
            CollegeAcademicRecord(
                student_id="STU-01", semester_code="SEM-IV", subject_code="CS401",
                checkpoint_name="MID_SEMESTER", midsem_1=25.0
            )

        # Activity > 10
        with self.assertRaises(ValidationError):
            CollegeAcademicRecord(
                student_id="STU-01", semester_code="SEM-IV", subject_code="CS401",
                checkpoint_name="MID_SEMESTER", activity_1=12.0
            )

        # Practical > 30
        with self.assertRaises(ValidationError):
            CollegeAcademicRecord(
                student_id="STU-01", semester_code="SEM-IV", subject_code="CS401",
                checkpoint_name="PRE_EXAM", practical_external=35.0
            )

    def test_attended_exceeds_conducted_classes(self):
        """Classes attended cannot exceed classes conducted."""
        with self.assertRaises(ValidationError):
            CollegeAcademicRecord(
                student_id="STU-01", semester_code="SEM-IV", subject_code="CS401",
                checkpoint_name="MID_SEMESTER", classes_attended=50, classes_conducted=40
            )

    def test_explicit_missingness_reporting(self):
        """Contract must explicitly identify missing/pending components."""
        record = CollegeAcademicRecord(
            student_id="STU-01",
            semester_code="SEM-IV",
            subject_code="CS401",
            checkpoint_name="EARLY_SEMESTER",
            midsem_1=17.0,
            attendance_percentage=90.0
        )
        missing = record.missing_assessment_fields()
        self.assertIn("midsem_2", missing)
        self.assertIn("midsem_3", missing)
        self.assertIn("practical_external", missing)
        self.assertIn("internship", missing)
        self.assertNotIn("midsem_1", missing)

    def test_validate_college_dataset_csv(self):
        """Validate batch CSV importing against CollegeAcademicRecord contract."""
        csv_text = """student_id,semester_code,subject_code,checkpoint_name,midsem_1,midsem_2,activity_1,attendance_percentage
STU-001,SEM-IV,CS401,MID_SEMESTER,18.0,17.5,9.0,92.0
STU-002,SEM-IV,CS401,MID_SEMESTER,14.0,15.0,7.0,85.0
"""
        is_valid, errors, valid_df, summary = validate_college_dataset_csv(io.StringIO(csv_text))
        self.assertTrue(is_valid)
        self.assertEqual(summary["total_rows"], 2)
        self.assertEqual(summary["valid_rows"], 2)
        self.assertEqual(summary["rejected_rows"], 0)

    def test_argon2id_hashing_and_verification(self):
        """Password hashing generates secure Argon2id / PBKDF2 hash and verifies correctly."""
        password = "ComplexSecurePassword@2026!"
        h = hash_password(password)
        self.assertNotEqual(h, password)
        # Verification
        valid, needs_rehash = verify_password(password, h)
        self.assertTrue(valid)
        # Invalid password
        bad_valid, _ = verify_password("WrongPassword", h)
        self.assertFalse(bad_valid)

    def test_legacy_hash_verification_flags_rehash(self):
        """Legacy SHA-256 hash verifies and explicitly flags for rehash."""
        password = "LegacyOldPassword#123"
        legacy_hash = hashlib.sha256((password + LEGACY_STATIC_SALT).encode("utf-8")).hexdigest()

        valid, needs_rehash = verify_password(password, legacy_hash)
        self.assertTrue(valid)
        self.assertTrue(needs_rehash)


if __name__ == "__main__":
    unittest.main()
