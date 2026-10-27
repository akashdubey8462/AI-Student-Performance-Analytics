"""ScholarIQ Database, Services, and Security Integration Test Suite.
Verifies persistence, transactions, authentication, role authorization, Argon2id migration, and intervention workflows
under strict test database isolation.
"""
import io
import unittest
from datetime import date, timedelta
from src.database.connection import get_db_session, init_db, drop_db, set_engine_url, DEFAULT_SQLITE_PATH
from src.database.models import User, Student, Course, Term, Enrollment, ModelVersion, Prediction, Intervention, AuditLog
from src.services.auth_service import (
    authenticate_user,
    has_permission,
    require_permission,
    hash_password,
    verify_password,
    LEGACY_STATIC_SALT
)
import hashlib
from src.services.student_service import list_students, get_student_by_id, create_student
from src.services.prediction_service import save_prediction, list_predictions, get_overview_kpis
from src.services.intervention_service import list_interventions, create_intervention, update_intervention_status
from src.services.import_service import import_roster_transactional


class TestDatabaseAndServices(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Bind to isolated in-memory test database
        set_engine_url("sqlite:///:memory:")
        init_db()

        # Seed explicit, hermetic test fixtures
        with get_db_session() as s:
            # 1. Test Users
            admin_user = User(
                username="admin",
                password_hash=hash_password("Admin@123"),
                full_name="Dr. Elena Vance (Dean of Academics)",
                role="admin",
                email="admin.test@scholariq.edu"
            )
            advisor_user = User(
                username="advisor",
                password_hash=hash_password("Advisor@123"),
                full_name="Sarah Jenkins (Senior Academic Advisor)",
                role="advisor",
                email="advisor.test@scholariq.edu"
            )
            s.add_all([admin_user, advisor_user])
            s.flush()

            # 2. Test Model Version
            model_ver = ModelVersion(
                version_tag="v2.0-clean",
                algorithm="RandomForestRegressor (Tuned, max_depth=6)",
                artifact_path="data/student_model_v2_clean.pkl",
                cv_r2=0.714,
                test_mae=8.31,
                is_active=True
            )
            s.add(model_ver)
            s.flush()

            # 3. Test Term & Course
            term = Term(term_code="2026-FALL", name="Fall 2026")
            s.add(term)
            s.flush()

            course = Course(course_code="MATH201", course_name="Applied Linear Algebra", term_id=term.id)
            s.add(course)
            s.flush()

            # 4. Baseline Student & Enrollment
            student = Student(student_id="STU-BASE-001", name="Baseline Student", gender=1, age=18)
            s.add(student)
            s.flush()

            enrollment = Enrollment(
                student_id=student.id,
                course_id=course.id,
                attendance_percentage=90.0,
                assignments_submitted=7,
                internal_marks=7.0,
                self_study_hours=3.5,
                previous_failures=0,
                extra_classes=0
            )
            s.add(enrollment)

    @classmethod
    def tearDownClass(cls):
        # Tear down in-memory tables and restore standard path
        drop_db()
        set_engine_url(f"sqlite:///{DEFAULT_SQLITE_PATH}")

    def test_01_user_authentication(self):
        # Admin auth success
        user = authenticate_user("admin", "Admin@123")
        self.assertIsNotNone(user)
        self.assertEqual(user["username"], "admin")
        self.assertEqual(user["role"], "admin")

        # Wrong password
        bad_auth = authenticate_user("admin", "WrongPassword")
        self.assertIsNone(bad_auth)

        # Nonexistent user
        no_user = authenticate_user("nonexistent_user", "Pass@123")
        self.assertIsNone(no_user)

    def test_02_role_permissions(self):
        self.assertTrue(has_permission("admin", "student"))
        self.assertTrue(has_permission("admin", "faculty"))
        self.assertTrue(has_permission("faculty", "advisor"))
        self.assertTrue(has_permission("advisor", "student"))
        self.assertFalse(has_permission("student", "faculty"))
        self.assertFalse(has_permission("student", "admin"))

    def test_03_student_crud_and_listing(self):
        new_stu = create_student(name="Test Integration Student", gender=0, age=19)
        self.assertIsNotNone(new_stu["id"])
        self.assertTrue(new_stu["student_id"].startswith("STU-"))

        students = list_students(search_query="Test Integration")
        self.assertTrue(any(s["name"] == "Test Integration Student" for s in students))

        detail = get_student_by_id(new_stu["id"])
        self.assertEqual(detail["name"], "Test Integration Student")
        self.assertEqual(detail["gender"], "Female")

    def test_04_prediction_persistence_and_snapshot(self):
        with get_db_session() as s:
            first_stu = s.query(Student).filter_by(student_id="STU-BASE-001").first()
            self.assertIsNotNone(first_stu)
            stu_id = first_stu.id

        features = {
            "Gender": 1,
            "Age": 18,
            "Assignments_Submitted": 8,
            "Internal_Marks": 7.0,
            "Attendance_Percentage": 92.0,
            "Self_Study_Hours": 3.0,
            "Previous_Failures": 0,
            "Extra_Classes": 0
        }

        res = save_prediction(student_db_id=stu_id, features=features, model_version_tag="v2.0-clean")
        self.assertIn("predicted_score", res)
        self.assertIn("prediction_db_id", res)
        self.assertGreaterEqual(res["predicted_score"], 0.0)
        self.assertLessEqual(res["predicted_score"], 100.0)

        # Check DB record
        with get_db_session() as s:
            pred_db = s.query(Prediction).filter_by(id=res["prediction_db_id"]).first()
            self.assertIsNotNone(pred_db)
            self.assertEqual(pred_db.student_id, stu_id)
            self.assertIn('"Internal_Marks": 7.0', pred_db.feature_snapshot_json)

    def test_05_intervention_lifecycle(self):
        with get_db_session() as s:
            stu = s.query(Student).filter_by(student_id="STU-BASE-001").first()
            adv = s.query(User).filter_by(role="advisor").first()
            stu_id = stu.id
            adv_id = adv.id

        # Create
        it_res = create_intervention(
            student_db_id=stu_id,
            priority="HIGH",
            category="Attendance",
            issue_summary="Attendance below threshold",
            student_action="Attend daily lectures",
            educator_action="Weekly check-in",
            follow_up_date=date.today() + timedelta(days=5),
            advisor_id=adv_id
        )
        self.assertEqual(it_res["status"], "Open")

        # Update
        updated = update_intervention_status(
            intervention_id=it_res["id"],
            new_status="Completed",
            notes="Student recovered attendance."
        )
        self.assertTrue(updated)

        with get_db_session() as s:
            it_db = s.query(Intervention).filter_by(id=it_res["id"]).first()
            self.assertEqual(it_db.status, "Completed")
            self.assertIn("Student recovered attendance", it_db.notes)

    def test_06_transactional_batch_import(self):
        csv_data = """Student_Name,Gender,Age,Assignments_Submitted,Internal_Marks,Attendance_Percentage,Self_Study_Hours,Previous_Failures,Extra_Classes
Batch Test One,1,18,7,6.5,88.0,3.0,0,0
Batch Test Two,0,17,9,8.0,95.0,4.0,0,1
"""
        success, res = import_roster_transactional(
            io.StringIO(csv_data),
            filename="test_batch.csv",
            course_code="MATH201"
        )
        self.assertTrue(success)
        self.assertEqual(res["total_rows"], 2)
        self.assertEqual(res["valid_rows"], 2)
        self.assertEqual(res["rejected_rows"], 0)

    def test_07_kpi_calculations(self):
        kpis = get_overview_kpis()
        self.assertIn("total_students", kpis)
        self.assertIn("average_predicted_score", kpis)
        self.assertIn("projected_pass_rate", kpis)
        self.assertGreater(kpis["total_students"], 0)

    def test_08_password_hashing_and_transparent_legacy_migration(self):
        """Verify that legacy SHA-256 hashes are automatically migrated to modern Argon2id upon valid login."""
        legacy_pass = "LegacyPass@2026"
        legacy_hash = hashlib.sha256((legacy_pass + LEGACY_STATIC_SALT).encode("utf-8")).hexdigest()

        with get_db_session() as s:
            migrating_user = User(
                username="legacy_user",
                password_hash=legacy_hash,
                full_name="Legacy Faculty User",
                role="faculty",
                email="legacy@scholariq.edu"
            )
            s.add(migrating_user)

        # Authenticate with plain password
        user = authenticate_user("legacy_user", legacy_pass)
        self.assertIsNotNone(user)
        self.assertEqual(user["username"], "legacy_user")

        # Verify hash was automatically upgraded in the database
        with get_db_session() as s:
            updated_user = s.query(User).filter_by(username="legacy_user").first()
            self.assertNotEqual(updated_user.password_hash, legacy_hash)
            # Must start with argon2 or pbkdf2
            self.assertTrue(
                updated_user.password_hash.startswith("$argon2") or updated_user.password_hash.startswith("pbkdf2_sha256$")
            )

    def test_09_anonymous_access_and_service_authorization_guard(self):
        """Verify that require_permission enforces fail-closed behavior for unauthenticated or unauthorized users."""
        # Unauthenticated user
        with self.assertRaises(PermissionError):
            require_permission(None, "faculty")

        # Unauthorized role
        student_user = {"username": "student_user", "role": "student"}
        with self.assertRaises(PermissionError):
            require_permission(student_user, "faculty")

        with self.assertRaises(PermissionError):
            require_permission(student_user, "admin")

        # Authorized role passes cleanly
        admin_user = {"username": "admin_user", "role": "admin"}
        require_permission(admin_user, "faculty")  # Should not raise


if __name__ == "__main__":
    unittest.main()
