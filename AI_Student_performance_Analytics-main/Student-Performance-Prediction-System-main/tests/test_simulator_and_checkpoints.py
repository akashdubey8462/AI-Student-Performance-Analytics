"""Unit tests for What-If Simulator, Semester Checkpoint Timeline, Latency Benchmarking, and Student Profile Updates.
"""
import unittest
import uuid
from src.services.simulator_service import simulate_what_if_scenario
from src.services.prediction_service import (
    save_prediction,
    get_student_prediction_timeline,
    record_actual_outcome
)
from src.services.student_service import (
    create_student,
    update_student_profile,
    update_student_enrollment,
    check_duplicate_student_id
)
from src.model_service import ModelService


class TestSimulatorAndCheckpoints(unittest.TestCase):
    """Test suite for what-if scenarios, checkpoint tracking, and student editing."""

    def test_what_if_simulation_attendance_boost(self):
        """Increasing attendance from 60% to 88% should trigger attendance bonus and increase ML score."""
        baseline = {
            "midsem_1": 15.0,
            "midsem_2": 15.0,
            "midsem_3": 14.0,
            "attendance_percentage": 60.0,  # 0 bonus marks
            "activity_1": 7.0,
            "activity_2": 8.0,
            "practical_external": 22.0,
            "term_work": 15.0,
            "pbl": 35.0,
            "internship": 35.0,
            "assignments_submitted": 6,
            "self_study_hours": 2.5,
            "previous_failures": 0,
            "extra_classes": 0,
            "gender": "Female",
            "age": 18
        }
        
        scenario = dict(baseline)
        scenario["attendance_percentage"] = 88.0  # +3 bonus marks

        sim = simulate_what_if_scenario(baseline, scenario)

        # Marking engine checks
        self.assertEqual(sim["baseline_marking"]["attendance_bonus"], 0)
        self.assertEqual(sim["scenario_marking"]["attendance_bonus"], 3)
        self.assertEqual(sim["bonus_delta"], 3)
        self.assertEqual(sim["subtotal_delta"], 3.0)

        # ML prediction checks
        self.assertGreater(sim["scenario_ml"]["predicted_score"], sim["baseline_ml"]["predicted_score"])
        self.assertGreater(sim["score_delta"], 0.0)
        self.assertGreaterEqual(sim["cgpa_scenario"], sim["cgpa_baseline"])
        self.assertTrue(len(sim["insights"]) > 0)

    def test_what_if_simulation_midsem_improvement(self):
        """Increasing midsem scores should increase midsem average, provisional subtotal, and ML score."""
        baseline = {
            "midsem_1": 10.0,
            "midsem_2": 12.0,
            "midsem_3": 11.0,
            "attendance_percentage": 75.0,
            "activity_1": 6.0,
            "activity_2": 7.0,
            "practical_external": 20.0,
            "term_work": 14.0,
            "pbl": 30.0,
            "internship": 30.0
        }
        scenario = dict(baseline)
        scenario["midsem_1"] = 18.0
        scenario["midsem_2"] = 19.0

        sim = simulate_what_if_scenario(baseline, scenario)

        self.assertGreater(sim["scenario_marking"]["midsem_avg"], sim["baseline_marking"]["midsem_avg"])
        self.assertGreater(sim["subtotal_delta"], 0.0)
        self.assertGreater(sim["score_delta"], 0.0)

    def test_model_inference_latency_benchmark(self):
        """Model inference latency benchmark must execute and return valid timing metrics."""
        service = ModelService(model_version="v2_clean")
        bench = service.measure_inference_latency(n_iterations=20)

        self.assertEqual(bench["iterations"], 20)
        self.assertGreater(bench["mean_ms"], 0.0)
        self.assertLess(bench["mean_ms"], 250.0)
        self.assertIn("p95_ms", bench)

    def test_student_checkpoint_progress_timeline(self):
        """Verify recording progressive checkpoints and tracking accuracy against actual final score."""
        uid = uuid.uuid4().hex[:6]
        created = create_student(name=f"Timeline Student {uid}", gender=1, age=19, student_id=f"STU-TL-{uid}")
        s_id = created["id"]

        # Checkpoint 1 (Midterm)
        f1 = {
            "Gender": 1, "Age": 19, "Assignments_Submitted": 5, "Internal_Marks": 5.0,
            "Attendance_Percentage": 70.0, "Self_Study_Hours": 2.0, "Previous_Failures": 0, "Extra_Classes": 0
        }
        save_prediction(s_id, f1, checkpoint_name="Milestone 1: Early Assessment")

        # Checkpoint 2 (Pre-Final)
        f2 = {
            "Gender": 1, "Age": 19, "Assignments_Submitted": 9, "Internal_Marks": 8.0,
            "Attendance_Percentage": 90.0, "Self_Study_Hours": 4.5, "Previous_Failures": 0, "Extra_Classes": 1
        }
        save_prediction(s_id, f2, checkpoint_name="Milestone 2: Pre-Final Comprehensive")

        timeline = get_student_prediction_timeline(s_id)
        self.assertEqual(len(timeline), 2)
        self.assertEqual(timeline[0]["checkpoint_name"], "Milestone 1: Early Assessment")
        self.assertEqual(timeline[1]["checkpoint_name"], "Milestone 2: Pre-Final Comprehensive")
        self.assertGreater(timeline[1]["predicted_score"], timeline[0]["predicted_score"])

    def test_student_update_and_duplicate_prevention(self):
        """Verify student profile editing and duplicate student_id prevention."""
        uid1 = uuid.uuid4().hex[:6]
        uid2 = uuid.uuid4().hex[:6]
        created = create_student(name=f"Edit Student {uid1}", gender=0, age=20, student_id=f"STU-ED-{uid1}")
        s_id = created["id"]

        # Update name and age
        ok, msg = update_student_profile(s_id, name="Updated Student Name", gender=0, age=21)
        self.assertTrue(ok)

        # Create another student
        created2 = create_student(name=f"Other Student {uid2}", gender=1, age=20, student_id=f"STU-ED-{uid2}")
        s2_id = created2["id"]

        # Attempt to change second student's ID to the first student's ID (must be blocked)
        duplicate_ok, dup_msg = update_student_profile(s2_id, name="Other Student", gender=1, age=20, student_id=f"STU-ED-{uid1}")
        self.assertFalse(duplicate_ok)
        self.assertIn("already in use", dup_msg)


if __name__ == "__main__":
    unittest.main()
