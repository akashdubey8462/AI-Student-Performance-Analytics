import unittest
from src.recommendations import generate_recommendations


class TestRecommendations(unittest.TestCase):
    def test_low_attendance_triggers_warning(self):
        student = {"Attendance_Percentage": 65.0, "Assignments_Submitted": 8, "Internal_Marks": 7.0, "Self_Study_Hours": 3.0, "Previous_Failures": 0}
        pred = {"predicted_score": 60.0}
        recs = generate_recommendations(student, pred)
        self.assertTrue(any(r["category"] == "Attendance & Engagement" for r in recs))

    def test_previous_failures_triggers_critical(self):
        student = {"Attendance_Percentage": 90.0, "Assignments_Submitted": 8, "Internal_Marks": 7.0, "Self_Study_Hours": 3.0, "Previous_Failures": 2}
        pred = {"predicted_score": 55.0}
        recs = generate_recommendations(student, pred)
        failure_rec = next((r for r in recs if r["category"] == "Curriculum Continuity"), None)
        self.assertIsNotNone(failure_rec)
        self.assertEqual(failure_rec["priority"], "CRITICAL")

    def test_exemplary_student_triggers_strength(self):
        student = {"Attendance_Percentage": 98.0, "Assignments_Submitted": 10, "Internal_Marks": 9.0, "Self_Study_Hours": 4.5, "Previous_Failures": 0, "Extra_Classes": 1}
        pred = {"predicted_score": 92.0}
        recs = generate_recommendations(student, pred)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["priority"], "STRENGTH")


if __name__ == "__main__":
    unittest.main()
