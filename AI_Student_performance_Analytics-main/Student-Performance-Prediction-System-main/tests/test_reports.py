import unittest
import pandas as pd
from src.report_generator import generate_text_report, generate_cohort_summary_csv


class TestReportGenerator(unittest.TestCase):
    def test_text_report_content(self):
        student = {
            "Student_Name": "Pardi",
            "Gender": 1,
            "Age": 20,
            "Attendance_Percentage": 90.0,
            "Assignments_Submitted": 8,
            "Internal_Marks": 7.0,
            "Self_Study_Hours": 3.0,
            "Previous_Failures": 0,
            "Extra_Classes": 1
        }
        pred = {
            "predicted_score": 78.5,
            "uncertainty_margin": 4.2,
            "score_lower": 74.3,
            "score_upper": 82.7,
            "grade": "B",
            "performance": "Very Good",
            "risk_level": "Low Risk",
            "model_version": "v2_clean"
        }
        recs = [
            {"priority": "STRENGTH", "category": "Exemplary", "issue": "Consistent", "student_action": "Keep going", "educator_action": "Mentorship"}
        ]
        
        report = generate_text_report(student, pred, recs)
        self.assertIn("SCHOLARIQ — ACADEMIC PERFORMANCE FORECAST", report)
        self.assertIn("Pardi", report)
        self.assertIn("78.5 / 100", report)
        self.assertIn("LIMITATIONS", report)

    def test_cohort_summary_csv(self):
        df = pd.DataFrame([{"Student_Name": "Student 1", "Predicted_Score": 85.0, "Grade": "A"}])
        csv_str = generate_cohort_summary_csv(df)
        self.assertIn("Student_Name,Predicted_Score,Grade", csv_str)
        self.assertIn("Student 1,85.0,A", csv_str)


if __name__ == "__main__":
    unittest.main()
