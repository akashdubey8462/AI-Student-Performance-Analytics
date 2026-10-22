import unittest
import pandas as pd
from src.model_service import ModelService
from src.config import FEATURE_COLUMNS


class TestModelService(unittest.TestCase):
    def setUp(self):
        self.service = ModelService(model_version="v2_clean")
        self.test_df = pd.DataFrame([{
            "Gender": 1,
            "Age": 17,
            "Assignments_Submitted": 8,
            "Internal_Marks": 7.0,
            "Attendance_Percentage": 92.0,
            "Self_Study_Hours": 3.0,
            "Previous_Failures": 0,
            "Extra_Classes": 0
        }])[FEATURE_COLUMNS]

    def test_model_loading(self):
        self.assertIsNotNone(self.service.model)
        self.assertIsNone(self.service.load_error)
        self.assertEqual(self.service.version, "v2_clean")

    def test_single_prediction_output(self):
        result = self.service.predict_single(self.test_df)
        self.assertIn("predicted_score", result)
        self.assertIn("uncertainty_margin", result)
        self.assertIn("grade", result)
        self.assertIn("risk_level", result)
        self.assertIn("feature_contributions", result)
        
        # Check boundary
        self.assertGreaterEqual(result["predicted_score"], 0.0)
        self.assertLessEqual(result["predicted_score"], 100.0)
        self.assertGreater(result["uncertainty_margin"], 0.0)

    def test_grade_mapping(self):
        self.assertEqual(self.service.grade_for_score(95.0)["grade"], "A+")
        self.assertEqual(self.service.grade_for_score(82.0)["grade"], "A")
        self.assertEqual(self.service.grade_for_score(72.0)["grade"], "B")
        self.assertEqual(self.service.grade_for_score(62.0)["grade"], "C")
        self.assertEqual(self.service.grade_for_score(52.0)["grade"], "D")
        self.assertEqual(self.service.grade_for_score(35.0)["grade"], "F")

    def test_batch_prediction(self):
        batch = pd.concat([self.test_df, self.test_df], ignore_index=True)
        batch["Student_Name"] = ["Student A", "Student B"]
        out_df = self.service.predict_batch(batch)
        self.assertEqual(len(out_df), 2)
        self.assertIn("Predicted_Score", out_df.columns)
        self.assertIn("Grade", out_df.columns)
        self.assertIn("Risk_Level", out_df.columns)
        self.assertIn("Passed", out_df.columns)


if __name__ == "__main__":
    unittest.main()
