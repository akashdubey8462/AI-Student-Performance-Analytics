import io
import unittest
import pandas as pd
from src.data_validation import validate_student_input, validate_batch_csv


class TestDataValidation(unittest.TestCase):
    def setUp(self):
        self.valid_input = {
            "Student_Name": "Aditi Sharma",
            "Gender": 0,
            "Age": 17,
            "Assignments_Submitted": 8,
            "Internal_Marks": 7.5,
            "Attendance_Percentage": 92.0,
            "Self_Study_Hours": 3.5,
            "Previous_Failures": 0,
            "Extra_Classes": 0
        }

    def test_valid_student_input(self):
        is_valid, errors, df = validate_student_input(self.valid_input)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 1)
        self.assertEqual(list(df.columns), [
            "Gender", "Age", "Assignments_Submitted", "Internal_Marks",
            "Attendance_Percentage", "Self_Study_Hours", "Previous_Failures", "Extra_Classes"
        ])

    def test_missing_student_name(self):
        data = self.valid_input.copy()
        data["Student_Name"] = "   "
        is_valid, errors, df = validate_student_input(data)
        self.assertFalse(is_valid)
        self.assertIn("Student name is required", errors[0])
        self.assertIsNone(df)

    def test_out_of_range_attributes(self):
        data = self.valid_input.copy()
        data["Attendance_Percentage"] = 150.0  # Out of range [0, 100]
        data["Internal_Marks"] = -2.0          # Out of range [0, 10]
        is_valid, errors, df = validate_student_input(data)
        self.assertFalse(is_valid)
        self.assertEqual(len(errors), 2)
        self.assertIsNone(df)

    def test_missing_attribute(self):
        data = self.valid_input.copy()
        del data["Internal_Marks"]
        is_valid, errors, df = validate_student_input(data)
        self.assertFalse(is_valid)
        self.assertTrue(any("Missing required attribute: 'Internal_Marks'" in e for e in errors))

    def test_valid_batch_csv(self):
        csv_content = """Student_Name,Gender,Age,Assignments_Submitted,Internal_Marks,Attendance_Percentage,Self_Study_Hours,Previous_Failures,Extra_Classes
Student 1,0,16,8,7.0,94.0,3.5,0,0
Student 2,1,18,5,5.0,80.0,2.0,1,1
"""
        is_valid, errors, clean_df, summary = validate_batch_csv(io.StringIO(csv_content))
        self.assertTrue(is_valid)
        self.assertEqual(summary["total_rows"], 2)
        self.assertEqual(summary["valid_rows"], 2)
        self.assertEqual(summary["rejected_rows"], 0)
        self.assertEqual(len(clean_df), 2)

    def test_batch_csv_with_partial_errors(self):
        csv_content = """Student_Name,Gender,Age,Assignments_Submitted,Internal_Marks,Attendance_Percentage,Self_Study_Hours,Previous_Failures,Extra_Classes
Student 1,0,16,8,7.0,94.0,3.5,0,0
Invalid Student,1,35,8,7.0,94.0,3.5,0,0
"""
        is_valid, errors, clean_df, summary = validate_batch_csv(io.StringIO(csv_content))
        self.assertTrue(is_valid)
        self.assertEqual(summary["total_rows"], 2)
        self.assertEqual(summary["valid_rows"], 1)
        self.assertEqual(summary["rejected_rows"], 1)
        self.assertEqual(len(clean_df), 1)

    def test_empty_csv(self):
        csv_content = ""
        is_valid, errors, clean_df, summary = validate_batch_csv(io.StringIO(csv_content))
        self.assertFalse(is_valid)


if __name__ == "__main__":
    unittest.main()
