"""Unit tests for ScholarIQ Academic Marking & Calculation Service.
Validates official marking rules, selection policies, attendance bonuses,
component validations, and incomplete-scheme warnings.
"""
import unittest
from src.services.marking_service import (
    calculate_attendance_bonus,
    calculate_midsem_average,
    calculate_counted_activity,
    calculate_academic_marks,
    DEFAULT_ACADEMIC_RULES
)
from src.services.rules_service import (
    get_academic_rules,
    save_academic_rules,
    reset_academic_rules,
    validate_academic_rules
)


class TestMarkingService(unittest.TestCase):
    """Test suite for academic marking rules and calculation engine."""

    def test_attendance_bonus_exact_boundaries(self):
        """Test non-cumulative attendance bonus thresholds at exact boundary percentages.
        Thresholds:
          <= 60% -> 0
          > 60% and <= 75% -> 1
          > 75% and <= 85% -> 2
          > 85% -> 3
        """
        # Strictly below or at 60%
        self.assertEqual(calculate_attendance_bonus(0.0), 0)
        self.assertEqual(calculate_attendance_bonus(59.0), 0)
        self.assertEqual(calculate_attendance_bonus(59.9), 0)
        self.assertEqual(calculate_attendance_bonus(60.0), 0)

        # Above 60% up to 75%
        self.assertEqual(calculate_attendance_bonus(60.1), 1)
        self.assertEqual(calculate_attendance_bonus(61.0), 1)
        self.assertEqual(calculate_attendance_bonus(74.9), 1)
        self.assertEqual(calculate_attendance_bonus(75.0), 1)

        # Above 75% up to 85%
        self.assertEqual(calculate_attendance_bonus(75.1), 2)
        self.assertEqual(calculate_attendance_bonus(76.0), 2)
        self.assertEqual(calculate_attendance_bonus(84.9), 2)
        self.assertEqual(calculate_attendance_bonus(85.0), 2)

        # Above 85%
        self.assertEqual(calculate_attendance_bonus(85.1), 3)
        self.assertEqual(calculate_attendance_bonus(86.0), 3)
        self.assertEqual(calculate_attendance_bonus(95.0), 3)
        self.assertEqual(calculate_attendance_bonus(100.0), 3)

    def test_attendance_bonus_invalid_ranges(self):
        """Attendance outside [0, 100] must raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_attendance_bonus(-1.0)
        with self.assertRaises(ValueError):
            calculate_attendance_bonus(101.0)

    def test_midsem_average_best_2_of_3(self):
        """Test mid-sem average: best 2 of 3 exams (sum divided by 2, NOT 3)."""
        # [18, 14, 16] -> best two are 18 and 16 -> avg = (18 + 16) / 2 = 17.0
        # If someone erroneously divided by 3, they would get (18+16)/3 = 11.33
        avg, selected = calculate_midsem_average(18.0, 14.0, 16.0, policy="best_2_of_3", max_marks=20.0)
        self.assertEqual(avg, 17.0)
        self.assertEqual(selected, [18.0, 16.0])

        # All equal
        avg2, selected2 = calculate_midsem_average(20.0, 20.0, 20.0)
        self.assertEqual(avg2, 20.0)
        self.assertEqual(selected2, [20.0, 20.0])

        # Lowest first
        avg3, selected3 = calculate_midsem_average(10.0, 15.0, 18.0)
        self.assertEqual(avg3, 16.5)
        self.assertEqual(selected3, [18.0, 15.0])

    def test_midsem_average_alternative_policies(self):
        """Test first 2 and all 3 selection policies."""
        # first_2 policy
        avg_f2, sel_f2 = calculate_midsem_average(12.0, 14.0, 20.0, policy="first_2")
        self.assertEqual(avg_f2, 13.0)
        self.assertEqual(sel_f2, [12.0, 14.0])

        # all_3 policy (divides by 3)
        avg_all, sel_all = calculate_midsem_average(15.0, 18.0, 12.0, policy="all_3")
        self.assertEqual(avg_all, 15.0)
        self.assertEqual(len(sel_all), 3)

    def test_midsem_out_of_range(self):
        """Marks > max_marks (20) or < 0 must raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_midsem_average(21.0, 15.0, 15.0)
        with self.assertRaises(ValueError):
            calculate_midsem_average(15.0, -1.0, 15.0)

    def test_counted_activity_max_of_two(self):
        """Higher of two activity marks must be selected."""
        # Act 2 higher
        score, winner = calculate_counted_activity(7.5, 9.0)
        self.assertEqual(score, 9.0)
        self.assertEqual(winner, 2)

        # Act 1 higher
        score2, winner2 = calculate_counted_activity(10.0, 8.5)
        self.assertEqual(score2, 10.0)
        self.assertEqual(winner2, 1)

        # Equal marks
        score3, winner3 = calculate_counted_activity(8.0, 8.0)
        self.assertEqual(score3, 8.0)
        self.assertEqual(winner3, 1)

    def test_activity_out_of_range(self):
        """Activity mark > 10 or < 0 must raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_counted_activity(11.0, 5.0)
        with self.assertRaises(ValueError):
            calculate_counted_activity(5.0, -2.0)

    def test_full_academic_calculation_provisional_subtotal(self):
        """Test complete calculation and verify provisional subtotal and 420-mark accounting."""
        inputs = {
            "midsem_1": 18.0,
            "midsem_2": 16.0,
            "midsem_3": 14.0,
            "attendance_percentage": 88.0,  # >85% -> 3 bonus marks
            "activity_1": 8.0,
            "activity_2": 9.5,              # higher of two: 9.5
            "practical_external": 26.0,     # max 30
            "term_work": 18.0,              # max 20
            "pbl": 45.0,                    # max 50
            "internship": 46.0              # max 50
        }
        res = calculate_academic_marks(inputs)

        self.assertTrue(res["is_valid"])
        self.assertEqual(len(res["errors"]), 0)
        self.assertEqual(res["midsem_avg"], 17.0)
        self.assertEqual(res["attendance_bonus"], 3)
        self.assertEqual(res["counted_activity"], 9.5)

        # Expected subtotal:
        # midsem_with_bonus (17.0 + 3 = 20.0) + act (9.5) + prac (26.0) + tw (18.0) + pbl (45.0) + intern (46.0) = 164.5
        expected_subtotal = 20.0 + 9.5 + 26.0 + 18.0 + 45.0 + 46.0
        self.assertEqual(res["provisional_subtotal"], expected_subtotal)

        # 420-mark incomplete scheme accounting:
        self.assertEqual(res["base_configured_max"], 180.0)
        self.assertEqual(res["total_target_marks"], 420.0)
        self.assertEqual(res["unallocated_marks"], 240.0)
        self.assertFalse(res["is_scheme_complete"])
        self.assertIsNotNone(res["scheme_warning"])
        self.assertIn("420", res["scheme_warning"])
        self.assertIn("240", res["scheme_warning"])

    def test_midsem_with_bonus_exceeds_max_warning(self):
        """When mid-sem avg + attendance bonus > max_marks, bonus is added and warning is issued without silent capping."""
        inputs = {
            "midsem_1": 20.0,
            "midsem_2": 20.0,
            "midsem_3": 19.0,
            "attendance_percentage": 90.0,  # bonus = 3
            "activity_1": 10.0,
            "activity_2": 8.0,
            "practical_external": 30.0,
            "term_work": 20.0,
            "pbl": 50.0,
            "internship": 50.0
        }
        res = calculate_academic_marks(inputs)
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["midsem_avg"], 20.0)
        self.assertEqual(res["attendance_bonus"], 3)
        self.assertEqual(res["midsem_with_bonus"], 23.0)
        self.assertIsNotNone(res["midsem_exceeded_warning"])
        self.assertIn("exceeds configured exam max", res["midsem_exceeded_warning"])

    def test_validation_errors_reported_when_exceeded(self):
        """Test that invalid values across practical, tw, pbl, internship are caught."""
        inputs = {
            "midsem_1": 25.0,  # Invalid (>20)
            "midsem_2": 15.0,
            "midsem_3": 15.0,
            "attendance_percentage": 105.0,  # Invalid (>100)
            "activity_1": 15.0,  # Invalid (>10)
            "activity_2": 5.0,
            "practical_external": 35.0,  # Invalid (>30)
            "term_work": 25.0,  # Invalid (>20)
            "pbl": 55.0,  # Invalid (>50)
            "internship": 60.0  # Invalid (>50)
        }
        res = calculate_academic_marks(inputs)
        self.assertFalse(res["is_valid"])
        self.assertGreaterEqual(len(res["errors"]), 5)


    def test_rules_service_defaults_and_validation(self):
        """Test academic rules retrieval, validation, and reset lifecycle."""
        # Ensure reset works
        rules = reset_academic_rules()
        self.assertEqual(rules["total_target_marks"], 420.0)
        self.assertEqual(rules["midsem"]["selection_policy"], "best_2_of_3")

        # Test valid modification
        rules_copy = dict(rules)
        rules_copy["midsem"]["selection_policy"] = "first_2"
        success, msg, saved = save_academic_rules(rules_copy)
        self.assertTrue(success)
        self.assertEqual(saved["midsem"]["selection_policy"], "first_2")

        loaded = get_academic_rules()
        self.assertEqual(loaded["midsem"]["selection_policy"], "first_2")

        # Test invalid policy rejection
        rules_copy["midsem"]["selection_policy"] = "invalid_policy"
        success2, msg2, _ = save_academic_rules(rules_copy)
        self.assertFalse(success2)

        # Reset back to default
        reset_rules = reset_academic_rules()
        self.assertEqual(reset_rules["midsem"]["selection_policy"], "best_2_of_3")


if __name__ == "__main__":
    unittest.main()
