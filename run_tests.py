"""ScholarIQ Unified Test Runner.
Executes all unit and integration tests across data validation, model service, recommendations, and reporting.
"""
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ROOT_DIR / "AI_Student_performance_Analytics-main" / "Student-Performance-Prediction-System-main"
TESTS_DIR = PROJECT_DIR / "tests"

# Ensure src is on Python search path
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

if __name__ == "__main__":
    print(f"Discovering and running tests in: {TESTS_DIR}")
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(TESTS_DIR), top_level_dir=str(PROJECT_DIR))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
