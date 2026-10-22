"""ScholarIQ Root Streamlit Entry Point.
Delegates execution to the production application located in the inner project directory.
Enables seamless execution from the workspace root via: `streamlit run app.py`.
"""
import sys
import runpy
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ROOT_DIR / "AI_Student_performance_Analytics-main" / "Student-Performance-Prediction-System-main"
APP_PATH = PROJECT_DIR / "Streamlit_App" / "app.py"

# Add project root and inner project root to sys.path
for p in [str(ROOT_DIR), str(PROJECT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

if __name__ == "__main__":
    if not APP_PATH.exists():
        raise FileNotFoundError(f"Target Streamlit app not found at {APP_PATH}")
    # Execute the target Streamlit script
    runpy.run_path(str(APP_PATH), run_name="__main__")
