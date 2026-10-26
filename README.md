# 🎓 ScholarIQ — Smart Student Semester Result Prediction System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.66-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-1.9-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/SQLAlchemy-2.0-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Tests-39%20Passed-3DDC97?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Latency-37.8ms%20Offline-blue?style=for-the-badge" />
</p>

<h3 align="center">
  A full-stack, leak-free predictive academic intelligence platform combining deterministic official institutional marking with supervised machine learning and what-if simulation.
</h3>

---

## 📌 Executive Product Overview

**ScholarIQ** is an institutional academic decision-support platform designed for faculty, academic advisors, and institutional leadership. It delivers two strictly separated, auditable evaluation engines:

1. **🏛️ Output A: Official Academic Marking Engine (Deterministic)**
   - Strict execution of configured college examination formulas.
   - Computes mid-semester examination averages, attendance bonus marks, and practical/project totals.
   - Handles the incomplete 420-mark college scheme without fabricating unallocated components.
   - Clearly flags the provisional subtotal (180 / 420 marks, 42.9% configured base).

2. **🤖 Output B: ML Semester Result Prediction (Supervised Machine Learning)**
   - Powered by an audited, leak-free **Random Forest Regressor (v2.0 Clean)** trained on verified pre-examination features ($G1/2$ continuous assessment).
   - Generates expected final semester scores (scale 0–100%), confidence intervals ($\pm$ margin), 10-point scale CGPA estimations, letter grades, and academic risk stratification.
   - Provides local factor attribution (waterfall/bar breakdown of what features drove the forecast).

3. **🔮 What-If Academic Scenario Simulator**
   - Enables faculty to test alternative scenarios (e.g. improved attendance or remedial midterm test scores) and observe side-by-side shifts in both calculated marks and ML predictions.

4. **📈 Longitudinal Semester Progress Tracking**
   - Records successive prediction checkpoints (Milestone 1, Milestone 2, Milestone 3) across the semester.
   - Tracks forecast evolution and measures prediction error against verified terminal outcomes once official exams are graded.

---

## 🏛️ Official College Marking Rules

The deterministic calculation engine adheres strictly to user-specified academic rules:

### A. Mid-Semester Examinations
- **Assessments**: 3 mid-semester exams, each carrying a maximum of **20 marks**.
- **Selection Policy**: Configurable (`best_2_of_3`, `first_2`, or `all_3`). Default: **Best 2 of 3**.
- **Formula**:
  $$\text{Mid-Semester Average} = \frac{\text{Test}_A + \text{Test}_B}{2.0}$$
  *(The sum of two exams is divided by 2, never accidentally divided by 3).*

### B. Academic Activities
- **Assessments**: Activity 1 (max 10 marks), Activity 2 (max 10 marks).
- **Rule**: Only the higher of the two scores counts:
  $$\text{Counted Activity Marks} = \max(\text{Activity}_1, \text{Activity}_2)$$

### C. Attendance Bonus Logic
- **Rules**: Non-cumulative bonus marks added directly to the mid-semester component.
- **Strict Boundary Logic Verified via Automated Tests**:
  - $\text{Attendance} \le 60\% \implies 0$ bonus marks
  - $\text{Attendance} > 60\% \text{ and } \le 75\% \implies 1$ bonus mark
  - $\text{Attendance} > 75\% \text{ and } \le 85\% \implies 2$ bonus marks
  - $\text{Attendance} > 85\% \implies 3$ bonus marks

| Tested Attendance % | Exact Bonus Awarded | Test Status |
| :---: | :---: | :---: |
| `59.0%` | **0 marks** | Passed |
| `60.0%` (Exact boundary) | **0 marks** | Passed |
| `61.0%` | **1 mark** | Passed |
| `75.0%` (Exact boundary) | **1 mark** | Passed |
| `76.0%` | **2 marks** | Passed |
| `85.0%` (Exact boundary) | **2 marks** | Passed |
| `86.0%` | **3 marks** | Passed |
| `100.0%` | **3 marks** | Passed |

- **Component Exceed Warning**: If $\text{Mid-Sem Avg} + \text{Bonus} > 20$, the score is preserved without arbitrary reduction, and an informative warning is displayed.

### D. Practical and Project Work
- **Practical External**: Maximum 30 marks
- **Term Work (TW)**: Maximum 20 marks
- **Project-Based Learning (PBL)**: Maximum 50 marks
- **Internship Assessment**: Maximum 50 marks

### E. Incomplete 420-Mark Scheme Handling
- **Configured Components Sum**: $20 + 10 + 30 + 20 + 50 + 50 = \mathbf{180 \text{ marks}}$ ($42.9\%$ of 420).
- **Unallocated Balance**: $\mathbf{240 \text{ marks}}$ (e.g. End-Semester Terminal Theory Examination).
- **Integrity Rule**: The application does **not** invent the missing 240 marks. It displays:
  > *"Marking scheme incomplete — official total and final grade are unavailable."*
- A **Provisional Subtotal** is presented for internal progress only.

---

## 🔬 Target Leakage Remediation & ML Integrity

In legacy pipelines, $G3$ (the final exam grade) was used to synthesize both `Internal_Marks` ($G3 / 2$) and `Final_Score` ($G3 \times 5$), resulting in an artificial **99.07% target leakage**.

**In Model v2.0 Clean (Active Production Model):**
$$\text{Internal\_Marks} = \frac{G1}{2.0}$$
`Internal_Marks` is derived from $G1$ — the **actual first-period midterm examination** administered months prior to the final exam. Feature importance is balanced:
- Interim Exams ($G1/2$): 67.2%
- Attendance Rate: 19.3%
- Tutoring / Extra Classes: 4.0%
- Student Age: 3.1%
- Coursework / Assignments: 2.3%
- Prior Academic Backlogs: 2.1%
- Daily Self-Study Hours: 1.4%

### Verified Model Performance (5-Fold Cross-Validation on $N=395$):
| Model | 5-Fold CV MAE | 5-Fold CV RMSE | Test MAE | Test $R^2$ |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (Dummy Mean)** | $17.01 \pm 3.15$ | $22.82 \pm 3.17$ | 18.23 pts | $-0.010$ |
| **Ridge Regression** | $9.91 \pm 1.25$ | $13.57 \pm 1.50$ | 10.33 pts | $0.652$ |
| **Random Forest (v2.0 Clean)** | **$8.10 \pm 1.01$** | **$11.73 \pm 1.71$** | **8.31 pts** | **0.722** |
| **HistGradientBoosting** | $8.81 \pm 0.87$ | $12.54 \pm 1.32$ | 9.08 pts | $0.686$ |

---

## ⚡ Execution Latency Benchmark

Single-student prediction latency was measured over 100 consecutive invocations on the local host environment:
- **Mean Inference Latency:** `37.81 ms`
- **Median Latency:** `36.81 ms`
- **95th Percentile (P95):** `42.75 ms`
- **Fastest Response:** `32.13 ms`
- **100% Offline Capability:** Runs completely locally with zero external network dependencies.

---

## 🧭 Application Sections (10 Institutional Workspaces)

1. **📊 Overview**: Cohort executive KPI summary, active term status, marking scheme completeness, and early intervention alerts.
2. **🎯 Student Result Prediction**: Primary interactive workspace with form inputs, live deterministic calculations (Output A), and ML forecast (Output B).
3. **🔮 What-If Simulator**: Side-by-side scenario modeling comparing baseline vs adjusted performance.
4. **👥 Student Records & Progress**: Student directory, profile editing, duplicate ID prevention, and progressive checkpoint tracking.
5. **📈 Academic Analytics**: Cohort grade distributions, risk stratification, and pass rate forecasting.
6. **📜 Prediction History**: Chronological log of predictions, input snapshots, model versions, and CSV export.
7. **🔬 Dataset & Model Info**: Dataset provenance, leakage remediation audit, Gini feature importance, and benchmarks.
8. **⚙️ Academic Rules Configuration**: Administrative interface to configure selection policies, component caps, attendance thresholds, and save/reset.
9. **📥 Import / Export**: Batch CSV intake with column/type validation, downloadable template, and database export.
10. **🛠️ Settings & Diagnostics**: Database connection diagnostics (SQLite WAL mode), table record counts, on-demand latency benchmarking, and database backup download (.db).

---

## 🚀 Installation & Quick Start

### 1. Setup Environment
```powershell
# Clone or navigate to the repository
cd c:\Users\offic\Downloads\AI_Student_performance_Analytics-main

# Activate virtual environment
.\AI_Student_performance_Analytics-main\Student-Performance-Prediction-System-main\.venv\Scripts\Activate.ps1
```

### 2. Run the Automated Test Suite (39 Tests)
```powershell
python run_tests.py
```
*Expected Output:*
```text
Ran 39 tests in 4.155s
OK
```

### 3. Launch the Application
```powershell
streamlit run AI_Student_performance_Analytics-main/Student-Performance-Prediction-System-main/Streamlit_App/app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 📄 License & Attribution

Developed as an academic analytics platform. Baseline training dataset adapted from the Portuguese Student Performance study (Cortez and Silva, 2008, University of Minho).
