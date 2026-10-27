# 🎓 ScholarIQ — Smart Student Semester Result Prediction System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.66-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-1.9-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Security-Argon2id%20%2B%20RBAC-00C853?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Tests-49%20Passed-3DDC97?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Latency-37.8ms%20Offline-blue?style=for-the-badge" />
</p>

<h3 align="center">
  A secure, industry-hardened academic decision-support platform combining deterministic institutional marking, supervised machine learning, longitudinal progress tracking, and verifiable database isolation.
</h3>

---

## 📌 Executive Product Architecture

**ScholarIQ** provides higher education institutions with an auditable, dual-engine academic evaluation and student success platform:

1. **🏛️ Output A: Official Academic Marking Engine (Deterministic)**
   - Strict implementation of institutional examination formulas and continuous assessment components.
   - Computes mid-semester examination averages, attendance bonus marks, and practical/project totals.
   - Handles the incomplete 420-mark college scheme without fabricating unallocated components.
   - Clearly marks the **provisional subtotal** (180 / 420 marks, 42.9% configured base).

2. **🤖 Output B: ML Semester Result Prediction (Supervised Machine Learning)**
   - Powered by an audited **Random Forest Regressor (v2.0 Clean)** trained with strict train/validation/test separation and cross-validation-only model selection.
   - Generates expected final semester score forecasts (scale 0–100%), heuristic uncertainty bounds ($\pm 5.0$), 10-point scale CGPA conversions, letter grades, and academic risk stratification.
   - Explicitly labelled as a **Research Baseline Benchmark** derived from Cortez & Silva (2008), with full feature provenance disclosed.

3. **🔮 What-If Academic Scenario Simulator**
   - Side-by-side comparative simulation allowing educators to model intervention outcomes (e.g. attendance recovery, remedial midsem tests) and evaluate shifts in both deterministic marks and ML projections.

4. **📈 Longitudinal Semester Progress Tracking**
   - Records successive prediction checkpoints (Milestone 1: Early Assessment, Milestone 2: Pre-Final) across the semester.
   - Tracks forecast evolution and measures residual error once official final examination scores are verified.

---

## 🔐 Security & Industry Hardening

The codebase has undergone comprehensive industry hardening to ensure institutional data protection and compliance:

- **Argon2id Password Hashing**: Passwords are encrypted using Argon2id (OWASP and NIST SP 800-63B recommended; memory cost: 64MB, time cost: 2 iterations, parallelism: 2) with automatic fallback to PBKDF2-HMAC-SHA256 (600,000 iterations).
- **Transparent Legacy Hash Migration**: Existing legacy SHA-256 password hashes are automatically verified and upgraded in-place to Argon2id upon the user's first successful sign-in.
- **Fail-Closed Authentication Gate**: All unauthenticated traffic is strictly blocked at the application boundary via an interactive login modal. The insecure default administrator session has been completely eliminated.
- **Role-Based Access Control (RBAC)**: Privileges (`admin`, `faculty`, `advisor`, `student`) are validated at both the UI navigation layer and the underlying service functions (`require_permission`). Arbitrary sidebar role switching has been permanently disabled.
- **Production Guardrails**: In production environments (`SCHOLARIQ_ENV=production`), automatic seeding of demo accounts and demo rosters is disabled.
- **Database Backup & Integrity Verification**: Point-in-time snapshot backup (`src/database/backup.py`) utilizing SQLite's online backup API, alongside automated `PRAGMA integrity_check` diagnostics.

---

## 🏛️ Official College Marking Rules

The deterministic calculation engine adheres strictly to configured institutional academic rules:

### A. Mid-Semester Examinations
- **Assessments**: 3 mid-semester exams, each carrying a maximum of **20 marks**.
- **Selection Policy**: Configurable (`best_2_of_3`, `first_2`, or `all_3`). Default: **Best 2 of 3**.
- **Formula**:
  $$\text{Mid-Semester Average} = \frac{\text{Test}_A + \text{Test}_B}{2.0}$$

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

| Attendance % | Bonus Awarded | Test Status |
| :---: | :---: | :---: |
| $\le 60.0\%$ | **0 marks** | Verified |
| $60.1\% - 75.0\%$ | **1 mark** | Verified |
| $75.1\% - 85.0\%$ | **2 marks** | Verified |
| $> 85.0\%$ | **3 marks** | Verified |

- **Component Exceed Warning**: If $\text{Mid-Sem Avg} + \text{Bonus} > 20$, the score is preserved without arbitrary reduction, and an informative warning is displayed.

### D. Practical and Project Work
- **Practical External**: Maximum 30 marks
- **Term Work (TW)**: Maximum 20 marks
- **Project-Based Learning (PBL)**: Maximum 50 marks
- **Internship Assessment**: Maximum 50 marks

### E. Incomplete 420-Mark Scheme Handling
- **Configured Components Sum**: $20 + 10 + 30 + 20 + 50 + 50 = \mathbf{180 \text{ marks}}$ ($42.9\%$ of 420).
- **Unallocated Balance**: $\mathbf{240 \text{ marks}}$ (e.g. End-Semester Terminal Theory Examination).
- The system flags: *"Marking scheme incomplete — official total and final grade are unavailable."* A **Provisional Subtotal** is presented for internal tracking.

---

## 🔬 Machine Learning Evaluation & Provenance

### Rigorous Evaluation Protocol (No Model Selection Bias)
1. **Train/Test Separation**: The dataset is split into an 80% Training set and a 20% strictly held-out untouched Test set.
2. **Cross-Validation Model Selection**: Candidate algorithms (Dummy Mean, Ridge Regression, Random Forest, HistGradientBoosting) were benchmarked strictly using 5-Fold Cross-Validation on the training split. **The held-out test set was never inspected during model selection.**
3. **Single Held-Out Test Evaluation**: The selected primary model was evaluated exactly once on the held-out test split.

| Model | 5-Fold CV MAE (Train) | 5-Fold CV RMSE (Train) | 5-Fold CV $R^2$ (Train) | Test MAE (Held-Out) | Test $R^2$ (Held-Out) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Dummy Mean)** | $17.01 \pm 3.15$ | $22.82 \pm 3.17$ | $-0.010 \pm 0.04$ | 18.23 pts | $-0.010$ |
| **Ridge Regression** | $9.91 \pm 1.25$ | $13.57 \pm 1.50$ | $0.652 \pm 0.08$ | 10.33 pts | $0.652$ |
| **Random Forest (v2.0 Clean)** | **$8.10 \pm 1.01$** | **$11.73 \pm 1.71$** | **$0.714 \pm 0.089$** | **8.31 pts** | **0.722** |
| **HistGradientBoosting** | $8.81 \pm 0.87$ | $12.54 \pm 1.32$ | $0.686 \pm 0.07$ | 9.08 pts | $0.686$ |

### Model Provenance Notice
Model `v2.0 Clean` is trained on the Cortez & Silva (2008) secondary school Portuguese language dataset. Features such as `Internal_Marks` ($G1/2$) and `Attendance_Percentage` ($100 - \text{absences}$) represent domain-mapped research proxies.
- **Genuine College ML Contract**: Full institutional deployment utilizes the `CollegeAcademicRecord` data contract (`src/schemas/college_contract.py`), capturing the 3 Mid-Semesters, Activities, External Practical (30), Term Work (20), PBL (50), and Internship (50).
- **Causal Disclaimer**: Statistical correlation does not establish causal necessity. Recommended interventions should be evaluated in context by academic advisors.

---

## 🧭 Application Sections (10 Institutional Workspaces)

1. **📊 Overview**: Cohort executive KPI summary, active term status, marking scheme completeness, and early intervention alerts.
2. **🎯 Student Result Prediction**: Primary interactive workspace with form inputs, live deterministic calculations (Output A), and ML forecast (Output B).
3. **🔮 What-If Simulator**: Side-by-side scenario modeling comparing baseline vs adjusted performance.
4. **👥 Student Records & Progress**: Searchable student directory, profile editing, duplicate ID prevention, and progressive checkpoint tracking.
5. **📈 Academic Analytics**: Cohort grade distributions, risk stratification, and pass rate forecasting.
6. **📜 Prediction History**: Chronological log of predictions, input snapshots, model versions, and CSV export.
7. **🔬 Dataset & Model Info**: Dataset provenance, leakage remediation audit, Gini feature importance, and benchmarks.
8. **⚙️ Academic Rules Configuration**: Administrative interface to configure selection policies, component caps, attendance thresholds, and save/reset.
9. **📥 Import / Export**: Batch CSV intake with column/type validation, downloadable template, and database export.
10. **🛠️ Settings & Diagnostics**: Database connection diagnostics (SQLite WAL mode), PRAGMA integrity checks, on-demand point-in-time database snapshots (.db), and latency benchmarks.

---

## 🚀 Installation & Quick Start

### 1. Setup Environment
```powershell
# Clone or navigate to the repository
cd c:\Users\offic\Downloads\AI_Student_performance_Analytics-main

# Install dependencies (includes argon2-cffi, pydantic, sqlalchemy)
pip install -r requirements.txt
```

### 2. Run the Hermetic Test Suite (49 Automated Tests)
```powershell
python run_tests.py
```
*Expected Output:*
```text
Ran 49 tests in 4.937s
OK
```

### 3. Launch the Application
```powershell
streamlit run AI_Student_performance_Analytics-main/Student-Performance-Prediction-System-main/Streamlit_App/app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser and sign in using your institutional account or development credentials (`admin` / `Admin@123`).
