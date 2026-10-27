# ScholarIQ 3.0 — Industry-Readiness & Security Audit

**Date:** October 27, 2026  
**Auditor:** Principal Software Architect & QA Lead  
**Repository:** [akashdubey8462/AI-Student-Performance-Analytics](https://github.com/akashdubey8462/AI-Student-Performance-Analytics)  
**Target Architecture:** Python 3.10–3.13, Streamlit, SQLAlchemy, SQLite/PostgreSQL, scikit-learn  

---

## 1. Executive Summary

A comprehensive architectural, security, and machine learning audit was conducted across the ScholarIQ repository. While the application possesses a functional deterministic marking engine, responsive Streamlit user interface, and relational entity models, critical security, test isolation, and ML evaluation vulnerabilities were identified that prevent production readiness.

This audit details all findings across severity levels, pinpoints exact code locations, demonstrates reproducibility, and defines the remediation implementation plan.

---

## 2. Audit Findings Matrix

| ID | Category | Severity | File & Location | Issue Description | Proposed Remediation |
|---|---|---|---|---|---|
| **SEC-01** | Security | **CRITICAL** | `Streamlit_App/app.py:101-109` | Default hard-coded session as `admin` (`Dr. Elena Vance`) grants unrestricted administrative access to any unauthenticated visitor. | Remove auto-admin session; enforce authentication guard and fail closed with login form. |
| **SEC-02** | Security | **HIGH** | `Streamlit_App/app.py:155-181` | Unrestricted role switcher in sidebar allows any client to switch to `admin`, `faculty`, or `advisor` without credentials. | Restrict role switching to authorized multi-role users; disable unauthenticated role spoofing in production. |
| **SEC-03** | Security | **HIGH** | `src/services/auth_service.py:17-21` | Static salt (`"scholariq_secure_salt_2026"`) with unsalted/un-iterated SHA-256 for password hashing. Vulnerable to rainbow tables and brute force. | Migrate to Argon2id (via `argon2-cffi`) with cryptographic per-user salt; provide automated migration path for existing hashes. |
| **SEC-04** | Security | **HIGH** | `src/services/student_service.py`, `src/services/prediction_service.py` | Service functions lack server-side authorization checks, relying solely on UI-level visibility. | Enforce session/identity and role authorization at the service layer. |
| **SEC-05** | Security | **MEDIUM** | `Streamlit_App/app.py:90-95`, `src/database/seed_data.py` | Automatic seeding of demo credentials (`Admin@123`, etc.) and demo records on startup in production. | Gate automatic seeding behind environment flag (`SCHOLARIQ_ENV != "production"` or explicit CLI command). |
| **CI-01** | CI / Tests | **CRITICAL** | `tests/test_database_and_services.py:18-20`, `.github/workflows/ci.yml` | Tests execute against shared persistent DB without test database isolation or explicit fixture seeding; fresh CI runners fail with Foreign Key and NoneType errors. | Implement isolated test DB (`sqlite:///:memory:` or temp DB) with explicit model/user/course fixture initialization in `setUpClass`. |
| **ML-01** | ML Pipeline | **HIGH** | `training/train.py:86-104` | Model selection bias: multiple candidate models evaluated on test set, and best estimator chosen via `test_r2` rather than cross-validation. | Separate Train/Val/Test; select model strictly on training data cross-validation; evaluate single selected model once on untouched test set. |
| **ML-02** | ML Validity | **HIGH** | `training/train.py:28-51`, `Streamlit_App/app.py` | UCI secondary school Portuguese dataset proxy features presented as validated college semester result predictor. | Clearly label UCI model as "Research Baseline Benchmark (Secondary School G1→G3)"; do not present as validated college model. |
| **ML-03** | ML Contract | **MEDIUM** | `src/model_service.py`, `Streamlit_App/app.py` | Lack of formal schema and timestamp contract for genuine college assessments (3 midsems, practical, TW, PBL, internship). | Implement formal versioned Pydantic data contract (`CollegeAcademicRecordContract`) with explicit missingness tracking. |
| **DB-01** | Database | **MEDIUM** | `src/database/models.py:114-135`, `src/services/prediction_service.py:53-56` | Prediction record lacks direct course/term linkage, and defaults to `student.enrollments[0]` rather than subject-specific enrollment. | Add `course_id` and `term_id` to `Prediction`; link predictions explicitly to target subject/enrollment. |
| **DB-02** | Database | **LOW** | `src/database/connection.py` | No Alembic migration tracking or automated backup/restore utilities for SQLite/PostgreSQL. | Provide Alembic migration structure and database backup/restore utility. |
| **UX-01** | UI / Ethics | **MEDIUM** | `src/database/seed_data.py:171-190`, `Streamlit_App/app.py` | Formula-generated demo scores (`interim * 7.5 + att * 0.18...`) saved as model predictions; uncertainty intervals heuristic without calibration disclaimers. | Label demo data explicitly; honestly disclose uncertainty intervals as empirical/heuristic; state correlation != causation. |

---

## 3. Deep Dive into Critical Vulnerabilities

### 3.1 SEC-01 & SEC-02: Authentication & Authorization Bypass
**Evidence in Code (`Streamlit_App/app.py`):**
```python
if "current_user" not in st.session_state:
    st.session_state["current_user"] = {
        "id": 1,
        "username": "admin",
        "full_name": "Dr. Elena Vance (Dean of Academics)",
        "role": "admin",
        "email": "dean.vance@scholariq.edu"
    }
```
Any user visiting the web URL immediately inherits administrative privileges, including access to student academic records, intervention history, and academic rules configuration. Furthermore, the identity switcher allows switching between roles without authentication.

**Remediation:**
- Initialize `st.session_state["current_user"] = None`.
- If `current_user` is None, render an authentication form requiring valid credentials against the database.
- Provide a "Sign In" / "Sign Out" flow.
- Enforce role-based access checks at the service level.

---

### 3.2 SEC-03: Weak Password Hashing with Static Salt
**Evidence in Code (`src/services/auth_service.py`):**
```python
def hash_password(password: str) -> str:
    salt = "scholariq_secure_salt_2026"
    return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()
```
Single-iteration SHA-256 with a static hard-coded salt can be calculated at hundreds of millions of hashes per second on consumer GPUs.

**Remediation:**
- Implement Argon2id password hashing using `argon2-cffi` (recommended by OWASP and NIST SP 800-63B).
- Provide a transparent migration path: upon login, if a user's stored hash matches the legacy SHA-256 pattern, verify it, immediately re-hash the cleartext password using Argon2id, and update the database record.

---

### 3.3 CI-01: CI Suite Failure & Lack of Test Isolation
**Evidence in Code (`tests/test_database_and_services.py`):**
```python
class TestDatabaseAndServices(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()  # <-- Does not seed users, courses, or model versions!
```
On a clean CI runner (Ubuntu, Python 3.10/3.11/3.12), `scholariq.db` does not exist. Calling `init_db()` creates empty tables.
When `test_01_user_authentication` tests `authenticate_user("admin", "Admin@123")`, the `users` table is empty, returning `None` and triggering `AssertionError: unexpectedly None`.
Subsequent tests fail with `ForeignKeyError` and `AttributeError: 'NoneType' object has no attribute 'id'` because `ModelVersion` and `Course` records were never initialized.

**Remediation:**
- In all test classes, use an isolated database (`sqlite:///:memory:` or isolated temporary test database).
- Provide explicit test setup that seeds required fixture records (test admin user, test model version, test course/term).
- Guarantee zero residue and zero dependence on the developer's local `scholariq.db`.

---

### 3.4 ML-01 & ML-02: Model Selection Bias & Research Dataset Provenance
**Evidence in Code (`training/train.py`):**
```python
for name, model in candidate_models.items():
    ...
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    test_r2 = r2_score(y_test, preds)
    benchmarks.append({"test_r2": test_r2, "instance": model})

best = sorted(benchmarks, key=lambda x: x["test_r2"], reverse=True)[0]
```
Evaluating multiple models on the test set and selecting the best using test-set metrics creates optimistic selection bias. The test set is no longer an unbiased held-out estimate.

Furthermore, the dataset used (`student_data.csv`) is the 2008 UCI Portuguese secondary school mathematics/language dataset. Transforming `G1` to `Internal_Marks` and synthetic questionnaire combinations into `Assignments_Submitted` does not constitute a validated college semester result prediction model.

**Remediation:**
1. Split data into Train (80%) and Test (20%).
2. Perform model selection and hyperparameter optimization strictly using 5-Fold Cross-Validation on the Train split.
3. Select the best estimator based on CV performance (`cv_r2` / `cv_mae`).
4. Perform exactly ONE final evaluation on the held-out Test set.
5. Formally document and label the model as "Research Baseline Model (UCI Secondary School Benchmark)".
6. Create an explicit data contract for genuine college academic performance records.

---

## 4. Remediation Roadmap & Execution Milestones

- **Milestone A:** Audit & CI failure reproduction (Completed).
- **Milestone B:** Security remediation (Argon2id, migration path, authentication gate, RBAC enforcement).
- **Milestone C:** ML evaluation correction (train/CV/test separation, unbiased selection, provenance disclosure).
- **Milestone D:** College-specific ML data contract & validation schema.
- **Milestone E:** Database isolation in tests, schema enhancements, backup/restore utility.
- **Milestone F:** Calibrated uncertainty & explainability disclaimers.
- **Milestone G:** UI polish, CI workflow hardening, and complete test verification.
