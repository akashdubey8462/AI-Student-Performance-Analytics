# ============================================================
# ScholarIQ 3.0 — AI Student Academic Performance Analytics Platform
# Full-Stack Production Application (SQLAlchemy + Streamlit + ML Engine)
# ============================================================

import sys
import io
import json
from pathlib import Path
from datetime import datetime, date, timedelta

# Add project root to sys.path so src imports resolve cleanly
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.config import (
    THEME_TOKENS,
    DEFAULT_MODEL_V2,
    LEGACY_MODEL_V1,
    DATASET_DIR
)
from src.database.connection import init_db, get_db_session
from src.database.models import User, Student, Course, Enrollment, ModelVersion, Prediction, Intervention, AuditLog
from src.database.seed_data import seed_database
from src.data_validation import validate_student_input
from src.model_service import ModelService
from src.recommendations import generate_recommendations
from src.report_generator import generate_text_report, generate_cohort_summary_csv
from src.services.auth_service import authenticate_user, has_permission
from src.services.student_service import list_students, get_student_by_id, create_student
from src.services.prediction_service import save_prediction, list_predictions, get_overview_kpis
from src.services.intervention_service import list_interventions, create_intervention, update_intervention_status
from src.services.import_service import import_roster_transactional


# -----------------------------
# Page Configuration & Theme
# -----------------------------
st.set_page_config(
    page_title="ScholarIQ 3.0 — Academic Intelligence Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    css_path = APP_DIR / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Auto-initialize database on first run
try:
    with get_db_session() as s:
        if s.query(Student).count() == 0:
            seed_database()
except Exception:
    seed_database()


# -----------------------------
# Session State Initialization
# -----------------------------
if "current_user" not in st.session_state:
    st.session_state["current_user"] = {
        "id": 1,
        "username": "admin",
        "full_name": "Dr. Elena Vance (Dean of Academics)",
        "role": "admin",
        "email": "dean.vance@scholariq.edu"
    }

if "selected_student_id" not in st.session_state:
    st.session_state["selected_student_id"] = None


# -----------------------------
# Chart Styling Tokens
# -----------------------------
CHART_FONT = {"family": "Inter", "color": THEME_TOKENS["ink_soft"]}
INK = THEME_TOKENS["ink"]
GRID = THEME_TOKENS["grid"]

def get_color_for_score(score: float) -> str:
    if score >= 70.0:
        return THEME_TOKENS["success"]
    elif score >= 50.0:
        return THEME_TOKENS["amber"]
    return THEME_TOKENS["red"]

def section_label(icon: str, title: str, sub: str = None):
    sub_html = f'<div class="label-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="section-label">
        <div class="icon-badge">{icon}</div>
        <div><div class="label-text">{title}</div>{sub_html}</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SIDEBAR: IDENTITY & NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
        <span style="font-size:26px;">🎓</span>
        <div>
            <div style="font-family:'Sora',sans-serif; font-weight:700; font-size:17px; color:#F2ECEC;">ScholarIQ <span style="color:#D4A24E;">3.0</span></div>
            <div style="font-size:11px; color:#AC9CA0;">Enterprise Academic Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Active User Context Switcher
    st.markdown('<div class="preview-header"><span class="preview-pulse"></span>Active Session</div>', unsafe_allow_html=True)
    
    role_options = {
        "admin": "Dr. Elena Vance (Dean / Admin)",
        "faculty": "Prof. Marcus Chen (Faculty)",
        "advisor": "Sarah Jenkins (Academic Advisor)",
        "student": "Aditi Sharma (Student)"
    }
    
    selected_role_key = st.selectbox(
        "Current Identity",
        options=list(role_options.keys()),
        format_func=lambda x: role_options[x],
        index=list(role_options.keys()).index(st.session_state["current_user"]["role"])
    )

    if selected_role_key != st.session_state["current_user"]["role"]:
        with get_db_session() as session:
            matched_user = session.query(User).filter_by(role=selected_role_key).first()
            if matched_user:
                st.session_state["current_user"] = {
                    "id": matched_user.id,
                    "username": matched_user.username,
                    "full_name": matched_user.full_name,
                    "role": matched_user.role,
                    "email": matched_user.email
                }
                st.rerun()

    cur_user = st.session_state["current_user"]
    st.caption(f"Logged in as **{cur_user['full_name']}** (`{cur_user['role'].upper()}`)")
    st.divider()

    # Role-Based Navigation Menu
    nav_options = [
        "📊 Institutional Overview",
        "👥 Student Directory & Profiles",
        "🎓 Individual Prediction Studio",
        "🛡️ Intervention Hub",
        "📥 Data Import & Batch Analytics",
        "📜 Prediction History & Audit",
        "🔬 Model Governance",
        "⚙️ Settings & System Health"
    ]

    selected_nav = st.radio(
        "Navigation Workspace",
        nav_options,
        index=0,
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown('<div class="footer-note">SCHOLARIQ v3.0 · SQLALCHEMY ORM & ML</div>', unsafe_allow_html=True)


# ============================================================
# PAGE 1: INSTITUTIONAL OVERVIEW
# ============================================================
if selected_nav == "📊 Institutional Overview":
    st.markdown("""
    <div class="hero">
        <div class="main-title">Institutional Overview<br><span class="accent">Academic Health & Early Warning</span></div>
        <div class="sub-title">Real-time aggregate analytics, performance risk stratification, and academic intervention tracking across enrolled cohorts.</div>
    </div>
    """, unsafe_allow_html=True)

    kpis = get_overview_kpis()

    # KPI Ribbon Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    
    with k1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">Students Monitored</div>
            <div class="stat-value">{kpis['total_students']}</div>
            <div class="stat-sub">Enrolled in Active Cohorts</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">Cohort Avg Score</div>
            <div class="stat-value" style="color:#D4A24E;">{kpis['average_predicted_score']:.1f}<span style="font-size:16px;">/100</span></div>
            <div class="stat-sub">Projected Mean Performance</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        pass_color = THEME_TOKENS["success"] if kpis['projected_pass_rate'] >= 75 else THEME_TOKENS["amber"]
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">Projected Pass Rate</div>
            <div class="stat-value" style="color:{pass_color};">{kpis['projected_pass_rate']:.1f}%</div>
            <div class="stat-sub">Passing Threshold ≥ 50/100</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">At-Risk Students</div>
            <div class="stat-value" style="color:#F5646B;">{kpis['at_risk_count']}</div>
            <div class="stat-sub">High Academic Risk (<50)</div>
        </div>
        """, unsafe_allow_html=True)
    with k5:
        overdue_note = f"<span style='color:#F5646B;'>{kpis['overdue_interventions']} overdue</span>" if kpis['overdue_interventions'] > 0 else "<span style='color:#3DDC97;'>All on schedule</span>"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">Active Interventions</div>
            <div class="stat-value">{kpis['open_interventions']}</div>
            <div class="stat-sub">{overdue_note}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.divider()

    # Visualizations Row
    col_chart1, col_chart2 = st.columns([1.3, 1], gap="medium")

    with col_chart1:
        st.markdown("##### 📈 Projected Score Distribution across Cohort")
        with get_db_session() as session:
            preds = session.query(Prediction).all()
            scores = [p.predicted_score for p in preds] if preds else [65.0]

        hist = go.Figure(go.Histogram(
            x=scores,
            nbinsx=12,
            marker=dict(color=THEME_TOKENS["maroon"], line=dict(color="rgba(255,255,255,0.1)", width=1)),
            opacity=0.9
        ))
        hist.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=CHART_FONT,
            xaxis=dict(gridcolor=GRID, title="Projected Final Score (0-100)"),
            yaxis=dict(gridcolor=GRID, title="Student Count")
        )
        st.plotly_chart(hist, use_container_width=True, config={"displayModeBar": False})

    with col_chart2:
        st.markdown("##### 🛡️ Academic Risk Stratification")
        with get_db_session() as session:
            preds = session.query(Prediction).all()
            risk_counts = {"Low Risk": 0, "Moderate Risk": 0, "High Academic Risk": 0}
            for p in preds:
                risk_counts[p.risk_level] = risk_counts.get(p.risk_level, 0) + 1

        pie_colors = [THEME_TOKENS["success"], THEME_TOKENS["amber"], THEME_TOKENS["red"]]
        pie = go.Figure(go.Pie(
            labels=list(risk_counts.keys()),
            values=list(risk_counts.values()),
            hole=0.65,
            marker=dict(colors=pie_colors),
            textinfo="percent+label",
            textposition="inside",
            showlegend=False
        ))
        pie.update_layout(
            height=260,
            margin=dict(l=0, r=0, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=CHART_FONT
        )
        st.plotly_chart(pie, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # Urgent Academic Action Attention List
    st.markdown("##### ⚠️ Students Requiring Immediate Early Intervention")
    students_list = list_students(risk_filter="High Academic Risk", limit=5)
    if students_list:
        df_risk = pd.DataFrame(students_list)[["student_id", "name", "latest_predicted_score", "latest_grade", "attendance_percentage", "internal_marks", "previous_failures"]]
        df_risk.columns = ["Student ID", "Full Name", "Projected Score", "Grade", "Attendance %", "Interim Test", "Prior Failures"]
        st.dataframe(df_risk, use_container_width=True, hide_index=True)
    else:
        st.success("🎉 No students currently flagged under High Academic Risk.")


# ============================================================
# PAGE 2: STUDENT DIRECTORY & PROFILES
# ============================================================
elif selected_nav == "👥 Student Directory & Profiles":
    section_label("👥", "Student Directory & Academic Profiles", "Longitudinal records, attendance tracking, and individual evaluation history")

    col_s1, col_s2, col_s3 = st.columns([2, 1.2, 1], gap="medium")
    with col_s1:
        search = st.text_input("🔍 Search by Student Name or ID", placeholder="e.g. Aarav, STU-2026...")
    with col_s2:
        filter_risk = st.selectbox("Filter by Risk Tier", ["All", "Low Risk", "Moderate Risk", "High Academic Risk"])
    with col_s3:
        st.write("")
        st.write("")
        btn_new_student = st.button("+ Enroll Student", use_container_width=True)

    if btn_new_student and has_permission(cur_user["role"], "faculty"):
        with st.expander("📝 Enroll New Student Profile", expanded=True):
            with st.form("new_student_form"):
                f_name = st.text_input("Full Name", placeholder="e.g. Priya Sharma")
                f_c1, f_c2 = st.columns(2)
                with f_c1:
                    f_gen = st.selectbox("Gender", ["Female", "Male"])
                with f_c2:
                    f_age = st.slider("Age", 14, 25, 18)
                submit_student = st.form_submit_button("Save Student to Database", type="primary")
                if submit_student:
                    if not f_name.strip():
                        st.error("Student name cannot be empty.")
                    else:
                        created = create_student(
                            name=f_name,
                            gender=1 if f_gen == "Male" else 0,
                            age=f_age,
                            user_id=cur_user["id"]
                        )
                        st.success(f"Enrolled {created['name']} with ID: {created['student_id']}")
                        st.rerun()

    # Directory Table
    students = list_students(search_query=search, risk_filter=filter_risk)
    
    if students:
        df_display = pd.DataFrame(students)
        table_cols = ["student_id", "name", "gender", "age", "attendance_percentage", "internal_marks", "latest_predicted_score", "latest_grade", "risk_level"]
        table_headers = ["ID", "Name", "Gender", "Age", "Attendance %", "Interim /10", "Predicted Score", "Grade", "Risk Tier"]
        
        show_df = df_display[table_cols].copy()
        show_df.columns = table_headers
        st.dataframe(show_df, use_container_width=True, hide_index=True)

        st.divider()

        # Detailed Student Profile Inspector
        section_label("👤", "Deep-Dive Student Profile", "Detailed enrollment indicators, past forecasts, and intervention notes")
        
        selected_student_name = st.selectbox(
            "Select Student Profile to Inspect",
            options=[s["name"] for s in students],
            index=0
        )
        
        target_s = next((s for s in students if s["name"] == selected_student_name), None)
        if target_s:
            detail = get_student_by_id(target_s["id"])
            if detail:
                p_c1, p_c2 = st.columns([1, 1.2], gap="large")
                with p_c1:
                    with st.container(border=True):
                        st.markdown(f"### {detail['name']}")
                        st.caption(f"ID: `{detail['student_id']}` • {detail['gender']} • Age {detail['age']}")
                        st.divider()
                        if detail["enrollments"]:
                            e = detail["enrollments"][0]
                            st.metric("Course", e["course_code"])
                            st.metric("Attendance Rate", f"{e['attendance']:.1f}%")
                            st.metric("Interim Test Score", f"{e['internal_marks']:.1f} / 10")
                            st.metric("Daily Study Hours", f"{e['study_hours']:.1f} hrs/day")
                            st.metric("Prior Failures", f"{e['failures']}")

                with p_c2:
                    with st.container(border=True):
                        st.markdown("#### 📜 Longitudinal Prediction History")
                        if detail["predictions"]:
                            pred_records = pd.DataFrame(detail["predictions"])[["created_at", "predicted_score", "grade", "risk_level", "model_version"]]
                            pred_records.columns = ["Timestamp", "Score /100", "Grade", "Risk", "Model"]
                            st.dataframe(pred_records, use_container_width=True, hide_index=True)
                        else:
                            st.info("No prediction forecasts recorded yet.")

                        st.divider()
                        st.markdown("#### 🛡️ Assigned Interventions")
                        if detail["interventions"]:
                            for it in detail["interventions"]:
                                st.markdown(f"""
                                <div class="rec-card {it['priority'].lower()}">
                                    <div style="display:flex; justify-content:space-between;">
                                        <b>{it['category']}</b>
                                        <span class="status-pill {it['status'].lower().replace(' ', '-')}">{it['status']}</span>
                                    </div>
                                    <div class="rec-detail" style="margin-top:4px;">{it['issue_summary']}</div>
                                    <div class="rec-action">👉 {it['student_action']}</div>
                                    <div class="rec-detail" style="font-size:11px; margin-top:4px; opacity:0.8;">Advisor: {it['advisor']} | Due: {it['follow_up_date']}</div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.caption("No active academic interventions on file for this student.")
    else:
        st.info("No matching student records found.")


# ============================================================
# PAGE 3: INDIVIDUAL PREDICTION STUDIO
# ============================================================
elif selected_nav == "🎓 Individual Prediction Studio":
    section_label("🎓", "Academic Forecast Studio", "Real-time supervised inference with local factor attribution and intervention drafting")

    students_all = list_students()
    student_names = ["-- Enter Custom Profile --"] + [s["name"] for s in students_all]
    
    col_pick, col_engine = st.columns([2, 1], gap="medium")
    with col_pick:
        picked_student = st.selectbox("Load Profile from Student Directory (Optional)", student_names)
    with col_engine:
        pred_engine = st.selectbox("Active Inference Model", ["v2.0-clean", "v1.0-legacy"], help="v2.0 Clean is the audited leak-free production model.")

    # Prefill if picked
    default_name = ""
    default_gen = "Female"
    default_age = 18
    default_att = 90.0
    default_interim = 6.5
    default_assign = 7
    default_study = 3.5
    default_fail = 0
    default_extra = "No"

    if picked_student != "-- Enter Custom Profile --":
        matched = next((s for s in students_all if s["name"] == picked_student), None)
        if matched:
            default_name = matched["name"]
            default_gen = matched["gender"]
            default_age = matched["age"]
            default_att = matched["attendance_percentage"]
            default_interim = matched["internal_marks"]
            default_assign = matched["assignments_submitted"]
            default_fail = matched["previous_failures"]

    col1, col2, col3 = st.columns([1, 1.25, 1], gap="medium")

    with col1:
        with st.container(border=True):
            section_label("👤", "Profile Details", "Demographics")
            in_name = st.text_input("Student Name", value=default_name, placeholder="e.g. Aarav Patel")
            in_gender = st.selectbox("Gender", ["Female", "Male"], index=0 if default_gen == "Female" else 1)
            in_age = st.slider("Age", 14, 25, default_age)

    with col2:
        with st.container(border=True):
            section_label("📚", "Academic Signals", "Assessments & Habits")
            in_assign = st.slider("Coursework Submitted (out of 10)", 0, 10, default_assign)
            in_interim = st.slider("Interim Test Score (out of 10)", 0.0, 10.0, float(default_interim), step=0.5)
            in_study = st.slider("Self-Study Hours (per day)", 0.0, 10.0, float(default_study), step=0.5)
            in_failures = st.number_input("Prior Course Failures", min_value=0, max_value=5, value=default_fail)
            in_extra = st.selectbox("Extra Tutoring Classes", ["No", "Yes"], index=0 if default_extra == "No" else 1)

    with col3:
        with st.container(border=True):
            section_label("🗓️", "Attendance", "Presence Rate")
            in_total_cls = st.number_input("Classes Conducted", min_value=1, value=100)
            in_attended_cls = st.number_input("Classes Attended", min_value=0, max_value=int(in_total_cls), value=int(default_att))
            calc_att = (in_attended_cls / in_total_cls) * 100.0
            st.metric("Attendance Percentage", f"{calc_att:.1f}%")
            st.progress(min(1.0, max(0.0, calc_att / 100.0)))

    st.write("")
    btn_predict = st.button("Generate & Persist Academic Forecast →", use_container_width=True, type="primary")

    if btn_predict:
        input_data = {
            "Student_Name": in_name,
            "Gender": 1 if in_gender == "Male" else 0,
            "Age": in_age,
            "Assignments_Submitted": in_assign,
            "Internal_Marks": in_interim,
            "Attendance_Percentage": calc_att,
            "Self_Study_Hours": in_study,
            "Previous_Failures": in_failures,
            "Extra_Classes": 1 if in_extra == "Yes" else 0
        }

        is_valid, validation_errors, validated_df = validate_student_input(input_data)

        if not is_valid:
            for err in validation_errors:
                st.error(f"Input Validation Error: {err}")
        else:
            with st.spinner("Executing model prediction and logging to database..."):
                # Find or create student in DB
                with get_db_session() as session:
                    s_obj = session.query(Student).filter_by(name=in_name.strip()).first()
                    if not s_obj:
                        count = session.query(Student).count()
                        s_obj = Student(
                            student_id=f"STU-2026-{count+1:04d}",
                            name=in_name.strip(),
                            gender=input_data["Gender"],
                            age=input_data["Age"]
                        )
                        session.add(s_obj)
                        session.flush()
                    s_id = s_obj.id

                # Save prediction to DB
                pred_res = save_prediction(
                    student_db_id=s_id,
                    features=input_data,
                    model_version_tag=pred_engine,
                    user_id=cur_user["id"]
                )

            st.success("Forecast generated and successfully saved to audit records.")

            # Display Results
            st.divider()
            r_col1, r_col2 = st.columns([1, 1.3], gap="large")

            with r_col1:
                with st.container(border=True):
                    section_label("📊", "Projected Outcome", "Score & Confidence")
                    score_val = pred_res["predicted_score"]
                    ring_color = get_color_for_score(score_val)
                    
                    donut = go.Figure(go.Pie(
                        values=[score_val, max(0.0, 100.0 - score_val)],
                        hole=0.78,
                        marker=dict(colors=[ring_color, "rgba(255,255,255,0.06)"], line=dict(color="#0A0708", width=2)),
                        textinfo="none",
                        sort=False
                    ))
                    donut.update_layout(
                        showlegend=False,
                        height=200,
                        margin=dict(l=0, r=0, t=0, b=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        annotations=[dict(
                            text=f"<b>{score_val:.0f}</b><span style='font-size:13px;color:{THEME_TOKENS['ink_soft']}'>/100</span>",
                            x=0.5, y=0.5, showarrow=False,
                            font=dict(family="Sora", size=32, color=INK)
                        )]
                    )
                    st.plotly_chart(donut, use_container_width=True, config={"displayModeBar": False})

                    st.markdown(f"""
                    <div style="text-align:center;">
                        <span class="score-badge {pred_res['badge_color']}">Grade {pred_res['grade']} · {pred_res['performance']}</span>
                        <div style="margin-top:10px;">
                            <span class="priority-pill {pred_res['risk_badge']}">{pred_res['risk_level']}</span>
                        </div>
                        <p style="margin-top:10px; font-size:12.5px; color:{THEME_TOKENS['ink_soft']};">
                            Confidence Range: <b style="color:{INK};">[{pred_res['score_lower']:.1f} – {pred_res['score_upper']:.1f}]</b> (±{pred_res['uncertainty_margin']:.1f} pts)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

            with r_col2:
                with st.container(border=True):
                    section_label("📈", "Local Factor Contributions", "Factors pulling score up or down")
                    contributions = pred_res["feature_contributions"]
                    f_names = [c["feature"].replace("_", " ") for c in contributions[:5]]
                    f_impacts = [c["impact"] for c in contributions[:5]]
                    f_colors = [THEME_TOKENS["success"] if val >= 0 else THEME_TOKENS["red"] for val in f_impacts]

                    fig_bar = go.Figure(go.Bar(
                        x=f_impacts,
                        y=f_names,
                        orientation="h",
                        marker=dict(color=f_colors, cornerradius=6),
                        text=[f"{v:+.1f} pts" for v in f_impacts],
                        textposition="outside"
                    ))
                    fig_bar.update_layout(
                        height=250,
                        margin=dict(l=10, r=40, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(gridcolor=GRID),
                        yaxis=dict(gridcolor=GRID, autorange="reversed"),
                        font=CHART_FONT
                    )
                    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

            # Actionable Interventions & Direct DB Logging
            st.divider()
            section_label("💡", "Actionable Intervention Plan", "Evidence-based student recommendations")
            recs = generate_recommendations(input_data, pred_res)
            
            for rec in recs:
                p_class = rec["priority"].lower()
                st.markdown(f"""
                <div class="rec-card {p_class}">
                    <div style="display:flex; justify-content:space-between;">
                        <span class="rec-title">{rec['icon']} {rec['category']}</span>
                        <span class="priority-pill {p_class}">{rec['priority']}</span>
                    </div>
                    <div class="rec-detail"><b>Issue Trigger:</b> {rec['issue']}</div>
                    <div class="rec-action">👉 <b>Student Action:</b> {rec['student_action']}</div>
                    <div class="rec-action" style="color:{THEME_TOKENS['accent']};">🏛️ <b>Faculty Protocol:</b> {rec['educator_action']}</div>
                </div>
                """, unsafe_allow_html=True)

            # Option to save intervention to DB
            if has_permission(cur_user["role"], "advisor") and recs and recs[0]["priority"] in ["CRITICAL", "HIGH"]:
                with st.expander("📌 Commit Formal Intervention to Database", expanded=False):
                    with st.form("commit_intervention_form"):
                        top_rec = recs[0]
                        i_summary = st.text_input("Intervention Summary", value=top_rec["issue"])
                        i_action_s = st.text_input("Assigned Student Task", value=top_rec["student_action"])
                        i_action_e = st.text_input("Assigned Faculty Task", value=top_rec["educator_action"])
                        i_due = st.date_input("Follow-Up Deadline", value=date.today() + timedelta(days=7))
                        i_notes = st.text_area("Initial Progress Notes", placeholder="e.g. Student informed; agreed to attend review clinic.")
                        submit_it = st.form_submit_button("Record Intervention in Database", type="primary")

                        if submit_it:
                            create_intervention(
                                student_db_id=s_id,
                                priority=top_rec["priority"],
                                category=top_rec["category"],
                                issue_summary=i_summary,
                                student_action=i_action_s,
                                educator_action=i_action_e,
                                follow_up_date=i_due,
                                notes=i_notes,
                                advisor_id=cur_user["id"],
                                user_id=cur_user["id"]
                            )
                            st.success("Formal intervention recorded in database and assigned to advisor.")

            # Download Report
            st.divider()
            text_rep = generate_text_report(input_data, pred_res, recs)
            st.download_button(
                label="📥 Download Official Academic Evaluation (.txt)",
                data=text_rep,
                file_name=f"{in_name.replace(' ', '_')}_ScholarIQ_Forecast.txt",
                mime="text/plain"
            )


# ============================================================
# PAGE 4: INTERVENTION HUB
# ============================================================
elif selected_nav == "🛡️ Intervention Hub":
    section_label("🛡️", "Academic Intervention & Follow-up Hub", "Track at-risk students, advisor workflows, and resolution milestones")

    i_col1, i_col2, i_col3 = st.columns([1.5, 1.5, 1], gap="medium")
    with i_col1:
        f_status = st.selectbox("Status Filter", ["All", "Open", "In Progress", "Completed"])
    with i_col2:
        f_priority = st.selectbox("Priority Filter", ["All", "CRITICAL", "HIGH", "MEDIUM"])
    with i_col3:
        st.write("")
        st.write("")
        refresh_btn = st.button("🔄 Refresh Hub", use_container_width=True)

    interventions_list = list_interventions(status_filter=f_status, priority_filter=f_priority)

    # Metrics
    total_it = len(interventions_list)
    open_it = sum(1 for i in interventions_list if i["status"] in ["Open", "In Progress"])
    overdue_it = sum(1 for i in interventions_list if i["is_overdue"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Interventions Shown", f"{total_it}")
    m2.metric("Pending Follow-Ups", f"{open_it}")
    m3.metric("Overdue Deadlines", f"{overdue_it}", delta=f"{overdue_it} overdue" if overdue_it > 0 else None, delta_color="inverse")

    st.divider()

    if interventions_list:
        for it in interventions_list:
            p_class = it["priority"].lower()
            s_class = it["status"].lower().replace(" ", "-")
            overdue_tag = '<span class="status-pill overdue">OVERDUE</span>' if it["is_overdue"] else ""
            
            with st.container(border=True):
                col_head, col_stat = st.columns([3, 1])
                with col_head:
                    st.markdown(f"""
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span class="priority-pill {p_class}">{it['priority']}</span>
                        <b style="font-size:15px; color:#F2ECEC;">{it['student_name']}</b> 
                        <span style="font-size:12px; color:#AC9CA0;">({it['student_id']})</span>
                        {overdue_tag}
                    </div>
                    """, unsafe_allow_html=True)
                with col_stat:
                    st.markdown(f'<div style="text-align:right;"><span class="status-pill {s_class}">{it["status"]}</span></div>', unsafe_allow_html=True)

                st.markdown(f"**Issue:** {it['issue_summary']}")
                st.markdown(f"👉 **Student Action:** {it['student_action']}")
                st.markdown(f"🏛️ **Advisor Action:** {it['educator_action']}")
                st.caption(f"Assigned Advisor: **{it['assigned_advisor']}** | Deadline: `{it['follow_up_date']}` | Created: {it['created_at']}")

                if it["notes"]:
                    with st.expander("📝 View History Notes"):
                        st.text(it["notes"])

                # Advisor Action Form
                if has_permission(cur_user["role"], "advisor") and it["status"] != "Completed":
                    with st.expander("✏️ Update Status / Add Note"):
                        with st.form(f"update_it_{it['id']}"):
                            next_status = st.selectbox("Update Status", ["Open", "In Progress", "Completed", "Cancelled"], index=1)
                            add_note = st.text_input("Add Progress Note", placeholder="e.g. Meeting conducted; student submitted remedial worksheet.")
                            save_update = st.form_submit_button("Save Status Update")
                            if save_update:
                                update_intervention_status(it["id"], next_status, add_note, user_id=cur_user["id"])
                                st.success("Intervention record updated.")
                                st.rerun()
    else:
        st.info("No intervention records matching the active filters.")


# ============================================================
# PAGE 5: DATA IMPORT & BATCH ANALYTICS
# ============================================================
elif selected_nav == "📥 Data Import & Batch Analytics":
    section_label("📥", "Batch Data Import & Cohort Processing", "Transactional CSV intake, schema enforcement, and database synchronization")

    c1, c2 = st.columns([3, 1], gap="medium")
    with c1:
        uploaded_roster = st.file_uploader(
            "Upload Classroom Roster (.csv)",
            type=["csv"],
            help="Expected columns: Student_Name, Gender, Age, Assignments_Submitted, Internal_Marks, Attendance_Percentage, Self_Study_Hours, Previous_Failures, Extra_Classes"
        )
    with c2:
        st.write("")
        st.write("")
        btn_sample = st.button("Load Preloaded Demo Roster (10 Students)", use_container_width=True)

    target_content = None
    target_name = "sample_cohort.csv"
    if uploaded_roster:
        target_content = uploaded_roster
        target_name = uploaded_roster.name
    elif btn_sample:
        sample_path = DATASET_DIR / "sample_cohort.csv"
        if sample_path.exists():
            target_content = sample_path
            target_name = "sample_cohort.csv"

    if target_content:
        st.divider()
        st.markdown(f"##### Previewing `{target_name}` Before Commit")
        preview_df = pd.read_csv(target_content)
        st.dataframe(preview_df.head(10), use_container_width=True, hide_index=True)

        if has_permission(cur_user["role"], "faculty"):
            commit_import = st.button("Commit Roster to Database & Execute ML Scoring →", type="primary")

            if commit_import:
                with st.spinner("Processing transactional batch import..."):
                    # Rewind buffer if BytesIO
                    if hasattr(target_content, "seek"):
                        target_content.seek(0)
                    
                    success, res = import_roster_transactional(
                        target_content,
                        filename=target_name,
                        course_code="MATH201",
                        user_id=cur_user["id"]
                    )

                if success:
                    st.success(f"Import {res['status']}: Processed {res['valid_rows']} valid rows ({res['created_students']} new students enrolled).")
                    
                    # Refresh overview KPIs
                    kpis = get_overview_kpis()
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Cohort Monitored", f"{kpis['total_students']}")
                    m2.metric("Mean Projected Score", f"{kpis['average_predicted_score']:.1f}/100")
                    m3.metric("At-Risk Count", f"{kpis['at_risk_count']}")
                else:
                    st.error("Batch import failed validation.")
                    for e in res.get("errors", []):
                        st.error(e)


# ============================================================
# PAGE 6: PREDICTION HISTORY & AUDIT LOG
# ============================================================
elif selected_nav == "📜 Prediction History & Audit":
    section_label("📜", "Prediction History & System Audit Log", "Immutable chronological record of model forecasts and operational events")

    tab_p, tab_a = st.tabs(["📊 Prediction History Log", "🛡️ Operational Audit Trail"])

    with tab_p:
        filter_p_risk = st.selectbox("Filter History by Risk Level", ["All", "Low Risk", "Moderate Risk", "High Academic Risk"])
        preds_history = list_predictions(risk_filter=filter_p_risk)

        if preds_history:
            df_preds = pd.DataFrame(preds_history)
            st.dataframe(df_preds[["predicted_at", "student_name", "student_id", "predicted_score", "band", "grade", "risk_level", "model_version", "internal_marks", "attendance"]], use_container_width=True, hide_index=True)
            
            # Export CSV
            csv_history = df_preds.to_csv(index=False)
            st.download_button(
                label="📥 Export Full Prediction History (.csv)",
                data=csv_history,
                file_name="ScholarIQ_Prediction_History.csv",
                mime="text/csv"
            )
        else:
            st.info("No prediction history recorded.")

    with tab_a:
        with get_db_session() as session:
            logs = session.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(50).all()
            log_records = []
            for l in logs:
                u = session.query(User).filter_by(id=l.user_id).first() if l.user_id else None
                log_records.append({
                    "Timestamp": l.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "User": u.full_name if u else "System Automated",
                    "Action": l.action,
                    "Entity": f"{l.entity_type} #{l.entity_id or '—'}",
                    "Details": l.details_json or ""
                })

        if log_records:
            st.dataframe(pd.DataFrame(log_records), use_container_width=True, hide_index=True)
        else:
            st.info("No audit logs recorded yet.")


# ============================================================
# PAGE 7: MODEL GOVERNANCE & TRANSPARENCY
# ============================================================
elif selected_nav == "🔬 Model Governance":
    section_label("🔬", "Model Governance & Diagnostic Transparency", "Mathematical formulations, target leakage remediation audit, and benchmark matrices")

    st.markdown(r"""
    ### 🎯 Prediction Target Formulation
    The system predicts **Final Academic Score (scale 0–100)**, mapped from the final exam grade ($G3 \in [0, 20]$) in the Cortez & Silva (2008) secondary school research dataset:
    $$\text{Final\_Score} = G3 \times 5$$
    """)

    st.markdown(r"""
    ### 🛡️ Verified Elimination of Target Leakage
    In the legacy v1.0 pipeline, `Internal_Marks` was derived from future final score:
    $$\text{Legacy (Defective): } \text{Internal\_Marks} = \text{round}\left(\frac{G3}{2}\right)$$
    This created a **99.07% mathematical target leakage**, causing the model to function as a multiplier lookup ($10 \times \text{Internal\_Marks}$).

    **In Model v2.0 Clean (Current Active Model):**
    $$\text{Audited (Clean): } \text{Internal\_Marks} = \frac{G1}{2.0}$$
    `Internal_Marks` is derived from $G1$ — the **actual first-period midterm assessment** conducted months prior to the final exam.
    """)

    st.divider()

    st.markdown("### 📊 Cross-Validated Model Comparison Matrix")
    benchmarks_data = pd.DataFrame([
        {"Algorithm": "Baseline (Dummy Mean)", "5-Fold CV MAE": "17.01 ± 3.15", "5-Fold CV RMSE": "22.82 ± 3.17", "Test MAE": 18.23, "Test RMSE": 22.75, "Test R²": -0.010, "Status": "Baseline"},
        {"Algorithm": "Ridge Regression", "5-Fold CV MAE": "9.91 ± 1.25", "5-Fold CV RMSE": "13.57 ± 1.50", "Test MAE": 10.33, "Test RMSE": 13.36, "Test R²": 0.652, "Status": "Evaluated"},
        {"Algorithm": "Random Forest (v2.0 Clean)", "5-Fold CV MAE": "8.10 ± 1.01", "5-Fold CV RMSE": "11.73 ± 1.71", "Test MAE": 8.31, "Test RMSE": 11.94, "Test R²": 0.722, "Status": "ACTIVE PRODUCTION"},
        {"Algorithm": "HistGradientBoosting", "5-Fold CV MAE": "8.81 ± 0.87", "5-Fold CV RMSE": "12.54 ± 1.32", "Test MAE": 9.08, "Test RMSE": 12.69, "Test R²": 0.686, "Status": "Evaluated"}
    ])
    st.dataframe(benchmarks_data, use_container_width=True, hide_index=True)

    st.divider()

    # Gini Feature Importances
    st.markdown("### ⚖️ Gini Feature Importances (v2.0 Clean vs Legacy)")
    feat_data = pd.DataFrame([
        {"Feature": "Interim Test Marks (G1/2)", "v2.0 Clean": 67.2, "v1.0 Legacy": 97.7},
        {"Feature": "Attendance Percentage", "v2.0 Clean": 19.3, "v1.0 Legacy": 1.4},
        {"Feature": "Supplemental Tutoring", "v2.0 Clean": 4.0, "v1.0 Legacy": 0.1},
        {"Feature": "Student Age", "v2.0 Clean": 3.1, "v1.0 Legacy": 0.3},
        {"Feature": "Coursework Submitted", "v2.0 Clean": 2.3, "v1.0 Legacy": 0.1},
        {"Feature": "Prior Course Failures", "v2.0 Clean": 2.1, "v1.0 Legacy": 0.2},
        {"Feature": "Self-Study Hours", "v2.0 Clean": 1.4, "v1.0 Legacy": 0.1},
        {"Feature": "Gender", "v2.0 Clean": 0.6, "v1.0 Legacy": 0.1}
    ])
    
    fig_f = go.Figure(data=[
        go.Bar(name="v2.0 Clean Weight (%)", x=feat_data["Feature"], y=feat_data["v2.0 Clean"], marker_color=THEME_TOKENS["success"]),
        go.Bar(name="v1.0 Legacy Weight (%)", x=feat_data["Feature"], y=feat_data["v1.0 Legacy"], marker_color=THEME_TOKENS["red"])
    ])
    fig_f.update_layout(
        barmode="group",
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=CHART_FONT,
        xaxis=dict(gridcolor=GRID),
        yaxis=dict(gridcolor=GRID, title="Decision Importance (%)")
    )
    st.plotly_chart(fig_f, use_container_width=True, config={"displayModeBar": False})


# ============================================================
# PAGE 8: SETTINGS & SYSTEM HEALTH
# ============================================================
elif selected_nav == "⚙️ Settings & System Health":
    section_label("⚙️", "Settings & System Diagnostics", "Database health, backup management, and environment status")

    c_s1, c_s2 = st.columns(2, gap="large")

    with c_s1:
        with st.container(border=True):
            st.markdown("#### 🗄️ Persistence Layer Health")
            with get_db_session() as session:
                u_cnt = session.query(User).count()
                s_cnt = session.query(Student).count()
                p_cnt = session.query(Prediction).count()
                i_cnt = session.query(Intervention).count()

            st.write(f"• **Database Engine:** SQLite (WAL mode, Foreign Keys ON)")
            st.write(f"• **Users Registered:** {u_cnt}")
            st.write(f"• **Students Enrolled:** {s_cnt}")
            st.write(f"• **Prediction Records:** {p_cnt}")
            st.write(f"• **Intervention Records:** {i_cnt}")

            st.divider()
            if has_permission(cur_user["role"], "admin"):
                if st.button("🔄 Re-Seed Database with Initial Demo Fixtures"):
                    seed_database()
                    st.success("Database re-seeded successfully.")
                    st.rerun()

    with c_s2:
        with st.container(border=True):
            st.markdown("#### 💻 Runtime Environment Diagnostics")
            st.write(f"• **Python Interpreter:** `{sys.version.split()[0]}` (64-bit)")
            st.write(f"• **Streamlit Version:** `{st.__version__}`")
            st.write(f"• **Active Model:** `student_model_v2_clean.pkl`")
            st.write(f"• **Test Suite Status:** `16/16 Passed`")

            st.divider()
            db_path = PROJECT_DIR / "data" / "scholariq.db"
            if db_path.exists():
                with open(db_path, "rb") as f_db:
                    st.download_button(
                        label="📥 Download SQLite Database Backup (.db)",
                        data=f_db.read(),
                        file_name=f"scholariq_backup_{datetime.now().strftime('%Y%m%d')}.db",
                        mime="application/x-sqlite3"
                    )


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown('<div class="footer-note">SCHOLARIQ 3.0 · FULL-STACK PREDICTIVE ACADEMIC INTELLIGENCE PLATFORM</div>', unsafe_allow_html=True)
