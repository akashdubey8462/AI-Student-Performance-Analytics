# ============================================================
# ScholarIQ — AI Student Academic Performance Analytics Platform
# Production Application Layer (v2.0 Clean Architecture)
# ============================================================

import sys
from pathlib import Path

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
from datetime import datetime

from src.config import (
    THEME_TOKENS,
    DEFAULT_MODEL_V2,
    DEFAULT_METADATA_V2,
    LEGACY_MODEL_V1,
    DATASET_DIR
)
from src.data_validation import validate_student_input, validate_batch_csv
from src.model_service import ModelService
from src.recommendations import generate_recommendations
from src.report_generator import generate_text_report, generate_cohort_summary_csv


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="ScholarIQ — AI Student Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Load CSS Theme
# -----------------------------
def load_css():
    css_path = APP_DIR / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("style.css not found — running with default styling.")

load_css()


# -----------------------------
# Model Service Factory (Cached)
# -----------------------------
@st.cache_resource
def get_model_service(version_key: str) -> ModelService:
    return ModelService(model_version=version_key)


# -----------------------------
# Chart Helpers
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
# HERO HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <div class="logo-lockup">
        <div class="logo-mark">🎓</div>
        <div class="logo-word">ScholarIQ <span class="dim">· Predictive Academic Intelligence</span></div>
    </div>
    <div class="main-title">Early-warning student analytics<br><span class="accent">before results day</span></div>
    <div class="sub-title">A scientifically audited machine learning platform predicting academic trajectories, identifying at-risk signals, and providing evidence-based faculty interventions.</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR CONFIGURATION
# ============================================================
with st.sidebar:
    st.markdown('<div class="preview-header"><span class="preview-pulse"></span>Platform Control</div>', unsafe_allow_html=True)
    
    model_choice = st.selectbox(
        "Inference Engine",
        options=["v2_clean", "legacy_v1"],
        format_func=lambda x: "Model v2.0 Clean (Audited • Leak-Free)" if x == "v2_clean" else "Model v1.0 Legacy (Historical Baseline)",
        help="Select between the audited clean model (trained on genuine pre-exam interim signals) and the historical baseline."
    )
    
    service = get_model_service(model_choice)
    
    if service.load_error:
        st.error(f"Engine Warning: {service.load_error}")
    else:
        leakage_badge = "VERIFIED CLEAN" if model_choice == "v2_clean" else "TARGET LEAKED (Audit Only)"
        pill_class = "strength" if model_choice == "v2_clean" else "critical"
        st.markdown(
            f'<div style="margin: 8px 0 16px 0;"><span class="priority-pill {pill_class}">{leakage_badge}</span></div>',
            unsafe_allow_html=True
        )
        
    st.divider()
    
    st.markdown("### 📌 Navigation & Capabilities")
    st.caption("• **Individual Forecast**: Student early warning & factor analysis\n• **Cohort Analytics**: CSV batch roster processing\n• **Model Transparency**: Benchmark audit & limitations")
    
    st.divider()
    st.markdown('<div class="footer-note">SCHOLARIQ v2.0 · SCIKIT-LEARN ENGINE</div>', unsafe_allow_html=True)


# ============================================================
# MAIN APPLICATION TABS
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "🎓 Individual Student Forecast",
    "📊 Cohort Batch Analytics",
    "🔬 Model Transparency & Audit"
])


# ============================================================
# TAB 1: INDIVIDUAL FORECAST
# ============================================================
with tab1:
    col1, col2, col3 = st.columns([1, 1.25, 1], gap="medium")

    with col1:
        with st.container(border=True):
            section_label("👤", "Student Profile", "Demographic & identification")
            student_name = st.text_input("Student Name", value="", placeholder="e.g. Aditi Sharma")
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 15, 25, 18)

    with col2:
        with st.container(border=True):
            section_label("📚", "Academic Signals", "Coursework & assessment data")
            assignments = st.slider("Coursework Submitted (out of 10)", 0, 10, 7)
            internal_marks = st.slider(
                "Interim Test Score (out of 10)",
                0.0, 10.0, 6.0, step=0.5,
                help="First-period midterm or internal assessment score. Validated prior assessment."
            )
            study_hours = st.slider("Self-Study Hours (per day)", 0.0, 10.0, 3.5, step=0.5)
            failures = st.number_input("Previous Course Failures", min_value=0, max_value=5, value=0)
            extra_classes = st.selectbox("Supplemental Remedial Classes", ["No", "Yes"])

    with col3:
        with st.container(border=True):
            section_label("🗓️", "Attendance", "Presence & classroom engagement")
            total_classes = st.number_input("Total Scheduled Classes", min_value=1, value=100)
            attended_classes = st.number_input(
                "Classes Attended", min_value=0, max_value=int(total_classes), value=90
            )
            attendance = (attended_classes / total_classes) * 100.0
            st.metric("Attendance Percentage", f"{attendance:.1f}%")
            st.progress(min(1.0, max(0.0, attendance / 100.0)))

    # Live Sidebar Summary (Only on Individual tab)
    with st.sidebar:
        st.markdown('<div class="preview-header"><span class="preview-pulse"></span>Live Preview</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="preview-card">
            <div class="preview-name">{student_name.strip() if student_name.strip() else "Unnamed Student"}</div>
            <div class="preview-meta">{gender} · Age {age}</div>
        </div>
        """, unsafe_allow_html=True)
        st.metric("Attendance Rate", f"{attendance:.1f}%")
        st.metric("Interim Score", f"{internal_marks:.1f}/10")
        st.metric("Daily Study", f"{study_hours:.1f} hrs")
        
        if attendance < 75.0:
            st.markdown('<div class="preview-tip warn">⚠️ Attendance below 75% threshold</div>', unsafe_allow_html=True)
        elif failures > 0:
            st.markdown(f'<div class="preview-tip warn">⚠️ {failures} academic backlog(s)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="preview-tip good">✅ Profile signals healthy</div>', unsafe_allow_html=True)

    st.write("")
    predict_btn = st.button("Generate Academic Forecast →", use_container_width=True, type="primary")

    if predict_btn:
        # Validate input
        raw_input = {
            "Student_Name": student_name,
            "Gender": 1 if gender == "Male" else 0,
            "Age": age,
            "Assignments_Submitted": assignments,
            "Internal_Marks": internal_marks,
            "Attendance_Percentage": attendance,
            "Self_Study_Hours": study_hours,
            "Previous_Failures": failures,
            "Extra_Classes": 1 if extra_classes == "Yes" else 0
        }
        
        is_valid, validation_errors, validated_df = validate_student_input(raw_input)
        
        if not is_valid:
            for err in validation_errors:
                st.error(f"Validation Error: {err}")
        elif service.model is None:
            st.error(f"Prediction blocked: Model could not be loaded ({service.load_error})")
        else:
            # Inference execution
            try:
                prediction_result = service.predict_single(validated_df)
            except Exception as e:
                st.error(f"Inference exception: {str(e)}")
                st.stop()
                
            pred_score = prediction_result["predicted_score"]
            lower_b = prediction_result["score_lower"]
            upper_b = prediction_result["score_upper"]
            margin = prediction_result["uncertainty_margin"]
            grade = prediction_result["grade"]
            performance = prediction_result["performance"]
            risk_level = prediction_result["risk_level"]
            risk_badge = prediction_result["risk_badge"]
            risk_desc = prediction_result["risk_desc"]
            contributions = prediction_result["feature_contributions"]
            
            # Generate actionable recommendations
            recs = generate_recommendations(raw_input, prediction_result)

            st.markdown('<div class="reveal">', unsafe_allow_html=True)
            st.divider()

            # Prediction Display (Donut Ring + Factor Contribution)
            col_res, col_factors = st.columns([1, 1.3], gap="large")

            with col_res:
                with st.container(border=True):
                    section_label("📊", "Forecast Result", "Continuous score & risk tier")
                    
                    ring_color = get_color_for_score(pred_score)
                    donut = go.Figure(go.Pie(
                        values=[pred_score, max(0.0, 100.0 - pred_score)],
                        hole=0.78,
                        marker=dict(colors=[ring_color, "rgba(255,255,255,0.06)"], line=dict(color="#0A0708", width=2)),
                        textinfo="none",
                        sort=False,
                        direction="clockwise"
                    ))
                    donut.update_layout(
                        showlegend=False,
                        height=200,
                        margin=dict(l=0, r=0, t=0, b=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        annotations=[dict(
                            text=f"<b>{pred_score:.0f}</b><span style='font-size:13px;color:{THEME_TOKENS['ink_soft']}'>/100</span>",
                            x=0.5, y=0.5, showarrow=False,
                            font=dict(family="Sora", size=32, color=INK)
                        )]
                    )
                    st.plotly_chart(donut, use_container_width=True, config={"displayModeBar": False})

                    st.markdown(f"""
                    <div style="text-align:center;">
                        <span class="score-badge {prediction_result['badge_color']}">Grade {grade} · {performance}</span>
                        <div style="margin-top:10px;">
                            <span class="priority-pill {risk_badge}">{risk_level}</span>
                        </div>
                        <p style="margin-top:12px;color:{THEME_TOKENS['ink_soft']};font-size:13px;line-height:1.5;">
                            Confidence Band: <b style="color:{INK};">[{lower_b:.1f} – {upper_b:.1f}]</b> (±{margin:.1f} pts)<br>
                            <span style="font-size:12px;opacity:0.8;">{risk_desc}</span>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

            with col_factors:
                with st.container(border=True):
                    section_label("📈", "Factor Contributions", "Local model attribution relative to average")
                    
                    factor_names = [c["feature"].replace("_", " ") for c in contributions[:5]]
                    impact_vals = [c["impact"] for c in contributions[:5]]
                    bar_colors = [THEME_TOKENS["success"] if v >= 0 else THEME_TOKENS["red"] for v in impact_vals]

                    fig_bar = go.Figure(go.Bar(
                        x=impact_vals,
                        y=factor_names,
                        orientation="h",
                        marker=dict(color=bar_colors, cornerradius=6),
                        text=[f"{v:+.1f} pts" for v in impact_vals],
                        textposition="outside",
                        textfont=dict(color=INK, family="Inter", size=12)
                    ))
                    fig_bar.update_layout(
                        height=250,
                        margin=dict(l=10, r=40, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor="rgba(255,255,255,0.2)"),
                        yaxis=dict(showgrid=False, autorange="reversed", color=THEME_TOKENS["ink_soft"]),
                        font=CHART_FONT
                    )
                    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
                    st.caption("Green bars indicate factors boosting the forecast; red bars indicate negative drag.")

            st.divider()

            # Recommendations Panel
            section_label("💡", "Evidence-Based Recommendations", "Prioritized interventions for student and faculty")
            for rec in recs:
                p_class = rec["priority"].lower()
                st.markdown(f"""
                <div class="rec-card {p_class}">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="rec-title">{rec['icon']} {rec['category']}</span>
                        <span class="priority-pill {p_class}">{rec['priority']}</span>
                    </div>
                    <div class="rec-detail"><b>Observed Issue:</b> {rec['issue']}</div>
                    <div class="rec-detail"><b>Evidence Rationale:</b> {rec['rationale']}</div>
                    <div class="rec-action">👉 <b>Student Action:</b> {rec['student_action']}</div>
                    <div class="rec-action" style="color:{THEME_TOKENS['accent']};">🏛️ <b>Faculty/Advisor Guidance:</b> {rec['educator_action']}</div>
                </div>
                """, unsafe_allow_html=True)

            st.divider()

            # Report Download
            section_label("📥", "Export Academic Evaluation", "Downloadable audit records")
            txt_report = generate_text_report(raw_input, prediction_result, recs)
            
            clean_filename = student_name.strip().replace(" ", "_") if student_name.strip() else "Student"
            st.download_button(
                label="Download Official Text Report (.txt)",
                data=txt_report,
                file_name=f"{clean_filename}_ScholarIQ_Report.txt",
                mime="text/plain",
                type="secondary"
            )
            st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# TAB 2: COHORT BATCH ANALYTICS
# ============================================================
with tab2:
    section_label("👥", "Cohort Batch Analysis", "Evaluate classroom rosters with instant risk stratification")
    
    col_up, col_btn = st.columns([3, 1], gap="medium")
    with col_up:
        uploaded_file = st.file_uploader(
            "Upload Class Roster CSV",
            type=["csv"],
            help="Upload a CSV with student attributes: Gender, Age, Assignments_Submitted, Internal_Marks, Attendance_Percentage, Self_Study_Hours, Previous_Failures, Extra_Classes"
        )
    with col_btn:
        st.write("")
        st.write("")
        load_sample = st.button("Load Demo Cohort (10 Students)", use_container_width=True)

    target_file = None
    if uploaded_file is not None:
        target_file = uploaded_file
    elif load_sample:
        sample_path = DATASET_DIR / "sample_cohort.csv"
        if sample_path.exists():
            target_file = str(sample_path)
        else:
            st.error("Sample cohort file not found in Dataset directory.")

    if target_file is not None:
        is_valid, batch_errs, clean_cohort_df, summary = validate_batch_csv(target_file)
        
        if not is_valid:
            for err in batch_errs:
                st.error(f"Cohort Error: {err}")
        else:
            # Batch Inference
            with st.spinner("Processing cohort predictions..."):
                scored_cohort = service.predict_batch(clean_cohort_df)
                
            st.success(f"Successfully processed {summary['valid_rows']} student records. ({summary['rejected_rows']} rejected rows).")
            
            # Metrics Ribbon
            m1, m2, m3, m4 = st.columns(4)
            avg_score = scored_cohort["Predicted_Score"].mean()
            pass_rate = (scored_cohort["Passed"].sum() / len(scored_cohort)) * 100.0
            at_risk_cnt = (scored_cohort["Risk_Level"] == "High Academic Risk").sum()
            outstanding_cnt = (scored_cohort["Grade"].isin(["A+", "A"])).sum()
            
            m1.metric("Class Average Score", f"{avg_score:.1f} / 100")
            m2.metric("Projected Pass Rate", f"{pass_rate:.1f}%")
            m3.metric("At-Risk Students", f"{at_risk_cnt}", delta=f"{at_risk_cnt} need intervention", delta_color="inverse")
            m4.metric("Top Honors (A/A+)", f"{outstanding_cnt}")
            
            st.divider()
            
            # Cohort Visualizations
            v1, v2 = st.columns([1.2, 1], gap="medium")
            
            with v1:
                st.markdown("##### 📊 Projected Score Distribution")
                hist_fig = px.histogram(
                    scored_cohort,
                    x="Predicted_Score",
                    nbins=12,
                    color_discrete_sequence=[THEME_TOKENS["maroon"]],
                    labels={"Predicted_Score": "Projected Score"}
                )
                hist_fig.update_layout(
                    height=260,
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=CHART_FONT,
                    xaxis=dict(gridcolor=GRID),
                    yaxis=dict(gridcolor=GRID)
                )
                st.plotly_chart(hist_fig, use_container_width=True, config={"displayModeBar": False})
                
            with v2:
                st.markdown("##### 🛡️ Risk Stratification")
                risk_counts = scored_cohort["Risk_Level"].value_counts().reset_index()
                risk_counts.columns = ["Risk_Level", "Count"]
                color_map = {
                    "Low Risk": THEME_TOKENS["success"],
                    "Moderate Risk": THEME_TOKENS["amber"],
                    "High Academic Risk": THEME_TOKENS["red"]
                }
                pie_fig = px.pie(
                    risk_counts,
                    names="Risk_Level",
                    values="Count",
                    hole=0.6,
                    color="Risk_Level",
                    color_discrete_map=color_map
                )
                pie_fig.update_layout(
                    height=260,
                    margin=dict(l=0, r=0, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=CHART_FONT,
                    showlegend=True
                )
                st.plotly_chart(pie_fig, use_container_width=True, config={"displayModeBar": False})
                
            st.divider()
            
            # Scored Table & Export
            st.markdown("##### 📋 Complete Evaluated Cohort Roster")
            display_cols = ["Student_Name", "Predicted_Score", "Grade", "Risk_Level", "Attendance_Percentage", "Internal_Marks", "Previous_Failures"]
            st.dataframe(scored_cohort[display_cols], use_container_width=True, hide_index=True)
            
            csv_data = generate_cohort_summary_csv(scored_cohort)
            st.download_button(
                label="📥 Download Scored Cohort CSV",
                data=csv_data,
                file_name="ScholarIQ_Cohort_Evaluated.csv",
                mime="text/csv",
                type="primary"
            )


# ============================================================
# TAB 3: MODEL TRANSPARENCY & AUDIT
# ============================================================
with tab3:
    section_label("🔬", "Model Diagnostics & Methodology", "Full disclosure of mathematical formulation, training, and benchmarks")
    
    st.markdown("""
    ### 🎯 Prediction Target Formulation
    The primary target is **Final Course Score (scale 0–100)**, mapped from the final academic examination grade ($G3 \in [0, 20]$) in the Cortez & Silva (2008) secondary school research dataset:
    $$\\text{Final\\_Score} = G3 \\times 5$$
    """)
    
    st.markdown("""
    ### 🛡️ Elimination of Target Leakage
    In the legacy version of this system, `Internal_Marks` was artificially synthesized by dividing the final exam grade by two:
    $$\\text{Legacy (Defective): } \\text{Internal\\_Marks} = \\text{round}\\left(\\frac{G3}{2}\\right)$$
    Because $\\text{Final\\_Score} = G3 \\times 5$, this created a **99.07% mathematical target leakage**, effectively turning the Random Forest into a multiplier lookup table ($10 \\times \\text{Internal\\_Marks}$).
    
    **In Model v2.0 Clean:**
    $$\\text{Audited (Clean): } \\text{Internal\\_Marks} = \\frac{G1}{2.0}$$
    `Internal_Marks` is derived from $G1$ — the **actual first-period midterm assessment** conducted months prior to the final exam. This represents valid pre-event data with zero target contamination.
    """)
    
    st.divider()
    
    # Benchmark Comparison Table
    st.markdown("### 📊 Cross-Validated Model Benchmarks")
    st.caption("5-Fold Cross-Validation on Training Set (N=316) and Untouched Test Set (N=79, 20%)")
    
    benchmarks_data = pd.DataFrame([
        {"Algorithm": "Baseline (Dummy Mean)", "5-Fold CV MAE": "17.01 ± 3.15", "5-Fold CV RMSE": "22.82 ± 3.17", "Test MAE": 18.23, "Test RMSE": 22.75, "Test R²": -0.010},
        {"Algorithm": "Ridge Regression", "5-Fold CV MAE": "9.91 ± 1.25", "5-Fold CV RMSE": "13.57 ± 1.50", "Test MAE": 10.33, "Test RMSE": 13.36, "Test R²": 0.652},
        {"Algorithm": "Random Forest (v2.0 Clean)", "5-Fold CV MAE": "8.10 ± 1.01", "5-Fold CV RMSE": "11.73 ± 1.71", "Test MAE": 8.31, "Test RMSE": 11.94, "Test R²": 0.722},
        {"Algorithm": "HistGradientBoosting", "5-Fold CV MAE": "8.81 ± 0.87", "5-Fold CV RMSE": "12.54 ± 1.32", "Test MAE": 9.08, "Test RMSE": 12.69, "Test R²": 0.686}
    ])
    st.dataframe(benchmarks_data, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Feature Importances Chart
    st.markdown("### ⚖️ Gini Feature Importances (v2.0 Clean)")
    if service.metadata and "feature_importances" in service.metadata:
        imps = service.metadata["feature_importances"]
        imp_df = pd.DataFrame(list(imps.items()), columns=["Feature", "Importance"])
        imp_df["Feature"] = imp_df["Feature"].str.replace("_", " ")
        imp_df["Percentage"] = imp_df["Importance"] * 100.0
        
        imp_fig = px.bar(
            imp_df,
            x="Percentage",
            y="Feature",
            orientation="h",
            color="Percentage",
            color_continuous_scale=["#3DDC97", "#D4A24E", "#8C1C3B"],
            labels={"Percentage": "Relative Importance (%)"}
        )
        imp_fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=CHART_FONT,
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(imp_fig, use_container_width=True, config={"displayModeBar": False})
        
    st.divider()
    
    # Limitations and Data Card
    st.markdown("""
    ### ⚠️ Limitations & Responsible AI Notice
    1. **Dataset Origin:** The underlying training dataset is sourced from Portuguese secondary school mathematics students (Cortez and Silva, 2008, N=395). While statistically robust for demonstration, real-world institutional deployment requires calibration on local cohort data.
    2. **Zero Infiltration / Dropouts:** In the historical dataset, 38 students (9.6%) scored zero on the final exam due to exam absence or dropout. The continuous regression model may underestimate zero-inflation risk without separate binary classification.
    3. **Human-in-the-Loop:** Predictions must serve strictly as **advisory early-warning signals**. They must never be used for automated grade assignment or punitive decisions.
    """)


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown('<div class="footer-note">SCHOLARIQ · PREDICTIVE ACADEMIC INTELLIGENCE · AUDITED FOR PRODUCTION</div>', unsafe_allow_html=True)
