"""ScholarIQ Recommendations Module.
Generates transparent, evidence-based academic intervention suggestions and educator guidance.
"""
from typing import Dict, Any, List


def generate_recommendations(student_data: Dict[str, Any], prediction_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Produce structured recommendations with observable triggers, rationale, and educator advice."""
    recommendations = []
    
    attendance = float(student_data.get("Attendance_Percentage", 90.0))
    assignments = int(student_data.get("Assignments_Submitted", 6))
    internal = float(student_data.get("Internal_Marks", 5.0))
    study_hours = float(student_data.get("Self_Study_Hours", 3.0))
    failures = int(student_data.get("Previous_Failures", 0))
    extra_classes = int(student_data.get("Extra_Classes", 0))
    predicted_score = float(prediction_data.get("predicted_score", 50.0))
    
    # 1. Critical: Prior Failures / Academic Backlog
    if failures > 0:
        recommendations.append({
            "priority": "CRITICAL",
            "category": "Curriculum Continuity",
            "icon": "⚠️",
            "issue": f"{failures} previous course failure(s) recorded.",
            "rationale": "Accumulated backlogs create conceptual gaps that compound in advanced subjects.",
            "student_action": "Prioritize remedial prerequisite revision and clear outstanding backlogs.",
            "educator_action": "Verify prerequisite mastery and assign peer tutoring or targeted academic mentoring."
        })

    # 2. High: Chronic Absenteeism
    if attendance < 75.0:
        recommendations.append({
            "priority": "HIGH",
            "category": "Attendance & Engagement",
            "icon": "📅",
            "issue": f"Attendance rate is {attendance:.1f}% (below the 75% institutional threshold).",
            "rationale": "Students missing more than 25% of sessions forfeit in-class discussions, practical labs, and quiz opportunities.",
            "student_action": "Commit to attending all remaining lecture hours to recover the attendance percentage.",
            "educator_action": "Investigate non-academic factors (health, commute, schedule clashes) and issue an early-warning notice."
        })
        
    # 3. High: Low Interim Assessment Performance
    if internal < 5.0:
        recommendations.append({
            "priority": "HIGH",
            "category": "Assessment Mastery",
            "icon": "📝",
            "issue": f"Interim marks are {internal:.1f}/10 (borderline passing or failing).",
            "rationale": "Early internal assessments account for major predictive weight in end-of-semester results.",
            "student_action": "Review graded internal tests with course instructors to pinpoint recurring conceptual mistakes.",
            "educator_action": "Provide formative mock quizzes and guided problem-solving sessions before final exams."
        })

    # 4. Medium: Low Self-Study Duration
    if study_hours < 3.0:
        recommendations.append({
            "priority": "MEDIUM",
            "category": "Study Habits",
            "icon": "📖",
            "issue": f"Self-study hours logged at {study_hours:.1f} hrs/day (below recommended 3+ hrs).",
            "rationale": "Independent study reinforces lecture material and improves problem retention.",
            "student_action": "Establish a consistent daily revision schedule of at least 2.5–3 hours using spaced repetition.",
            "educator_action": "Provide structured study timetables and chapter-wise summary sheets."
        })

    # 5. Medium: Incomplete Assignments
    if assignments < 7:
        recommendations.append({
            "priority": "MEDIUM",
            "category": "Coursework",
            "icon": "📚",
            "issue": f"Submitted only {assignments}/10 coursework assignments.",
            "rationale": "Coursework reinforces practical application and builds continuous confidence.",
            "student_action": "Submit all remaining assignments before deadlines, even if partially completed for feedback.",
            "educator_action": "Check for submission blockers and offer brief assignment clinics for struggling students."
        })

    # 6. Informational: Academic Support Opportunity
    if extra_classes == 0 and predicted_score < 70.0:
        recommendations.append({
            "priority": "MEDIUM",
            "category": "Academic Support",
            "icon": "🎯",
            "issue": "Not currently enrolled in departmental remedial or extra support classes.",
            "rationale": "Extra classes provide high-touch faculty interaction for topics requiring reinforcement.",
            "student_action": "Enroll in university review sessions or faculty office hours for challenging topics.",
            "educator_action": "Directly invite student to supplemental instruction sessions."
        })

    # Positive Profile
    if not recommendations:
        recommendations.append({
            "priority": "STRENGTH",
            "category": "Exemplary Performance",
            "icon": "🎉",
            "issue": "Profile shows strong consistency across all monitored academic indicators.",
            "rationale": "High attendance, active assignment completion, and consistent study hours strongly correlate with top-tier outcomes.",
            "student_action": "Maintain your disciplined study cadence and consider peer tutoring to deepen conceptual mastery.",
            "educator_action": "Encourage participation in advanced honors projects, undergraduate research, or competition teams."
        })

    return recommendations
