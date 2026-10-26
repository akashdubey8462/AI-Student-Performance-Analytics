"""ScholarIQ Academic Marking & Calculation Service.
Implements the official university marking rules, selection policies, attendance bonuses,
component-level validations, and incomplete-scheme provisional accounting.
"""
from typing import Dict, Any, Tuple, List, Optional


# Default Institutional Marking Configuration
DEFAULT_ACADEMIC_RULES = {
    "total_target_marks": 420.0,
    "midsem": {
        "max_marks_per_exam": 20.0,
        "selection_policy": "best_2_of_3",  # options: "best_2_of_3", "first_2", "all_3"
        "weight_in_total": 20.0,
        "mandatory": True
    },
    "attendance": {
        "thresholds": [
            {"min_pct": 85.0, "bonus": 3},
            {"min_pct": 75.0, "bonus": 2},
            {"min_pct": 60.0, "bonus": 1}
        ],
        "default_bonus": 0,
        "add_to": "midsem"
    },
    "activities": {
        "max_marks_per_activity": 10.0,
        "selection_rule": "higher_of_two",
        "weight_in_total": 10.0,
        "mandatory": True
    },
    "practical_external": {
        "max_marks": 30.0,
        "mandatory": True
    },
    "term_work": {
        "max_marks": 20.0,
        "mandatory": True
    },
    "pbl": {
        "max_marks": 50.0,
        "mandatory": True
    },
    "internship": {
        "max_marks": 50.0,
        "mandatory": True
    },
    "grading_scale": {
        "scale_type": "10_point_cgpa",
        "unconfirmed": True,
        "passing_threshold_pct": 40.0,
        "bands": [
            {"min_pct": 90.0, "grade": "O", "grade_point": 10.0, "desc": "Outstanding"},
            {"min_pct": 80.0, "grade": "A+", "grade_point": 9.0, "desc": "Excellent"},
            {"min_pct": 70.0, "grade": "A", "grade_point": 8.0, "desc": "Very Good"},
            {"min_pct": 60.0, "grade": "B+", "grade_point": 7.0, "desc": "Good"},
            {"min_pct": 50.0, "grade": "B", "grade_point": 6.0, "desc": "Above Average"},
            {"min_pct": 40.0, "grade": "C", "grade_point": 5.0, "desc": "Pass"},
            {"min_pct": 0.0, "grade": "F", "grade_point": 0.0, "desc": "Fail"}
        ]
    }
}


def calculate_attendance_bonus(attendance_pct: float, thresholds: Optional[List[Dict[str, Any]]] = None) -> int:
    """Calculate attendance bonus marks based on strict non-cumulative institutional thresholds.
    
    Rules:
      - Attendance > 85%: 3 bonus marks
      - Attendance > 75%: 2 bonus marks
      - Attendance > 60%: 1 bonus mark
      - Attendance <= 60%: 0 bonus marks
      
    Boundary verification:
      59% -> 0
      60% -> 0
      61% -> 1
      75% -> 1
      76% -> 2
      85% -> 2
      86% -> 3
    """
    if attendance_pct < 0.0 or attendance_pct > 100.0:
        raise ValueError(f"Attendance percentage must be between 0.0 and 100.0, received {attendance_pct}")

    if thresholds is None:
        thresholds = DEFAULT_ACADEMIC_RULES["attendance"]["thresholds"]

    # Sort descending by min_pct to evaluate highest threshold first
    sorted_thresholds = sorted(thresholds, key=lambda t: t["min_pct"], reverse=True)
    for t in sorted_thresholds:
        # User rule explicitly specifies 'above X%', which translates to strict inequality (attendance > min_pct)
        if attendance_pct > t["min_pct"]:
            return int(t["bonus"])
            
    return 0


def calculate_midsem_average(
    midsem_1: float,
    midsem_2: float,
    midsem_3: float,
    policy: str = "best_2_of_3",
    max_marks: float = 20.0
) -> Tuple[float, List[float]]:
    """Calculate the mid-semester examination average based on configured selection policy.
    
    Validates that each exam is within [0, max_marks].
    Policy 'best_2_of_3' averages the two highest examination scores.
    """
    for idx, score in enumerate([midsem_1, midsem_2, midsem_3], 1):
        if score < 0.0 or score > max_marks:
            raise ValueError(f"Mid-Semester Exam {idx} mark must be between 0 and {max_marks}, received {score}")

    scores = [float(midsem_1), float(midsem_2), float(midsem_3)]
    
    if policy == "best_2_of_3":
        selected = sorted(scores, reverse=True)[:2]
        avg = sum(selected) / 2.0
    elif policy == "first_2":
        selected = [scores[0], scores[1]]
        avg = sum(selected) / 2.0
    elif policy == "all_3":
        selected = scores
        avg = sum(selected) / 3.0
    else:
        # Default fallback to best 2
        selected = sorted(scores, reverse=True)[:2]
        avg = sum(selected) / 2.0

    return round(avg, 2), selected


def calculate_counted_activity(
    activity_1: float,
    activity_2: float,
    max_marks: float = 10.0
) -> Tuple[float, int]:
    """Calculate counted activity score using the higher-of-two policy.
    
    Formula: max(Activity 1, Activity 2).
    Returns (counted_score, winning_activity_number).
    """
    for idx, score in enumerate([activity_1, activity_2], 1):
        if score < 0.0 or score > max_marks:
            raise ValueError(f"Activity {idx} mark must be between 0 and {max_marks}, received {score}")

    score_1 = float(activity_1)
    score_2 = float(activity_2)
    
    if score_1 >= score_2:
        return score_1, 1
    return score_2, 2


def calculate_academic_marks(
    inputs: Dict[str, Any],
    custom_rules: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Calculate complete deterministic academic marks based on user's official rules.
    
    Accurately tracks provisional subtotal and highlights unconfigured portions of the 420-mark scheme.
    """
    rules = custom_rules or DEFAULT_ACADEMIC_RULES
    errors = []
    
    # 1. Mid-Semester Component
    m1 = float(inputs.get("midsem_1", 0.0))
    m2 = float(inputs.get("midsem_2", 0.0))
    m3 = float(inputs.get("midsem_3", 0.0))
    m_policy = rules["midsem"]["selection_policy"]
    m_max = rules["midsem"]["max_marks_per_exam"]
    
    try:
        midsem_avg, selected_exams = calculate_midsem_average(m1, m2, m3, policy=m_policy, max_marks=m_max)
    except ValueError as e:
        errors.append(str(e))
        midsem_avg, selected_exams = 0.0, [0.0, 0.0]

    # 2. Attendance & Bonus
    att_pct = float(inputs.get("attendance_percentage", 0.0))
    try:
        att_bonus = calculate_attendance_bonus(att_pct, rules["attendance"]["thresholds"])
    except ValueError as e:
        errors.append(str(e))
        att_bonus = 0

    midsem_with_bonus = round(midsem_avg + att_bonus, 2)
    midsem_exceeded_warning = None
    if midsem_with_bonus > m_max:
        midsem_exceeded_warning = (
            f"Mid-semester score with attendance bonus ({midsem_with_bonus:.1f}) exceeds configured exam max "
            f"({m_max:.1f}). Preserved as {midsem_with_bonus:.1f} without arbitrary rescaling until official college policy is confirmed."
        )

    # 3. Academic Activities
    act_1 = float(inputs.get("activity_1", 0.0))
    act_2 = float(inputs.get("activity_2", 0.0))
    act_max = rules["activities"]["max_marks_per_activity"]
    try:
        counted_act, win_act = calculate_counted_activity(act_1, act_2, max_marks=act_max)
    except ValueError as e:
        errors.append(str(e))
        counted_act, win_act = 0.0, 1

    # 4. Practical & Project Components
    prac_ext = float(inputs.get("practical_external", 0.0))
    tw = float(inputs.get("term_work", 0.0))
    pbl = float(inputs.get("pbl", 0.0))
    internship = float(inputs.get("internship", 0.0))

    prac_max = rules["practical_external"]["max_marks"]
    tw_max = rules["term_work"]["max_marks"]
    pbl_max = rules["pbl"]["max_marks"]
    intern_max = rules["internship"]["max_marks"]

    if prac_ext < 0.0 or prac_ext > prac_max:
        errors.append(f"Practical External marks must be between 0 and {prac_max}, received {prac_ext}")
    if tw < 0.0 or tw > tw_max:
        errors.append(f"Term Work (TW) marks must be between 0 and {tw_max}, received {tw}")
    if pbl < 0.0 or pbl > pbl_max:
        errors.append(f"Project-Based Learning (PBL) marks must be between 0 and {pbl_max}, received {pbl}")
    if internship < 0.0 or internship > intern_max:
        errors.append(f"Internship marks must be between 0 and {intern_max}, received {internship}")

    # 5. Provisional Subtotal & 420-Mark Accounting
    # Configured maximum marks:
    # Mid-sem (20) + Activities (10) + Practical (30) + TW (20) + PBL (50) + Internship (50) = 180
    base_configured_max = m_max + act_max + prac_max + tw_max + pbl_max + intern_max
    
    # Subtotal scored by student (including bonus)
    provisional_subtotal = round(
        midsem_with_bonus + counted_act + prac_ext + tw + pbl + internship, 2
    )
    
    target_total = float(rules.get("total_target_marks", 420.0))
    unallocated_marks = max(0.0, target_total - base_configured_max)
    is_scheme_complete = (unallocated_marks == 0.0)

    # Component Breakdown Registry
    components = [
        {
            "component": "Mid-Semester Examinations",
            "component_name": "Mid-Semester Examinations",
            "assessment_type": "Continuous Theory Assessment",
            "selection_policy": m_policy,
            "rule": f"{m_policy.replace('_', ' ').title()} (Tests: {m1:.1f}, {m2:.1f}, {m3:.1f})",
            "scored": midsem_avg,
            "bonus": att_bonus,
            "total_with_bonus": midsem_with_bonus,
            "max_marks": m_max,
            "contributes_to_final": True,
            "mandatory": True,
            "status": "Configured",
            "weighting": f"{m_max:.0f} / {target_total:.0f}"
        },
        {
            "component": "Academic Activities",
            "component_name": "Academic Activities",
            "assessment_type": "Continuous Academic Activities",
            "selection_policy": "higher_of_two",
            "rule": f"Higher of Act 1 ({act_1:.1f}) & Act 2 ({act_2:.1f})",
            "scored": counted_act,
            "bonus": 0,
            "total_with_bonus": counted_act,
            "max_marks": act_max,
            "contributes_to_final": True,
            "mandatory": True,
            "status": "Configured",
            "weighting": f"{act_max:.0f} / {target_total:.0f}"
        },
        {
            "component": "Practical External Examination",
            "component_name": "Practical External Examination",
            "assessment_type": "Laboratory / Practical Evaluation",
            "selection_policy": "direct_score",
            "rule": "External Lab Evaluation",
            "scored": prac_ext,
            "bonus": 0,
            "total_with_bonus": prac_ext,
            "max_marks": prac_max,
            "contributes_to_final": True,
            "mandatory": True,
            "status": "Configured",
            "weighting": f"{prac_max:.0f} / {target_total:.0f}"
        },
        {
            "component": "Term Work (TW)",
            "component_name": "Term Work (TW)",
            "assessment_type": "Sessional & Lab Performance",
            "selection_policy": "direct_score",
            "rule": "Continuous Laboratory & Sessional Work",
            "scored": tw,
            "bonus": 0,
            "total_with_bonus": tw,
            "max_marks": tw_max,
            "contributes_to_final": True,
            "mandatory": True,
            "status": "Configured",
            "weighting": f"{tw_max:.0f} / {target_total:.0f}"
        },
        {
            "component": "Project-Based Learning (PBL)",
            "component_name": "Project-Based Learning (PBL)",
            "assessment_type": "Capstone / Team Problem-Solving",
            "selection_policy": "direct_score",
            "rule": "Capstone / Team Problem-Solving Project",
            "scored": pbl,
            "bonus": 0,
            "total_with_bonus": pbl,
            "max_marks": pbl_max,
            "contributes_to_final": True,
            "mandatory": True,
            "status": "Configured",
            "weighting": f"{pbl_max:.0f} / {target_total:.0f}"
        },
        {
            "component": "Internship Assessment",
            "component_name": "Internship Assessment",
            "assessment_type": "Industry / Research Internship",
            "selection_policy": "direct_score",
            "rule": "Industry / Research Internship Evaluation",
            "scored": internship,
            "bonus": 0,
            "total_with_bonus": internship,
            "max_marks": intern_max,
            "contributes_to_final": True,
            "mandatory": True,
            "status": "Configured",
            "weighting": f"{intern_max:.0f} / {target_total:.0f}"
        },
        {
            "component": "Unconfigured Semester Components",
            "component_name": "Unconfigured Semester Components (e.g. End-Sem Theory)",
            "assessment_type": "Terminal Examination",
            "selection_policy": "pending_definition",
            "rule": "Awaiting Official Scheme Confirmation (e.g. End-Sem Theory)",
            "scored": None,
            "bonus": 0,
            "total_with_bonus": None,
            "max_marks": unallocated_marks,
            "contributes_to_final": True,
            "mandatory": True,
            "status": "Pending Configuration",
            "weighting": f"{unallocated_marks:.0f} / {target_total:.0f}"
        }
    ]

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "provisional_subtotal": provisional_subtotal,
        "base_configured_max": base_configured_max,
        "total_target_marks": target_total,
        "unallocated_marks": unallocated_marks,
        "is_scheme_complete": is_scheme_complete,
        "completeness_percentage": round((base_configured_max / target_total) * 100.0, 1),
        "midsem_avg": midsem_avg,
        "attendance_bonus": att_bonus,
        "midsem_with_bonus": midsem_with_bonus,
        "midsem_exceeded_warning": midsem_exceeded_warning,
        "counted_activity": counted_act,
        "selected_midsem_exams": selected_exams,
        "winning_activity": win_act,
        "components": components,
        "scheme_warning": (
            f"Marking scheme incomplete — official total and final grade are unavailable. "
            f"Only {base_configured_max:.0f} of {target_total:.0f} marks ({(base_configured_max / target_total) * 100:.1f}%) "
            f"are currently configured. {unallocated_marks:.0f} marks remain unallocated (e.g., End-Semester Theory). "
            f"Provisional subtotal ({provisional_subtotal:.1f} / {base_configured_max:.0f}) reflects internal assessments only."
        ) if not is_scheme_complete else None
    }
