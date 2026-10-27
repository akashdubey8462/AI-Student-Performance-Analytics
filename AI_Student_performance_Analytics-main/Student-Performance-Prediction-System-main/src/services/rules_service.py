"""ScholarIQ Academic Rules Configuration & Persistence Service.
Manages the institutional marking policies, component caps, attendance thresholds,
and persistence to disk/database with validation and reset capabilities.
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from copy import deepcopy
from .marking_service import DEFAULT_ACADEMIC_RULES

RULES_FILE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "academic_rules.json"


def get_academic_rules() -> Dict[str, Any]:
    """Retrieve the currently active academic rules configuration.
    Falls back to DEFAULT_ACADEMIC_RULES if no custom configuration exists.
    """
    if RULES_FILE_PATH.exists():
        try:
            with open(RULES_FILE_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
                # Merge with default to guarantee all keys exist
                rules = deepcopy(DEFAULT_ACADEMIC_RULES)
                _deep_update(rules, saved)
                return rules
        except Exception:
            return deepcopy(DEFAULT_ACADEMIC_RULES)
    return deepcopy(DEFAULT_ACADEMIC_RULES)


def save_academic_rules(new_rules: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """Validate and persist new academic rules configuration.
    Returns (success: bool, message: str, rules: Dict).
    """
    valid, err_msg = validate_academic_rules(new_rules)
    if not valid:
        return False, err_msg, get_academic_rules()

    try:
        RULES_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RULES_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(new_rules, f, indent=2)
        return True, "Academic rules updated successfully.", new_rules
    except Exception as e:
        return False, f"Failed to persist academic rules: {str(e)}", get_academic_rules()


def reset_academic_rules() -> Dict[str, Any]:
    """Reset configuration to institutional default rules."""
    defaults = deepcopy(DEFAULT_ACADEMIC_RULES)
    try:
        RULES_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RULES_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(defaults, f, indent=2)
    except Exception:
        pass
    return defaults


def validate_academic_rules(rules: Dict[str, Any]) -> tuple:
    """Validate that configured rules contain valid numerical bounds and allowed policies."""
    if not isinstance(rules, dict):
        return False, "Rules payload must be a JSON object"

    # Total target marks
    target = rules.get("total_target_marks")
    if target is None or not isinstance(target, (int, float)) or target <= 0:
        return False, "total_target_marks must be a positive number"

    # Midsem
    midsem = rules.get("midsem", {})
    if midsem.get("selection_policy") not in ["best_2_of_3", "first_2", "all_3"]:
        return False, f"Invalid midsem selection_policy: {midsem.get('selection_policy')}"
    if float(midsem.get("max_marks_per_exam", 0)) <= 0:
        return False, "Midsem max_marks_per_exam must be positive"

    # Attendance thresholds
    att = rules.get("attendance", {})
    thresholds = att.get("thresholds", [])
    if not isinstance(thresholds, list):
        return False, "Attendance thresholds must be a list"
    for t in thresholds:
        if "min_pct" not in t or "bonus" not in t:
            return False, "Each attendance threshold must have 'min_pct' and 'bonus'"
        if t["min_pct"] < 0 or t["min_pct"] > 100:
            return False, "Attendance threshold min_pct must be between 0 and 100"

    # Activities
    act = rules.get("activities", {})
    if float(act.get("max_marks_per_activity", 0)) <= 0:
        return False, "Activity max_marks_per_activity must be positive"

    # Practical, TW, PBL, Internship
    for comp in ["practical_external", "term_work", "pbl", "internship"]:
        c = rules.get(comp, {})
        if float(c.get("max_marks", 0)) <= 0:
            return False, f"{comp} max_marks must be positive"

    return True, "Valid"


def _deep_update(target: dict, source: dict):
    """Recursively update nested dictionary."""
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


