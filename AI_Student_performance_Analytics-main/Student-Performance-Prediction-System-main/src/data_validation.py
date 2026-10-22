"""ScholarIQ Data Validation Module.
Handles single-student input validation and batch CSV roster parsing.
"""
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
from .config import FEATURE_COLUMNS, FEATURE_RANGES


def validate_student_input(data: Dict[str, Any]) -> Tuple[bool, List[str], Optional[pd.DataFrame]]:
    """Validate a single student's input dictionary against expected schema and bounds.
    
    Returns:
        (is_valid, error_messages, dataframe_ready_for_model)
    """
    errors = []
    processed = {}
    
    # Check student name
    name = str(data.get("Student_Name", "")).strip()
    if not name:
        errors.append("Student name is required to personalize the academic report.")
        
    for feature in FEATURE_COLUMNS:
        if feature not in data:
            errors.append(f"Missing required attribute: '{feature}'")
            continue
            
        val = data[feature]
        rules = FEATURE_RANGES[feature]
        
        try:
            converted = rules["type"](val)
            if converted < rules["min"] or converted > rules["max"]:
                errors.append(
                    f"{feature} value ({converted}) is out of legitimate range "
                    f"[{rules['min']}, {rules['max']}]."
                )
            processed[feature] = [converted]
        except (ValueError, TypeError):
            errors.append(f"Invalid format for {feature}: expected {rules['type'].__name__}, received '{val}'.")
            
    if errors:
        return False, errors, None
        
    df = pd.DataFrame(processed)[FEATURE_COLUMNS]
    return True, [], df


def validate_batch_csv(file_or_df: Any) -> Tuple[bool, List[str], Optional[pd.DataFrame], Dict[str, Any]]:
    """Validate a batch CSV roster for cohort inference.
    
    Returns:
        (is_valid, errors, cleaned_df, summary_stats)
    """
    errors = []
    summary = {
        "total_rows": 0,
        "valid_rows": 0,
        "rejected_rows": 0,
        "missing_columns": [],
        "row_errors": []
    }
    
    try:
        if isinstance(file_or_df, pd.DataFrame):
            df = file_or_df.copy()
        else:
            df = pd.read_csv(file_or_df)
    except Exception as e:
        return False, [f"Failed to parse CSV file: {str(e)}"], None, summary
        
    summary["total_rows"] = len(df)
    if len(df) == 0:
        return False, ["Uploaded CSV file contains no data rows."], None, summary
        
    # Check required columns
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        summary["missing_columns"] = missing_cols
        return False, [f"Missing required columns in CSV: {', '.join(missing_cols)}"], None, summary
        
    # Optional name column
    if "Student_Name" not in df.columns:
        df["Student_Name"] = [f"Student {i+1}" for i in range(len(df))]
        
    valid_indices = []
    row_errors = []
    
    for idx, row in df.iterrows():
        row_issues = []
        for feature in FEATURE_COLUMNS:
            val = row[feature]
            if pd.isna(val):
                row_issues.append(f"{feature} is null")
                continue
                
            rules = FEATURE_RANGES[feature]
            try:
                converted = rules["type"](val)
                if converted < rules["min"] or converted > rules["max"]:
                    row_issues.append(f"{feature} ({converted}) out of bounds [{rules['min']}, {rules['max']}]")
            except (ValueError, TypeError):
                row_issues.append(f"{feature} invalid type ({val})")
                
        if row_issues:
            row_errors.append(f"Row {idx + 1}: " + "; ".join(row_issues))
        else:
            valid_indices.append(idx)
            
    summary["valid_rows"] = len(valid_indices)
    summary["rejected_rows"] = len(df) - len(valid_indices)
    summary["row_errors"] = row_errors[:20]  # Cap preview at 20 rows
    
    if len(valid_indices) == 0:
        return False, ["All rows failed validation checks."], None, summary
        
    clean_df = df.loc[valid_indices].copy()
    
    # Cast types properly
    for feature in FEATURE_COLUMNS:
        clean_df[feature] = clean_df[feature].astype(FEATURE_RANGES[feature]["type"])
        
    return True, [], clean_df, summary
