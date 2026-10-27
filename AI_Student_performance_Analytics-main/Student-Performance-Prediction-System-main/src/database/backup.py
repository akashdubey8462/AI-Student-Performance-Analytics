"""ScholarIQ Database Backup, Restore, and Integrity Verification Utility.
Provides zero-data-loss snapshots, schema verification, and restore routines.
"""
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from .connection import DATABASE_URL, DEFAULT_SQLITE_PATH, get_db_session, engine
from .models import User, Student, Course, Enrollment, ModelVersion, Prediction, Intervention


def get_sqlite_file_path() -> Optional[Path]:
    """Extract SQLite database file path from connection URL."""
    if not DATABASE_URL.startswith("sqlite"):
        return None
    clean = DATABASE_URL.replace("sqlite:///", "")
    return Path(clean) if clean != ":memory:" else None


def create_database_backup(destination_dir: Optional[Path] = None) -> Path:
    """Create a point-in-time timestamped backup copy of the SQLite database.
    
    Uses SQLite's online backup API to ensure snapshot consistency even under concurrent writes.
    """
    source_path = get_sqlite_file_path()
    if not source_path or not source_path.exists():
        raise FileNotFoundError(f"Source database file not found at {source_path}")

    dest_dir = destination_dir or (source_path.parent / "backups")
    dest_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_file = dest_dir / f"scholariq_backup_{timestamp}.db"

    # Use SQLite native backup API for consistency
    src_conn = sqlite3.connect(str(source_path))
    dst_conn = sqlite3.connect(str(backup_file))
    try:
        with dst_conn:
            src_conn.backup(dst_conn, pages=100)
    finally:
        dst_conn.close()
        src_conn.close()

    return backup_file


def restore_database_backup(backup_file: Path) -> bool:
    """Safely restore database from a verified backup file."""
    if not backup_file.exists():
        raise FileNotFoundError(f"Backup file not found at {backup_file}")

    target_path = get_sqlite_file_path()
    if not target_path:
        raise ValueError("Cannot restore non-SQLite database using file snapshot routine.")

    # Verify backup integrity first
    test_conn = sqlite3.connect(str(backup_file))
    try:
        cursor = test_conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        res = cursor.fetchone()
        if not res or res[0] != "ok":
            raise ValueError(f"Backup file corrupted: {res}")
    finally:
        test_conn.close()

    # Perform restoration
    shutil.copy2(backup_file, target_path)
    return True


def verify_database_integrity() -> Dict[str, Any]:
    """Execute PRAGMA integrity check and verify row counts across key academic entities."""
    source_path = get_sqlite_file_path()
    if not source_path or not source_path.exists():
        return {"status": "UNAVAILABLE", "message": "In-memory or remote database."}

    conn = sqlite3.connect(str(source_path))
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        integrity = cursor.fetchone()[0]
        cursor.execute("PRAGMA foreign_key_check;")
        fk_violations = cursor.fetchall()
    finally:
        conn.close()

    counts = {}
    with get_db_session() as s:
        counts["users"] = s.query(User).count()
        counts["students"] = s.query(Student).count()
        counts["courses"] = s.query(Course).count()
        counts["enrollments"] = s.query(Enrollment).count()
        counts["predictions"] = s.query(Prediction).count()
        counts["interventions"] = s.query(Intervention).count()
        counts["models"] = s.query(ModelVersion).count()

    is_healthy = (integrity == "ok") and (len(fk_violations) == 0)
    return {
        "status": "HEALTHY" if is_healthy else "CORRUPTED",
        "sqlite_integrity": integrity,
        "foreign_key_violations": len(fk_violations),
        "table_row_counts": counts,
        "checked_at": datetime.now(timezone.utc).isoformat()
    }
