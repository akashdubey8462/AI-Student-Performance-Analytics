"""ScholarIQ Authentication and Authorization Service.
Handles credentials, role hierarchies, and session context.
"""
import hashlib
from typing import Optional, Dict, Any
from ..database.connection import get_db_session
from ..database.models import User, AuditLog

ROLE_HIERARCHY = {
    "admin": ["admin", "faculty", "advisor", "student"],
    "faculty": ["faculty", "advisor", "student"],
    "advisor": ["advisor", "student"],
    "student": ["student"]
}


def hash_password(password: str) -> str:
    """Create SHA-256 hash with static application salt."""
    salt = "scholariq_secure_salt_2026"
    return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Verify credentials and return user dict or None."""
    hashed = hash_password(password.strip())
    with get_db_session() as session:
        user = session.query(User).filter_by(username=username.strip(), is_active=True).first()
        if user and user.password_hash == hashed:
            # Log successful login
            session.add(AuditLog(
                user_id=user.id,
                action="USER_LOGIN_SUCCESS",
                entity_type="USER",
                entity_id=user.id,
                details_json=f'{{"username": "{user.username}", "role": "{user.role}"}}'
            ))
            return {
                "id": user.id,
                "username": user.username,
                "full_name": user.full_name,
                "role": user.role,
                "email": user.email
            }
        elif user:
            # Log failed attempt
            session.add(AuditLog(
                user_id=user.id,
                action="USER_LOGIN_FAILED_PASSWORD",
                entity_type="USER",
                entity_id=user.id,
                details_json=f'{{"username": "{username}"}}'
            ))
    return None


def has_permission(user_role: str, target_role: str) -> bool:
    """Check if user_role has access rights to features reserved for target_role."""
    allowed = ROLE_HIERARCHY.get(user_role, [])
    return target_role in allowed
