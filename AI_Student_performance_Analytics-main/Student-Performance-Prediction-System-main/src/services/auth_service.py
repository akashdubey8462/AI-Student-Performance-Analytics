"""ScholarIQ Authentication and Authorization Service.
Provides Argon2id password hashing, transparent legacy migration,
role hierarchies, and authorization enforcement.
"""
import hashlib
import os
import secrets
from typing import Optional, Dict, Any, Tuple
from ..database.connection import get_db_session
from ..database.models import User, AuditLog

# Try importing argon2-cffi, with robust fallback to standard library PBKDF2-HMAC
try:
    from argon2 import PasswordHasher
    from argon2.exceptions import VerifyMismatchError, VerificationError
    _ARGON2_AVAILABLE = True
    _ph = PasswordHasher(time_cost=2, memory_cost=65536, parallelism=2, hash_len=32)
except ImportError:
    _ARGON2_AVAILABLE = False
    _ph = None

ROLE_HIERARCHY = {
    "admin": ["admin", "faculty", "advisor", "student"],
    "faculty": ["faculty", "advisor", "student"],
    "advisor": ["advisor", "student"],
    "student": ["student"]
}

LEGACY_STATIC_SALT = "scholariq_secure_salt_2026"


def hash_password(password: str) -> str:
    """Hash password using Argon2id (OWASP recommended) or PBKDF2-HMAC-SHA256 fallback."""
    if _ARGON2_AVAILABLE and _ph is not None:
        return _ph.hash(password)
    
    # Standard library fallback: PBKDF2-HMAC-SHA256 with 600,000 iterations & random 16-byte salt
    salt = secrets.token_hex(16)
    iterations = 600000
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
    return f"pbkdf2_sha256${iterations}${salt}${derived.hex()}"


def verify_password(plain_password: str, stored_hash: str) -> Tuple[bool, bool]:
    """Verify password against stored hash.
    
    Returns:
        (is_valid: bool, needs_rehash: bool)
    """
    if not stored_hash or not plain_password:
        return False, False

    # 1. Check Argon2id format
    if stored_hash.startswith("$argon2"):
        if not _ARGON2_AVAILABLE or _ph is None:
            return False, False
        try:
            _ph.verify(stored_hash, plain_password)
            needs_rehash = _ph.check_needs_rehash(stored_hash)
            return True, needs_rehash
        except (VerifyMismatchError, VerificationError):
            return False, False

    # 2. Check PBKDF2 format
    if stored_hash.startswith("pbkdf2_sha256$"):
        parts = stored_hash.split("$")
        if len(parts) == 4:
            _, iterations_str, salt, hash_hex = parts
            try:
                iterations = int(iterations_str)
                derived = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt.encode("utf-8"), iterations)
                if secrets.compare_digest(derived.hex(), hash_hex):
                    # If Argon2 is now available, flag for rehash to Argon2id
                    return True, _ARGON2_AVAILABLE
            except Exception:
                return False, False
        return False, False

    # 3. Check Legacy SHA-256 with static salt (64 hex characters)
    legacy_hash = hashlib.sha256((plain_password + LEGACY_STATIC_SALT).encode("utf-8")).hexdigest()
    if secrets.compare_digest(legacy_hash, stored_hash):
        # Successfully matched legacy hash; MUST be rehashed to Argon2id
        return True, True

    return False, False


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Verify credentials, transparently migrate legacy hashes, and return user dict or None."""
    if not username or not password:
        return None

    clean_username = username.strip()
    clean_password = password.strip()

    with get_db_session() as session:
        user = session.query(User).filter_by(username=clean_username, is_active=True).first()
        if not user:
            return None

        is_valid, needs_rehash = verify_password(clean_password, user.password_hash)

        if is_valid:
            # Transparent migration: rehash to modern Argon2id if old format or outdated parameters
            if needs_rehash:
                user.password_hash = hash_password(clean_password)
                session.add(user)

            # Log audit entry
            session.add(AuditLog(
                user_id=user.id,
                action="USER_LOGIN_SUCCESS",
                entity_type="USER",
                entity_id=user.id,
                details_json=f'{{"username": "{user.username}", "role": "{user.role}", "migrated_hash": {str(needs_rehash).lower()}}}'
            ))

            return {
                "id": user.id,
                "username": user.username,
                "full_name": user.full_name,
                "role": user.role,
                "email": user.email
            }
        else:
            # Log failed attempt
            session.add(AuditLog(
                user_id=user.id,
                action="USER_LOGIN_FAILED_PASSWORD",
                entity_type="USER",
                entity_id=user.id,
                details_json=f'{{"username": "{clean_username}"}}'
            ))
            return None


def has_permission(user_role: str, target_role: str) -> bool:
    """Check if user_role has access rights to features reserved for target_role."""
    allowed = ROLE_HIERARCHY.get(user_role, [])
    return target_role in allowed


def require_permission(current_user: Optional[Dict[str, Any]], required_role: str) -> None:
    """Enforce authorization guard at service level. Raises PermissionError if unauthorized."""
    if not current_user:
        raise PermissionError("Authentication required: No active user session.")
    
    role = current_user.get("role", "")
    if not has_permission(role, required_role):
        raise PermissionError(f"Access denied: Role '{role}' does not possess required privilege '{required_role}'.")
