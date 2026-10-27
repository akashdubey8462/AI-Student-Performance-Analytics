"""ScholarIQ Database Connection & Session Management.
Supports SQLite (default), PostgreSQL via SQLAlchemy, and isolated in-memory test databases.
"""
import os
from pathlib import Path
from contextlib import contextmanager
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent.parent
DEFAULT_DB_DIR = PROJECT_DIR / "data"
DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_SQLITE_PATH = DEFAULT_DB_DIR / "scholariq.db"

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_SQLITE_PATH}")

Base = declarative_base()


def build_engine(url: str):
    """Construct an engine configured appropriately for SQLite WAL or in-memory StaticPool."""
    is_sqlite = url.startswith("sqlite")
    engine_kwargs = {}
    
    if is_sqlite:
        engine_kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url:
            engine_kwargs["poolclass"] = StaticPool

    eng = create_engine(url, echo=False, **engine_kwargs)

    # Enable Foreign Key enforcement and WAL mode for persistent SQLite
    if is_sqlite:
        @event.listens_for(eng, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            if ":memory:" not in url:
                cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    return eng


engine = build_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def set_engine_url(url: str):
    """Rebind engine and session factory to a new database URL (used for test isolation)."""
    global engine, SessionLocal, DATABASE_URL
    DATABASE_URL = url
    engine = build_engine(url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine


@contextmanager
def get_db_session():
    """Transactional database session context manager."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(target_engine=None):
    """Create all tables in the database."""
    from . import models  # noqa
    eng = target_engine or engine
    Base.metadata.create_all(bind=eng)


def drop_db(target_engine=None):
    """Drop all tables in the database (used in test teardown)."""
    from . import models  # noqa
    eng = target_engine or engine
    Base.metadata.drop_all(bind=eng)
