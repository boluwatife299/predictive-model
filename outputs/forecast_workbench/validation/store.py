"""
Storage backend resolver for Forecast Workbench.

Resolves a SQLAlchemy engine from configuration, with this precedence:

    1. ``DATABASE_URL`` environment variable
    2. ``st.secrets["DATABASE_URL"]`` (Streamlit Cloud / local secrets.toml)
    3. Local SQLite file next to this module  (default fallback)

In the cloud the SQLite file lives on an ephemeral container disk and is
wiped on every restart/redeploy — that is exactly why forecast history kept
disappearing. Point ``DATABASE_URL`` at a managed Postgres (e.g. Supabase)
and history survives. Locally, with no URL set, SQLite is perfectly fine.

The schema is defined with SQLAlchemy Core so it is portable across SQLite
and Postgres (autoincrement primary keys, types, etc. are handled per-engine).
"""
from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
)
from sqlalchemy.engine import Engine

# Default SQLite location — survives across *local* sessions only.
_SQLITE_PATH = Path(__file__).parent / "model_runs.db"

metadata = MetaData()

model_runs = Table(
    "model_runs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_at", DateTime(timezone=True), nullable=False),
    Column("ticker", String(64), nullable=False),
    Column("asset_class", String(64), nullable=False),
    Column("model_name", String(128), nullable=False),
    Column("horizon_days", Integer, nullable=False),
    Column("s0", Float, nullable=False),
    Column("predicted_p50", Float, nullable=False),
    Column("predicted_p5", Float, nullable=False),
    Column("predicted_p95", Float, nullable=False),
    Column("mu", Float, nullable=False),
    Column("sigma", Float, nullable=False),
    Column("params", Text, nullable=False),
)


def _resolve_url() -> str:
    """Return the database URL using the documented precedence."""
    url = os.getenv("DATABASE_URL")
    src = "env DATABASE_URL"
    if not url:
        try:
            import streamlit as st  # imported lazily; tracker may run headless

            url = st.secrets.get("DATABASE_URL")  # type: ignore[assignment]
            src = "st.secrets DATABASE_URL"
        except Exception:
            url = None
    if not url:
        return f"sqlite:///{_SQLITE_PATH}", False, "no DATABASE_URL configured"

    # Supabase / Heroku hand out "postgres://" which SQLAlchemy rejects;
    # normalise to the driver-qualified form.
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url, True, src


_engine: Engine | None = None
_SQLITE_FALLBACK = f"sqlite:///{_SQLITE_PATH}"

# Diagnostic status, populated on first get_engine() call.
_status: dict = {
    "configured": False,   # was a DATABASE_URL found at all?
    "connected": False,    # did we actually connect to managed Postgres?
    "driver": "sqlite",
    "source": "default",
    "error": None,         # connection error reason, if any
}


def get_engine() -> Engine:
    """
    Return a process-wide cached SQLAlchemy engine.

    If a DATABASE_URL is configured but the connection fails, we fall back to
    local SQLite (so the app still runs) and record the failure reason in the
    status dict — never crash the whole app over a bad connection string.
    """
    global _engine
    if _engine is not None:
        return _engine

    url, configured, source = _resolve_url()
    _status["configured"] = configured
    _status["source"] = source

    try:
        connect_args = {}
        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False  # Streamlit threads reruns
        engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
        metadata.create_all(engine)          # forces a real connection
        _engine = engine
        _status["driver"] = engine.url.drivername
        _status["connected"] = not engine.url.drivername.startswith("sqlite")
        _status["error"] = None if configured else "no DATABASE_URL configured"
        return _engine
    except Exception as exc:
        # Managed DB unreachable / bad credentials — degrade to SQLite, remember why.
        _status["connected"] = False
        _status["error"] = f"{type(exc).__name__}: {exc}"
        engine = create_engine(_SQLITE_FALLBACK, connect_args={"check_same_thread": False})
        metadata.create_all(engine)
        _engine = engine
        _status["driver"] = "sqlite (fallback)"
        return _engine


def get_status() -> dict:
    """Return a copy of the persistence diagnostic status (engine resolved lazily)."""
    get_engine()
    return dict(_status)


def is_cloud_persistent() -> bool:
    """True when backed by a managed DB rather than ephemeral local SQLite."""
    return get_status()["connected"]
