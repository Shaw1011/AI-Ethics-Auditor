"""
Database initialisation and session management.
Sessions are scoped per-request and cleaned up automatically.
"""
from __future__ import annotations

from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker

Base = declarative_base()

_Session: scoped_session | None = None


def init_db(app) -> None:
    """Bind the engine to the app config and create all tables."""
    global _Session

    engine = create_engine(
        app.config["SQLALCHEMY_DATABASE_URI"],
        **app.config.get("SQLALCHEMY_ENGINE_OPTIONS", {}),
    )

    # Enable WAL mode for SQLite concurrency (ignored by other DBs)
    @event.listens_for(engine, "connect")
    def _set_wal(dbapi_conn, _):
        if "sqlite" in app.config["SQLALCHEMY_DATABASE_URI"]:
            dbapi_conn.execute("PRAGMA journal_mode=WAL")
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

    _Session = scoped_session(
        sessionmaker(autocommit=False, autoflush=False, bind=engine)
    )
    Base.query = _Session.query_property()

    # Import models so they register with Base before create_all
    from app.models import audit, model, token_blocklist, user  # noqa: F401

    import os
    os.makedirs("data", exist_ok=True)

    with app.app_context():
        Base.metadata.create_all(bind=engine)

    # Tear down session at end of each request
    @app.teardown_appcontext
    def _remove_session(_exc=None):
        if _Session:
            _Session.remove()


def get_session() -> scoped_session:
    if _Session is None:
        raise RuntimeError("Database has not been initialised. Call init_db(app) first.")
    return _Session
