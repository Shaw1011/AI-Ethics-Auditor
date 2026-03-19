"""
pytest fixtures.

Uses a function-scoped app with a fresh in-memory SQLite database per test,
so each test is fully isolated with no shared state.
"""
from __future__ import annotations
import pytest
from app import create_app
from app.config import TestingConfig
from app.database import Base, get_session


@pytest.fixture()
def app():
    """Fresh Flask app + fresh in-memory DB for every single test."""
    application = create_app(TestingConfig)
    with application.app_context():
        yield application


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def db(app):
    yield get_session()


@pytest.fixture()
def regular_user(db):
    from app.models.user import User
    u = User(username="testuser", email="test@test.com", is_admin=False)
    u.set_password("Str0ngP@ss!")
    db.add(u)
    db.commit()
    return u


@pytest.fixture()
def admin_user(db):
    from app.models.user import User
    u = User(username="admin", email="admin@test.com", is_admin=True)
    u.set_password("Adm1nP@ss!")
    db.add(u)
    db.commit()
    return u


def get_token(client, username: str, password: str) -> str:
    res = client.post("/api/auth/login", json={"username": username, "password": password})
    return res.get_json()["access_token"]