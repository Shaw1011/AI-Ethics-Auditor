"""Auth endpoint tests."""
import json
import pytest


class TestRegister:
    def test_register_success(self, client):
        res = client.post("/api/auth/register", json={
            "username": "newuser", "email": "new@test.com", "password": "Str0ngP@ss!"
        })
        assert res.status_code == 201
        data = res.get_json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["user"]["username"] == "newuser"

    def test_register_duplicate_username(self, client, regular_user):
        res = client.post("/api/auth/register", json={
            "username": "testuser", "email": "other@test.com", "password": "Str0ngP@ss!"
        })
        assert res.status_code == 409

    def test_register_invalid_email(self, client):
        res = client.post("/api/auth/register", json={
            "username": "user2", "email": "notanemail", "password": "Str0ngP@ss!"
        })
        assert res.status_code == 422

    def test_register_password_too_short(self, client):
        res = client.post("/api/auth/register", json={
            "username": "user3", "email": "u3@test.com", "password": "123"
        })
        assert res.status_code == 422

    def test_register_invalid_username_chars(self, client):
        res = client.post("/api/auth/register", json={
            "username": "bad user!", "email": "bad@test.com", "password": "Str0ngP@ss!"
        })
        assert res.status_code == 422


class TestLogin:
    def test_login_success(self, client, regular_user):
        res = client.post("/api/auth/login", json={
            "username": "testuser", "password": "Str0ngP@ss!"
        })
        assert res.status_code == 200
        assert "access_token" in res.get_json()

    def test_login_wrong_password(self, client, regular_user):
        res = client.post("/api/auth/login", json={
            "username": "testuser", "password": "wrongpassword"
        })
        assert res.status_code == 401

    def test_login_nonexistent_user(self, client):
        res = client.post("/api/auth/login", json={
            "username": "ghost", "password": "Str0ngP@ss!"
        })
        assert res.status_code == 401

    def test_login_returns_same_error_for_wrong_user_and_wrong_pass(self, client, regular_user):
        """Prevent user enumeration — errors must be identical."""
        r1 = client.post("/api/auth/login", json={"username": "ghost", "password": "x"})
        r2 = client.post("/api/auth/login", json={"username": "testuser", "password": "x"})
        assert r1.get_json()["error"]["code"] == r2.get_json()["error"]["code"]


class TestLogout:
    def test_logout_revokes_token(self, client, regular_user):
        login = client.post("/api/auth/login", json={"username": "testuser", "password": "Str0ngP@ss!"})
        token = login.get_json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        logout = client.delete("/api/auth/logout", headers=headers)
        assert logout.status_code == 200
        # Token must now be rejected
        me = client.get("/api/auth/me", headers=headers)
        assert me.status_code == 401
