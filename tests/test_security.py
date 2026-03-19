"""Security-specific tests."""
from tests.conftest import get_token


class TestSecurity:
    def test_sql_injection_in_login(self, client):
        res = client.post("/api/auth/login", json={
            "username": "' OR 1=1 --", "password": "x"
        })
        assert res.status_code in (401, 422)

    def test_oversized_input_rejected(self, client):
        res = client.post("/api/auth/register", json={
            "username": "a" * 300, "email": "x@x.com", "password": "Str0ngP@ss!"
        })
        assert res.status_code == 422

    def test_missing_token_returns_401(self, client):
        res = client.post("/api/models/", json={"name": "m", "model_type": "classification"})
        assert res.status_code == 401

    def test_tampered_token_rejected(self, client, regular_user):
        token = get_token(client, "testuser", "Str0ngP@ss!")
        bad = token[:-10] + "tampered!!"
        res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {bad}"})
        assert res.status_code in (401, 422)

    def test_upload_rejects_oversized_name(self, client, regular_user):
        token = get_token(client, "testuser", "Str0ngP@ss!")
        res = client.post("/api/models/",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "x" * 200, "model_type": "classification"})
        assert res.status_code == 422

    def test_404_returns_json_not_html(self, client):
        res = client.get("/api/does-not-exist")
        assert res.content_type == "application/json"
        assert res.status_code == 404

    def test_error_envelope_structure(self, client):
        res = client.get("/api/models/9999")
        data = res.get_json()
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]
