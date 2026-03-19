"""Audit API tests."""
from tests.conftest import get_token
from app.models.model import Model
from app.database import get_session


class TestAudits:
    def _create_model(self, client, token: str) -> int:
        res = client.post("/api/models/",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "Test Model", "model_type": "classification"})
        return res.get_json()["id"]

    def test_list_audits_empty(self, client):
        res = client.get("/api/audits/")
        assert res.status_code == 200
        assert res.get_json()["meta"]["total"] == 0

    def test_create_audit_success(self, client, regular_user):
        token = get_token(client, "testuser", "Str0ngP@ss!")
        model_id = self._create_model(client, token)
        res = client.post("/api/audits/",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "My Audit", "model_id": model_id})
        assert res.status_code == 201
        assert res.get_json()["status"] == "pending"

    def test_create_audit_nonexistent_model(self, client, regular_user):
        token = get_token(client, "testuser", "Str0ngP@ss!")
        res = client.post("/api/audits/",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "Audit", "model_id": 9999})
        assert res.status_code == 404

    def test_run_audit_fails_without_model_file(self, client, regular_user):
        """
        Running an audit on a model with no uploaded file must fail
        cleanly — not silently substitute a dummy model.
        """
        token = get_token(client, "testuser", "Str0ngP@ss!")
        model_id = self._create_model(client, token)
        create = client.post("/api/audits/",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "Audit", "model_id": model_id})
        audit_id = create.get_json()["id"]
        run = client.post(f"/api/audits/{audit_id}/run",
            headers={"Authorization": f"Bearer {token}"})
        assert run.status_code in (422, 500)
        assert run.get_json()["error"]["code"] in ("AUDIT_FAILED", "INTERNAL_ERROR")

    def test_export_requires_completed_audit(self, client, regular_user):
        token = get_token(client, "testuser", "Str0ngP@ss!")
        model_id = self._create_model(client, token)
        create = client.post("/api/audits/",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "Audit", "model_id": model_id})
        audit_id = create.get_json()["id"]
        res = client.get(f"/api/audits/{audit_id}/export?format=json")
        assert res.status_code == 409

    def test_audit_pagination(self, client, regular_user):
        token = get_token(client, "testuser", "Str0ngP@ss!")
        model_id = self._create_model(client, token)
        for i in range(5):
            client.post("/api/audits/",
                headers={"Authorization": f"Bearer {token}"},
                json={"name": f"Audit {i}", "model_id": model_id})
        res = client.get("/api/audits/?page=1&size=3")
        data = res.get_json()
        assert len(data["items"]) == 3
        assert data["meta"]["total"] == 5
        assert data["meta"]["pages"] == 2
