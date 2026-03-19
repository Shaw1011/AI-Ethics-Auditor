"""Model API tests."""
from tests.conftest import get_token


class TestModels:
    def test_list_models_empty(self, client):
        res = client.get("/api/models/")
        assert res.status_code == 200
        data = res.get_json()
        assert "items" in data
        assert data["meta"]["total"] == 0

    def test_create_model_requires_auth(self, client):
        res = client.post("/api/models/", json={"name": "m", "model_type": "classification"})
        assert res.status_code == 401

    def test_create_model_success(self, client, regular_user):
        token = get_token(client, "testuser", "Str0ngP@ss!")
        res = client.post("/api/models/",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "My Model", "model_type": "classification", "version": "1.0"})
        assert res.status_code == 201
        assert res.get_json()["name"] == "My Model"

    def test_create_model_invalid_type(self, client, regular_user):
        token = get_token(client, "testuser", "Str0ngP@ss!")
        res = client.post("/api/models/",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "m", "model_type": "invalid_type"})
        assert res.status_code == 422

    def test_get_model_not_found(self, client):
        res = client.get("/api/models/9999")
        assert res.status_code == 404

    def test_delete_model_forbidden_for_other_user(self, client, regular_user, admin_user):
        # admin creates a model
        admin_token = get_token(client, "admin", "Adm1nP@ss!")
        create = client.post("/api/models/",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"name": "Admin Model", "model_type": "classification"})
        model_id = create.get_json()["id"]

        # regular user tries to delete it
        user_token = get_token(client, "testuser", "Str0ngP@ss!")
        res = client.delete(f"/api/models/{model_id}",
            headers={"Authorization": f"Bearer {user_token}"})
        assert res.status_code == 403

    def test_upload_rejects_non_joblib(self, client, regular_user):
        token = get_token(client, "testuser", "Str0ngP@ss!")
        data = {"file": (b"fake pickle", "model.pkl", "application/octet-stream")}
        res = client.post("/api/models/upload",
            headers={"Authorization": f"Bearer {token}"},
            data={"name": "bad", "model_type": "classification",
                  "file": (b"data", "model.pkl")},
            content_type="multipart/form-data")
        assert res.status_code == 422
