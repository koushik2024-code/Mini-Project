import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient

from api import main
from api.main import app
from auth.google_auth import GoogleAuthError
from tests.auth_helpers import auth_headers_for

GOOGLE_PROFILE = {
    "google_sub": "google-abc-123",
    "email": "signup@gmail.com",
    "name": "New User",
    "picture": "http://pic/avatar.png",
}


@pytest.fixture
def anon():
    return TestClient(app)


def _mock_verify(profile=None, error=None):
    """Patch the Google verifier so tests never call Google."""
    if error is not None:
        return patch.object(main.google_verifier, "verify", side_effect=error)
    return patch.object(main.google_verifier, "verify", return_value=profile or GOOGLE_PROFILE)


class TestGoogleLogin:
    """POST /auth/google"""

    def test_successful_login_returns_token_and_user(self, anon):
        with _mock_verify():
            response = anon.post("/auth/google", json={"credential": "fake-google-token"})

        assert response.status_code == 200
        data = response.json()

        assert data["token_type"] == "bearer"
        assert data["access_token"].count(".") == 2
        assert data["expires_in"] > 0
        assert data["user"]["email"] == "signup@gmail.com"
        assert data["user"]["name"] == "New User"
        assert data["user"]["id"] > 0

    def test_issued_token_works_on_protected_routes(self, anon):
        with _mock_verify():
            token = anon.post(
                "/auth/google", json={"credential": "fake"}
            ).json()["access_token"]

        c = TestClient(app, headers={"Authorization": f"Bearer {token}"})
        assert c.get("/auth/me").status_code == 200
        assert c.get("/models").status_code == 200

    def test_login_creates_the_user(self, anon):
        with _mock_verify({**GOOGLE_PROFILE, "google_sub": "brand-new-sub"}):
            anon.post("/auth/google", json={"credential": "fake"})

        assert main.user_store.get_by_sub("brand-new-sub") is not None

    def test_repeat_login_reuses_the_same_user(self, anon):
        with _mock_verify({**GOOGLE_PROFILE, "google_sub": "repeat-sub"}):
            first = anon.post("/auth/google", json={"credential": "fake"}).json()
            second = anon.post("/auth/google", json={"credential": "fake"}).json()

        assert first["user"]["id"] == second["user"]["id"]

    def test_rejected_google_token_returns_401(self, anon):
        with _mock_verify(error=GoogleAuthError("Invalid Google credential")):
            response = anon.post("/auth/google", json={"credential": "bad"})

        assert response.status_code == 401
        assert "detail" in response.json()

    def test_unconfigured_server_returns_401(self, anon):
        with _mock_verify(error=GoogleAuthError("Google sign-in is not configured on the server")):
            response = anon.post("/auth/google", json={"credential": "anything"})

        assert response.status_code == 401

    def test_missing_credential_is_rejected(self, anon):
        assert anon.post("/auth/google", json={}).status_code == 422

    def test_empty_credential_is_rejected(self, anon):
        assert anon.post("/auth/google", json={"credential": ""}).status_code == 422

    def test_login_endpoint_is_public(self, anon):
        # Must be reachable without a token, or nobody could ever sign in
        with _mock_verify():
            assert anon.post("/auth/google", json={"credential": "fake"}).status_code == 200


class TestCurrentUser:
    """GET /auth/me"""

    def test_returns_profile(self):
        user = main.user_store.upsert("sub-me", "me@example.com", "Me", "http://pic")
        c = TestClient(app, headers=auth_headers_for(user))

        response = c.get("/auth/me")
        assert response.status_code == 200
        assert response.json() == {
            "id": user["id"],
            "email": "me@example.com",
            "name": "Me",
            "picture": "http://pic",
        }

    def test_requires_auth(self, anon):
        assert anon.get("/auth/me").status_code == 401

    def test_rejects_forged_token(self, anon):
        c = TestClient(app, headers={"Authorization": "Bearer forged.token.here"})
        assert c.get("/auth/me").status_code == 401


class TestLogout:
    """POST /auth/logout"""

    def test_logout_succeeds(self):
        user = main.user_store.upsert("sub-out", "out@example.com")
        c = TestClient(app, headers=auth_headers_for(user))

        assert c.post("/auth/logout").status_code == 200

    def test_logout_requires_auth(self, anon):
        assert anon.post("/auth/logout").status_code == 401


class TestHealthReportsAuth:
    """GET /health exposes whether sign-in can work"""

    def test_health_includes_auth_configured(self, anon):
        data = anon.get("/health").json()
        assert "auth_configured" in data
        assert isinstance(data["auth_configured"], bool)
