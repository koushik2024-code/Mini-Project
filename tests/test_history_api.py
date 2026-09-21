import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient

from api import main
from api.main import app
from db.history_store import HistoryConfig, HistoryStore
from tests.auth_helpers import auth_headers_for


@pytest.fixture
def store(monkeypatch):
    """Throwaway in-memory history store for the API to use."""
    s = HistoryStore(HistoryConfig(db_path=":memory:", max_entries=0))
    monkeypatch.setattr(main, "history_store", s)
    yield s
    s.close()


@pytest.fixture
def user():
    return main.user_store.upsert("sub-primary", "primary@example.com", "Primary")


@pytest.fixture
def other_user():
    return main.user_store.upsert("sub-other", "other@example.com", "Other")


@pytest.fixture
def client(store, user):
    """Signed-in client for the primary user."""
    return TestClient(app, headers=auth_headers_for(user))


def _seed(user_id, count=3):
    """Populate the patched store directly, without hitting the LLM."""
    for i in range(count):
        main.history_store.add(
            user_id=user_id,
            query=f"seeded query {i}",
            response=f"seeded response {i}",
            predicted_class="easy" if i % 2 == 0 else "hard",
            confidence=0.8,
            selected_model="qwen3:1.7b",
            fallback_used=False,
            latency_ms=100 + i,
        )


class TestAuthRequired:
    """Every history endpoint rejects unauthenticated callers"""

    @pytest.fixture
    def anon(self, store):
        return TestClient(app)

    def test_list_requires_auth(self, anon):
        assert anon.get("/history").status_code == 401

    def test_get_requires_auth(self, anon):
        assert anon.get("/history/1").status_code == 401

    def test_stats_requires_auth(self, anon):
        assert anon.get("/history/stats").status_code == 401

    def test_delete_requires_auth(self, anon):
        assert anon.delete("/history/1").status_code == 401

    def test_clear_requires_auth(self, anon):
        assert anon.delete("/history").status_code == 401

    def test_chat_requires_auth(self, anon):
        assert anon.post("/chat", json={"query": "hi"}).status_code == 401

    def test_models_requires_auth(self, anon):
        assert anon.get("/models").status_code == 401

    def test_health_stays_public(self, anon):
        # Health checks run before anyone can sign in
        assert anon.get("/health").status_code == 200

    def test_garbage_token_rejected(self, store):
        c = TestClient(app, headers={"Authorization": "Bearer not-a-real-token"})
        assert c.get("/history").status_code == 401

    def test_malformed_header_rejected(self, store):
        c = TestClient(app, headers={"Authorization": "Basic abc123"})
        assert c.get("/history").status_code == 401

    def test_token_for_deleted_user_rejected(self, store):
        ghost = main.user_store.upsert("sub-ghost", "ghost@example.com")
        headers = auth_headers_for(ghost)
        main.user_store._conn.execute("DELETE FROM users WHERE id = ?", (ghost["id"],))
        main.user_store._conn.commit()

        c = TestClient(app, headers=headers)
        assert c.get("/history").status_code == 401

    def test_401_sets_www_authenticate(self, anon):
        assert "www-authenticate" in {
            k.lower() for k in anon.get("/history").headers
        }


class TestUserIsolationApi:
    """One signed-in user must not reach another's history"""

    def test_list_only_shows_own_entries(self, store, user, other_user):
        _seed(user["id"], 2)
        _seed(other_user["id"], 3)

        mine = TestClient(app, headers=auth_headers_for(user))
        theirs = TestClient(app, headers=auth_headers_for(other_user))

        assert mine.get("/history").json()["total"] == 2
        assert theirs.get("/history").json()["total"] == 3

    def test_cannot_read_another_users_entry(self, store, user, other_user):
        _seed(other_user["id"], 1)
        their_id = (
            TestClient(app, headers=auth_headers_for(other_user))
            .get("/history").json()["entries"][0]["id"]
        )

        mine = TestClient(app, headers=auth_headers_for(user))
        assert mine.get(f"/history/{their_id}").status_code == 404

    def test_cannot_delete_another_users_entry(self, store, user, other_user):
        _seed(other_user["id"], 1)
        theirs = TestClient(app, headers=auth_headers_for(other_user))
        their_id = theirs.get("/history").json()["entries"][0]["id"]

        mine = TestClient(app, headers=auth_headers_for(user))
        assert mine.delete(f"/history/{their_id}").status_code == 404
        assert theirs.get("/history").json()["total"] == 1

    def test_clear_does_not_touch_another_user(self, store, user, other_user):
        _seed(user["id"], 2)
        _seed(other_user["id"], 2)

        mine = TestClient(app, headers=auth_headers_for(user))
        theirs = TestClient(app, headers=auth_headers_for(other_user))

        assert mine.delete("/history").json() == {"deleted": 2}
        assert theirs.get("/history").json()["total"] == 2

    def test_stats_are_per_user(self, store, user, other_user):
        _seed(user["id"], 1)
        _seed(other_user["id"], 3)

        mine = TestClient(app, headers=auth_headers_for(user))
        assert mine.get("/history/stats").json()["total"] == 1


class TestListHistory:
    """GET /history"""

    def test_empty_history(self, client):
        response = client.get("/history")

        assert response.status_code == 200
        data = response.json()
        assert data["entries"] == []
        assert data["total"] == 0
        assert data["limit"] == 50
        assert data["offset"] == 0

    def test_returns_entries_newest_first(self, client, user):
        _seed(user["id"], 3)
        data = client.get("/history").json()

        assert data["total"] == 3
        assert [e["query"] for e in data["entries"]] == [
            "seeded query 2",
            "seeded query 1",
            "seeded query 0",
        ]

    def test_entry_shape(self, client, user):
        _seed(user["id"], 1)
        entry = client.get("/history").json()["entries"][0]

        for key in [
            "id", "query", "response", "predicted_class", "confidence",
            "selected_model", "fallback_used", "latency_ms", "created_at"
        ]:
            assert key in entry, f"Missing key: {key}"

        assert isinstance(entry["fallback_used"], bool)
        assert isinstance(entry["latency_ms"], int)

    def test_limit_and_offset(self, client, user):
        _seed(user["id"], 5)
        data = client.get("/history?limit=2&offset=1").json()

        assert len(data["entries"]) == 2
        assert data["total"] == 5
        assert data["limit"] == 2
        assert data["offset"] == 1

    def test_search_filter(self, client, user):
        main.history_store.add(user_id=user["id"], query="binary search tree", predicted_class="medium")
        main.history_store.add(user_id=user["id"], query="hash map basics", predicted_class="easy")

        data = client.get("/history?q=binary").json()
        assert data["total"] == 1
        assert data["entries"][0]["query"] == "binary search tree"

    def test_blank_search_is_ignored(self, client, user):
        _seed(user["id"], 2)
        assert client.get("/history?q=%20%20").json()["total"] == 2

    def test_limit_out_of_range_rejected(self, client):
        assert client.get("/history?limit=0").status_code == 422
        assert client.get("/history?limit=500").status_code == 422

    def test_negative_offset_rejected(self, client):
        assert client.get("/history?offset=-1").status_code == 422


class TestGetHistoryEntry:
    """GET /history/{id}"""

    def test_returns_full_entry(self, client, user):
        _seed(user["id"], 1)
        entry_id = client.get("/history").json()["entries"][0]["id"]

        response = client.get(f"/history/{entry_id}")
        assert response.status_code == 200
        assert response.json()["response"] == "seeded response 0"

    def test_missing_entry_returns_404(self, client):
        response = client.get("/history/99999")
        assert response.status_code == 404
        assert "detail" in response.json()


class TestHistoryStats:
    """GET /history/stats"""

    def test_stats_route_not_shadowed_by_id_route(self, client, user):
        _seed(user["id"], 3)
        response = client.get("/history/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert data["by_class"] == {"easy": 2, "hard": 1}
        assert "avg_latency_ms" in data
        assert "avg_confidence" in data


class TestDeleteHistory:
    """DELETE /history and DELETE /history/{id}"""

    def test_delete_single_entry(self, client, user):
        _seed(user["id"], 2)
        entry_id = client.get("/history").json()["entries"][0]["id"]

        response = client.delete(f"/history/{entry_id}")
        assert response.status_code == 200
        assert response.json() == {"deleted": 1, "id": entry_id}
        assert client.get("/history").json()["total"] == 1

    def test_delete_missing_entry_returns_404(self, client):
        assert client.delete("/history/99999").status_code == 404

    def test_clear_all(self, client, user):
        _seed(user["id"], 4)

        response = client.delete("/history")
        assert response.status_code == 200
        assert response.json() == {"deleted": 4}
        assert client.get("/history").json()["total"] == 0

    def test_clear_empty_history(self, client):
        assert client.delete("/history").json() == {"deleted": 0}


class TestHistoryDisabled:
    """History endpoints when the feature flag is off"""

    @pytest.fixture
    def disabled_client(self, monkeypatch, user):
        monkeypatch.setattr(main, "history_store", None)
        return TestClient(app, headers=auth_headers_for(user))

    def test_list_returns_503(self, disabled_client):
        assert disabled_client.get("/history").status_code == 503

    def test_get_returns_503(self, disabled_client):
        assert disabled_client.get("/history/1").status_code == 503

    def test_delete_returns_503(self, disabled_client):
        assert disabled_client.delete("/history/1").status_code == 503

    def test_clear_returns_503(self, disabled_client):
        assert disabled_client.delete("/history").status_code == 503


class TestChatRecordsHistory:
    """POST /chat persists the search against the signed-in user"""

    def test_chat_saves_entry_and_returns_history_id(self, client):
        response = client.post("/chat", json={"query": "What is 2+2?"})
        assert response.status_code == 200

        data = response.json()
        assert "history_id" in data

        entries = client.get("/history").json()["entries"]
        assert len(entries) == 1
        assert entries[0]["id"] == data["history_id"]
        assert entries[0]["query"] == "What is 2+2?"
        assert entries[0]["response"] == data["response"]
        assert entries[0]["predicted_class"] == data["predicted_class"]
        assert entries[0]["selected_model"] == data["selected_model"]

    def test_entry_is_owned_by_the_caller(self, client, user, other_user):
        client.post("/chat", json={"query": "What is 2+2?"})

        theirs = TestClient(app, headers=auth_headers_for(other_user))
        assert theirs.get("/history").json()["total"] == 0

    def test_empty_query_is_not_recorded(self, client):
        assert client.post("/chat", json={"query": ""}).status_code == 400
        assert client.get("/history").json()["total"] == 0

    def test_chat_succeeds_when_history_write_fails(self, client, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(main.history_store, "add", boom)

        response = client.post("/chat", json={"query": "What is 2+2?"})
        assert response.status_code == 200
        assert "history_id" not in response.json()

    def test_chat_works_when_history_disabled(self, client, monkeypatch):
        monkeypatch.setattr(main, "history_store", None)

        response = client.post("/chat", json={"query": "What is 2+2?"})
        assert response.status_code == 200
        assert "history_id" not in response.json()
