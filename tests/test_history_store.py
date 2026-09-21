import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from db.history_store import HistoryConfig, HistoryStore

USER = 1
OTHER_USER = 2


@pytest.fixture
def store():
    """In-memory SQLite store, fresh for every test."""
    s = HistoryStore(HistoryConfig(db_path=":memory:", max_entries=0))
    yield s
    s.close()


def _add(store, query, user_id=USER, **overrides):
    payload = {
        "response": f"answer to {query}",
        "predicted_class": "easy",
        "confidence": 0.9,
        "selected_model": "qwen3:1.7b",
        "fallback_used": False,
        "latency_ms": 120,
    }
    payload.update(overrides)
    return store.add(user_id=user_id, query=query, **payload)


class TestSchema:
    """Schema creation for the search_history table"""

    def test_table_created(self, store):
        row = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='search_history'"
        ).fetchone()
        assert row is not None

    def test_user_id_column_exists(self, store):
        columns = {
            r["name"] for r in store._conn.execute("PRAGMA table_info(search_history)")
        }
        assert "user_id" in columns

    def test_indexes_created(self, store):
        names = {
            r["name"]
            for r in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
        }
        assert "idx_search_history_created_at" in names
        assert "idx_search_history_user" in names

    def test_init_is_idempotent(self, tmp_path):
        db_path = str(tmp_path / "history.db")
        first = HistoryStore(HistoryConfig(db_path=db_path))
        _add(first, "persisted query")
        first.close()

        second = HistoryStore(HistoryConfig(db_path=db_path))
        assert second.count(USER) == 1
        second.close()

    def test_creates_parent_directory(self, tmp_path):
        db_path = str(tmp_path / "nested" / "dir" / "history.db")
        s = HistoryStore(HistoryConfig(db_path=db_path))
        s.close()
        assert Path(db_path).exists()


class TestMigration:
    """Upgrading a database written before authentication existed"""

    def test_adds_user_id_to_legacy_table(self, tmp_path):
        db_path = str(tmp_path / "legacy.db")

        # Build the pre-auth schema by hand
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE search_history (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                query           TEXT    NOT NULL,
                response        TEXT    NOT NULL DEFAULT '',
                predicted_class TEXT    NOT NULL,
                confidence      REAL    NOT NULL,
                selected_model  TEXT    NOT NULL,
                fallback_used   INTEGER NOT NULL DEFAULT 0,
                latency_ms      INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT    NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO search_history "
            "(query, response, predicted_class, confidence, selected_model, created_at) "
            "VALUES ('legacy row', 'old answer', 'easy', 0.5, 'qwen3:1.7b', '2026-01-01T00:00:00+00:00')"
        )
        conn.commit()
        conn.close()

        store = HistoryStore(HistoryConfig(db_path=db_path))

        columns = {
            r["name"] for r in store._conn.execute("PRAGMA table_info(search_history)")
        }
        assert "user_id" in columns

        # The orphaned row belongs to nobody, so no user sees it
        assert store.count(USER) == 0
        assert store.list(USER) == []

        # ...but the row itself is preserved, not deleted
        total = store._conn.execute("SELECT COUNT(*) AS n FROM search_history").fetchone()
        assert total["n"] == 1
        store.close()

    def test_migration_is_idempotent(self, tmp_path):
        db_path = str(tmp_path / "twice.db")
        first = HistoryStore(HistoryConfig(db_path=db_path))
        _add(first, "row")
        first.close()

        second = HistoryStore(HistoryConfig(db_path=db_path))
        assert second.count(USER) == 1
        second.close()


class TestUserIsolation:
    """One user must never see or touch another's history"""

    def test_list_only_returns_own_entries(self, store):
        _add(store, "mine", user_id=USER)
        _add(store, "theirs", user_id=OTHER_USER)

        assert [e["query"] for e in store.list(USER)] == ["mine"]
        assert [e["query"] for e in store.list(OTHER_USER)] == ["theirs"]

    def test_count_is_per_user(self, store):
        _add(store, "a", user_id=USER)
        _add(store, "b", user_id=USER)
        _add(store, "c", user_id=OTHER_USER)

        assert store.count(USER) == 2
        assert store.count(OTHER_USER) == 1

    def test_cannot_get_another_users_entry(self, store):
        theirs = _add(store, "secret", user_id=OTHER_USER)
        assert store.get(USER, theirs["id"]) is None
        assert store.get(OTHER_USER, theirs["id"]) is not None

    def test_cannot_delete_another_users_entry(self, store):
        theirs = _add(store, "secret", user_id=OTHER_USER)

        assert store.delete(USER, theirs["id"]) is False
        assert store.get(OTHER_USER, theirs["id"]) is not None

    def test_clear_only_affects_own_entries(self, store):
        _add(store, "mine", user_id=USER)
        _add(store, "theirs", user_id=OTHER_USER)

        assert store.clear(USER) == 1
        assert store.count(USER) == 0
        assert store.count(OTHER_USER) == 1

    def test_stats_are_per_user(self, store):
        _add(store, "a", user_id=USER, predicted_class="easy")
        _add(store, "b", user_id=OTHER_USER, predicted_class="hard")
        _add(store, "c", user_id=OTHER_USER, predicted_class="hard")

        assert store.stats(USER)["by_class"] == {"easy": 1}
        assert store.stats(OTHER_USER)["by_class"] == {"hard": 2}

    def test_search_does_not_cross_users(self, store):
        _add(store, "binary search", user_id=OTHER_USER)
        assert store.list(USER, search="binary") == []

    def test_pruning_is_per_user(self):
        s = HistoryStore(HistoryConfig(db_path=":memory:", max_entries=2))
        for i in range(4):
            _add(s, f"mine {i}", user_id=USER)
        for i in range(4):
            _add(s, f"theirs {i}", user_id=OTHER_USER)

        # Each user keeps their own 2 most recent - one user's volume of
        # searches must not evict another's
        assert s.count(USER) == 2
        assert s.count(OTHER_USER) == 2
        s.close()


class TestAdd:
    """Recording searches"""

    def test_add_returns_stored_row(self, store):
        entry = _add(store, "What is 2+2?")

        assert entry["id"] > 0
        assert entry["user_id"] == USER
        assert entry["query"] == "What is 2+2?"
        assert entry["response"] == "answer to What is 2+2?"
        assert entry["predicted_class"] == "easy"
        assert entry["confidence"] == 0.9
        assert entry["selected_model"] == "qwen3:1.7b"
        assert entry["fallback_used"] is False
        assert entry["latency_ms"] == 120
        assert entry["created_at"]

    def test_fallback_used_roundtrips_as_bool(self, store):
        entry = _add(store, "hard one", fallback_used=True)
        assert entry["fallback_used"] is True
        assert store.get(USER, entry["id"])["fallback_used"] is True

    def test_ids_increment(self, store):
        first = _add(store, "first")
        second = _add(store, "second")
        assert second["id"] > first["id"]

    def test_duplicate_queries_are_separate_rows(self, store):
        _add(store, "same")
        _add(store, "same")
        assert store.count(USER) == 2

    def test_prunes_beyond_max_entries(self):
        s = HistoryStore(HistoryConfig(db_path=":memory:", max_entries=3))
        for i in range(6):
            _add(s, f"query {i}")

        assert s.count(USER) == 3
        assert [e["query"] for e in s.list(USER)] == ["query 5", "query 4", "query 3"]
        s.close()

    def test_max_entries_zero_disables_pruning(self, store):
        for i in range(10):
            _add(store, f"query {i}")
        assert store.count(USER) == 10


class TestListAndGet:
    """Reading history back"""

    def test_list_is_newest_first(self, store):
        _add(store, "oldest")
        _add(store, "middle")
        _add(store, "newest")

        assert [e["query"] for e in store.list(USER)] == ["newest", "middle", "oldest"]

    def test_list_empty(self, store):
        assert store.list(USER) == []

    def test_list_limit(self, store):
        for i in range(5):
            _add(store, f"query {i}")
        assert len(store.list(USER, limit=2)) == 2

    def test_list_offset(self, store):
        for i in range(5):
            _add(store, f"query {i}")

        page = store.list(USER, limit=2, offset=2)
        assert [e["query"] for e in page] == ["query 2", "query 1"]

    def test_list_search_filters(self, store):
        _add(store, "explain binary search trees")
        _add(store, "what is a hash map")
        _add(store, "binary heap complexity")

        results = store.list(USER, search="binary")
        assert len(results) == 2
        assert all("binary" in e["query"] for e in results)

    def test_list_search_is_case_insensitive(self, store):
        _add(store, "Explain Binary Search")
        assert len(store.list(USER, search="binary")) == 1

    def test_list_search_escapes_wildcards(self, store):
        _add(store, "100% coverage")
        _add(store, "unrelated query")

        assert len(store.list(USER, search="%")) == 1
        assert store.list(USER, search="%")[0]["query"] == "100% coverage"

    def test_list_search_escapes_underscore(self, store):
        _add(store, "snake_case naming")
        _add(store, "snakeXcase naming")

        results = store.list(USER, search="snake_case")
        assert len(results) == 1
        assert results[0]["query"] == "snake_case naming"

    def test_get_returns_entry(self, store):
        entry = _add(store, "findable")
        assert store.get(USER, entry["id"])["query"] == "findable"

    def test_get_missing_returns_none(self, store):
        assert store.get(USER, 99999) is None

    def test_count_honours_search(self, store):
        _add(store, "binary search")
        _add(store, "hash map")

        assert store.count(USER) == 2
        assert store.count(USER, search="binary") == 1


class TestDelete:
    """Removing history"""

    def test_delete_removes_entry(self, store):
        entry = _add(store, "delete me")

        assert store.delete(USER, entry["id"]) is True
        assert store.get(USER, entry["id"]) is None
        assert store.count(USER) == 0

    def test_delete_missing_returns_false(self, store):
        assert store.delete(USER, 99999) is False

    def test_delete_leaves_other_entries(self, store):
        keep = _add(store, "keep")
        drop = _add(store, "drop")

        store.delete(USER, drop["id"])
        assert [e["id"] for e in store.list(USER)] == [keep["id"]]

    def test_clear_removes_everything(self, store):
        for i in range(4):
            _add(store, f"query {i}")

        assert store.clear(USER) == 4
        assert store.count(USER) == 0

    def test_clear_on_empty_store(self, store):
        assert store.clear(USER) == 0


class TestStats:
    """Aggregate statistics"""

    def test_stats_on_empty_store(self, store):
        stats = store.stats(USER)
        assert stats["total"] == 0
        assert stats["by_class"] == {}
        assert stats["avg_latency_ms"] == 0
        assert stats["avg_confidence"] == 0

    def test_stats_counts_by_class(self, store):
        _add(store, "a", predicted_class="easy")
        _add(store, "b", predicted_class="easy")
        _add(store, "c", predicted_class="hard")

        stats = store.stats(USER)
        assert stats["total"] == 3
        assert stats["by_class"] == {"easy": 2, "hard": 1}

    def test_stats_averages(self, store):
        _add(store, "a", latency_ms=100, confidence=0.5)
        _add(store, "b", latency_ms=300, confidence=0.9)

        stats = store.stats(USER)
        assert stats["avg_latency_ms"] == 200.0
        assert stats["avg_confidence"] == 0.7


class TestSqlInjection:
    """Parameterised queries protect against injection"""

    def test_malicious_query_text_is_stored_literally(self, store):
        malicious = "'; DROP TABLE search_history; --"
        entry = _add(store, malicious)

        assert store.get(USER, entry["id"])["query"] == malicious
        assert store.count(USER) == 1

    def test_malicious_search_term_is_safe(self, store):
        _add(store, "benign")

        assert store.list(USER, search="'; DROP TABLE search_history; --") == []
        assert store.count(USER) == 1

    def test_table_survives_injection_attempt(self, store):
        _add(store, "'; DROP TABLE search_history; --")
        # A dropped table would raise OperationalError here
        try:
            store.list(USER)
        except sqlite3.OperationalError:
            pytest.fail("search_history table was dropped")
