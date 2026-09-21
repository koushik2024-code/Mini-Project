import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from db.user_store import UserConfig, UserStore


@pytest.fixture
def store():
    s = UserStore(UserConfig(db_path=":memory:"))
    yield s
    s.close()


class TestSchema:
    """Schema creation for the users table"""

    def test_table_created(self, store):
        row = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
        ).fetchone()
        assert row is not None

    def test_google_sub_is_unique(self, store):
        sql = store._conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='users'"
        ).fetchone()[0]
        assert "UNIQUE" in sql.upper()

    def test_persists_across_instances(self, tmp_path):
        db_path = str(tmp_path / "users.db")
        first = UserStore(UserConfig(db_path=db_path))
        first.upsert("sub-1", "a@example.com")
        first.close()

        second = UserStore(UserConfig(db_path=db_path))
        assert second.count() == 1
        second.close()


class TestUpsert:
    """Creating and refreshing users"""

    def test_creates_user(self, store):
        user = store.upsert("sub-1", "a@example.com", "Ada", "http://pic")

        assert user["id"] > 0
        assert user["google_sub"] == "sub-1"
        assert user["email"] == "a@example.com"
        assert user["name"] == "Ada"
        assert user["picture"] == "http://pic"
        assert user["created_at"]
        assert user["last_login_at"]

    def test_second_login_reuses_same_row(self, store):
        first = store.upsert("sub-1", "a@example.com", "Ada")
        second = store.upsert("sub-1", "a@example.com", "Ada")

        assert first["id"] == second["id"]
        assert store.count() == 1

    def test_profile_changes_are_picked_up(self, store):
        store.upsert("sub-1", "old@example.com", "Old Name", "old-pic")
        updated = store.upsert("sub-1", "new@example.com", "New Name", "new-pic")

        assert updated["email"] == "new@example.com"
        assert updated["name"] == "New Name"
        assert updated["picture"] == "new-pic"
        assert store.count() == 1

    def test_created_at_is_preserved_on_relogin(self, store):
        first = store.upsert("sub-1", "a@example.com")
        second = store.upsert("sub-1", "a@example.com")

        assert second["created_at"] == first["created_at"]

    def test_different_subs_are_different_users(self, store):
        a = store.upsert("sub-1", "a@example.com")
        b = store.upsert("sub-2", "b@example.com")

        assert a["id"] != b["id"]
        assert store.count() == 2

    def test_same_email_different_sub_are_separate(self, store):
        # Email can be reassigned; the Google subject id is the real identity
        a = store.upsert("sub-1", "shared@example.com")
        b = store.upsert("sub-2", "shared@example.com")

        assert a["id"] != b["id"]
        assert store.count() == 2


class TestLookup:
    """Reading users back"""

    def test_get_by_id(self, store):
        created = store.upsert("sub-1", "a@example.com", "Ada")
        assert store.get(created["id"])["email"] == "a@example.com"

    def test_get_missing_returns_none(self, store):
        assert store.get(99999) is None

    def test_get_by_sub(self, store):
        store.upsert("sub-1", "a@example.com")
        assert store.get_by_sub("sub-1")["email"] == "a@example.com"

    def test_get_by_missing_sub_returns_none(self, store):
        assert store.get_by_sub("nope") is None

    def test_count_on_empty_store(self, store):
        assert store.count() == 0


class TestSqlInjection:
    """Parameterised queries protect against injection"""

    def test_malicious_sub_is_stored_literally(self, store):
        malicious = "'; DROP TABLE users; --"
        user = store.upsert(malicious, "a@example.com")

        assert store.get_by_sub(malicious)["id"] == user["id"]
        assert store.count() == 1
