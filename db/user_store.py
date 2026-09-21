"""SQL-backed store for authenticated Google users.

Shares the same SQLite file as the search history so a user row and their
searches stay together and can be joined.
"""

import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    google_sub    TEXT    NOT NULL UNIQUE,
    email         TEXT    NOT NULL,
    name          TEXT    NOT NULL DEFAULT '',
    picture       TEXT    NOT NULL DEFAULT '',
    created_at    TEXT    NOT NULL,
    last_login_at TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);
"""


@dataclass
class UserConfig:
    """Configuration for the user store."""
    db_path: str = "data/history.db"


class UserStore:
    """Persist Google-authenticated users.

    Identity is keyed on the Google subject id (``sub``), which is stable for
    an account. Email is stored for display and can change, so it is never
    used as the primary key.
    """

    def __init__(self, config: UserConfig | None = None):
        self.config = config or UserConfig()
        self._lock = threading.Lock()

        if self.config.db_path != ":memory:":
            Path(self.config.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self.config.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        logger.info(f"User store ready at {self.config.db_path}")

    def upsert(
        self,
        google_sub: str,
        email: str,
        name: str = "",
        picture: str = "",
    ) -> dict[str, Any]:
        """Create the user on first sign-in, refresh their profile after that."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO users (
                    google_sub, email, name, picture, created_at, last_login_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (google_sub) DO UPDATE SET
                    email         = excluded.email,
                    name          = excluded.name,
                    picture       = excluded.picture,
                    last_login_at = excluded.last_login_at
                """,
                (google_sub, email, name, picture, now, now),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM users WHERE google_sub = ?", (google_sub,)
            ).fetchone()

        return self._to_dict(row)

    def get(self, user_id: int) -> dict[str, Any] | None:
        """Fetch a user by primary key."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return self._to_dict(row) if row else None

    def get_by_sub(self, google_sub: str) -> dict[str, Any] | None:
        """Fetch a user by their Google subject id."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE google_sub = ?", (google_sub,)
            ).fetchone()
        return self._to_dict(row) if row else None

    def count(self) -> int:
        """Total registered users."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()
        return int(row["n"])

    def close(self) -> None:
        """Close the underlying connection."""
        with self._lock:
            self._conn.close()

    @staticmethod
    def _to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "google_sub": row["google_sub"],
            "email": row["email"],
            "name": row["name"],
            "picture": row["picture"],
            "created_at": row["created_at"],
            "last_login_at": row["last_login_at"],
        }
