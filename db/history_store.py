"""SQL-backed store for user search (query) history.

Uses SQLite via the standard-library ``sqlite3`` driver, so no extra
dependency and no server is required - the whole history lives in a single
file alongside the rest of the project's file-based storage.

Every row belongs to a user (``user_id``), and every read and write is scoped
to one user, so one account can never see or delete another's searches.
"""

import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The table is created first, then migrated, then indexed. Indexing has to
# come last: on a database written before authentication existed the table
# already exists without user_id, so an index on that column would fail
# before the migration has a chance to add it.
SCHEMA_TABLE = """
CREATE TABLE IF NOT EXISTS search_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER,
    query           TEXT    NOT NULL,
    response        TEXT    NOT NULL DEFAULT '',
    predicted_class TEXT    NOT NULL,
    confidence      REAL    NOT NULL,
    selected_model  TEXT    NOT NULL,
    fallback_used   INTEGER NOT NULL DEFAULT 0,
    latency_ms      INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    NOT NULL
);
"""

SCHEMA_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_search_history_created_at
    ON search_history (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_search_history_user
    ON search_history (user_id, id DESC);
"""


@dataclass
class HistoryConfig:
    """Configuration for the search history store."""
    db_path: str = "data/history.db"
    # Keep only the N most recent entries per user (0 disables pruning)
    max_entries: int = 500


class HistoryStore:
    """Persist and query per-user search history in SQLite.

    A single connection is shared across threads (FastAPI runs sync endpoint
    handlers in a worker threadpool), guarded by a lock so writes stay
    serialized. Keeping one connection also lets tests use ``:memory:``.
    """

    def __init__(self, config: HistoryConfig | None = None):
        self.config = config or HistoryConfig()
        self._lock = threading.Lock()

        if self.config.db_path != ":memory:":
            Path(self.config.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self.config.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA_TABLE)
        self._migrate()
        self._conn.executescript(SCHEMA_INDEXES)
        self._conn.commit()
        logger.info(f"Search history store ready at {self.config.db_path}")

    def _migrate(self) -> None:
        """Bring a pre-auth database up to the current schema."""
        columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(search_history)")
        }

        if "user_id" not in columns:
            # Rows written before authentication existed have no owner. They
            # stay in the table but belong to nobody, so no account sees them.
            self._conn.execute("ALTER TABLE search_history ADD COLUMN user_id INTEGER")
            logger.info("Migrated search_history: added user_id column")

    # ------------------------------------------------------------------ write

    def add(
        self,
        user_id: int,
        query: str,
        response: str = "",
        predicted_class: str = "",
        confidence: float = 0.0,
        selected_model: str = "",
        fallback_used: bool = False,
        latency_ms: int = 0,
    ) -> dict[str, Any]:
        """Record one search for a user and return the stored row."""
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO search_history (
                    user_id, query, response, predicted_class, confidence,
                    selected_model, fallback_used, latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(user_id),
                    query,
                    response,
                    predicted_class,
                    float(confidence),
                    selected_model,
                    1 if fallback_used else 0,
                    int(latency_ms),
                    created_at,
                ),
            )
            entry_id = cursor.lastrowid
            self._prune_locked(int(user_id))
            self._conn.commit()

        return {
            "id": int(entry_id or 0),
            "user_id": int(user_id),
            "query": query,
            "response": response,
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "selected_model": selected_model,
            "fallback_used": bool(fallback_used),
            "latency_ms": int(latency_ms),
            "created_at": created_at,
        }

    def delete(self, user_id: int, entry_id: int) -> bool:
        """Delete one of this user's entries. True if a row was removed."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM search_history WHERE id = ? AND user_id = ?",
                (entry_id, int(user_id)),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def clear(self, user_id: int) -> int:
        """Delete all of this user's entries. Returns rows removed."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM search_history WHERE user_id = ?", (int(user_id),)
            )
            self._conn.commit()
            return cursor.rowcount

    # ------------------------------------------------------------------- read

    def get(self, user_id: int, entry_id: int) -> dict[str, Any] | None:
        """Fetch one of this user's entries by id."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM search_history WHERE id = ? AND user_id = ?",
                (entry_id, int(user_id)),
            ).fetchone()
        return self._to_dict(row) if row else None

    def list(
        self,
        user_id: int,
        limit: int = 50,
        offset: int = 0,
        search: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return this user's entries newest-first, optionally filtered."""
        sql = "SELECT * FROM search_history WHERE user_id = ?"
        params: list[Any] = [int(user_id)]

        if search:
            sql += " AND query LIKE ? ESCAPE '\\'"
            params.append(f"%{self._escape_like(search)}%")

        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([max(0, limit), max(0, offset)])

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._to_dict(row) for row in rows]

    def count(self, user_id: int, search: str | None = None) -> int:
        """Count this user's entries, honouring the same filter as ``list``."""
        sql = "SELECT COUNT(*) AS n FROM search_history WHERE user_id = ?"
        params: list[Any] = [int(user_id)]

        if search:
            sql += " AND query LIKE ? ESCAPE '\\'"
            params.append(f"%{self._escape_like(search)}%")

        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return int(row["n"])

    def stats(self, user_id: int) -> dict[str, Any]:
        """Aggregate counts per difficulty tier for one user, plus totals."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT predicted_class, COUNT(*) AS n
                FROM search_history
                WHERE user_id = ?
                GROUP BY predicted_class
                """,
                (int(user_id),),
            ).fetchall()
            totals = self._conn.execute(
                """
                SELECT COUNT(*) AS total,
                       COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
                       COALESCE(AVG(confidence), 0) AS avg_confidence
                FROM search_history
                WHERE user_id = ?
                """,
                (int(user_id),),
            ).fetchone()

        return {
            "total": int(totals["total"]),
            "avg_latency_ms": round(float(totals["avg_latency_ms"]), 1),
            "avg_confidence": round(float(totals["avg_confidence"]), 4),
            "by_class": {row["predicted_class"]: int(row["n"]) for row in rows},
        }

    def close(self) -> None:
        """Close the underlying connection."""
        with self._lock:
            self._conn.close()

    # --------------------------------------------------------------- internal

    def _prune_locked(self, user_id: int) -> None:
        """Drop this user's rows beyond ``max_entries``. Caller holds the lock."""
        if self.config.max_entries <= 0:
            return

        self._conn.execute(
            """
            DELETE FROM search_history
            WHERE user_id = ?
              AND id NOT IN (
                SELECT id FROM search_history
                WHERE user_id = ?
                ORDER BY id DESC LIMIT ?
              )
            """,
            (user_id, user_id, self.config.max_entries),
        )

    @staticmethod
    def _escape_like(value: str) -> str:
        """Escape LIKE wildcards so a search for '%' is literal."""
        return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    @staticmethod
    def _to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "query": row["query"],
            "response": row["response"],
            "predicted_class": row["predicted_class"],
            "confidence": row["confidence"],
            "selected_model": row["selected_model"],
            "fallback_used": bool(row["fallback_used"]),
            "latency_ms": row["latency_ms"],
            "created_at": row["created_at"],
        }
