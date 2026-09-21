"""Shared pytest configuration.

Runs before any test module imports ``api.main``, so the API's module-level
history store is built against an in-memory database instead of the
developer's real ``data/history.db``.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("LLM_ROUTER_HISTORY_DB", ":memory:")
