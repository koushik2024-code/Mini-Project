"""Helpers for authenticating tests against the gated API.

Authentication is always on, so tests sign in the same way the app does -
by minting a real session token from the app's own SessionManager. Nothing
here bypasses the auth layer; it only skips the Google round trip, which is
the one part we cannot perform offline.
"""

from typing import Any

DEFAULT_SUB = "test-google-sub-1"
DEFAULT_EMAIL = "tester@example.com"


def make_user(
    google_sub: str = DEFAULT_SUB,
    email: str = DEFAULT_EMAIL,
    name: str = "Test User",
) -> dict[str, Any]:
    """Create (or refresh) a user directly in the API's user store."""
    from api import main

    return main.user_store.upsert(
        google_sub=google_sub, email=email, name=name, picture=""
    )


def auth_headers(
    google_sub: str = DEFAULT_SUB,
    email: str = DEFAULT_EMAIL,
    name: str = "Test User",
) -> dict[str, str]:
    """Bearer headers for a signed-in test user, creating them if needed."""
    from api import main

    user = make_user(google_sub, email, name)
    token = main.session_manager.issue(user["id"], user["email"])
    return {"Authorization": f"Bearer {token}"}


def auth_headers_for(user: dict[str, Any]) -> dict[str, str]:
    """Bearer headers for an existing user row."""
    from api import main

    token = main.session_manager.issue(user["id"], user["email"])
    return {"Authorization": f"Bearer {token}"}
