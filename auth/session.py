"""Backend-issued session tokens.

Google's ID token proves who someone is, but it expires in about an hour and
is not ours to re-issue. After verifying it once we mint our own short JWT,
which the frontend sends as a Bearer token on every request.
"""

import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"
# PyJWT warns below this; HS256 keys shorter than the digest weaken the MAC
MIN_SECRET_BYTES = 32


class SessionError(Exception):
    """Raised when a session token is missing, expired, or forged."""


@dataclass
class SessionConfig:
    """Configuration for issuing and verifying session tokens."""
    secret: str = ""
    ttl_hours: int = 24
    issuer: str = "llm-router"


class SessionManager:
    """Issues and verifies the app's own session JWTs."""

    def __init__(self, config: SessionConfig | None = None):
        self.config = config or SessionConfig()

        if not self.config.secret:
            # Never fall back to a fixed default - a predictable signing key
            # lets anyone mint a session. A random key is safe; it just means
            # existing sessions stop working when the server restarts.
            self.config.secret = secrets.token_urlsafe(48)
            logger.warning(
                "No session secret configured - generated an ephemeral one. "
                "Set SESSION_SECRET to keep users signed in across restarts."
            )
        elif len(self.config.secret.encode()) < MIN_SECRET_BYTES:
            raise ValueError(
                f"Session secret must be at least {MIN_SECRET_BYTES} bytes; "
                f"got {len(self.config.secret.encode())}"
            )

    def issue(self, user_id: int, email: str) -> str:
        """Mint a session token for a signed-in user."""
        now = datetime.now(timezone.utc)
        payload = {
            "sub": str(user_id),
            "email": email,
            "iss": self.config.issuer,
            "iat": now,
            "exp": now + timedelta(hours=self.config.ttl_hours),
        }
        return jwt.encode(payload, self.config.secret, algorithm=ALGORITHM)

    def verify(self, token: str) -> dict[str, Any]:
        """Verify a session token and return its claims."""
        if not token:
            raise SessionError("Missing session token")

        try:
            claims = jwt.decode(
                token,
                self.config.secret,
                algorithms=[ALGORITHM],
                issuer=self.config.issuer,
                options={"require": ["exp", "iat", "sub", "iss"]},
            )
        except jwt.ExpiredSignatureError as exc:
            raise SessionError("Session has expired") from exc
        except jwt.InvalidTokenError as exc:
            raise SessionError("Invalid session token") from exc

        try:
            user_id = int(claims["sub"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SessionError("Session token has an invalid subject") from exc

        return {"user_id": user_id, "email": claims.get("email", "")}

    @property
    def ttl_seconds(self) -> int:
        """Token lifetime, for the client to schedule a refresh."""
        return self.config.ttl_hours * 3600
