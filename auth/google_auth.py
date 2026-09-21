"""Verification of Google Sign-In ID tokens.

The React frontend runs Google Identity Services, which hands it a signed ID
token (a JWT). That token is posted here; we verify its signature against
Google's public keys, check the audience matches our OAuth client, and pull
the user's profile out of the claims.
"""

import logging
from dataclasses import dataclass
from typing import Any

from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token

logger = logging.getLogger(__name__)

# Google always issues ID tokens under one of these issuers
VALID_ISSUERS = ("accounts.google.com", "https://accounts.google.com")


class GoogleAuthError(Exception):
    """Raised when a Google ID token cannot be trusted."""


@dataclass
class GoogleAuthConfig:
    """Configuration for Google Sign-In verification."""
    client_id: str = ""
    # Require Google to have verified the address before we trust it
    require_verified_email: bool = True


class GoogleVerifier:
    """Verifies Google ID tokens and extracts the user's profile."""

    def __init__(self, config: GoogleAuthConfig | None = None):
        self.config = config or GoogleAuthConfig()
        self._request = google_requests.Request()

    @property
    def configured(self) -> bool:
        """True when a Google OAuth client id is available."""
        return bool(self.config.client_id)

    def verify(self, credential: str) -> dict[str, Any]:
        """Verify an ID token and return the caller's profile.

        Raises GoogleAuthError if the token is missing, malformed, expired,
        issued for another application, or lacks a verified email.
        """
        if not credential:
            raise GoogleAuthError("Missing Google credential")

        if not self.configured:
            raise GoogleAuthError("Google sign-in is not configured on the server")

        try:
            claims = google_id_token.verify_oauth2_token(
                credential, self._request, self.config.client_id
            )
        except ValueError as exc:
            # Covers bad signature, wrong audience, and expired tokens
            raise GoogleAuthError(f"Invalid Google credential: {exc}") from exc

        if claims.get("iss") not in VALID_ISSUERS:
            raise GoogleAuthError("Invalid Google credential: unexpected issuer")

        email = claims.get("email", "")
        if not email:
            raise GoogleAuthError("Google account has no email address")

        if self.config.require_verified_email and not claims.get("email_verified"):
            raise GoogleAuthError("Google account email is not verified")

        google_sub = claims.get("sub", "")
        if not google_sub:
            raise GoogleAuthError("Google credential has no subject id")

        return {
            "google_sub": google_sub,
            "email": email,
            "name": claims.get("name", ""),
            "picture": claims.get("picture", ""),
        }
