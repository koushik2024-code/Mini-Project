import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import jwt
import pytest

from auth.google_auth import GoogleAuthConfig, GoogleAuthError, GoogleVerifier
from auth.session import SessionConfig, SessionError, SessionManager

SECRET = "t" * 48
OTHER_SECRET = "z" * 48


@pytest.fixture
def manager():
    return SessionManager(SessionConfig(secret=SECRET, ttl_hours=1))


class TestSessionIssuing:
    """Minting session tokens"""

    def test_issue_returns_a_jwt(self, manager):
        token = manager.issue(1, "a@example.com")
        assert token.count(".") == 2

    def test_roundtrip(self, manager):
        claims = manager.verify(manager.issue(42, "a@example.com"))
        assert claims == {"user_id": 42, "email": "a@example.com"}

    def test_token_carries_expiry_and_issuer(self, manager):
        decoded = jwt.decode(
            manager.issue(1, "a@example.com"),
            SECRET,
            algorithms=["HS256"],
            issuer="llm-router",
        )
        assert decoded["iss"] == "llm-router"
        assert decoded["exp"] > decoded["iat"]

    def test_ttl_seconds(self):
        assert SessionManager(SessionConfig(secret=SECRET, ttl_hours=24)).ttl_seconds == 86400


class TestSessionSecret:
    """Signing key handling"""

    def test_missing_secret_generates_a_random_one(self):
        first = SessionManager(SessionConfig(secret=""))
        second = SessionManager(SessionConfig(secret=""))

        # A fixed fallback would let anyone forge sessions, so each instance
        # must get its own unpredictable key
        assert first.config.secret != second.config.secret
        assert len(first.config.secret) >= 32

    def test_token_from_one_instance_fails_on_another(self):
        first = SessionManager(SessionConfig(secret=""))
        second = SessionManager(SessionConfig(secret=""))

        with pytest.raises(SessionError):
            second.verify(first.issue(1, "a@example.com"))

    def test_short_secret_is_rejected(self):
        with pytest.raises(ValueError, match="at least 32 bytes"):
            SessionManager(SessionConfig(secret="tooshort"))

    def test_exactly_32_bytes_is_accepted(self):
        SessionManager(SessionConfig(secret="x" * 32))


class TestSessionVerification:
    """Rejecting bad tokens"""

    def test_empty_token(self, manager):
        with pytest.raises(SessionError, match="Missing"):
            manager.verify("")

    def test_garbage_token(self, manager):
        with pytest.raises(SessionError, match="Invalid"):
            manager.verify("not-a-jwt")

    def test_token_signed_with_another_key(self, manager):
        forged = jwt.encode(
            {"sub": "1", "iss": "llm-router", "iat": int(time.time()),
             "exp": int(time.time()) + 3600},
            OTHER_SECRET,
            algorithm="HS256",
        )
        with pytest.raises(SessionError, match="Invalid"):
            manager.verify(forged)

    def test_expired_token(self, manager):
        expired = jwt.encode(
            {"sub": "1", "email": "a@example.com", "iss": "llm-router",
             "iat": int(time.time()) - 7200, "exp": int(time.time()) - 3600},
            SECRET,
            algorithm="HS256",
        )
        with pytest.raises(SessionError, match="expired"):
            manager.verify(expired)

    def test_wrong_issuer(self, manager):
        wrong = jwt.encode(
            {"sub": "1", "iss": "somebody-else", "iat": int(time.time()),
             "exp": int(time.time()) + 3600},
            SECRET,
            algorithm="HS256",
        )
        with pytest.raises(SessionError):
            manager.verify(wrong)

    def test_none_algorithm_is_rejected(self, manager):
        # The classic JWT downgrade attack: unsigned token claiming alg=none
        unsigned = jwt.encode(
            {"sub": "1", "iss": "llm-router", "iat": int(time.time()),
             "exp": int(time.time()) + 3600},
            key="",
            algorithm="none",
        )
        with pytest.raises(SessionError):
            manager.verify(unsigned)

    def test_missing_required_claims(self, manager):
        incomplete = jwt.encode({"sub": "1"}, SECRET, algorithm="HS256")
        with pytest.raises(SessionError):
            manager.verify(incomplete)

    def test_non_numeric_subject(self, manager):
        bad = jwt.encode(
            {"sub": "not-a-number", "iss": "llm-router",
             "iat": int(time.time()), "exp": int(time.time()) + 3600},
            SECRET,
            algorithm="HS256",
        )
        with pytest.raises(SessionError, match="subject"):
            manager.verify(bad)


class TestGoogleVerifier:
    """Google ID token verification"""

    def test_not_configured_without_client_id(self):
        assert GoogleVerifier(GoogleAuthConfig(client_id="")).configured is False

    def test_configured_with_client_id(self):
        assert GoogleVerifier(GoogleAuthConfig(client_id="abc.apps.googleusercontent.com")).configured

    def test_empty_credential_rejected(self):
        verifier = GoogleVerifier(GoogleAuthConfig(client_id="abc"))
        with pytest.raises(GoogleAuthError, match="Missing"):
            verifier.verify("")

    def test_unconfigured_server_rejects_sign_in(self):
        verifier = GoogleVerifier(GoogleAuthConfig(client_id=""))
        with pytest.raises(GoogleAuthError, match="not configured"):
            verifier.verify("some-token")

    def test_valid_token_returns_profile(self):
        verifier = GoogleVerifier(GoogleAuthConfig(client_id="abc"))
        claims = {
            "iss": "https://accounts.google.com",
            "sub": "google-123",
            "email": "user@gmail.com",
            "email_verified": True,
            "name": "Real User",
            "picture": "http://pic",
        }

        with patch("auth.google_auth.google_id_token.verify_oauth2_token", return_value=claims):
            profile = verifier.verify("token")

        assert profile == {
            "google_sub": "google-123",
            "email": "user@gmail.com",
            "name": "Real User",
            "picture": "http://pic",
        }

    def test_bad_signature_is_rejected(self):
        verifier = GoogleVerifier(GoogleAuthConfig(client_id="abc"))

        with patch(
            "auth.google_auth.google_id_token.verify_oauth2_token",
            side_effect=ValueError("Token signature is invalid"),
        ):
            with pytest.raises(GoogleAuthError, match="Invalid Google credential"):
                verifier.verify("token")

    def test_token_for_another_audience_is_rejected(self):
        verifier = GoogleVerifier(GoogleAuthConfig(client_id="abc"))

        with patch(
            "auth.google_auth.google_id_token.verify_oauth2_token",
            side_effect=ValueError("Token has wrong audience"),
        ):
            with pytest.raises(GoogleAuthError):
                verifier.verify("token")

    def test_untrusted_issuer_is_rejected(self):
        verifier = GoogleVerifier(GoogleAuthConfig(client_id="abc"))
        claims = {
            "iss": "https://evil.example.com",
            "sub": "google-123",
            "email": "user@gmail.com",
            "email_verified": True,
        }

        with patch("auth.google_auth.google_id_token.verify_oauth2_token", return_value=claims):
            with pytest.raises(GoogleAuthError, match="issuer"):
                verifier.verify("token")

    def test_unverified_email_is_rejected(self):
        verifier = GoogleVerifier(GoogleAuthConfig(client_id="abc"))
        claims = {
            "iss": "accounts.google.com",
            "sub": "google-123",
            "email": "user@gmail.com",
            "email_verified": False,
        }

        with patch("auth.google_auth.google_id_token.verify_oauth2_token", return_value=claims):
            with pytest.raises(GoogleAuthError, match="not verified"):
                verifier.verify("token")

    def test_unverified_email_allowed_when_check_disabled(self):
        verifier = GoogleVerifier(
            GoogleAuthConfig(client_id="abc", require_verified_email=False)
        )
        claims = {
            "iss": "accounts.google.com",
            "sub": "google-123",
            "email": "user@gmail.com",
            "email_verified": False,
        }

        with patch("auth.google_auth.google_id_token.verify_oauth2_token", return_value=claims):
            assert verifier.verify("token")["email"] == "user@gmail.com"

    def test_missing_email_is_rejected(self):
        verifier = GoogleVerifier(GoogleAuthConfig(client_id="abc"))
        claims = {"iss": "accounts.google.com", "sub": "google-123"}

        with patch("auth.google_auth.google_id_token.verify_oauth2_token", return_value=claims):
            with pytest.raises(GoogleAuthError, match="no email"):
                verifier.verify("token")

    def test_missing_subject_is_rejected(self):
        verifier = GoogleVerifier(GoogleAuthConfig(client_id="abc"))
        claims = {
            "iss": "accounts.google.com",
            "email": "user@gmail.com",
            "email_verified": True,
            "sub": "",
        }

        with patch("auth.google_auth.google_id_token.verify_oauth2_token", return_value=claims):
            with pytest.raises(GoogleAuthError, match="subject"):
                verifier.verify("token")
