"""Shared-password sessions and administrator-token checks."""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Optional

from fastapi import HTTPException, Request

SESSION_COOKIE_NAME = "jupiteriq_session"
SESSION_TTL_SECONDS = 12 * 60 * 60


def _auth_secrets() -> tuple[str, str]:
    password = os.getenv("CHAT_ACCESS_PASSWORD", "")
    session_secret = os.getenv("CHAT_SESSION_SECRET", "")
    if len(password) < 32 or len(session_secret) < 32:
        raise HTTPException(
            status_code=503,
            detail="Chat sign-in is not configured. Set the required authentication secrets.",
        )
    return password, session_secret


def chat_auth_is_configured() -> bool:
    return len(os.getenv("CHAT_ACCESS_PASSWORD", "")) >= 32 and len(
        os.getenv("CHAT_SESSION_SECRET", "")
    ) >= 32


def verify_login_password(candidate: str) -> bool:
    password, _ = _auth_secrets()
    return hmac.compare_digest(candidate.encode("utf-8"), password.encode("utf-8"))


def create_session_token() -> str:
    password, session_secret = _auth_secrets()
    timestamp = str(int(time.time()))
    password_fingerprint = hashlib.sha256(password.encode("utf-8")).hexdigest()
    message = f"v1:{timestamp}:{password_fingerprint}".encode("utf-8")
    signature = hmac.new(session_secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    return f"v1.{timestamp}.{signature}"


def validate_session_token(token: Optional[str]) -> bool:
    if not token or not chat_auth_is_configured():
        return False
    try:
        version, timestamp_text, supplied_signature = token.split(".", maxsplit=2)
        if version != "v1":
            return False
        timestamp = int(timestamp_text)
        now = int(time.time())
        if timestamp > now + 60 or now - timestamp > SESSION_TTL_SECONDS:
            return False
        password, session_secret = _auth_secrets()
        password_fingerprint = hashlib.sha256(password.encode("utf-8")).hexdigest()
        message = f"v1:{timestamp_text}:{password_fingerprint}".encode("utf-8")
        expected_signature = hmac.new(
            session_secret.encode("utf-8"), message, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(supplied_signature, expected_signature)
    except (ValueError, HTTPException):
        return False


def require_chat_session(request: Request) -> None:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not validate_session_token(token):
        raise HTTPException(
            status_code=401,
            detail="Sign in to continue.",
            headers={"WWW-Authenticate": "Cookie"},
        )


def require_source_admin(request: Request) -> None:
    expected = os.getenv("SOURCE_ADMIN_TOKEN", "")
    if len(expected) < 32:
        raise HTTPException(status_code=503, detail="Source management is not configured.")
    authorization = request.headers.get("Authorization", "")
    scheme, separator, supplied = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer" or not supplied:
        raise HTTPException(status_code=403, detail="Administrator access is required.")
    if not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(status_code=403, detail="Administrator access is required.")
