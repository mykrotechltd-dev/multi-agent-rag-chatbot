"""Password sign-in endpoints for the FastAPI browser app."""

import os

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from security import (
    SESSION_COOKIE_NAME,
    SESSION_TTL_SECONDS,
    create_session_token,
    validate_session_token,
    verify_login_password,
)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
limiter = Limiter(key_func=get_remote_address)


class LoginRequest(BaseModel):
    password: str = Field(..., min_length=1, max_length=256)


def setup_auth_routes():
    @router.post("/login")
    @limiter.limit("5/minute")
    async def login(payload: LoginRequest, request: Request, response: Response):
        """Validate the shared password and set an HTTP-only browser session."""
        if not verify_login_password(payload.password):
            raise HTTPException(status_code=401, detail="The password is incorrect.")
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=create_session_token(),
            max_age=SESSION_TTL_SECONDS,
            httponly=True,
            secure=request.url.scheme == "https" or bool(os.getenv("VERCEL")),
            samesite="strict",
            path="/",
        )
        return {"authenticated": True}

    @router.post("/logout")
    async def logout(request: Request, response: Response):
        response.delete_cookie(
            key=SESSION_COOKIE_NAME,
            path="/",
            secure=request.url.scheme == "https" or bool(os.getenv("VERCEL")),
            httponly=True,
            samesite="strict",
        )
        return {"authenticated": False}

    @router.get("/session")
    async def session_status(request: Request):
        return {
            "authenticated": validate_session_token(
                request.cookies.get(SESSION_COOKIE_NAME)
            )
        }

    return router
