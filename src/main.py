"""FastAPI application setup for the multi-agent RAG chatbot."""

import os
import sys
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# Add src to path if not already there
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from querying import setup_query_routes
from querying.agents import Orchestrator
from querying.auth_routes import limiter, setup_auth_routes
from querying.sources_routes import router as sources_router
from security import require_chat_session, validate_session_token


def _configured_cors_origins() -> list[str]:
    return [
        origin.strip().rstrip("/")
        for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
        if origin.strip() and origin.strip() != "*"
    ]


def create_app() -> FastAPI:
    """Create and configure the signed-in chat and source manager."""
    app = FastAPI(
        title="JupiterIQ Multi-Agent RAG Chatbot",
        description="Multi-agent RAG chatbot for handling customer inquiries across departments",
        version="0.1.0",
    )

    # Same-origin browser access is the default. Cross-origin clients must be explicitly named.
    allowed_origins = _configured_cors_origins()
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Accept", "Authorization", "Content-Type"],
        )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    orchestrator = Orchestrator()
    app.include_router(setup_auth_routes())
    app.include_router(
        setup_query_routes(orchestrator),
        dependencies=[Depends(require_chat_session)],
    )
    app.include_router(sources_router)

    @app.get("/login", response_class=HTMLResponse)
    def login_page():
        return FileResponse(src_path / "static" / "login.html", media_type="text/html")

    @app.get("/", response_class=HTMLResponse)
    def read_root(request: Request):
        """Serve the chat only to a browser with a valid session cookie."""
        if not validate_session_token(request.cookies.get("jupiteriq_session")):
            return RedirectResponse(url="/login?next=%2F", status_code=303)
        return FileResponse(src_path / "static" / "chat.html", media_type="text/html")

    @app.get("/sources", response_class=HTMLResponse)
    def source_manager_page(request: Request):
        """Serve the source manager only after chat sign-in."""
        if not validate_session_token(request.cookies.get("jupiteriq_session")):
            return RedirectResponse(url="/login?next=%2Fsources", status_code=303)
        return FileResponse(src_path / "static" / "sources.html", media_type="text/html")

    @app.get("/health")
    @app.head("/health")
    def health_check():
        """Public liveness endpoint for hosting health checks."""
        return {"status": "ok"}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
