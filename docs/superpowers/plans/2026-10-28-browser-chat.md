# Browser Chat Interface Implementation Plan

> **For agentic workers:** Follow the approved design in `docs/superpowers/specs/2026-10-28-browser-chat-design.md`. Keep the existing API behavior for clients that omit a session ID.

**Goal:** Replace the JSON root response with an accessible, interactive JupiterIQ chat page that talks to the existing query API.

**Architecture:** FastAPI serves a same-origin HTML page from `src/static/chat.html`. The page uses native browser JavaScript and `POST /api/v1/query`; an optional browser UUID session ID lets the page keep conversations separate while older API clients retain IP-derived IDs.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic, HTML, CSS, vanilla JavaScript; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-28-browser-chat-design.md`

## Global Constraints

- Preserve `/health`, `/docs`, `/redoc`, and all `/api/v1/*` routes.
- Keep provider credentials exclusively on the server.
- Treat model responses and source content as plain text in the browser.
- Do not add or run tests unless the user asks.
- Preserve unrelated existing local changes and publish to `mykrotechltd-dev/multi-agent-rag-chatbot`; do not use the local checkout's mismatched Git remote.

## Review Focus

- Whitespace-only composer input must not issue a request; review the submit handler's trim/empty check.
- Invalid browser session IDs must fail validation, while requests with no session ID must retain the IP fallback; review the Pydantic field and route selection.
- API or network failures must show readable feedback and re-enable sending; review the fetch error paths.
- Model output and source text containing HTML-like content must remain inert; review all rendering paths for `textContent` use and absence of unsafe HTML insertion.
- Missing or non-JSON error bodies must not break the chat; review response parsing and the generic fallback message.

---

## File Map

- Modify `src/querying/models.py`: add an optional, constrained `session_id` to `QueryRequest`.
- Modify `src/querying/routes.py`: honor a supplied browser session ID and keep the IP-derived fallback for existing clients.
- Modify `src/main.py`: serve the chat document from `GET /` as HTML.
- Create `src/static/chat.html`: accessible responsive layout, browser state, composer interactions, API calls, loading/errors, copied-tab ownership detection, and source disclosure.

## Task 1: Add the FastAPI Chat Page

**Files:**
- Modify: `src/querying/models.py`
- Modify: `src/querying/routes.py`
- Modify: `src/main.py`
- Create: `src/static/chat.html`

**Interfaces:**
- `QueryRequest.session_id: Optional[str]`, optional and constrained to `web_` plus a 36-character lowercase UUID.
- Query route chooses `request.session_id` when provided; otherwise it calls the existing `generate_session_id_from_ip(client_ip)`.
- Root route returns `FileResponse(src_path / "static" / "chat.html", media_type="text/html")`.
- Browser submits `{ query, session_id }` to `/api/v1/query` and consumes `content`, `agents_used`, `sources`, and `session_id`.

- [ ] Add the optional validated session ID field without changing the query or similarity fields.
- [ ] Use a supplied session ID in the query handler; preserve the current IP fallback and return the selected ID in `QueryResponse`.
- [ ] Create the single-file responsive chat page. Use accessible labels and focus styles; support prompt chips, Enter-to-send, Shift+Enter for newline, disabled/loading send state, inline failures, new chat, and sessionStorage transcript/session state. Use BroadcastChannel to detect copied tab IDs and rotate the duplicate to a fresh conversation. Render messages, agent names, and source details using safe text nodes.
- [ ] Replace the root JSON body with a `FileResponse` for the chat page; keep all other routes unchanged.
- [ ] Review the four files against the Review Focus list and confirm the UI asset path is relative to `main.py` so it resolves in the deployed Python app.
- [ ] Publish the feature and approved design/plan documents as one commit to GitHub `main` using the current main SHA as the parent; allow the connected Vercel project to deploy from that commit.

