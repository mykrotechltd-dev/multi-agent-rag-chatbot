# Browser Chat Interface Design

**Date:** 2026-10-28  
**Status:** Approved for implementation

## Intent

Give visitors to the deployed application a usable browser chat instead of the JSON service description currently shown at `/`. Keep the FastAPI service and existing API documentation available.

## Goals

- Show an accessible, responsive chat page at `/` that sends messages to `POST /api/v1/query`.
- Support a new chat, sample prompt chips, multiline input, send/loading/error states, and source details on assistant answers.
- Keep the conversation associated with a browser-generated per-tab session ID so separate browser tabs do not share the API's IP-based conversation context; detect copied tab storage and rotate the duplicate's ID.
- Preserve `/health`, `/docs`, `/redoc`, and all `/api/v1/*` routes.
- Render API-provided text as text, not executable HTML.

## Non-goals

- No separate React/Node build, accounts, authentication, server-side conversation database, streaming responses, or changes to agent behavior.
- No API key or other provider secret in browser code.

## Constraints

- The application is a Python FastAPI service deployed to Vercel; the query API already returns answer text, agents, sources, and a session ID.
- The local checkout contains unrelated uncommitted changes and points at a different Git remote than the user's GitHub repository. Implementation and publication must preserve those changes and target `mykrotechltd-dev/multi-agent-rag-chatbot`.
- Do not add or run tests unless the user asks; review changes and use a low-cost HTTP page check after deployment if publishing proceeds.

## Approaches

1. **FastAPI-served page at `/` (selected).** Add a self-contained HTML/CSS/JavaScript page and serve it from the existing app. It reuses same-origin API calls, avoids a second build pipeline, and replaces the JSON landing response with the requested chat.
2. **Separate React frontend.** This gives a conventional component build but adds a second frontend toolchain and deployment surface for a small single-page experience.

The user selected the FastAPI page. A separate `/chat` route would leave `/` showing JSON and therefore would not meet the requested browser entry point.

## Proposed behavior and implementation calls

- Serve a static chat document from the FastAPI root route; preserve the current health and API routes. The OpenAPI pages remain at `/docs` and `/redoc`.
- Use an original, responsive JupiterIQ chat layout with a clear header, welcome prompts, transcript, composer, send action, loading indicator, and inline errors.
- Keep messages in browser session storage, generate a per-tab UUID-based session ID, detect copied session storage with `BroadcastChannel`, and allow starting a fresh chat. If tab coordination is unavailable, start a fresh chat rather than reuse a potentially copied ID. Add an optional validated session ID to `QueryRequest`; when absent, retain the current IP-derived session behavior for existing API clients.
- Show agent names and expandable source content returned by the query endpoint. Treat all response content as plain text to avoid injecting HTML from model output.
- Use native HTML, CSS, and JavaScript with no new runtime dependencies. Keep frontend assets under `src` and resolve their path relative to `main.py` for Vercel.
- Send normal API failures to the chat as readable errors, keep the composer usable after a failure, and disable duplicate sends while a request is running.

## Approval requested

Approve this design to proceed with the implementation plan and code changes, or request a change to its behavior or scope.

