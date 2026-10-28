"""Admin-only upload and source management API."""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from indexing.source_files import MAX_UPLOAD_BYTES
from querying.tools.qdrant_source_store import (
    ALLOWED_SCOPES,
    DuplicateSourceError,
    QdrantNotConfigured,
    QdrantSourceError,
    QdrantSourceStore,
    SourceSummary,
    get_qdrant_source_store,
)
from security import require_chat_session, require_source_admin

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/api/v1/sources",
    tags=["sources"],
    dependencies=[Depends(require_chat_session), Depends(require_source_admin)],
)


def _get_store() -> QdrantSourceStore:
    try:
        return get_qdrant_source_store()
    except QdrantNotConfigured as exc:
        raise HTTPException(status_code=503, detail="Qdrant source storage is not configured.") from exc
    except Exception as exc:
        logger.exception("Could not initialize Qdrant source storage")
        raise HTTPException(status_code=503, detail="Qdrant source storage is unavailable.") from exc


def _serialize_summary(source: SourceSummary) -> dict:
    return {
        "source_id": source.source_id,
        "filename": source.filename,
        "scope": source.scope,
        "file_format": source.file_format,
        "chunk_count": source.chunk_count,
        "uploaded_at": source.uploaded_at,
        "embedding_model": source.embedding_model,
    }


def _normalize_source_id(source_id: str) -> str:
    try:
        return str(UUID(source_id))
    except (ValueError, AttributeError) as exc:
        raise HTTPException(status_code=404, detail="Source not found.") from exc


@router.get("")
def list_sources():
    """List uploaded files and chunk counts."""
    try:
        store = _get_store()
        sources = store.list_sources()
    except Exception as exc:
        logger.exception("Failed to list Qdrant sources")
        raise HTTPException(status_code=503, detail="Sources could not be loaded from Qdrant.") from exc
    return {"sources": [_serialize_summary(source) for source in sources]}


@router.post("")
async def upload_source(
    file: UploadFile = File(...),
    scope: str = Form("all"),
):
    """Parse, chunk, embed, and persist a source document."""
    try:
        if scope not in ALLOWED_SCOPES:
            raise HTTPException(status_code=422, detail="Choose a valid source scope.")
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Files must be 4 MB or smaller.")
        store = _get_store()
        source = await run_in_threadpool(store.index_source, file.filename, raw, scope)
    except HTTPException:
        raise
    except DuplicateSourceError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except QdrantNotConfigured as exc:
        raise HTTPException(status_code=503, detail="Qdrant source storage is not configured.") from exc
    except QdrantSourceError as exc:
        logger.warning("Qdrant source indexing failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected source indexing failure")
        raise HTTPException(status_code=503, detail="The file could not be indexed. Try again later.") from exc
    finally:
        await file.close()
    return {"source": _serialize_summary(source)}


@router.get("/{source_id}/chunks")
def list_source_chunks(source_id: str):
    """Return text chunks for the selected source in document order."""
    normalized_id = _normalize_source_id(source_id)
    try:
        store = _get_store()
        chunks = store.list_chunks(normalized_id)
    except Exception as exc:
        logger.exception("Failed to list chunks for Qdrant source %s", normalized_id)
        raise HTTPException(status_code=503, detail="Source chunks could not be loaded.") from exc
    if chunks is None:
        raise HTTPException(status_code=404, detail="Source not found.")
    return {"source_id": normalized_id, "chunks": chunks}


@router.delete("/{source_id}")
def delete_source(source_id: str):
    """Delete every stored chunk for the selected source."""
    normalized_id = _normalize_source_id(source_id)
    try:
        store = _get_store()
        deleted_chunks = store.delete_source(normalized_id)
    except Exception as exc:
        logger.exception("Failed to delete Qdrant source %s", normalized_id)
        raise HTTPException(status_code=503, detail="The source could not be deleted.") from exc
    if deleted_chunks == 0:
        raise HTTPException(status_code=404, detail="Source not found.")
    return {"source_id": normalized_id, "deleted_chunks": deleted_chunks}
