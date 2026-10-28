"""Persistent Qdrant storage for operator-uploaded source documents."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Optional
from uuid import uuid4

from qdrant_client import QdrantClient, models

from config import EMBEDDING_MODEL
from indexing.embeddings import initialize_embeddings_model
from indexing.source_files import ParsedSource, chunk_uploaded_source, parse_uploaded_file

logger = logging.getLogger(__name__)

EMBEDDING_BATCH_SIZE = 64
DEFAULT_COLLECTION = "uploaded_sources"
ALLOWED_SCOPES = {"all", "finance", "hr", "legal", "tech", "general_knowledge"}


class QdrantSourceError(RuntimeError):
    """A configuration or persistence error for uploaded sources."""


class QdrantNotConfigured(QdrantSourceError):
    """Raised when required Qdrant connection settings are missing."""


class DuplicateSourceError(QdrantSourceError):
    def __init__(self, filename: str):
        self.filename = filename
        super().__init__(f"This document is already indexed as {filename}.")


@dataclass(frozen=True)
class SourceSummary:
    source_id: str
    filename: str
    scope: str
    file_format: str
    chunk_count: int
    uploaded_at: str
    embedding_model: str


@dataclass(frozen=True)
class RetrievedSource:
    content: str
    metadata: dict[str, Any]
    similarity: float
    distance: float


class QdrantSourceStore:
    """Indexes, searches, lists, previews, and removes uploaded source chunks."""

    def __init__(self, url: str, api_key: Optional[str], collection_name: str):
        self.client = QdrantClient(url=url, api_key=api_key or None, timeout=15)
        self.collection_name = collection_name
        self._payload_indexes_ready = False
        self._embeddings = None

    @classmethod
    def from_environment(cls) -> "QdrantSourceStore":
        url = os.getenv("QDRANT_URL", "").strip()
        if not url:
            raise QdrantNotConfigured("QDRANT_URL is not configured.")
        api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
        collection = os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION).strip() or DEFAULT_COLLECTION
        return cls(url=url, api_key=api_key, collection_name=collection)

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = initialize_embeddings_model()
        return self._embeddings

    def _exists(self) -> bool:
        return self.client.collection_exists(self.collection_name)

    def _ensure_indexes(self) -> None:
        if self._payload_indexes_ready:
            return
        for field_name in ("kind", "source_id", "content_hash", "status", "scope", "embedding_model"):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        self._payload_indexes_ready = True

    def _ensure_collection(self, vector_size: int) -> None:
        if not self._exists():
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            self._payload_indexes_ready = False
            self._ensure_indexes()
            return

        info = self.client.get_collection(self.collection_name)
        vector_config = info.config.params.vectors
        if not isinstance(vector_config, models.VectorParams):
            raise QdrantSourceError(
                "The configured Qdrant collection uses named vectors. Use a collection with one unnamed cosine vector."
            )
        if vector_config.size != vector_size or vector_config.distance != models.Distance.COSINE:
            raise QdrantSourceError(
                "The Qdrant collection dimension or distance does not match the active embedding model. "
                "Choose an empty compatible collection or reindex its contents; existing data was not changed."
            )
        self._ensure_indexes()

    @staticmethod
    def _field_match(key: str, value: str) -> models.FieldCondition:
        return models.FieldCondition(key=key, match=models.MatchValue(value=value))

    def _filter_for_source(self, source_id: str) -> models.Filter:
        return models.Filter(
            must=[
                self._field_match("kind", "source_chunk"),
                self._field_match("source_id", source_id),
            ]
        )

    def _filter_for_duplicate(self, content_hash: str) -> models.Filter:
        return models.Filter(
            must=[
                self._field_match("kind", "source_chunk"),
                self._field_match("content_hash", content_hash),
            ]
        )

    def find_duplicate(self, content_hash: str) -> Optional[str]:
        if not self._exists():
            return None
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=self._filter_for_duplicate(content_hash),
            limit=1,
            with_payload=["filename"],
            with_vectors=False,
        )
        if not points:
            return None
        return str((points[0].payload or {}).get("filename") or "an existing source")

    def index_source(self, filename: Optional[str], raw: bytes, scope: str) -> SourceSummary:
        if scope not in ALLOWED_SCOPES:
            raise ValueError("Choose a valid source scope.")
        parsed: ParsedSource = parse_uploaded_file(filename, raw)
        chunks = chunk_uploaded_source(parsed, scope)
        duplicate_name = self.find_duplicate(parsed.content_hash)
        if duplicate_name:
            raise DuplicateSourceError(duplicate_name)

        source_id = str(uuid4())
        uploaded_at = datetime.now(timezone.utc).isoformat()
        point_ids: list[str] = []

        try:
            for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
                batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
                vectors = self.embeddings.embed_documents([chunk.page_content for chunk in batch])
                if not vectors or len(vectors) != len(batch):
                    raise QdrantSourceError("The embedding service returned an incomplete batch.")
                vector_size = len(vectors[0])
                if any(len(vector) != vector_size for vector in vectors):
                    raise QdrantSourceError("The embedding service returned inconsistent vector sizes.")
                self._ensure_collection(vector_size)

                points = []
                for offset, (chunk, vector) in enumerate(zip(batch, vectors)):
                    point_id = str(uuid4())
                    point_ids.append(point_id)
                    payload = {
                        "kind": "source_chunk",
                        "source_id": source_id,
                        "filename": parsed.filename,
                        "scope": scope,
                        "chunk_index": start + offset,
                        "uploaded_at": uploaded_at,
                        "content_hash": parsed.content_hash,
                        "embedding_model": EMBEDDING_MODEL,
                        "status": "pending",
                        "content": chunk.page_content,
                        "metadata": chunk.metadata,
                    }
                    points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))

                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                )

            source_filter = self._filter_for_source(source_id)
            self.client.set_payload(
                collection_name=self.collection_name,
                payload={"status": "ready"},
                points=models.FilterSelector(filter=source_filter),
                wait=True,
            )
        except Exception as exc:
            if point_ids:
                try:
                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=models.FilterSelector(filter=self._filter_for_source(source_id)),
                        wait=True,
                    )
                except Exception as cleanup_error:
                    logger.exception("Failed to clean up pending Qdrant source %s", source_id)
                    raise QdrantSourceError(
                        f"Indexing outcome is uncertain. Inspect or delete source {source_id} before retrying."
                    ) from cleanup_error
            if isinstance(exc, QdrantSourceError):
                raise
            raise QdrantSourceError("The source could not be indexed. No searchable chunks were kept.") from exc

        return SourceSummary(
            source_id=source_id,
            filename=parsed.filename,
            scope=scope,
            file_format=parsed.extension.lstrip("."),
            chunk_count=len(chunks),
            uploaded_at=uploaded_at,
            embedding_model=EMBEDDING_MODEL,
        )

    def _scroll(self, query_filter: models.Filter, payload_fields: list[str]):
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=256,
                offset=offset,
                with_payload=payload_fields,
                with_vectors=False,
            )
            yield from points
            if offset is None:
                break

    def list_sources(self) -> list[SourceSummary]:
        if not self._exists():
            return []
        query_filter = models.Filter(
            must=[
                self._field_match("kind", "source_chunk"),
                self._field_match("status", "ready"),
            ]
        )
        summaries: dict[str, dict[str, Any]] = {}
        fields = [
            "source_id",
            "filename",
            "scope",
            "uploaded_at",
            "embedding_model",
            "metadata",
            "status",
        ]
        for point in self._scroll(query_filter, fields):
            payload = point.payload or {}
            source_id = str(payload.get("source_id", ""))
            if not source_id:
                continue
            summary = summaries.setdefault(
                source_id,
                {
                    "source_id": source_id,
                    "filename": str(payload.get("filename") or "Untitled source"),
                    "scope": str(payload.get("scope") or "all"),
                    "uploaded_at": str(payload.get("uploaded_at") or ""),
                    "embedding_model": str(payload.get("embedding_model") or ""),
                    "file_format": str((payload.get("metadata") or {}).get("format") or ""),
                    "chunk_count": 0,
                },
            )
            summary["chunk_count"] += 1
        sources = [SourceSummary(**summary) for summary in summaries.values()]
        return sorted(sources, key=lambda source: source.uploaded_at, reverse=True)

    def list_chunks(self, source_id: str) -> Optional[list[dict[str, Any]]]:
        if not self._exists():
            return None
        query_filter = models.Filter(
            must=[
                self._field_match("kind", "source_chunk"),
                self._field_match("source_id", source_id),
                self._field_match("status", "ready"),
            ]
        )
        chunks = []
        for point in self._scroll(query_filter, ["chunk_index", "content", "metadata"]):
            payload = point.payload or {}
            chunks.append(
                {
                    "chunk_index": int(payload.get("chunk_index", 0)),
                    "content": str(payload.get("content", "")),
                    "metadata": payload.get("metadata") or {},
                }
            )
        if not chunks:
            return None
        return sorted(chunks, key=lambda chunk: chunk["chunk_index"])

    def delete_source(self, source_id: str) -> int:
        if not self._exists():
            return 0
        count = self.client.count(
            collection_name=self.collection_name,
            count_filter=self._filter_for_source(source_id),
            exact=True,
        ).count
        if not count:
            return 0
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=self._filter_for_source(source_id)),
            wait=True,
        )
        return int(count)

    def search(
        self,
        query_vector: list[float],
        agent_name: str,
        limit: int,
    ) -> list[RetrievedSource]:
        if not self._exists():
            return []
        self._ensure_collection(len(query_vector))
        query_filter = models.Filter(
            must=[
                self._field_match("kind", "source_chunk"),
                self._field_match("status", "ready"),
                self._field_match("embedding_model", EMBEDDING_MODEL),
            ],
            should=[
                self._field_match("scope", "all"),
                self._field_match("scope", agent_name),
            ],
        )
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        matches = []
        for point in response.points:
            payload = point.payload or {}
            content = payload.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            similarity = float(point.score)
            metadata = dict(payload.get("metadata") or {})
            metadata.update(
                {
                    "source": str(payload.get("filename") or "Uploaded source"),
                    "source_id": str(payload.get("source_id") or ""),
                    "scope": str(payload.get("scope") or "all"),
                    "chunk_index": int(payload.get("chunk_index", 0)),
                    "uploaded_at": str(payload.get("uploaded_at") or ""),
                }
            )
            matches.append(
                RetrievedSource(
                    content=content,
                    metadata=metadata,
                    similarity=similarity,
                    distance=1.0 - similarity,
                )
            )
        return matches


@lru_cache(maxsize=1)
def get_qdrant_source_store() -> QdrantSourceStore:
    """Create one reusable client per warm application process."""
    return QdrantSourceStore.from_environment()
