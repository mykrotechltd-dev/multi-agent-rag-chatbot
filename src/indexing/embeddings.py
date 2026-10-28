"""Initialize embeddings and vector stores (Chroma/Faiss) for document chunks."""

import os
import shutil
import stat
import tempfile
import uuid
from pathlib import Path
from typing import List, Union

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()

from config import (
    EMBEDDING_MODEL,
    VECTOR_STORE_TYPE,
    VECTOR_STORE_PATH,
)
from utils.model_provider import get_model_api_config


def initialize_embeddings_model():
    """
    Initialize the configured OpenAI-compatible embeddings model.
    
    Returns:
        Initialized OpenAIEmbeddings model.
    
    Raises:
        ValueError: If no model API key is set.
    """
    api_key, base_url = get_model_api_config()
    if not api_key:
        raise ValueError(
            "No model API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY."
        )

    print(f"Using embeddings API at {base_url or 'https://api.openai.com/v1'}")
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=api_key,
        openai_api_base=base_url,
    )


def _get_writable_chroma_directory(
    source_directory: Path,
    handbook_name: str,
) -> Path:
    """Copy a bundled Chroma store to Vercel's writable temporary filesystem.

    Vercel function bundles are read-only. Chroma opens its SQLite database in
    read/write mode, so the deployed index must be copied to /tmp before use.
    The source index is never modified.
    """
    if not (os.getenv("VERCEL") or os.getenv("VERCEL_ENV")):
        return source_directory

    target_directory = (
        Path(tempfile.gettempdir())
        / "multi-agent-rag-chatbot"
        / "vectorstore"
        / handbook_name
    )
    import fcntl

    target_directory = (
        Path(tempfile.gettempdir())
        / "multi-agent-rag-chatbot"
        / "vectorstore"
        / handbook_name
    )
    target_directory.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target_directory.with_name(f"{target_directory.name}.lock")

    # Concurrent cold starts can share /tmp. Lock each index copy to avoid
    # duplicate temporary copies exhausting the function's scratch space.
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if not target_directory.exists():
                staging_directory = target_directory.with_name(
                    f"{target_directory.name}-{uuid.uuid4().hex}.copying"
                )
                try:
                    shutil.copytree(source_directory, staging_directory)
                    os.replace(staging_directory, target_directory)
                except Exception:
                    shutil.rmtree(staging_directory, ignore_errors=True)
                    raise

            # copytree preserves source permissions, which may be read-only.
            for root, directories, files in os.walk(target_directory):
                os.chmod(root, os.stat(root).st_mode | stat.S_IWUSR)
                for name in files:
                    file_path = Path(root) / name
                    os.chmod(file_path, os.stat(file_path).st_mode | stat.S_IWUSR)
                for name in directories:
                    directory_path = Path(root) / name
                    os.chmod(directory_path, os.stat(directory_path).st_mode | stat.S_IWUSR)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return target_directory


def generate_embeddings(
    chunks,
    handbook_name: str,
    vector_store_type: str = None,
    base_persist_directory: Path = None,
) -> Union[Chroma, FAISS]:
    """
    Generate embeddings and create vector store for document chunks.
    Encapsulates all embedding and vector store creation logic.
    
    Args:
        chunks: List of Document chunks to embed and store.
        handbook_name: Name of the handbook (used for directory/collection naming).
        vector_store_type: Type of vector store ("chroma" or "faiss"). Defaults to config.
        base_persist_directory: Base directory for vector stores. Defaults to config.
    
    Returns:
        Initialized vector store with chunks and embeddings.
    
    Raises:
        ValueError: If OPENAI_API_KEY is not set.
    """
    # Initialize embeddings model (supports OpenAI and OpenRouter)
    embeddings_model = initialize_embeddings_model()
    
    # Create vector store
    if vector_store_type is None:
        vector_store_type = VECTOR_STORE_TYPE
    
    if base_persist_directory is None:
        base_persist_directory = VECTOR_STORE_PATH
    
    # Create handbook-specific directory
    persist_directory = base_persist_directory / handbook_name
    persist_directory.mkdir(parents=True, exist_ok=True)
    
    collection_name = handbook_name  # Use handbook name as collection name
    
    print(f"Generating embeddings and creating {vector_store_type.upper()} vector store for {handbook_name} with {len(chunks)} chunks...")
    
    if vector_store_type.lower() == "chroma":
        # Delete existing collection if it exists to avoid duplicates
        # Chroma.from_documents() appends to existing collections, causing duplicates
        import chromadb
        client = chromadb.PersistentClient(path=str(persist_directory))
        try:
            client.delete_collection(name=collection_name)
            print(f"Deleted existing collection '{collection_name}' to avoid duplicates")
        except Exception:
            # Collection doesn't exist, which is fine
            pass
        
        # Generate IDs matching JSONL format for consistency
        ids = [
            f"{chunk.metadata.get('handbook', handbook_name)}_{i}"
            for i, chunk in enumerate(chunks)
        ]
        
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=str(persist_directory),
            collection_name=collection_name,
            ids=ids,  # Use consistent IDs matching JSONL format
            collection_metadata={"hnsw:space": "cosine"},
        )
        print(f"Chroma vector store created and persisted to {persist_directory}")
        
    elif vector_store_type.lower() == "faiss":
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings_model,
        )
        # Save FAISS index
        faiss_path = persist_directory / "faiss_index"
        vector_store.save_local(str(faiss_path))
        print(f"FAISS vector store created and saved to {faiss_path}")
        
    else:
        raise ValueError(f"Unknown vector store type: {vector_store_type}. Use 'chroma' or 'faiss'")
    
    return vector_store


def load_vector_store(
    handbook_name: str,
    vector_store_type: str = None,
    base_persist_directory: Path = None,
) -> Union[Chroma, FAISS]:
    """
    Load an existing vector store for a specific handbook.
    
    Args:
        handbook_name: Name of the handbook.
        vector_store_type: Type of vector store ("chroma" or "faiss"). Defaults to config.
        base_persist_directory: Base directory for vector stores. Defaults to config.
    
    Returns:
        Loaded vector store.
    
    Raises:
        ValueError: If OPENAI_API_KEY is not set.
    """
    # Initialize embeddings model (supports OpenAI and OpenRouter)
    embeddings_model = initialize_embeddings_model()
    
    if vector_store_type is None:
        vector_store_type = VECTOR_STORE_TYPE
    
    if base_persist_directory is None:
        base_persist_directory = VECTOR_STORE_PATH
    
    # Handbook-specific directory
    persist_directory = base_persist_directory / handbook_name
    collection_name = handbook_name
    
    if vector_store_type.lower() == "chroma":
        # Check if directory exists
        if not persist_directory.exists():
            raise FileNotFoundError(
                f"Vector store directory does not exist: {persist_directory}. "
                f"Please run the indexing script to create vector stores."
            )
        
        persist_directory = _get_writable_chroma_directory(
            persist_directory,
            handbook_name,
        )

        vector_store = Chroma(
            persist_directory=str(persist_directory),
            embedding_function=embeddings_model,
            collection_name=collection_name,
            # collection_metadata={"hnsw:space": "cosine"}, # Use cosine similarity
        )
        
        # Check if collection has documents
        try:
            collection = vector_store._collection
            count = collection.count()
            if count == 0:
                print(f"WARNING: Vector store for {handbook_name} exists but is empty (0 documents)")
            else:
                print(f"Loaded Chroma vector store from {persist_directory} ({count} documents)")
        except Exception as e:
            print(f"WARNING: Could not verify document count for {handbook_name}: {e}")
            print(f"Loaded Chroma vector store from {persist_directory}")
        
    elif vector_store_type.lower() == "faiss":
        faiss_path = persist_directory / "faiss_index"
        vector_store = FAISS.load_local(
            str(faiss_path),
            embeddings_model,
            allow_dangerous_deserialization=True,
        )
        print(f"Loaded FAISS vector store from {faiss_path}")
        
    else:
        raise ValueError(f"Unknown vector store type: {vector_store_type}. Use 'chroma' or 'faiss'")
    
    return vector_store


def generate_embedding_for_text(text: str) -> List[float]:
    """
    Generate a single embedding for a text string (for queries).
    
    Args:
        text: Text to embed.
    
    Returns:
        Embedding vector.
    
    Raises:
        ValueError: If OPENAI_API_KEY is not set.
    """
    # Initialize embeddings model (supports OpenAI and OpenRouter)
    embeddings_model = initialize_embeddings_model()
    
    return embeddings_model.embed_query(text)

