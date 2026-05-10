"""ChromaDB vector store wrapper.

Manages the lifecycle of the vector database: creating collections,
adding documents with embeddings, and querying for similar content.
"""

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "property_knowledge"
CHROMA_DIR = Path("data/chroma")


def get_client(persist_dir: Path | str | None = None) -> chromadb.ClientAPI:
    """Get or create a ChromaDB client with persistent storage."""
    persist_path = Path(persist_dir or CHROMA_DIR)
    persist_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(persist_path),
        settings=Settings(anonymized_telemetry=False),
    )
    return client


def get_or_create_collection(
    name: str = DEFAULT_COLLECTION,
    client: chromadb.ClientAPI | None = None,
) -> chromadb.Collection:
    """Get or create a ChromaDB collection."""
    if client is None:
        client = get_client()

    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Collection '%s': %d documents", name, collection.count())
    return collection


def add_documents(
    texts: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict] | None = None,
    ids: list[str] | None = None,
    collection_name: str = DEFAULT_COLLECTION,
    client: chromadb.ClientAPI | None = None,
):
    """Add documents with pre-computed embeddings to the vector store.

    Args:
        texts: Document texts.
        embeddings: Pre-computed embedding vectors.
        metadatas: Optional metadata for each document.
        ids: Optional document IDs (generated if not provided).
        collection_name: Target collection name.
        client: ChromaDB client instance.
    """
    collection = get_or_create_collection(collection_name, client)

    if ids is None:
        ids = [f"doc_{i}" for i in range(len(texts))]

    if metadatas is None:
        metadatas = [{} for _ in texts]

    # ChromaDB has a batch size limit — chunk if needed
    batch_size = 500
    for i in range(0, len(texts), batch_size):
        batch_end = min(i + batch_size, len(texts))
        collection.add(
            documents=texts[i:batch_end],
            embeddings=embeddings[i:batch_end],
            metadatas=metadatas[i:batch_end],
            ids=ids[i:batch_end],
        )

    logger.info("Added %d documents to collection '%s'", len(texts), collection_name)


def query_collection(
    query_embedding: list[float],
    n_results: int = 5,
    collection_name: str = DEFAULT_COLLECTION,
    client: chromadb.ClientAPI | None = None,
    where: dict | None = None,
) -> list[dict]:
    """Query the vector store for similar documents.

    Args:
        query_embedding: Query embedding vector.
        n_results: Number of results to return.
        collection_name: Collection to query.
        client: ChromaDB client instance.
        where: Optional metadata filter.

    Returns:
        List of result dicts with 'text', 'score', 'metadata'.
    """
    collection = get_or_create_collection(collection_name, client)

    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": min(n_results, collection.count()) if collection.count() > 0 else 1,
    }
    if where:
        query_params["where"] = where

    results = collection.query(**query_params)

    formatted = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            formatted.append(
                {
                    "text": doc,
                    "score": 1 - results["distances"][0][i],  # Convert distance to similarity
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "id": results["ids"][0][i] if results["ids"] else None,
                }
            )

    return formatted


def get_collection_stats(collection_name: str = DEFAULT_COLLECTION) -> dict:
    """Get statistics about a collection."""
    client = get_client()
    try:
        collection = client.get_collection(collection_name)
        return {
            "name": collection_name,
            "count": collection.count(),
            "status": "active",
        }
    except Exception:
        return {
            "name": collection_name,
            "count": 0,
            "status": "not_found",
        }
