"""Sentence embedding module.

Wraps sentence-transformers to generate embeddings for text chunks.
Uses all-MiniLM-L6-v2 by default — fast, small, and effective for retrieval.
"""

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_model(model_name: str = DEFAULT_MODEL):
    """Load the sentence-transformer model (cached)."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)
    return model


def embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed.
        model_name: Name of the sentence-transformers model.
        batch_size: Number of texts to process per batch.
        show_progress: Whether to show a progress bar.

    Returns:
        numpy array of shape (n_texts, embedding_dim).
    """
    model = _load_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
    )
    logger.info("Generated %d embeddings (dim=%d)", len(texts), embeddings.shape[1])
    return embeddings


def embed_query(query: str, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Generate an embedding for a single query string.

    Returns:
        numpy array of shape (embedding_dim,).
    """
    model = _load_model(model_name)
    embedding = model.encode([query], normalize_embeddings=True)
    return embedding[0]
