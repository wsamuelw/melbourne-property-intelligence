"""RAG (Retrieval-Augmented Generation) pipeline.

Retrieves relevant documents from the vector store and generates
responses using an LLM with source citations.
"""

import logging
from dataclasses import dataclass, field

from src.index.embedder import embed_query
from src.index.vectorstore import query_collection
from src.query.llm_client import LLMClient, get_llm_client

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict] = field(default_factory=list)
    query: str = ""


SYSTEM_PROMPT = """You are a Melbourne property market expert. Answer questions about
Melbourne property auctions, suburbs, prices, and market trends using ONLY the provided
context. If the context doesn't contain enough information, say so.

Always cite your sources. Format citations as [Source: source_name].
Be specific with numbers and data when available."""


def build_rag_prompt(query: str, context_docs: list[dict]) -> str:
    """Build a RAG prompt with retrieved context.

    Args:
        query: User question.
        context_docs: Retrieved documents with scores.

    Returns:
        Formatted prompt string.
    """
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        source = doc.get("metadata", {}).get("source", "unknown")
        context_parts.append(f"[{i}] (Source: {source})\n{doc['text']}")

    context = "\n\n".join(context_parts)

    return f"""{SYSTEM_PROMPT}

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""


def rag_query(
    query: str,
    n_results: int = 5,
    llm_client: LLMClient | None = None,
) -> RAGResponse:
    """Execute a RAG query: retrieve context, then generate a response.

    Args:
        query: User question about Melbourne property market.
        n_results: Number of context documents to retrieve.
        llm_client: LLM client instance (auto-detected if None).

    Returns:
        RAGResponse with answer and source citations.
    """
    if llm_client is None:
        llm_client = get_llm_client()

    # Step 1: Embed the query
    logger.info("Embedding query: %s", query[:100])
    query_embedding = embed_query(query)

    # Step 2: Retrieve relevant documents
    logger.info("Retrieving %d relevant documents", n_results)
    context_docs = query_collection(
        query_embedding=query_embedding.tolist(),
        n_results=n_results,
    )

    if not context_docs:
        return RAGResponse(
            answer="No relevant information found in the knowledge base. "
            "Try running the data ingestion pipeline first.",
            sources=[],
            query=query,
        )

    # Step 3: Build prompt and generate response
    prompt = build_rag_prompt(query, context_docs)
    logger.info("Generating response with %d context documents", len(context_docs))

    answer = llm_client.generate(prompt)

    # Step 4: Format sources
    sources = [
        {
            "text": doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"],
            "score": round(doc["score"], 3),
            "source": doc.get("metadata", {}).get("source", "unknown"),
        }
        for doc in context_docs
    ]

    return RAGResponse(
        answer=answer,
        sources=sources,
        query=query,
    )
