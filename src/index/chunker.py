from __future__ import annotations
"""Text chunking strategies for RAG.

Splits documents into overlapping chunks suitable for embedding
and retrieval. Supports fixed-size, sentence-aware, and recursive splitting.
"""

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    chunk_id: str
    source: str
    metadata: dict | None = None


def chunk_fixed_size(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    source: str = "",
) -> list[Chunk]:
    """Split text into fixed-size chunks with overlap.

    Args:
        text: Input text.
        chunk_size: Target chunk size in characters.
        overlap: Number of overlapping characters between chunks.
        source: Source identifier for the chunks.

    Returns:
        List of Chunk objects.
    """
    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        if chunk_text.strip():
            chunks.append(
                Chunk(
                    text=chunk_text.strip(),
                    chunk_id=f"{source}::chunk_{chunk_idx}",
                    source=source,
                    metadata={"start": start, "end": end},
                )
            )
            chunk_idx += 1

        start += chunk_size - overlap

    return chunks


def chunk_by_sentence(
    text: str,
    max_chunk_size: int = 512,
    source: str = "",
) -> list[Chunk]:
    """Split text by sentences, grouping into chunks up to max_chunk_size.

    Respects sentence boundaries to avoid cutting mid-sentence.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk: list[str] = []
    current_size = 0
    chunk_idx = 0

    for sentence in sentences:
        sentence_size = len(sentence)

        if current_size + sentence_size > max_chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_id=f"{source}::chunk_{chunk_idx}",
                    source=source,
                )
            )
            chunk_idx += 1
            current_chunk = []
            current_size = 0

        current_chunk.append(sentence)
        current_size += sentence_size

    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(
            Chunk(
                text=chunk_text,
                chunk_id=f"{source}::chunk_{chunk_idx}",
                source=source,
            )
        )

    return chunks


def chunk_by_paragraph(
    text: str,
    max_chunk_size: int = 1024,
    source: str = "",
) -> list[Chunk]:
    """Split text by double newlines (paragraphs).

    Falls back to sentence splitting if a paragraph exceeds max_chunk_size.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    chunk_idx = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) <= max_chunk_size:
            chunks.append(
                Chunk(
                    text=para,
                    chunk_id=f"{source}::chunk_{chunk_idx}",
                    source=source,
                )
            )
            chunk_idx += 1
        else:
            # Fall back to sentence splitting for long paragraphs
            sub_chunks = chunk_by_sentence(para, max_chunk_size, source)
            for sc in sub_chunks:
                sc.chunk_id = f"{source}::chunk_{chunk_idx}"
                chunks.append(sc)
                chunk_idx += 1

    return chunks
