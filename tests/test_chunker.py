"""Tests for text chunking modules."""

import pytest

from src.index.chunker import chunk_by_paragraph, chunk_by_sentence, chunk_fixed_size


class TestChunkFixedSize:
    def test_short_text(self):
        chunks = chunk_fixed_size("Hello world", chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world"

    def test_long_text(self):
        text = "word " * 200  # 1000 chars
        chunks = chunk_fixed_size(text, chunk_size=250, overlap=50)
        assert len(chunks) > 1

    def test_empty_text(self):
        chunks = chunk_fixed_size("")
        assert len(chunks) == 0


class TestChunkBySentence:
    def test_single_sentence(self):
        chunks = chunk_by_sentence("This is one sentence.")
        assert len(chunks) == 1

    def test_multiple_sentences(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunk_by_sentence(text, max_chunk_size=50)
        assert len(chunks) >= 1

    def test_long_sentence(self):
        text = "This is a very long sentence " * 50
        chunks = chunk_by_sentence(text, max_chunk_size=100)
        assert len(chunks) >= 1


class TestChunkByParagraph:
    def test_paragraphs(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunk_by_paragraph(text)
        assert len(chunks) == 3

    def test_long_paragraph_fallback(self):
        text = "Long paragraph. " * 100
        chunks = chunk_by_paragraph(text, max_chunk_size=100)
        assert len(chunks) >= 1
