"""``chunker`` 单元测试。

覆盖 token 感知分块器与去重逻辑：
- ``chunk_text``: 段落聚合、重叠、确定性 UUID、max_chunks 截断
- ``dedup_chunks``: 精确匹配（None embedder）与余弦相似度（真嵌入）
- ``_cosine_sim``: 正交/相反/零向量边界条件
- ``_sha256_hex``: 确定性哈希
"""

from __future__ import annotations

import math

import pytest

from persona_distillation.chunker import (
    Chunk,
    _cosine_sim,
    _sha256_hex,
    chunk_text,
    dedup_chunks,
)


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_empty_text_returns_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_single_paragraph_produces_one_chunk(self):
        text = "这是一段文字，没有段落分隔。"
        chunks = chunk_text(text, target_tokens=50)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].index == 0
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == len(text)
        assert chunks[0].uuid

    def test_multiple_paragraphs_merged(self):
        text = "第一段。\n\n第二段。\n\n第三段。"
        chunks = chunk_text(text, target_tokens=500)
        assert len(chunks) == 1
        assert "第一段" in chunks[0].text
        assert "第二段" in chunks[0].text
        assert "第三段" in chunks[0].text

    def test_deterministic_uuid(self):
        text = "确定性UUID测试文本。"
        c1 = chunk_text(text, target_tokens=50)
        c2 = chunk_text(text, target_tokens=50)
        assert len(c1) == len(c2)
        assert c1[0].uuid == c2[0].uuid
        assert c1[0].text == c2[0].text

    def test_different_text_produces_different_uuids(self):
        c1 = chunk_text("文本A", target_tokens=50)
        c2 = chunk_text("文本B", target_tokens=50)
        assert c1[0].uuid != c2[0].uuid

    def test_max_chunks_truncates(self):
        paragraphs = "\n\n".join([f"段落{i}。" * 20 for i in range(10)])
        chunks = chunk_text(paragraphs, target_tokens=50, max_chunks=3)
        assert len(chunks) <= 3

    def test_chunk_count_with_overlap(self):
        text = "\n\n".join([f"第{i}段内容。" * 10 for i in range(5)])
        chunks = chunk_text(text, target_tokens=30)
        assert len(chunks) >= 2
        for i in range(len(chunks)):
            assert chunks[i].index == i
            assert chunks[i].token_count > 0

    def test_chunk_has_source_info(self):
        text = "需要分块的文本内容。"
        chunks = chunk_text(text, target_tokens=50)
        source = chunks[0].with_source("corpus.txt", 1)
        assert source["source_file"] == "corpus.txt"
        assert source["chunk_index"] == 0
        assert source["total_chunks"] == 1
        assert "text" in source
        assert "uuid" in source


# ---------------------------------------------------------------------------
# _cosine_sim
# ---------------------------------------------------------------------------


class TestCosineSim:
    def test_identical_vectors(self):
        assert _cosine_sim([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_sim([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine_sim([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert _cosine_sim([0.0, 0.0], [1.0, 2.0]) == 0.0
        assert _cosine_sim([1.0, 2.0], [0.0, 0.0]) == 0.0

    def test_empty_vectors(self):
        assert _cosine_sim([], []) == 0.0

    def test_mismatched_length(self):
        assert _cosine_sim([1.0], [1.0, 2.0]) == 0.0

    def test_normalized_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.5, 0.5, 0.0]
        expected = 0.5 / math.sqrt(0.5)
        assert _cosine_sim(a, b) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _sha256_hex
# ---------------------------------------------------------------------------


class TestSha256Hex:
    def test_deterministic(self):
        h1 = _sha256_hex("hello")
        h2 = _sha256_hex("hello")
        assert h1 == h2

    def test_different_inputs(self):
        assert _sha256_hex("hello") != _sha256_hex("world")

    def test_length_is_64(self):
        assert len(_sha256_hex("test")) == 64


# ---------------------------------------------------------------------------
# dedup_chunks
# ---------------------------------------------------------------------------


class TestDedupChunks:
    def test_empty_list_returns_empty(self):
        assert dedup_chunks([]) == []

    def test_exact_match_dedup(self):
        chunks = [
            Chunk(text="same content", index=0, char_start=0, char_end=12, token_count=5),
            Chunk(text="same content", index=1, char_start=12, char_end=24, token_count=5),
            Chunk(text="different", index=2, char_start=24, char_end=33, token_count=5),
        ]
        result = dedup_chunks(chunks, embedder=None)
        assert len(result) == 2
        assert result[0].index == 0
        assert result[1].index == 2

    def test_near_duplicate_with_real_embedder(self):
        class _FakeEmbedder:
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                vectors = []
                for t in texts:
                    if "hello world there" in t:
                        vectors.append([0.99, 0.01])
                    elif "hello world" in t and "there" not in t:
                        vectors.append([1.0, 0.0])
                    else:
                        vectors.append([0.0, 1.0])
                return vectors

        chunks = [
            Chunk(text="hello world", index=0, char_start=0, char_end=11, token_count=5),
            Chunk(text="hello world there", index=1, char_start=11, char_end=28, token_count=7),
            Chunk(text="completely different topic", index=2, char_start=28, char_end=53, token_count=8),
        ]
        result = dedup_chunks(chunks, embedder=_FakeEmbedder(), threshold=0.95)
        assert len(result) == 2
        assert result[0].text == "hello world"
        assert result[1].text == "completely different topic"

    def test_no_duplicates_keeps_all(self):
        chunks = [
            Chunk(text="AAA", index=0, char_start=0, char_end=3, token_count=1),
            Chunk(text="BBB", index=1, char_start=3, char_end=6, token_count=1),
            Chunk(text="CCC", index=2, char_start=6, char_end=9, token_count=1),
        ]
        result = dedup_chunks(chunks, embedder=None)
        assert len(result) == 3

    def test_preserves_order(self):
        chunks = [
            Chunk(text="first", index=0, char_start=0, char_end=5, token_count=2),
            Chunk(text="first", index=1, char_start=5, char_end=10, token_count=2),
            Chunk(text="second", index=2, char_start=10, char_end=16, token_count=2),
        ]
        result = dedup_chunks(chunks, embedder=None)
        assert [c.text for c in result] == ["first", "second"]

    def test_embedder_exception_falls_back_to_exact(self):
        class _BrokenEmbedder:
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                raise RuntimeError("API down")

        chunks = [
            Chunk(text="same", index=0, char_start=0, char_end=4, token_count=1),
            Chunk(text="same", index=1, char_start=4, char_end=8, token_count=1),
            Chunk(text="diff", index=2, char_start=8, char_end=12, token_count=1),
        ]
        result = dedup_chunks(chunks, embedder=_BrokenEmbedder())
        assert len(result) == 2