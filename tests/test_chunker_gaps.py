"""chunker 去重逻辑的覆盖率缺口测试（Issue #18.a）。

`dedup_chunks` / `_cosine_sim` / `_sha256_hex` 在 commit 972d563（修复 16 个
GitHub issue）中新增，用于在 NER 之前去除重复 chunk，避免同一文本被多次识别
浪费 LLM 调用。此前无任何单元测试覆盖——本文件补齐：

- SHA-256 摘要的确定性
- 余弦相似度的边界条件（空向量 / 长度不匹配 / 零向量 / 正交 / 相同）
- dedup_chunks 三条路径：
  * 精确匹配（None / HashEmbeddings）—— 退化为 SHA-256 完全匹配
  * 真嵌入路径 —— cosine ≥ threshold 视为重复
  * embedder 调用失败 —— 安全降级到精确匹配

所有用例确定性、离线、无 LLM 依赖。跑法：``python -m pytest tests/test_chunker_gaps.py``。
"""
from __future__ import annotations

from persona_distillation.chunker import (
    Chunk,
    _cosine_sim,
    _sha256_hex,
    dedup_chunks,
)
from persona_distillation.intake.embedder import HashEmbeddings


# ---------------------------------------------------------------------------
# _sha256_hex
# ---------------------------------------------------------------------------
def test_sha256_hex_deterministic() -> None:
    """相同文本产生相同摘要；不同文本产生不同摘要；与编码无关的稳定性。"""
    assert _sha256_hex("荒川老师") == _sha256_hex("荒川老师")
    assert _sha256_hex("荒川老师") != _sha256_hex("荒川老师 ")
    # 空串也是合法输入
    assert isinstance(_sha256_hex(""), str) and len(_sha256_hex("")) == 64


# ---------------------------------------------------------------------------
# _cosine_sim
# ---------------------------------------------------------------------------
def test_cosine_sim_edge_cases() -> None:
    """余弦相似度边界：空/长度不匹配返回 0；相同向量返回 1；正交返回 0。"""
    # 空向量
    assert _cosine_sim([], [1.0, 2.0]) == 0.0
    assert _cosine_sim([1.0], []) == 0.0
    # 长度不匹配
    assert _cosine_sim([1.0, 2.0], [1.0]) == 0.0
    # 相同向量 → 1.0
    assert _cosine_sim([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0
    # 正交向量 → 0.0
    assert _cosine_sim([1.0, 0.0], [0.0, 1.0]) == 0.0
    # 零向量不应抛错（norm 兜底为 1.0，结果为 0.0）
    assert _cosine_sim([0.0, 0.0], [1.0, 1.0]) == 0.0


# ---------------------------------------------------------------------------
# dedup_chunks — 精确匹配路径（None / HashEmbeddings）
# ---------------------------------------------------------------------------
def _chunk(text: str, index: int = 0) -> Chunk:
    return Chunk(text=text, index=index, char_start=0, char_end=len(text), token_count=len(text))


def test_dedup_chunks_empty() -> None:
    """空列表入参 → 空列表出参。"""
    assert dedup_chunks([]) == []


def test_dedup_chunks_exact_match_removes_duplicates() -> None:
    """None embedder：SHA-256 精确匹配，去重保留首次出现，保持原顺序。"""
    a = _chunk("荒川老师点点头。", 0)
    b = _chunk("小明跑过来。", 1)
    a2 = _chunk("荒川老师点点头。", 2)  # 与 a 文本完全相同
    out = dedup_chunks([a, b, a2], embedder=None)
    assert out == [a, b]
    # 近似但非完全相同的不应被去重
    c = _chunk("荒川老师点点头!。", 3)
    assert dedup_chunks([a, c], embedder=None) == [a, c]


def test_dedup_chunks_hash_embeddings_uses_exact_match() -> None:
    """HashEmbeddings 是伪嵌入，cosine 无意义，必须走精确匹配路径。"""
    a = _chunk("同一段文本。", 0)
    b = _chunk("另一段文本。", 1)
    a2 = _chunk("同一段文本。", 2)
    out = dedup_chunks([a, b, a2], embedder=HashEmbeddings(dim=32))
    assert out == [a, b]


# ---------------------------------------------------------------------------
# dedup_chunks — 真嵌入路径
# ---------------------------------------------------------------------------
class _FakeEmbedder:
    """受控伪嵌入器：按文本返回预设向量，模拟真嵌入行为（非 HashEmbeddings）。"""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors.get(t, [0.0, 0.0]) for t in texts]


def test_dedup_chunks_real_embedder_removes_near_duplicates() -> None:
    """真嵌入路径：cosine ≥ threshold 视为重复被跳过。"""
    # v1 与 v2 几乎平行（cosine≈1），v3 与它们正交
    v1 = [1.0, 0.01]
    v2 = [1.0, 0.0]
    v3 = [0.0, 1.0]
    emb = _FakeEmbedder({"A": v1, "B": v2, "C": v3})
    a, b, c = _chunk("A", 0), _chunk("B", 1), _chunk("C", 2)
    out = dedup_chunks([a, b, c], embedder=emb, threshold=0.95)
    # B 与 A cosine≈1 ≥ 0.95 → 视为重复跳过；C 正交 → 保留
    assert out == [a, c]


def test_dedup_chunks_real_embedder_threshold_boundary() -> None:
    """阈值边界：低于阈值的不被视为重复。"""
    # cosine(v1,v2) = 1/(1*sqrt(2)) ≈ 0.707
    emb = _FakeEmbedder({"A": [1.0, 0.0], "B": [1.0, 1.0]})
    a, b = _chunk("A", 0), _chunk("B", 1)
    # threshold=0.95：0.707 < 0.95 → 不去重
    assert dedup_chunks([a, b], embedder=emb, threshold=0.95) == [a, b]
    # threshold=0.5：0.707 ≥ 0.5 → 去重
    assert dedup_chunks([a, b], embedder=emb, threshold=0.5) == [a]


class _ExplodingEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:  # noqa: ARG002
        raise RuntimeError("embedder unavailable")


def test_dedup_chunks_embedder_failure_falls_back_to_exact() -> None:
    """真嵌入调用抛错时，安全降级到精确匹配，不崩溃且仍能去重完全相同的 chunk。"""
    a = _chunk("同一段文本。", 0)
    b = _chunk("另一段。", 1)
    a2 = _chunk("同一段文本。", 2)
    out = dedup_chunks([a, b, a2], embedder=_ExplodingEmbedder())
    assert out == [a, b]
