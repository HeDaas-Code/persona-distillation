"""Token 感知分块器。

优先用 tiktoken 估算 token 数（与多数 LLM 计费一致），不可用时退化为字符数。
按段落聚合，保证分块不会从句子中间劈开，并保留 ``chunk_overlap`` 重叠以维持语境。
"""
from __future__ import annotations

import hashlib
import math
import re
import uuid as _uuid
from dataclasses import dataclass
from typing import Any

try:
    import tiktoken  # type: ignore

    _ENC = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_ENC.encode(text))

    _HAS_TIKTOKEN = True
except Exception:  # pragma: no cover - 环境降级
    _HAS_TIKTOKEN = False

    def _count_tokens(text: str) -> int:
        return len(text)


@dataclass
class Chunk:
    """一个分块。

    ``uuid`` 是基于分块正文 SHA-256(前 16 hex) 与分块索引生成的确定性 UUID v5：
    同一份输入文本 + 同一分块顺序 → 同一 ``uuid``。用于跨运行的 chunk 级缓存命中。
    """

    text: str
    index: int
    char_start: int
    char_end: int
    token_count: int
    uuid: str = ""

    def with_source(self, source_file: str, total: int) -> dict:
        return {
            "source_file": source_file,
            "chunk_index": self.index,
            "total_chunks": total,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "token_count": self.token_count,
            "text": self.text,
            "uuid": self.uuid,
        }


_PARA_SPLIT = re.compile(r"\n\s*\n")


def _split_paragraphs(text: str) -> list[tuple[int, int, str]]:
    """返回 [(char_start, char_end, paragraph), ...]，保留段落定位。"""
    paras: list[tuple[int, int, str]] = []
    pos = 0
    for m in _PARA_SPLIT.split(text):
        if not m:
            continue
        start = text.find(m, pos)
        if start < 0:
            start = pos
        end = start + len(m)
        paras.append((start, end, m))
        pos = end
    if not paras and text.strip():
        paras.append((0, len(text), text))
    return paras


def chunk_text(
    text: str,
    *,
    target_tokens: int = 1800,
    overlap_tokens: int = 200,
    max_chunks: int = 0,
) -> list[Chunk]:
    """将长文本切分为带重叠的分块。

    Args:
        text: 原文。
        target_tokens: 每块目标 token 数。
        overlap_tokens: 重叠 token 数（用于跨块语境延续）。
        max_chunks: 最多返回多少块，0 表示不限。
    """
    if not text.strip():
        return []

    paras = _split_paragraphs(text)
    chunks: list[Chunk] = []
    buf: list[tuple[int, int, str]] = []
    buf_tokens = 0
    char_cursor = 0

    def _flush() -> None:
        nonlocal buf, buf_tokens
        if not buf:
            return
        body = "\n\n".join(p for _, _, p in buf)
        # 确定性 UUID v5：基于 (块索引, 块正文 SHA-256[:16])。
        # 同一份输入文本恒定产生同一 uuid，用于跨运行的 chunk 级缓存键。
        chunk_index = len(chunks)
        chunk_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
        chunk_uuid = str(_uuid.uuid5(_uuid.NAMESPACE_URL, f"{chunk_index}\0{chunk_hash}"))
        chunks.append(
            Chunk(
                text=body,
                index=chunk_index,
                char_start=buf[0][0],
                char_end=buf[-1][1],
                token_count=buf_tokens,
                uuid=chunk_uuid,
            )
        )
        # 重叠：保留尾部若干段落，直到累计 token 接近 overlap_tokens
        tail: list[tuple[int, int, str]] = []
        tail_tokens = 0
        for s, e, p in reversed(buf):
            t = _count_tokens(p)
            if tail_tokens + t > overlap_tokens and tail:
                break
            tail.insert(0, (s, e, p))
            tail_tokens += t
        buf = tail
        buf_tokens = tail_tokens

    for s, e, p in paras:
        t = _count_tokens(p)
        if buf and buf_tokens + t > target_tokens and t <= target_tokens:
            _flush()
            if max_chunks and len(chunks) >= max_chunks:
                return chunks
        buf.append((s, e, p))
        buf_tokens += t
        char_cursor = e
        # 单段超长且超过 target：直接成块后清空缓冲
        if buf_tokens >= target_tokens and len(buf) == 1:
            _flush()
            if max_chunks and len(chunks) >= max_chunks:
                return chunks

    _flush()
    if max_chunks:
        chunks = chunks[:max_chunks]
    return chunks


# ---------------------------------------------------------------------------
# Issue #18.a: chunk 去重
# ---------------------------------------------------------------------------
def _sha256_hex(text: str) -> str:
    """文本 SHA-256 十六进制摘要。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """余弦相似度（向量未归一化时也能用）。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def dedup_chunks(
    chunks: list[Chunk],
    embedder: Any | None = None,
    *,
    threshold: float = 0.95,
) -> list[Chunk]:
    """对分块列表去重，返回去重后的 ``Chunk`` 列表（保留首次出现）。

    Issue #18.a：在 NER 之前去除重复 chunk，避免同一文本被多次识别浪费 LLM 调用。

    - **HashEmbeddings / None embedder**：退化为 SHA-256 精确匹配（文本完全
      相同才算重复）。HashEmbeddings 的「向量」由 hash 派生，cosine 相似度
      无意义，必须走精确匹配。
    - **真嵌入 embedder**：对每个 chunk 的 ``text`` 取 embedding，与已保留
      chunk 的 embedding 算 cosine；≥ ``threshold`` 视为重复跳过。

    Args:
        chunks: 原始分块列表。
        embedder: langchain ``Embeddings``（或类似接口的对象，需有
            ``embed_documents``）。``None`` 或 :class:`HashEmbeddings` 走精确匹配。
        threshold: cosine 相似度阈值，仅在使用真嵌入时生效。

    Returns:
        去重后的 ``Chunk`` 列表（按原顺序保留首次出现的 chunk）。
    """
    if not chunks:
        return []

    # 判断是否走精确匹配：None / HashEmbeddings
    use_exact = embedder is None
    if not use_exact:
        try:
            from persona_distillation.intake.embedder import HashEmbeddings

            if isinstance(embedder, HashEmbeddings):
                use_exact = True
        except Exception:  # noqa: BLE001
            # 模块未就绪时安全降级到精确匹配
            use_exact = True

    kept: list[Chunk] = []
    if use_exact:
        seen_hashes: set[str] = set()
        for c in chunks:
            h = _sha256_hex(c.text)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            kept.append(c)
        return kept

    # 真嵌入路径
    try:
        kept_vecs: list[list[float]] = []
        for c in chunks:
            vec = embedder.embed_documents([c.text])[0]
            is_dup = False
            for kv in kept_vecs:
                if _cosine_sim(vec, kv) >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(c)
                kept_vecs.append(vec)
    except Exception:  # noqa: BLE001
        # embedder 调用失败时安全降级到精确匹配
        seen_hashes: set[str] = set()
        for c in chunks:
            h = _sha256_hex(c.text)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            kept.append(c)
    return kept
