"""Token 感知分块器。

优先用 tiktoken 估算 token 数（与多数 LLM 计费一致），不可用时退化为字符数。
按段落聚合，保证分块不会从句子中间劈开，并保留 ``chunk_overlap`` 重叠以维持语境。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

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
    """一个分块。"""

    text: str
    index: int
    char_start: int
    char_end: int
    token_count: int

    def with_source(self, source_file: str, total: int) -> dict:
        return {
            "source_file": source_file,
            "chunk_index": self.index,
            "total_chunks": total,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "token_count": self.token_count,
            "text": self.text,
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
        chunks.append(
            Chunk(
                text=body,
                index=len(chunks),
                char_start=buf[0][0],
                char_end=buf[-1][1],
                token_count=buf_tokens,
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
