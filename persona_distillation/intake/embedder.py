"""嵌入 + 重排序模型工厂。

为 :class:`IndexStore` 提供 langchain 兼容的 ``Embeddings``，为 :func:`build_profile`
提供 cross-encoder reranker。同时给出一个零依赖的 :class:`HashEmbeddings`，
供离线 / 单元测试场景使用（向量由文本 hash 派生，维度固定）。
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

try:
    from langchain_core.embeddings import Embeddings
except Exception:  # pragma: no cover - langchain_core 可能不存在
    Embeddings = object  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# 零依赖 fallback：基于 hash 的伪 embedding
# ---------------------------------------------------------------------------
class HashEmbeddings(Embeddings if isinstance(Embeddings, type) else object):
    """基于文本 hash 的伪嵌入。

    - 维度固定（默认 256）
    - 相同文本 → 相同向量（确定性）
    - 仅供离线 / 单测使用；生产请用 :func:`build_embedder`
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        h = hashlib.sha512(text.encode("utf-8")).digest()
        # 重复 hash 直到填满 dim
        buf = bytearray()
        while len(buf) < self.dim:
            buf.extend(h)
            h = hashlib.sha512(h).digest()
        vec = [((buf[i] - 128) / 128.0) for i in range(self.dim)]
        # L2 归一化
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


# ---------------------------------------------------------------------------
# 真实嵌入：langchain + HuggingFace
# ---------------------------------------------------------------------------
def build_embedder(model_name: str = "BAAI/bge-m3", **kwargs: Any) -> Any:
    """构造 langchain Embeddings。

    使用 ``HuggingFaceBgeEmbeddings``（中文 + 多语种 + 维度 1024）。
    任何导入失败时退回 :class:`HashEmbeddings` 以保证系统可启动。
    """
    try:
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings

        return HuggingFaceBgeEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
            **kwargs,
        )
    except Exception:
        return HashEmbeddings()


# ---------------------------------------------------------------------------
# 重排序：cross-encoder
# ---------------------------------------------------------------------------
def build_reranker(model_name: str = "BAAI/bge-reranker-base", top_n: int = 6) -> Any:
    """构造 ``CrossEncoderReranker``（langchain）。

    任何依赖缺失时返回 ``None``——调用方应兼容 None 跳过 rerank。
    """
    try:
        from langchain.retrievers.document_compressors import CrossEncoderReranker
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder

        return CrossEncoderReranker(
            model=HuggingFaceCrossEncoder(model_name=model_name),
            top_n=top_n,
        )
    except Exception:
        return None
