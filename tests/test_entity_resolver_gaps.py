"""跨 chunk 实体归并器的覆盖率缺口测试（Issue #17）。

`entity_resolver.py` 在 commit 972d563（修复 16 个 GitHub issue）中作为新文件
引入，实现「别名 / 字符串相似 / 嵌入相似」三重信号融合的实体归并。此前无任何
单元测试覆盖——本文件补齐：

- 纯字符串相似度算法：levenshtein / jaro / jaro_winkler / strings_similar
- 余弦相似度边界、CharacterSignals / ResolveResult 聚合属性
- resolve_entities 集成行为：≥2 重自动合并 / 1 重待确认 / 0 重跳过 /
  无嵌入降级 / auto_merge=False / 嵌入信号隔离 / target 选择

所有用例确定性、离线（HashEmbeddings + SQLite 临时目录）、无 LLM 依赖。
跑法：``python -m pytest tests/test_entity_resolver_gaps.py``。
"""
from __future__ import annotations

import tempfile

from persona_distillation.intake.embedder import HashEmbeddings
from persona_distillation.intake.entity_resolver import (
    CharacterSignals,
    ResolveResult,
    _cosine,
    jaro,
    jaro_winkler,
    levenshtein,
    resolve_entities,
    strings_similar,
)
from persona_distillation.intake.index_store import IndexStore
from persona_distillation.intake.schemas import IndexCategory, NameIndexEntry


# ---------------------------------------------------------------------------
# levenshtein
# ---------------------------------------------------------------------------
def test_levenshtein_basic() -> None:
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "") == 3
    assert levenshtein("kitten", "sitting") == 3  # 经典用例
    # 单字符替换
    assert levenshtein("cat", "cot") == 1
    # 插入
    assert levenshtein("cat", "cats") == 1
    # 换位在 Levenshtein 中算 2（两次替换）
    assert levenshtein("ab", "ba") == 2


# ---------------------------------------------------------------------------
# jaro / jaro_winkler
# ---------------------------------------------------------------------------
def test_jaro_basic() -> None:
    assert jaro("abc", "abc") == 1.0
    assert jaro("", "x") == 0.0
    assert jaro("abc", "xyz") == 0.0  # 无匹配字符
    # 经典用例：MARTHA / MARHTA = 0.9444
    assert abs(jaro("MARTHA", "MARHTA") - 0.9444) < 1e-3


def test_jaro_winkler_basic() -> None:
    assert jaro_winkler("abc", "abc") == 1.0
    # 公共前缀加权：MARTHA/MARHTA 前缀 3 → 0.961
    assert abs(jaro_winkler("MARTHA", "MARHTA") - 0.961) < 1e-3
    # 无公共前缀时不额外加权
    assert jaro_winkler("xyz", "abc") == 0.0


# ---------------------------------------------------------------------------
# strings_similar
# ---------------------------------------------------------------------------
def test_strings_similar() -> None:
    # 空值守卫
    assert strings_similar("", "x") is False
    assert strings_similar("x", "") is False
    # 完全相同
    assert strings_similar("荒川", "荒川") is True
    # Levenshtein ≤ 2 路径（删 2 字）
    assert strings_similar("荒川善次", "荒川") is True
    # Jaro-Winkler ≥ 0.85 路径（轻微拼写差异）
    assert strings_similar("MARTHA", "MARHTA") is True
    # 完全不同 → False
    assert strings_similar("荒川善次", "高桥太郎") is False


# ---------------------------------------------------------------------------
# _cosine（entity_resolver 内部实现，与 chunker._cosine_sim 独立）
# ---------------------------------------------------------------------------
def test_entity_resolver_cosine_edge_cases() -> None:
    assert _cosine([], [1.0]) == 0.0
    assert _cosine([1.0, 2.0], [1.0]) == 0.0  # 长度不匹配
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


# ---------------------------------------------------------------------------
# CharacterSignals / ResolveResult 聚合属性
# ---------------------------------------------------------------------------
def test_character_signals_hit_count() -> None:
    sig = CharacterSignals(name_a="a", name_b="b")
    assert sig.hit_count == 0
    sig.alias_hit = True
    assert sig.hit_count == 1
    sig.string_hit = True
    sig.embedding_hit = True
    assert sig.hit_count == 3


def test_resolve_result_total_actions() -> None:
    r = ResolveResult()
    assert r.total_actions == 0
    r.auto_merged.append(("a", "b"))
    assert r.total_actions == 1
    r.pending.append(CharacterSignals(name_a="c", name_b="d", string_hit=True))
    assert r.total_actions == 2


# ---------------------------------------------------------------------------
# resolve_entities 集成（HashEmbeddings → 仅别名+字符串两重信号）
# ---------------------------------------------------------------------------
def _entry(
    name: str, *, text: str, aliases: list[str] | None = None, chunk_index: int = 0
) -> NameIndexEntry:
    return NameIndexEntry(
        character_name=name,
        aliases=aliases or [],
        category=IndexCategory.SPEECH,
        text=text,
        source="a.txt",
        chunk_index=chunk_index,
    )


def _store() -> IndexStore:
    return IndexStore(tempfile.mkdtemp(), embedding=HashEmbeddings(dim=32))


def test_resolve_entities_single_character_noop() -> None:
    """<2 个人物 → 空结果，不报错。"""
    store = _store()
    store.add(_entry("荒川", text="嘛。"))
    result = resolve_entities(store, llm=None)
    assert result.auto_merged == []
    assert result.pending == []
    store.close()


def test_resolve_entities_alias_plus_string_auto_merge() -> None:
    """别名交叉 + 字符串相似（2 重，无嵌入）→ 自动合并到 mention 更多的 target。"""
    store = _store()
    # 荒川善次 带 alias「荒川」；荒川 是独立条目 → 经典跨 chunk 归并场景
    store.add(_entry("荒川善次", text="嘛，再看看吧。", aliases=["荒川"], chunk_index=0))
    store.add(_entry("荒川善次", text="书不还价。", chunk_index=1))
    store.add(_entry("荒川", text="五十二岁。", chunk_index=2))
    result = resolve_entities(store, llm=None, auto_merge=True)
    # source=荒川 → target=荒川善次（mention 2 > 1）
    assert ("荒川", "荒川善次") in result.auto_merged
    # 合并后荒川 应消失
    names = {c["character_name"] for c in store.list_characters()}
    assert "荒川" not in names
    assert "荒川善次" in names
    store.close()


def test_resolve_entities_single_hit_goes_pending() -> None:
    """仅别名交叉 1 重（字符串不相似）→ 待确认，不自动合并。"""
    store = _store()
    store.add(_entry("荒川善次", text="嘛。", aliases=["Sensei"]))
    store.add(_entry("Sensei", text="hello."))
    result = resolve_entities(store, llm=None, auto_merge=True)
    assert result.auto_merged == []
    assert len(result.pending) == 1
    assert result.pending[0].alias_hit is True
    assert result.pending[0].string_hit is False
    # 未合并：两个名字都仍在
    names = {c["character_name"] for c in store.list_characters()}
    assert names == {"荒川善次", "Sensei"}
    store.close()


def test_resolve_entities_zero_hits_skipped() -> None:
    """无别名交叉、字符串不相似（0 重）→ 不出现在结果里。"""
    store = _store()
    store.add(_entry("荒川善次", text="嘛。"))
    store.add(_entry("高桥太郎", text="你好。"))
    result = resolve_entities(store, llm=None)
    assert result.auto_merged == []
    assert result.pending == []
    store.close()


def test_resolve_entities_auto_merge_false_goes_pending() -> None:
    """auto_merge=False 时，即使 2 重命中也进 pending 而非合并。"""
    store = _store()
    store.add(_entry("荒川善次", text="嘛。", aliases=["荒川"]))
    store.add(_entry("荒川", text="书。"))
    result = resolve_entities(store, llm=None, auto_merge=False)
    assert result.auto_merged == []
    assert len(result.pending) == 1
    assert result.pending[0].hit_count >= 2
    store.close()


# ---------------------------------------------------------------------------
# resolve_entities — 嵌入信号（受控伪嵌入，非 HashEmbeddings）
# ---------------------------------------------------------------------------
class _ConstantEmbedder:
    """对所有文本返回同一向量 → 任意两人物 mean embedding cosine = 1.0。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:  # noqa: ARG002
        return [[1.0, 0.0] for _ in texts]


def test_resolve_entities_embedding_signal_isolated() -> None:
    """无别名、字符串不相似，但嵌入一致 → embedding_hit=True，1 重 → 待确认。"""
    store = IndexStore(tempfile.mkdtemp(), embedding=_ConstantEmbedder())
    store.add(_entry("荒川善次", text="嘛。"))
    store.add(_entry("高桥太郎", text="你好。"))
    result = resolve_entities(store, llm=None, auto_merge=True, embedding_cos_min=0.8)
    assert result.auto_merged == []  # 仅 1 重不合并
    assert len(result.pending) == 1
    sig = result.pending[0]
    assert sig.embedding_hit is True
    assert sig.embedding_score >= 0.8
    assert sig.alias_hit is False
    assert sig.string_hit is False
    store.close()
