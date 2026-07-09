"""IndexStore 检索与归并的覆盖率缺口测试（Issue #12 / #17）。

commit 972d563（修复 16 个 GitHub issue）中两处关键改动此前无针对性单测：

- **#12**：``search`` 在 ``HashEmbeddings``（伪嵌入）/ Chroma 不可用时，必须
  退化到 SQLite ``LIKE`` 关键词匹配，并支持 ``character_name`` 过滤与 ``k`` 上限。
  若退化路径失效，离线/单测场景会返回空结果或随机向量结果，掩盖真实检索问题。
- **#17**：``merge_characters`` 是跨 chunk 实体归并的底层原子操作，负责把源人物
  全部条目并入目标并传播别名。边界（source==target / 不存在）与别名合并正确性
  直接影响实体归并的数据完整性。

所有用例确定性、离线（HashEmbeddings + SQLite 临时目录）、无 LLM 依赖。
跑法：``python -m pytest tests/test_index_store_gaps.py``。
"""
from __future__ import annotations

import tempfile

from persona_distillation.intake.embedder import HashEmbeddings
from persona_distillation.intake.index_store import IndexStore
from persona_distillation.intake.schemas import IndexCategory, NameIndexEntry


def _store() -> IndexStore:
    return IndexStore(tempfile.mkdtemp(), embedding=HashEmbeddings(dim=32))


def _entry(
    name: str, *, text: str, category: IndexCategory = IndexCategory.SPEECH,
    aliases: list[str] | None = None, chunk_index: int = 0,
) -> NameIndexEntry:
    return NameIndexEntry(
        character_name=name,
        aliases=aliases or [],
        category=category,
        text=text,
        source="a.txt",
        chunk_index=chunk_index,
    )


# ---------------------------------------------------------------------------
# search — HashEmbeddings → SQLite LIKE 退化（Issue #12）
# ---------------------------------------------------------------------------
def test_search_like_fallback_finds_by_keyword() -> None:
    """HashEmbeddings 下 search 走 LIKE：按文本关键词命中。"""
    store = _store()
    store.add(_entry("荒川", text="嘛，再看看吧。这版是岩波文库。"))
    store.add(_entry("小明", text="来旧书店买书。"))
    results = store.search("岩波")
    assert len(results) == 1
    assert "岩波" in results[0].text
    store.close()


def test_search_like_fallback_respects_character_filter() -> None:
    """character_name 过滤：仅返回该人物的命中。"""
    store = _store()
    store.add(_entry("荒川", text="书不还价。"))
    store.add(_entry("小明", text="书很便宜。"))  # 同含「书」字
    results = store.search("书", character_name="荒川")
    assert len(results) == 1
    assert results[0].character_name == "荒川"
    store.close()


def test_search_like_fallback_respects_k_limit() -> None:
    """k 上限：命中数超过 k 时只返回前 k 条。"""
    store = _store()
    for i in range(5):
        store.add(_entry(f"人物{i}", text=f"第{i}条记录。", chunk_index=i))
    results = store.search("记录", k=2)
    assert len(results) == 2
    store.close()


def test_search_like_no_match_returns_empty() -> None:
    """关键词不存在 → 空列表（不抛错）。"""
    store = _store()
    store.add(_entry("荒川", text="嘛。"))
    assert store.search("不存在的关键词XYZ") == []
    store.close()


# ---------------------------------------------------------------------------
# merge_characters（Issue #17 底层原子操作）
# ---------------------------------------------------------------------------
def test_merge_characters_source_equals_target_returns_zero() -> None:
    store = _store()
    store.add(_entry("荒川", text="嘛。"))
    assert store.merge_characters("荒川", "荒川") == 0
    store.close()


def test_merge_characters_nonexistent_source_returns_zero() -> None:
    store = _store()
    store.add(_entry("荒川", text="嘛。"))
    assert store.merge_characters("不存在", "荒川") == 0
    # target 不变
    assert store.count() == 1
    store.close()


def test_merge_characters_moves_entries_and_propagates_alias() -> None:
    """合并：源条目迁入 target，返回迁移数，源名作为别名写入 target。"""
    store = _store()
    store.add(_entry("荒川", text="嘛。", aliases=["老师"]))
    store.add(_entry("荒川", text="书。", chunk_index=1))
    store.add(_entry("荒川善次", text="五十二岁。", category=IndexCategory.APPEARANCE))

    moved = store.merge_characters("荒川", "荒川善次")
    assert moved == 2  # 荒川 的 2 条迁入
    # 源人物消失，target 累计 3 条
    names = {c["character_name"] for c in store.list_characters()}
    assert names == {"荒川善次"}
    assert store.count() == 3
    # 「荒川」作为别名传播到 target
    target_aliases = set()
    for e in store.get_character_entries("荒川善次"):
        target_aliases |= set(e.aliases)
    assert "荒川" in target_aliases
    # 原别名「老师」也保留
    assert "老师" in target_aliases
    store.close()
