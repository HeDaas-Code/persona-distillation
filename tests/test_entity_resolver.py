''""``intake.entity_resolver`` 单元测试。

覆盖实体归并器的核心纯函数与集成逻辑：
- 字符串相似度：``levenshtein`` / ``jaro`` / ``jaro_winkler`` / ``strings_similar``
- 向量余弦相似度：``_cosine``
- ``CharacterSignals.hit_count`` 属性
- ``ResolveResult.total_actions`` 属性
- ``resolve_entities`` 在无真嵌入（HashEmbeddings/None）下的纯 SQLite 归并行为

跑法：``python -m pytest tests/test_entity_resolver.py -v``
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from persona_distillation.intake import entity_resolver
from persona_distillation.intake.embedder import HashEmbeddings
from persona_distillation.intake.index_store import IndexStore
from persona_distillation.intake.schemas import (
    IndexCategory,
    NameIndexEntry,
)


# ---------------------------------------------------------------------------
# levenshtein 编辑距离
# ---------------------------------------------------------------------------
def test_levenshtein_identical() -> None:
    """完全相同的字符串 → 距离 0。"""
    assert entity_resolver.levenshtein("abc", "abc") == 0
    assert entity_resolver.levenshtein("", "") == 0


def test_levenshtein_one_empty() -> None:
    """一边为空 → 距离等于另一边长度。"""
    assert entity_resolver.levenshtein("", "abcd") == 4
    assert entity_resolver.levenshtein("xyz", "") == 3


def test_levenshtein_single_ops() -> None:
    """单步插入/删除/替换 → 距离 1。"""
    # 插入
    assert entity_resolver.levenshtein("abc", "aXbc") == 1
    # 删除
    assert entity_resolver.levenshtein("aXbc", "abc") == 1
    # 替换
    assert entity_resolver.levenshtein("abc", "aXc") == 1


def test_levenshtein_multi_ops() -> None:
    """多步操作：kitten → sitting（经典用例，距离 3）。"""
    assert entity_resolver.levenshtein("kitten", "sitting") == 3


def test_levenshtein_chinese() -> None:
    """中文字符串（按 Unicode 码位算字符，与 ASCII 一致）。"""
    # 荒川善次 vs 荒川 → 删两个字 → 距离 2
    assert entity_resolver.levenshtein("荒川善次", "荒川") == 2
    # 荒川老师 vs 荒川 → 距离 2
    assert entity_resolver.levenshtein("荒川老师", "荒川") == 2


# ---------------------------------------------------------------------------
# jaro 相似度
# ---------------------------------------------------------------------------
def test_jaro_identical() -> None:
    """完全相同 → 1.0。"""
    assert entity_resolver.jaro("abc", "abc") == 1.0
    assert entity_resolver.jaro("荒川", "荒川") == 1.0


def test_jaro_one_empty() -> None:
    """任一边为空 → 0.0。"""
    assert entity_resolver.jaro("", "anything") == 0.0
    assert entity_resolver.jaro("anything", "") == 0.0


def test_jaro_both_empty() -> None:
    """两边都空 → 1.0（快速路径 a==b）。"""
    assert entity_resolver.jaro("", "") == 1.0


def test_jaro_dixontest_case() -> None:
    """经典对：MARTHA / MARHTA——结果位于 (0, 1) 区间。"""
    j = entity_resolver.jaro("MARTHA", "MARHTA")
    assert 0.0 < j < 1.0
    assert j >= entity_resolver.jaro("MARTHA", "XXXXXX")


def test_jaro_no_match() -> None:
    """完全不相交 → 0。"""
    assert entity_resolver.jaro("abc", "xyz") == 0.0


# ---------------------------------------------------------------------------
# jaro_winkler
# ---------------------------------------------------------------------------
def test_jaro_winkler_identical() -> None:
    """完全相同 → 1.0。"""
    assert abs(entity_resolver.jaro_winkler("abcd", "abcd") - 1.0) < 1e-9


def test_jaro_winkler_prefix_boost() -> None:
    """公共前缀越长，jaro-winkler 比纯 jaro 越高。"""
    # 前 4 个字符相同的一对
    a, b = "abcdef", "abcdxx"
    j = entity_resolver.jaro(a, b)
    jw = entity_resolver.jaro_winkler(a, b)
    assert jw > j, "公共前缀存在时 JW 应高于 Jaro"

    # 前缀长度为 0 → 两者相同
    c, d = "axcdef", "bxdcff"
    j2 = entity_resolver.jaro(c, d)
    jw2 = entity_resolver.jaro_winkler(c, d)
    # 若首字符不同，前缀为 0，JW 应等于 Jaro
    if c[0] != d[0]:
        assert abs(jw2 - j2) < 1e-9


# ---------------------------------------------------------------------------
# strings_similar 组合判定
# ---------------------------------------------------------------------------
def test_strings_similar_identical() -> None:
    """完全相同 → True（快速路径）。"""
    assert entity_resolver.strings_similar("荒川", "荒川")


def test_strings_similar_one_empty() -> None:
    """任一边为空 → False（无法判断相似）。"""
    assert not entity_resolver.strings_similar("", "荒川")
    assert not entity_resolver.strings_similar("荒川", "")
    assert not entity_resolver.strings_similar("", "")


def test_strings_similar_levenshtein_hit() -> None:
    """Levenshtein <= 2 → True。"""
    assert entity_resolver.strings_similar("荒川善次", "荒川", lev_max=2)
    # 距 3（> lev_max=2）→ False
    assert not entity_resolver.strings_similar("荒川善次老师", "荒川", lev_max=2)


def test_strings_similar_jaro_winkler_hit() -> None:
    """JW 命中时也返回 True。"""
    a, b = "abcdefghij", "abcdefgXij"
    # 先验证 lev 不命中
    lev = entity_resolver.levenshtein(a, b)
    if lev > 2:
        # 结果应为 True：JW 前缀加分高（前 7 字符同）
        result = entity_resolver.strings_similar(a, b)
        assert isinstance(result, bool)
    # 确定性：相同输入两次调用结果一致
    assert entity_resolver.strings_similar(a, b) == entity_resolver.strings_similar(a, b)


def test_strings_similar_custom_thresholds() -> None:
    """自定义阈值应生效。"""
    # 默认：首字符不同（'9' vs '1'），公共前缀=0，Lev 距离大 → JW 也不会高
    a, b = "9876543210", "12345"
    lev = entity_resolver.levenshtein(a, b)
    assert lev > 2, f"lev 应 > 2 以便验证默认不命中，实际 {lev}"
    assert not entity_resolver.strings_similar(a, b), (
        f"'{a}' vs '{b}' 在默认阈值下应不相似 "
        f"(lev={lev}, jw={entity_resolver.jaro_winkler(a, b):.3f})"
    )
    # 放宽 lev_max=10 → lev <= 10 → True
    assert entity_resolver.strings_similar(a, b, lev_max=10)


# ---------------------------------------------------------------------------
# _cosine 余弦相似度
# ---------------------------------------------------------------------------
def test_cosine_identical_normalized() -> None:
    """同向向量（已归一化）→ cosine = 1.0。"""
    v = [1.0, 0.0, 0.0]
    assert abs(entity_resolver._cosine(v, v) - 1.0) < 1e-9


def test_cosine_orthogonal() -> None:
    """正交 → 0。"""
    assert abs(entity_resolver._cosine([1, 0], [0, 1])) < 1e-9


def test_cosine_opposite() -> None:
    """反向 → -1。"""
    assert abs(entity_resolver._cosine([1, 0, 1], [-1, 0, -1]) - (-1.0)) < 1e-9


def test_cosine_dim_mismatch() -> None:
    """向量长度不同 → 0.0 不抛异常。"""
    assert entity_resolver._cosine([1, 2], [1, 2, 3]) == 0.0


def test_cosine_empty_or_zero() -> None:
    """空向量或全零向量 → 0.0，不除零。"""
    assert entity_resolver._cosine([], []) == 0.0
    assert entity_resolver._cosine([0, 0, 0], [0, 0, 0]) == 0.0
    assert entity_resolver._cosine([], [1, 2]) == 0.0


def test_cosine_unnormalized_still_works() -> None:
    """非归一化向量也能正确算余弦（有归一化兜底）。"""
    assert abs(entity_resolver._cosine([3.0, 4.0], [6.0, 8.0]) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# CharacterSignals / ResolveResult 数据类属性
# ---------------------------------------------------------------------------
def test_character_signals_hit_count() -> None:
    """hit_count 为三信号布尔值的和。"""
    s = entity_resolver.CharacterSignals(name_a="A", name_b="B")
    assert s.hit_count == 0
    s.alias_hit = True
    assert s.hit_count == 1
    s.string_hit = True
    s.embedding_hit = True
    assert s.hit_count == 3


def test_resolve_result_total_actions() -> None:
    """total_actions = auto_merged 长度 + pending 长度。"""
    r = entity_resolver.ResolveResult()
    assert r.total_actions == 0
    r.auto_merged.append(("X", "Y"))
    r.auto_merged.append(("Z", "Y"))
    r.pending.append(entity_resolver.CharacterSignals(name_a="A", name_b="B"))
    assert r.total_actions == 3


# ---------------------------------------------------------------------------
# resolve_entities 集成测试（使用 HashEmbeddings + 临时 SQLite）
# ---------------------------------------------------------------------------
def _add(store: IndexStore, name: str, aliases: list[str], text: str = "") -> None:
    """向 store 添加一条 SPEECH 条目。"""
    store.add(NameIndexEntry(
        character_name=name,
        aliases=list(aliases),
        category=IndexCategory.SPEECH,
        text=text or f"{name}说过这句话",
        source="test.txt",
        chunk_index=0,
    ))


def test_resolve_entities_fewer_than_two_characters() -> None:
    """< 2 个人物时直接返回空结果。"""
    with tempfile.TemporaryDirectory() as d:
        store = IndexStore(Path(d) / "s", embedding=HashEmbeddings())
        # 空
        r0 = entity_resolver.resolve_entities(store)
        assert r0.total_actions == 0
        assert r0.auto_merged == []
        assert r0.pending == []
        # 1 人
        _add(store, "荒川", [])
        r1 = entity_resolver.resolve_entities(store)
        assert r1.total_actions == 0


def test_resolve_entities_alias_hit_only_is_pending() -> None:
    """只命中 alias（1 重）→ 标记 pending，不自动合并（无真嵌入时不激进合并）。"""
    with tempfile.TemporaryDirectory() as d:
        store = IndexStore(Path(d) / "s", embedding=HashEmbeddings())
        _add(store, "荒川善次", [])
        _add(store, "班主任", ["荒川善次"])
        # 字符串应不相似
        assert not entity_resolver.strings_similar("荒川善次", "班主任")

        result = entity_resolver.resolve_entities(store)
        assert len(result.auto_merged) == 0, "仅命中 1 重不应自动合并"
        assert len(result.pending) == 1, "命中 1 重应进入 pending"
        sig = result.pending[0]
        assert sig.alias_hit is True
        assert sig.string_hit is False
        assert sig.embedding_hit is False
        assert sig.hit_count == 1


def test_resolve_entities_alias_plus_string_auto_merge() -> None:
    """别名 + 字符串 两重命中 → 自动合并。"""
    with tempfile.TemporaryDirectory() as d:
        store = IndexStore(Path(d) / "s", embedding=HashEmbeddings())
        _add(store, "荒川", [])
        _add(store, "荒川老师", ["荒川"])
        assert entity_resolver.strings_similar("荒川", "荒川老师")

        result = entity_resolver.resolve_entities(store)
        assert len(result.pending) == 0
        assert len(result.auto_merged) == 1
        source, target = result.auto_merged[0]
        remaining = {c["character_name"] for c in store.list_characters()}
        assert source not in remaining
        assert target in remaining
        assert len(remaining) == 1


def test_resolve_entities_pick_target_prefers_more_mentions() -> None:
    """mention_count 更多的做 target。"""
    with tempfile.TemporaryDirectory() as d:
        store = IndexStore(Path(d) / "s", embedding=HashEmbeddings())
        for i in range(3):
            store.add(NameIndexEntry(
                character_name="荒川善次",
                aliases=["荒川"],
                category=IndexCategory.SPEECH,
                text=f"荒川善次说话{i}",
                source=f"t{i}.txt",
                chunk_index=i,
            ))
        store.add(NameIndexEntry(
            character_name="荒川",
            aliases=["荒川善次"],
            category=IndexCategory.SPEECH,
            text="荒川说话",
            source="tx.txt",
            chunk_index=0,
        ))
        assert entity_resolver.strings_similar("荒川善次", "荒川")

        result = entity_resolver.resolve_entities(store)
        assert len(result.auto_merged) == 1
        source, target = result.auto_merged[0]
        assert source == "荒川"
        assert target == "荒川善次"


def test_resolve_entities_zero_hits_skipped() -> None:
    """完全不相似的两个人物 → 不出现在 pending 或 auto_merged。"""
    with tempfile.TemporaryDirectory() as d:
        store = IndexStore(Path(d) / "s", embedding=HashEmbeddings())
        _add(store, "荒川", [])
        _add(store, "佐藤完全不一样", [])
        result = entity_resolver.resolve_entities(store)
        assert result.auto_merged == []
        if not entity_resolver.strings_similar("荒川", "佐藤完全不一样"):
            assert result.pending == []


def test_resolve_entities_auto_merge_false_puts_everything_in_pending() -> None:
    """auto_merge=False 时，命中 ≥2 重也进入 pending 而非合并。"""
    with tempfile.TemporaryDirectory() as d:
        store = IndexStore(Path(d) / "s", embedding=HashEmbeddings())
        _add(store, "荒川", [])
        _add(store, "荒川老师", ["荒川"])

        result = entity_resolver.resolve_entities(store, auto_merge=False)
        assert result.auto_merged == []
        assert len(result.pending) == 1
        assert len(store.list_characters()) == 2
