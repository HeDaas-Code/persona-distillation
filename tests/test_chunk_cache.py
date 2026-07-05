"""chunk-progress-cache-uuid spec 的正式单元/集成测试。

覆盖 SubTask 7.1-7.4：
  7.1  _normalize_llm_output 三种格式 + chunk UUID 确定性
  7.2  IndexStore 缓存方法往返一致性
  7.3  集成：首次 5 块 → 第二次跳过 5 块
  7.4  集成：max_chunks=3 续传

跑法：``.venv/bin/python -m tests.test_chunk_cache``

风格沿用 ``tests/smoke_test.py``：简单 print + assert + 返回码，
不强制 pytest 框架（但函数名 ``test_*`` 也兼容 pytest 收集）。
所有 LLM 路径走 ``offline=True`` 启发式 NER，不依赖 API key。
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# 辅助：构造能切成 N 块的测试文件
# ---------------------------------------------------------------------------
def _make_corpus_file(tmp: Path, n_paragraphs: int) -> Path:
    """写一个能切成 >= n_paragraphs 块的中文测试文件。

    每段约 700+ 字符（中文 tiktoken ≈ 2 tokens/字 → ~1400 tokens/段），
    超过 intake_chunk_size=1200，保证每段独立成块。
    """
    test_file = tmp / "corpus.txt"
    parts: list[str] = []
    for i in range(n_paragraphs):
        # 每段一个不同人名 + 重复内容撑长，保证分块器切成独立块
        parts.append(
            f"第{i}段：张三{str(i)}号说这是段落开头。"
            + "荒川老师点点头。" * 80
            + f"李四{str(i)}号出现了。"
        )
    test_file.write_text("\n\n".join(parts), encoding="utf-8")
    return test_file


# ---------------------------------------------------------------------------
# SubTask 7.1: _normalize_llm_output + chunk UUID 确定性
# ---------------------------------------------------------------------------
def test_normalize_llm_output_and_chunk_uuid() -> None:
    from persona_distillation.chunker import chunk_text
    from persona_distillation.intake.name_extractor import _normalize_llm_output
    from persona_distillation.intake.schemas import IndexCategory

    # ---- 格式 1: 扁平 category + evidence ----
    data_flat = {
        "mentions": [
            {"name": "荒川", "category": "speech", "evidence": "嘛，再看看吧。"},
            {"name": "小明", "category": "event", "evidence": "来旧书店买书。"},
        ]
    }
    m_flat = _normalize_llm_output(data_flat)
    assert len(m_flat) == 2, f"扁平格式应解析 2 条，实际 {len(m_flat)}"
    assert m_flat[0].name == "荒川"
    assert m_flat[0].category == IndexCategory.SPEECH
    assert m_flat[0].evidence == "嘛，再看看吧。"
    assert m_flat[1].category == IndexCategory.EVENT

    # ---- 格式 2: 嵌套 speech/appearance/event 数组 ----
    data_nested = {
        "mentions": [
            {
                "name": "荒川",
                "speech": ["嘛，再看看吧。", "书不还价。"],
                "appearance": [{"evidence": "穿藏青色开衫"}],
                "event": "年轻时在东京念书。",
            }
        ]
    }
    m_nested = _normalize_llm_output(data_nested)
    # 2 speech + 1 appearance + 1 event = 4
    assert len(m_nested) == 4, f"嵌套格式应解析 4 条，实际 {len(m_nested)}"
    cats = [m.category for m in m_nested]
    assert cats.count(IndexCategory.SPEECH) == 2
    assert cats.count(IndexCategory.APPEARANCE) == 1
    assert cats.count(IndexCategory.EVENT) == 1
    # 字符串元素和 {"evidence": ...} 对象都能解析
    speeches = [m.evidence for m in m_nested if m.category == IndexCategory.SPEECH]
    assert "嘛，再看看吧。" in speeches
    assert "书不还价。" in speeches
    assert m_nested[2].evidence == "穿藏青色开衫"  # appearance

    # ---- 格式 3: type 字段（category 的别名）----
    data_type = {
        "mentions": [
            {"name": "中林", "type": "event", "evidence": "来买字典。"},
        ]
    }
    m_type = _normalize_llm_output(data_type)
    assert len(m_type) == 1
    assert m_type[0].name == "中林"
    assert m_type[0].category == IndexCategory.EVENT
    assert m_type[0].evidence == "来买字典。"

    # ---- 边界：空 mentions / 非 dict 元素 ----
    assert _normalize_llm_output({"mentions": []}) == []
    assert _normalize_llm_output({"mentions": ["不是dict", None, 42]}) == []
    assert _normalize_llm_output({}) == []  # 无 mentions 键

    # ---- chunk UUID 确定性：同输入同输出 ----
    text = "段落一的内容。\n\n段落二的内容。\n\n段落三的内容。"
    chunks_a = chunk_text(text, target_tokens=1200, overlap_tokens=120)
    chunks_b = chunk_text(text, target_tokens=1200, overlap_tokens=120)
    assert len(chunks_a) == len(chunks_b), "同输入应切出相同块数"
    for ca, cb in zip(chunks_a, chunks_b):
        assert ca.uuid == cb.uuid, (
            f"chunk {ca.index} UUID 不确定：{ca.uuid} vs {cb.uuid}"
        )
        assert len(ca.uuid) == 36, f"UUID 应是 36 字符标准格式，实际 {len(ca.uuid)}"

    # ---- 不同 chunk 的 uuid 不同 ----
    uuids = {c.uuid for c in chunks_a}
    assert len(uuids) == len(chunks_a), "同语料内不同 chunk 的 uuid 应不同"

    # ---- 不同输入 → 不同 uuid（同 index 也不冲突）----
    text2 = "完全不同的内容。"
    chunks_c = chunk_text(text2, target_tokens=1200, overlap_tokens=120)
    if chunks_c:
        assert chunks_c[0].uuid != chunks_a[0].uuid, (
            "不同内容同索引的 chunk uuid 应不同"
        )

    print("[7.1] _normalize_llm_output 三种格式 + chunk UUID 确定性 OK")


# ---------------------------------------------------------------------------
# SubTask 7.2: IndexStore 缓存方法往返一致性
# ---------------------------------------------------------------------------
def test_index_store_cache_roundtrip() -> None:
    from persona_distillation.intake.embedder import HashEmbeddings
    from persona_distillation.intake.index_store import IndexStore

    with tempfile.TemporaryDirectory() as td:
        store = IndexStore(td, embedding=HashEmbeddings(dim=64))

        corpus_uuid = "test-corpus-uuid-7.2"
        chunk_uuid_a = "chunk-uuid-a"
        chunk_uuid_b = "chunk-uuid-b"
        content_hash_a = hashlib.sha256(b"chunk A text").hexdigest()
        content_hash_b = hashlib.sha256(b"chunk B text").hexdigest()

        # ---- register_corpus：首次 True，重复 False ----
        is_new = store.register_corpus(
            corpus_uuid, source_path="a.txt",
            content_hash="doc-hash", total_chunks=2,
        )
        assert is_new is True, "首次注册应返回 True"
        is_new2 = store.register_corpus(
            corpus_uuid, source_path="a.txt",
            content_hash="doc-hash", total_chunks=2,
        )
        assert is_new2 is False, "重复注册应返回 False"

        # ---- is_chunk_processed：未处理返回 None ----
        assert store.is_chunk_processed(corpus_uuid, chunk_uuid_a) is None
        assert store.is_chunk_processed(corpus_uuid, chunk_uuid_b) is None

        # ---- mark_chunk_processed 后 is_chunk_processed 返回正确 hash ----
        store.mark_chunk_processed(corpus_uuid, chunk_uuid_a, content_hash_a, mention_count=3)
        got_a = store.is_chunk_processed(corpus_uuid, chunk_uuid_a)
        assert got_a == content_hash_a, (
            f"mark 后应返回 content_hash，实际 {got_a}"
        )
        # chunk_uuid_b 仍未处理
        assert store.is_chunk_processed(corpus_uuid, chunk_uuid_b) is None

        # 标记第二个 chunk
        store.mark_chunk_processed(corpus_uuid, chunk_uuid_b, content_hash_b, mention_count=5)

        # ---- get_corpus_progress 返回 (processed=2, total=2) ----
        processed, total = store.get_corpus_progress(corpus_uuid)
        assert (processed, total) == (2, 2), (
            f"mark 后 progress 应是 (2,2)，实际 ({processed},{total})"
        )

        # ---- 未注册的 corpus_uuid 返回 (0, 0) ----
        assert store.get_corpus_progress("nonexistent-uuid") == (0, 0)

        # ---- mark 重复调用会刷新（INSERT OR REPLACE）----
        store.mark_chunk_processed(corpus_uuid, chunk_uuid_a, content_hash_a, mention_count=99)
        # 进度仍是 2/2（不会因为 REPLACE 多算）
        processed2, total2 = store.get_corpus_progress(corpus_uuid)
        assert (processed2, total2) == (2, 2), "REPLACE 不应导致进度多算"

        # ---- update_corpus_progress 同步缓存列 ----
        store.update_corpus_progress(corpus_uuid)
        # 通过查 registry 表的 processed_chunks 列验证同步
        cur = store._conn.execute(
            "SELECT processed_chunks FROM corpus_registry WHERE corpus_uuid = ?",
            (corpus_uuid,),
        )
        cached = cur.fetchone()[0]
        assert cached == 2, f"update_corpus_progress 应同步到缓存列，实际 {cached}"

        store.close()

    print("[7.2] IndexStore register/is_processed/mark/progress 往返一致 OK")


# ---------------------------------------------------------------------------
# SubTask 7.3: 集成 — 首次 5 块 → 第二次跳过 5 块
# ---------------------------------------------------------------------------
def test_integration_first_run_then_skip_all() -> None:
    from persona_distillation.config import DistillationConfig
    from persona_distillation.intake.tools import build_intake_context, build_intake_tools

    with tempfile.TemporaryDirectory() as td:
        cfg = DistillationConfig(
            workdir=td, offline=True, dry_run=True, show_progress=False,
        )
        ctx = build_intake_context(cfg)
        tools = build_intake_tools(ctx)
        intake = [t for t in tools if t.name == "intake_corpus"][0]

        # 写能切成 5 块的测试文件
        test_file = _make_corpus_file(Path(td), n_paragraphs=5)

        # 加载并取 corpus_uuid（用于后面查 progress）
        from persona_distillation.loader import load_corpus
        docs = load_corpus(test_file)
        assert len(docs) == 1
        corpus_uuid = docs[0].corpus_uuid
        assert corpus_uuid, "LoadedDoc 应已算出 corpus_uuid"

        # ---- 第一次调用：处理全部 5 块 ----
        r1 = intake.invoke({"path": str(test_file)})
        # 验证全部处理，无跳过、无剩余
        assert "5/5 块" in r1, f"第一次应处理 5/5 块，输出: {r1}"
        assert "跳过" not in r1, f"第一次不应有跳过，输出: {r1}"
        assert "剩余" not in r1, f"第一次不应有剩余，输出: {r1}"

        # 验证 progress 是 (5, 5)
        processed, total = ctx.store.get_corpus_progress(corpus_uuid)
        assert total == 5, f"总块数应是 5，实际 {total}"
        assert processed == 5, f"已处理应是 5，实际 {processed}"

        # 记下第一次的索引条数，第二次不应增加
        count_after_first = ctx.store.count()
        assert count_after_first > 0, "第一次应写入索引"

        # ---- 第二次同路径调用：跳过全部 5 块 ----
        r2 = intake.invoke({"path": str(test_file)})
        assert "5/5 块" in r2, f"第二次也应 5/5（全部跳过），输出: {r2}"
        assert "跳过 5 块" in r2, f"第二次应跳过 5 块，输出: {r2}"
        # 第二次不应新增任何索引
        count_after_second = ctx.store.count()
        assert count_after_second == count_after_first, (
            f"第二次不应新增索引：第一次 {count_after_first}，"
            f"第二次 {count_after_second}"
        )
        # 第二次「新增 X 条提及」中的 X 应为 0（全部跳过）
        # 检查单文件汇总行：跳过场景下 file_line 仍包含「新增 0 条提及」
        assert "新增 0 条提及" in r2, (
            f"全部跳过时新增 mention 应为 0，输出: {r2}"
        )

        # progress 仍是 (5, 5)
        processed2, total2 = ctx.store.get_corpus_progress(corpus_uuid)
        assert (processed2, total2) == (5, 5), (
            f"第二次后 progress 仍应是 (5,5)，实际 ({processed2},{total2})"
        )

        ctx.store.close()

    print("[7.3] 集成：首次 5 块 → 第二次跳过 5 块 OK")


# ---------------------------------------------------------------------------
# SubTask 7.4: 集成 — max_chunks=3 续传
# ---------------------------------------------------------------------------
def test_integration_max_chunks_resume() -> None:
    from persona_distillation.config import DistillationConfig
    from persona_distillation.intake.tools import build_intake_context, build_intake_tools

    with tempfile.TemporaryDirectory() as td:
        cfg = DistillationConfig(
            workdir=td, offline=True, dry_run=True, show_progress=False,
        )
        ctx = build_intake_context(cfg)
        tools = build_intake_tools(ctx)
        intake = [t for t in tools if t.name == "intake_corpus"][0]

        # 写能切成 6 块的测试文件
        test_file = _make_corpus_file(Path(td), n_paragraphs=6)

        from persona_distillation.loader import load_corpus
        docs = load_corpus(test_file)
        corpus_uuid = docs[0].corpus_uuid

        # ---- 第一次调用：max_chunks=3，处理 3 块，剩余 3 块 ----
        r1 = intake.invoke({"path": str(test_file), "max_chunks": 3})
        # 应处理 3 块，剩余 3 块（6 - 3）
        assert "剩余" in r1, f"第一次 max_chunks=3 应有剩余，输出: {r1}"
        assert "剩余 3 块" in r1, f"第一次应剩余 3 块，输出: {r1}"
        # 不应有跳过（首次调用，无缓存）
        assert "跳过" not in r1, f"第一次不应有跳过，输出: {r1}"
        # 验证 progress 是 (3, 6)
        processed1, total1 = ctx.store.get_corpus_progress(corpus_uuid)
        assert (processed1, total1) == (3, 6), (
            f"第一次后 progress 应是 (3,6)，实际 ({processed1},{total1})"
        )

        count_after_first = ctx.store.count()

        # ---- 第二次调用：max_chunks=3，跳过 3 块 + 处理 3 新块 ----
        r2 = intake.invoke({"path": str(test_file), "max_chunks": 3})
        # 第二次应有跳过（前 3 块缓存命中）
        assert "跳过" in r2, f"第二次应有跳过（前 3 块缓存命中），输出: {r2}"
        assert "跳过 3 块" in r2, f"第二次应跳过 3 块，输出: {r2}"
        # 第二次处理完剩余 3 块，不应再有剩余
        assert "剩余" not in r2, (
            f"第二次处理完最后 3 块后不应有剩余，输出: {r2}"
        )
        # 第二次应新增索引（处理了 3 个新 chunk）
        count_after_second = ctx.store.count()
        assert count_after_second > count_after_first, (
            f"第二次应新增索引：第一次 {count_after_first}，"
            f"第二次 {count_after_second}"
        )
        # 验证 progress 是 (6, 6)
        processed2, total2 = ctx.store.get_corpus_progress(corpus_uuid)
        assert (processed2, total2) == (6, 6), (
            f"第二次后 progress 应是 (6,6)，实际 ({processed2},{total2})"
        )

        # ---- 第三次调用：max_chunks=3，应全部跳过（已全部处理）----
        r3 = intake.invoke({"path": str(test_file), "max_chunks": 3})
        assert "跳过" in r3, f"第三次应全部跳过，输出: {r3}"
        assert "剩余" not in r3, f"第三次不应有剩余，输出: {r3}"
        # 索引数不应变化
        count_after_third = ctx.store.count()
        assert count_after_third == count_after_second, (
            f"第三次不应新增索引：第二次 {count_after_second}，"
            f"第三次 {count_after_third}"
        )

        ctx.store.close()

    print("[7.4] 集成：max_chunks=3 续传 OK")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> int:
    tests = [
        ("7.1", test_normalize_llm_output_and_chunk_uuid),
        ("7.2", test_index_store_cache_roundtrip),
        ("7.3", test_integration_first_run_then_skip_all),
        ("7.4", test_integration_max_chunks_resume),
    ]
    failures: list[str] = []
    for label, fn in tests:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback

            tb = traceback.format_exc()
            failures.append(f"[{label}] {fn.__name__}: {e}\n{tb}")
            print(f"[{label}] FAIL {fn.__name__}: {e}")
            print(tb)
    print()
    if failures:
        print(f"=== {len(failures)} FAILURES ===")
        for f in failures:
            print(f"  - {f.splitlines()[0]}")
        return 1
    print("=== ALL 4 CHUNK-CACHE TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
