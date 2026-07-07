"""chunk-progress-cache-uuid spec 的正式单元/集成测试。

覆盖 SubTask 7.1-7.4：
  7.1  _normalize_llm_output 三种格式 + chunk UUID 确定性
  7.2  IndexStore 缓存方法往返一致性
  7.3  集成：load_and_chunk + index_characters 端到端（Phase 1 重构后重写）
  7.4  集成：index_characters 增量分批写库（Phase 1 重构后重写）

跑法：``.venv/bin/python -m tests.test_chunk_cache``

风格沿用 ``tests/smoke_test.py``：简单 print + assert + 返回码，
不强制 pytest 框架（但函数名 ``test_*`` 也兼容 pytest 收集）。
所有 LLM 路径走 ``offline=True`` 启发式 NER，不依赖 API key。

注：Phase 1 重构（#15）移除了 ``intake_corpus`` 工具，新流程是
``load_and_chunk``（分块，不做 NER）+ ``task(intake_ner)`` SubAgent 批量 NER
+ ``index_characters``（写库）。旧 7.3/7.4 测的「intake_corpus 缓存命中跳过」
逻辑已不适用，改为测新流程的端到端 + 增量写库。chunk-progress-cache 的
底层基础设施（register_corpus / is_chunk_processed / mark_chunk_processed /
get_corpus_progress / update_corpus_progress）仍保留在 IndexStore，由 7.2 覆盖。
"""
from __future__ import annotations

import hashlib
import json
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
# SubTask 7.3（重写）: 集成 — load_and_chunk + index_characters 端到端
# ---------------------------------------------------------------------------
def test_integration_load_and_chunk_then_index() -> None:
    """Phase 1 重构后重写：旧 7.3 测的 intake_corpus 缓存命中已不适用。

    新流程：load_and_chunk（分块）→ 合成 NerBatchResult（模拟 intake_ner
    SubAgent 产出）→ index_characters（写库）→ list_characters（查询）。

    验证：
    - load_and_chunk 返回合法 chunks JSON（5 块）
    - index_characters 把全部 5 个 chunk 的 mentions 写入 IndexStore
    - 写入后 store.count() == 5，list_characters 能查到 5 位人物
    """
    from persona_distillation.config import DistillationConfig
    from persona_distillation.intake.tools import build_intake_context, build_intake_tools

    with tempfile.TemporaryDirectory() as td:
        cfg = DistillationConfig(
            workdir=td, offline=True, dry_run=True, show_progress=False,
        )
        ctx = build_intake_context(cfg)
        tools = build_intake_tools(ctx)
        load_and_chunk = [t for t in tools if t.name == "load_and_chunk"][0]
        index_characters = [t for t in tools if t.name == "index_characters"][0]
        list_characters = [t for t in tools if t.name == "list_characters"][0]

        # 写能切成 5 块的测试文件
        test_file = _make_corpus_file(Path(td), n_paragraphs=5)

        # ---- 第 1 步: load_and_chunk 拿到 chunks JSON（不做 NER）----
        r1 = load_and_chunk.invoke({"path": str(test_file)})
        chunks = json.loads(r1)
        assert len(chunks) == 5, f"应切出 5 块，实际 {len(chunks)}"
        # 每块应带 chunk_meta 字段（供后续 index_characters 透传）
        for c in chunks:
            assert "source" in c and "chunk_index" in c and "corpus_uuid" in c
            assert "char_start" in c and "content_hash" in c and "total_chunks" in c

        # ---- 第 2 步: 合成 NerBatchResult（模拟 intake_ner SubAgent 产出）----
        # 每个 chunk 构造 1 条 mention，用 chunk 文本前 20 字符作 evidence
        ner_items = []
        for chunk in chunks:
            evidence = chunk["text"][:20]
            ner_items.append({
                "chunk_meta": {
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                    "char_start": chunk["char_start"],
                    "corpus_uuid": chunk["corpus_uuid"],
                    "content_hash": chunk["content_hash"],
                    "total_chunks": chunk["total_chunks"],
                },
                "mentions": [{
                    "name": f"张三{chunk['chunk_index']}号",
                    "aliases": [],
                    "category": "event",
                    "evidence": evidence,
                    "char_start": 0,
                    "char_end": len(evidence),
                }],
            })
        ner_json = json.dumps({"items": ner_items}, ensure_ascii=False)

        # ---- 第 3 步: index_characters 写库 ----
        r2 = index_characters.invoke({"ner_results_json": ner_json})
        assert "索引建立完成" in r2, f"应提示索引建立完成，输出: {r2}"
        assert "5 个 chunk" in r2, f"应处理 5 个 chunk，输出: {r2}"

        # 验证索引库有 5 条索引（每 chunk 1 条 mention）
        assert ctx.store.count() == 5, (
            f"应有 5 条索引，实际 {ctx.store.count()}"
        )

        # ---- 第 4 步: list_characters 能查到 5 位人物 ----
        r3 = list_characters.invoke({})
        assert "5 位人物" in r3, f"应识别 5 位人物，输出: {r3}"

        ctx.store.close()

    print("[7.3] 集成：load_and_chunk + index_characters 端到端 OK")


# ---------------------------------------------------------------------------
# SubTask 7.4（重写）: 集成 — index_characters 增量分批写库
# ---------------------------------------------------------------------------
def test_integration_index_characters_incremental() -> None:
    """Phase 1 重构后重写：旧 7.4 测的 max_chunks 续传已不适用。

    新流程下 SubAgent 一次性接收全部 chunk，不再有 max_chunks 概念。但
    index_characters 仍可被多次调用以增量建索引（例如大体量语料分批处理）。
    本测试验证：
    - 第一次 index_characters 写入前 3 个 chunk 的 mentions → 3 条索引
    - 第二次 index_characters 写入后 3 个 chunk 的 mentions → 累计 6 条
    - 两次调用互不干扰，索引库最终含全部 6 条索引
    """
    from persona_distillation.config import DistillationConfig
    from persona_distillation.intake.tools import build_intake_context, build_intake_tools

    with tempfile.TemporaryDirectory() as td:
        cfg = DistillationConfig(
            workdir=td, offline=True, dry_run=True, show_progress=False,
        )
        ctx = build_intake_context(cfg)
        tools = build_intake_tools(ctx)
        load_and_chunk = [t for t in tools if t.name == "load_and_chunk"][0]
        index_characters = [t for t in tools if t.name == "index_characters"][0]

        # 写能切成 6 块的测试文件
        test_file = _make_corpus_file(Path(td), n_paragraphs=6)

        # load_and_chunk 拿到 6 块
        r1 = load_and_chunk.invoke({"path": str(test_file)})
        chunks = json.loads(r1)
        assert len(chunks) == 6, f"应切出 6 块，实际 {len(chunks)}"

        def _build_ner_json(chunk_subset: list[dict]) -> str:
            """从 chunk 子集合成 NerBatchResult JSON。"""
            items = []
            for chunk in chunk_subset:
                evidence = chunk["text"][:20]
                items.append({
                    "chunk_meta": {
                        "source": chunk["source"],
                        "chunk_index": chunk["chunk_index"],
                        "char_start": chunk["char_start"],
                        "corpus_uuid": chunk["corpus_uuid"],
                        "content_hash": chunk["content_hash"],
                        "total_chunks": chunk["total_chunks"],
                    },
                    "mentions": [{
                        "name": f"张三{chunk['chunk_index']}号",
                        "aliases": [],
                        "category": "event",
                        "evidence": evidence,
                        "char_start": 0,
                        "char_end": len(evidence),
                    }],
                })
            return json.dumps({"items": items}, ensure_ascii=False)

        # ---- 第一次: 写入前 3 块 ----
        ner_json_1 = _build_ner_json(chunks[:3])
        r2 = index_characters.invoke({"ner_results_json": ner_json_1})
        assert "3 个 chunk" in r2, f"第一次应处理 3 个 chunk，输出: {r2}"
        assert ctx.store.count() == 3, (
            f"第一次后应有 3 条索引，实际 {ctx.store.count()}"
        )

        # ---- 第二次: 写入后 3 块（增量）----
        ner_json_2 = _build_ner_json(chunks[3:])
        r3 = index_characters.invoke({"ner_results_json": ner_json_2})
        assert "3 个 chunk" in r3, f"第二次应处理 3 个 chunk，输出: {r3}"
        assert ctx.store.count() == 6, (
            f"第二次后应有 6 条索引，实际 {ctx.store.count()}"
        )

        # 验证 6 位人物都识别到了（张三0号 ~ 张三5号）
        chars = ctx.store.list_characters()
        assert len(chars) == 6, f"应有 6 位人物，实际 {len(chars)}"

        ctx.store.close()

    print("[7.4] 集成：index_characters 增量分批写库 OK")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> int:
    tests = [
        ("7.1", test_normalize_llm_output_and_chunk_uuid),
        ("7.2", test_index_store_cache_roundtrip),
        ("7.3", test_integration_load_and_chunk_then_index),
        ("7.4", test_integration_index_characters_incremental),
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
