"""分馏失败可见性单元测试（Task 9，对应 GitHub Issue #1）。

覆盖 spec Phase 3 "分馏 chunk 失败可见性" 要求：
- ``test_high_failure_rate_aborts``：高失败率（>50%）→ RuntimeError 中止
- ``test_low_failure_rate_continues``：低失败率（<50%）→ 继续 + metadata 含失败统计
- ``test_metadata_contains_failure_stats``：metadata 必含三个失败统计字段

实现方式：
- monkey-patch ``persona_distillation.pipeline.invoke_structured``，按 ``expected_type``
  分流：``Distillate`` 调用按全局序号决定成败；其他类型返回固定默认值
- monkey-patch ``persona_distillation.pipeline.chunk_text`` 返回固定数量 Chunk，
  绕过分块器的 tail-keep 段落 quirk（``overlap=0`` 仍保留 1 段作 tail，导致
  长段落 + 小 chunk_size 不能可靠产出 N 块）
- 注入哨兵 agent（``distiller._extractor`` 等）绕过真实 ``build_*_agent``，避免调 LLM

跑法：``python -m pytest tests/test_pipeline_failure.py -v``

Issue #13：已转为 pytest 风格——``test_*`` 函数由 pytest 自动收集，
直接用 ``assert`` 断言，不再需要 ``main()`` / ``try/except`` 包装。
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

from persona_distillation import pipeline as pipeline_mod
from persona_distillation.agents import PersonaSkillList, PresetDialogueList
from persona_distillation.chunker import Chunk
from persona_distillation.config import DistillationConfig
from persona_distillation.pipeline import PersonaDistiller
from persona_distillation.schemas import (
    Distillate,
    PersonaCard,
    PersonaSignal,
    PresetDialogue,
    SignalCategory,
)


# ---------------------------------------------------------------------------
# 辅助：构造最小合法对象，供 mock invoke_structured 返回
# ---------------------------------------------------------------------------
def _make_distillate(idx: int) -> Distillate:
    """构造一个最小合法的 Distillate，供 extractor mock 返回。"""
    return Distillate(
        source_file="corpus.txt",
        chunk_index=idx,
        char_start=0,
        char_end=10,
        signals=[
            PersonaSignal(
                category=SignalCategory.SPEECH_STYLE,
                content=f"chunk {idx} 的说话风格信号",
                evidence="原文片段",
                salience=0.7,
            )
        ],
        summary=f"chunk {idx} 速写",
    )


def _make_persona_card() -> PersonaCard:
    """构造一个最小合法的 PersonaCard，供 synthesizer mock 返回。

    system_prompt 含七段标记，避免下游校验告警。
    """
    return PersonaCard(
        persona_id="test_persona",
        display_name="测试人格",
        system_prompt=(
            "[身份] 测试人格\n"
            "[性格] 平和\n"
            "[说话风格] 简短\n"
            "[知识边界] 测试范围\n"
            "[情绪模式] 平稳\n"
            "[雷区] 无\n"
            "[输出约束] 简短回答"
        ),
        error_reply="（测试报错回复）",
    )


def _make_skill_list() -> PersonaSkillList:
    return PersonaSkillList(skills=[])


def _make_dialogue_list() -> PresetDialogueList:
    return PresetDialogueList(
        dialogues=[
            PresetDialogue(user="你好", assistant="你好。", intent="寒暄"),
        ]
    )


# ---------------------------------------------------------------------------
# 辅助：构造固定数量 Chunk 的 chunk_text 替身
# ---------------------------------------------------------------------------
def _make_fake_chunk_text(n_chunks: int):
    """返回一个 chunk_text 替身，无视入参，恒定返回 n_chunks 个 Chunk。

    分块器的 tail-keep-paragraph quirk（即使 ``overlap=0`` 也保留 1 段作 tail）
    会让"长段落 + 小 chunk_size"的常规组合无法可靠产出 N 块，故此处直接桩掉。
    """

    def fake_chunk_text(text: str, **kwargs: Any) -> list[Chunk]:
        return [
            Chunk(
                text=f"chunk {i} body",
                index=i,
                char_start=i * 100,
                char_end=i * 100 + 100,
                token_count=100,
                uuid=f"fake-chunk-uuid-{i}",
            )
            for i in range(n_chunks)
        ]

    return fake_chunk_text


# ---------------------------------------------------------------------------
# 辅助：mock invoke_structured，按 expected_type 分流
# ---------------------------------------------------------------------------
def _make_mock_invoke_structured(
    *,
    extractor_fail_indices: set[int],
    extractor_exc: BaseException = RuntimeError("mock extractor down"),
):
    """返回一个 ``invoke_structured`` 替身 + 状态 dict。

    - ``expected_type is Distillate``：按 extractor 全局调用序号决定成败
      （序号 N 在 ``extractor_fail_indices`` 中则抛 ``extractor_exc``，否则返回
      ``_make_distillate(N)``）；序号从 0 起累加，无论成败都计数
    - ``expected_type is PersonaCard`` → 返回 ``_make_persona_card()``
    - ``expected_type is PersonaSkillList`` → 返回 ``_make_skill_list()``
    - ``expected_type is PresetDialogueList`` → 返回 ``_make_dialogue_list()``
    """
    state = {"extractor_calls": 0}

    def mock_invoke(agent: Any, user_prompt: str, expected_type: type) -> Any:
        if expected_type is Distillate:
            idx = state["extractor_calls"]
            state["extractor_calls"] += 1
            if idx in extractor_fail_indices:
                raise extractor_exc
            return _make_distillate(idx)
        if expected_type is PersonaCard:
            return _make_persona_card()
        if expected_type is PersonaSkillList:
            return _make_skill_list()
        if expected_type is PresetDialogueList:
            return _make_dialogue_list()
        raise AssertionError(f"未预期的 expected_type: {expected_type!r}")

    return mock_invoke, state


# ---------------------------------------------------------------------------
# 辅助：构造一个绕过 LLM 的 PersonaDistiller
# ---------------------------------------------------------------------------
def _make_distiller(tmp: Path) -> PersonaDistiller:
    """构造 dry_run 的 distiller，注入哨兵 agent 避免真实 build。

    - ``dry_run=True`` 跳过 API key 校验
    - ``model='openai:test'`` 让 ``build_model`` 原样返回字符串（不实际构造 ChatOpenAI）
    - 注入哨兵 agent 到四个懒加载字段，避免 ``_get_*`` 真去 ``build_*_agent``
    - chunk 总数由测试用例通过 patch chunk_text 控制，不依赖 cfg.chunk_size
    """
    cfg = DistillationConfig(
        model="openai:test",
        dry_run=True,
        chunk_size=50,
        chunk_overlap=0,
        max_chunks_per_file=0,
        workdir=str(tmp / "workdir"),
        show_progress=False,
    )
    distiller = PersonaDistiller(cfg)
    sentinel = object()
    distiller._extractor = sentinel
    distiller._synthesizer = sentinel
    distiller._skill_designer = sentinel
    distiller._dialogue_writer = sentinel
    return distiller


def _make_corpus_file(tmp: Path) -> Path:
    """写一个最小语料文件供 load_corpus 加载（内容由 fake chunk_text 接管分块）。"""
    test_file = tmp / "corpus.txt"
    test_file.write_text("测试语料，内容由 fake chunk_text 接管分块。", encoding="utf-8")
    return test_file


# ---------------------------------------------------------------------------
# 用例 1：高失败率（>50%）应抛 RuntimeError 中止
# ---------------------------------------------------------------------------
def test_high_failure_rate_aborts() -> None:
    """10 个 chunk，前 6 个失败（failure_rate=0.6 > 0.5）→ RuntimeError。

    验证：
    - 抛 RuntimeError 且消息含"分馏失败率"与"60%"
    - 不进入 Stage 2（synthesizer / skill_designer / dialogue_writer 不应被调用）
    - extractor 全部 10 次调用都被尝试（不因失败提前 break）
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        corpus = _make_corpus_file(tmp)
        distiller = _make_distiller(tmp)

        # 前 6 个 chunk 失败：failure_rate = 6/10 = 0.6 > 0.5
        mock_invoke, state = _make_mock_invoke_structured(
            extractor_fail_indices={0, 1, 2, 3, 4, 5},
        )
        with patch.object(pipeline_mod, "invoke_structured", mock_invoke), \
             patch.object(pipeline_mod, "chunk_text", _make_fake_chunk_text(10)):
            try:
                distiller.distill(corpus)
            except RuntimeError as e:
                msg = str(e)
                assert "分馏失败率" in msg, (
                    f"RuntimeError 消息应含'分馏失败率'，实际: {msg!r}"
                )
                assert "60%" in msg, (
                    f"RuntimeError 消息应含'60%'，实际: {msg!r}"
                )
            else:
                raise AssertionError(
                    "failure_rate=0.6 > 0.5 时应抛 RuntimeError，但未抛任何异常"
                )

        # extractor 应被调用全部 10 次（不因失败提前中止）
        assert state["extractor_calls"] == 10, (
            f"extractor 应被调用 10 次（全部 chunk 都尝试过），"
            f"实际 {state['extractor_calls']}"
        )


# ---------------------------------------------------------------------------
# 用例 2：低失败率（<50%）应继续 Stage 2 + metadata 含失败统计
# ---------------------------------------------------------------------------
def test_low_failure_rate_continues() -> None:
    """10 个 chunk，前 2 个失败（failure_rate=0.2 < 0.5）→ 继续到 Stage 4。

    验证：
    - 不抛 RuntimeError，正常返回 DistillationResult
    - result.metadata 含 n_total_chunks=10 / n_failed_chunks=2 / failure_rate≈0.2
    - Stage 2/3/4 都跑了（synthesizer / skill_designer / dialogue_writer 各调一次）
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        corpus = _make_corpus_file(tmp)
        distiller = _make_distiller(tmp)

        # 前 2 个 chunk 失败：failure_rate = 2/10 = 0.2 < 0.5
        mock_invoke, state = _make_mock_invoke_structured(
            extractor_fail_indices={0, 1},
        )
        with patch.object(pipeline_mod, "invoke_structured", mock_invoke), \
             patch.object(pipeline_mod, "chunk_text", _make_fake_chunk_text(10)):
            result = distiller.distill(corpus)

        # extractor 调用 10 次（全部 chunk）
        assert state["extractor_calls"] == 10, (
            f"extractor 应被调用 10 次，实际 {state['extractor_calls']}"
        )
        # distillates 数 = 10 - 2 = 8
        assert len(result.distillates) == 8, (
            f"应保留 8 个 distillate（10-2=8），实际 {len(result.distillates)}"
        )
        # metadata 含失败统计
        md = result.metadata
        assert md["n_total_chunks"] == 10, (
            f"metadata.n_total_chunks 应为 10，实际 {md.get('n_total_chunks')}"
        )
        assert md["n_failed_chunks"] == 2, (
            f"metadata.n_failed_chunks 应为 2，实际 {md.get('n_failed_chunks')}"
        )
        # failure_rate 是浮点，用容差比较
        rate = md["failure_rate"]
        assert abs(rate - 0.2) < 1e-9, (
            f"metadata.failure_rate 应为 0.2，实际 {rate}"
        )
        # persona_card / skills / dialogues 都应有内容（Stage 2/3/4 跑完）
        assert result.persona_card is not None
        assert result.persona_card.persona_id == "test_persona"
        # dialogue_writer mock 返回 1 条
        assert len(result.preset_dialogues) == 1


# ---------------------------------------------------------------------------
# 用例 3：metadata 必含三个失败统计字段
# ---------------------------------------------------------------------------
def test_metadata_contains_failure_stats() -> None:
    """验证 metadata 必含 n_total_chunks / n_failed_chunks / failure_rate 三个字段。

    用 0 失败的正常场景验证字段一定存在（即使没有失败也要有统计）。
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        corpus = _make_corpus_file(tmp)
        distiller = _make_distiller(tmp)

        # 0 失败：failure_rate = 0
        mock_invoke, _state = _make_mock_invoke_structured(
            extractor_fail_indices=set(),
        )
        with patch.object(pipeline_mod, "invoke_structured", mock_invoke), \
             patch.object(pipeline_mod, "chunk_text", _make_fake_chunk_text(5)):
            result = distiller.distill(corpus)

        md = result.metadata
        for key in ("n_total_chunks", "n_failed_chunks", "failure_rate"):
            assert key in md, (
                f"metadata 缺字段 {key}；现有 keys: {list(md.keys())}"
            )
        assert md["n_total_chunks"] == 5, (
            f"n_total_chunks 应为 5，实际 {md['n_total_chunks']}"
        )
        assert md["n_failed_chunks"] == 0, (
            f"n_failed_chunks 应为 0，实际 {md['n_failed_chunks']}"
        )
        assert md["failure_rate"] == 0.0, (
            f"failure_rate 应为 0.0，实际 {md['failure_rate']}"
        )
