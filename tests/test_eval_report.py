"""``eval.report.build_report`` 单元测试。

Issue #13：已转为 pytest 风格——``test_*`` 函数由 pytest 自动收集，
直接用 ``assert`` 断言，不再需要 ``main()`` / ``try/except`` 包装。
所有 LLM 路径走 ``FakeReportLLM``，不调真实 API。

跑法：``python -m pytest tests/test_eval_report.py -v``
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from persona_distillation.eval import report
from persona_distillation.schemas import (
    AntiPattern,
    DecisionHeuristic,
    EvalReport,
    ExpressionDNA,
    HonestBoundary,
    MentalModel,
    PersonaCard,
    VerificationResult,
)


# ---------------------------------------------------------------------------
# FakeLLM：根据 SystemMessage 内容区分 fidelity / identifiability probe / guess
# ---------------------------------------------------------------------------
class _FakeResp:
    """模拟 LangChain 返回对象：有 ``.content`` 属性。"""

    def __init__(self, content: Any) -> None:
        self.content = content


class FakeReportLLM:
    """报告评估用 Fake LLM：``invoke`` 根据 system prompt 内容区分调用方。

    三个调用方（参考 ``eval/fidelity.py`` / ``eval/identifiability.py``）：
    - fidelity judge：system 含 "人格蒸馏质量评估"（``_JUDGE_SYSTEM``）
    - identifiability probe：system 含 "保持人物口吻"（``_PROBE_SYSTEM_SUFFIX``，
      拼在 card.system_prompt 之后）
    - identifiability guess：system 含 "人物识别"（``_GUESS_SYSTEM``）

    - ``raise_on_fidelity=True`` 时，fidelity 调用抛异常（用于测失败兜底）
    - ``call_count`` / ``fidelity_calls`` / ``probe_calls`` / ``guess_calls`` 分别计数
    """

    def __init__(
        self,
        fidelity_resp: str = '{"score": 0.85, "reasons": ["风格还原到位", "心智模型准确", "价值底线清晰"]}',
        probe_resp: str | None = None,
        guess_resp: str = '{"name": "荒川老师", "confidence": 0.9}',
        raise_on_fidelity: bool = False,
    ) -> None:
        self.fidelity_resp = fidelity_resp
        # probe 回答：5 段 Q&A，与 _PROBES 数量对齐
        self.probe_resp = probe_resp or (
            "Q: 你怎么看待朋友？\nA: 朋友是镜子。\n\n"
            "Q: 最近在忙什么？\nA: 嘛，再看看吧。\n\n"
            "Q: 遇到不公时会怎样？\nA: 直接说。\n\n"
            "Q: 你最看重什么？\nA: 专注。\n\n"
            "Q: 用一个比喻描述自己。\nA: 像一本旧书。"
        )
        self.guess_resp = guess_resp
        self.raise_on_fidelity = raise_on_fidelity
        self.call_count = 0
        self.fidelity_calls = 0
        self.probe_calls = 0
        self.guess_calls = 0

    def invoke(self, messages: list) -> _FakeResp:
        self.call_count += 1
        # 第 1 条应是 SystemMessage，取其 content 判断调用方
        sys_content = ""
        if messages:
            first = messages[0]
            content = getattr(first, "content", "") or ""
            sys_content = str(content)

        # fidelity judge：_JUDGE_SYSTEM 含 "人格蒸馏质量评估"
        if "人格蒸馏质量评估" in sys_content:
            self.fidelity_calls += 1
            if self.raise_on_fidelity:
                raise RuntimeError("fidelity judge down")
            return _FakeResp(self.fidelity_resp)
        # identifiability guess：_GUESS_SYSTEM 含 "人物识别"
        if "人物识别" in sys_content:
            self.guess_calls += 1
            return _FakeResp(self.guess_resp)
        # 兜底：identifiability probe（system = card.system_prompt + _PROBE_SYSTEM_SUFFIX）
        self.probe_calls += 1
        return _FakeResp(self.probe_resp)


# ---------------------------------------------------------------------------
# 辅助：构造完整 PersonaCard（与 test_eval_coverage 同款）
# ---------------------------------------------------------------------------
def _passing_model(name: str = "聚焦即说不") -> MentalModel:
    return MentalModel(
        name=name,
        principle=f"{name}的核心原理",
        verification=VerificationResult(
            cross_domain=True,
            generative=True,
            exclusive=True,
        ),
    )


def _full_card(persona_id: str = "arakawa_sensei") -> PersonaCard:
    """构造完整 PersonaCard，display_name 与 FakeReportLLM 默认 guess 名一致。"""
    return PersonaCard(
        persona_id=persona_id,
        display_name="荒川老师",
        system_prompt="[身份] 语文老师\n[性格] 严厉但关心学生",
        expression_dna=ExpressionDNA(
            vocabulary=[f"词{i}" for i in range(12)],
            rhythm="短句、极端确定",
            rhetorical_tics=[f"修辞{i}" for i in range(5)],
            signature_metaphors=[f"比喻{i}" for i in range(3)],
            opening_samples=[f"开场{i}" for i in range(2)],
        ),
        mental_models=[_passing_model(f"模型{i}") for i in range(3)],
        anti_patterns=[
            AntiPattern(pattern=f"反模式{i}", reason="理由") for i in range(4)
        ],
        honest_boundaries=[
            HonestBoundary(limitation=f"边界{i}", reason="原因") for i in range(2)
        ],
        decision_heuristics=[
            DecisionHeuristic(rule=f"规则{i}", trigger="触发") for i in range(5)
        ],
    )


# ---------------------------------------------------------------------------
# SubTask 11.2 用例
# ---------------------------------------------------------------------------
def test_offline_report() -> None:
    """llm=None 离线模式：只跑 coverage，fidelity/identifiability 为 None。"""
    card = _full_card()
    rep = report.build_report(card, llm=None)

    # fidelity / identifiability 应为 None
    assert rep.fidelity is None, (
        f"离线模式 fidelity 应为 None，实际 {rep.fidelity}"
    )
    assert rep.identifiability is None, (
        f"离线模式 identifiability 应为 None，实际 {rep.identifiability}"
    )

    # coverage 应非 None
    assert rep.coverage is not None
    assert rep.coverage.total_score > 0

    # overall_score == coverage.total_score（离线仅基于 coverage）
    assert abs(rep.overall_score - rep.coverage.total_score) < 1e-9, (
        f"离线 overall_score 应 == coverage.total_score，"
        f"实际 overall={rep.overall_score} coverage={rep.coverage.total_score}"
    )
    assert 0.0 <= rep.overall_score <= 1.0


def test_online_report() -> None:
    """llm=FakeLLM 在线模式：三个评估器都跑，overall 加权正确。"""
    card = _full_card()
    llm = FakeReportLLM()  # 默认 fidelity=0.85 / guess=荒川老师（匹配 display_name → correct=True）
    rep = report.build_report(card, llm=llm)

    # 三个评估器都跑
    assert rep.fidelity is not None, "在线模式 fidelity 不应为 None"
    assert rep.identifiability is not None, "在线模式 identifiability 不应为 None"
    assert rep.coverage is not None

    # fidelity 调了 1 次，identifiability 调了 2 次（probe + guess）
    assert llm.fidelity_calls == 1, (
        f"fidelity 应调 1 次，实际 {llm.fidelity_calls}"
    )
    assert llm.probe_calls == 1, (
        f"identifiability probe 应调 1 次，实际 {llm.probe_calls}"
    )
    assert llm.guess_calls == 1, (
        f"identifiability guess 应调 1 次，实际 {llm.guess_calls}"
    )

    # overall 加权：coverage*0.3 + fidelity*0.4 + identifiability*0.3
    # identifiability correct=True → ident_score=1.0
    fid_score = rep.fidelity.score
    ident_score = 1.0 if rep.identifiability.correct else (
        rep.identifiability.confidence * 0.5
    )
    expected_overall = (
        rep.coverage.total_score * report.W_COVERAGE
        + fid_score * report.W_FIDELITY
        + ident_score * report.W_IDENTIFIABILITY
    )
    expected_overall = max(0.0, min(1.0, expected_overall))
    assert abs(rep.overall_score - expected_overall) < 1e-9, (
        f"overall 加权不正确：实际 {rep.overall_score}，"
        f"期望 {expected_overall}（cov={rep.coverage.total_score}*0.3 + "
        f"fid={fid_score}*0.4 + ident={ident_score}*0.3）"
    )

    # fidelity 非失败分
    assert rep.fidelity.score >= 0, (
        f"fidelity 不应失败，score={rep.fidelity.score}"
    )
    assert len(rep.fidelity.reasons) == 3, (
        f"fidelity reasons 应有 3 条，实际 {len(rep.fidelity.reasons)}"
    )


def test_fidelity_failure() -> None:
    """FakeLLM 在 fidelity 调用抛异常：fidelity.score==-1.0，overall 中 fidelity 计 0，不抛异常。"""
    card = _full_card()
    # raise_on_fidelity=True：fidelity 调用抛异常，但 probe/guess 正常
    llm = FakeReportLLM(raise_on_fidelity=True)
    # 不抛异常（fidelity.evaluate 内部 catch）
    rep = report.build_report(card, llm=llm)

    # fidelity 应返回失败分
    assert rep.fidelity is not None
    assert rep.fidelity.score == -1.0, (
        f"fidelity 失败时 score 应为 -1.0，实际 {rep.fidelity.score}"
    )

    # identifiability 仍正常跑（fidelity 失败不影响 identifiability）
    assert rep.identifiability is not None, (
        "fidelity 失败不应影响 identifiability 运行"
    )

    # overall 中 fidelity 计 0：overall = coverage*0.3 + 0*0.4 + ident_score*0.3
    ident_score = 1.0 if rep.identifiability.correct else (
        rep.identifiability.confidence * 0.5
    )
    if rep.identifiability.confidence < 0:
        ident_score = 0.0
    expected_overall = (
        rep.coverage.total_score * report.W_COVERAGE
        + 0.0 * report.W_FIDELITY  # fidelity 失败计 0
        + ident_score * report.W_IDENTIFIABILITY
    )
    expected_overall = max(0.0, min(1.0, expected_overall))
    assert abs(rep.overall_score - expected_overall) < 1e-9, (
        f"fidelity 失败时 overall 应计 fidelity=0：实际 {rep.overall_score}，"
        f"期望 {expected_overall}（cov*0.3 + 0 + ident*0.3）"
    )

    # 验证 fidelity 确实被调过（且抛了异常但仍计入 1 次）
    assert llm.fidelity_calls == 1, (
        f"fidelity 应被调 1 次（即便抛异常），实际 {llm.fidelity_calls}"
    )


def test_identifiability_correct() -> None:
    """FakeLLM 猜对名字：correct==True，identifiability 分项计 1.0。"""
    card = _full_card()  # display_name="荒川老师"
    # guess 返回 "荒川老师"，与 display_name 完全匹配 → correct=True
    llm = FakeReportLLM(
        guess_resp='{"name": "荒川老师", "confidence": 0.9}',
    )
    rep = report.build_report(card, llm=llm)

    assert rep.identifiability is not None, "identifiability 不应为 None"

    # correct == True
    assert rep.identifiability.correct is True, (
        f"猜对名字时 correct 应为 True，实际 {rep.identifiability.correct}"
    )
    # guessed_name 应非空
    assert rep.identifiability.guessed_name, (
        f"guessed_name 不应为空，实际 {rep.identifiability.guessed_name!r}"
    )
    # confidence 应在 [0,1]（非 -1 失败）
    assert 0.0 <= rep.identifiability.confidence <= 1.0, (
        f"confidence 应在 [0,1]，实际 {rep.identifiability.confidence}"
    )

    # identifiability 分项计 1.0：correct 时 ident_score=1.0
    # overall = coverage*0.3 + fidelity*0.4 + 1.0*0.3
    expected_overall = (
        rep.coverage.total_score * report.W_COVERAGE
        + rep.fidelity.score * report.W_FIDELITY
        + 1.0 * report.W_IDENTIFIABILITY  # correct → ident_score=1.0
    )
    expected_overall = max(0.0, min(1.0, expected_overall))
    assert abs(rep.overall_score - expected_overall) < 1e-9, (
        f"correct=True 时 identifiability 分项应计 1.0："
        f"实际 overall={rep.overall_score}，"
        f"期望 {expected_overall}（cov*0.3 + fid*0.4 + 1.0*0.3）"
    )

    # probes 应非空（probe 生成成功）
    assert len(rep.identifiability.probes) >= 1, (
        f"probes 不应为空，实际 {rep.identifiability.probes}"
    )


def test_eval_report_serialization() -> None:
    """EvalReport round-trip：build → model_dump_json → from_json → 字段一致。"""
    card = _full_card()
    llm = FakeReportLLM()
    rep = report.build_report(card, llm=llm)

    # 序列化
    json_str = rep.model_dump_json(indent=2)
    assert json_str, "model_dump_json 不应返回空字符串"

    # 写到临时文件，再用 from_json 加载
    with tempfile.TemporaryDirectory() as td:
        report_file = Path(td) / "eval_report.json"
        report_file.write_text(json_str, encoding="utf-8")

        # from_json 接受文件路径
        loaded = EvalReport.from_json(report_file)

        # 字段一致
        assert loaded.overall_score == rep.overall_score, (
            f"overall_score 不一致：{loaded.overall_score} vs {rep.overall_score}"
        )
        assert loaded.coverage.total_score == rep.coverage.total_score, (
            f"coverage.total_score 不一致"
        )
        assert (
            loaded.coverage.expression_dna_richness
            == rep.coverage.expression_dna_richness
        )
        assert loaded.coverage.mental_model_count == rep.coverage.mental_model_count
        assert (
            loaded.coverage.mental_model_verification_pass_rate
            == rep.coverage.mental_model_verification_pass_rate
        )

        # fidelity round-trip
        assert loaded.fidelity is not None, "loaded fidelity 不应为 None"
        assert loaded.fidelity.score == rep.fidelity.score, (
            f"fidelity.score 不一致：{loaded.fidelity.score} vs {rep.fidelity.score}"
        )
        assert loaded.fidelity.reasons == rep.fidelity.reasons, (
            f"fidelity.reasons 不一致：{loaded.fidelity.reasons} vs {rep.fidelity.reasons}"
        )

        # identifiability round-trip
        assert loaded.identifiability is not None, "loaded identifiability 不应为 None"
        assert loaded.identifiability.correct == rep.identifiability.correct
        assert loaded.identifiability.guessed_name == rep.identifiability.guessed_name
        assert loaded.identifiability.confidence == rep.identifiability.confidence

        # from_json 也能接受目录路径
        loaded_from_dir = EvalReport.from_json(Path(td))
        assert loaded_from_dir.overall_score == rep.overall_score, (
            "from_json 接受目录路径时应等价于接受文件路径"
        )

    # to_markdown 也能正常输出（不抛异常）
    md = rep.to_markdown()
    assert "蒸馏质量评估报告" in md, (
        f"to_markdown 应含标题，实际: {md[:80]!r}"
    )
    assert "覆盖度" in md
    assert "忠实度" in md
    assert "可识别度" in md
