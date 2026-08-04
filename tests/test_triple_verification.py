"""``triple_verification`` 单元测试。

覆盖三重验证法的规则式初筛与完整流程：
- ``_keywords``: 中英文关键词提取
- ``_domain_count``: 跨域复现检查
- ``verify_mental_models``: 完整三重验证（跨域/生成力/排他性）
- ``filter_verified``: 验证后仅保留通过的模型
- ``VerificationReport.apply``: 验证结果写回模型
"""

from __future__ import annotations

from persona_distillation.schemas import (
    Distillate,
    MentalModel,
    PersonaSignal,
    SignalCategory,
    VerificationResult,
)
from persona_distillation.triple_verification import (
    VerificationReport,
    _collect_evidence,
    _domain_count,
    _keywords,
    filter_verified,
    verify_mental_models,
)


def _make_model(
    name: str = "test_model",
    principle: str = "test principle",
    *,
    cross_domain: bool = False,
    cross_domain_evidence: list[str] | None = None,
    generative: bool = False,
    generative_example: str = "",
    exclusive: bool = False,
    exclusivity_note: str = "",
) -> MentalModel:
    return MentalModel(
        name=name,
        principle=principle,
        verification=VerificationResult(
            cross_domain=cross_domain,
            cross_domain_evidence=cross_domain_evidence or [],
            generative=generative,
            generative_example=generative_example,
            exclusive=exclusive,
            exclusivity_note=exclusivity_note,
        ),
    )


def _make_distillate(
    signals: list[PersonaSignal] | None = None,
    summary: str = "test summary",
) -> Distillate:
    return Distillate(
        source_file="test.txt",
        chunk_index=0,
        char_start=0,
        char_end=100,
        signals=signals or [],
        summary=summary,
    )


# ---------------------------------------------------------------------------
# _keywords
# ---------------------------------------------------------------------------


class TestKeywords:
    def test_chinese_extraction(self):
        kws = _keywords("专注 是说不对 100 个好主意")
        assert "专注" in kws
        assert "是说不对" in kws
        assert "个好主意" in kws

    def test_english_extraction(self):
        kws = _keywords("focus is saying no to a hundred good ideas")
        assert "focus" in kws
        assert "saying" in kws
        assert "hundred" in kws
        assert "ideas" in kws

    def test_mixed_language(self):
        kws = _keywords("专注 focus 是说不")
        assert any("专注" in k or "focus" in k for k in kws)

    def test_short_words_excluded(self):
        kws = _keywords("a I to is")
        assert "a" not in kws
        assert "I" not in kws

    def test_empty_string(self):
        assert _keywords("") == []

    def test_punctuation_only(self):
        assert _keywords("!!!???.") == []


# ---------------------------------------------------------------------------
# _collect_evidence
# ---------------------------------------------------------------------------


class TestCollectEvidence:
    def test_empty_distillates(self):
        assert _collect_evidence([]) == []

    def test_collects_all_signals(self):
        signals = [
            PersonaSignal(
                category=SignalCategory.CATCHPHRASE,
                content="关键词",
                evidence="他说：关键词在这里",
                salience=0.9,
            ),
            PersonaSignal(
                category=SignalCategory.VALUES,
                content="信念",
                evidence="信念来源",
                salience=0.8,
            ),
        ]
        distillates = [_make_distillate(signals=signals)]
        evidence = _collect_evidence(distillates)
        assert len(evidence) == 2
        assert evidence[0][0] == SignalCategory.CATCHPHRASE
        assert evidence[0][1] == "关键词"
        assert evidence[1][0] == SignalCategory.VALUES

    def test_multiple_distillates(self):
        d1 = _make_distillate(signals=[
            PersonaSignal(category=SignalCategory.CATCHPHRASE, content="a", evidence="e1", salience=0.9),
        ])
        d2 = _make_distillate(signals=[
            PersonaSignal(category=SignalCategory.VALUES, content="b", evidence="e2", salience=0.9),
        ])
        evidence = _collect_evidence([d1, d2])
        assert len(evidence) == 2


# ---------------------------------------------------------------------------
# _domain_count
# ---------------------------------------------------------------------------


class TestDomainCount:
    def test_passes_with_two_domains(self):
        model = _make_model(principle="杠杆 财富 职业")
        evidence = [
            (SignalCategory.CATCHPHRASE, "杠杆在财富领域", "财富领域用杠杆"),
            (SignalCategory.VALUES, "杠杆在职业选择", "职业选择用杠杆"),
        ]
        passed, proofs = _domain_count(model, evidence)
        assert passed is True
        assert len(proofs) >= 2

    def test_fails_with_single_domain(self):
        model = _make_model(principle="杠杆")
        evidence = [
            (SignalCategory.CATCHPHRASE, "杠杆在财富", "财富杠杆"),
            (SignalCategory.CATCHPHRASE, "杠杆在投资", "投资杠杆"),
        ]
        passed, proofs = _domain_count(model, evidence)
        assert passed is False

    def test_no_keywords_fails(self):
        model = _make_model(principle="...")
        evidence = [
            (SignalCategory.CATCHPHRASE, "...", "..."),
        ]
        passed, proofs = _domain_count(model, evidence)
        assert passed is False
        assert proofs == []


# ---------------------------------------------------------------------------
# verify_mental_models
# ---------------------------------------------------------------------------


class TestVerifyMentalModels:
    def test_all_three_pass(self):
        model = _make_model(
            name="反脆弱",
            principle="反脆弱 杠杆 选择权",
            cross_domain=True,
            cross_domain_evidence=["[财富] 反脆弱", "[职业] 反脆弱"],
            generative=True,
            generative_example="如何扩张→先反脆弱设计",
            exclusive=True,
            exclusivity_note="不是所有聪明人都这样想",
        )
        distillates = [_make_distillate(signals=[
            PersonaSignal(category=SignalCategory.CATCHPHRASE, content="反脆弱", evidence="财富领域", salience=0.9),
            PersonaSignal(category=SignalCategory.VALUES, content="反脆弱", evidence="职业领域", salience=0.8),
        ])]
        reports = verify_mental_models([model], distillates)
        assert len(reports) == 1
        assert reports[0].passed is True

    def test_fails_generative_without_example(self):
        model = _make_model(
            cross_domain=True,
            generative=False,
            generative_example="",
            exclusive=True,
            exclusivity_note="unique",
        )
        reports = verify_mental_models([model], [])
        assert len(reports) == 1
        assert reports[0].passed is False
        assert any("生成力" in r for r in reports[0].reasons)

    def test_fails_exclusive_without_note(self):
        model = _make_model(
            cross_domain=True,
            generative=True,
            generative_example="推断",
            exclusive=False,
            exclusivity_note="",
        )
        reports = verify_mental_models([model], [])
        assert reports[0].passed is False
        assert any("排他性" in r for r in reports[0].reasons)

    def test_fails_cross_domain_without_evidence(self):
        model = _make_model(
            cross_domain=False,
            cross_domain_evidence=[],
            generative=True,
            generative_example="推断",
            exclusive=True,
            exclusivity_note="unique",
        )
        reports = verify_mental_models([model], [])
        assert reports[0].passed is False

    def test_llm_initially_passes_but_rule_fails(self):
        model = _make_model(
            cross_domain=True,
            cross_domain_evidence=["[领域A] 证据"],
            generative=True,
            generative_example="",
            exclusive=True,
            exclusivity_note="",
        )
        reports = verify_mental_models([model], [])
        assert reports[0].passed is False

    def test_multiple_models_independent(self):
        good = _make_model(
            name="好模型",
            principle="跨域 复现",
            cross_domain=True,
            cross_domain_evidence=["[A] 跨域", "[B] 复现"],
            generative=True,
            generative_example="新问题立场",
            exclusive=True,
            exclusivity_note="独特",
        )
        bad = _make_model(name="坏模型")
        distillates = [_make_distillate(signals=[
            PersonaSignal(category=SignalCategory.CATCHPHRASE, content="跨域", evidence="A", salience=0.9),
            PersonaSignal(category=SignalCategory.VALUES, content="复现", evidence="B", salience=0.9),
        ])]
        reports = verify_mental_models([good, bad], distillates)
        assert len(reports) == 2
        assert reports[0].passed is True
        assert reports[1].passed is False


# ---------------------------------------------------------------------------
# filter_verified
# ---------------------------------------------------------------------------


class TestFilterVerified:
    def test_filters_only_passed(self):
        good = _make_model(
            name="通过模型",
            cross_domain=True,
            cross_domain_evidence=["[X] 证据1", "[Y] 证据2"],
            generative=True,
            generative_example="推断示例",
            exclusive=True,
            exclusivity_note="独特说明",
        )
        bad = _make_model(name="未通过")
        distillates = [_make_distillate(signals=[
            PersonaSignal(category=SignalCategory.CATCHPHRASE, content="证据1", evidence="X", salience=0.9),
            PersonaSignal(category=SignalCategory.VALUES, content="证据2", evidence="Y", salience=0.9),
        ])]
        passed, reports = filter_verified([good, bad], distillates)
        assert len(passed) == 1
        assert passed[0].name == "通过模型"
        assert len(reports) == 2

    def test_empty_candidates(self):
        passed, reports = filter_verified([], [])
        assert passed == []
        assert reports == []

    def test_all_pass(self):
        m1 = _make_model(
            name="M1",
            cross_domain=True,
            cross_domain_evidence=["[A] 证据", "[B] 证据"],
            generative=True,
            generative_example="推断",
            exclusive=True,
            exclusivity_note="独特",
        )
        distillates = [_make_distillate(signals=[
            PersonaSignal(category=SignalCategory.CATCHPHRASE, content="证据", evidence="A", salience=0.9),
            PersonaSignal(category=SignalCategory.VALUES, content="证据", evidence="B", salience=0.9),
        ])]
        passed, reports = filter_verified([m1], distillates)
        assert len(passed) == 1
        assert len(reports) == 1


# ---------------------------------------------------------------------------
# VerificationReport.apply
# ---------------------------------------------------------------------------


class TestVerificationReportApply:
    def test_apply_updates_model(self):
        model = _make_model(name="test")
        report = VerificationReport(model=model, passed=False, reasons=["排他性失败"])
        updated = report.apply()
        assert updated.name == "test"
        assert isinstance(updated, MentalModel)

    def test_apply_preserves_cross_domain_evidence(self):
        model = _make_model(
            cross_domain=True,
            cross_domain_evidence=["[A] 证据1", "[B] 证据2"],
        )
        report = VerificationReport(model=model, passed=True, reasons=["通过"])
        updated = report.apply()
        assert updated.verification.cross_domain_evidence == ["[A] 证据1", "[B] 证据2"]