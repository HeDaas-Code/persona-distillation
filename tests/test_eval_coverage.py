"""``eval.coverage.evaluate`` 单元测试。

Issue #13：已转为 pytest 风格——``test_*`` 函数由 pytest 自动收集，
直接用 ``assert`` 断言，不再需要 ``main()`` / ``try/except`` 包装。
覆盖度评估器是纯规则，无需 LLM。

跑法：``python -m pytest tests/test_eval_coverage.py -v``
"""
from __future__ import annotations

from persona_distillation.eval import coverage
from persona_distillation.schemas import (
    AntiPattern,
    DecisionHeuristic,
    ExpressionDNA,
    HonestBoundary,
    MentalModel,
    PersonaCard,
    PersonaSkill,
    VerificationResult,
)


# ---------------------------------------------------------------------------
# 辅助：构造 DNA 元素
# ---------------------------------------------------------------------------
def _passing_model(name: str = "聚焦即说不") -> MentalModel:
    """构造三重验证全部通过的心智模型。"""
    return MentalModel(
        name=name,
        principle=f"{name}的核心原理",
        verification=VerificationResult(
            cross_domain=True,
            cross_domain_evidence=["[产品]", "[招聘]"],
            generative=True,
            generative_example="问如何扩张→先问能砍掉什么",
            exclusive=True,
            exclusivity_note="多数人靠加法扩张",
        ),
    )


def _failing_model(name: str = "通用常识") -> MentalModel:
    """构造三重验证未通过的心智模型（全 False）。"""
    return MentalModel(
        name=name,
        principle="要努力",
        verification=VerificationResult(),  # cross_domain/generative/exclusive 全 False
    )


def _full_dna() -> ExpressionDNA:
    """构造 spec 场景的 DNA：vocab=12 / rhythm=非空 / tics=5 / metaphors=3 / openings=2。"""
    return ExpressionDNA(
        vocabulary=[f"词{i}" for i in range(12)],  # 12 个，cap=15
        rhythm="短句、极端确定",  # 非空
        rhetorical_tics=[f"修辞{i}" for i in range(5)],  # 5 个，cap=5
        signature_metaphors=[f"比喻{i}" for i in range(3)],  # 3 个，cap=3
        opening_samples=[f"开场{i}" for i in range(2)],  # 2 个，cap=2
    )


def _full_card(persona_id: str = "arakawa_sensei") -> PersonaCard:
    """spec 场景完整卡：mm=3 全过 / ap=4 / hb=2 / dh=5。"""
    return PersonaCard(
        persona_id=persona_id,
        display_name="荒川老师",
        system_prompt="[身份] 语文老师\n[性格] 严厉但关心学生",
        expression_dna=_full_dna(),
        mental_models=[_passing_model(f"模型{i}") for i in range(3)],
        anti_patterns=[
            AntiPattern(pattern=f"反模式{i}", reason="拒绝理由", evidence="证据")
            for i in range(4)
        ],
        honest_boundaries=[
            HonestBoundary(limitation=f"边界{i}", reason="原因") for i in range(2)
        ],
        decision_heuristics=[
            DecisionHeuristic(rule=f"规则{i}", trigger="触发", example="示例")
            for i in range(5)
        ],
    )


# ---------------------------------------------------------------------------
# SubTask 11.1 用例
# ---------------------------------------------------------------------------
def test_full_card_coverage() -> None:
    """完整 PersonaCard（spec 场景）覆盖度评估。"""
    card = _full_card()
    score = coverage.evaluate(card)

    # 表达 DNA 丰富度接近 1.0（5 子字段都达上限，仅 vocab=12/15=0.8 拉低）
    # 0.8*0.30 + 1.0*0.15 + 1.0*0.20 + 1.0*0.20 + 1.0*0.15 = 0.94
    assert score.expression_dna_richness >= 0.9, (
        f"expression_dna_richness 应接近 1.0，实际 {score.expression_dna_richness}"
    )
    assert score.expression_dna_richness <= 1.0

    # 心智模型数 == 3，全部通过 → 通过率 1.0
    assert score.mental_model_count == 3, (
        f"mental_model_count 应为 3，实际 {score.mental_model_count}"
    )
    assert score.mental_model_verification_pass_rate == 1.0, (
        f"三重验证通过率应为 1.0，实际 {score.mental_model_verification_pass_rate}"
    )

    # 反模式数 == 4
    assert score.anti_pattern_count == 4, (
        f"anti_pattern_count 应为 4，实际 {score.anti_pattern_count}"
    )

    # total_score > 0.7
    assert score.total_score > 0.7, (
        f"total_score 应 > 0.7，实际 {score.total_score}"
    )
    assert 0.0 <= score.total_score <= 1.0

    # details dict 含各分项原始数值
    expected_keys = {
        "vocabulary_count",
        "rhythm_present",
        "rhetorical_tics_count",
        "signature_metaphors_count",
        "opening_samples_count",
        "expression_dna_richness",
        "mental_model_count",
        "mental_model_passed",
        "mental_model_verification_pass_rate",
        "anti_pattern_count",
        "honest_boundary_count",
        "decision_heuristic_count",
        "weights",
    }
    missing = expected_keys - set(score.details.keys())
    assert not missing, f"details 缺字段: {missing}，实际 keys={list(score.details.keys())}"
    # 不应有 warning（卡是完整的）
    assert "warning" not in score.details, (
        f"完整卡不应有 warning，实际 details={score.details}"
    )


def test_empty_card_coverage() -> None:
    """DNA 全空 PersonaCard——total_score 接近 0，details 含 warning，不抛异常。"""
    card = PersonaCard(
        persona_id="empty_persona",
        system_prompt="(空 system prompt)",
        # expression_dna / mental_models / ... 全部默认空
    )
    # 不抛异常
    score = coverage.evaluate(card)

    # total_score 接近 0（< 0.1）
    assert score.total_score < 0.1, (
        f"空卡 total_score 应 < 0.1，实际 {score.total_score}"
    )
    assert score.total_score >= 0.0

    # details 含 warning 字段
    assert "warning" in score.details, (
        f"空卡 details 应含 warning，实际 keys={list(score.details.keys())}"
    )
    assert "回填失败" in score.details["warning"], (
        f"warning 文案应含'回填失败'，实际: {score.details['warning']!r}"
    )

    # 各分项计数为 0
    assert score.expression_dna_richness == 0.0
    assert score.mental_model_count == 0
    assert score.mental_model_verification_pass_rate == 0.0
    assert score.anti_pattern_count == 0
    assert score.honest_boundary_count == 0
    assert score.decision_heuristic_count == 0


def test_partial_verification() -> None:
    """三重验证部分通过：5 个模型，3 过 2 不过——pass_rate=0.6，total_score 反映通过率。"""
    # 5 个模型：前 3 个通过，后 2 个失败
    models = [_passing_model(f"通过模型{i}") for i in range(3)]
    models += [_failing_model(f"失败模型{i}") for i in range(2)]

    card = PersonaCard(
        persona_id="partial_persona",
        system_prompt="[身份] 测试人物",
        expression_dna=_full_dna(),
        mental_models=models,
        anti_patterns=[
            AntiPattern(pattern=f"反模式{i}", reason="理由")
            for i in range(3)
        ],
        honest_boundaries=[HonestBoundary(limitation="边界", reason="原因")],
        decision_heuristics=[
            DecisionHeuristic(rule=f"规则{i}", trigger="触发") for i in range(3)
        ],
    )
    score = coverage.evaluate(card)

    assert score.mental_model_count == 5, (
        f"mental_model_count 应为 5，实际 {score.mental_model_count}"
    )
    # pass_rate = 3/5 = 0.6
    assert abs(score.mental_model_verification_pass_rate - 0.6) < 1e-6, (
        f"三重验证通过率应为 0.6，实际 {score.mental_model_verification_pass_rate}"
    )

    # total_score 反映通过率：与"5 个全过"版本对比，部分通过的分数应更低
    all_pass_card = card.model_copy(update={
        "mental_models": [_passing_model(f"全过模型{i}") for i in range(5)],
    })
    all_pass_score = coverage.evaluate(all_pass_card)
    assert score.total_score < all_pass_score.total_score, (
        f"部分通过（{score.total_score}）应 < 全过（{all_pass_score.total_score}），"
        f"pass_rate={score.mental_model_verification_pass_rate}"
    )

    # details 也应反映
    assert score.details["mental_model_passed"] == 3, (
        f"details.mental_model_passed 应为 3，实际 {score.details['mental_model_passed']}"
    )
    assert abs(score.details["mental_model_verification_pass_rate"] - 0.6) < 1e-6


def test_skills_and_distillates_ignored() -> None:
    """coverage 不依赖 skills/distillates——传 None 与传非 None 结果一致。"""
    card = _full_card()

    # 传 None
    score_none = coverage.evaluate(card, skills=None, distillates=None)

    # 传非 None（构造一些非空但无意义的 skills / distillates）
    from persona_distillation.schemas import (
        Distillate,
        PersonaSignal,
        SignalCategory,
    )
    skills = [
        PersonaSkill(
            name="some-skill",
            description="测试 skill，不应影响 coverage",
        )
    ]
    distillates = [
        Distillate(
            source_file="a.txt",
            chunk_index=0,
            char_start=0,
            char_end=10,
            signals=[
                PersonaSignal(
                    category=SignalCategory.CATCHPHRASE,
                    content="嘛",
                )
            ],
            summary="测试摘要",
        )
    ]
    score_non_none = coverage.evaluate(card, skills=skills, distillates=distillates)

    # 两次结果应完全一致（coverage 只看 card）
    assert score_none.total_score == score_non_none.total_score, (
        f"coverage 不应依赖 skills/distillates，"
        f"None={score_none.total_score} vs 非None={score_non_none.total_score}"
    )
    assert (
        score_none.expression_dna_richness
        == score_non_none.expression_dna_richness
    )
    assert score_none.mental_model_count == score_non_none.mental_model_count
    assert (
        score_none.mental_model_verification_pass_rate
        == score_non_none.mental_model_verification_pass_rate
    )
    assert score_none.details == score_non_none.details, (
        "两次评估的 details 应完全一致"
    )
