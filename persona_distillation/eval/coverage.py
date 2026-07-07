"""覆盖度评估器：纯规则统计 PersonaCard 的 DNA 五层完整度。

无需 LLM，对蒸馏产出的结构化字段做计数与加权，给出 ∈ [0,1] 的覆盖度分数。
"""
from __future__ import annotations

from persona_distillation.schemas import (
    CoverageScore,
    Distillate,
    PersonaCard,
    PersonaSkill,
)


# expression_dna_richness 5 子字段权重（合计 1.0）与上限
_W_VOCAB = 0.30       # 词汇偏好
_W_RHYTHM = 0.15      # 句式节奏
_W_TICS = 0.20        # 修辞习惯
_W_METAPHORS = 0.20   # 标志性比喻
_W_OPENINGS = 0.15    # 开场白示范
_CAP_VOCAB = 15
_CAP_TICS = 5
_CAP_METAPHORS = 3
_CAP_OPENINGS = 2

# total_score 各分项权重（合计 1.0）
_W_EXPR_DNA = 0.25
_W_MENTAL_MODEL = 0.30
_W_ANTI_PATTERN = 0.20
_W_HONEST_BOUNDARY = 0.10
_W_DECISION_HEURISTIC = 0.15
# 各分项数量达上限视为满分
_CAP_MENTAL_MODEL = 5
_CAP_ANTI_PATTERN = 3
_CAP_HONEST_BOUNDARY = 2
_CAP_DECISION_HEURISTIC = 3


def _cap_ratio(count: int, cap: int) -> float:
    """min(count / cap, 1.0)；cap <= 0 时返回 0。"""
    if cap <= 0:
        return 0.0
    return min(count / cap, 1.0)


def evaluate(
    card: PersonaCard,
    skills: list[PersonaSkill] | None = None,
    distillates: list[Distillate] | None = None,
) -> CoverageScore:
    """评估 PersonaCard 的 DNA 五层覆盖度。

    Args:
        card: 待评估的人格卡
        skills: 人格 skill 列表（当前未参与评分，预留扩展位）
        distillates: 蒸馏中间产物（当前未参与评分，预留扩展位）

    Returns:
        :class:`CoverageScore`——含 total_score / expression_dna_richness /
        各分项原始计数 / details 人类可读明细。DNA 全空时不抛异常，
        details 会附 ``warning`` 字段。
    """
    # 1. 统计 expression_dna 5 个子字段数量
    edna = card.expression_dna
    vocab_n = len(edna.vocabulary)
    rhythm_n = 1 if edna.rhythm else 0  # rhythm 是 str，有内容算 1
    tics_n = len(edna.rhetorical_tics)
    metaphors_n = len(edna.signature_metaphors)
    openings_n = len(edna.opening_samples)

    # expression_dna_richness: 5 子字段加权归一化
    expression_dna_richness = (
        _cap_ratio(vocab_n, _CAP_VOCAB) * _W_VOCAB
        + _cap_ratio(rhythm_n, 1) * _W_RHYTHM
        + _cap_ratio(tics_n, _CAP_TICS) * _W_TICS
        + _cap_ratio(metaphors_n, _CAP_METAPHORS) * _W_METAPHORS
        + _cap_ratio(openings_n, _CAP_OPENINGS) * _W_OPENINGS
    )
    # 浮点钳制
    expression_dna_richness = max(0.0, min(1.0, expression_dna_richness))

    # 2. mental_models：数量 + 三重验证通过率
    mm_count = len(card.mental_models)
    if mm_count > 0:
        passed = sum(1 for m in card.mental_models if m.verification.passed)
        pass_rate = passed / mm_count
    else:
        passed = 0
        pass_rate = 0.0

    # 3. anti_patterns / honest_boundaries / decision_heuristics
    ap_n = len(card.anti_patterns)
    hb_n = len(card.honest_boundaries)
    dh_n = len(card.decision_heuristics)

    # 4. DNA 全空检测
    all_empty = (
        vocab_n == 0
        and rhythm_n == 0
        and tics_n == 0
        and metaphors_n == 0
        and openings_n == 0
        and mm_count == 0
        and ap_n == 0
        and hb_n == 0
        and dh_n == 0
    )

    # 5. total_score 加权——mental_model 分项要数量够 + 验证通过
    expr_dna_term = expression_dna_richness * _W_EXPR_DNA
    mental_model_term = (
        _cap_ratio(mm_count, _CAP_MENTAL_MODEL) * pass_rate * _W_MENTAL_MODEL
    )
    anti_pattern_term = _cap_ratio(ap_n, _CAP_ANTI_PATTERN) * _W_ANTI_PATTERN
    honest_boundary_term = _cap_ratio(hb_n, _CAP_HONEST_BOUNDARY) * _W_HONEST_BOUNDARY
    decision_heuristic_term = (
        _cap_ratio(dh_n, _CAP_DECISION_HEURISTIC) * _W_DECISION_HEURISTIC
    )

    total_score = (
        expr_dna_term
        + mental_model_term
        + anti_pattern_term
        + honest_boundary_term
        + decision_heuristic_term
    )
    total_score = max(0.0, min(1.0, total_score))

    # 6. details dict 含各分项原始数值，便于人类可读
    details: dict = {
        "vocabulary_count": vocab_n,
        "rhythm_present": bool(rhythm_n),
        "rhetorical_tics_count": tics_n,
        "signature_metaphors_count": metaphors_n,
        "opening_samples_count": openings_n,
        "expression_dna_richness": round(expression_dna_richness, 4),
        "mental_model_count": mm_count,
        "mental_model_passed": passed,
        "mental_model_verification_pass_rate": round(pass_rate, 4),
        "anti_pattern_count": ap_n,
        "honest_boundary_count": hb_n,
        "decision_heuristic_count": dh_n,
        "weights": {
            "expression_dna": _W_EXPR_DNA,
            "mental_model": _W_MENTAL_MODEL,
            "anti_pattern": _W_ANTI_PATTERN,
            "honest_boundary": _W_HONEST_BOUNDARY,
            "decision_heuristic": _W_DECISION_HEURISTIC,
        },
    }

    # 7. all_empty 时 details 加 warning
    if all_empty:
        details["warning"] = "DNA 五层全空，疑似回填失败"

    return CoverageScore(
        total_score=total_score,
        details=details,
        expression_dna_richness=expression_dna_richness,
        mental_model_count=mm_count,
        mental_model_verification_pass_rate=pass_rate,
        anti_pattern_count=ap_n,
        honest_boundary_count=hb_n,
        decision_heuristic_count=dh_n,
    )
