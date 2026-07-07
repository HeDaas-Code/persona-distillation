"""评估报告汇总器：合并三个评估器结果。

对外暴露 :func:`build_report`，根据是否传入 ``llm`` 决定跑离线（仅 coverage）
还是完整（coverage + fidelity + identifiability）评估，并加权得到 overall_score。
"""
from __future__ import annotations

import logging
from typing import Any

from persona_distillation.eval import coverage, fidelity, identifiability
from persona_distillation.schemas import (
    Distillate,
    EvalReport,
    PersonaCard,
    PersonaSkill,
)

logger = logging.getLogger(__name__)

# overall_score 加权权重（合计 1.0）
W_COVERAGE = 0.3
W_FIDELITY = 0.4
W_IDENTIFIABILITY = 0.3


def build_report(
    card: PersonaCard,
    skills: list[PersonaSkill] | None = None,
    distillates: list[Distillate] | None = None,
    llm: Any = None,
) -> EvalReport:
    """构建完整评估报告。

    Args:
        card: 待评估的人格卡
        skills: 人格 skill 列表（参与 coverage 评估的扩展位，当前未使用）
        distillates: 蒸馏中间产物（fidelity/identifiability 用作"原语料"参考）
        llm: LangChain ``BaseChatModel``；``None`` 时只跑 coverage（离线模式），
            fidelity / identifiability 字段为 None

    Returns:
        :class:`EvalReport`——含 coverage / fidelity / identifiability /
        overall_score / created_at。在线模式下 fidelity / identifiability 任一
        失败（score=-1 / confidence=-1）该分项计 0，不影响其它评估。
    """
    cov = coverage.evaluate(card, skills, distillates)

    if llm is None:
        # 离线模式：只跑 coverage，overall_score 仅基于 coverage
        return EvalReport(
            coverage=cov,
            fidelity=None,
            identifiability=None,
            overall_score=cov.total_score,
        )

    # 在线模式：跑全部三个评估器
    fid = fidelity.evaluate(card, distillates or [], llm)
    ident = identifiability.evaluate(card, distillates or [], llm)

    # overall_score 加权
    # fidelity.score == -1 表示失败，计 0
    fid_score = max(fid.score, 0.0)

    # identifiability：confidence == -1 表示失败，计 0；
    # 否则 correct 时用 1.0（认对了就满分），incorrect 时用 confidence * 0.5
    # （认错但有一定确信度，给半折分）
    if ident.confidence < 0:
        ident_score = 0.0
    elif ident.correct:
        ident_score = 1.0
    else:
        ident_score = ident.confidence * 0.5

    overall = (
        cov.total_score * W_COVERAGE
        + fid_score * W_FIDELITY
        + ident_score * W_IDENTIFIABILITY
    )
    overall = max(0.0, min(1.0, overall))

    return EvalReport(
        coverage=cov,
        fidelity=fid,
        identifiability=ident,
        overall_score=overall,
    )
