"""忠实度评估器：LLM-as-judge 对比 PersonaCard 与原语料，打分忠实度。

让独立 LLM 作为评委，基于 distillates 摘要与 PersonaCard 摘要，
判断该卡对原人物表达风格 / 心智模型 / 价值底线的还原程度，输出 0~1 分 + 3 条理由。
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from persona_distillation.schemas import (
    Distillate,
    FidelityScore,
    PersonaCard,
)

logger = logging.getLogger(__name__)


_JUDGE_SYSTEM = (
    "你是人格蒸馏质量评估的独立评委。基于原语料摘要与一张 PersonaCard，"
    "判断该卡对原人物表达风格 / 心智模型 / 价值底线的还原程度。"
    "严格输出 JSON：{\"score\": <0~1 浮点>, \"reasons\": [<3 条短理由>]}"
)


def _extract_content(resp: Any) -> str:
    """从 LangChain 响应中提取纯文本，兼容 content 为 list 的情况。"""
    content = getattr(resp, "content", "") or ""
    if isinstance(content, list):
        # langchain 新版 content 可能是 [{"type": "text", "text": "..."}, ...]
        content = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in content
        )
    return str(content).strip()


def _parse_json(text: str) -> dict:
    """从可能含 markdown ```json``` 包裹的文本中解析 JSON 对象。

    策略：先去掉 ``` 包裹（若有），再取第一个 ``{`` 到最后一个 ``}`` 的子串，
    最后 ``json.loads``。失败抛原异常给上层 catch。
    """
    text = text.strip()
    # 去掉 markdown ```json ... ``` 或 ``` ... ``` 包裹
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]  # 去掉首行 ``` 或 ```json
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
    # 取第一个 { 到最后一个 } 之间内容
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        text = text[start : end + 1]
    return json.loads(text)


def _build_corpus_digest(distillates: list[Distillate]) -> str:
    """把 distillates 的 signals + summary 拼成人类可读语料摘要。"""
    if not distillates:
        return "(无原语料摘要)"
    parts: list[str] = []
    for idx, d in enumerate(distillates, 1):
        parts.append(f"### 分块 {idx}（来源: {d.source_file}, chunk={d.chunk_index}）")
        if d.summary:
            parts.append(f"摘要: {d.summary}")
        for sig in d.signals:
            evidence = f"｜证据: {sig.evidence}" if sig.evidence else ""
            parts.append(f"- [{sig.category.value}] {sig.content}{evidence}")
    return "\n".join(parts)


def _build_card_digest(card: PersonaCard) -> str:
    """把 PersonaCard 的 system_prompt + DNA 五层拼成人类可读摘要。"""
    edna = card.expression_dna
    parts: list[str] = []
    parts.append(f"### PersonaCard (persona_id={card.persona_id})")
    parts.append("## system_prompt")
    parts.append(card.system_prompt or "(空)")
    parts.append("## expression_dna")
    parts.append(f"- vocabulary: {edna.vocabulary}")
    parts.append(f"- rhythm: {edna.rhythm or '(空)'}")
    parts.append(f"- rhetorical_tics: {edna.rhetorical_tics}")
    parts.append(f"- signature_metaphors: {edna.signature_metaphors}")
    parts.append(f"- opening_samples: {edna.opening_samples}")
    parts.append("## mental_models")
    for m in card.mental_models:
        v = m.verification
        parts.append(
            f"- {m.name}：{m.principle}"
            f"（cross_domain={v.cross_domain}, generative={v.generative}, "
            f"exclusive={v.exclusive}）"
        )
    parts.append("## decision_heuristics")
    for h in card.decision_heuristics:
        parts.append(f"- {h.rule}（触发: {h.trigger}）")
    parts.append("## anti_patterns")
    for ap in card.anti_patterns:
        parts.append(f"- {ap.pattern}（理由: {ap.reason}）")
    parts.append("## honest_boundaries")
    for hb in card.honest_boundaries:
        parts.append(f"- {hb.limitation}（理由: {hb.reason}）")
    return "\n".join(parts)


def evaluate(
    card: PersonaCard,
    distillates: list[Distillate],
    llm: Any,
) -> FidelityScore:
    """让独立 LLM 作为 judge 评估 PersonaCard 对原人物的忠实还原度。

    Args:
        card: 待评估的人格卡
        distillates: 蒸馏中间产物，作为"原语料"参考喂给 judge
        llm: LangChain ``BaseChatModel``，作为独立 judge

    Returns:
        :class:`FidelityScore`——score ∈ [0,1]；judge 调用失败时返回
        ``score=-1.0`` 并把异常信息塞进 reasons，不抛异常。
    """
    try:
        corpus_digest = _build_corpus_digest(distillates)
        card_digest = _build_card_digest(card)
        user_prompt = (
            "## 原语料摘要\n\n"
            f"{corpus_digest}\n\n"
            "## PersonaCard 摘要\n\n"
            f"{card_digest}\n\n"
            "## 任务\n"
            "基于原语料，判断该 PersonaCard 在多大程度上忠实还原了人物？\n"
            "打分 0~1（1 = 完美还原，0 = 完全偏离），并给出 3 条理由。\n"
            '严格输出 JSON：{"score": <float>, "reasons": ["...", "...", "..."]}'
        )
        resp = llm.invoke([
            SystemMessage(content=_JUDGE_SYSTEM),
            HumanMessage(content=user_prompt),
        ])
        text = _extract_content(resp)
        data = _parse_json(text)

        score = float(data.get("score", 0.0))
        score = max(0.0, min(1.0, score))

        reasons_raw = data.get("reasons", [])
        if isinstance(reasons_raw, str):
            reasons = [reasons_raw]
        else:
            reasons = [str(r) for r in reasons_raw]

        return FidelityScore(score=score, reasons=reasons)
    except Exception as e:
        logger.warning("fidelity 评估失败: %s", e)
        return FidelityScore(score=-1.0, reasons=[f"judge 调用失败: {e}"])
