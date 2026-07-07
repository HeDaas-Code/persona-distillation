"""可识别度评估器：probe + 盲猜，看 PersonaCard 是否能让 judge 认出人物。

两步流程：
1. Probe 生成——让 LLM 扮演 PersonaCard 描述的人物，回答 5 个标准化通用问题；
2. 盲猜——把 probe Q&A 喂给独立 judge（不给 PersonaCard），让其从原语料里
   挑出"这是哪位人物"。最后模糊匹配 judge 猜的名字与 card.display_name /
   persona_id 是否一致。
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from persona_distillation.schemas import (
    Distillate,
    IdentifiabilityScore,
    PersonaCard,
)

logger = logging.getLogger(__name__)


# 5 个标准化 probe 问题——与具体人物无关的通用提问，迫使模型用人格口吻回答
_PROBES = [
    "你怎么看待朋友？",
    "最近在忙什么？",
    "遇到不公时会怎样？",
    "你最看重什么？",
    "用一个比喻描述自己。",
]


_PROBE_SYSTEM_SUFFIX = (
    "\n\n请逐个回答下列问题，每个问题独立成段，"
    "以 \"Q: <问题>\\nA: <回答>\" 格式输出，保持人物口吻。"
)

_GUESS_SYSTEM = (
    "你是人物识别的独立评委。给你一组 probe Q&A（某匿名人物以第一人称回答的"
    "标准化问题），以及一份原语料摘要（含多人原话与事件）。"
    "判断这些 Q&A 最像原语料中的哪位人物。"
    "严格输出 JSON：{\"name\": \"<人物名>\", \"confidence\": <0~1 浮点>}"
)


def _extract_content(resp: Any) -> str:
    """从 LangChain 响应中提取纯文本，兼容 content 为 list 的情况。"""
    content = getattr(resp, "content", "") or ""
    if isinstance(content, list):
        content = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in content
        )
    return str(content).strip()


def _parse_json(text: str) -> dict:
    """从可能含 markdown ```json``` 包裹的文本中解析 JSON 对象。"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
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
        if d.summary:
            parts.append(f"- 分块{idx}摘要: {d.summary}")
        for sig in d.signals:
            parts.append(f"- [{sig.category.value}] {sig.content}")
    return "\n".join(parts)


def _split_probe_qa(text: str) -> list[str]:
    """把 LLM 整体回答按 "Q:" 分段，拆成 probe 数量等长的字符串列表。

    拆不出 ≥2 段时回退为整段文本单元素列表。
    """
    if not text:
        return []
    upper = text.upper()
    indices: list[int] = []
    i = 0
    while True:
        idx = upper.find("Q:", i)
        if idx == -1:
            break
        indices.append(idx)
        i = idx + 2
    if len(indices) >= 2:
        segments: list[str] = []
        for j, start in enumerate(indices):
            end = indices[j + 1] if j + 1 < len(indices) else len(text)
            segments.append(text[start:end].strip())
        return segments
    # 回退：整段当作一条 probe 回答
    return [text.strip()]


def _fuzzy_match(guessed: str, card: PersonaCard) -> bool:
    """判断 guessed 是否匹配 card.display_name 或 card.persona_id。

    含即算对，大小写不敏感；任一方向子串命中都算匹配。
    """
    if not guessed:
        return False
    g = guessed.strip().lower()
    if not g:
        return False
    candidates = [
        c.lower() for c in (card.display_name, card.persona_id) if c
    ]
    for c in candidates:
        if g in c or c in g:
            return True
    return False


def evaluate(
    card: PersonaCard,
    distillates: list[Distillate],
    llm: Any,
    n_probes: int = 5,
) -> IdentifiabilityScore:
    """两步：probe 生成 + 盲猜。

    Args:
        card: 待评估的人格卡
        distillates: 蒸馏中间产物，盲猜阶段作为"原语料"参考
        llm: LangChain ``BaseChatModel``
        n_probes: 使用前几个标准化 probe（≤5）

    Returns:
        :class:`IdentifiabilityScore`——confidence ∈ [0,1]；
        probe 生成或盲猜失败时返回 ``confidence=-1.0``，不抛异常。
    """
    try:
        # Step 1: Probe 生成——一次性给所有 probe，让 LLM 扮演人物回答
        n = max(1, min(n_probes, len(_PROBES)))
        selected_probes = _PROBES[:n]
        probe_user = "请逐个回答以下问题：\n" + "\n".join(
            f"{i}. {q}" for i, q in enumerate(selected_probes, 1)
        )
        system_prompt = (card.system_prompt or "").rstrip() + _PROBE_SYSTEM_SUFFIX
        probe_resp = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=probe_user),
        ])
        probe_text = _extract_content(probe_resp)
        probes = _split_probe_qa(probe_text)

        # Step 2: 盲猜——把 probe Q&A 喂给独立 judge（不给 PersonaCard）
        corpus_digest = _build_corpus_digest(distillates)
        probe_qa_text = (
            "\n\n".join(probes) if probes else "(probe 生成失败，无内容)"
        )
        guess_user = (
            "## 原语料摘要（含多位人物）\n\n"
            f"{corpus_digest}\n\n"
            "## Probe Q&A（某匿名人物的回答）\n\n"
            f"{probe_qa_text}\n\n"
            "## 任务\n"
            "判断上述 Q&A 最像原语料中的哪位人物？给出名字与置信度（0~1）。\n"
            '严格输出 JSON：{"name": "...", "confidence": 0.x}'
        )
        guess_resp = llm.invoke([
            SystemMessage(content=_GUESS_SYSTEM),
            HumanMessage(content=guess_user),
        ])
        guess_text = _extract_content(guess_resp)
        data = _parse_json(guess_text)

        guessed_name = str(data.get("name", "")).strip()
        confidence = float(data.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))

        # Step 3: 模糊匹配
        correct = _fuzzy_match(guessed_name, card)

        return IdentifiabilityScore(
            probes=probes,
            guessed_name=guessed_name,
            confidence=confidence,
            correct=correct,
            error="",
        )
    except Exception as e:
        logger.warning("identifiability 评估失败: %s", e)
        return IdentifiabilityScore(
            correct=False,
            confidence=-1.0,
            guessed_name="",
            error=str(e),
        )
