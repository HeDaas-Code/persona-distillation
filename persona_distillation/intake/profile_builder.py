"""人物档案构建器：从索引聚合 + 选 top-k + LLM 生成 summary。

步骤：
1. 从 :class:`IndexStore` 拉取该人物的全部索引条目
2. 按 category 分组
3. 各类内用 reranker（或简单 salience 截断）取 top-n
4. 调 LLM 生成 ≤200 字 summary（可选；LLM 不可用时用拼装版）
"""
from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel

from persona_distillation.intake.index_store import IndexStore
from persona_distillation.intake.schemas import (
    CharacterProfile,
    IndexCategory,
    NameIndexEntry,
)


def _topk_by_text(
    entries: list[NameIndexEntry],
    query: str,
    top_n: int,
    reranker: Any = None,
) -> list[NameIndexEntry]:
    """对一组条目做 top-n 选择。

    - 若有 reranker：先取 top-k * 2，再用 cross-encoder 重排取 top_n
    - 否则按 entry 长度 + chunk 顺序截断
    """
    if not entries:
        return []

    if reranker is not None:
        try:
            from langchain_core.documents import Document

            docs = [
                Document(page_content=e.text, metadata={"uuid": e.uuid, "_idx": i})
                for i, e in enumerate(entries)
            ]
            compressed = reranker.compress_documents(docs, query)
            # 按压缩后顺序取 top_n
            ids: list[str] = [d.metadata.get("uuid", "") for d in compressed[:top_n]]
            by_uuid = {e.uuid: e for e in entries}
            return [by_uuid[u] for u in ids if u in by_uuid]
        except Exception:
            pass

    # 退化：按文本长度 + 顺序
    return entries[:top_n]


def _take(
    entries: list[NameIndexEntry],
    max_entries: int,
    reranker: Any = None,
    query: str = "",
) -> list[NameIndexEntry]:
    """为蒸馏语料保留条目（与 _topk_by_text 的搜索 top-k 语义不同）。

    - ``max_entries=0``：返回全部条目（reranker 可用时按 salience 排序但不截断，
      让蒸馏器看到全量语料）
    - ``max_entries>0``：截断到 max_entries 条（极端长文本时控制内存）

    与 ``_topk_by_text`` 的区别：后者是为搜索结果展示设计的 top-k 截断；
    本函数是为蒸馏语料重建设计的全量保留，仅在用户显式配置上限时截断。
    """
    if not entries:
        return []
    if max_entries > 0 and len(entries) > max_entries:
        # 有 reranker 时先排序再截断，保留最 salient 的 max_entries 条
        if reranker is not None:
            sorted_entries = _rerank_sort(entries, reranker, query)
            return sorted_entries[:max_entries]
        return entries[:max_entries]
    # max_entries=0 或条目数未超上限：全部保留（reranker 可用时排序但不截断）
    if reranker is not None and len(entries) > 1:
        return _rerank_sort(entries, reranker, query)
    return entries


def _rerank_sort(
    entries: list[NameIndexEntry],
    reranker: Any,
    query: str,
) -> list[NameIndexEntry]:
    """用 reranker 对 entries 按 salience 排序（不截断）。失败时返回原顺序。"""
    try:
        from langchain_core.documents import Document

        docs = [
            Document(page_content=e.text, metadata={"uuid": e.uuid, "_idx": i})
            for i, e in enumerate(entries)
        ]
        compressed = reranker.compress_documents(docs, query)
        ids: list[str] = [d.metadata.get("uuid", "") for d in compressed]
        by_uuid = {e.uuid: e for e in entries}
        return [by_uuid[u] for u in ids if u in by_uuid]
    except Exception:
        return entries


def _fallback_summary(profile: CharacterProfile) -> str:
    """无 LLM 时的兜底 summary。"""
    parts = [
        f"{profile.character_name}：在 {profile.mention_count} 处被提及，",
        f"对话 {profile.speech_count} 条、外貌 {profile.appearance_count} 条、事件 {profile.event_count} 条。",
    ]
    return "".join(parts)


SUMMARY_PROMPT = """你是人物档案撰写者。

【人物名】{name}（别名：{aliases}）
【统计】{stats}

【对话摘录】（top {n_speech}）
{speech_block}

【外貌摘录】（top {n_app}）
{appearance_block}

【事件摘录】（top {n_ev}）
{event_block}

请撰写一段 ≤200 字的人物档案摘要，包含：
1) 身份定位（一句话）
2) 行为特征（3~5 条要点）
3) 原文引用（保留 1~2 条最具代表性的原话或描述）

仅输出纯文本，不要分点列表。
"""


def build_profile(
    name: str,
    store: IndexStore,
    *,
    reranker: Any = None,
    llm: BaseChatModel | None = None,
    top_n: int = 6,           # 仅用于 summary LLM block，控制上下文长度
    max_entries: int = 0,     # 0=不限，>0=每类 excerpts 上限（蒸馏语料全量保留）
) -> CharacterProfile:
    """从索引里聚合一个人物档案。"""
    all_entries = store.get_character_entries(name)

    speech = [e for e in all_entries if e.category == IndexCategory.SPEECH]
    appearance = [e for e in all_entries if e.category == IndexCategory.APPEARANCE]
    event = [e for e in all_entries if e.category == IndexCategory.EVENT]

    # 收集 aliases
    aliases: list[str] = []
    seen: set[str] = set()
    for e in all_entries:
        for a in e.aliases:
            if a and a not in seen:
                seen.add(a)
                aliases.append(a)

    profile = CharacterProfile(
        character_name=name,
        aliases=aliases,
        mention_count=len(all_entries),
        speech_count=len(speech),
        appearance_count=len(appearance),
        event_count=len(event),
        speech_excerpts=_take(speech, max_entries, reranker, name),
        appearance_excerpts=_take(appearance, max_entries, reranker, name),
        event_excerpts=_take(event, max_entries, reranker, name),
    )

    # ---- 调 LLM 生成 summary ----
    if llm is not None:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            def _block(entries: list[NameIndexEntry]) -> str:
                if not entries:
                    return "（无）"
                return "\n".join(f"- {e.text}" for e in entries)

            # summary LLM block 用 top-6（控制 LLM 上下文长度），与 excerpts 的全量保留解耦
            speech_top = _topk_by_text(speech, name, top_n, reranker)
            appearance_top = _topk_by_text(appearance, name, top_n, reranker)
            event_top = _topk_by_text(event, name, top_n, reranker)
            prompt = SUMMARY_PROMPT.format(
                name=name,
                aliases=", ".join(aliases) or "（无）",
                stats=f"提及 {profile.mention_count} 次 / 对话 {profile.speech_count} / 外貌 {profile.appearance_count} / 事件 {profile.event_count}",
                n_speech=len(speech_top),
                n_app=len(appearance_top),
                n_ev=len(event_top),
                speech_block=_block(speech_top),
                appearance_block=_block(appearance_top),
                event_block=_block(event_top),
            )
            resp = llm.invoke(
                [
                    SystemMessage(content="你是人物档案撰写者，输出纯文本。"),
                    HumanMessage(content=prompt),
                ]
            )
            text = getattr(resp, "content", "") or ""
            if isinstance(text, list):
                text = "".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in text
                )
            text = str(text).strip()
            if text:
                profile = profile.model_copy(update={"summary": text[:400]})
        except Exception:
            pass

    if not profile.summary:
        profile = profile.model_copy(update={"summary": _fallback_summary(profile)})

    return profile
