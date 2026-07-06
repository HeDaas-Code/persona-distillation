"""OC 共创 Phase 1：骨架文本生成。

4 个 sub-agent 各写一类文本（独白/对话/事件/回忆），落盘到
``<workdir>/<persona_id>/oc_corpus/``，供 Phase 2 访谈的 character_player 读取作为人设基础。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from persona_distillation.prompts import (
    dialogue_writer_prompt,
    event_writer_prompt,
    memory_writer_prompt,
    monologue_writer_prompt,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OC 设定：用户与主理人共同捏造的角色基础
# ---------------------------------------------------------------------------
class OCSetting(BaseModel):
    """OC（原创角色）设定。

    age 用 str 而非 int，兼容"未知" / "三十出头" / "永远 17 岁"等非数值表达。
    """

    name: str
    age: str
    background: str
    traits: str
    worldview: str
    catchphrase: str

    def to_prompt_text(self) -> str:
        """把设定拼成人类可读文本，供注入 system prompt。"""
        return (
            f"姓名：{self.name}\n"
            f"年龄：{self.age}\n"
            f"背景：{self.background}\n"
            f"性格核心：{self.traits}\n"
            f"世界观：{self.worldview}\n"
            f"口头禅：{self.catchphrase}"
        )


# ---------------------------------------------------------------------------
# LLM 调用 helper：兼容 content 为 list 的 langchain 新版返回
# ---------------------------------------------------------------------------
def _invoke_llm(llm: Any, system_prompt: str, user_prompt: str) -> str:
    """调 LLM 返回纯文本，自动处理 content 为 list 的情况。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    resp = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    content = getattr(resp, "content", "") or ""
    if isinstance(content, list):
        # langchain 新版 content 可能是 [{"type": "text", "text": "..."}, ...]
        content = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in content
        )
    return str(content).strip()


# ---------------------------------------------------------------------------
# 4 类文本生成配置：(key, 文件名, 中文标题, prompt 构造器, 用户指令)
# ---------------------------------------------------------------------------
_WRITERS = [
    (
        "monologue",
        "monologue.md",
        "独白",
        monologue_writer_prompt,
        "请按 system prompt 要求撰写这段第一人称内心独白。",
    ),
    (
        "dialogue",
        "dialogue.md",
        "对话",
        dialogue_writer_prompt,
        "请按 system prompt 要求撰写 ≥3 段不同关系的对话片段。",
    ),
    (
        "event",
        "event.md",
        "事件",
        event_writer_prompt,
        "请按 system prompt 要求撰写 ≥2 个标志性事件叙述。",
    ),
    (
        "memory",
        "memory.md",
        "回忆",
        memory_writer_prompt,
        "请按 system prompt 要求撰写 ≥2 段过往回忆。",
    ),
]


def generate_oc_corpus(
    setting: OCSetting,
    workdir: str | Path,
    persona_id: str,
    llm: Any,
) -> dict:
    """Phase 1：调 4 个 sub-agent 生成骨架文本并落盘。

    Args:
        setting: OC 设定
        workdir: 工作目录（产物落在 ``<workdir>/<persona_id>/oc_corpus/``）
        persona_id: 人格 ID（决定子目录名）
        llm: 已构造好的 LangChain ``BaseChatModel``

    Returns:
        ``{"paths": {...}, "word_counts": {...}, "corpus_dir": str}``

    Raises:
        任意 LLM 调用或落盘异常向上抛（让上层工具捕获）。
    """
    corpus_dir = Path(workdir) / persona_id / "oc_corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    setting_text = setting.to_prompt_text()
    paths: dict[str, str] = {}
    word_counts: dict[str, int] = {}

    for key, filename, label, prompt_fn, user_instr in _WRITERS:
        system_prompt = prompt_fn(setting_text)
        logger.info("[Phase 1] 生成 %s 骨架 (persona=%s)", key, persona_id)
        body = _invoke_llm(llm, system_prompt, user_instr)
        # 文件开头加一行标题，然后是正文
        content = f"# {label} · {setting.name}\n\n{body}"
        out_path = corpus_dir / filename
        out_path.write_text(content, encoding="utf-8")
        paths[key] = str(out_path)
        word_counts[key] = len(body)
        logger.info(
            "[Phase 1] %s 完成: %d 字 → %s", key, word_counts[key], out_path
        )

    return {
        "paths": paths,
        "word_counts": word_counts,
        "corpus_dir": str(corpus_dir),
    }
