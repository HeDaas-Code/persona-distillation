"""OC 共创 Phase 2：血肉访谈。

character_player sub-agent 读取 Phase 1 骨架作为人设基础，
主理人（interviewer）对 character_player 做 N 轮访谈，
访谈记录落盘到 ``<workdir>/<persona_id>/interview.md``。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from persona_distillation.intake.oc_writer import OCSetting
from persona_distillation.prompts import (
    character_player_system,
    interviewer_system,
)

logger = logging.getLogger(__name__)


# 4 类骨架文件名（与 oc_writer 保持一致）
_SKELETON_FILES = ["monologue.md", "dialogue.md", "event.md", "memory.md"]


def _invoke_llm(llm: Any, messages: list) -> str:
    """调 LLM 返回纯文本，自动处理 content 为 list 的情况。"""
    resp = llm.invoke(messages)
    content = getattr(resp, "content", "") or ""
    if isinstance(content, list):
        content = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in content
        )
    return str(content).strip()


def _load_skeleton(corpus_dir: Path) -> str:
    """读取 oc_corpus 下 4 个骨架文件，拼成一段人设基础文本。"""
    parts: list[str] = []
    for fname in _SKELETON_FILES:
        p = corpus_dir / fname
        if p.exists():
            parts.append(p.read_text(encoding="utf-8"))
    if not parts:
        raise FileNotFoundError(
            f"骨架目录 {corpus_dir} 下未找到任何骨架文件 "
            f"(期望: {_SKELETON_FILES})"
        )
    return "\n\n---\n\n".join(parts)


def run_interview(
    setting: OCSetting,
    n_rounds: int,
    workdir: str | Path,
    persona_id: str,
    llm: Any,
) -> dict:
    """Phase 2：主理人对 character_player 做 N 轮访谈并落盘。

    Args:
        setting: OC 设定（与 Phase 1 一致）
        n_rounds: 访谈轮数
        workdir: 工作目录
        persona_id: 人格 ID（决定骨架与访谈文件位置）
        llm: 已构造好的 LangChain ``BaseChatModel``

    Returns:
        ``{"path": str, "rounds": N}``

    Raises:
        FileNotFoundError: 骨架目录不存在时抛出（提示先调 generate_oc_corpus）。
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    corpus_dir = Path(workdir) / persona_id / "oc_corpus"
    if not corpus_dir.exists():
        raise FileNotFoundError(
            "骨架目录不存在，请先调 generate_oc_corpus 生成骨架"
        )

    # 读骨架 → 构造 character_player / interviewer system prompt
    skeleton_text = _load_skeleton(corpus_dir)
    character_player_sys = character_player_system(
        setting.to_prompt_text(), skeleton_text
    )
    interviewer_sys = interviewer_system()

    # 累积的 Q&A 历史：HumanMessage(问题) + AIMessage(回答)
    # 双方共享这份 transcript——character_player 视角下"人类发问、OC 作答"语义自洽；
    # interviewer 也能从内容读出 Q&A 流向，进而追问。
    history: list = []
    qa_blocks: list[str] = [f"# 角色访谈记录 · {setting.name}", ""]

    for i in range(1, n_rounds + 1):
        logger.info(
            "[Phase 2] 第 %d/%d 轮访谈 (persona=%s)", i, n_rounds, persona_id
        )

        # 1. interviewer 基于已有上下文出题
        question = _invoke_llm(llm, [
            SystemMessage(content=interviewer_sys),
            *history,
            HumanMessage(
                content="请基于已有上下文提出下一个访谈问题，只输出问题本身。"
            ),
        ])

        # 2. character_player 以 OC 身份回答
        answer = _invoke_llm(llm, [
            SystemMessage(content=character_player_sys),
            *history,
            HumanMessage(content=question),
        ])

        # 3. 累积到 history（供下一轮双方参考）
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))

        # 4. 拼访谈记录 markdown
        qa_blocks.append(f"## 第 {i} 轮")
        qa_blocks.append("")
        qa_blocks.append(f"**主理人**：{question}")
        qa_blocks.append("")
        qa_blocks.append(f"**{setting.name}**：{answer}")
        qa_blocks.append("")

    out_path = Path(workdir) / persona_id / "interview.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(qa_blocks), encoding="utf-8")
    logger.info("[Phase 2] 访谈记录落盘: %s", out_path)

    return {"path": str(out_path), "rounds": n_rounds}
