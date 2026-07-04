"""DeepAgents 子智能体与主智能体工厂。

提供两种使用方式：

1. **自主编排模式** —— :func:`build_orchestrator` 返回一个带四个子智能体
   （extractor / synthesizer / skill_designer / dialogue_writer）与"蒸馏 skill"
   的主 deepagent，由其自行用 ``task`` 工具调度。适合交互式/探索性蒸馏。

2. **确定性流水线模式** —— :func:`build_extractor_agent` 等返回各阶段独立
   deepagent，配合 :class:`~persona_distillation.pipeline.PersonaDistiller`
   按序调用、强制 ``response_format`` 结构化产出。适合可复现的批量蒸馏。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from deepagents import SubAgent, create_deep_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel

from persona_distillation.config import DistillationConfig
from persona_distillation.prompts import (
    DISTILLATION_SKILL_MD,
    DIALOGUE_WRITER_SYSTEM,
    EXTRACTOR_SYSTEM,
    ORCHESTRATOR_SYSTEM,
    dialogue_writer_system,
    skill_designer_system,
    synthesizer_system,
)
from pydantic import BaseModel

from persona_distillation.schemas import (
    Distillate,
    PersonaCard,
    PersonaSkill,
    PresetDialogue,
)


# ---------------------------------------------------------------------------
# 模型构造：把 ``minimax:<model>`` 解析成 ChatOpenAI + minimax base_url；
# 其余 ``provider:model`` 字符串原样交给 deepagents 的 init_chat_model。
# ---------------------------------------------------------------------------
def build_model(cfg: DistillationConfig) -> str | BaseChatModel:
    """根据 ``cfg.model`` 返回 deepagents 可用的 model 参数。

    - ``minimax:<model_name>`` → 用 ``ChatOpenAI`` 指向 minimax OpenAI 兼容端点，
      需环境变量 ``MINIMAX_API_KEY``（或 ``cfg.minimax_api_key_env`` 指定的变量）。
    - 其它（如 ``openai:gpt-4o-mini`` / ``anthropic:claude-sonnet-4-5``）→ 原样返回
      字符串，由 deepagents 走 ``init_chat_model`` 处理。
    """
    model = cfg.model
    if not model.startswith("minimax:"):
        return model

    import os

    from langchain_openai import ChatOpenAI

    model_name = model.split(":", 1)[1]
    api_key = os.environ.get(cfg.minimax_api_key_env, "")
    if not api_key:
        raise RuntimeError(
            f"未设置环境变量 {cfg.minimax_api_key_env}；请在 "
            f"https://platform.minimax.io 获取后 export {cfg.minimax_api_key_env}=..."
        )
    return ChatOpenAI(
        model=model_name,
        base_url=cfg.minimax_base_url,
        api_key=api_key,
        temperature=0.6,
    )


# ---------------------------------------------------------------------------
# response_format 需要单个对象，故为列表型产出包一层包装
# ---------------------------------------------------------------------------
class PersonaSkillList(BaseModel):
    """skill_designer 的结构化产出包装。"""

    skills: list[PersonaSkill] = []


class PresetDialogueList(BaseModel):
    """dialogue_writer 的结构化产出包装。"""

    dialogues: list[PresetDialogue] = []


# ---------------------------------------------------------------------------
# 蒸馏 skill 落盘：让主智能体通过 SkillsMiddleware 加载框架自身的"蒸馏skills"
# ---------------------------------------------------------------------------
def write_distillation_skill(skills_dir: str | Path) -> str:
    """把蒸馏方法论写成一个 SKILL.md，返回其所在目录（供 ``skills=`` 参数使用）。"""
    root = Path(skills_dir) / "persona-distillation"
    root.mkdir(parents=True, exist_ok=True)
    (root / "SKILL.md").write_text(DISTILLATION_SKILL_MD, encoding="utf-8")
    return str(root.parent) + "/"


# ---------------------------------------------------------------------------
# 子智能体声明（供 orchestrator 的 subagents= 使用）
# ---------------------------------------------------------------------------
def build_subagents(cfg: DistillationConfig) -> list[SubAgent]:
    """返回四个子智能体的声明。"""
    persona_id_hint = (
        f"用户已指定 persona_id=`{cfg.persona_id}`，必须原样使用。"
        if cfg.persona_id
        else "用户未指定，由你根据角色推断。"
    )
    return [
        SubAgent(
            name="extractor",
            description=(
                "对一个文本分块进行人格分馏：按 SignalCategory 塔板分离信号，"
                "每条附 evidence 与 salience，产出 Distillate。"
            ),
            system_prompt=EXTRACTOR_SYSTEM,
            response_format=Distillate,  # type: ignore[call-arg]
        ),
        SubAgent(
            name="synthesizer",
            description=(
                "读取全部 Distillate，执行冷凝+提纯，产出最终 PersonaCard"
                "（persona_id / system_prompt / error_reply）。"
            ),
            system_prompt=synthesizer_system(cfg.salience_threshold, persona_id_hint),
            response_format=PersonaCard,  # type: ignore[call-arg]
        ),
        SubAgent(
            name="skill_designer",
            description=(
                f"基于 PersonaCard 设计至多 {cfg.max_skills} 个 PersonaSkill，"
                "覆盖问候/拒绝/安抚/深聊/回忆/告别等能力面。"
            ),
            system_prompt=skill_designer_system(cfg.max_skills),
            response_format=PersonaSkillList,  # type: ignore[call-arg]
        ),
        SubAgent(
            name="dialogue_writer",
            description=(
                f"基于 PersonaCard 撰写 {cfg.max_preset_dialogues} 组 PresetDialogue，"
                "覆盖寒暄/探问/踩雷/求助/倾诉/告别等意图。"
            ),
            system_prompt=dialogue_writer_system(cfg.max_preset_dialogues),
            response_format=PresetDialogueList,  # type: ignore[call-arg]
        ),
    ]


# ---------------------------------------------------------------------------
# 主智能体（自主编排模式）
# ---------------------------------------------------------------------------
def build_orchestrator(
    cfg: DistillationConfig,
    *,
    middleware: list[AgentMiddleware] | None = None,
) -> Any:
    """构建带子智能体的主 deepagent（自主编排模式）。

    ``create_deep_agent`` 默认已注入 ``FilesystemMiddleware``（StateBackend），
    提供默认的 ``ls/read_file/write_file/edit_file/glob/grep`` 工具，故此处不再
    重复添加；中间产物由 Python 端的流水线落盘到 ``cfg.workdir``。
    """
    workdir = cfg.resolve_workdir()
    skills_dir = write_distillation_skill(workdir / "skills")
    extra = list(cfg.extra_skills_dirs or [])
    skills = [skills_dir, *extra]

    return create_deep_agent(
        model=build_model(cfg),
        system_prompt=ORCHESTRATOR_SYSTEM,
        subagents=build_subagents(cfg),
        skills=skills,
        middleware=middleware or (),
        checkpointer=False,
        debug=cfg.debug,
        name="persona-distillation-orchestrator",
    )


# ---------------------------------------------------------------------------
# 各阶段独立 agent（确定性流水线模式）
# ---------------------------------------------------------------------------
def build_extractor_agent(cfg: DistillationConfig) -> Any:
    return create_deep_agent(
        model=build_model(cfg),
        system_prompt=EXTRACTOR_SYSTEM,
        response_format=Distillate,
        checkpointer=False,
        debug=cfg.debug,
        name="persona-extractor",
    )


def build_synthesizer_agent(cfg: DistillationConfig) -> Any:
    persona_id_hint = (
        f"用户已指定 persona_id=`{cfg.persona_id}`，必须原样使用。"
        if cfg.persona_id
        else "用户未指定，由你根据角色推断。"
    )
    return create_deep_agent(
        model=build_model(cfg),
        system_prompt=synthesizer_system(cfg.salience_threshold, persona_id_hint),
        response_format=PersonaCard,
        checkpointer=False,
        debug=cfg.debug,
        name="persona-synthesizer",
    )


def build_skill_designer_agent(cfg: DistillationConfig) -> Any:
    return create_deep_agent(
        model=build_model(cfg),
        system_prompt=skill_designer_system(cfg.max_skills),
        response_format=PersonaSkillList,
        checkpointer=False,
        debug=cfg.debug,
        name="persona-skill-designer",
    )


def build_dialogue_writer_agent(cfg: DistillationConfig) -> Any:
    return create_deep_agent(
        model=build_model(cfg),
        system_prompt=dialogue_writer_system(cfg.max_preset_dialogues),
        response_format=PresetDialogueList,
        checkpointer=False,
        debug=cfg.debug,
        name="persona-dialogue-writer",
    )


# ---------------------------------------------------------------------------
# 调用 helper：兼容 structured_response 与消息内容两种返回
# ---------------------------------------------------------------------------
def invoke_structured(agent: Any, user_prompt: str, expected_type: type) -> Any:
    """调用 deepagent 并取回结构化结果。

    优先读 ``structured_response``；若缺失则尝试把最后一条 AIMessage 内容解析为 JSON。
    """
    result = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
    sr = result.get("structured_response")
    if sr is not None:
        if isinstance(sr, expected_type):
            return sr
        if isinstance(sr, dict):
            return expected_type.model_validate(sr)
    # 退化：从最后一条消息内容里抢救 JSON
    import json

    messages = result.get("messages") or []
    for msg in reversed(messages):
        content = getattr(msg, "content", None) or ""
        if isinstance(content, list):
            content = "".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        if not isinstance(content, str) or "{" not in content:
            continue
        start = content.find("{")
        end = content.rfind("}")
        if 0 <= start < end:
            try:
                return expected_type.model_validate(json.loads(content[start : end + 1]))
            except Exception:
                continue
    raise ValueError(
        f"未能从 agent 返回中解析出 {expected_type.__name__}；原始 keys={list(result.keys())}"
    )
