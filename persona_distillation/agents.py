"""DeepAgents 工厂：把 LLM 模型转成可调用的 agent 图。

对外接口：
- :func:`build_orchestrator`         —— 顶层 deepagent（含 4 个子 agent + skill）
- :func:`build_extractor_agent`      —— Stage 1 蒸馏
- :func:`build_synthesizer_agent`    —— Stage 2-3 冷凝 + 提纯
- :func:`build_skill_designer_agent` —— Stage 4a 技能设计
- :func:`build_dialogue_writer_agent`—— Stage 4b 对话撰写
- :func:`build_intake_orchestrator`  —— 主理人 Agent（intake 子包入口）
- :func:`invoke_structured`          —— 通用 LLM 调用 + 结构化输出包装
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from langchain_core.language_models import BaseChatModel
from deepagents import create_deep_agent
from deepagents.middleware import SubAgent
from langchain.agents.middleware import AgentMiddleware

from persona_distillation.config import DistillationConfig
from persona_distillation.prompts import (
    BRIDGER_SYSTEM,
    DISTILLATION_SKILL_MD,
    DIALOGUE_WRITER_SYSTEM,
    EXTRACTOR_SYSTEM,
    INTAKE_NER_SYSTEM,
    INTAKE_ORCHESTRATOR_SYSTEM,
    ORCHESTRATOR_SYSTEM,
    PROFILE_BUILDER_SYSTEM,
    dialogue_writer_system,
    skill_designer_system,
    synthesizer_system,
)
from pydantic import BaseModel

from persona_distillation.intake.schemas import (
    CharacterProfile,
    NameExtractionResult,
)
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
        # P0-2: 清晰的错误信息，引导用户快速修复
        logger.error("未设置环境变量 %s", cfg.minimax_api_key_env)
        raise RuntimeError(
            f"未设置环境变量 {cfg.minimax_api_key_env}；请在 "
            f"https://api.minimax.io 获取后 export {cfg.minimax_api_key_env}=... "
            f"（或设置 cfg.dry_run=True 跳过校验用于测试）"
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
# intake 子包：主理人 Agent + 3 个子智能体
# ---------------------------------------------------------------------------
def build_intake_subagents(cfg: DistillationConfig) -> list[SubAgent]:
    """返回 3 个 intake 子智能体。"""
    return [
        SubAgent(
            name="intake_ner",
            description=(
                "对单分块做人物识别与分类：识别所有人物 + 规范化名字 + 标注 "
                "speech/appearance/event 类别，产出 NameExtractionResult。"
            ),
            system_prompt=INTAKE_NER_SYSTEM,
            response_format=NameExtractionResult,  # type: ignore[call-arg]
        ),
        SubAgent(
            name="profile_builder",
            description=(
                "从 IndexStore 拉取单人物索引条目，跨类别重排序，撰写 ≤200 字人物档案。"
            ),
            system_prompt=PROFILE_BUILDER_SYSTEM,
            response_format=CharacterProfile,  # type: ignore[call-arg]
        ),
        SubAgent(
            name="bridger",
            description=(
                "接收 CharacterProfile 调工具完成蒸馏桥接：重建临时语料目录 + 启动 PersonaDistiller。"
            ),
            system_prompt=BRIDGER_SYSTEM,
            # 不强制 response_format：bridger 的产出是落盘的文件路径
        ),
    ]


def build_intake_orchestrator(
    cfg: DistillationConfig,
    *,
    middleware: list[AgentMiddleware] | None = None,
) -> Any:
    """主理人 Agent：3 个 intake 子 agent + 框架 skill + 工具。

    入口是交互式 REPL，由用户自然语言驱动。
    """
    workdir = cfg.resolve_workdir()
    skills_dir = write_distillation_skill(workdir / "skills")
    extra = list(cfg.extra_skills_dirs or [])
    skills = [skills_dir, *extra]

    return create_deep_agent(
        model=build_model(cfg),
        system_prompt=INTAKE_ORCHESTRATOR_SYSTEM,
        subagents=build_intake_subagents(cfg),
        skills=skills,
        middleware=middleware or (),
        checkpointer=False,
        debug=cfg.debug,
        name="persona-intake-conductor",
    )


# ---------------------------------------------------------------------------
# 调用 helper：兼容 structured_response 与消息内容两种返回
# ---------------------------------------------------------------------------
# P1-1: 重试装饰器（带指数回退 + 抖动）—— 仅依赖标准库，避免引入 tenacity
import random
import time as _time
from functools import wraps
from typing import Callable


_RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)


def _retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 8.0,
    enabled: bool = True,
):
    """指数回退 + 抖动重试装饰器。

    适用于：瞬时网络错误、超时、限流。仅依赖标准库。
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled:
                return func(*args, **kwargs)
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except _RETRYABLE_EXCEPTIONS as e:
                    last_exc = e
                    if attempt >= max_attempts:
                        break
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    delay += random.uniform(0, 0.3)  # 抖动
                    logger.warning(
                        "LLM call %s 第 %d/%d 次失败 (%s: %s)，%.2fs 后重试",
                        getattr(func, "__name__", "<func>"),
                        attempt,
                        max_attempts,
                        type(e).__name__,
                        e,
                        delay,
                    )
                    _time.sleep(delay)
            # 全部失败
            logger.error("LLM call 重试 %d 次后仍失败：%s", max_attempts, last_exc)
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator


def invoke_structured(agent: Any, user_prompt: str, expected_type: type) -> Any:
    """调用 deepagent 并取回结构化结果。

    优先读 ``structured_response``；若缺失则尝试把最后一条 AIMessage 内容解析为 JSON。
    内部使用 :func:`_retry_with_backoff` 处理瞬时 LLM 失败。
    """
    # P1-1: 重试
    result = _retry_with_backoff(max_attempts=3, base_delay=1.0)(
        lambda: agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
    )()
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
            except json.JSONDecodeError as e:
                logger.warning("JSON 解析失败 (尝试最后一条消息): %s", e)
                continue
            except Exception as e:  # noqa: BLE001
                # 校验失败等也记录而非静默
                logger.warning("Pydantic 校验失败 (尝试最后一条消息): %s", e)
                continue
    raise ValueError(
        f"未能从 agent 返回中解析出 {expected_type.__name__}；原始 keys={list(result.keys())}"
    )
