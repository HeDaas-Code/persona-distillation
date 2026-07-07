"""DeepAgents 工厂：把 LLM 模型转成可调用的 agent 图。

对外接口：
- :func:`build_extractor_agent`      —— Stage 1 蒸馏（pipeline.py 确定性流水线逐块调用）
- :func:`build_synthesizer_agent`    —— Stage 2-3 冷凝 + 提纯
- :func:`build_skill_designer_agent` —— Stage 4a 技能设计
- :func:`build_dialogue_writer_agent`—— Stage 4b 对话撰写
- :func:`build_subagents`            —— 4 个蒸馏 SubAgent 声明（供 build_intake_orchestrator 复用）
- :func:`build_intake_orchestrator`  —— 主理人 Agent（intake 子包入口；接线 7 个 SubAgent + 纯 IO 工具）
- :func:`invoke_structured`          —— 通用 LLM 调用 + 结构化输出包装

Phase 1 重构（#15）：旧版 ``build_orchestrator`` / ``build_intake_subagents`` 已删除——
前者是与主理人并行的"自主编排模式"死代码（从未被 CLI / WebUI 调用），
后者的 SubAgent 定义已合并到 ``build_intake_orchestrator``。
"""
from __future__ import annotations

import logging
import os
import random
import time as _time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

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
    EXTRACTOR_BATCH_SYSTEM,
    EXTRACTOR_SYSTEM,
    INTAKE_NER_SYSTEM,
    INTAKE_ORCHESTRATOR_SYSTEM,
    PROFILE_BUILDER_SYSTEM,
    dialogue_writer_system,
    intake_orchestrator_system,
    skill_designer_system,
    synthesizer_system,
)
from pydantic import BaseModel

from persona_distillation.intake.schemas import (
    CharacterProfile,
    NerBatchResult,
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

    from langchain_openai import ChatOpenAI

    # 归一化代理 scheme：httpx 只认 socks5:// / socks4://，不认 socks://。
    # Clash 等 GUI 代理工具常把 ALL_PROXY 设成 socks://（缺版本号），导致
    # ChatOpenAI 构造时 httpx 抛 "Unknown scheme for proxy URL"。
    # 这里在本地构造归一化后的代理 dict 传给 http_client，不改 os.environ（避免全局副作用）。
    proxy_map: dict[str, str] = {}
    for _var, _scheme in (
        ("ALL_PROXY", "all://"),
        ("HTTP_PROXY", "http://"),
        ("HTTPS_PROXY", "https://"),
    ):
        _val = os.environ.get(_var, "") or os.environ.get(_var.lower(), "")
        if not _val:
            continue
        if _val.startswith("socks://"):
            _val = "socks5://" + _val[len("socks://"):]
            logger.debug("代理 scheme 归一化: %s socks:// → socks5://", _var)
        proxy_map[_scheme] = _val

    # 仅在有代理时构造 httpx.Client；失败则回退到默认 http_client
    http_client = None
    if proxy_map:
        try:
            import httpx
            http_client = httpx.Client(proxies=proxy_map)
        except Exception as e:  # noqa: BLE001
            logger.warning("构造 httpx Client 失败，回退到默认 http_client: %s", e)
            http_client = None

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
        http_client=http_client,
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


class DistillateList(BaseModel):
    """extractor SubAgent（批量模式）的结构化产出包装。

    pipeline.py 的确定性流水线仍用单条 ``Distillate``（逐块调用），不变；
    主理人 Agent 一次性分派全部 chunk 时用本包装接收 Distillate 列表。
    """

    distillates: list[Distillate] = []


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
# intake 子包：主理人 Agent（接线 7 个 SubAgent + 纯 IO/查询工具）
# ---------------------------------------------------------------------------
def build_intake_orchestrator(
    cfg: DistillationConfig,
    *,
    middleware: list[AgentMiddleware] | None = None,
) -> Any:
    """主理人 Agent（intake 子包入口）：接线 7 个 SubAgent + 纯 IO/查询工具。

    入口是交互式 REPL，由用户自然语言驱动；主理人通过 ``task`` 工具分派
    SubAgent，而非用 Python 黑箱工具包办（Phase 1 重构 #15）。

    7 个 SubAgent：
    - ``intake_ner``      —— 批量 NER（接收全部 chunk，产出 NerBatchResult）
    - ``profile_builder`` —— 拉取人物索引条目，撰写 CharacterProfile
    - ``extractor``       —— 批量分馏（EXTRACTOR_BATCH_SYSTEM + DistillateList，
                              与 pipeline.py 逐块模式区分）
    - ``synthesizer``     —— 冷凝 + 提纯，产出 PersonaCard
    - ``skill_designer``  —— 基于 PersonaCard 设计 PersonaSkill 列表
    - ``dialogue_writer`` —— 基于 PersonaCard 撰写 PresetDialogue 列表
    - ``bridger``         —— 收尾汇报（最终产物落盘后给主理人最终回复素材）

    Python 工具（见 :mod:`persona_distillation.intake.tools`）只保留纯 IO/查询
    与 SubAgent 间文件交接：``load_text`` / ``load_and_chunk`` / ``index_characters``
    / ``list_characters`` / ``search_index`` / ``get_character_entries`` /
    ``save_distillates`` / ``load_distillates`` + OC 共创的 ``generate_oc_corpus``
    / ``run_character_interview``。

    设计说明：
    - ``FilesystemMiddleware`` 默认的 StateBackend 是内存虚拟 FS，无法读真实磁盘；
      摄入语料必须走 ``load_text`` / ``load_and_chunk`` 工具，不能依赖 ``ls``/``read_file``。
      这一点已写入 system prompt。
    - 4 个蒸馏 SubAgent 复用 ``build_subagents()`` 的声明，但 extractor 在主理人
      模式下改用 ``EXTRACTOR_BATCH_SYSTEM`` + ``DistillateList``（一次性处理全部 chunk）；
      ``build_subagents()`` 自身保留原 ``EXTRACTOR_SYSTEM`` + 单条 ``Distillate``，
      不影响 pipeline.py 的逐块确定性流水线。
    - 3 个 intake SubAgent（intake_ner / profile_builder / bridger）在此内联声明
      （旧 ``build_intake_subagents`` 已删除）。
    - 工具闭包共享同一个 :class:`IntakeContext`（IndexStore / llm / workdir），
      跨工具调用状态一致。
    """
    from persona_distillation.intake.tools import build_intake_context, build_intake_tools

    ctx = build_intake_context(cfg)
    tools = build_intake_tools(ctx)
    skills_dir = write_distillation_skill(ctx.workdir / "skills")
    extra = list(cfg.extra_skills_dirs or [])
    skills = [skills_dir, *extra]

    # 4 个蒸馏 SubAgent：复用 build_subagents()，但把 extractor 切到批量模式
    # （主理人一次性分派全部 chunk；pipeline.py 仍走 EXTRACTOR_SYSTEM 逐块模式，互不影响）
    # 注：``SubAgent`` 是 TypedDict，实例即 dict，按 key 访问而非属性。
    subagents: list[SubAgent] = []
    for sa in build_subagents(cfg):
        if sa["name"] == "extractor":
            subagents.append(
                SubAgent(
                    name="extractor",
                    description=(
                        "一次性接收全部文本分块（JSON 数组），批量按 SignalCategory "
                        "塔板分馏，每条附 evidence 与 salience，产出 DistillateList。"
                    ),
                    system_prompt=EXTRACTOR_BATCH_SYSTEM,
                    response_format=DistillateList,  # type: ignore[call-arg]
                )
            )
        else:
            subagents.append(sa)

    # 3 个 intake SubAgent：内联声明（旧 build_intake_subagents 已删除，定义合并到此处）
    subagents.extend([
        SubAgent(
            name="intake_ner",
            description=(
                "一次性接收全部 chunk（JSON 数组），批量识别人物 + 规范化名字 + "
                "标注 speech/appearance/event 类别，产出 NerBatchResult（每个 chunk "
                "对应一个 NerBatchItem，必须透传 chunk_meta）。"
            ),
            system_prompt=INTAKE_NER_SYSTEM,
            response_format=NerBatchResult,  # type: ignore[call-arg]
        ),
        SubAgent(
            name="profile_builder",
            description=(
                "接收指定人物的索引条目 JSON 数组（来自 get_character_entries 工具），"
                "跨类别重排序，撰写 CharacterProfile（含 ≤200 字摘要）。"
            ),
            system_prompt=PROFILE_BUILDER_SYSTEM,
            response_format=CharacterProfile,  # type: ignore[call-arg]
        ),
        SubAgent(
            name="bridger",
            description=(
                "蒸馏产物落盘后的最后一公里汇报员：汇总 PersonaCard / Skills / "
                "PresetDialogue 的落盘路径与关键统计，给主理人最终回复用户的素材。"
                "不执行蒸馏——蒸馏由 extractor/synthesizer/skill_designer/dialogue_writer 完成。"
            ),
            system_prompt=BRIDGER_SYSTEM,
            # 不强制 response_format：bridger 的产出是给主理人汇报用的文本
        ),
    ])

    return create_deep_agent(
        model=build_model(cfg),
        system_prompt=intake_orchestrator_system(ctx.workdir),
        tools=tools,
        subagents=subagents,
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
# （random / time / functools.wraps / typing.Callable 已在文件顶部导入）
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


def _extract_first_json_object(text: str) -> str | None:
    """用栈匹配提取第一个完整 JSON 对象。返回 JSON 字符串或 None。

    通过花括号配对计数（栈匹配）找到第一个完整 ``{...}`` 子串；
    跟踪字符串字面量（``"`` 开关）忽略其中的花括号，转义的 ``\\"`` 不计。
    比 ``find("{")/rfind("}")`` 安全——后者会把"第一个 ``{`` 到最后一个 ``}``
    之间"的整段文本当成 JSON，跨多个对象或含花括号自然语言时必然解析失败。

    注意：本函数只做花括号配对，不验证 JSON 语法。无效候选（如 ``{我们}``）
    也会被返回，由调用方通过 ``json.loads`` 拒绝——这是 ``invoke_structured``
    抵御"含花括号自然语言误判"的最后一道防线。
    """
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]
    return None


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
        if not isinstance(content, str):
            continue
        # 用栈匹配提取第一个完整 JSON 对象，避免 find/rfind 把含花括号的
        # 自然语言（如"关于{我们}发现"）或跨多对象文本误截成非法 JSON
        candidate = _extract_first_json_object(content)
        if candidate is None:
            # 当前消息无可识别的花括号配对子串，继续看下一条
            continue
        try:
            return expected_type.model_validate(json.loads(candidate))
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
