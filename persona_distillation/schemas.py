"""蒸馏输入/输出的结构化 schema。

字段命名与角色卡暗色界面一一对应：
- ``PersonaCard.persona_id``      → 左侧"人格ID"
- ``PersonaCard.system_prompt``   → 左侧"系统提示词"
- ``PersonaCard.error_reply``     → 左侧"自定义报错回复信息"
- ``PersonaSkill``                → 右侧"Skills 选择"
- ``PresetDialogue``              → 右侧"预设对话"
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class SignalCategory(str, Enum):
    """人格信号的分馏组分（fractional-distillation 的"塔板"）。"""

    SPEECH_STYLE = "speech_style"        # 说话风格：句式、语气、节奏
    CATCHPHRASE = "catchphrase"          # 口头禅 / 标志性表达
    VALUES = "values"                    # 价值观 / 信念
    KNOWLEDGE = "knowledge"              # 知识边界 / 擅长领域
    EMOTION = "emotion"                  # 情绪模式 / 反应倾向
    RELATIONSHIPS = "relationships"      # 关系网络 / 称谓习惯
    SIGNATURE_EVENT = "signature_event"  # 标志性事件 / 记忆锚点
    BACKGROUND = "background"            # 身世背景
    TABOO = "taboo"                      # 雷区 / 禁忌 / 不可触碰的话题
    MANNERISM = "mannerism"              # 小动作 / 习惯姿态


class PersonaSignal(BaseModel):
    """从单个分块中蒸馏出的一条人格信号。"""

    category: SignalCategory
    content: str = Field(..., description="用第一人称或客观陈述记录的人格特征")
    evidence: str = Field("", description="原文中支撑该判断的引文片段")
    salience: float = Field(
        0.5, ge=0.0, le=1.0, description="显著度，提纯阶段据此取舍"
    )


# ---------------------------------------------------------------------------
# DNA 级别认知操作系统（参考 nuwa-skill 五层方法论）
# ---------------------------------------------------------------------------
class VerificationResult(BaseModel):
    """心智模型的三重验证结果（nuwa-skill triple-verification）。"""

    cross_domain: bool = Field(
        False, description="跨域复现：该模型在 ≥2 个不同领域出现"
    )
    cross_domain_evidence: list[str] = Field(
        default_factory=list, description="跨域出现的领域与原文证据"
    )
    generative: bool = Field(
        False, description="有生成力：能预测此人对新问题的立场"
    )
    generative_example: str = Field("", description="生成力示例：一个新问题与推断立场")
    exclusive: bool = Field(
        False, description="有排他性：不是所有聪明人都会这样想"
    )
    exclusivity_note: str = Field("", description="排他性说明：与常识的差异点")

    @property
    def passed(self) -> bool:
        """三重验证全部通过才算合格心智模型。"""
        return self.cross_domain and self.generative and self.exclusive


class MentalModel(BaseModel):
    """心智模型——人物看世界的"镜片"。

    必须通过三重验证（跨域复现/生成力/排他性）才会被收录，
    体现的是 HOW they think 而非 WHAT they said。
    """

    name: str = Field(..., description="模型名称，如「聚焦即说不」")
    principle: str = Field(..., description="核心原理，一句话")
    verification: VerificationResult = Field(default_factory=VerificationResult)
    application: str = Field("", description="如何应用于新问题")


class DecisionHeuristic(BaseModel):
    """决策启发式——人物的推理捷径与判断规则。"""

    rule: str = Field(..., description="规则，如「先问物理极限，再谈优化」")
    trigger: str = Field(..., description="触发场景")
    example: str = Field("", description="原文中的应用实例")


class AntiPattern(BaseModel):
    """反模式——人物主动拒绝的事、价值观底线。

    体现「绝对不会做什么」，比正向规则更能勾勒人格边界。
    """

    pattern: str = Field(..., description="被拒绝的模式，如「把书当摆设」")
    reason: str = Field(..., description="拒绝理由")
    evidence: str = Field("", description="原文证据")


class HonestBoundary(BaseModel):
    """诚实边界——该 skill 真正做不到的事。

    一个不说明自身局限的 skill 不值得信任。
    参考 nuwa-skill：蒸馏不了直觉 / 捕捉不了突变 / 公开言论≠真实信念。
    """

    limitation: str = Field(..., description="做不到的事")
    reason: str = Field(..., description="为什么做不到")


class ExpressionDNA(BaseModel):
    """表达 DNA——语气、节奏、用词偏好、标志性比喻。

    不是语录合集，而是可复现的表达生成规则。
    """

    vocabulary: list[str] = Field(
        default_factory=list, description="高频/偏好词汇"
    )
    rhythm: str = Field("", description="句式节奏特征，如「短句、极端确定」")
    rhetorical_tics: list[str] = Field(
        default_factory=list, description="修辞习惯，如「二元对立」「反问」"
    )
    signature_metaphors: list[str] = Field(
        default_factory=list, description="标志性比喻"
    )
    opening_samples: list[str] = Field(
        default_factory=list, description="开场白示范"
    )


class Distillate(BaseModel):
    """单个分块的"蒸馏液"——分馏阶段的一次产出。"""

    source_file: str
    chunk_index: int
    char_start: int
    char_end: int
    signals: list[PersonaSignal] = Field(default_factory=list)
    summary: str = Field("", description="该分块的人格速写（≤120字）")


class PersonaCard(BaseModel):
    """人格卡——对应角色卡暗色界面左侧与基础信息。

    ``system_prompt`` 是最终注入目标 Agent 的系统提示词，
    ``persona_id`` 同时用作落盘子目录名。
    """

    persona_id: str = Field(..., description="人格ID，小写字母/数字/连字符")
    display_name: str = ""
    system_prompt: str = Field(..., description="系统提示词（完整可注入）")
    error_reply: str = Field(..., description="LLM 请求失败时的自定义报错回复")
    tags: list[str] = Field(default_factory=list)
    traits_summary: str = Field("", description="供下游 skills/对话阶段复用的特质摘要")
    # DNA 汇总：从 distillates 提纯得到，供 skill_designer 复用
    expression_dna: ExpressionDNA = Field(default_factory=ExpressionDNA)
    mental_models: list[MentalModel] = Field(default_factory=list)
    decision_heuristics: list[DecisionHeuristic] = Field(default_factory=list)
    anti_patterns: list[AntiPattern] = Field(default_factory=list)
    honest_boundaries: list[HonestBoundary] = Field(default_factory=list)


class PersonaSkill(BaseModel):
    """人格 Skill——对应界面右侧"Skills 选择"。

    DNA 级别认知操作系统（参考 nuwa-skill）：每个 skill 不再是简单流程说明，
    而是一套可运行的认知操作系统，包含五层——表达DNA/心智模型/决策启发式/反模式/诚实边界。

    每个 Skill 会落盘为一个目录 + ``SKILL.md``，遵循 Anthropic Agent Skills 规范，
    可被 deepagents 的 ``SkillsMiddleware`` 直接加载。
    """

    name: str = Field(..., description="小写字母数字与连字符，≤64字符")
    description: str = Field(..., description="该 skill 做什么，≤1024字符")
    when_to_use: str = ""
    # DNA 五层
    expression_dna: ExpressionDNA = Field(default_factory=ExpressionDNA)
    mental_models: list[MentalModel] = Field(
        default_factory=list,
        description="3~7 个经三重验证的心智模型；未通过的不得收录",
    )
    decision_heuristics: list[DecisionHeuristic] = Field(default_factory=list)
    anti_patterns: list[AntiPattern] = Field(default_factory=list)
    honest_boundaries: list[HonestBoundary] = Field(default_factory=list)
    # 兼容旧字段（落盘时合并进 instructions 段落）
    instructions: str = ""
    license: str = "MIT"


class PresetDialogue(BaseModel):
    """预设对话——对应界面右侧"预设对话"的一组对话对。"""

    user: str
    assistant: str
    intent: str = Field("", description="该对话对展示的人格侧重点")


class DistillationResult(BaseModel):
    """一次完整蒸馏的最终产物。"""

    persona_card: PersonaCard
    skills: list[PersonaSkill] = Field(default_factory=list)
    preset_dialogues: list[PresetDialogue] = Field(default_factory=list)
    distillates: list[Distillate] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ---- 落盘 -----------------------------------------------------------
    def save(self, output_dir: str | Path) -> Path:
        """将结果写入目录，结构与角色卡界面字段对齐。"""
        from persona_distillation.renderer import render_persona_card
        from persona_distillation.skills_writer import write_skills

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        render_persona_card(self, out)
        write_skills(self.persona_card.persona_id, self.skills, out)

        (out / "preset_dialogues.json").write_text(
            self.model_dump_json(indent=2, include={"preset_dialogues"}),
            encoding="utf-8",
        )
        (out / "distillates.jsonl").write_text(
            "\n".join(d.model_dump_json() for d in self.distillates),
            encoding="utf-8",
        )
        (out / "distillation_result.json").write_text(
            self.model_dump_json(indent=2, exclude={"distillates"}),
            encoding="utf-8",
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> "DistillationResult":
        import json

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        try:
            return cls.model_validate(data)
        except ValidationError:
            # 兼容只存了 persona_card 的情况
            return cls.model_validate({"persona_card": data})
