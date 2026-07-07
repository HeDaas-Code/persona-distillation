"""Pydantic 数据契约：所有跨模块流动的数据结构都在这里定义。

字段命名与角色卡暗色界面一一对应：
- ``PersonaCard.persona_id``      → 左侧"人格ID"
- ``PersonaCard.system_prompt``   → 左侧"系统提示词"
- ``PersonaCard.error_reply``     → 左侧"自定义报错回复信息"
- ``PersonaSkill``                → 右侧"Skills 选择"
- ``PresetDialogue``              → 右侧"预设对话"

DNA 五层（均在 :class:`PersonaCard` / :class:`PersonaSkill` 中）：
:class:`ExpressionDNA` / :class:`MentalModel` / :class:`DecisionHeuristic`
/ :class:`AntiPattern` / :class:`HonestBoundary`
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)


# 当前 schema 版本号。修改任何持久化模型字段时必须 bump 该值。
SCHEMA_VERSION = 1


class SchemaVersionError(ValueError):
    """加载数据时 schema_version 与当前不匹配。"""


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

    schema_version: int = Field(default=SCHEMA_VERSION, description="schema 版本号")
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

    # P0 红线：LLM 经常塞 schema 外字段（如 ``tools`` / ``skills``），
    # 一律拒绝，让前端/上游调用方显式收敛。
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION, description="schema 版本号")
    persona_id: str = Field(..., description="人格ID，小写字母/数字/连字符")
    display_name: str = ""
    system_prompt: str = Field(..., description="系统提示词（完整可注入）")
    error_reply: str = Field(
        default="", description="LLM 请求失败时的自定义报错回复（可空，pipeline 兜底）"
    )
    tags: list[str] = Field(default_factory=list)
    traits_summary: str = Field("", description="供下游 skills/对话阶段复用的特质摘要")
    # DNA 汇总：从 distillates 提纯得到，供 skill_designer 复用
    expression_dna: ExpressionDNA = Field(default_factory=ExpressionDNA)
    mental_models: list[MentalModel] = Field(default_factory=list)
    decision_heuristics: list[DecisionHeuristic] = Field(default_factory=list)
    anti_patterns: list[AntiPattern] = Field(default_factory=list)
    honest_boundaries: list[HonestBoundary] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        # P1: DNA 完整性检查——5 字段全空时报 WARNING，提示后处理回填
        edna_empty = (
            not self.expression_dna.vocabulary
            and not self.expression_dna.rhythm
            and not self.expression_dna.rhetorical_tics
            and not self.expression_dna.signature_metaphors
            and not self.expression_dna.opening_samples
        )
        all_empty = (
            edna_empty
            and len(self.mental_models) == 0
            and len(self.decision_heuristics) == 0
            and len(self.anti_patterns) == 0
            and len(self.honest_boundaries) == 0
        )
        if all_empty:
            logger.warning(
                "PersonaCard(persona_id=%s) DNA 字段全空, system_prompt 长度=%d; 建议后处理回填",
                self.persona_id,
                len(self.system_prompt or ""),
            )


class PersonaSkill(BaseModel):
    """人格 Skill——对应界面右侧"Skills 选择"。

    DNA 级别认知操作系统（参考 nuwa-skill）：每个 skill 不再是简单流程说明，
    而是一套可运行的认知操作系统，包含五层——表达DNA/心智模型/决策启发式/反模式/诚实边界。

    每个 Skill 会落盘为一个目录 + ``SKILL.md``，遵循 Anthropic Agent Skills 规范，
    可被 deepagents 的 ``SkillsMiddleware`` 直接加载。
    """

    schema_version: int = Field(default=SCHEMA_VERSION, description="schema 版本号")
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

    schema_version: int = Field(default=SCHEMA_VERSION, description="schema 版本号")
    user: str
    assistant: str
    intent: str = Field("", description="该对话对展示的人格侧重点")


class DistillationResult(BaseModel):
    """一次完整蒸馏的最终产物。"""

    schema_version: int = Field(default=SCHEMA_VERSION, description="schema 版本号")
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

        # renderer 用标准库 json.dumps 序列化 metadata，不支持 Pydantic 对象；
        # 若 metadata 含 EvalReport，渲染时传一份 model_dump() 后的纯 dict 副本，
        # 避免触发 "Object of type EvalReport is not JSON serializable"。
        eval_report = self.metadata.get("eval_report")
        has_eval_report = isinstance(eval_report, EvalReport)
        if has_eval_report:
            safe_result = self.model_copy(
                update={
                    "metadata": {
                        k: (
                            v.model_dump()
                            if isinstance(v, EvalReport)
                            else v
                        )
                        for k, v in self.metadata.items()
                    }
                }
            )
            render_persona_card(safe_result, out)
        else:
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
        # 若 metadata 含 EvalReport 对象，额外落盘 eval_report.json
        # （EvalReport 与本类同文件定义，无需额外 import）
        if has_eval_report:
            (out / "eval_report.json").write_text(
                eval_report.model_dump_json(indent=2),
                encoding="utf-8",
            )
        logger.info("distillation_result saved to %s", out)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "DistillationResult":
        """从 ``distillation_result.json`` 加载（带 schema 版本校验）。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        # 1. schema_version 校验（P1-3）
        ver = data.get("schema_version")
        if ver is None:
            # 旧文件无版本号，假定 v1（兼容期）
            logger.warning("加载的 distillation_result.json 无 schema_version 字段，假定 v%d", SCHEMA_VERSION)
            data["schema_version"] = SCHEMA_VERSION
        elif ver != SCHEMA_VERSION:
            raise SchemaVersionError(
                f"schema_version 不匹配：文件={ver}，当前={SCHEMA_VERSION}。"
                f" 请运行 migration 或重新蒸馏。"
            )

        try:
            return cls.model_validate(data)
        except ValidationError:
            # 兼容只存了 persona_card 的情况
            return cls.model_validate({"persona_card": data, "schema_version": SCHEMA_VERSION})


# ---------------------------------------------------------------------------
# 蒸馏质量评估数据契约（eval/ 子包产出）
# ---------------------------------------------------------------------------
class CoverageScore(BaseModel):
    """覆盖度评估结果（纯规则统计，无需 LLM）。"""

    total_score: float = Field(ge=0.0, le=1.0, description="加权汇总分 ∈ [0,1]")
    details: dict[str, Any] = Field(
        default_factory=dict, description="各分项原始数值，人类可读"
    )
    expression_dna_richness: float = Field(
        ge=0.0, le=1.0, description="表达DNA丰富度"
    )
    mental_model_count: int = Field(
        ge=0, description="心智模型总数（含未通过验证的）"
    )
    mental_model_verification_pass_rate: float = Field(
        ge=0.0, le=1.0, description="三重验证通过率"
    )
    anti_pattern_count: int = Field(ge=0)
    honest_boundary_count: int = Field(ge=0)
    decision_heuristic_count: int = Field(ge=0)


class FidelityScore(BaseModel):
    """忠实度评估结果（LLM-as-judge）。score=-1 表示 judge 调用失败。"""

    score: float = Field(
        ge=-1.0, le=1.0, description="忠实度分数 ∈ [0,1]，-1 表示失败"
    )
    reasons: list[str] = Field(
        default_factory=list, description="judge 给出的理由（通常 3 条）"
    )


class IdentifiabilityScore(BaseModel):
    """可识别度评估结果（LLM-as-judge 盲猜人物）。confidence=-1 表示失败。"""

    probes: list[str] = Field(
        default_factory=list, description="probe Q&A 文本列表"
    )
    guessed_name: str = Field(default="", description="judge 猜的人物名")
    confidence: float = Field(
        ge=-1.0, le=1.0, description="judge 置信度 ∈ [0,1]，-1 表示失败"
    )
    correct: bool = Field(default=False, description="盲猜是否正确（模糊匹配）")
    error: str = Field(default="", description="失败时的错误信息")


class EvalReport(BaseModel):
    """蒸馏质量评估报告。"""

    coverage: CoverageScore
    fidelity: FidelityScore | None = None
    identifiability: IdentifiabilityScore | None = None
    overall_score: float = Field(
        ge=0.0,
        le=1.0,
        description="加权汇总（coverage 0.3 + fidelity 0.4 + identifiability 0.3）",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="评估时间",
    )

    @classmethod
    def from_json(cls, path: str | Path) -> "EvalReport":
        """从 ``eval_report.json`` 反序列化。

        ``path`` 既可以是含 ``eval_report.json`` 的目录，也可以是该文件本身。
        损坏或结构不匹配时抛清晰错误，便于上游 CLI 友好提示。
        """
        p = Path(path)
        if p.is_dir():
            p = p / "eval_report.json"
        if not p.exists():
            raise FileNotFoundError(f"评估报告不存在: {p}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"评估报告 JSON 损坏: {p}: {e}") from e
        try:
            return cls.model_validate(data)
        except ValidationError as e:
            raise ValueError(
                f"评估报告结构与 EvalReport 不匹配: {p}: {e}"
            ) from e

    def to_markdown(self) -> str:
        """人类可读 Markdown 摘要，含总评分与所有非 None 分项。"""
        lines: list[str] = [
            "# 蒸馏质量评估报告",
            "",
            f"- **总评分**: {self.overall_score:.3f}",
            f"- **评估时间**: {self.created_at}",
            "",
        ]

        # 覆盖度（必有）
        c = self.coverage
        lines.append("## 覆盖度 (Coverage)")
        lines.append(f"- 总分: {c.total_score:.3f}")
        lines.append(f"- 表达DNA丰富度: {c.expression_dna_richness:.3f}")
        lines.append(
            f"- 心智模型数: {c.mental_model_count}"
            f"（通过率 {c.mental_model_verification_pass_rate:.2%}）"
        )
        lines.append(f"- 反模式数: {c.anti_pattern_count}")
        lines.append(f"- 诚实边界数: {c.honest_boundary_count}")
        lines.append(f"- 决策启发式数: {c.decision_heuristic_count}")
        if c.details:
            lines.append("- 明细:")
            for k, v in c.details.items():
                lines.append(f"  - {k}: {v}")
        lines.append("")

        # 忠实度（可选）
        if self.fidelity is not None:
            f = self.fidelity
            lines.append("## 忠实度 (Fidelity)")
            if f.score < 0:
                lines.append(f"- 评估失败（score={f.score}）")
            else:
                lines.append(f"- 分数: {f.score:.3f}")
            if f.reasons:
                lines.append("- 理由:")
                for r in f.reasons:
                    lines.append(f"  - {r}")
            lines.append("")

        # 可识别度（可选）
        if self.identifiability is not None:
            i = self.identifiability
            lines.append("## 可识别度 (Identifiability)")
            if i.confidence < 0:
                lines.append(f"- 评估失败（confidence={i.confidence}）")
                if i.error:
                    lines.append(f"- 错误: {i.error}")
            else:
                lines.append(f"- 猜测人物: {i.guessed_name or '(空)'}")
                lines.append(f"- 置信度: {i.confidence:.3f}")
                lines.append(f"- 盲猜正确: {'是' if i.correct else '否'}")
            if i.probes:
                lines.append("- Probe Q&A:")
                for idx, q in enumerate(i.probes, 1):
                    lines.append(f"  {idx}. {q}")
            lines.append("")

        return "\n".join(lines)
