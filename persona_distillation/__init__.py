"""Persona Distillation Framework.

基于 LangChain DeepAgents 的人格蒸馏框架：摄入多文本长文本语料，通过
"分馏 → 冷凝 → 提纯" 三段式蒸馏方法论，产出符合角色卡界面结构的人格卡
（人格ID / 系统提示词 / 自定义报错回复）、人格 Skills 与预设对话。

典型用法::

    from persona_distillation import PersonaDistiller, DistillationConfig

    distiller = PersonaDistiller(DistillationConfig())  # 默认 minimax:MiniMax-M3
    result = distiller.distill("./corpus", persona_id="arakawa_sensei")
    result.save("./output")

环境变量::

    export MINIMAX_API_KEY=sk-...   # 见 https://platform.minimax.io
"""

from persona_distillation.config import DistillationConfig
from persona_distillation.schemas import (
    AntiPattern,
    DecisionHeuristic,
    Distillate,
    DistillationResult,
    ExpressionDNA,
    HonestBoundary,
    MentalModel,
    PersonaCard,
    PersonaSignal,
    PersonaSkill,
    PresetDialogue,
    SignalCategory,
    VerificationResult,
)
from persona_distillation.pipeline import PersonaDistiller
from persona_distillation.loader import LoadedDoc, load_corpus
from persona_distillation.chunker import Chunk, chunk_text
from persona_distillation.triple_verification import verify_mental_models

__all__ = [
    "DistillationConfig",
    "PersonaDistiller",
    "PersonaCard",
    "PersonaSignal",
    "SignalCategory",
    "Distillate",
    "PersonaSkill",
    "PresetDialogue",
    "DistillationResult",
    "LoadedDoc",
    "load_corpus",
    "Chunk",
    "chunk_text",
    # DNA 级别
    "ExpressionDNA",
    "MentalModel",
    "VerificationResult",
    "DecisionHeuristic",
    "AntiPattern",
    "HonestBoundary",
    "verify_mental_models",
]

__version__ = "0.1.0"
