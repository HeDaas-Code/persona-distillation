"""蒸馏质量评估子包。

对外导出三个评估器入口 + 汇总器 + 数据契约：

- :func:`evaluate_coverage`        —— 纯规则覆盖度评估
- :func:`evaluate_fidelity`        —— LLM-as-judge 忠实度评估
- :func:`evaluate_identifiability` —— LLM-as-judge 可识别度评估（盲猜）
- :func:`build_report`             —— 汇总三个评估器，产出 :class:`EvalReport`
- :class:`EvalReport` / :class:`CoverageScore` / :class:`FidelityScore` /
  :class:`IdentifiabilityScore` —— 数据契约（定义在 :mod:`persona_distillation.schemas`）
"""
from persona_distillation.eval.coverage import evaluate as evaluate_coverage
from persona_distillation.eval.fidelity import evaluate as evaluate_fidelity
from persona_distillation.eval.identifiability import (
    evaluate as evaluate_identifiability,
)
from persona_distillation.eval.report import build_report
from persona_distillation.schemas import (
    CoverageScore,
    EvalReport,
    FidelityScore,
    IdentifiabilityScore,
)

__all__ = [
    "evaluate_coverage",
    "evaluate_fidelity",
    "evaluate_identifiability",
    "build_report",
    "EvalReport",
    "CoverageScore",
    "FidelityScore",
    "IdentifiabilityScore",
]
