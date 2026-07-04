"""三重验证法（Triple Verification）——参考 nuwa-skill 心智模型准入门槛。

一个候选心智模型必须同时通过三重验证才会被收录进 PersonaSkill：

1. **跨域复现 (Cross-Domain Recurrence)**
   该模型必须在此人讨论的 ≥2 个不同领域中出现。
   一次性表态不算，反复出现的才是真信念。
   示例：纳瓦尔的"杠杆"——在财富/个人成长/职业选择三域复现。

2. **有生成力 (Generative Power)**
   用这个模型可以推断此人对**新问题**的可能立场，而非只是描述既有观点。
   示例：芒格的"逆向思维"——面对"如何成功"→他先想"如何确保失败"。

3. **有排他性 (Exclusivity)**
   不是所有聪明人都会这样想，体现此人的独特视角。
   示例："反脆弱"是塔勒布的，不是通用智慧。

未通过的模型一律丢弃——宁缺毋滥，避免把通用常识误当人格特质。
"""
from __future__ import annotations

from dataclasses import dataclass

from persona_distillation.schemas import (
    Distillate,
    MentalModel,
    PersonaSignal,
    SignalCategory,
    VerificationResult,
)


@dataclass
class VerificationReport:
    """单个心智模型的验证报告。"""

    model: MentalModel
    passed: bool
    reasons: list[str]

    def apply(self) -> MentalModel:
        """把验证结果写回模型的 verification 字段。"""
        return self.model.model_copy(update={"verification": _to_result(self)})


def _to_result(report: VerificationReport) -> VerificationResult:
    v = report.model.verification
    return v.model_copy(
        update={
            "cross_domain": v.cross_domain,
            "generative": v.generative,
            "exclusive": v.exclusive,
        }
    )


# ---------------------------------------------------------------------------
# 规则式初筛：从 distillates 收集证据，给候选模型打分
# ---------------------------------------------------------------------------
def _collect_evidence(
    distillates: list[Distillate],
) -> list[tuple[SignalCategory, str, str]]:
    """汇总所有信号 (category, content, evidence)。"""
    out: list[tuple[SignalCategory, str, str]] = []
    for d in distillates:
        for s in d.signals:
            out.append((s.category, s.content, s.evidence))
    return out


def _domain_count(model: MentalModel, evidence: list[tuple[SignalCategory, str, str]]) -> tuple[bool, list[str]]:
    """跨域复现初筛：检查模型 principle/content 的关键词是否横跨多个 signal category。

    distillates 里的 SignalCategory 视作"领域"近似——不同类别代表不同语境。
    若模型关键词仅在单一类别出现，则跨域复现失败。
    """
    keywords = _keywords(model.principle + " " + model.name)
    if not keywords:
        return False, []
    domains: set[str] = set()
    proofs: list[str] = []
    for cat, content, ev in evidence:
        if any(k in content or k in ev for k in keywords):
            domains.add(cat.value)
            proofs.append(f"[{cat.value}] {ev[:60]}")
    passed = len(domains) >= 2
    return passed, proofs[:4]


def _keywords(text: str) -> list[str]:
    """从一句话里抽出关键词（长度≥2的中文片段 / 英文词）。"""
    import re

    # 中文连续片段
    cn = re.findall(r"[\u4e00-\u9fa5]{2,}", text)
    # 英文词
    en = [w for w in re.findall(r"[A-Za-z]{3,}", text)]
    return cn + en


def verify_mental_models(
    candidates: list[MentalModel],
    distillates: list[Distillate],
) -> list[VerificationReport]:
    """对一批候选心智模型执行三重验证。

    Parameters:
        candidates: skill_designer / synthesizer 提出的候选模型（已含 LLM 给的 verification 初判）。
        distillates: 全部蒸馏液，用于跨域证据收集。

    Returns:
        每个候选模型一份 :class:`VerificationReport`，``passed=True`` 才可收录。
    """
    evidence = _collect_evidence(distillates)
    reports: list[VerificationReport] = []

    for m in candidates:
        reasons: list[str] = []
        v = m.verification

        # 1. 跨域复现：用 distillates 证据复核 LLM 的初判
        cd_passed, proofs = _domain_count(m, evidence)
        if not v.cross_domain and not cd_passed:
            reasons.append("跨域复现失败：仅在单一领域出现，可能是一次性表态而非真信念")
            v = v.model_copy(
                update={
                    "cross_domain": False,
                    "cross_domain_evidence": proofs,
                }
            )
        else:
            v = v.model_copy(
                update={
                    "cross_domain": True,
                    "cross_domain_evidence": proofs or v.cross_domain_evidence,
                }
            )

        # 2. 有生成力：LLM 必须给出 generative_example
        if not v.generative or not v.generative_example.strip():
            reasons.append("生成力不足：未给出对新问题的立场推断")
            v = v.model_copy(update={"generative": False})
        else:
            v = v.model_copy(update={"generative": True})

        # 3. 有排他性：LLM 必须给出 exclusivity_note
        if not v.exclusive or not v.exclusivity_note.strip():
            reasons.append("排他性不足：未说明与常识的差异，可能是通用智慧而非个人特质")
            v = v.model_copy(update={"exclusive": False})
        else:
            v = v.model_copy(update={"exclusive": True})

        passed = v.passed
        if passed:
            reasons.append("✓ 三重验证通过：跨域复现 + 有生成力 + 有排他性")
        reports.append(VerificationReport(model=m.model_copy(update={"verification": v}), passed=passed, reasons=reasons))

    return reports


def filter_verified(
    candidates: list[MentalModel],
    distillates: list[Distillate],
) -> tuple[list[MentalModel], list[VerificationReport]]:
    """执行三重验证并只保留通过的模型。

    Returns:
        (通过模型列表, 全部验证报告)
    """
    reports = verify_mental_models(candidates, distillates)
    passed = [r.model for r in reports if r.passed]
    return passed, reports
