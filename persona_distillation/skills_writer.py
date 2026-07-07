"""把 PersonaSkill 列表写成 deepagents SkillsMiddleware 可加载的目录结构。

每个 skill 落盘为 ``skills/<name>/SKILL.md``，遵循 Anthropic Agent Skills 规范：
YAML frontmatter（name/description/license）+ Markdown 正文。

DNA 级别模板（参考 nuwa-skill）：
  - 角色扮演规则（用「我」、免责声明一次、退出锚）
  - 回答工作流 (Agentic Protocol)
  - 心智模型（含三重验证证据）
  - 决策启发式
  - 表达 DNA
  - 反模式
  - 诚实边界
"""
from __future__ import annotations

import re
from pathlib import Path

from persona_distillation.schemas import (
    ExpressionDNA,
    HonestBoundary,
    PersonaSkill,
)

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def _ensure_unique(name: str, existing: list[str] | set[str] | None) -> str:
    """若 ``name`` 已在 ``existing`` 中，追加 ``-2``、``-3``… 直到唯一。

    用于避免多个非法 skill 名 sanitize 成相同 fallback 后产生目录冲突。
    """
    if not existing:
        return name
    seen_set = set(existing)
    if name not in seen_set:
        return name
    i = 2
    while f"{name}-{i}" in seen_set:
        i += 1
    return f"{name}-{i}"


def _sanitize_skill_name(
    name: str,
    fallback: str = "skill",
    existing: list[str] | set[str] | None = None,
) -> str:
    """清洗 skill 名并去重。

    - 小写、仅保留 ``a-z0-9-``、合并多余连字符
    - 不合法（空或不符合 ``_NAME_RE``）则用 ``fallback``
    - 若结果在 ``existing`` 中已存在，追加 ``-2``、``-3``… 直到唯一
    """
    n = name.strip().lower()
    n = re.sub(r"[^a-z0-9-]+", "-", n)
    n = re.sub(r"-+", "-", n).strip("-")
    if not n or not _NAME_RE.match(n):
        n = fallback
    return _ensure_unique(n, existing)


def write_skills(
    persona_id: str,
    skills: list[PersonaSkill],
    out_dir: str | Path,
) -> list[Path]:
    """写出全部 skill 目录，返回 SKILL.md 路径列表。"""
    out = Path(out_dir) / "skills"
    out.mkdir(parents=True, exist_ok=True)

    pid = _sanitize_skill_name(persona_id, "persona")
    written: list[Path] = []
    seen_names: set[str] = set()

    for i, sk in enumerate(skills):
        # 先清洗（不加前缀），再用 pid 前缀补齐，最后对 seen_names 去重
        name = _sanitize_skill_name(sk.name, f"{pid}-skill-{i}")
        if not name.startswith(pid):
            name = f"{pid}-{name}"[:64]
        # 加前缀后可能再次冲突，统一在这里确保唯一
        name = _ensure_unique(name, seen_names)
        seen_names.add(name)

        skill_dir = out / name
        skill_dir.mkdir(exist_ok=True)
        md = _skill_md(name, pid, sk)
        target = skill_dir / "SKILL.md"
        target.write_text(md, encoding="utf-8")
        written.append(target)
    return written


def _skill_md(name: str, persona_id: str, sk: PersonaSkill) -> str:
    desc = (sk.description or "").strip()[:1024]
    when = (sk.when_to_use or "").strip()
    license_ = (sk.license or "MIT").strip()
    display_name = persona_id.replace("-", " ").title()

    parts: list[str] = [
        "---",
        f"name: {name}",
        f"description: {desc}",
        f"license: {license_}",
        "---",
        "",
        f"# {display_name} · 思维操作系统",
        "",
        "## 角色扮演规则（最重要）",
        f"**此 Skill 激活后，直接以 {display_name} 的身份回应。**",
        "- 用「我」而非第三人称转述",
        "- 直接用此人的语气、节奏、词汇回答问题",
        "- 遇到不确定的问题，用此人会有的方式回应——或直接质疑问题本身，或给出出人意料的类比",
        ("- 🛑 **STOP（仅一次）**：首次激活时输出免责声明一次——"
         "「我以该视角和你聊，基于公开语料推断，非本人观点」。后续对话**绝不**重复，重复算违规"),
        "- 不说「如果是XX，他可能会...」这类跳出角色的 meta 表述",
        ('🚪 **EXIT TRIGGER（显性退出锚）**：用户说「退出」「切回正常」「不用扮演了」'
         "「跳出角色」时 → 立即恢复正常模式"),
        "",
        "## When to Use",
        when or "- 见 description。",
        "",
        "## 回答工作流 (Agentic Protocol)",
        (sk.instructions or "（待补充）").strip() or
        "1. 先用 mental_models 中的镜片重新框定用户问题。\n"
        "2. 用 decision_heuristics 给出判断，而非直接给答案。\n"
        "3. 用 expression_dna 的节奏与词汇表达。\n"
        "4. 触及 anti_patterns 时果断拒绝，而非顺从。",
        "",
    ]

    # ---- 心智模型（含三重验证证据） ----
    if sk.mental_models:
        parts.append("## 心智模型 (Mental Models)")
        parts.append("> 每个模型均通过三重验证：跨域复现 · 有生成力 · 有排他性。")
        parts.append("")
        for m in sk.mental_models:
            parts.append(f"### {m.name}")
            parts.append(f"**原理**：{m.principle}")
            v = m.verification
            if v.cross_domain_evidence:
                parts.append("**跨域复现证据**：")
                for ev in v.cross_domain_evidence:
                    parts.append(f"- {ev}")
            if v.generative_example:
                parts.append(f"**生成力示例**：{v.generative_example}")
            if v.exclusivity_note:
                parts.append(f"**排他性**：{v.exclusivity_note}")
            if m.application:
                parts.append(f"**应用**：{m.application}")
            parts.append("")

    # ---- 决策启发式 ----
    if sk.decision_heuristics:
        parts.append("## 决策启发式 (Decision Heuristics)")
        for h in sk.decision_heuristics:
            parts.append(f"- **{h.rule}** —— 触发：{h.trigger}")
            if h.example:
                parts.append(f"  - 例：{h.example}")
        parts.append("")

    # ---- 表达 DNA ----
    dna = sk.expression_dna
    parts.append("## 表达 DNA (Expression DNA)")
    parts.append(_render_dna(dna))
    parts.append("")

    # ---- 反模式 ----
    if sk.anti_patterns:
        parts.append("## 反模式 (Anti-Patterns) —— 绝对不会做什么")
        for a in sk.anti_patterns:
            parts.append(f"- 🚫 **{a.pattern}** —— {a.reason}")
            if a.evidence:
                parts.append(f"  - 证据：{a.evidence}")
        parts.append("")

    # ---- 诚实边界 ----
    parts.append("## 诚实边界 (Honest Boundaries)")
    boundaries = list(sk.honest_boundaries)
    # 兜底：至少声明两条通用局限
    if not any("直觉" in b.limitation for b in boundaries):
        boundaries.append(HonestBoundary(
            limitation="无法蒸馏直觉",
            reason="框架能提取，灵感不能。该 skill 只复现可表述的思考路径。",
        ))
    if not any("快照" in b.limitation or "公开" in b.limitation for b in boundaries):
        boundaries.append(HonestBoundary(
            limitation="仅基于公开语料的快照",
            reason="不等于本人真实信念，且不随本人后续变化更新。",
        ))
    for b in boundaries:
        parts.append(f"- ⚠️ **{b.limitation}** —— {b.reason}")
    parts.append("")

    return "\n".join(parts)


def _render_dna(dna: ExpressionDNA) -> str:
    lines: list[str] = []
    if dna.vocabulary:
        lines.append(f"- **偏好词汇**：{', '.join(dna.vocabulary[:12])}")
    if dna.rhythm:
        lines.append(f"- **节奏**：{dna.rhythm}")
    if dna.rhetorical_tics:
        lines.append(f"- **修辞习惯**：{', '.join(dna.rhetorical_tics)}")
    if dna.signature_metaphors:
        lines.append(f"- **标志性比喻**：{', '.join(dna.signature_metaphors)}")
    if dna.opening_samples:
        lines.append("- **开场白示范**：")
        for s in dna.opening_samples:
            lines.append(f"  > {s}")
    if not lines:
        lines.append("- （待提炼）")
    return "\n".join(lines)
