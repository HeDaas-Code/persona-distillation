"""WebUI 共享状态与工具函数。

集中存放 4 个 Tab 共用的内容，避免 ``tab_*.py`` 之间互相依赖：

- Gradio 跨版本兼容构造器（``_chatbot`` / ``_blocks`` / ``_GRADIO_MAJOR``）
- 通用文件/目录 helper（``_safe_read`` / ``_list_persona_dirs`` / ``_load_result``）
- 日志收集器 :class:`_LogHandler`（蒸馏 / OC 共创 / 评估三个 generator 共用）
- 渲染函数（``_format_result_summary`` / ``_render_*``）
- 评估区刷新函数（``_refresh_eval_area`` / ``_reset_eval_area``）
- 跨 Tab 联动函数（``_jump_to_browse`` / ``_jump_to_browse_refresh`` / ``_jump_to_distill``）
- 主题样式常量（``_CUSTOM_CSS`` / ``_HEADER_MD``）

各 ``tab_*.py`` 通过 ``from persona_distillation.webui.state import ...`` 复用；
``__init__.py`` 的 ``build_ui`` 在 4 个 Tab 组件都创建完成后，
用这里的联动函数组装跨 Tab 事件绑定。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import gradio as gr
    # Gradio 6.x 移除了 Chatbot 的 type 参数（默认 messages），4.x/5.x 需要 type="messages"
    _GRADIO_MAJOR = int(gr.__version__.split(".")[0])
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "WebUI 依赖 gradio，请先安装：pip install gradio>=4.44.0"
    ) from e


def _chatbot(**kwargs):
    """跨版本 Chatbot 构造：6.x 不再接受 type 参数。"""
    if _GRADIO_MAJOR >= 6:
        kwargs.pop("type", None)
    return gr.Chatbot(**kwargs)


def _blocks(**kwargs):
    """跨版本 Blocks 构造：6.x 把 theme/css 移到 launch()。"""
    if _GRADIO_MAJOR >= 6:
        kwargs.pop("theme", None)
        kwargs.pop("css", None)
    return gr.Blocks(**kwargs)


from persona_distillation.schemas import (
    AntiPattern,
    DecisionHeuristic,
    DistillationResult,
    EvalReport,
    ExpressionDNA,
    MentalModel,
    PersonaCard,
    PersonaSkill,
    PresetDialogue,
)


# ===========================================================================
# 通用 helper
# ===========================================================================
def _safe_read(p: Path, limit: int = 0) -> str:
    """读文件，limit>0 时只取前 limit 字符。"""
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
        return text if limit <= 0 else text[:limit]
    except Exception as e:  # noqa: BLE001
        return f"<读取失败: {e}>"


def _list_persona_dirs(base: str) -> list[str]:
    """列出 base 下含 ``persona_card.json`` 的子目录。"""
    base_path = Path(base).expanduser()
    if not base_path.exists():
        return []
    out: list[str] = []
    for sub in sorted(base_path.iterdir()):
        if sub.is_dir() and (sub / "persona_card.json").exists():
            out.append(str(sub))
    return out


def _load_result(out_dir: str) -> DistillationResult | None:
    """从输出目录加载 ``distillation_result.json``。"""
    p = Path(out_dir) / "distillation_result.json"
    if not p.exists():
        return None
    try:
        return DistillationResult.load(p)
    except Exception as e:  # noqa: BLE001
        logger.warning("加载 %s 失败: %s", p, e)
        return None


# ===========================================================================
# 日志收集器
# ===========================================================================
class _LogHandler(logging.Handler):
    """把 ``persona_distillation`` logger 的记录收集到一个 list 里。"""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))

    def reset(self) -> None:
        self.records.clear()


# ===========================================================================
# 渲染函数（蒸馏摘要 + 产物浏览 Markdown）
# ===========================================================================
def _format_result_summary(result: DistillationResult, out: Path) -> str:
    card = result.persona_card
    md = result.metadata
    lines = [
        "## ✅ 蒸馏完成",
        "",
        f"- **人格ID**: `{card.persona_id}`",
        f"- **显示名**: {card.display_name or '(未设置)'}",
        f"- **标签**: {', '.join(card.tags) or '(无)'}",
        f"- **报错回复**: {card.error_reply}",
        f"- **系统提示词长度**: {len(card.system_prompt)} 字符",
        f"- **Skills 数**: {len(result.skills)} → {', '.join(s.name for s in result.skills) or '(无)'}",
        f"- **预设对话数**: {len(result.preset_dialogues)}",
        f"- **蒸馏液块数**: {len(result.distillates)}",
        f"- **耗时**: {md.get('elapsed_sec', '?')} 秒",
        f"- **产物目录**: `{out}`",
        "",
        "### DNA 五层产出",
        f"- 🗣️ 表达 DNA: vocab={len(card.expression_dna.vocabulary)}, "
        f"rhythm={'✓' if card.expression_dna.rhythm else '✗'}, "
        f"signature_metaphors={len(card.expression_dna.signature_metaphors)}",
        f"- 🧠 心智模型: {len(card.mental_models)} 个（均通过三重验证）",
        f"- ⚖️ 决策启发式: {len(card.decision_heuristics)} 条",
        f"- 🚫 反模式: {len(card.anti_patterns)} 条",
        f"- 📏 诚实边界: {len(card.honest_boundaries)} 条",
        "",
        "> 💡 点击下方「👉 立即查看完整产物」按钮可一键跳转查看完整内容。",
    ]
    return "\n".join(lines)


def _render_persona_card_md(card: PersonaCard) -> str:
    """把 PersonaCard 渲染成 Markdown。"""
    lines = [
        f"# 🪪 人格卡：{card.display_name or card.persona_id}",
        "",
        f"- **人格ID**: `{card.persona_id}`",
        f"- **显示名**: {card.display_name or '(未设置)'}",
        f"- **标签**: {', '.join(card.tags) or '(无)'}",
        f"- **报错回复**: {card.error_reply}",
        f"- **特质摘要**: {card.traits_summary or '(未生成)'}",
        "",
        "## 📜 系统提示词",
        "",
        "```",
        card.system_prompt,
        "```",
    ]
    return "\n".join(lines)


def _render_dna_md(card: PersonaCard) -> str:
    """渲染 DNA 五层 + 三重验证证据。"""
    lines = ["# 🧬 DNA 五层认知操作系统", ""]

    # 1. 表达 DNA
    lines.append("## 🗣️ 表达 DNA (Expression DNA)")
    dna: ExpressionDNA = card.expression_dna
    if dna.vocabulary:
        lines.append(f"- **偏好词汇**: {', '.join(dna.vocabulary[:15])}")
    if dna.rhythm:
        lines.append(f"- **节奏**: {dna.rhythm}")
    if dna.rhetorical_tics:
        lines.append(f"- **修辞习惯**: {', '.join(dna.rhetorical_tics)}")
    if dna.signature_metaphors:
        lines.append(f"- **标志性比喻**: {', '.join(dna.signature_metaphors)}")
    if dna.opening_samples:
        lines.append("- **开场白示范**:")
        for s in dna.opening_samples:
            lines.append(f"  > {s}")
    if not (dna.vocabulary or dna.rhythm or dna.rhetorical_tics
            or dna.signature_metaphors or dna.opening_samples):
        lines.append("_（全空——可能 LLM 把 DNA 拼进了 system_prompt 文本，"
                     "检查 backfill 是否生效）_")
    lines.append("")

    # 2. 心智模型（含三重验证证据）
    lines.append("## 🧠 心智模型 (Mental Models)")
    lines.append("> 每个模型均通过三重验证：跨域复现 · 有生成力 · 有排他性")
    lines.append("")
    if not card.mental_models:
        lines.append("_（无——三重验证未通过任何模型，宁缺毋滥原则）_")
    for m in card.mental_models:
        lines.extend(_render_mental_model_md(m))
        lines.append("")

    # 3. 决策启发式
    lines.append("## ⚖️ 决策启发式 (Decision Heuristics)")
    if not card.decision_heuristics:
        lines.append("_（无）_")
    for h in card.decision_heuristics:
        lines.extend(_render_heuristic_md(h))
    lines.append("")

    # 4. 反模式
    lines.append("## 🚫 反模式 (Anti-Patterns) —— 绝对不会做什么")
    if not card.anti_patterns:
        lines.append("_（无）_")
    for a in card.anti_patterns:
        lines.extend(_render_anti_pattern_md(a))
    lines.append("")

    # 5. 诚实边界
    lines.append("## 📏 诚实边界 (Honest Boundaries)")
    if not card.honest_boundaries:
        lines.append("_（无）_")
    for b in card.honest_boundaries:
        lines.append(f"- ⚠️ **{b.limitation}** —— {b.reason}")
    return "\n".join(lines)


def _render_mental_model_md(m: MentalModel) -> list[str]:
    v = m.verification
    badge = "✅ 通过" if v.passed else "⚠️ 未全过"
    lines = [
        f"### {m.name}  `{badge}`",
        f"**原理**: {m.principle}",
    ]
    lines.append(f"- 跨域复现: {'✅' if v.cross_domain else '❌'}  "
                 + (f"{'; '.join(v.cross_domain_evidence)}" if v.cross_domain_evidence else ""))
    lines.append(f"- 有生成力: {'✅' if v.generative else '❌'}  "
                 + (v.generative_example if v.generative_example else ""))
    lines.append(f"- 有排他性: {'✅' if v.exclusive else '❌'}  "
                 + (v.exclusivity_note if v.exclusivity_note else ""))
    if m.application:
        lines.append(f"- **应用**: {m.application}")
    return lines


def _render_heuristic_md(h: DecisionHeuristic) -> list[str]:
    out = [f"- **{h.rule}** —— 触发：{h.trigger}"]
    if h.example:
        out.append(f"  - 例：{h.example}")
    return out


def _render_anti_pattern_md(a: AntiPattern) -> list[str]:
    out = [f"- 🚫 **{a.pattern}** —— {a.reason}"]
    if a.evidence:
        out.append(f"  - 证据：{a.evidence}")
    return out


def _render_skill_md(skill: PersonaSkill, out_dir: str) -> str:
    """读 SKILL.md 文件原文（已经由 skills_writer 落盘）。"""
    p = Path(out_dir) / "skills" / skill.name / "SKILL.md"
    if p.exists():
        return _safe_read(p)
    # 文件不存在就拼接一个最小版本
    return _render_skill_fallback(skill)


def _render_skill_fallback(sk: PersonaSkill) -> str:
    lines = [
        f"# {sk.name}",
        "",
        sk.description,
        "",
        f"**When to Use**: {sk.when_to_use or '(待补充)'}",
        "",
        "## Instructions",
        sk.instructions or "(待补充)",
    ]
    return "\n".join(lines)


def _render_dialogues(dialogues: list[PresetDialogue]) -> list[list[str]]:
    """渲染成 chat 气泡格式 [[user, assistant, intent], ...]。"""
    return [[d.user, d.assistant, d.intent or "—"] for d in dialogues]


# ===========================================================================
# 评估区刷新 / 重置（被 Tab 2 内部与跨 Tab 联动共享）
# ===========================================================================
def _refresh_eval_area(out_dir: str) -> tuple[str, Any, Any]:
    """加载产物后刷新评估区：根据 ``eval_report.json`` 是否存在展示内容 + 切换按钮可见性。

    返回三元组 ``(eval_md 内容, 生成按钮可见性, 重新评估按钮可见性)``：
    - ``out_dir`` 为空或 ``distillation_result.json`` 不存在（未加载产物）→ 评估区
      显示提示，两个按钮都隐藏
    - 已加载但 ``eval_report.json`` 不存在 → 显示"尚未评估"提示 + 显示「📊 生成评估」
    - 已加载且 ``eval_report.json`` 存在 → 显示 ``EvalReport.to_markdown()`` + 显示
      「🔄 重新评估」
    - ``eval_report.json`` 损坏 → 显示错误提示 + 显示「🔄 重新评估」让用户可覆盖重生成
    """
    if not out_dir:
        return ("_（未加载产物）_", gr.update(visible=False), gr.update(visible=False))

    # 检查是否已加载（distillation_result.json 存在才算"已加载"）
    result_path = Path(out_dir) / "distillation_result.json"
    if not result_path.exists():
        return ("_（未加载产物）_", gr.update(visible=False), gr.update(visible=False))

    eval_path = Path(out_dir) / "eval_report.json"
    if not eval_path.exists():
        return (
            "**尚未评估**——点下方「📊 生成评估」按钮启动 LLM-as-judge 评估"
            "（coverage 纯规则 + fidelity/identifiability 走 LLM judge）。",
            gr.update(visible=True),
            gr.update(visible=False),
        )
    try:
        report = EvalReport.from_json(eval_path)
        return (
            report.to_markdown(),
            gr.update(visible=False),
            gr.update(visible=True),
        )
    except Exception as e:  # noqa: BLE001
        return (
            f"⚠️ 评估报告加载失败：{e}\n\n可点「🔄 重新评估」覆盖重生成。",
            gr.update(visible=False),
            gr.update(visible=True),
        )


def _reset_eval_area() -> tuple[str, Any, Any]:
    """重置评估区到"未加载"状态。

    用于主理人 Agent Tab 的「查看最新产物」按钮——仅刷新下拉不预加载，
    评估区也应清空，与其它面板的"未加载"状态保持一致。
    """
    return ("_（未加载产物）_", gr.update(visible=False), gr.update(visible=False))


# ===========================================================================
# 全局跨 Tab 联动核心函数（所有蒸馏入口共享）
# ===========================================================================
def _jump_to_browse(out_dir: str, base: str, result_holder: dict[str, Any]) -> tuple:
    """全局联动：切到产物浏览 Tab + 刷新下拉 + 选中 + 加载。

    蒸馏 Tab / OC 共创 Tab / Agent Tab 三个入口共享此函数。
    返回 8 个组件更新，对应：
        (gr.Tabs(selected="browse"), gr.Dropdown(choices, value=out_dir),
         card_md, dna_md, skill_md, skill_dropdown, dialogues, status)
    """
    # 延迟导入避免 state ↔ tab_browse 循环依赖
    from persona_distillation.webui.tab_browse import _on_load_output

    opts = _list_persona_dirs(base)
    if out_dir and out_dir not in opts:
        # out_dir 可能不在 base 下（如自定义输出目录），确保选中项出现在列表里
        opts = [out_dir, *opts]
    load_result = _on_load_output(out_dir, base, result_holder)
    # load_result[0] 是 dropdown 的 gr.update()（无变更），这里用自己的刷新覆盖
    return (
        gr.Tabs(selected="browse"),
        gr.update(choices=opts, value=out_dir),
        load_result[1],  # card_md
        load_result[2],  # dna_md
        load_result[3],  # skill_md
        load_result[4],  # skill_dropdown
        load_result[5],  # dialogues
        load_result[6],  # status
    )


def _jump_to_browse_refresh(base: str) -> tuple:
    """切到产物浏览 Tab + 仅刷新下拉（不预选不加载）。

    用于主理人 Agent Tab：无法精确解析产物路径，让用户自己选。
    返回 8 个组件更新（与 :func:`_jump_to_browse` 结构一致）。
    """
    opts = _list_persona_dirs(base)
    value = opts[0] if opts else None
    return (
        gr.Tabs(selected="browse"),
        gr.update(choices=opts, value=value),
        "（请选择一个产物目录后点「加载」）",  # card_md
        "",  # dna_md
        "（未加载）",  # skill_md
        gr.update(choices=[], value=None),  # skill_dropdown
        [],  # dialogues
        "未加载",  # status
    )


def _jump_to_distill() -> tuple:
    """反向联动：切到蒸馏 Tab（语料路径由用户手动填，不可靠推断故不预填）。

    返回 1 个组件更新：``gr.Tabs(selected="distill")``
    """
    return (gr.Tabs(selected="distill"),)


# ===========================================================================
# 主题与样式
# ===========================================================================
_CUSTOM_CSS = """
.gradio-container { max-width: 1400px !important; }
.md-header { border-bottom: 1px solid var(--border-color-primary); padding-bottom: 8px; }
footer { display: none !important; }
"""

_HEADER_MD = """
# 🧪 人格蒸馏 · 调试 WebUI

> 四 Tab 调试面板：**蒸馏参数** · **产物浏览** · **主理人 Agent 对话** · **OC 共创** · 技术栈 Gradio
"""
