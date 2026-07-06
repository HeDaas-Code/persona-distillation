"""Gradio WebUI——人格蒸馏框架的调试面板。

四个 Tab：
- **蒸馏**：表单填参数 → 后台跑 ``PersonaDistiller.distill`` → 实时日志 → 完成摘要
- **产物浏览**：选 ``out/`` 子目录 → 渲染 persona_card / DNA 五层 / Skills / 预设对话
- **主理人 Agent**：聊天界面驱动 intake 子包（人物识别 → 档案 → 蒸馏）
- **OC 共创**：捏造 OC 设定 → 三阶段串联（骨架生成 → 血肉访谈 → 蒸馏）

所有蒸馏入口完成后均可一键跳转到产物浏览 Tab（全局跨 Tab 联动）。

启动方式::

    python -m persona_distillation.main webui [--host 0.0.0.0] [--port 7860] [--share]

无需 API key 也能启动（dry_run 模式下蒸馏 Tab 会直接报缺 key 的错，但
产物浏览 Tab 仍可离线查看已有产出）。
"""
from __future__ import annotations

import logging
import traceback
from collections.abc import Generator
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


from persona_distillation.config import DistillationConfig
from persona_distillation.pipeline import PersonaDistiller
from persona_distillation.schemas import (
    AntiPattern,
    DecisionHeuristic,
    DistillationResult,
    ExpressionDNA,
    HonestBoundary,
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
# Tab 1：蒸馏面板
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


def _run_distill(
    input_path: str,
    output_dir: str,
    model: str,
    persona_id: str,
    chunk_size: int,
    chunk_overlap: int,
    max_chunks: int,
    salience: float,
    max_skills: int,
    max_dialogues: int,
    error_reply: str,
    workdir: str,
    debug: bool,
) -> Generator[tuple[str, str, Any], None, None]:
    """执行蒸馏，流式输出日志。yield 给 Gradio 让 UI 实时刷新。

    yield 三元组 ``(日志, 摘要, 联动按钮可见性更新)``：
    - 蒸馏成功后第 3 项为 ``gr.update(visible=True)`` 显示「立即查看完整产物」按钮
    - 其他情况第 3 项为 ``gr.update(visible=False)`` 隐藏按钮
    """
    btn_hidden = gr.update(visible=False)
    log_box = ""
    yield log_box, "🔧 构造配置中...", btn_hidden

    if not input_path.strip():
        yield "❌ 错误：语料路径不能为空", "", btn_hidden
        return
    if not output_dir.strip():
        yield "❌ 错误：输出目录不能为空", "", btn_hidden

    # 安装日志收集器
    handler = _LogHandler()
    handler.setLevel(logging.DEBUG if debug else logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")
    )
    pd_logger = logging.getLogger("persona_distillation")
    pd_logger.addHandler(handler)
    prev_level = pd_logger.level
    pd_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    try:
        cfg = DistillationConfig(
            model=model,
            persona_id=persona_id or "",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_chunks_per_file=max_chunks,
            salience_threshold=salience,
            max_skills=max_skills,
            max_preset_dialogues=max_dialogues,
            default_error_reply=error_reply,
            workdir=workdir or "",
            debug=debug,
        )
    except Exception as e:  # noqa: BLE001
        pd_logger.removeHandler(handler)
        pd_logger.setLevel(prev_level)
        yield log_box + f"\n❌ 配置错误：{e}", "", btn_hidden
        return

    log_box = "\n".join(handler.records) + f"\n🚀 启动 PersonaDistiller (model={model})...\n"
    yield log_box, "🚀 启动蒸馏中...", btn_hidden

    try:
        distiller = PersonaDistiller(cfg)
        result = distiller.distill(input_path, output_dir=output_dir)

        # 把 handler 累积的最后日志合进去
        log_box = "\n".join(handler.records)
        summary = _format_result_summary(result, Path(output_dir))
        yield log_box + "\n\n✅ 蒸馏完成！", summary, gr.update(visible=True)
    except Exception as e:  # noqa: BLE001
        log_box = "\n".join(handler.records)
        tb = traceback.format_exc()
        yield log_box + f"\n\n❌ 蒸馏失败：{e}\n\n{tb}", "", btn_hidden
    finally:
        pd_logger.removeHandler(handler)
        pd_logger.setLevel(prev_level)


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


# ===========================================================================
# OC 共创三阶段串联 generator（Phase 1 骨架 → Phase 2 血肉 → Phase 3 蒸馏）
# ===========================================================================
def _run_cocreate(
    oc_name: str,
    oc_age: str,
    oc_background: str,
    oc_traits: str,
    oc_worldview: str,
    oc_catchphrase: str,
    n_rounds: int,
    persona_id: str,
    model: str,
    output_dir: str,
    workdir: str,
    chunk_size: int,
    chunk_overlap: int,
    salience: float,
    max_skills: int,
    max_dialogues: int,
    error_reply: str,
    debug: bool,
) -> Generator[tuple[str, str, Any], None, None]:
    """OC 共创三阶段串联 generator。

    流式 yield ``(日志, 摘要, 联动按钮可见性更新)`` 三元组：
    - **Phase 1 骨架**：4 类文本逐个生成（独白/对话/事件/回忆）
    - **Phase 2 血肉**：N 轮访谈（``run_interview`` 同步执行，先 yield 进度后显示结果）
    - **Phase 3 蒸馏**：骨架 + 血肉合并入 ``PersonaDistiller``

    自 Phase 1 之前就安装 :class:`_LogHandler` 到 ``persona_distillation`` logger，
    让 ``oc_writer`` / ``interview`` / ``PersonaDistiller`` 内部的 ``logger.info``
    全程自动进 handler，进而流式进 WebUI 日志框（不再只显示阶段级 start/完成 标记）。
    generator 的 ``finally`` 里统一 ``removeHandler``，确保异常路径也清理。

    每阶段 try/except，失败时 yield 错误日志并 return，已生成的产物保留不清理。
    Phase 3 成功后第 3 项为 ``gr.update(visible=True)`` 显示联动按钮。
    """
    from persona_distillation.agents import build_model
    from persona_distillation.intake.interview import run_interview
    from persona_distillation.intake.oc_writer import OCSetting, generate_oc_corpus

    btn_hidden = gr.update(visible=False)
    log_box = ""
    yield log_box, "🔧 构造配置中...", btn_hidden

    # ---- 基本校验 ----
    if not oc_name.strip():
        yield "❌ 错误：OC 姓名不能为空", "", btn_hidden
        return
    if not output_dir.strip():
        yield "❌ 错误：输出目录不能为空", "", btn_hidden
        return
    if not workdir.strip():
        yield "❌ 错误：工作目录不能为空（OC 共创需要工作目录落盘骨架与访谈）", "", btn_hidden
        return

    # persona_id 为空时从姓名派生（复用 bridge.slugify，失败则用极简兜底）
    pid = persona_id.strip()
    if not pid:
        try:
            from persona_distillation.intake.bridge import slugify
            pid = slugify(oc_name.strip()) or "oc"
        except Exception:  # noqa: BLE001
            pid = (
                "".join(
                    c if c.isalnum() or c in "-_" else "_"
                    for c in oc_name.strip()
                )
                or "oc"
            )
        log_box += f"⚠️ persona_id 为空，从姓名派生为 `{pid}`\n"

    # ---- 构造 OCSetting ----
    try:
        setting = OCSetting(
            name=oc_name.strip(),
            age=oc_age.strip() or "未知",
            background=oc_background.strip(),
            traits=oc_traits.strip(),
            worldview=oc_worldview.strip(),
            catchphrase=oc_catchphrase.strip(),
        )
    except Exception as e:  # noqa: BLE001
        yield log_box + f"\n❌ OC 设定构造失败：{e}", "", btn_hidden
        return

    # ---- 构造 DistillationConfig ----
    try:
        cfg = DistillationConfig(
            model=model,
            persona_id=pid,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            salience_threshold=salience,
            max_skills=max_skills,
            max_preset_dialogues=max_dialogues,
            default_error_reply=error_reply,
            workdir=workdir,
            debug=debug,
        )
    except Exception as e:  # noqa: BLE001
        yield log_box + f"\n❌ 配置错误：{e}", "", btn_hidden
        return

    # ---- 构造 LLM ----
    log_box += f"🚀 构造 LLM (model={model})...\n"
    yield log_box, "🚀 构造 LLM 中...", btn_hidden
    try:
        llm = build_model(cfg)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        yield log_box + f"\n❌ LLM 构造失败：{e}\n\n{tb}", "", btn_hidden
        return

    # ---- 提前安装 _LogHandler：覆盖 Phase 1/2/3 全程 ----
    # 这样 oc_writer / interview / PersonaDistiller 内部的 logger.info 都会进 handler，
    # 进而流式进 WebUI 日志框（不再只显示阶段级 start/完成 标记）。
    # 通过 Python logging 的 propagate 机制，子 logger
    # (persona_distillation.intake.oc_writer / .interview) 的记录会冒泡到本 handler。
    handler = _LogHandler()
    handler.setLevel(logging.DEBUG if debug else logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")
    )
    pd_logger = logging.getLogger("persona_distillation")
    pd_logger.addHandler(handler)
    prev_level = pd_logger.level
    pd_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # 已合并进 log_box 的 handler.records 索引；每次 yield 前 drain 新增记录，避免重复渲染
    log_cursor = 0

    def _drain_new_logs() -> str:
        nonlocal log_cursor
        new = "\n".join(handler.records[log_cursor:])
        log_cursor = len(handler.records)
        return new

    try:
        # ==========================================================
        # Phase 1：骨架生成（独白 / 对话 / 事件 / 回忆）
        # ==========================================================
        log_box += (
            "\n" + "=" * 50
            + "\n[Phase 1 骨架] 开始生成 4 类文本...\n"
            + "=" * 50 + "\n"
        )
        yield log_box, "📝 [Phase 1 骨架] 生成独白/对话/事件/回忆中...", btn_hidden

        try:
            phase1 = generate_oc_corpus(setting, workdir, pid, llm)
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            new_logs = _drain_new_logs()
            yield (
                log_box + (new_logs + "\n" if new_logs else "")
                + f"\n❌ [Phase 1 骨架] 失败：{e}\n\n{tb}",
                "",
                btn_hidden,
            )
            return

        # 合并 Phase 1 期间 oc_writer 内部逐类 logger.info 进度
        new_logs = _drain_new_logs()
        if new_logs:
            log_box += new_logs + "\n"
        for key, count in phase1["word_counts"].items():
            log_box += f"  ✅ {key}: {count} 字\n"
        log_box += f"[Phase 1 骨架] 完成 → {phase1['corpus_dir']}\n"
        yield log_box, "✅ [Phase 1 骨架] 完成，进入 Phase 2...", btn_hidden

        # ==========================================================
        # Phase 2：血肉访谈（run_interview 同步，先 yield 进度后展示结果）
        # ==========================================================
        rounds_int = max(1, int(n_rounds) if n_rounds else 8)
        log_box += (
            "\n" + "=" * 50
            + f"\n[Phase 2 血肉] 开始访谈 ({rounds_int} 轮)...\n"
            + "=" * 50 + "\n"
        )
        yield log_box, f"🎤 [Phase 2 血肉] 访谈中（{rounds_int} 轮，同步执行）...", btn_hidden

        try:
            phase2 = run_interview(setting, rounds_int, workdir, pid, llm)
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            new_logs = _drain_new_logs()
            yield (
                log_box + (new_logs + "\n" if new_logs else "")
                + f"\n❌ [Phase 2 血肉] 失败：{e}\n\n{tb}",
                "",
                btn_hidden,
            )
            return

        # 合并 Phase 2 期间 interview 内部逐轮 logger.info 进度
        new_logs = _drain_new_logs()
        if new_logs:
            log_box += new_logs + "\n"
        log_box += f"[Phase 2 血肉] 完成 → {phase2['path']}（{phase2['rounds']} 轮）\n"
        yield log_box, "✅ [Phase 2 血肉] 完成，进入 Phase 3 蒸馏...", btn_hidden

        # ==========================================================
        # Phase 3：蒸馏（骨架 + 血肉合并，复用同一份 _LogHandler）
        # ==========================================================
        log_box += (
            "\n" + "=" * 50
            + "\n[Phase 3 蒸馏] 启动 PersonaDistiller...\n"
            + "=" * 50 + "\n"
        )
        yield log_box, "🔬 [Phase 3 蒸馏] 蒸馏骨架+血肉中...", btn_hidden

        # 蒸馏输入是 <workdir>/<persona_id>/ 目录（含 oc_corpus/ + interview.md）
        distill_input = str(Path(workdir) / pid)

        try:
            distiller = PersonaDistiller(cfg)
            result = distiller.distill(distill_input, output_dir=output_dir)

            # 合并 Phase 3 期间 PersonaDistiller 内部 logger.info + 摘要 + 显示联动按钮
            new_logs = _drain_new_logs()
            if new_logs:
                log_box += new_logs + "\n"
            summary = _format_result_summary(result, Path(output_dir))
            log_box += "\n✅ OC 共创蒸馏完成！"
            yield log_box, summary, gr.update(visible=True)
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            new_logs = _drain_new_logs()
            if new_logs:
                log_box += new_logs + "\n"
            yield (
                log_box + f"\n\n❌ [Phase 3 蒸馏] 失败：{e}\n\n{tb}",
                "",
                btn_hidden,
            )
    finally:
        # 统一清理 handler，覆盖 Phase 1/2/3 全程，异常路径也保证卸载
        pd_logger.removeHandler(handler)
        pd_logger.setLevel(prev_level)


# ===========================================================================
# Tab 2：产物浏览面板
# ===========================================================================
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


def _on_load_output(out_dir: str, base: str, result_holder: dict[str, Any]):
    """点击「加载」后填充所有产物浏览组件，并把 result 缓存到 holder。"""
    if not out_dir:
        # 列出可选项让用户先选
        opts = _list_persona_dirs(base)
        value = opts[0] if opts else None
        return (
            gr.update(choices=opts, value=value),
            "请选择一个产物目录后点「加载」",
            "",
            "",
            gr.update(choices=[], value=None),
            [],
            "未加载",
        )

    result = _load_result(out_dir)
    result_holder["result"] = result  # 缓存供 skill 切换时复用
    if result is None:
        return (
            gr.update(),
            f"❌ 找不到 `{out_dir}/distillation_result.json`，无法加载。",
            "",
            "",
            gr.update(choices=[], value=None),
            [],
            "未加载",
        )

    card = result.persona_card
    card_md = _render_persona_card_md(card)
    dna_md = _render_dna_md(card)
    skill_choices = [s.name for s in result.skills] or ["(无 skill)"]
    first_skill_md = (
        _render_skill_md(result.skills[0], out_dir) if result.skills else "(无 skill)"
    )
    dialogues = _render_dialogues(result.preset_dialogues)
    status = (f"✅ 加载成功：persona_id=`{card.persona_id}`，"
              f"skills={len(result.skills)}，对话={len(result.preset_dialogues)}")
    return (
        gr.update(),
        card_md,
        dna_md,
        first_skill_md,
        gr.update(choices=skill_choices, value=skill_choices[0]),
        dialogues,
        status,
    )


def _on_select_skill(skill_name: str, out_dir: str, result_holder: dict[str, Any]) -> str:
    """切换选中的 skill 时返回对应 SKILL.md。"""
    if not skill_name or skill_name == "(无 skill)":
        return "(无 skill)"
    result = result_holder.get("result")
    if result is None:
        # 重新加载
        result = _load_result(out_dir)
        result_holder["result"] = result
    if result is None:
        return f"❌ 无法重新加载结果目录：{out_dir}"
    for sk in result.skills:
        if sk.name == skill_name:
            return _render_skill_md(sk, out_dir)
    return f"❌ 找不到 skill: {skill_name}"


def _on_load_output_with_jump(
    out_dir: str, base: str, result_holder: dict[str, Any]
) -> tuple:
    """``_on_load_output`` 的联动增强版：加载产物后顺带控制反向联动按钮可见性。

    返回 8 项（``_on_load_output`` 的 7 项 + 反向联动按钮可见性更新）：
    加载成功且 result 非空时显示「🔄 用相同语料在蒸馏 Tab 重跑」按钮。
    """
    load_result = _on_load_output(out_dir, base, result_holder)
    show_btn = bool(out_dir) and result_holder.get("result") is not None
    return (*load_result, gr.update(visible=show_btn))


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
# Tab 3：主理人 Agent 对话
# ===========================================================================
def _build_agent_holder():
    """会话级状态：缓存的 agent + 配置。"""
    return {"agent": None, "cfg": None}


def _on_chat_init(
    workdir: str,
    model: str,
    offline: bool,
    chunk_size: int,
    chunk_overlap: int,
    profile_max_entries: int,
    no_progress: bool,
    debug: bool,
    holder: dict[str, Any],
) -> str:
    """初始化主理人 Agent。holder 是闭包变量，不可序列化字段直接挂进去。"""
    from persona_distillation.agents import build_intake_orchestrator

    cfg = DistillationConfig(
        model=model,
        workdir=workdir or "./intake_workdir",
        intake_chunk_size=chunk_size,
        intake_chunk_overlap=chunk_overlap,
        profile_max_entries=profile_max_entries,
        offline=offline,
        debug=debug,
        show_progress=not no_progress,
    )
    try:
        agent = build_intake_orchestrator(cfg)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        return f"❌ Agent 启动失败：{e}\n\n{tb}"

    holder["agent"] = agent
    holder["cfg"] = cfg
    workdir_resolved = Path(cfg.workdir).resolve()
    return (f"✅ 主理人 Agent 已就绪\n"
            f"- 模型: `{model}`\n"
            f"- 工作目录: `{workdir_resolved}`\n"
            f"- 离线: {offline}\n"
            f"- 模式: 5 步预处理 + 蒸馏闭环\n\n"
            f"接下来在对话框里说一句话开始，例如：\n"
            f"> 「先摄入 ./examples/sample_corpus」\n"
            f"> 「列出已识别的人物」\n"
            f"> 「蒸馏第 1 个」")


def _on_chat_send(message: str, history: list, holder: dict[str, Any]) -> tuple[str, list, str]:
    """发送一条消息给主理人 Agent。"""
    if not message.strip():
        return "", history, ""
    agent = holder.get("agent")
    if agent is None:
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "❌ Agent 尚未初始化，请先在下方点「启动 / 重置 Agent」。"},
        ]
        return "", history, ""

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": message}]}
        )
        messages = result.get("messages") or []
        content = ""
        if messages:
            raw = getattr(messages[-1], "content", "") or ""
            if isinstance(raw, list):
                content = "".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in raw
                )
            else:
                content = str(raw)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        content = f"❌ 调用出错：{e}\n\n{tb}"

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": content},
    ]
    return "", history, ""


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


# ===========================================================================
# 构建 Blocks
# ===========================================================================
def build_ui(
    *,
    default_input: str = "./examples/sample_corpus",
    default_output: str = "./out",
    default_workdir: str = "./intake_workdir",
    default_model: str = "minimax:MiniMax-M3",
    share: bool = False,
) -> "gr.Blocks":
    """构建 Gradio Blocks 实例。"""
    # 用于在 Tab2/Tab3 之间共享会话状态
    result_holder: dict[str, Any] = {"result": None}
    agent_holder = _build_agent_holder()

    with _blocks(
        title="人格蒸馏 · 调试 WebUI",
        theme=gr.themes.Soft(),
        css=_CUSTOM_CSS,
    ) as demo:
        gr.Markdown(_HEADER_MD)

        with gr.Tabs() as tabs:
            # -----------------------------------------------------------------
            # Tab 1: 蒸馏
            # -----------------------------------------------------------------
            with gr.Tab("🔥 蒸馏", id="distill"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 参数")
                        in_input = gr.Textbox(
                            label="语料路径 (文件/目录)",
                            value=default_input,
                            placeholder="./examples/sample_corpus",
                        )
                        in_output = gr.Textbox(
                            label="输出目录",
                            value=default_output,
                            placeholder="./out",
                        )
                        in_model = gr.Textbox(label="模型", value=default_model)
                        in_persona_id = gr.Textbox(
                            label="人格ID (留空则 LLM 推断)",
                            value="",
                            placeholder="arakawa_sensei",
                        )
                        with gr.Accordion("高级参数", open=False):
                            in_chunk_size = gr.Slider(
                                400, 4000, value=1800, step=100, label="分块大小 (tokens)"
                            )
                            in_chunk_overlap = gr.Slider(
                                0, 800, value=200, step=50, label="分块重叠 (tokens)"
                            )
                            in_max_chunks = gr.Number(
                                value=0, label="单文件最大分块 (0=不限)", precision=0
                            )
                            in_salience = gr.Slider(
                                0.0, 1.0, value=0.35, step=0.05, label="显著度阈值"
                            )
                            in_max_skills = gr.Slider(
                                1, 12, value=6, step=1, label="Skills 数量上限"
                            )
                            in_max_dialogues = gr.Slider(
                                0, 20, value=8, step=1, label="预设对话数上限"
                            )
                            in_error_reply = gr.Textbox(
                                label="报错兜底文案",
                                value="（人格暂时失语，请稍后再试。）",
                            )
                            in_workdir = gr.Textbox(
                                label="中间产物目录 (留空则临时目录)",
                                value="",
                            )
                            in_debug = gr.Checkbox(label="DEBUG 日志", value=False)

                        btn_run = gr.Button("🚀 开始蒸馏", variant="primary")
                        gr.Markdown(
                            "> 点击后日志会实时刷新。耗时取决于语料长度与 LLM 速度。"
                        )

                    with gr.Column(scale=2):
                        out_status = gr.Markdown("待命")
                        out_log = gr.Textbox(
                            label="实时日志",
                            lines=20,
                            max_lines=40,
                            interactive=False,
                            autoscroll=True,
                        )
                        out_summary = gr.Markdown("")
                        btn_jump_distill = gr.Button(
                            "👉 立即查看完整产物", visible=False, variant="secondary"
                        )

                btn_run.click(
                    _run_distill,
                    inputs=[
                        in_input, in_output, in_model, in_persona_id,
                        in_chunk_size, in_chunk_overlap, in_max_chunks,
                        in_salience, in_max_skills, in_max_dialogues,
                        in_error_reply, in_workdir, in_debug,
                    ],
                    outputs=[out_log, out_summary, btn_jump_distill],
                )

            # -----------------------------------------------------------------
            # Tab 2: 产物浏览
            # -----------------------------------------------------------------
            with gr.Tab("📂 产物浏览", id="browse"):
                with gr.Row():
                    in_out_base = gr.Textbox(
                        label="产物根目录", value=default_output, scale=3
                    )
                    in_out_dir = gr.Dropdown(
                        label="选择产物子目录",
                        choices=_list_persona_dirs(default_output),
                        interactive=True,
                        scale=3,
                    )
                    btn_list = gr.Button("🔄 刷新列表", scale=1)
                    btn_load = gr.Button("📥 加载", variant="primary", scale=1)

                browse_status = gr.Markdown("未加载")

                with gr.Row():
                    with gr.Column(scale=1):
                        out_card_md = gr.Markdown("（未加载）", label="人格卡")
                    with gr.Column(scale=1):
                        out_dna_md = gr.Markdown("（未加载）", label="DNA 五层")

                gr.Markdown("### 🎯 Skills")
                with gr.Row():
                    in_skill_name = gr.Dropdown(
                        label="选择 Skill",
                        choices=[],
                        interactive=True,
                        scale=1,
                    )
                    out_skill_md = gr.Markdown("（未加载）", scale=2)

                gr.Markdown("### 💬 预设对话")
                out_dialogues = gr.Dataframe(
                    headers=["user", "assistant", "intent"],
                    datatype=["str", "str", "str"],
                    value=[],
                    interactive=False,
                    wrap=True,
                )

                # 反向联动按钮：加载产物后显示，点击切到蒸馏 Tab 重跑
                btn_jump_to_distill = gr.Button(
                    "🔄 用相同语料在蒸馏 Tab 重跑", visible=False, variant="secondary"
                )

                # 事件
                btn_list.click(
                    lambda base: gr.update(choices=_list_persona_dirs(base)),
                    inputs=in_out_base,
                    outputs=in_out_dir,
                )
                btn_load.click(
                    lambda out_dir, base: _on_load_output_with_jump(
                        out_dir, base, result_holder
                    ),
                    inputs=[in_out_dir, in_out_base],
                    outputs=[
                        in_out_dir, out_card_md, out_dna_md,
                        out_skill_md, in_skill_name, out_dialogues, browse_status,
                        btn_jump_to_distill,
                    ],
                )
                in_skill_name.change(
                    lambda name, base: _on_select_skill(name, base, result_holder),
                    inputs=[in_skill_name, in_out_dir],
                    outputs=out_skill_md,
                )

            # -----------------------------------------------------------------
            # Tab 3: 主理人 Agent 对话
            # -----------------------------------------------------------------
            with gr.Tab("💬 主理人 Agent", id="agent"):
                gr.Markdown(
                    "### 主理人 Agent\n"
                    "先点「启动 / 重置 Agent」构建一个会话级 Agent；"
                    "再在对话框里用自然语言驱动 5 步流程：摄入语料 → 列人物 → 选人物 → 档案 → 蒸馏。"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        chat_workdir = gr.Textbox(
                            label="工作目录", value=default_workdir
                        )
                        chat_model = gr.Textbox(label="模型", value=default_model)
                        with gr.Accordion("intake 参数", open=False):
                            chat_chunk_size = gr.Number(
                                value=1200, label="intake 分块大小", precision=0
                            )
                            chat_chunk_overlap = gr.Number(
                                value=120, label="intake 分块重叠", precision=0
                            )
                            chat_profile_max = gr.Number(
                                value=0, label="档案每类条目上限 (0=不限)", precision=0
                            )
                        chat_offline = gr.Checkbox(label="离线模式", value=False)
                        chat_no_progress = gr.Checkbox(label="关闭进度条", value=False)
                        chat_debug = gr.Checkbox(label="DEBUG 日志", value=False)
                        btn_init_agent = gr.Button(
                            "🔄 启动 / 重置 Agent", variant="primary"
                        )
                        chat_status = gr.Markdown("Agent 尚未初始化")

                    with gr.Column(scale=2):
                        chatbot = _chatbot(
                            label="对话",
                            type="messages",
                            height=520,
                            placeholder="先点左侧「启动 / 重置 Agent」，再在这里说话",
                        )
                        chat_input = gr.Textbox(
                            label="输入",
                            placeholder="例：先摄入 ./examples/sample_corpus",
                            lines=2,
                        )
                        with gr.Row():
                            btn_send = gr.Button("📤 发送", variant="primary")
                            btn_clear = gr.Button("🧹 清空对话")
                        # 联动按钮：Agent 蒸馏完成后点此跳转到产物浏览 Tab（仅刷新下拉）
                        btn_agent_jump = gr.Button(
                            "👉 查看最新产物", variant="secondary"
                        )

                btn_init_agent.click(
                    lambda wd, m, off, cs, co, pme, np_, dbg: _on_chat_init(
                        wd, m, off, cs, co, pme, np_, dbg, agent_holder
                    ),
                    inputs=[
                        chat_workdir, chat_model, chat_offline,
                        chat_chunk_size, chat_chunk_overlap,
                        chat_profile_max, chat_no_progress, chat_debug,
                    ],
                    outputs=chat_status,
                )
                btn_send.click(
                    lambda msg, hist: _on_chat_send(msg, hist, agent_holder),
                    inputs=[chat_input, chatbot],
                    outputs=[chat_input, chatbot, chat_status],
                )
                btn_clear.click(lambda: ([], ""), outputs=[chatbot, chat_input])

            # -----------------------------------------------------------------
            # Tab 4: OC 共创（三阶段串联：骨架 → 血肉 → 蒸馏）
            # -----------------------------------------------------------------
            with gr.Tab("✨ OC 共创", id="cocreate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### OC 设定")
                        oc_name = gr.Textbox(
                            label="姓名", placeholder="如：林深"
                        )
                        oc_age = gr.Textbox(
                            label="年龄", placeholder="如：25 / 永远17岁 / 未知"
                        )
                        oc_background = gr.Textbox(
                            label="背景", lines=2, placeholder="身世、职业、经历..."
                        )
                        oc_traits = gr.Textbox(
                            label="性格核心", lines=2,
                            placeholder="如：外冷内热、执拗、温柔",
                        )
                        oc_worldview = gr.Textbox(
                            label="世界观", lines=2,
                            placeholder="对世界/人生的核心看法",
                        )
                        oc_catchphrase = gr.Textbox(
                            label="口头禅", placeholder="如：嘛，再看看吧。"
                        )

                        gr.Markdown("### 访谈与蒸馏参数")
                        oc_rounds = gr.Number(
                            value=8, label="访谈轮数", precision=0
                        )
                        oc_persona_id = gr.Textbox(
                            label="人格ID (留空则从姓名派生)", value="",
                            placeholder="lin_shen",
                        )
                        oc_model = gr.Textbox(label="模型", value=default_model)
                        oc_output = gr.Textbox(
                            label="输出目录", value=default_output, placeholder="./out"
                        )
                        oc_workdir = gr.Textbox(
                            label="工作目录 (骨架/访谈落盘)", value=default_workdir
                        )
                        with gr.Accordion("高级蒸馏参数", open=False):
                            oc_chunk_size = gr.Slider(
                                400, 4000, value=1800, step=100,
                                label="分块大小 (tokens)",
                            )
                            oc_chunk_overlap = gr.Slider(
                                0, 800, value=200, step=50,
                                label="分块重叠 (tokens)",
                            )
                            oc_salience = gr.Slider(
                                0.0, 1.0, value=0.35, step=0.05, label="显著度阈值"
                            )
                            oc_max_skills = gr.Slider(
                                1, 12, value=6, step=1, label="Skills 数量上限"
                            )
                            oc_max_dialogues = gr.Slider(
                                0, 20, value=8, step=1, label="预设对话数上限"
                            )
                            oc_error_reply = gr.Textbox(
                                label="报错兜底文案",
                                value="（人格暂时失语，请稍后再试。）",
                            )
                            oc_debug = gr.Checkbox(label="DEBUG 日志", value=False)

                        btn_run_cocreate = gr.Button(
                            "🚀 开始 OC 共创蒸馏", variant="primary"
                        )
                        gr.Markdown(
                            "> 三阶段串联：骨架生成 → 血肉访谈 → 蒸馏。"
                            "耗时较长，请耐心等待日志流式刷新。"
                        )

                    with gr.Column(scale=2):
                        oc_status = gr.Markdown("待命")
                        oc_log = gr.Textbox(
                            label="实时日志",
                            lines=24,
                            max_lines=48,
                            interactive=False,
                            autoscroll=True,
                        )
                        oc_summary = gr.Markdown("")
                        btn_jump_cocreate = gr.Button(
                            "👉 立即查看完整产物", visible=False, variant="secondary"
                        )

                btn_run_cocreate.click(
                    _run_cocreate,
                    inputs=[
                        oc_name, oc_age, oc_background, oc_traits,
                        oc_worldview, oc_catchphrase,
                        oc_rounds, oc_persona_id, oc_model, oc_output, oc_workdir,
                        oc_chunk_size, oc_chunk_overlap, oc_salience,
                        oc_max_skills, oc_max_dialogues, oc_error_reply, oc_debug,
                    ],
                    outputs=[oc_log, oc_summary, btn_jump_cocreate],
                )

        # =================================================================
        # 跨 Tab 联动事件绑定（所有 Tab 组件均已定义，此时引用安全）
        # =================================================================
        # 蒸馏 Tab → 产物浏览 Tab（输出目录从 in_output 读取）
        btn_jump_distill.click(
            lambda out_dir, base: _jump_to_browse(out_dir, base, result_holder),
            inputs=[in_output, in_out_base],
            outputs=[
                tabs, in_out_dir, out_card_md, out_dna_md,
                out_skill_md, in_skill_name, out_dialogues, browse_status,
            ],
        )
        # OC 共创 Tab → 产物浏览 Tab（输出目录从 oc_output 读取）
        btn_jump_cocreate.click(
            lambda out_dir, base: _jump_to_browse(out_dir, base, result_holder),
            inputs=[oc_output, in_out_base],
            outputs=[
                tabs, in_out_dir, out_card_md, out_dna_md,
                out_skill_md, in_skill_name, out_dialogues, browse_status,
            ],
        )
        # 主理人 Agent Tab → 产物浏览 Tab（仅刷新下拉，不预选不加载）
        btn_agent_jump.click(
            lambda base: _jump_to_browse_refresh(base),
            inputs=[in_out_base],
            outputs=[
                tabs, in_out_dir, out_card_md, out_dna_md,
                out_skill_md, in_skill_name, out_dialogues, browse_status,
            ],
        )
        # 产物浏览 Tab → 蒸馏 Tab（反向联动：切到蒸馏 Tab 重跑）
        btn_jump_to_distill.click(
            lambda: _jump_to_distill(),
            inputs=None,
            outputs=[tabs],
        )

    return demo


# ===========================================================================
# 入口
# ===========================================================================
def launch(
    *,
    host: str = "0.0.0.0",
    port: int = 7860,
    share: bool = False,
    default_input: str = "./examples/sample_corpus",
    default_output: str = "./out",
    default_workdir: str = "./intake_workdir",
    default_model: str = "minimax:MiniMax-M3",
) -> None:
    """启动 WebUI。"""
    demo = build_ui(
        default_input=default_input,
        default_output=default_output,
        default_workdir=default_workdir,
        default_model=default_model,
        share=share,
    )
    logger.info("启动 WebUI: host=%s port=%d share=%s", host, port, share)
    # Gradio 6.x 把 theme/css 从 Blocks 移到 launch()
    launch_kwargs: dict[str, Any] = {
        "server_name": host,
        "server_port": port,
        "share": share,
        "show_error": True,
    }
    if _GRADIO_MAJOR >= 6:
        launch_kwargs["theme"] = gr.themes.Soft()
        launch_kwargs["css"] = _CUSTOM_CSS
    demo.launch(**launch_kwargs)
