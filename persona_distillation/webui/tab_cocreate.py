"""Tab 4：OC 共创面板。

捏造 OC 设定 → 三阶段串联（骨架生成 → 血肉访谈 → 蒸馏）。

OC 共创蒸馏成功后通过 ``btn_jump_cocreate`` 一键跳转到产物浏览 Tab
（跨 Tab 联动在 ``__init__.build_ui`` 里组装）。
"""
from __future__ import annotations

import logging
import traceback
from collections.abc import Generator
from pathlib import Path
from typing import Any

import gradio as gr

from persona_distillation.config import DistillationConfig
from persona_distillation.pipeline import PersonaDistiller
from persona_distillation.webui.state import (
    _LogHandler,
    _format_result_summary,
)


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


def build_tab_cocreate(
    *,
    default_output: str,
    default_workdir: str,
    default_model: str,
) -> dict[str, Any]:
    """构建 OC 共创 Tab 的 UI 组件 + 内部事件绑定。

    返回组件 dict，供 ``__init__.build_ui`` 组装跨 Tab 联动：
    - ``oc_output``：输出目录输入框（联动跳转的输入）
    - ``btn_jump_cocreate``：蒸馏完成后显示的「立即查看完整产物」按钮（联动触发器）
    """
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

    return {
        "oc_output": oc_output,
        "btn_jump_cocreate": btn_jump_cocreate,
    }
