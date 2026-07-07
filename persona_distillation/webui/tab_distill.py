"""Tab 1：蒸馏面板。

表单填参数 → 后台跑 ``PersonaDistiller.distill`` → 实时日志 → 完成摘要。
蒸馏成功后通过 ``btn_jump_distill`` 一键跳转到产物浏览 Tab（跨 Tab 联动
在 ``__init__.build_ui`` 里组装）。
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


def build_tab_distill(
    *,
    default_input: str,
    default_output: str,
    default_model: str,
) -> dict[str, Any]:
    """构建蒸馏 Tab 的 UI 组件 + 内部事件绑定。

    返回组件 dict，供 ``__init__.build_ui`` 组装跨 Tab 联动：
    - ``in_output``：输出目录输入框（联动跳转的输入）
    - ``btn_jump_distill``：蒸馏完成后显示的「立即查看完整产物」按钮（联动触发器）
    """
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

    return {
        "in_output": in_output,
        "btn_jump_distill": btn_jump_distill,
    }
