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

拆分自原 ``webui.py`` 单文件（Issue #9）：本 ``__init__.py`` 只保留
``build_ui`` 入口与 ``launch`` 启动器，4 个 Tab 的 UI 与事件分别落在
``tab_distill`` / ``tab_browse`` / ``tab_agent`` / ``tab_cocreate``，
共享状态与跨 Tab 联动函数集中在 :mod:`persona_distillation.webui.state`。
``from persona_distillation.webui import build_ui`` 仍可用。
"""
from __future__ import annotations

import logging
from typing import Any

try:
    import gradio as gr
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "WebUI 依赖 gradio，请先安装：pip install gradio>=4.44.0"
    ) from e

from persona_distillation.webui.state import (
    _CUSTOM_CSS,
    _GRADIO_MAJOR,
    _HEADER_MD,
    _blocks,
    _jump_to_browse,
    _jump_to_browse_refresh,
    _jump_to_distill,
    _refresh_eval_area,
    _reset_eval_area,
)
from persona_distillation.webui.tab_agent import build_tab_agent
from persona_distillation.webui.tab_browse import build_tab_browse
from persona_distillation.webui.tab_cocreate import build_tab_cocreate
from persona_distillation.webui.tab_distill import build_tab_distill

logger = logging.getLogger(__name__)


def build_ui(
    *,
    default_input: str = "./examples/sample_corpus",
    default_output: str = "./out",
    default_workdir: str = "./intake_workdir",
    default_model: str = "minimax:MiniMax-M3",
    share: bool = False,
) -> "gr.Blocks":
    """构建 Gradio Blocks 实例。"""
    # 用于在 Tab2/Tab3 之间共享会话状态（原 webui.py 的闭包变量，
    # 拆分后改为 build_ui 局部变量，通过参数传给 build_tab_* 与联动 lambda）
    result_holder: dict[str, Any] = {"result": None}
    agent_holder: dict[str, Any] = {"agent": None, "cfg": None}

    with _blocks(
        title="人格蒸馏 · 调试 WebUI",
        theme=gr.themes.Soft(),
        css=_CUSTOM_CSS,
    ) as demo:
        gr.Markdown(_HEADER_MD)

        with gr.Tabs() as tabs:
            # 4 个 Tab 各自构建 UI + 内部事件绑定，返回跨 Tab 联动需要的组件引用
            distill = build_tab_distill(
                default_input=default_input,
                default_output=default_output,
                default_model=default_model,
            )
            browse = build_tab_browse(
                default_output=default_output,
                default_model=default_model,
                result_holder=result_holder,
            )
            agent = build_tab_agent(
                default_workdir=default_workdir,
                default_model=default_model,
                agent_holder=agent_holder,
            )
            cocreate = build_tab_cocreate(
                default_output=default_output,
                default_workdir=default_workdir,
                default_model=default_model,
            )

        # =================================================================
        # 跨 Tab 联动事件绑定（所有 Tab 组件均已定义，此时引用安全）
        # =================================================================
        # 蒸馏 Tab → 产物浏览 Tab（输出目录从 in_output 读取）
        distill["btn_jump_distill"].click(
            lambda out_dir, base: _jump_to_browse(out_dir, base, result_holder),
            inputs=[distill["in_output"], browse["in_out_base"]],
            outputs=[
                tabs, browse["in_out_dir"], browse["out_card_md"],
                browse["out_dna_md"], browse["out_skill_md"],
                browse["in_skill_name"], browse["out_dialogues"],
                browse["browse_status"],
            ],
        ).then(
            # 跳转后顺带刷新评估区（in_out_dir 已被 _jump_to_browse 设为新产物路径）
            _refresh_eval_area,
            inputs=[browse["in_out_dir"]],
            outputs=[browse["eval_md"], browse["btn_eval_generate"], browse["btn_eval_rerun"]],
        )
        # OC 共创 Tab → 产物浏览 Tab（输出目录从 oc_output 读取）
        cocreate["btn_jump_cocreate"].click(
            lambda out_dir, base: _jump_to_browse(out_dir, base, result_holder),
            inputs=[cocreate["oc_output"], browse["in_out_base"]],
            outputs=[
                tabs, browse["in_out_dir"], browse["out_card_md"],
                browse["out_dna_md"], browse["out_skill_md"],
                browse["in_skill_name"], browse["out_dialogues"],
                browse["browse_status"],
            ],
        ).then(
            _refresh_eval_area,
            inputs=[browse["in_out_dir"]],
            outputs=[browse["eval_md"], browse["btn_eval_generate"], browse["btn_eval_rerun"]],
        )
        # 主理人 Agent Tab → 产物浏览 Tab（仅刷新下拉，不预选不加载）
        agent["btn_agent_jump"].click(
            lambda base: _jump_to_browse_refresh(base),
            inputs=[browse["in_out_base"]],
            outputs=[
                tabs, browse["in_out_dir"], browse["out_card_md"],
                browse["out_dna_md"], browse["out_skill_md"],
                browse["in_skill_name"], browse["out_dialogues"],
                browse["browse_status"],
            ],
        ).then(
            # 仅刷新下拉不预加载，评估区重置到"未加载"状态
            _reset_eval_area,
            inputs=None,
            outputs=[browse["eval_md"], browse["btn_eval_generate"], browse["btn_eval_rerun"]],
        )
        # 产物浏览 Tab → 蒸馏 Tab（反向联动：切到蒸馏 Tab 重跑）
        browse["btn_jump_to_distill"].click(
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
