"""Tab 3：主理人 Agent 对话面板。

先点「启动 / 重置 Agent」构建一个会话级 Agent；再在对话框里用自然语言驱动
5 步流程：摄入语料 → 列人物 → 选人物 → 档案 → 蒸馏。

Agent 蒸馏完成后可点 ``btn_agent_jump`` 跳转到产物浏览 Tab（仅刷新下拉，
让用户自己选产物目录——因为 Agent 流程无法精确解析产物路径）。
跨 Tab 联动在 ``__init__.build_ui`` 里组装。
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import gradio as gr

from persona_distillation.config import DistillationConfig
from persona_distillation.webui.state import _chatbot


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


def build_tab_agent(
    *,
    default_workdir: str,
    default_model: str,
    agent_holder: dict[str, Any],
) -> dict[str, Any]:
    """构建主理人 Agent Tab 的 UI 组件 + 内部事件绑定。

    返回组件 dict，供 ``__init__.build_ui`` 组装跨 Tab 联动：
    - ``btn_agent_jump``：跳转到产物浏览 Tab 的触发器
    """
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

    return {
        "btn_agent_jump": btn_agent_jump,
    }
