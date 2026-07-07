"""Tab 2：产物浏览面板。

选 ``out/`` 子目录 → 渲染 persona_card / DNA 五层 / Skills / 预设对话；
含 LLM-as-judge 评估区（``_run_eval`` generator 流式刷新日志 + 报告）。

加载产物后会显示反向联动按钮（``btn_jump_to_distill``），点击切到蒸馏 Tab
重跑；正向联动（其它 Tab 跳到本 Tab）在 ``__init__.build_ui`` 里组装。
"""
from __future__ import annotations

import logging
import traceback
from collections.abc import Generator
from pathlib import Path
from typing import Any

import gradio as gr

from persona_distillation.config import DistillationConfig
from persona_distillation.schemas import (
    Distillate,
    DistillationResult,
)
from persona_distillation.webui.state import (
    _LogHandler,
    _list_persona_dirs,
    _load_result,
    _refresh_eval_area,
    _render_dialogues,
    _render_dna_md,
    _render_persona_card_md,
    _render_skill_md,
)


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
# 评估 generator：后台调 build_report + LLM，流式 yield 日志 + 评估区刷新
# ===========================================================================
def _run_eval(
    out_dir: str,
    model: str,
    offline: bool,
) -> Generator[tuple[str, str, Any, Any], None, None]:
    """后台跑蒸馏质量评估，流式 yield ``(日志, 评估区Markdown, 生成按钮可见性, 重新评估按钮可见性)``。

    流程：
    1. 加载 ``<out_dir>/distillation_result.json`` + ``distillates.jsonl``
    2. 安装 :class:`_LogHandler` 到 ``persona_distillation`` logger，捕获
       fidelity / identifiability 评估器内部的 ``logger.info`` 流式进 UI
    3. ``offline=False`` 时构造 LLM（minimax 走 ChatOpenAI，其它 provider 走
       ``init_chat_model``），失败则回退到离线模式（仅 coverage）
    4. 调 :func:`eval.report.build_report`，落盘 ``eval_report.json``
    5. 完成后调 :func:`_refresh_eval_area` 刷新评估区 + 按钮可见性

    评估失败（无 API key / LLM 调用异常）时不崩溃，按磁盘上 ``eval_report.json``
    的实际存在状态决定按钮可见性，让用户可重试。
    """
    # 评估运行期间隐藏两个按钮，防双击
    both_hidden = (gr.update(visible=False), gr.update(visible=False))
    log_box = ""

    if not out_dir:
        yield "❌ 错误：未选择产物目录", "", *both_hidden
        return

    # ---- 加载 distillation_result.json ----
    result_path = Path(out_dir) / "distillation_result.json"
    if not result_path.exists():
        yield f"❌ 找不到 `{result_path}`", "", *both_hidden
        return
    try:
        result = DistillationResult.load(result_path)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        yield f"❌ 加载 distillation_result.json 失败：{e}\n\n{tb}", "", *both_hidden
        return

    # ---- 加载 distillates.jsonl（fidelity/identifiability 用作"原语料"参考）----
    distillates: list[Distillate] = []
    distillates_path = Path(out_dir) / "distillates.jsonl"
    if distillates_path.exists():
        try:
            for line in distillates_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    distillates.append(Distillate.model_validate_json(line))
        except Exception as e:  # noqa: BLE001
            log_box += f"⚠️ 加载 distillates.jsonl 失败（继续评估）: {e}\n"

    log_box += (
        f"✅ 加载产物: persona_id=`{result.persona_card.persona_id}`, "
        f"skills={len(result.skills)}, distillates={len(distillates)}\n"
    )
    yield log_box + "\n🚀 启动评估中...", "", *both_hidden

    # ---- 安装 _LogHandler：捕获评估器内部的 logger.info ----
    handler = _LogHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")
    )
    pd_logger = logging.getLogger("persona_distillation")
    pd_logger.addHandler(handler)
    prev_level = pd_logger.level
    pd_logger.setLevel(logging.INFO)

    log_cursor = 0

    def _drain_new_logs() -> str:
        nonlocal log_cursor
        new = "\n".join(handler.records[log_cursor:])
        log_cursor = len(handler.records)
        return new

    try:
        # ---- 构造 LLM（除非 offline）----
        llm: Any = None
        if not offline:
            log_box += f"🚀 构造 LLM (model={model})...\n"
            yield log_box, "", *both_hidden
            from persona_distillation.agents import build_model

            # dry_run=True 跳过 DistillationConfig 的 API key 校验，
            # 让 build_model 自己抛 RuntimeError（我们 catch 后回退到离线）
            cfg = DistillationConfig(model=model, dry_run=True)
            try:
                m = build_model(cfg)
                # build_model 对 minimax 返回 BaseChatModel，对其它 provider 返回字符串
                # 字符串需走 init_chat_model 转成 BaseChatModel 才能给 eval 评估器用
                if isinstance(m, str):
                    from langchain.chat_models import init_chat_model
                    m = init_chat_model(m)  # type: ignore[arg-type]
                llm = m
            except Exception as e:  # noqa: BLE001
                log_box += (
                    f"⚠️ LLM 构造失败：{e}\n"
                    "→ 回退到离线模式（仅跑 coverage 评估）。\n"
                )
                llm = None
        else:
            log_box += "📴 离线模式：仅跑 coverage 评估，不调 LLM。\n"

        # ---- 调 build_report ----
        from persona_distillation.eval.report import build_report

        log_box += "📊 跑评估中（coverage / fidelity / identifiability）...\n"
        yield log_box, "", *both_hidden

        report = build_report(
            card=result.persona_card,
            skills=result.skills,
            distillates=distillates,
            llm=llm,
        )

        new_logs = _drain_new_logs()
        if new_logs:
            log_box += new_logs + "\n"

        # ---- 落盘 eval_report.json ----
        eval_path = Path(out_dir) / "eval_report.json"
        eval_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        log_box += (
            f"\n✅ 评估完成 → {eval_path}\n"
            f"   overall_score = {report.overall_score:.3f}\n"
        )

        # 完成后按磁盘状态刷新评估区 + 按钮可见性
        final_md, final_gen_vis, final_rerun_vis = _refresh_eval_area(out_dir)
        yield log_box, final_md, final_gen_vis, final_rerun_vis
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        new_logs = _drain_new_logs()
        if new_logs:
            log_box += new_logs + "\n"
        log_box += f"\n❌ 评估失败：{e}\n\n{tb}"
        # 失败时按磁盘上 eval_report.json 的实际状态决定按钮可见性
        final_md, final_gen_vis, final_rerun_vis = _refresh_eval_area(out_dir)
        yield log_box, final_md, final_gen_vis, final_rerun_vis
    finally:
        pd_logger.removeHandler(handler)
        pd_logger.setLevel(prev_level)


def build_tab_browse(
    *,
    default_output: str,
    default_model: str,
    result_holder: dict[str, Any],
) -> dict[str, Any]:
    """构建产物浏览 Tab 的 UI 组件 + 内部事件绑定。

    返回组件 dict，供 ``__init__.build_ui`` 组装跨 Tab 联动（正向跳转的
    outputs + 反向跳转的触发器 ``btn_jump_to_distill`` + 评估区刷新目标）。
    """
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

        # ---- 📊 评估区（Accordion 默认折叠，不参与 _on_load_output 返回值）----
        # 加载产物后通过 btn_load.click().then(_refresh_eval_area) 单独刷新
        with gr.Accordion("📊 评估", open=False):
            eval_md = gr.Markdown("_（未加载产物）_")
            with gr.Row():
                eval_model = gr.Textbox(
                    label="评估模型",
                    value=default_model,
                    scale=2,
                    interactive=True,
                )
                eval_offline = gr.Checkbox(
                    label="离线模式（仅 coverage，不调 LLM）",
                    value=False,
                    scale=1,
                )
            eval_log = gr.Textbox(
                label="评估日志",
                lines=8,
                max_lines=16,
                interactive=False,
                autoscroll=True,
            )
            with gr.Row():
                # 不存在 eval_report.json 时显示
                btn_eval_generate = gr.Button(
                    "📊 生成评估", variant="primary", visible=False
                )
                # 存在 eval_report.json 时显示（覆盖重生成）
                btn_eval_rerun = gr.Button(
                    "🔄 重新评估", variant="secondary", visible=False
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
        ).then(
            # 加载产物后单独刷新评估区（不污染 _on_load_output 的返回值结构）
            _refresh_eval_area,
            inputs=[in_out_dir],
            outputs=[eval_md, btn_eval_generate, btn_eval_rerun],
        )
        in_skill_name.change(
            lambda name, base: _on_select_skill(name, base, result_holder),
            inputs=[in_skill_name, in_out_dir],
            outputs=out_skill_md,
        )
        # 生成评估 / 重新评估 共用 _run_eval generator
        btn_eval_generate.click(
            _run_eval,
            inputs=[in_out_dir, eval_model, eval_offline],
            outputs=[eval_log, eval_md, btn_eval_generate, btn_eval_rerun],
        )
        btn_eval_rerun.click(
            _run_eval,
            inputs=[in_out_dir, eval_model, eval_offline],
            outputs=[eval_log, eval_md, btn_eval_generate, btn_eval_rerun],
        )

    return {
        "in_out_base": in_out_base,
        "in_out_dir": in_out_dir,
        "out_card_md": out_card_md,
        "out_dna_md": out_dna_md,
        "out_skill_md": out_skill_md,
        "in_skill_name": in_skill_name,
        "out_dialogues": out_dialogues,
        "browse_status": browse_status,
        "btn_jump_to_distill": btn_jump_to_distill,
        "eval_md": eval_md,
        "btn_eval_generate": btn_eval_generate,
        "btn_eval_rerun": btn_eval_rerun,
    }
