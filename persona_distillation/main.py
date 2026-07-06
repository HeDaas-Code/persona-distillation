"""命令行入口。

示例::

    export MINIMAX_API_KEY=sk-...
    python -m persona_distillation.main distill ./corpus ./out \\
        --model minimax:MiniMax-M3 --persona-id arakawa_sensei

    python -m persona_distillation.main inspect ./out
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

from persona_distillation.config import DistillationConfig
from persona_distillation.pipeline import PersonaDistiller
from persona_distillation.schemas import DistillationResult


def _configure_logging(debug: bool = False) -> None:
    """配置日志：默认 INFO，--debug 时 DEBUG。

    persona_distillation 的 logger 始终至少 INFO，方便排查 NER/蒸馏问题。
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=logging.WARNING,  # 第三方库保持 WARNING
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    logging.getLogger("persona_distillation").setLevel(level)


def _cmd_distill(args: argparse.Namespace) -> int:
    cfg = DistillationConfig(
        model=args.model,
        persona_id=args.persona_id or "",
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_chunks_per_file=args.max_chunks_per_file,
        salience_threshold=args.salience_threshold,
        max_skills=args.max_skills,
        max_preset_dialogues=args.max_dialogues,
        default_error_reply=args.error_reply,
        workdir=args.workdir or "",
        debug=args.debug,
    )
    distiller = PersonaDistiller(cfg)
    result = distiller.distill(args.input, output_dir=args.output)
    _print_summary(result, Path(args.output))
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    p = Path(args.path)
    result_path = p / "distillation_result.json" if p.is_dir() else p
    if not result_path.exists():
        logger.error("找不到结果文件: %s", result_path)
        return 1
    result = DistillationResult.load(result_path)
    _print_summary(result, result_path.parent if p.is_dir() else p)
    return 0


def _print_summary(result: DistillationResult, out: Path) -> None:
    card = result.persona_card
    print("\n" + "=" * 60)
    print(f"人格ID        : {card.persona_id}")
    print(f"显示名        : {card.display_name or '(未设置)'}")
    print(f"标签          : {', '.join(card.tags) or '(无)'}")
    print(f"报错回复      : {card.error_reply}")
    print(f"系统提示词长度: {len(card.system_prompt)} 字符")
    print(f"Skills 数     : {len(result.skills)}  → {[s.name for s in result.skills]}")
    print(f"预设对话数    : {len(result.preset_dialogues)}")
    print(f"蒸馏液块数    : {len(result.distillates)}")
    print(f"元信息        : {result.metadata}")
    print("=" * 60)
    print(f"产物目录      : {out}")
    print("  - persona_card.json     角色卡（左/右面板字段）")
    print("  - persona_card.md       可读版系统提示词")
    print("  - preset_dialogues.json 预设对话")
    print("  - distillates.jsonl     分馏液（可审计）")
    print("  - skills/<name>/SKILL.md 人格 Skills")


def _cmd_chat(args: argparse.Namespace) -> int:
    """启动主理人 Agent（交互式 REPL）。"""
    from persona_distillation.agents import build_intake_orchestrator

    cfg = DistillationConfig(
        model=args.model,
        workdir=args.workdir,
        embedding_model=args.embedding_model,
        rerank_model=args.rerank_model,
        intake_chunk_size=args.chunk_size,
        intake_chunk_overlap=args.chunk_overlap,
        offline=args.offline,
        debug=args.debug,
        show_progress=not getattr(args, "no_progress", False),
        profile_max_entries=args.profile_max_entries,
    )
    try:
        agent = build_intake_orchestrator(cfg)
    except Exception as e:
        logger.error("主理人 Agent 启动失败: %s", e, exc_info=True)
        return 1

    workdir = Path(args.workdir).resolve()
    print("=" * 60)
    print(f"  人格蒸馏 · 主理人 Agent  已就绪")
    print(f"  模型: {args.model}    工作目录: {workdir}")
    print(f"  退出: 输入 退出 / quit / exit")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n[你] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[主理人] 再见。")
            return 0
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "退出", "切回正常", "不用扮演了"):
            print("[主理人] 再见。")
            return 0
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]}
            )
        except Exception as e:
            logger.error("主理人 Agent 调用出错: %s", e, exc_info=True)
            continue
        messages = result.get("messages") or []
        if messages:
            content = getattr(messages[-1], "content", "") or ""
            if isinstance(content, list):
                content = "".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in content
                )
            print(f"\n[主理人] {content}")
    # unreachable
    return 0


def _cmd_webui(args: argparse.Namespace) -> int:
    """启动 Gradio WebUI（调试面板）。"""
    from persona_distillation.webui import launch

    launch(
        host=args.host,
        port=args.port,
        share=args.share,
        default_input=args.default_input,
        default_output=args.default_output,
        default_workdir=args.default_workdir,
        default_model=args.model,
    )
    return 0


# ---------------------------------------------------------------------------
# cocreate 子命令：OC 共创三阶段（骨架 → 血肉访谈 → 蒸馏）
# ---------------------------------------------------------------------------
def _slugify_persona_id(name: str) -> str:
    """从 name 生成 ASCII-only persona_id；无法生成时返回空串。

    规则：小写化 → 空白转下划线 → 去除非 [a-z0-9_] 字符 → 去首尾下划线。
    若 name 含非 ASCII 字符（如中文/日文），slugify 后会为空，调用方应提示
    用户用 ``--persona-id`` 显式指定。
    """
    n = name.strip().lower()
    n = re.sub(r"\s+", "_", n)
    n = re.sub(r"[^a-z0-9_]+", "", n)
    n = n.strip("_")
    return n


def _prompt_oc_setting(args: argparse.Namespace):
    """交互式引导：依次提示 OC 设定字段 + 访谈轮数，返回设定摘要让用户确认。

    Returns:
        ``(OCSetting, rounds, persona_id)`` 或 ``None``（用户取消 / CI 无 TTY）。
    """
    from persona_distillation.intake.oc_writer import OCSetting

    try:
        while True:
            print("\n=== OC 共创蒸馏 · 交互式引导 ===")
            name = input("姓名: ").strip()
            age = input("年龄（如 25 / 未知 / 三十出头）: ").strip()
            background = input("背景（一句话）: ").strip()
            traits = input("性格核心（一句话）: ").strip()
            worldview = input("世界观（一句话）: ").strip()
            catchphrase = input("口头禅: ").strip()
            rounds_str = input("访谈轮数（默认 8）: ").strip()
            try:
                rounds = int(rounds_str) if rounds_str else 8
            except ValueError:
                print("[警告] 访谈轮数不是数字，已回退为 8。")
                rounds = 8

            print("\n=== 设定摘要 ===")
            print(f"姓名      : {name}")
            print(f"年龄      : {age}")
            print(f"背景      : {background}")
            print(f"性格核心  : {traits}")
            print(f"世界观    : {worldview}")
            print(f"口头禅    : {catchphrase}")
            print(f"访谈轮数  : {rounds}")
            confirm = input("\n确认以上设定？(y=继续 / n=重新输入): ").strip().lower()
            if confirm == "y":
                break

        persona_id = _slugify_persona_id(name)
        if not persona_id:
            print(f"无法从姓名 '{name}' 自动生成 ASCII persona_id。")
            persona_id = input(
                "请手动输入 persona_id（小写字母/数字/连字符，回车取消）: "
            ).strip()
            if not persona_id:
                print("未提供 persona_id，已取消。")
                return None
        return OCSetting(
            name=name,
            age=age,
            background=background,
            traits=traits,
            worldview=worldview,
            catchphrase=catchphrase,
        ), rounds, persona_id
    except EOFError:
        # CI 环境无 TTY，input() 立即 EOF
        print("\n[错误] 当前环境无 TTY，无法进入交互式引导。")
        print(
            "请改用非交互式参数：--name / --age / --background / --traits / "
            "--worldview / --catchphrase / --rounds / --output / --persona-id"
        )
        return None


def _cmd_cocreate(args: argparse.Namespace) -> int:
    """OC 共创蒸馏：一键走完 Phase 1 骨架 → Phase 2 访谈 → Phase 3 蒸馏。"""
    from persona_distillation.agents import build_model
    from persona_distillation.intake.interview import run_interview
    from persona_distillation.intake.oc_writer import (
        OCSetting,
        generate_oc_corpus,
    )

    # ---- 解析 OC 设定（交互式 or 非交互式） ----
    if args.name:
        # 非交互式：--name 提供即跳过引导
        setting = OCSetting(
            name=args.name,
            age=args.age,
            background=args.background,
            traits=args.traits,
            worldview=args.worldview,
            catchphrase=args.catchphrase,
        )
        rounds = args.rounds
        persona_id = args.persona_id or _slugify_persona_id(args.name)
        if not persona_id:
            logger.error(
                "无法从姓名 %r 自动生成 ASCII persona_id（含非 ASCII 字符），"
                "请用 --persona-id 显式指定。",
                args.name,
            )
            return 1
    else:
        # 交互式引导
        result = _prompt_oc_setting(args)
        if result is None:
            return 1
        setting, rounds, persona_id = result

    # ---- 构造 cfg + llm ----
    # cocreate 默认 workdir 用 ./oc_workdir，便于用户检视中间产物（oc_corpus/ + interview.md）
    workdir = args.workdir or "./oc_workdir"
    Path(workdir).mkdir(parents=True, exist_ok=True)
    cfg = DistillationConfig(
        model=args.model,
        persona_id=persona_id,
        workdir=workdir,
        debug=args.debug,
    )
    try:
        llm = build_model(cfg)
    except Exception as e:
        logger.error("构造 LLM 失败: %s", e, exc_info=True)
        return 1

    persona_dir = Path(workdir) / persona_id

    # ---- Phase 1：骨架生成 ----
    print(f"\n[Phase 1 骨架] 生成 4 类骨架文本 (persona={persona_id}) ...")
    try:
        phase1 = generate_oc_corpus(setting, workdir, persona_id, llm)
    except Exception as e:
        logger.error("[Phase 1 骨架] 失败: %s", e, exc_info=True)
        return 1
    for key, wc in phase1["word_counts"].items():
        print(f"  - {key:<10s} {wc:>5d} 字  →  {phase1['paths'][key]}")

    # ---- Phase 2：血肉访谈 ----
    print(f"\n[Phase 2 血肉] 进行 {rounds} 轮访谈 (persona={persona_id}) ...")
    try:
        phase2 = run_interview(setting, rounds, workdir, persona_id, llm)
    except Exception as e:
        logger.error("[Phase 2 血肉] 失败: %s", e, exc_info=True)
        return 1
    print(f"  - {phase2['rounds']} 轮访谈记录  →  {phase2['path']}")

    # ---- Phase 3：蒸馏（输入是 <workdir>/<persona_id>/ 目录，含 oc_corpus/ + interview.md） ----
    print(
        f"\n[Phase 3 蒸馏] 基于 {persona_dir}/ 下的 oc_corpus/ + interview.md 执行人格蒸馏 ..."
    )
    try:
        distiller = PersonaDistiller(cfg)
        result = distiller.distill(
            input_path=persona_dir,
            output_dir=args.output,
        )
    except Exception as e:
        logger.error("[Phase 3 蒸馏] 失败: %s", e, exc_info=True)
        return 1
    _print_summary(result, Path(args.output))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="persona-distillation",
        description="基于 DeepAgents 的人格蒸馏框架",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("distill", help="对语料执行人格蒸馏")
    d.add_argument("input", help="语料文件或目录")
    d.add_argument("output", help="产物输出目录")
    d.add_argument("--model", default="minimax:MiniMax-M3", help="provider:model 字符串（默认 minimax:MiniMax-M3）")
    d.add_argument("--persona-id", default="", help="指定人格ID")
    d.add_argument("--chunk-size", type=int, default=1800)
    d.add_argument("--chunk-overlap", type=int, default=200)
    d.add_argument("--max-chunks-per-file", type=int, default=0, help="0=不限")
    d.add_argument("--salience-threshold", type=float, default=0.35)
    d.add_argument("--max-skills", type=int, default=6)
    d.add_argument("--max-dialogues", type=int, default=8)
    d.add_argument("--error-reply", default="（人格暂时失语，请稍后再试。）")
    d.add_argument("--workdir", default="", help="中间产物目录")
    d.add_argument("--debug", action="store_true")
    d.set_defaults(func=_cmd_distill)

    ins = sub.add_parser("inspect", help="查看已蒸馏结果")
    ins.add_argument("path", help="结果目录或 distillation_result.json")
    ins.set_defaults(func=_cmd_inspect)

    chat = sub.add_parser("chat", help="启动主理人 Agent（交互式蒸馏）")
    chat.add_argument("--model", default="minimax:MiniMax-M3", help="模型字符串")
    chat.add_argument("--workdir", default="./intake_workdir", help="工作目录")
    chat.add_argument("--embedding-model", default="BAAI/bge-m3", help="嵌入模型")
    chat.add_argument("--rerank-model", default="BAAI/bge-reranker-base", help="重排序模型")
    chat.add_argument("--chunk-size", type=int, default=1200, help="intake 分块大小")
    chat.add_argument("--chunk-overlap", type=int, default=120, help="intake 分块重叠")
    chat.add_argument("--offline", action="store_true", help="离线模式（用伪 embedding）")
    chat.add_argument("--no-progress", action="store_true", help="关闭分块解析进度条")
    chat.add_argument("--profile-max-entries", type=int, default=0,
                      help="蒸馏语料每类 excerpts 上限（0=不限，保留全部；默认 0）")
    chat.add_argument("--debug", action="store_true")
    chat.set_defaults(func=_cmd_chat)

    web = sub.add_parser("webui", help="启动 Gradio 调试 WebUI")
    web.add_argument("--host", default="0.0.0.0", help="监听地址")
    web.add_argument("--port", type=int, default=7860, help="监听端口")
    web.add_argument("--share", action="store_true", help="启用 Gradio 公网分享链接")
    web.add_argument("--model", default="minimax:MiniMax-M3", help="默认模型")
    web.add_argument("--default-input", default="./examples/sample_corpus",
                     help="蒸馏 Tab 的默认语料路径")
    web.add_argument("--default-output", default="./out",
                     help="蒸馏 Tab 与产物浏览 Tab 的默认产物根目录")
    web.add_argument("--default-workdir", default="./intake_workdir",
                     help="主理人 Agent Tab 的默认工作目录")
    web.add_argument("--debug", action="store_true")
    web.set_defaults(func=_cmd_webui)

    cc = sub.add_parser(
        "cocreate",
        help="OC 共创蒸馏：一键走完骨架→访谈→蒸馏三阶段",
    )
    cc.add_argument("--name", default="",
                    help="OC 姓名（提供则跳过交互式引导）")
    cc.add_argument("--age", default="", help="OC 年龄")
    cc.add_argument("--background", default="", help="OC 背景")
    cc.add_argument("--traits", default="", help="OC 性格核心")
    cc.add_argument("--worldview", default="", help="OC 世界观")
    cc.add_argument("--catchphrase", default="", help="OC 口头禅")
    cc.add_argument("--rounds", type=int, default=8,
                    help="访谈轮数（默认 8）")
    cc.add_argument("--output", required=True,
                    help="蒸馏产物输出目录")
    cc.add_argument("--persona-id", default="",
                    help="人格 ID（留空则从 name 自动生成 ASCII slug）")
    cc.add_argument("--model", default="minimax:MiniMax-M3",
                    help="provider:model 字符串（默认 minimax:MiniMax-M3）")
    cc.add_argument("--workdir", default="",
                    help="中间产物（oc_corpus/ + interview.md）目录（默认 ./oc_workdir）")
    cc.add_argument("--debug", action="store_true")
    cc.set_defaults(func=_cmd_cocreate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(debug=getattr(args, "debug", False))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
