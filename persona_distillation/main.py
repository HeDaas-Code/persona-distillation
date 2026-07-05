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
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

from persona_distillation.config import DistillationConfig
from persona_distillation.pipeline import PersonaDistiller
from persona_distillation.schemas import DistillationResult


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
    chat.add_argument("--debug", action="store_true")
    chat.set_defaults(func=_cmd_chat)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
