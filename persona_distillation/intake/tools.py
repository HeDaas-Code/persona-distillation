"""主理人 Agent 的工具桥接层。

把 ``persona_distillation`` 包里已有的 Python 函数（``load_corpus`` /
``chunk_text`` / ``extract_names_from_chunk`` / ``IndexStore`` /
``build_profile`` / ``distill_character``）包装成 LangChain 工具，
让主理人 Agent 通过工具调用真实地读取磁盘、写入索引、启动蒸馏。

设计要点：
- ``IntakeContext`` 一次构造、被所有工具闭包共享（同一个 IndexStore / llm / workdir）
- 路径解析三段式：绝对路径 → 相对 CWD → 相对 workdir，命中即用
- 工具出错返回人类可读字符串而非抛异常，方便 LLM 兜底
- ``intake_corpus`` 把 5 步流程里的"接收文本 + 预处理"合并成一次调用，
  内部走确定性 Python 编排（``load_corpus → chunk_text → extract_names_from_chunk → store.add``），
  避免让 LLM 逐块委派子 agent 造成的不稳定
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool

from persona_distillation.config import DistillationConfig
from persona_distillation.loader import load_corpus
from persona_distillation.chunker import chunk_text
from persona_distillation.intake.embedder import HashEmbeddings, build_embedder, build_reranker
from persona_distillation.intake.index_store import IndexStore
from persona_distillation.intake.name_extractor import extract_names_from_chunk
from persona_distillation.intake.progress import ProgressReporter
from persona_distillation.intake.schemas import NameIndexEntry
from persona_distillation.intake.profile_builder import build_profile as _build_profile
from persona_distillation.intake.bridge import (
    distill_character as _distill_character,
    slugify,
)
from persona_distillation.intake.oc_writer import (
    OCSetting,
    generate_oc_corpus as _generate_oc_corpus,
)
from persona_distillation.intake.interview import run_interview as _run_interview

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 上下文：一次构造，被所有工具闭包共享
# ---------------------------------------------------------------------------
@dataclass
class IntakeContext:
    """主理人 Agent 工具共享的运行时上下文。"""

    cfg: DistillationConfig
    workdir: Path
    store: IndexStore
    llm: BaseChatModel | None = None
    reranker: Any | None = None


# ---------------------------------------------------------------------------
# 构造助手
# ---------------------------------------------------------------------------
def _build_intake_llm(cfg: DistillationConfig) -> BaseChatModel | None:
    """为 intake 工具构造一个真实的 ``BaseChatModel``。

    NER 与 profile 撰写都需要 LLM；离线模式返回 ``None``（走启发式 / 兜底）。
    """
    if cfg.offline:
        return None
    from persona_distillation.agents import build_model

    m = build_model(cfg)
    if isinstance(m, BaseChatModel):
        return m
    # build_model 返回字符串（非 minimax provider）→ 交给 init_chat_model
    try:
        from langchain.chat_models import init_chat_model

        return init_chat_model(m)  # type: ignore[arg-type]
    except Exception as e:  # noqa: BLE001
        logger.warning("intake LLM 初始化失败，NER/profile 将走兜底: %s", e)
        return None


def build_intake_context(cfg: DistillationConfig) -> IntakeContext:
    """根据 ``cfg`` 构造 ``IntakeContext``：建 workdir、IndexStore、llm、reranker。"""
    workdir = cfg.resolve_workdir()

    if cfg.offline:
        embedder: Any = HashEmbeddings()
    else:
        embedder = build_embedder(cfg.embedding_model)

    store = IndexStore(workdir / "index", embedding=embedder)
    llm = _build_intake_llm(cfg)

    reranker = None if cfg.offline else build_reranker(cfg.rerank_model, top_n=cfg.rerank_top_n)

    logger.info(
        "IntakeContext 就绪: workdir=%s offline=%s llm=%s reranker=%s embedder=%s",
        workdir,
        cfg.offline,
        type(llm).__name__ if llm is not None else "None(将走启发式)",
        "yes" if reranker is not None else "no",
        type(embedder).__name__,
    )
    return IntakeContext(cfg=cfg, workdir=workdir, store=store, llm=llm, reranker=reranker)


# ---------------------------------------------------------------------------
# 路径解析：绝对路径 → 相对 CWD → 相对 workdir
# ---------------------------------------------------------------------------
def _resolve_path(path_str: str, workdir: Path) -> Path:
    """三段式解析用户给的路径，返回真实存在的 ``Path``。

    Raises:
        FileNotFoundError: 三段都找不到时抛出，附三段候选路径方便排查。
    """
    p = Path(path_str).expanduser()
    candidates = [p]
    if not p.is_absolute():
        candidates.append(Path.cwd() / p)
        candidates.append(workdir / p)
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        f"找不到路径: {path_str}（已尝试: {[str(c) for c in candidates]}）"
    )


# ---------------------------------------------------------------------------
# 工具工厂
# ---------------------------------------------------------------------------
def build_intake_tools(ctx: IntakeContext) -> list[BaseTool]:
    """返回主理人 Agent 可用的工具列表。所有工具闭包共享 ``ctx``。"""

    @tool
    def load_text(path: str) -> str:
        """加载文件或目录的文本，返回文档清单（不索引、不分块）。

        用于第 1 步「接收文本」——先确认能读到，再决定是否走 intake_corpus。
        支持 .txt/.md/.json/.jsonl/.csv 等多种格式，目录会递归扫描。
        """
        try:
            p = _resolve_path(path, ctx.workdir)
            docs = load_corpus(p)
            lines = [f"已加载 {len(docs)} 篇文档（来源: {p}）："]
            for d in docs:
                lines.append(f"  - {d.relpath}  ({d.meta.get('size_chars', 0)} 字符)")
            lines.append(
                "下一步：调 intake_corpus 让我分块 + 识别人物 + 入库；"
                "或调 list_characters 看已索引的人物。"
            )
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            return f"加载失败: {e}\n请确认路径正确——可给绝对路径，或相对当前目录 / 工作目录({ctx.workdir})的相对路径。"

    @tool
    def intake_corpus(path: str, max_chunks: int = 0) -> str:
        """加载 + 分块 + 人物识别 + 入库（Chroma+SQLite）。

        这是 5 步流程里第 1~2 步的合并执行。完成后调 list_characters 看识别到的人物。

        支持断点续传：同一文件再次调用会跳过已处理的 chunk（基于 chunk.uuid +
        content_hash 缓存命中）。若 chunk 内容变化则删除旧索引重处理。

        Args:
            path: 文件或目录路径（绝对 / 相对 CWD / 相对 workdir 均可）。
            max_chunks: 本次最多处理多少个**新** chunk（已缓存的自动跳过，不计数）。
                        0 表示不限制，处理全部。用于长文本分多次摄入。
        """
        try:
            p = _resolve_path(path, ctx.workdir)
            docs = load_corpus(p)
        except Exception as e:  # noqa: BLE001
            return f"加载失败: {e}"

        before = ctx.store.count()
        total_new_mentions = 0
        total_skipped = 0
        total_processed = 0
        per_file: list[str] = []

        logger.info(
            "intake_corpus 开始: %d 篇文档, llm=%s, chunk_size=%d, max_chunks=%d",
            len(docs),
            type(ctx.llm).__name__ if ctx.llm is not None else "None(启发式)",
            ctx.cfg.intake_chunk_size,
            max_chunks,
        )

        for doc in docs:
            # 不再传 max_chunks 给 chunk_text——分块始终保持完整，限制只在
            # 「新 chunk 处理」环节生效（缓存命中的不算），这样断点续传才能
            # 在二次调用时跳过前 N 块继续处理后续。
            chunks = chunk_text(
                doc.text,
                target_tokens=ctx.cfg.intake_chunk_size,
                overlap_tokens=ctx.cfg.intake_chunk_overlap,
            )

            # 语料 UUID 绑定：LoadedDoc 已基于 content_hash 算好确定性 corpus_uuid，
            # 同一文件内容（即便路径/编码不同）→ 同一 uuid，是缓存命中的根。
            corpus_uuid = doc.corpus_uuid
            content_hash = doc.content_hash

            # 注册语料：首次返回 True 写入 registry，重复返回 False（断点续传场景，
            # 保留原进度不覆盖）。即使返回 False，后面仍要逐 chunk 检查缓存。
            is_new = ctx.store.register_corpus(
                corpus_uuid, doc.relpath, content_hash, len(chunks)
            )
            if not is_new:
                logger.info("语料已注册（断点续传）: %s uuid=%s", doc.relpath, corpus_uuid)

            # 进度指示器：输出到 stderr 不污染 LLM 对话流
            reporter = ProgressReporter(
                total=len(chunks),
                label=doc.relpath,
                show=ctx.cfg.show_progress,
            )

            # file_new 是 chunk 计数用于 max_chunks 限制；file_mentions 是
            # mention 计数用于显示，两者语义不同不要混用。
            file_new = 0          # 本次新处理的 chunk 数（用于 max_chunks 限制）
            file_mentions = 0     # 本次新处理的 mention 数（用于显示）
            file_skipped = 0      # 缓存命中跳过的 chunk 数
            file_processed = 0    # 已处理总数（含跳过），用于进度条

            for chunk in chunks:
                # 缓存检查：基于 (corpus_uuid, chunk.uuid) 查询。chunk.uuid 是
                # 确定性 v5（基于块索引 + 块正文 SHA-256[:16]），同一输入恒定。
                if ctx.cfg.enable_chunk_cache:
                    cached_hash = ctx.store.is_chunk_processed(
                        corpus_uuid, chunk.uuid
                    )
                    if cached_hash is not None:
                        # chunk 现算完整 SHA-256（与 chunk.uuid 里的 [:16] 不同，
                        # 用完整哈希做内容比对更严格）
                        chunk_hash = hashlib.sha256(
                            chunk.text.encode("utf-8")
                        ).hexdigest()
                        if cached_hash == chunk_hash:
                            # 内容没变，跳过（断点续传的核心命中点）
                            file_skipped += 1
                            file_processed += 1
                            reporter.update(file_processed, "")
                            logger.info(
                                "跳过已缓存 chunk %d (source=%s)",
                                chunk.index, doc.relpath,
                            )
                            continue
                        else:
                            # 内容变了：删旧索引条目后重处理，避免重复写入
                            logger.warning(
                                "chunk %d 内容变更 (source=%s)，重新处理",
                                chunk.index, doc.relpath,
                            )
                            ctx.store.delete_chunk_entries(
                                corpus_uuid, doc.relpath, chunk.index
                            )

                # max_chunks 限制：只数新处理的 chunk，不数缓存命中的跳过，
                # 这样长文本分多次摄入时每次都能精确控制成本。
                if max_chunks > 0 and file_new >= max_chunks:
                    reporter.update(file_processed, "")
                    break

                # NER 提取（启发式或 LLM）
                mentions = extract_names_from_chunk(
                    chunk,
                    source=doc.relpath,
                    llm=ctx.llm,
                    detect_injection=ctx.cfg.detect_injection,
                )

                # 写入索引：必须透传 corpus_uuid，否则后续按语料失效缓存会失效
                chunk_hash = hashlib.sha256(
                    chunk.text.encode("utf-8")
                ).hexdigest()
                chunk_mention_count = 0
                for m in mentions:
                    try:
                        entry = NameIndexEntry.from_mention(
                            m,
                            chunk_index=chunk.index,
                            source=doc.relpath,
                            global_char_start=chunk.char_start,
                            corpus_uuid=corpus_uuid,
                        )
                        ctx.store.add(entry)
                        chunk_mention_count += 1
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "索引写入失败 (source=%s chunk=%d): %s",
                            doc.relpath, chunk.index, e,
                        )

                # 标记 chunk 已处理（写 processed_chunks 表，便于下次断点续传）
                if ctx.cfg.enable_chunk_cache:
                    ctx.store.mark_chunk_processed(
                        corpus_uuid, chunk.uuid, chunk_hash, chunk_mention_count
                    )

                file_new += 1
                file_processed += 1
                file_mentions += chunk_mention_count
                total_new_mentions += chunk_mention_count

                # 进度更新：显示当前 chunk 提取到的人物名（截断到 50 字避免行爆宽）
                current_names = ", ".join(
                    sorted({m.name for m in mentions})
                )[:50]
                reporter.update(file_processed, current_names)

            # 批次结束：把 processed_chunks 表的实际行数同步回
            # corpus_registry.processed_chunks 缓存列
            if ctx.cfg.enable_chunk_cache:
                ctx.store.update_corpus_progress(corpus_uuid)

            # 进度收尾：打印本文件的最终统计
            remaining = len(chunks) - file_processed
            summary_parts = [f"{file_processed}/{len(chunks)} 块", f"新增 {file_mentions} 条提及"]
            if file_skipped > 0:
                summary_parts.append(f"跳过 {file_skipped} 块 (缓存命中)")
            if remaining > 0:
                summary_parts.append(f"剩余 {remaining} 块未处理")
            reporter.finish(", ".join(summary_parts))

            total_skipped += file_skipped
            total_processed += file_processed

            # 单文件汇总行（用于最终输出）
            file_line = f"  - {doc.relpath}: {file_processed}/{len(chunks)} 块 / 新增 {file_mentions} 条提及"
            if file_skipped > 0:
                file_line += f" / 跳过 {file_skipped} 块"
            if remaining > 0:
                file_line += f" / 剩余 {remaining} 块（可再次调用继续）"
            per_file.append(file_line)

        added = ctx.store.count() - before
        lines = [
            f"摄入完成：{len(docs)} 篇文档 / 本次新增 {added} 条索引（{total_new_mentions} 条提及）。",
            *per_file,
            f"索引库: {ctx.store.db_dir}",
        ]
        if total_skipped > 0:
            lines.append(f"缓存命中: 跳过 {total_skipped} 块（断点续传生效）")
        # 检查是否有任何文件还剩未处理 chunk（max_chunks 提前 break 的场景）
        has_remaining = any("剩余" in pf for pf in per_file)
        if has_remaining:
            lines.append(
                "提示: 部分分块未处理，可再次调用 intake_corpus 同路径继续，"
                "或直接调 list_characters 基于现有数据开始。"
            )
        else:
            lines.append("下一步：调 list_characters 查看识别到的人物。")
        return "\n".join(lines)

    @tool
    def list_characters() -> str:
        """列出已索引的全部人物 + 各类（speech/appearance/event）计数。"""
        try:
            chars = ctx.store.list_characters()
            if not chars:
                return "索引库里还没有人物。请先调 intake_corpus <文件或目录> 摄入语料。"
            lines = [f"已识别 {len(chars)} 位人物："]
            for i, c in enumerate(chars, 1):
                by_cat = c.get("by_category", {})
                cat_str = "/".join(f"{k}:{v}" for k, v in sorted(by_cat.items()))
                lines.append(f"  {i}. {c['character_name']}  (提及 {c['mention_count']} 次 | {cat_str})")
            lines.append("选择要蒸馏的人物：调 build_profile <名字或编号>。")
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            return f"列出人物失败: {e}"

    @tool
    def search_index(query: str, character_name: str = "") -> str:
        """按关键词检索索引条目（向量 + 关键词兜底）。

        Args:
            query: 检索关键词或问句。
            character_name: 可选，限定人物；留空则全库检索。
        """
        try:
            name = character_name.strip() or None
            results = ctx.store.search(query, character_name=name, k=ctx.cfg.index_top_k)
            if not results:
                return f"未检索到与「{query}」相关的条目。"
            lines = [f"检索「{query}」命中 {len(results)} 条："]
            for i, e in enumerate(results, 1):
                lines.append(f"  {i}. [{e.category.value}] {e.character_name}: {e.text[:120]}")
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            return f"检索失败: {e}"

    @tool
    def build_profile(character_name: str) -> str:
        """从索引聚合指定人物的档案（speech/appearance/event + 200字摘要）。

        用于 5 步流程第 5 步「档案」——完成后让用户确认是否蒸馏。
        """
        try:
            chars = ctx.store.list_characters()
            names = [c["character_name"] for c in chars]
            resolved = _resolve_character_name(character_name, names)
            if resolved is None:
                return f"未找到人物「{character_name}」。已识别: {names or '（空）'}"

            profile = _build_profile(
                resolved,
                ctx.store,
                reranker=ctx.reranker,
                llm=ctx.llm,
                top_n=ctx.cfg.rerank_top_n,
                max_entries=ctx.cfg.profile_max_entries,
            )
            lines = [
                f"人物档案 · {profile.character_name}",
                f"  别名: {', '.join(profile.aliases) or '(无)'}",
                f"  提及 {profile.mention_count} 次 | 对话 {profile.speech_count} | 外貌 {profile.appearance_count} | 事件 {profile.event_count}",
                "",
                "摘要：",
                profile.summary,
                "",
                f"确认蒸馏此人物？调 distill_character {profile.character_name} 启动。",
            ]
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            return f"档案构建失败: {e}"

    @tool
    def distill_character(character_name: str) -> str:
        """启动指定人物的四阶段蒸馏（extractor → synthesizer → skill_designer → dialogue_writer）。

        会把档案重建成临时语料目录喂给 PersonaDistiller，产物落到
        <workdir>/distilled/<persona_id>/ 下。耗时较长，请耐心等待。
        """
        try:
            chars = ctx.store.list_characters()
            names = [c["character_name"] for c in chars]
            resolved = _resolve_character_name(character_name, names)
            if resolved is None:
                return f"未找到人物「{character_name}」。已识别: {names or '（空）'}"

            profile = _build_profile(
                resolved,
                ctx.store,
                reranker=ctx.reranker,
                llm=ctx.llm,
                top_n=ctx.cfg.rerank_top_n,
                max_entries=ctx.cfg.profile_max_entries,
            )
            result = _distill_character(profile, ctx.cfg, ctx.workdir)
            out_dir = ctx.workdir / "distilled" / result.persona_card.persona_id
            lines = [
                f"蒸馏完成 · {result.persona_card.display_name or result.persona_card.persona_id}",
                f"  人格ID: {result.persona_card.persona_id}",
                f"  Skills: {len(result.skills)} 个 → {[s.name for s in result.skills]}",
                f"  预设对话: {len(result.preset_dialogues)} 组",
                f"  分馏液: {len(result.distillates)} 块",
                f"  产物目录: {out_dir}",
                "  - persona_card.json / persona_card.md",
                "  - preset_dialogues.json",
                "  - distillates.jsonl",
                "  - skills/<name>/SKILL.md",
            ]
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            logger.error("蒸馏失败: %s", e, exc_info=True)
            return f"蒸馏失败: {e}\n（详见日志；可重试或换个人物）"

    @tool
    def generate_oc_corpus(setting_json: str) -> str:
        """从 OC 设定生成骨架语料（独白/对话/事件/回忆 4 类）。

        用于 OC 共创蒸馏 Phase 1。setting_json 是一段 JSON 字符串，需含：
        name / age / background / traits / worldview / catchphrase 六字段。

        4 类文本落到 <workdir>/<persona_id>/oc_corpus/，完成后可调
        run_character_interview 完善血肉，再调 distill_character 蒸馏。
        """
        # 解析 OC 设定 JSON → OCSetting
        try:
            setting = OCSetting.model_validate_json(setting_json)
        except Exception as e:  # noqa: BLE001
            return (
                f"OC 设定解析失败: {e}\n"
                "setting_json 需含 name/age/background/traits/worldview/catchphrase 六字段。"
            )

        # 离线模式兜底：没有 LLM 就无法生成
        if ctx.llm is None:
            return (
                "当前为离线模式（无可用 LLM），无法生成 OC 骨架语料。\n"
                "请配置 DistillationConfig.model 后重试。"
            )

        persona_id = slugify(setting.name)
        try:
            result = _generate_oc_corpus(setting, ctx.workdir, persona_id, ctx.llm)
        except Exception as e:  # noqa: BLE001
            logger.error("OC 骨架生成失败: %s", e, exc_info=True)
            return f"OC 骨架生成失败: {e}\n（详见日志；可重试）"

        paths = result.get("paths", {})
        word_counts = result.get("word_counts", {})
        corpus_dir = result.get("corpus_dir", "")
        lines = [
            f"OC 骨架生成完成 · {setting.name}",
            f"  人格ID: {persona_id}",
            f"  骨架目录: {corpus_dir}",
        ]
        for key in ("monologue", "dialogue", "event", "memory"):
            if key in paths:
                lines.append(f"  - {key}: {word_counts.get(key, 0)} 字 → {paths[key]}")
        lines.append(
            "下一步：调 run_character_interview 完善血肉，再调 distill_character 蒸馏。"
        )
        return "\n".join(lines)

    @tool
    def run_character_interview(setting_json: str, n_rounds: int = 8) -> str:
        """基于 OC 骨架对 OC 做访谈，完善角色血肉。

        用于 OC 共创蒸馏 Phase 2。需先调 generate_oc_corpus 生成骨架。
        setting_json 同 generate_oc_corpus（name/age/background/traits/worldview/catchphrase）。
        n_rounds 默认 8。

        访谈记录落 <workdir>/<persona_id>/interview.md，完成后调 distill_character 蒸馏。
        """
        # 解析 OC 设定 JSON → OCSetting
        try:
            setting = OCSetting.model_validate_json(setting_json)
        except Exception as e:  # noqa: BLE001
            return (
                f"OC 设定解析失败: {e}\n"
                "setting_json 需含 name/age/background/traits/worldview/catchphrase 六字段。"
            )

        # 离线模式兜底：没有 LLM 就无法访谈
        if ctx.llm is None:
            return (
                "当前为离线模式（无可用 LLM），无法执行角色访谈。\n"
                "请配置 DistillationConfig.model 后重试。"
            )

        persona_id = slugify(setting.name)
        try:
            result = _run_interview(setting, n_rounds, ctx.workdir, persona_id, ctx.llm)
        except FileNotFoundError as e:
            # 骨架不存在：引导先调 generate_oc_corpus
            return (
                f"骨架不存在: {e}\n"
                "请先调 generate_oc_corpus 生成骨架后再访谈。"
            )
        except Exception as e:  # noqa: BLE001
            logger.error("OC 访谈失败: %s", e, exc_info=True)
            return f"OC 访谈失败: {e}\n（详见日志；可重试）"

        path = result.get("path", "")
        rounds = result.get("rounds", n_rounds)
        distill_dir = ctx.workdir / persona_id
        lines = [
            f"OC 访谈完成 · {setting.name}",
            f"  人格ID: {persona_id}",
            f"  轮数: {rounds}",
            f"  访谈记录: {path}",
            f"  蒸馏目录: {distill_dir}",
            "下一步：调 distill_character 启动蒸馏"
            f"（蒸馏输入目录 {distill_dir}/，loader 会递归读取 oc_corpus/ + interview.md）。",
        ]
        return "\n".join(lines)

    return [
        load_text,
        intake_corpus,
        list_characters,
        search_index,
        build_profile,
        distill_character,
        generate_oc_corpus,
        run_character_interview,
    ]


# ---------------------------------------------------------------------------
# 人物名模糊匹配：编号 / 全名 / 别名
# ---------------------------------------------------------------------------
def _resolve_character_name(user_input: str, known: list[str]) -> str | None:
    """把用户输入解析成索引库里的人物名。

    支持编号（1/2/3…）、精确名、子串匹配。
    """
    s = user_input.strip()
    if not s:
        return None
    # 编号
    if s.isdigit():
        idx = int(s) - 1
        if 0 <= idx < len(known):
            return known[idx]
    # 精确
    if s in known:
        return s
    # 子串
    hits = [n for n in known if s in n or n in s]
    if len(hits) == 1:
        return hits[0]
    return None
