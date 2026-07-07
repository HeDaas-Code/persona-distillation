"""主理人 Agent 的工具桥接层（Phase 1 重构后版本）。

设计要点（与旧版的区别）：
- **NER / 档案 / 蒸馏不再藏在 Python 工具里**——这些 LLM 推理任务由主理人通过
  ``task`` 分派 SubAgent 完成（intake_ner / profile_builder / extractor / synthesizer /
  skill_designer / dialogue_writer / bridger），见 ``agents.build_intake_orchestrator``。
- Python 工具只保留**纯 IO/查询**与**SubAgent 间文件交接**：
  ``load_text`` / ``load_and_chunk`` / ``index_characters`` / ``list_characters`` /
  ``search_index`` / ``get_character_entries`` / ``save_distillates`` / ``load_distillates``
  + OC 共创的 ``generate_oc_corpus`` / ``run_character_interview``。
- ``IntakeContext`` 一次构造、被所有工具闭包共享（同一个 IndexStore / llm / workdir）。
- 路径解析三段式：绝对路径 → 相对 CWD → 相对 workdir，命中即用。
- 工具出错返回人类可读字符串而非抛异常，方便 LLM 兜底。

旧版的 ``intake_corpus``（load+chunk+NER+入库一条龙）、``build_profile``、
``distill_character`` 三个 Python 黑箱工具已移除——它们的职责分别由
``load_and_chunk`` + ``task(intake_ner)`` + ``index_characters``、
``task(profile_builder)``、``task(extractor/synthesizer/skill_designer/dialogue_writer)``
接管。底层 Python 函数（``bridge.distill_character`` / ``profile_builder.build_profile``）
仍保留，供 CLI ``distill`` / WebUI 蒸馏 Tab / ``pipeline.PersonaDistiller.distill()`` 直跑。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool

from persona_distillation.config import DistillationConfig
from persona_distillation.loader import load_corpus
from persona_distillation.chunker import chunk_text, dedup_chunks
from persona_distillation.intake.embedder import HashEmbeddings, build_embedder, build_reranker
from persona_distillation.intake.index_store import IndexStore
from persona_distillation.intake.schemas import (
    NameIndexEntry,
    NameMention,
    NerBatchResult,
)
from persona_distillation.intake.bridge import slugify
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

        用于第 1 步「接收文本」——先确认能读到，再决定是否走 load_and_chunk。
        支持 .txt/.md/.json/.jsonl/.csv 等多种格式，目录会递归扫描。
        """
        try:
            p = _resolve_path(path, ctx.workdir)
            docs = load_corpus(p)
            lines = [f"已加载 {len(docs)} 篇文档（来源: {p}）："]
            for d in docs:
                lines.append(f"  - {d.relpath}  ({d.meta.get('size_chars', 0)} 字符)")
            lines.append(
                "下一步：调 load_and_chunk <path> 让我分块（不做 NER）；"
                "或调 list_characters 看已索引的人物。"
            )
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            return f"加载失败: {e}\n请确认路径正确——可给绝对路径，或相对当前目录 / 工作目录({ctx.workdir})的相对路径。"

    @tool
    def load_and_chunk(path: str) -> str:
        """加载文件或目录 + 分块，返回 chunk 列表 JSON（**不做 NER**）。

        用于 9 步主流程第 1 步「接收文本 + 分块」。返回的 JSON 数组每个元素含：
        source / chunk_index / text / char_start / char_end / token_count /
        corpus_uuid / content_hash / total_chunks。

        下一步：把返回的 chunk 列表 JSON 交给 `task(intake_ner)` SubAgent 批量做 NER，
        再把 NerBatchResult JSON 调 index_characters 写入索引库。
        """
        try:
            p = _resolve_path(path, ctx.workdir)
            docs = load_corpus(p)
        except Exception as e:  # noqa: BLE001
            return f"加载失败: {e}"

        chunks_out: list[dict] = []
        # Issue #18.a: chunk 去重（NER 之前）。embedder 取自 IndexStore 的嵌入模型；
        # 离线模式（HashEmbeddings）或 embedder=None 时退化为 SHA-256 精确匹配。
        embedder_for_dedup = getattr(ctx.store, "_embedding", None)
        dedup_threshold = ctx.cfg.chunk_dedup_threshold
        total_dups = 0
        for doc in docs:
            chunks = chunk_text(
                doc.text,
                target_tokens=ctx.cfg.intake_chunk_size,
                overlap_tokens=ctx.cfg.intake_chunk_overlap,
            )
            before_dedup = len(chunks)
            chunks = dedup_chunks(chunks, embedder_for_dedup, threshold=dedup_threshold)
            total_dups += before_dedup - len(chunks)
            # total_chunks 用原始 chunk 总数（去重不重排 index，便于定位）
            total = before_dedup
            for chunk in chunks:
                chunks_out.append({
                    "source": doc.relpath,
                    "chunk_index": chunk.index,
                    "text": chunk.text,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "token_count": chunk.token_count,
                    "uuid": chunk.uuid,
                    "corpus_uuid": doc.corpus_uuid,
                    "content_hash": doc.content_hash,
                    "total_chunks": total,
                })

        logger.info(
            "load_and_chunk: %d 篇文档 / %d 块 (chunk_size=%d, 去重 %d 块)",
            len(docs), len(chunks_out), ctx.cfg.intake_chunk_size, total_dups,
        )
        if not chunks_out:
            return "未切出任何分块——请确认文件非空且可读。"
        return json.dumps(chunks_out, ensure_ascii=False)

    @tool
    def index_characters(ner_results_json: str) -> str:
        """接收 NerBatchResult JSON，写入 IndexStore（Chroma + SQLite）。

        用于 9 步主流程第 3 步「建索引」。输入是 intake_ner SubAgent 产出的
        NerBatchResult 序列化 JSON（含 items 列表，每个 item 有 chunk_meta + mentions）。
        内部调 IndexStore.add() 写入索引条目，保留 corpus_uuid / source / chunk_index /
        char_start 定位信息。

        完成后调 list_characters 看识别到的人物。
        """
        try:
            data = json.loads(ner_results_json)
        except json.JSONDecodeError as e:
            return f"NER 结果 JSON 解析失败: {e}"

        # 兼容 SubAgent 直接返回 NerBatchResult 对象的 dict 形式（{"items": [...]}）
        # 或裸返回 items 列表（[...]）两种情况
        if isinstance(data, dict) and "items" in data:
            items = data.get("items", [])
        elif isinstance(data, list):
            items = data
        else:
            return "NER 结果应是 NerBatchResult（含 items 列表）或 items 数组"

        before = ctx.store.count()
        total_mentions = 0
        per_chunk: list[str] = []

        for item in items:
            if not isinstance(item, dict):
                continue
            chunk_meta = item.get("chunk_meta", {}) or {}
            mentions_data = item.get("mentions", []) or []
            source = str(chunk_meta.get("source", "unknown"))
            try:
                chunk_index = int(chunk_meta.get("chunk_index", 0))
            except (TypeError, ValueError):
                chunk_index = 0
            try:
                global_char_start = int(chunk_meta.get("char_start", 0))
            except (TypeError, ValueError):
                global_char_start = 0
            corpus_uuid = str(chunk_meta.get("corpus_uuid", ""))

            # 解析 mentions → NameMention
            mentions: list[NameMention] = []
            for m_data in mentions_data:
                try:
                    mentions.append(NameMention.model_validate(m_data))
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "mention 解析失败 (source=%s chunk=%d): %s",
                        source, chunk_index, e,
                    )

            # 写入索引：复用 intake_corpus 旧版的 from_mention 逻辑
            chunk_mention_count = 0
            for m in mentions:
                try:
                    entry = NameIndexEntry.from_mention(
                        m,
                        chunk_index=chunk_index,
                        source=source,
                        global_char_start=global_char_start,
                        corpus_uuid=corpus_uuid,
                    )
                    ctx.store.add(entry)
                    chunk_mention_count += 1
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "索引写入失败 (source=%s chunk=%d): %s",
                        source, chunk_index, e,
                    )

            total_mentions += chunk_mention_count
            per_chunk.append(f"  - {source}[{chunk_index}]: {chunk_mention_count} 条提及")

        added = ctx.store.count() - before
        lines = [
            f"索引建立完成：{len(items)} 个 chunk / 新增 {added} 条索引（{total_mentions} 条提及）。",
            *per_chunk,
            f"索引库: {ctx.store.db_dir}",
            "下一步：调 list_characters 查看识别到的人物。",
        ]
        return "\n".join(lines)

    @tool
    def list_characters() -> str:
        """列出已索引的全部人物 + 各类（speech/appearance/event）计数 + aliases。

        若 ``cfg.auto_merge=True``，调用前会先做一次跨 chunk 实体归并
        （见 :func:`~persona_distillation.intake.entity_resolver.resolve_entities`），
        把同一人物在不同 chunk 里被识别成的多个称谓（如「荒川善次」/「老师」/「荒川」）
        合并成一条。命中 ≥2 重信号（别名交叉 / 字符串相似 / 嵌入相似）的自动合并；
        仅命中 1 重的对照会在列表末尾标「待确认」。
        """
        try:
            # 自动归并：在 list 之前调一次 resolve_entities
            merge_summary = ""
            if ctx.cfg.auto_merge:
                try:
                    from persona_distillation.intake.entity_resolver import resolve_entities

                    res = resolve_entities(
                        ctx.store,
                        llm=ctx.llm,
                        auto_merge=True,
                        threshold=ctx.cfg.auto_merge_threshold,
                    )
                    if res.auto_merged or res.pending:
                        parts = []
                        if res.auto_merged:
                            parts.append(
                                f"自动合并 {len(res.auto_merged)} 对："
                                + ", ".join(f"{s}→{t}" for s, t in res.auto_merged)
                            )
                        if res.pending:
                            parts.append(f"待确认 {len(res.pending)} 对")
                        merge_summary = "（归并：" + "；".join(parts) + "）"
                except Exception as e:  # noqa: BLE001
                    logger.warning("auto_merge 失败，跳过: %s", e)
                    merge_summary = f"（auto_merge 跳过: {e}）"

            chars = ctx.store.list_characters()
            if not chars:
                return (
                    "索引库里还没有人物。请先走 9 步主流程前 3 步：\n"
                    "  1) load_and_chunk <path> 分块\n"
                    "  2) task(intake_ner) 批量 NER\n"
                    "  3) index_characters <ner_results_json> 建索引"
                )
            header = f"已识别 {len(chars)} 位人物"
            if merge_summary:
                header += merge_summary
            lines = [header + "："]
            for i, c in enumerate(chars, 1):
                by_cat = c.get("by_category", {})
                cat_str = "/".join(f"{k}:{v}" for k, v in sorted(by_cat.items()))
                aliases = c.get("aliases") or []
                alias_str = f" | 别名: {','.join(aliases)}" if aliases else ""
                lines.append(
                    f"  {i}. {c['character_name']}  "
                    f"(提及 {c['mention_count']} 次 | {cat_str}{alias_str})"
                )
            # 「待确认」对照提示：重新跑一次只读 resolve（auto_merge=False）拿 pending 列表
            if ctx.cfg.auto_merge:
                try:
                    from persona_distillation.intake.entity_resolver import resolve_entities

                    res2 = resolve_entities(
                        ctx.store,
                        llm=ctx.llm,
                        auto_merge=False,
                        threshold=ctx.cfg.auto_merge_threshold,
                    )
                    if res2.pending:
                        lines.append("")
                        lines.append("以下人物对照仅命中 1 重信号，待确认是否同一人：")
                        for sig in res2.pending:
                            tags = []
                            if sig.alias_hit:
                                tags.append("别名")
                            if sig.string_hit:
                                tags.append("字符串")
                            if sig.embedding_hit:
                                tags.append(f"嵌入{sig.embedding_score:.2f}")
                            lines.append(
                                f"  - {sig.name_a} ↔ {sig.name_b}  "
                                f"(命中: {'/'.join(tags) or '无'})"
                            )
                except Exception as e:  # noqa: BLE001
                    logger.debug("pending 复算失败: %s", e)
            lines.append(
                "选择要蒸馏的人物：调 get_character_entries <名字或编号> 拿索引条目，"
                "再 task(profile_builder) 建档案。"
            )
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            return f"列出人物失败: {e}"

    @tool
    def resolve_characters() -> str:
        """跨 chunk 实体归并：把同一人物在不同 chunk 里被识别成的多个称谓合并。

        用于 list_characters 之前的手动归并（若 cfg.auto_merge=False，
        list_characters 不会自动归并，需主理人显式调本工具）。

        三重信号融合判断是否同一人：
        a. 别名交叉：A 的 aliases 含 B 的 name（或反之）
        b. 字符串相似：Levenshtein ≤ 2 或 Jaro-Winkler ≥ cfg.auto_merge_threshold
        c. 嵌入相似：各人物 top-5 evidence 的 embedding 平均值，cosine ≥ 0.8
           （无真嵌入时降级，只用 a+b）

        命中 ≥2 重自动合并；命中 1 重标记「待确认」。
        """
        try:
            from persona_distillation.intake.entity_resolver import resolve_entities

            res = resolve_entities(
                ctx.store,
                llm=ctx.llm,
                auto_merge=True,
                threshold=ctx.cfg.auto_merge_threshold,
            )
            lines = [f"实体归并完成（阈值 {ctx.cfg.auto_merge_threshold}）："]
            if res.auto_merged:
                lines.append(f"  自动合并 {len(res.auto_merged)} 对：")
                for source, target in res.auto_merged:
                    lines.append(f"    - {source} → {target}")
            else:
                lines.append("  无自动合并")
            if res.pending:
                lines.append(f"  待确认 {len(res.pending)} 对（仅命中 1 重信号）：")
                for sig in res.pending:
                    tags = []
                    if sig.alias_hit:
                        tags.append("别名")
                    if sig.string_hit:
                        tags.append("字符串")
                    if sig.embedding_hit:
                        tags.append(f"嵌入{sig.embedding_score:.2f}")
                    lines.append(
                        f"    - {sig.name_a} ↔ {sig.name_b}  (命中: {'/'.join(tags)})"
                    )
            if res.skipped:
                lines.append(f"  跳过 {len(res.skipped)} 对（合并失败）")
            lines.append("下一步：调 list_characters 查看归并后的人物列表。")
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            return f"实体归并失败: {e}"

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
    def get_character_entries(character_name: str) -> str:
        """获取指定人物的全部索引条目（JSON 数组），供 profile_builder SubAgent 使用。

        用于 9 步主流程第 5 步——把返回的 JSON + 人物名一起交给 task(profile_builder)
        SubAgent，由 SubAgent 撰写 CharacterProfile（含 ≤200 字摘要）。

        返回的 JSON 数组每个元素含：category / text / source / chunk_index /
        char_start / char_end / aliases。
        """
        try:
            chars = ctx.store.list_characters()
            names = [c["character_name"] for c in chars]
            resolved = _resolve_character_name(character_name, names)
            if resolved is None:
                return f"未找到人物「{character_name}」。已识别: {names or '（空）'}"

            entries = ctx.store.get_character_entries(resolved)
            if not entries:
                return f"人物「{resolved}」暂无索引条目。"
            items = []
            for e in entries:
                items.append({
                    "category": e.category.value,
                    "text": e.text,
                    "source": e.source,
                    "chunk_index": e.chunk_index,
                    "char_start": e.char_start,
                    "char_end": e.char_end,
                    "aliases": list(e.aliases),
                })
            return json.dumps(items, ensure_ascii=False)
        except Exception as e:  # noqa: BLE001
            return f"获取条目失败: {e}"

    @tool
    def save_distillates(distillates_json: str) -> str:
        """把 DistillateList JSON 写到 <workdir>/distillates.json（SubAgent 间文件交接）。

        用于 9 步主流程第 6 步之后——extractor SubAgent 产出 DistillateList 后，
        主理人调本工具持久化，避免在 context 里长期保留大体量中间产物。
        下一个 SubAgent（synthesizer）通过 load_distillates 取回。
        """
        path = ctx.workdir / "distillates.json"
        try:
            path.write_text(distillates_json, encoding="utf-8")
            # 简单校验：能解析成 JSON 数组或 {distillates: [...]}
            try:
                data = json.loads(distillates_json)
                if isinstance(data, dict) and "distillates" in data:
                    n = len(data.get("distillates") or [])
                elif isinstance(data, list):
                    n = len(data)
                else:
                    n = -1
            except json.JSONDecodeError:
                n = -1
            if n < 0:
                return f"已写入 {path}，但内容不是合法 JSON 数组/DistillateList。"
            return f"已持久化 {n} 条 Distillate → {path}。下一步：调 load_distillates 取回，再 task(synthesizer)。"
        except Exception as e:  # noqa: BLE001
            return f"写入 distillates.json 失败: {e}"

    @tool
    def load_distillates() -> str:
        """读回 <workdir>/distillates.json 的 JSON 字符串（SubAgent 间文件交接）。

        用于 9 步主流程第 7 步——分派 synthesizer SubAgent 前，主理人调本工具
        取回 extractor 的 DistillateList，连同 CharacterProfile 一起放进 task prompt。
        """
        path = ctx.workdir / "distillates.json"
        if not path.exists():
            return (
                f"文件不存在: {path}。\n"
                "请先 task(extractor) 产出 DistillateList，再调 save_distillates 持久化。"
            )
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            return f"读取 distillates.json 失败: {e}"

    @tool
    def generate_oc_corpus(setting_json: str) -> str:
        """从 OC 设定生成骨架语料（独白/对话/事件/回忆 4 类）。

        用于 OC 共创蒸馏 Phase 1。setting_json 是一段 JSON 字符串，需含：
        name / age / background / traits / worldview / catchphrase 六字段。

        4 类文本落到 <workdir>/<persona_id>/oc_corpus/，完成后可调
        run_character_interview 完善血肉，再走 load_and_chunk + task(extractor) 等蒸馏步骤。
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
            "下一步：调 run_character_interview 完善血肉；然后调 load_and_chunk "
            f"<{corpus_dir}> 分块，再走 task(extractor) → task(synthesizer) → "
            "task(skill_designer) → task(dialogue_writer) 蒸馏。"
        )
        return "\n".join(lines)

    @tool
    def run_character_interview(setting_json: str, n_rounds: int = 8) -> str:
        """基于 OC 骨架对 OC 做访谈，完善角色血肉。

        用于 OC 共创蒸馏 Phase 2。需先调 generate_oc_corpus 生成骨架。
        setting_json 同 generate_oc_corpus（name/age/background/traits/worldview/catchphrase）。
        n_rounds 默认 8。

        访谈记录落 <workdir>/<persona_id>/interview.md，完成后走 load_and_chunk + 蒸馏。
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
            "下一步：调 load_and_chunk "
            f"<{distill_dir}> 分块（loader 会递归读取 oc_corpus/ + interview.md），"
            "再走 task(extractor) → task(synthesizer) → task(skill_designer) → "
            "task(dialogue_writer) 蒸馏。",
        ]
        return "\n".join(lines)

    return [
        load_text,
        load_and_chunk,
        index_characters,
        resolve_characters,
        list_characters,
        search_index,
        get_character_entries,
        save_distillates,
        load_distillates,
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
