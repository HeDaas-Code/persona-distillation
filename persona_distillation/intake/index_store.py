"""索引服务：Chroma（向量）+ SQLite（元数据）双写。

- **Chroma** 存 embedding（按 text 算）
- **SQLite** 存元数据（uuid/character_name/category/source/positions/text）
- 两者通过 ``uuid`` 关联

设计目标：
- 嵌入式运行（不开 server）—— ``chromadb.PersistentClient``
- 离线可用 —— ``embedding`` 接受任何 langchain ``Embeddings``，包括 :class:`HashEmbeddings`
- 查询接口简明 —— ``get_character_entries`` / ``list_characters`` / ``search``
- 原子写入 —— P0-3 修复：Chroma 失败时回滚 SQLite
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

from persona_distillation.intake.embedder import HashEmbeddings
from persona_distillation.intake.schemas import (
    IndexCategory,
    NameIndexEntry,
)


def _table_cols() -> str:
    return (
        "uuid TEXT PRIMARY KEY, "
        "character_name TEXT NOT NULL, "
        "category TEXT NOT NULL, "
        "source TEXT NOT NULL, "
        "chunk_index INTEGER NOT NULL, "
        "char_start INTEGER NOT NULL, "
        "char_end INTEGER NOT NULL, "
        "text TEXT NOT NULL, "
        "aliases TEXT NOT NULL DEFAULT '', "
        "corpus_uuid TEXT NOT NULL DEFAULT '', "
        "relation_to TEXT, "
        "co_mentioned TEXT NOT NULL DEFAULT ''"
    )


# 读取 entries 时统一选取的列（含关系提取新增列），保证 _row_to_entry 解包一致
_ENTRY_SELECT_COLS = (
    "uuid, character_name, category, source, chunk_index, "
    "char_start, char_end, text, aliases, corpus_uuid, "
    "relation_to, co_mentioned"
)


class IndexStore:
    """人名索引：Chroma + SQLite 双写。"""

    COLLECTION_NAME = "persona_index"

    def __init__(self, db_dir: str | Path, embedding: Any | None = None) -> None:
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        # ---- SQLite ----
        self._sqlite_path = self.db_dir / "meta.sqlite"
        self._conn = sqlite3.connect(str(self._sqlite_path), check_same_thread=False)
        # P0-3: 启用 WAL 模式以提升并发安全
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        except Exception as e:  # noqa: BLE001
            logger.warning("启用 WAL 模式失败: %s", e)
        self._init_db()

        # ---- Chroma ----
        self._embedding = embedding
        self._collection: Any = None
        try:
            from langchain_chroma import Chroma  # type: ignore

            if self._embedding is not None:
                self._collection = Chroma(
                    collection_name=self.COLLECTION_NAME,
                    embedding_function=self._embedding,
                    persist_directory=str(self.db_dir / "chroma"),
                )
        except Exception as e:  # noqa: BLE001
            # chromadb 不可用 → 退化到纯 SQLite
            logger.warning("Chroma 初始化失败，退化到 SQLite-only 模式: %s", e)
            self._collection = None

    def _init_db(self) -> None:
        """初始化/升级 SQLite schema。

        - 创建 ``entries`` 表（新库）
        - 旧库兼容：``ALTER TABLE entries ADD COLUMN corpus_uuid``（已存在则忽略）
        - 创建索引（``idx_name`` / ``idx_cat`` / ``idx_corpus``）
        - 创建缓存表 ``corpus_registry`` 与 ``processed_chunks``（chunk-progress-cache）
        """
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS entries ({_table_cols()})"
        )
        # 旧库兼容：entries 表可能没有 corpus_uuid 列
        try:
            self._conn.execute(
                "ALTER TABLE entries ADD COLUMN corpus_uuid TEXT NOT NULL DEFAULT ''"
            )
        except sqlite3.OperationalError:
            pass  # 列已存在
        # 旧库兼容：entries 表可能没有 relation_to / co_mentioned 列（关系提取新增）
        try:
            self._conn.execute("ALTER TABLE entries ADD COLUMN relation_to TEXT")
        except sqlite3.OperationalError:
            pass  # 列已存在
        try:
            self._conn.execute(
                "ALTER TABLE entries ADD COLUMN co_mentioned TEXT NOT NULL DEFAULT ''"
            )
        except sqlite3.OperationalError:
            pass  # 列已存在
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_name ON entries(character_name)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cat ON entries(category)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_corpus ON entries(corpus_uuid)"
        )

        # ---- chunk-progress-cache 缓存表 ----
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS corpus_registry (
                corpus_uuid TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                total_chunks INTEGER NOT NULL DEFAULT 0,
                processed_chunks INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_chunks (
                corpus_uuid TEXT NOT NULL,
                chunk_uuid TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                processed_at TEXT NOT NULL,
                mention_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (corpus_uuid, chunk_uuid)
            )
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # 写
    # ------------------------------------------------------------------
    def add(self, entry: NameIndexEntry) -> None:
        """写入一条索引。P0-3: 双写原子化 —— Chroma 失败时回滚 SQLite。"""
        meta = entry.to_metadata()
        # SQLite
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO entries "
                "(uuid, character_name, category, source, chunk_index, "
                "char_start, char_end, text, aliases, corpus_uuid, "
                "relation_to, co_mentioned) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.uuid,
                    entry.character_name,
                    entry.category.value,
                    entry.source,
                    entry.chunk_index,
                    entry.char_start,
                    entry.char_end,
                    entry.text,
                    ",".join(entry.aliases),
                    entry.corpus_uuid,
                    entry.relation_to,
                    ",".join(entry.co_mentioned),
                ),
            )
            self._conn.commit()
        except Exception as e:  # noqa: BLE001
            logger.error("SQLite INSERT 失败: %s", e, exc_info=True)
            raise

        # Chroma
        if self._collection is not None:
            try:
                self._collection.add_texts(
                    texts=[entry.text],
                    metadatas=[meta],
                    ids=[entry.uuid],
                )
            except Exception as e:  # noqa: BLE001
                # P0-3: 回滚 SQLite（保证一致性）
                logger.error(
                    "Chroma add 失败，回滚 SQLite (uuid=%s): %s", entry.uuid, e,
                    exc_info=True,
                )
                try:
                    self._conn.execute("DELETE FROM entries WHERE uuid=?", (entry.uuid,))
                    self._conn.commit()
                except Exception as rb_e:  # noqa: BLE001
                    logger.error("SQLite 回滚失败: %s", rb_e, exc_info=True)
                raise

    def add_many(self, entries: Iterable[NameIndexEntry]) -> None:
        """批量写入索引（单线程，原子化）。

        Issue #16: NER 并行产出 entries 后，由主线程串行调本方法写库——
        SQLite 并发写会 ``database is locked``，Chroma 并发写可能损坏，
        所以写库必须串行。本方法把全部 entries 收敛到一次 SQLite 事务 +
        一次 Chroma ``add_texts`` 调用，比逐条 ``add()`` 快很多（少 N-1 次
        commit + N-1 次 Chroma round-trip）。

        原子性（继承 P0-3 策略）：Chroma 失败时回滚 SQLite 全部插入。
        """
        entries_list = list(entries)
        if not entries_list:
            logger.info("add_many: 空列表，跳过")
            return

        # ---- SQLite 批量插入（单事务）----
        try:
            self._conn.execute("BEGIN")
            for e in entries_list:
                self._conn.execute(
                    "INSERT OR REPLACE INTO entries "
                    "(uuid, character_name, category, source, chunk_index, "
                    "char_start, char_end, text, aliases, corpus_uuid, "
                    "relation_to, co_mentioned) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        e.uuid,
                        e.character_name,
                        e.category.value,
                        e.source,
                        e.chunk_index,
                        e.char_start,
                        e.char_end,
                        e.text,
                        ",".join(e.aliases),
                        e.corpus_uuid,
                        e.relation_to,
                        ",".join(e.co_mentioned),
                    ),
                )
            self._conn.commit()
        except Exception as e:  # noqa: BLE001
            # SQLite 失败：回滚事务，不写 Chroma
            try:
                self._conn.execute("ROLLBACK")
            except Exception:  # noqa: BLE001
                pass
            logger.error("add_many SQLite 批量插入失败: %s", e, exc_info=True)
            raise

        # ---- Chroma 批量插入 ----
        if self._collection is not None:
            try:
                self._collection.add_texts(
                    texts=[e.text for e in entries_list],
                    metadatas=[e.to_metadata() for e in entries_list],
                    ids=[e.uuid for e in entries_list],
                )
            except Exception as e:  # noqa: BLE001
                # P0-3: 回滚 SQLite（保证一致性）
                logger.error(
                    "add_many Chroma 批量写入失败，回滚 SQLite (%d 条): %s",
                    len(entries_list), e, exc_info=True,
                )
                uuids = [e.uuid for e in entries_list]
                placeholders = ",".join("?" * len(uuids))
                try:
                    self._conn.execute(
                        f"DELETE FROM entries WHERE uuid IN ({placeholders})",
                        uuids,
                    )
                    self._conn.commit()
                except Exception as rb_e:  # noqa: BLE001
                    logger.error("add_many SQLite 回滚失败: %s", rb_e, exc_info=True)
                raise

        logger.info("add_many: 写入 %d 条索引", len(entries_list))

    def recover_from_crash(self) -> int:
        """P0-3: 从崩溃恢复 —— 用 SQLite 中的全部 text 重建 Chroma 索引。

        Returns:
            重建的条目数
        """
        if self._collection is None:
            logger.warning("recover_from_crash: Chroma 不可用，跳过")
            return 0
        try:
            cur = self._conn.execute(
                f"SELECT {_ENTRY_SELECT_COLS} FROM entries"
            )
            rows = cur.fetchall()
            for row in rows:
                ent = _row_to_entry(row)
                try:
                    self._collection.add_texts(
                        texts=[ent.text],
                        metadatas=[ent.to_metadata()],
                        ids=[ent.uuid],
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("recover 跳过 uuid=%s: %s", ent.uuid, e)
            logger.info("recover_from_crash: 重建 %d 条 Chroma 索引", len(rows))
            return len(rows)
        except Exception as e:  # noqa: BLE001
            logger.error("recover_from_crash 失败: %s", e, exc_info=True)
            return 0

    # ------------------------------------------------------------------
    # 读
    # ------------------------------------------------------------------
    def get(self, uuid: str) -> NameIndexEntry | None:
        cur = self._conn.execute(
            f"SELECT {_ENTRY_SELECT_COLS} FROM entries WHERE uuid=?",
            (uuid,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return _row_to_entry(row)

    def get_character_entries(
        self, character_name: str, category: IndexCategory | None = None
    ) -> list[NameIndexEntry]:
        if category is None:
            cur = self._conn.execute(
                f"SELECT {_ENTRY_SELECT_COLS} FROM entries "
                "WHERE character_name=? ORDER BY chunk_index, char_start",
                (character_name,),
            )
        else:
            cur = self._conn.execute(
                f"SELECT {_ENTRY_SELECT_COLS} FROM entries "
                "WHERE character_name=? AND category=? "
                "ORDER BY chunk_index, char_start",
                (character_name, category.value),
            )
        return [_row_to_entry(r) for r in cur.fetchall()]

    def list_characters(self) -> list[dict[str, Any]]:
        """聚合视图：人物 → 各类计数 + aliases 列表。

        每条返回的 dict 包含：
        - ``character_name``  人物名
        - ``mention_count``   总提及次数
        - ``by_category``     ``{category: count}``
        - ``aliases``         去重后的别名列表（按首次出现顺序）
        """
        cur = self._conn.execute(
            "SELECT character_name, category, COUNT(*) AS n, aliases "
            "FROM entries GROUP BY character_name, category, aliases"
        )
        agg: dict[str, dict[str, Any]] = {}
        alias_seen: dict[str, set[str]] = {}
        for name, cat, n, aliases_csv in cur.fetchall():
            slot = agg.setdefault(
                name, {"character_name": name, "mention_count": 0, "by_category": {}, "aliases": []}
            )
            slot["mention_count"] += n
            slot["by_category"][cat] = slot["by_category"].get(cat, 0) + n
            seen = alias_seen.setdefault(name, set())
            for a in (aliases_csv or "").split(","):
                a = a.strip()
                if a and a != name and a not in seen:
                    seen.add(a)
                    slot["aliases"].append(a)
        return sorted(agg.values(), key=lambda x: -x["mention_count"])

    # ------------------------------------------------------------------
    # 跨 chunk 实体归并
    # ------------------------------------------------------------------
    def merge_characters(self, source: str, target: str) -> int:
        """把 ``source`` 人物的全部索引条目并入 ``target``，返回迁移条目数。

        实现细节：
        1. SQLite ``UPDATE entries SET character_name=target WHERE character_name=source``
        2. aliases 合并：每条迁移后的条目 aliases 追加 ``source``（去重），
           保留原 NER 抓到的同块别名，避免丢失身份信息。
        3. Chroma metadata 同步：best-effort 调 ``update_document`` 刷新
           ``character_name`` / ``aliases``；Chroma 不可用时只更新 SQLite（留 TODO）。

        Args:
            source: 被合并的源人物名（合并后消失）。
            target: 合并目标人物名（合并后保留）。

        Returns:
            实际迁移的条目数（``source == target`` 或 source 不存在时返回 0）。
        """
        if not source or not target or source == target:
            return 0

        # 取出所有 source 条目（含 uuid + aliases），用于 Chroma 同步
        cur = self._conn.execute(
            "SELECT uuid, aliases FROM entries WHERE character_name=?",
            (source,),
        )
        rows = cur.fetchall()
        if not rows:
            return 0

        # 收集 target 现有 aliases（合并后保证 target 也带上 source 名）
        target_aliases_cur = self._conn.execute(
            "SELECT aliases FROM entries WHERE character_name=?",
            (target,),
        )
        target_alias_set: set[str] = set()
        for (csv,) in target_aliases_cur.fetchall():
            for a in (csv or "").split(","):
                a = a.strip()
                if a:
                    target_alias_set.add(a)
        target_alias_set.add(source)  # source 名 → target 的别名

        # 逐条更新：character_name=target + aliases 追加 source（去重）
        for uuid, csv in rows:
            cur_aliases = [a for a in (csv or "").split(",") if a.strip()]
            seen: set[str] = set(cur_aliases)
            merged: list[str] = list(cur_aliases)
            # source 名作为新别名追加（如果还没有）
            if source not in seen:
                merged.append(source)
                seen.add(source)
            # target 名本身不作为别名保留（避免循环引用）
            merged = [a for a in merged if a != target]
            new_csv = ",".join(merged)
            self._conn.execute(
                "UPDATE entries SET character_name=?, aliases=? WHERE uuid=?",
                (target, new_csv, uuid),
            )
        self._conn.commit()

        # ---- Chroma metadata 同步（best-effort）----
        # TODO: 当 Chroma 启用时，metadata 里的 character_name 也应同步刷新；
        #       此处用 update_document 尝试，失败只告警不抛错——recover_from_crash
        #       可在下次启动时由 SQLite 重建 Chroma 兜底。
        if self._collection is not None:
            for uuid, _ in rows:
                try:
                    # 用最新的 SQLite 条目重建 metadata
                    entry = self.get(uuid)
                    if entry is None:
                        continue
                    try:
                        # 优先用 update_document（langchain_chroma 提供）
                        self._collection.update_document(
                            document_id=uuid,  # type: ignore[call-arg]
                            metadata=entry.to_metadata(),
                        )
                    except (AttributeError, TypeError):
                        # 不同版本签名不同；尝试底层 _collection.update
                        underlying = getattr(self._collection, "_collection", None)
                        if underlying is not None:
                            underlying.update(
                                ids=[uuid],
                                metadatas=[entry.to_metadata()],
                            )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Chroma metadata 同步失败 (uuid=%s source=%s target=%s): %s",
                        uuid, source, target, e,
                    )

        return len(rows)

    def search(
        self, query: str, character_name: str | None = None, k: int = 10
    ) -> list[NameIndexEntry]:
        """向量检索 + 过滤。

        若 Chroma 不可用，或嵌入模型是 :class:`HashEmbeddings`（离线伪嵌入，
        向量由文本 hash 派生，相似度无意义），退化到关键词 LIKE 匹配。
        """
        # HashEmbeddings 的向量由文本 hash 派生，相似度无意义；
        # 离线模式下跳过向量检索，直接走 SQLite LIKE，避免返回随机结果
        use_vector = (
            self._collection is not None
            and not isinstance(self._embedding, HashEmbeddings)
        )
        if use_vector:
            try:
                where = {"character_name": character_name} if character_name else None
                docs = self._collection.similarity_search(query, k=k, filter=where)
                # 把 LangChain Document 转回 NameIndexEntry
                out: list[NameIndexEntry] = []
                for d in docs:
                    uuid = d.metadata.get("uuid", "")
                    e = self.get(uuid)
                    if e is not None:
                        out.append(e)
                return out
            except Exception as e:  # noqa: BLE001
                logger.warning("Chroma similarity_search 失败，退化到 LIKE: %s", e)

        # ---- 关键词 fallback（Chroma 不可用 / HashEmbeddings / 向量检索失败）----
        # 离线模式下用 LIKE 匹配原文 text 与角色名 character_name
        like = f"%{query}%"
        sql = (
            f"SELECT {_ENTRY_SELECT_COLS} FROM entries "
            "WHERE (text LIKE ? OR character_name LIKE ?)"
        )
        params: list[Any] = [like, like]
        if character_name:
            sql += " AND character_name=?"
            params.append(character_name)
        sql += " LIMIT ?"
        params.append(k)
        cur = self._conn.execute(sql, params)
        return [_row_to_entry(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # 维护
    # ------------------------------------------------------------------
    def count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM entries")
        return int(cur.fetchone()[0])

    def clear(self) -> None:
        self._conn.execute("DELETE FROM entries")
        # 同步清空缓存表，避免留下「已处理」的虚假记录
        self._conn.execute("DELETE FROM corpus_registry")
        self._conn.execute("DELETE FROM processed_chunks")
        self._conn.commit()
        if self._collection is not None:
            try:
                self._collection.delete_collection()
            except Exception as e:  # noqa: BLE001
                logger.warning("Chroma delete_collection 失败: %s", e)

    # ------------------------------------------------------------------
    # 缓存（chunk-progress-cache）
    # ------------------------------------------------------------------
    def register_corpus(
        self,
        corpus_uuid: str,
        source_path: str,
        content_hash: str,
        total_chunks: int,
    ) -> bool:
        """注册新语料。

        ``corpus_uuid`` 已存在则返回 ``False``（不覆盖，保留原进度）；
        新注册返回 ``True``。
        """
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            "SELECT 1 FROM corpus_registry WHERE corpus_uuid = ?", (corpus_uuid,)
        )
        if cur.fetchone():
            return False  # 已存在
        self._conn.execute(
            "INSERT INTO corpus_registry "
            "(corpus_uuid, source_path, content_hash, total_chunks, "
            "processed_chunks, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 0, ?, ?)",
            (corpus_uuid, source_path, content_hash, total_chunks, now, now),
        )
        self._conn.commit()
        return True

    def is_chunk_processed(self, corpus_uuid: str, chunk_uuid: str) -> str | None:
        """查询 chunk 是否已处理。

        返回已存的 ``content_hash``；未处理返回 ``None``。
        调用方可对比 content_hash 判断 chunk 内容是否变更。
        """
        cur = self._conn.execute(
            "SELECT content_hash FROM processed_chunks "
            "WHERE corpus_uuid = ? AND chunk_uuid = ?",
            (corpus_uuid, chunk_uuid),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def mark_chunk_processed(
        self,
        corpus_uuid: str,
        chunk_uuid: str,
        content_hash: str,
        mention_count: int,
    ) -> None:
        """标记 chunk 为已处理（``INSERT OR REPLACE``）。

        重复调用会刷新 ``content_hash`` / ``processed_at`` / ``mention_count``。
        注意：本方法不刷新 ``corpus_registry.processed_chunks``，调用方应在
        批次结束后调用 :meth:`update_corpus_progress` 重算。
        """
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO processed_chunks "
            "(corpus_uuid, chunk_uuid, content_hash, processed_at, mention_count) "
            "VALUES (?, ?, ?, ?, ?)",
            (corpus_uuid, chunk_uuid, content_hash, now, mention_count),
        )
        self._conn.commit()

    def get_corpus_progress(self, corpus_uuid: str) -> tuple[int, int]:
        """返回 ``(processed_chunks, total_chunks)``。

        未注册返回 ``(0, 0)``。

        ``processed_chunks`` 取 ``processed_chunks`` 表的实时 COUNT（而非
        ``corpus_registry.processed_chunks`` 缓存列），这样在
        :meth:`mark_chunk_processed` 之后无需先调
        :meth:`update_corpus_progress` 即可读到准确进度。``update_corpus_progress``
        仍负责把该 COUNT 同步回缓存列，供批量列举语料进度时使用。
        """
        cur = self._conn.execute(
            "SELECT total_chunks FROM corpus_registry WHERE corpus_uuid = ?",
            (corpus_uuid,),
        )
        row = cur.fetchone()
        if not row:
            return (0, 0)
        total = int(row[0])
        cur2 = self._conn.execute(
            "SELECT COUNT(*) FROM processed_chunks WHERE corpus_uuid = ?",
            (corpus_uuid,),
        )
        processed = int(cur2.fetchone()[0])
        return (processed, total)

    def update_corpus_progress(self, corpus_uuid: str) -> None:
        """重算 ``corpus_registry.processed_chunks`` 并刷新 ``updated_at``。

        以 ``processed_chunks`` 表的实际行数为准，避免增量更新错位。
        """
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM processed_chunks WHERE corpus_uuid = ?",
            (corpus_uuid,),
        )
        n = int(cur.fetchone()[0])
        self._conn.execute(
            "UPDATE corpus_registry SET processed_chunks = ?, updated_at = ? "
            "WHERE corpus_uuid = ?",
            (n, now, corpus_uuid),
        )
        self._conn.commit()

    def delete_chunk_entries(
        self, corpus_uuid: str, source: str, chunk_index: int
    ) -> int:
        """删除指定 chunk 的所有旧索引条目（SQLite + Chroma）。

        通过 ``(corpus_uuid, source, chunk_index)`` 三元组定位——复用
        ``entries`` 表既有列，无需新增 ``chunk_uuid`` 列。

        返回删除条数。Chroma 删除失败只告警不抛错（SQLite 已删，下次重建
        可由 :meth:`recover_from_crash` 兜底）。
        """
        cur = self._conn.execute(
            "SELECT uuid FROM entries "
            "WHERE corpus_uuid = ? AND source = ? AND chunk_index = ?",
            (corpus_uuid, source, chunk_index),
        )
        uuids = [row[0] for row in cur.fetchall()]
        if not uuids:
            return 0
        # 删 SQLite
        self._conn.execute(
            "DELETE FROM entries "
            "WHERE corpus_uuid = ? AND source = ? AND chunk_index = ?",
            (corpus_uuid, source, chunk_index),
        )
        self._conn.commit()
        # 删 Chroma
        if self._collection is not None:
            try:
                self._collection.delete(ids=uuids)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Chroma 删除失败 (chunk_index=%d): %s", chunk_index, e
                )
        return len(uuids)

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception as e:  # noqa: BLE001
            logger.warning("SQLite close 失败: %s", e)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _row_to_entry(row: tuple) -> NameIndexEntry:
    (
        uuid,
        character_name,
        category,
        source,
        chunk_index,
        char_start,
        char_end,
        text,
        aliases,
        corpus_uuid,
        relation_to,
        co_mentioned,
    ) = row
    return NameIndexEntry(
        uuid=uuid,
        character_name=character_name,
        category=IndexCategory(category),
        source=source,
        chunk_index=int(chunk_index),
        char_start=int(char_start),
        char_end=int(char_end),
        text=text,
        aliases=[a for a in aliases.split(",") if a],
        corpus_uuid=corpus_uuid or "",
        relation_to=relation_to or None,
        co_mentioned=[c for c in (co_mentioned or "").split(",") if c],
    )
