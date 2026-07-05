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
        "corpus_uuid TEXT NOT NULL DEFAULT ''"
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
                "char_start, char_end, text, aliases, corpus_uuid) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
        n = 0
        for e in entries:
            self.add(e)
            n += 1
        logger.info("add_many: 写入 %d 条索引", n)

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
                "SELECT uuid, character_name, category, source, chunk_index, "
                "char_start, char_end, text, aliases, corpus_uuid FROM entries"
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
            "SELECT uuid, character_name, category, source, chunk_index, "
            "char_start, char_end, text, aliases, corpus_uuid FROM entries "
            "WHERE uuid=?",
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
                "SELECT uuid, character_name, category, source, chunk_index, "
                "char_start, char_end, text, aliases, corpus_uuid FROM entries "
                "WHERE character_name=? ORDER BY chunk_index, char_start",
                (character_name,),
            )
        else:
            cur = self._conn.execute(
                "SELECT uuid, character_name, category, source, chunk_index, "
                "char_start, char_end, text, aliases, corpus_uuid FROM entries "
                "WHERE character_name=? AND category=? "
                "ORDER BY chunk_index, char_start",
                (character_name, category.value),
            )
        return [_row_to_entry(r) for r in cur.fetchall()]

    def list_characters(self) -> list[dict[str, Any]]:
        """聚合视图：人物 → 各类计数。"""
        cur = self._conn.execute(
            "SELECT character_name, category, COUNT(*) AS n FROM entries "
            "GROUP BY character_name, category"
        )
        agg: dict[str, dict[str, Any]] = {}
        for name, cat, n in cur.fetchall():
            slot = agg.setdefault(
                name, {"character_name": name, "mention_count": 0, "by_category": {}}
            )
            slot["mention_count"] += n
            slot["by_category"][cat] = n
        return sorted(agg.values(), key=lambda x: -x["mention_count"])

    def search(
        self, query: str, character_name: str | None = None, k: int = 10
    ) -> list[NameIndexEntry]:
        """向量检索 + 过滤。

        若 Chroma 不可用，退化到关键词 LIKE 匹配。
        """
        if self._collection is not None:
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

        # ---- 关键词 fallback ----
        sql = (
            "SELECT uuid, character_name, category, source, chunk_index, "
            "char_start, char_end, text, aliases, corpus_uuid FROM entries "
            "WHERE text LIKE ?"
        )
        params: list[Any] = [f"%{query}%"]
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
    )
