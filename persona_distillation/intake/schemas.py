"""intake 子包的结构化 schema。

包含：
- :class:`IndexCategory`    —— 索引类别枚举（speech/appearance/event）
- :class:`NameMention`      —— LLM-NER 单条提及（原始结构化输出）
- :class:`NameExtractionResult` —— LLM-NER 单分块结果
- :class:`NameIndexEntry`   —— 入库后的索引条目（含 uuid）
- :class:`CharacterProfile` —— 聚合后的人物档案
"""
from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class IndexCategory(str, Enum):
    """索引类别。

    - ``speech``     该角色说过的话（直接引语 / 对话）
    - ``appearance`` 关于该角色外貌的描述
    - ``event``      与该角色相关的事件
    """

    SPEECH = "speech"
    APPEARANCE = "appearance"
    EVENT = "event"


class NameMention(BaseModel):
    """LLM-NER 从单分块里识别出的一条人物提及。"""

    name: str = Field(..., description="规范化后的人名（消歧后的统一标识）")
    aliases: list[str] = Field(
        default_factory=list, description="同义称谓（如「老师」「荒川」）"
    )
    category: IndexCategory = Field(..., description="提及类别")
    evidence: str = Field(..., description="原文引文，≤120 字")
    char_start: int = Field(0, description="在所属分块内的相对起止")
    char_end: int = Field(0, description="在所属分块内的相对起止")
    # 同一 evidence 中同时出现的人物名（关系提取用）
    co_mentioned: list[str] = Field(
        default_factory=list,
        description="同一 evidence 中同时出现的人物名（关系提取用）",
    )
    # 与主人物的关系（如「学生」「上级」「对手」「亲人」）
    relation_to: str | None = Field(
        None, description="与主人物的关系（如「学生」「上级」「对手」「亲人」）"
    )


class NameExtractionResult(BaseModel):
    """LLM-NER 对单分块的产出。"""

    mentions: list[NameMention] = Field(default_factory=list)


class NerBatchItem(BaseModel):
    """intake_ner SubAgent 单个 chunk 的批量产出。

    在 ``NameExtractionResult`` 基础上多带一份 ``chunk_meta``——``index_characters``
    Python 工具依赖这份元信息（source / chunk_index / char_start / corpus_uuid /
    content_hash）才能把 mention 重建为带定位的 :class:`NameIndexEntry`。
    SubAgent 必须原样透传 chunk_meta，不要改写。
    """

    chunk_meta: dict[str, Any] = Field(
        default_factory=dict,
        description="chunk 元信息，原样透传：source / chunk_index / char_start / "
                    "corpus_uuid / content_hash / total_chunks",
    )
    mentions: list[NameMention] = Field(default_factory=list)


class NerBatchResult(BaseModel):
    """intake_ner SubAgent 批量产出的包装（response_format 用）。

    一次性接收全部 chunk，对每个 chunk 输出一个 :class:`NerBatchItem`。
    """

    items: list[NerBatchItem] = Field(default_factory=list)


class NameIndexEntry(BaseModel):
    """入库索引条目（带 uuid 与定位信息）。

    两类 UUID 不要混淆：
    - ``uuid``         —— 随机 v4，每条索引条目独立标识（用于点查/删除）
    - ``corpus_uuid``  —— 确定性 v5，由 LoadedDoc.corpus_uuid 透传而来，
                          标识「这条索引来自哪篇语料」，便于按语料聚合/失效缓存
    """

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    character_name: str
    aliases: list[str] = Field(default_factory=list)
    category: IndexCategory
    text: str = Field(..., description="原文片段（已 copy 到该条独立存储）")
    source: str = Field(..., description="来源文件 relpath")
    corpus_uuid: str = Field(default="", description="所属语料的确定性 UUID")
    char_start: int = 0
    char_end: int = 0
    chunk_index: int = 0
    embedding: list[float] | None = None
    # 与主人物的关系（关系提取 + 条件脱名用）
    relation_to: str | None = Field(None, description="与主人物的关系")
    # 同一 evidence 中同时出现的人物名（关系提取 + 条件脱名用）
    co_mentioned: list[str] = Field(
        default_factory=list, description="同一 evidence 中同时出现的人物名"
    )

    @classmethod
    def from_mention(
        cls,
        m: NameMention,
        *,
        chunk_index: int,
        source: str,
        global_char_start: int = 0,
        corpus_uuid: str = "",
    ) -> "NameIndexEntry":
        """从 LLM 输出 + 分块定位构造索引条目。

        ``corpus_uuid`` 由调用方从 ``LoadedDoc.corpus_uuid`` 透传，
        不传则留空（向后兼容旧调用点）。

        ``relation_to`` / ``co_mentioned`` 从 :class:`NameMention` 透传，
        供 ``rebuild_corpus_dir`` 条件脱名时生成角色标签。
        """
        return cls(
            character_name=m.name,
            aliases=list(m.aliases),
            category=m.category,
            text=m.evidence,
            source=source,
            corpus_uuid=corpus_uuid,
            chunk_index=chunk_index,
            char_start=global_char_start + m.char_start,
            char_end=global_char_start + m.char_end,
            relation_to=m.relation_to,
            co_mentioned=list(m.co_mentioned),
        )

    def to_metadata(self) -> dict[str, Any]:
        """转为 Chroma / SQLite 通用元数据。"""
        return {
            "uuid": self.uuid,
            "character_name": self.character_name,
            "category": self.category.value,
            "source": self.source,
            "corpus_uuid": self.corpus_uuid,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "text": self.text,
            "aliases": ",".join(self.aliases),
            "relation_to": self.relation_to or "",
            "co_mentioned": ",".join(self.co_mentioned),
        }


class CharacterProfile(BaseModel):
    """聚合后的人物档案——主理人 Agent 让用户从这些条目里选择。"""

    character_name: str
    aliases: list[str] = Field(default_factory=list)
    mention_count: int = 0
    speech_count: int = 0
    appearance_count: int = 0
    event_count: int = 0
    # 各类的 top-k 索引条目（来自 index_store + rerank）
    speech_excerpts: list[NameIndexEntry] = Field(default_factory=list)
    appearance_excerpts: list[NameIndexEntry] = Field(default_factory=list)
    event_excerpts: list[NameIndexEntry] = Field(default_factory=list)
    # LLM 生成的总结
    summary: str = ""
