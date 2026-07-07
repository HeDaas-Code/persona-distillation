"""intake —— 人格蒸馏的预处理子包。

提供：
- :class:`IndexStore`         —— Chroma + SQLite 双写的人名索引
- :class:`CharacterProfile`   —— 聚合后的人物档案
- :func:`extract_names_from_chunk` —— LLM-NER 识别分块中的人物与类别
- :func:`build_profile`       —— 从索引聚合人物档案
- :func:`distill_character`   —— 桥接 PersonaDistiller

典型用法::

    from persona_distillation.intake import (
        IndexStore, extract_names_from_chunk, build_profile, distill_character,
    )

    store = IndexStore(Path("./index"))
    for chunk in chunks:
        mentions = extract_names_from_chunk(chunk, source=..., llm=...)
        for m in mentions:
            store.add(NameIndexEntry.from_mention(m, chunk, source))

    profile = build_profile("荒川善次", store, reranker=..., llm=...)
    result = distill_character(profile, cfg, workdir=Path("./out"))
"""
from __future__ import annotations

from persona_distillation.intake.schemas import (
    CharacterProfile,
    IndexCategory,
    NameExtractionResult,
    NameIndexEntry,
    NameMention,
)
from persona_distillation.intake.index_store import IndexStore
from persona_distillation.intake.name_extractor import extract_names_from_chunk
from persona_distillation.intake.profile_builder import build_profile
from persona_distillation.intake.bridge import distill_character, rebuild_corpus_dir
from persona_distillation.intake.embedder import (
    HashEmbeddings,
    build_embedder,
    build_reranker,
)
from persona_distillation.intake.dna_extractor import (
    backfill_dna_from_system_prompt,
    extract_dna_from_system_prompt,
)
from persona_distillation.intake.tools import (
    IntakeContext,
    build_intake_context,
    build_intake_tools,
)
from persona_distillation.intake.entity_resolver import (
    CharacterSignals,
    ResolveResult,
    resolve_entities,
)

__all__ = [
    "IndexCategory",
    "NameMention",
    "NameExtractionResult",
    "NameIndexEntry",
    "CharacterProfile",
    "IndexStore",
    "extract_names_from_chunk",
    "build_profile",
    "rebuild_corpus_dir",
    "distill_character",
    "HashEmbeddings",
    "build_embedder",
    "build_reranker",
    "extract_dna_from_system_prompt",
    "backfill_dna_from_system_prompt",
    "IntakeContext",
    "build_intake_context",
    "build_intake_tools",
    # 跨 chunk 实体归并
    "CharacterSignals",
    "ResolveResult",
    "resolve_entities",
]
