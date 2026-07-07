"""蒸馏桥接：从 :class:`CharacterProfile` 重建临时语料目录 → 调 :class:`PersonaDistiller`。

落盘结构::

    <workdir>/<persona_id>/
        speech.md        # 该角色说过的对话
        appearance.md    # 外貌描述
        events.md        # 相关事件

随后整目录喂给现有 :func:`PersonaDistiller.distill`，复用 extractor → synthesizer →
skill_designer → dialogue_writer 全流程。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from persona_distillation.config import DistillationConfig
from persona_distillation.intake.schemas import CharacterProfile
from persona_distillation.schemas import DistillationResult


_NAME_RE = re.compile(r"^[a-z0-9\u4e00-\u9fa5][a-z0-9\u4e00-\u9fa5-]{0,63}$")


def slugify(name: str, fallback: str = "persona") -> str:
    """人名 → persona_id。

    支持中英混排：
    - 英文 / 数字 / 连字符：保留
    - 中日韩统一表意文字：保留
    - 其它字符 → 折叠为 ``-``
    """
    import re as _re

    n = name.strip().lower()
    # CJK 字符 + 英文 + 数字 + 连字符保留，其它折叠为 -
    n = _re.sub(r"[\s_]+", "-", n)
    n = _re.sub(r"[^\u4e00-\u9fa5a-z0-9-]+", "-", n)
    n = _re.sub(r"-+", "-", n).strip("-")
    if not n or not _NAME_RE.match(n):
        n = fallback
    return n


# ---------------------------------------------------------------------------
# 语料落盘
# ---------------------------------------------------------------------------
def _format_block(
    title: str, items: list, anonymizer: Any = None
) -> str:
    """把索引条目列表格式化为 markdown 块。

    ``anonymizer`` 为可选的文本脱名函数（``str -> str``）——传入时对每条
    ``e.text`` 应用脱名后再写入，避免 evidence 里的其他人名泄漏给 synthesizer。
    """
    if not items:
        return f"# {title}\n\n（无）\n"
    lines = [f"# {title}", ""]
    for i, e in enumerate(items, 1):
        lines.append(f"## 条目 {i}")
        body = anonymizer(e.text) if anonymizer else e.text
        lines.append(f"> {body}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 条件脱名（Issue #19）：把 evidence 里的其他人名替换为角色标签
# ---------------------------------------------------------------------------
def _all_excerpts(profile: CharacterProfile) -> list:
    """档案下全部索引条目（speech + appearance + event）。"""
    return (
        list(profile.speech_excerpts)
        + list(profile.appearance_excerpts)
        + list(profile.event_excerpts)
    )


def _collect_other_characters(
    profile: CharacterProfile,
    target: str,
    index_store: Any,
) -> tuple[dict[str, str | None], dict[str, list[str]]]:
    """收集目标人物之外的其他人物名 → (relation_to, aliases)。

    数据源：
    1. profile 各 excerpts 的 ``co_mentioned`` 字段——无 IndexStore 时也能脱名（兜底）
    2. ``index_store.list_characters()``（若提供）——补全其他人物 + relation_to + aliases
    """
    name_to_relation: dict[str, str | None] = {}
    name_to_aliases: dict[str, list[str]] = {}

    target_names = {target, *profile.aliases}

    # 1) 从 profile excerpts 的 co_mentioned 收集（脱名兜底）
    for e in _all_excerpts(profile):
        for name in getattr(e, "co_mentioned", []) or []:
            if name and name != target and name not in target_names:
                name_to_relation.setdefault(name, None)
                name_to_aliases.setdefault(name, [])

    # 2) 从 IndexStore 补全其他人物 + relation_to + aliases
    if index_store is not None:
        try:
            for ch in index_store.list_characters():
                name = ch.get("character_name")
                if not name or name == target or name in target_names:
                    continue
                name_to_relation.setdefault(name, None)
                als = ch.get("aliases") or []
                name_to_aliases.setdefault(name, list(als))
                # 查询该人物条目，取第一个非 None 的 relation_to（与主人物的关系）
                try:
                    entries = index_store.get_character_entries(name)
                except Exception:  # noqa: BLE001
                    entries = []
                for ent in entries:
                    if getattr(ent, "relation_to", None):
                        name_to_relation[name] = ent.relation_to
                        break
        except Exception:  # noqa: BLE001
            pass

    return name_to_relation, name_to_aliases


def _build_labels(
    name_to_relation: dict[str, str | None],
) -> dict[str, str]:
    """根据 relation_to 生成 ``[学生A]`` / ``[人物B]`` 风格标签。

    同一关系类型内字母递增（A、B、C...）；``relation_to`` 为 None 时统一用「人物」。
    """
    relation_counter: dict[str, int] = {}
    name_to_label: dict[str, str] = {}
    for name, rel in name_to_relation.items():
        rel_key = rel if rel else "人物"
        idx = relation_counter.get(rel_key, 0)
        letter = chr(ord("A") + idx)
        relation_counter[rel_key] = idx + 1
        name_to_label[name] = f"[{rel_key}{letter}]"
    return name_to_label


def _build_replacements(
    name_to_label: dict[str, str],
    name_to_aliases: dict[str, list[str]],
    target_names: set[str],
) -> list[tuple[str, str]]:
    """构造 ``(名字, 标签)`` 替换列表，长名字优先，避免短名先替换长名子串。

    替换范围：其他人物的规范化名 + 别名（别名长度需 ≥2，避免误伤单字词）。
    目标人物的名 / 别名一律排除，确保目标人物名原样保留。
    """
    mapping: dict[str, str] = {}
    for name, label in name_to_label.items():
        if name and name not in target_names:
            mapping[name] = label
    for name, label in name_to_label.items():
        for alias in name_to_aliases.get(name, []):
            if (
                alias
                and len(alias) >= 2
                and alias not in target_names
                and alias not in mapping
            ):
                mapping[alias] = label
    # 长名字优先替换，避免「荒川」先把「荒川善次」截断
    return sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True)


def _anonymize_text(text: str, replacements: list[tuple[str, str]]) -> str:
    """按替换列表脱名（``replacements`` 已按长度降序，先长后短）。"""
    result = text
    for name, label in replacements:
        if name:
            result = result.replace(name, label)
    return result


def _count_co_occurrences(name: str, excerpts: list) -> int:
    """统计其他人物与目标的共现次数。

    优先用 ``co_mentioned`` 字段；若全部为空，退化为在 evidence text 中计数。
    """
    co_count = sum(
        1 for e in excerpts if name in (getattr(e, "co_mentioned", []) or [])
    )
    if co_count > 0:
        return co_count
    return sum(getattr(e, "text", "").count(name) for e in excerpts)


def rebuild_corpus_dir(
    profile: CharacterProfile,
    workdir: Path,
    *,
    anonymize_relations: bool = True,
    target_name: str | None = None,
    index_store: Any = None,
) -> Path:
    """从档案重建临时语料目录，返回该目录路径。

    Parameters:
        profile: 人物档案
        workdir: 工作目录（语料目录落在 ``workdir/<pid>/``）
        anonymize_relations: 是否对其他人物名字做条件脱名（默认 True）。
            开启后保留目标人物名，其他人物名替换为 ``[学生A]`` / ``[人物B]``
            等角色标签，并额外产出 ``relationships.json`` 关系图谱。可关闭以
            恢复原样落盘。
        target_name: 目标人物名（脱名时保留）；``None`` 时取 ``profile.character_name``。
        index_store: 可选的 :class:`IndexStore`，用于补全其他人物名 + relation_to。
            不传时仅依据 profile excerpts 的 ``co_mentioned`` 字段脱名（兜底）。
    """
    pid = slugify(profile.character_name)
    out = workdir / pid
    out.mkdir(parents=True, exist_ok=True)

    target = target_name or profile.character_name

    # ---- 条件脱名预处理 ----
    replacements: list[tuple[str, str]] = []
    relations_doc: dict[str, Any] | None = None
    if anonymize_relations:
        target_names = {target, *profile.aliases}
        name_to_relation, name_to_aliases = _collect_other_characters(
            profile, target, index_store
        )
        name_to_label = _build_labels(name_to_relation)
        replacements = _build_replacements(
            name_to_label, name_to_aliases, target_names
        )
        excerpts = _all_excerpts(profile)
        relations = [
            {
                "name": n,
                "relation": rel,
                "co_occurrences": _count_co_occurrences(n, excerpts),
            }
            for n, rel in name_to_relation.items()
        ]
        relations.sort(key=lambda x: -x["co_occurrences"])
        relations_doc = {"target": target, "relations": relations}

    def _anon(text: str) -> str:
        return _anonymize_text(text, replacements) if replacements else text

    (out / "speech.md").write_text(
        _format_block(
            f"{profile.character_name} · 对话语录",
            profile.speech_excerpts,
            anonymizer=_anon,
        ),
        encoding="utf-8",
    )
    (out / "appearance.md").write_text(
        _format_block(
            f"{profile.character_name} · 外貌描述",
            profile.appearance_excerpts,
            anonymizer=_anon,
        ),
        encoding="utf-8",
    )
    (out / "events.md").write_text(
        _format_block(
            f"{profile.character_name} · 相关事件",
            profile.event_excerpts,
            anonymizer=_anon,
        ),
        encoding="utf-8",
    )
    # 人物总结也做脱名，避免 summary 泄漏其他人名
    (out / "summary.md").write_text(
        _anon(f"# {profile.character_name} · 档案摘要\n\n{profile.summary}\n"),
        encoding="utf-8",
    )

    # ---- 关系图谱产出（Issue #19）----
    if relations_doc is not None:
        (out / "relationships.json").write_text(
            json.dumps(relations_doc, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return out


# ---------------------------------------------------------------------------
# 蒸馏入口
# ---------------------------------------------------------------------------
def distill_character(
    profile: CharacterProfile,
    cfg: DistillationConfig,
    workdir: Path,
    *,
    distiller_factory: Any = None,
) -> DistillationResult:
    """从档案启动蒸馏，返回 :class:`DistillationResult`。

    Parameters:
        profile: 人物档案
        cfg: 蒸馏配置（``persona_id`` 会被覆盖为该角色名）
        workdir: 工作目录（中间产物 + 最终产物都落这里）
        distiller_factory: 可选，用于注入自定义的 ``PersonaDistiller``（默认用真品）
    """
    pid = slugify(profile.character_name)
    cfg = cfg.model_copy(update={"persona_id": pid})

    corpus_dir = rebuild_corpus_dir(profile, workdir)

    if distiller_factory is None:
        from persona_distillation.pipeline import PersonaDistiller

        distiller_factory = PersonaDistiller

    distiller = distiller_factory(cfg)
    output_dir = workdir / "distilled" / pid
    return distiller.distill(
        corpus_dir,
        persona_id=pid,
        output_dir=output_dir,
    )
