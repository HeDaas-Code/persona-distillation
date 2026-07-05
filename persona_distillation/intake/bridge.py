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
def _format_block(title: str, items: list) -> str:
    if not items:
        return f"# {title}\n\n（无）\n"
    lines = [f"# {title}", ""]
    for i, e in enumerate(items, 1):
        lines.append(f"## 条目 {i}")
        lines.append(f"> {e.text}")
        lines.append("")
    return "\n".join(lines)


def rebuild_corpus_dir(profile: CharacterProfile, workdir: Path) -> Path:
    """从档案重建临时语料目录，返回该目录路径。"""
    pid = slugify(profile.character_name)
    out = workdir / pid
    out.mkdir(parents=True, exist_ok=True)

    (out / "speech.md").write_text(
        _format_block(f"{profile.character_name} · 对话语录", profile.speech_excerpts),
        encoding="utf-8",
    )
    (out / "appearance.md").write_text(
        _format_block(f"{profile.character_name} · 外貌描述", profile.appearance_excerpts),
        encoding="utf-8",
    )
    (out / "events.md").write_text(
        _format_block(f"{profile.character_name} · 相关事件", profile.event_excerpts),
        encoding="utf-8",
    )
    # 额外：人物总结（供 synthesizer 参考）
    (out / "summary.md").write_text(
        f"# {profile.character_name} · 档案摘要\n\n{profile.summary}\n",
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
    cfg = cfg.model_copy(update={"persona_id": pid}) if hasattr(cfg, "model_copy") else cfg
    # dataclass 替换
    import dataclasses

    if dataclasses.is_dataclass(cfg):
        cfg = dataclasses.replace(cfg, persona_id=pid)

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
