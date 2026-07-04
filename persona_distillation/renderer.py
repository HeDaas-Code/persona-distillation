"""人格卡渲染器：把 DistillationResult 写成与角色卡暗色界面对齐的产物。

落盘清单：
- ``persona_card.json``        —— 左侧面板三字段 + 元信息
- ``persona_card.md``          —— 人类可读版系统提示词
- ``preset_dialogues.json``    —— 右侧"预设对话"
- ``distillates.jsonl``        —— 中间蒸馏液（可审计）
- ``distillation_result.json`` —— 除蒸馏液外的完整结果
- ``skills/<persona_id>-*/SKILL.md`` —— 右侧"Skills"（见 skills_writer）
"""
from __future__ import annotations

import json
from pathlib import Path

from persona_distillation.schemas import DistillationResult, PersonaCard


def render_persona_card(result: DistillationResult, out: Path) -> Path:
    """写出人格卡相关文件。"""
    out.mkdir(parents=True, exist_ok=True)
    card = result.persona_card

    # 1. 角色卡 JSON（字段对齐界面左侧）
    card_payload = {
        "persona_id": card.persona_id,
        "display_name": card.display_name,
        "system_prompt": card.system_prompt,
        "error_reply": card.error_reply,
        "tags": card.tags,
        "traits_summary": card.traits_summary,
        # 右侧三块选择默认值，便于直接导入平台
        "tools": {"mode": "default_all"},
        "skills": {
            "mode": "specified",
            "paths": [f"skills/{s.name}/" for s in result.skills],
            "names": [s.name for s in result.skills],
        },
        "preset_dialogues": [d.model_dump() for d in result.preset_dialogues],
        "metadata": result.metadata,
    }
    (out / "persona_card.json").write_text(
        json.dumps(card_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 2. 人类可读版系统提示词
    (out / "persona_card.md").write_text(_card_markdown(card), encoding="utf-8")

    return out / "persona_card.json"


def _card_markdown(card: PersonaCard) -> str:
    lines = [
        f"# 人格卡：{card.display_name or card.persona_id}",
        "",
        f"- **人格ID**: `{card.persona_id}`",
        f"- **标签**: {', '.join(card.tags) or '（无）'}",
        f"- **自定义报错回复**: {card.error_reply}",
        "",
        "## 系统提示词",
        "",
        "```",
        card.system_prompt,
        "```",
    ]
    if card.traits_summary:
        lines += ["", "## 特质摘要", "", card.traits_summary]
    return "\n".join(lines) + "\n"
