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
    # 用 model_dump 拿到全部字段（包括 DNA 五层），再叠加右侧三块选择默认值
    card_payload = card.model_dump(exclude_none=True)
    # 右侧三块选择默认值，便于直接导入平台
    card_payload["tools"] = {"mode": "default_all"}
    card_payload["skills"] = {
        "mode": "specified",
        "paths": [f"skills/{s.name}/" for s in result.skills],
        "names": [s.name for s in result.skills],
    }
    card_payload["preset_dialogues"] = [d.model_dump() for d in result.preset_dialogues]
    card_payload["metadata"] = result.metadata
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
