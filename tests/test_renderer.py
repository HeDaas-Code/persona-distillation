"""``renderer`` 单元测试。

覆盖人格卡渲染器的文件写出与 Markdown 生成逻辑：
- ``render_persona_card``: JSON 文件落盘、右侧三块默认值注入
- ``_card_markdown``: Markdown 渲染、可选特质摘要段落
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from persona_distillation.renderer import _card_markdown, render_persona_card
from persona_distillation.schemas import (
    Distillate,
    DistillationResult,
    PersonaCard,
    PersonaSignal,
    PersonaSkill,
    PresetDialogue,
    SignalCategory,
)


def _mini_card(**overrides) -> PersonaCard:
    base = dict(
        persona_id="test_id",
        display_name="测试角色",
        system_prompt="[身份] 测试\n[性格] 温和",
        error_reply="抱歉，出错了。",
        tags=["测试", "温和"],
    )
    base.update(overrides)
    return PersonaCard(**base)


def _mini_result(
    card: PersonaCard | None = None,
    skills: list[PersonaSkill] | None = None,
    dialogues: list[PresetDialogue] | None = None,
    distillates: list[Distillate] | None = None,
) -> DistillationResult:
    return DistillationResult(
        persona_card=card or _mini_card(),
        skills=skills or [],
        preset_dialogues=dialogues or [],
        distillates=distillates or [],
    )


# ---------------------------------------------------------------------------
# render_persona_card
# ---------------------------------------------------------------------------


class TestRenderPersonaCard:
    def test_creates_output_directory(self):
        result = _mini_result()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "deep" / "nested"
            render_persona_card(result, out)
            assert out.is_dir()

    def test_writes_persona_card_json(self):
        result = _mini_result()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            render_persona_card(result, out)
            json_path = out / "persona_card.json"
            assert json_path.exists()
            data = json.loads(json_path.read_text(encoding="utf-8"))
            assert data["persona_id"] == "test_id"
            assert data["display_name"] == "测试角色"

    def test_writes_persona_card_md(self):
        result = _mini_result()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            render_persona_card(result, out)
            md_path = out / "persona_card.md"
            assert md_path.exists()
            content = md_path.read_text(encoding="utf-8")
            assert "人格卡" in content
            assert "test_id" in content

    def test_injects_tools_default(self):
        result = _mini_result()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            render_persona_card(result, out)
            data = json.loads((out / "persona_card.json").read_text(encoding="utf-8"))
            assert data["tools"] == {"mode": "default_all"}

    def test_injects_skills_with_paths(self):
        skill = PersonaSkill(
            name="test-skill",
            description="测试技能",
            when_to_use="测试时使用",
        )
        result = _mini_result(skills=[skill])
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            render_persona_card(result, out)
            data = json.loads((out / "persona_card.json").read_text(encoding="utf-8"))
            assert data["skills"]["mode"] == "specified"
            assert "skills/test-skill/" in data["skills"]["paths"]
            assert "test-skill" in data["skills"]["names"]

    def test_injects_preset_dialogues(self):
        dialogue = PresetDialogue(
            user="你好",
            assistant="你好呀",
            intent="寒暄",
        )
        result = _mini_result(dialogues=[dialogue])
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            render_persona_card(result, out)
            data = json.loads((out / "persona_card.json").read_text(encoding="utf-8"))
            assert len(data["preset_dialogues"]) == 1
            assert data["preset_dialogues"][0]["user"] == "你好"

    def test_returns_output_path(self):
        result = _mini_result()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            returned = render_persona_card(result, out)
            assert returned == out / "persona_card.json"


# ---------------------------------------------------------------------------
# _card_markdown
# ---------------------------------------------------------------------------


class TestCardMarkdown:
    def test_contains_display_name(self):
        card = _mini_card()
        md = _card_markdown(card)
        assert "测试角色" in md

    def test_contains_persona_id(self):
        card = _mini_card()
        md = _card_markdown(card)
        assert "test_id" in md

    def test_contains_tags(self):
        card = _mini_card()
        md = _card_markdown(card)
        assert "测试" in md
        assert "温和" in md

    def test_no_tags_shows_none(self):
        card = _mini_card(tags=[])
        md = _card_markdown(card)
        assert "（无）" in md

    def test_contains_system_prompt(self):
        card = _mini_card()
        md = _card_markdown(card)
        assert "测试" in md
        assert "温和" in md

    def test_contains_error_reply(self):
        card = _mini_card()
        md = _card_markdown(card)
        assert "抱歉，出错了" in md

    def test_traits_summary_included_when_present(self):
        card = _mini_card(traits_summary="此人温和且有智慧。")
        md = _card_markdown(card)
        assert "特质摘要" in md
        assert "温和且有智慧" in md

    def test_traits_summary_excluded_when_absent(self):
        card = _mini_card(traits_summary="")
        md = _card_markdown(card)
        assert "特质摘要" not in md