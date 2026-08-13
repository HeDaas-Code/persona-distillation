"""``skills_writer`` 单元测试：覆盖名称清洗、去重、SKILL.md 输出完整性。

``skills_writer.py`` 是 PersonaSkill 落盘层，但此前只有 2 处间接引用。
本测试直接锁死其核心契约：

1. ``_sanitize_skill_name``：非法字符清洗、空名 fallback、长度 64 上限
2. ``_ensure_unique``：重名时 -2、-3… 递增直到唯一
3. ``write_skills``：多 skill 批量写出、pid 前缀、最终 SKILL.md 路径列表
4. ``_skill_md``：YAML frontmatter 格式、DNA 五层渲染、HonestBoundary 兜底

跑法：``python -m pytest tests/test_skills_writer.py -v``
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from persona_distillation import skills_writer
from persona_distillation.schemas import (
    AntiPattern,
    DecisionHeuristic,
    ExpressionDNA,
    HonestBoundary,
    MentalModel,
    PersonaSkill,
    VerificationResult,
)


# ======================================================================
# 1. _ensure_unique —— 去重递增
# ======================================================================


class TestEnsureUnique:
    def test_no_existing_returns_original(self) -> None:
        assert skills_writer._ensure_unique("alpha", None) == "alpha"
        assert skills_writer._ensure_unique("alpha", []) == "alpha"

    def test_name_not_in_existing_returns_original(self) -> None:
        assert skills_writer._ensure_unique("alpha", ["beta", "gamma"]) == "alpha"

    def test_first_collision_adds_minus_2(self) -> None:
        assert skills_writer._ensure_unique("alpha", ["alpha"]) == "alpha-2"

    def test_second_collision_adds_minus_3(self) -> None:
        assert skills_writer._ensure_unique("alpha", ["alpha", "alpha-2"]) == "alpha-3"

    def test_skips_existing_numbers(self) -> None:
        result = skills_writer._ensure_unique("alpha", ["alpha", "alpha-2", "alpha-3", "alpha-5"])
        assert result == "alpha-4"

    def test_set_input_works(self) -> None:
        assert skills_writer._ensure_unique("a", {"a"}) == "a-2"


# ======================================================================
# 2. _sanitize_skill_name —— 名称清洗
# ======================================================================


class TestSanitizeSkillName:
    def test_already_valid_unchanged(self) -> None:
        assert skills_writer._sanitize_skill_name("my-skill-1") == "my-skill-1"

    def test_uppercase_lowercased(self) -> None:
        assert skills_writer._sanitize_skill_name("MY-SKILL") == "my-skill"

    def test_spaces_replaced_with_hyphens(self) -> None:
        assert skills_writer._sanitize_skill_name("my skill name") == "my-skill-name"

    def test_special_chars_stripped(self) -> None:
        assert skills_writer._sanitize_skill_name("my@skill#name!") == "my-skill-name"

    def test_repeated_hyphens_collapsed(self) -> None:
        assert skills_writer._sanitize_skill_name("my---skill") == "my-skill"

    def test_leading_trailing_hyphens_stripped(self) -> None:
        assert skills_writer._sanitize_skill_name("---my-skill---") == "my-skill"

    def test_empty_after_cleaning_uses_fallback(self) -> None:
        assert skills_writer._sanitize_skill_name("!!!", fallback="fb") == "fb"
        assert skills_writer._sanitize_skill_name("", fallback="fb") == "fb"

    def test_existing_dedup_after_sanitize(self) -> None:
        result = skills_writer._sanitize_skill_name(
            "my skill", fallback="skill", existing=["my-skill"]
        )
        assert result == "my-skill-2"

    def test_valid_length_limit(self) -> None:
        """输出必须匹配 _NAME_RE（小写字母/数字开头，≤64字符）。"""
        long_name = "a" * 100
        result = skills_writer._sanitize_skill_name(long_name)
        assert len(result) <= 64
        assert re.match(r"^[a-z0-9][a-z0-9-]{0,63}$", result)


# ======================================================================
# 3. write_skills —— 端到端写出
# ======================================================================


def _mini_skill(name: str = "test-skill") -> PersonaSkill:
    return PersonaSkill(
        name=name,
        description="A minimal test skill",
        expression_dna=ExpressionDNA(vocabulary=["test", "hello"], rhythm="短句"),
    )


class TestWriteSkills:
    def test_single_skill_creates_directory_and_file(self, tmp_path: Path) -> None:
        written = skills_writer.write_skills(
            persona_id="my-persona",
            skills=[_mini_skill("first-skill")],
            out_dir=tmp_path,
        )
        assert len(written) == 1
        p = written[0]
        assert p.name == "SKILL.md"
        assert p.exists()
        assert "skills" in str(p.parent)

    def test_multiple_skills_all_written(self, tmp_path: Path) -> None:
        skills = [_mini_skill(f"skill-{i}") for i in range(3)]
        written = skills_writer.write_skills(
            persona_id="persona", skills=skills, out_dir=tmp_path
        )
        assert len(written) == 3
        all_names = {p.parent.name for p in written}
        assert len(all_names) == 3

    def test_persona_id_sanitized_in_prefix(self, tmp_path: Path) -> None:
        written = skills_writer.write_skills(
            persona_id="My Persona!!!",
            skills=[_mini_skill("core")],
            out_dir=tmp_path,
        )
        assert written[0].parent.name.startswith("my-persona")

    def test_yaml_frontmatter_present(self, tmp_path: Path) -> None:
        written = skills_writer.write_skills(
            persona_id="p", skills=[_mini_skill("s")], out_dir=tmp_path
        )
        content = written[0].read_text()
        assert content.startswith("---")
        assert "\nname: " in content
        assert "\ndescription: " in content
        assert "\nlicense: " in content
        assert content.count("---") >= 2

    def test_character_count(self, tmp_path: Path) -> None:
        """描述被截断到 1024 字符。"""
        long_desc = "x" * 2000
        written = skills_writer.write_skills(
            persona_id="p",
            skills=[PersonaSkill(name="s", description=long_desc)],
            out_dir=tmp_path,
        )
        content = written[0].read_text()
        # frontmatter 中的 description 应被截断
        desc_line = [l for l in content.split("\n") if l.startswith("description:")][0]
        assert len(desc_line) < 1100


# ======================================================================
# 4. _skill_md —— MD 内容完整性 + HonestBoundary 兜底
# ======================================================================


class TestSkillMdContent:
    def test_honest_boundary_fallback_when_empty(self, tmp_path: Path) -> None:
        """即使没指定 honest_boundaries，也自动追加两条通用局限。"""
        written = skills_writer.write_skills(
            persona_id="p",
            skills=[PersonaSkill(name="s", description="d")],
            out_dir=tmp_path,
        )
        content = written[0].read_text()
        assert "无法蒸馏直觉" in content
        assert "仅基于公开语料" in content

    def test_honest_boundary_not_duplicated_when_present(self, tmp_path: Path) -> None:
        written = skills_writer.write_skills(
            persona_id="p",
            skills=[
                PersonaSkill(
                    name="s",
                    description="d",
                    honest_boundaries=[
                        HonestBoundary(
                            limitation="无法蒸馏直觉", reason="框架能提取，灵感不能。"
                        ),
                        HonestBoundary(
                            limitation="自定义边界", reason="测试用"
                        ),
                    ],
                )
            ],
            out_dir=tmp_path,
        )
        content = written[0].read_text()
        # 自定义边界应出现
        assert "自定义边界" in content
        # "仅基于公开语料" 的兜底仍应追加（因为没显式声明）
        assert "仅基于公开语料" in content
        # "无法蒸馏直觉" 只出现一次（用户已提供，不重复追加）
        assert content.count("无法蒸馏直觉") == 1

    def test_mental_models_rendered(self, tmp_path: Path) -> None:
        skill = PersonaSkill(
            name="s",
            description="d",
            mental_models=[
                MentalModel(
                    name="聚焦即说不",
                    principle="资源有限，聚焦取舍",
                    verification=VerificationResult(
                        cross_domain=True,
                        cross_domain_evidence=["商业决策", "工程调试"],
                        generative=True,
                        generative_example="新问题推断",
                        exclusive=True,
                        exclusivity_note="vs 平均主义者",
                    ),
                    application="面对新任务先问：什么是真正重要的？",
                )
            ],
        )
        written = skills_writer.write_skills(
            persona_id="p", skills=[skill], out_dir=tmp_path
        )
        content = written[0].read_text()
        assert "心智模型" in content
        assert "聚焦即说不" in content
        assert "跨域复现证据" in content
        assert "生成力示例" in content
        assert "排他性" in content
        assert "三重验证" in content

    def test_decision_heuristics_rendered(self, tmp_path: Path) -> None:
        skill = PersonaSkill(
            name="s",
            description="d",
            decision_heuristics=[
                DecisionHeuristic(rule="先问物理极限", trigger="优化场景", example="测试例")
            ],
        )
        written = skills_writer.write_skills(
            persona_id="p", skills=[skill], out_dir=tmp_path
        )
        content = written[0].read_text()
        assert "决策启发式" in content
        assert "先问物理极限" in content

    def test_anti_patterns_rendered(self, tmp_path: Path) -> None:
        skill = PersonaSkill(
            name="s",
            description="d",
            anti_patterns=[
                AntiPattern(pattern="把书当摆设", reason="读书要读完", evidence="原文证据")
            ],
        )
        written = skills_writer.write_skills(
            persona_id="p", skills=[skill], out_dir=tmp_path
        )
        content = written[0].read_text()
        assert "反模式" in content
        assert "把书当摆设" in content
        assert "原文证据" in content

    def test_identity_rules_present(self, tmp_path: Path) -> None:
        written = skills_writer.write_skills(
            persona_id="荒川老师", skills=[_mini_skill()], out_dir=tmp_path
        )
        content = written[0].read_text()
        assert "角色扮演规则" in content
        assert "STOP" in content or "STOP（仅一次）" in content
        assert "EXIT TRIGGER" in content
        assert "用「我」" in content
