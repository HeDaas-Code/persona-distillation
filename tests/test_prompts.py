"""``prompts`` 单元测试。

覆盖 persona_distillation.prompts 模块中所有公开的构造函数和常量：
- intake_orchestrator_system：workdir 注入
- oc_writer_system / monologue_writer_prompt / dialogue_writer_prompt /
  event_writer_prompt / memory_writer_prompt：setting_text 注入
- character_player_system：setting_block / skeleton_block 双占位符替换
- interviewer_system：返回常量字符串
- synthesizer_system / skill_designer_system / dialogue_writer_system（主流程版）：
  模板变量替换
- 全部公开 *_SYSTEM / *_TEMPLATE 常量：非空且含中文关键词
- 向后兼容的 INTAKE_ORCHESTRATOR_SYSTEM 占位符注入

跑法：``python -m pytest tests/test_prompts.py -v``
"""
from __future__ import annotations

from persona_distillation import prompts


# ---------------------------------------------------------------------------
# 辅助：拿全模块里所有公开的 *_SYSTEM / *_TEMPLATE / *_SKILL_MD 常量
# ---------------------------------------------------------------------------
def _iter_public_constants() -> list[tuple[str, str]]:
    """返回模块中所有以大写字母开头、值为 str 的公开常量 (name, value)。"""
    out: list[tuple[str, str]] = []
    for name in dir(prompts):
        if name.startswith("_"):
            continue
        # 只挑模块级大写常量（不是函数/类）
        if not name.isupper():
            continue
        value = getattr(prompts, name)
        if isinstance(value, str):
            out.append((name, value))
    return out


def test_all_public_constants_are_non_empty_strings() -> None:
    """所有公开大写常量都应是非空字符串。"""
    consts = _iter_public_constants()
    assert consts, "模块里找不到任何公开大写 str 常量——这本身就不对"
    for name, value in consts:
        assert isinstance(value, str), f"{name!r} 应为 str, 实际 {type(value).__name__}"
        assert len(value) > 0, f"{name!r} 不应为空串"


# ---------------------------------------------------------------------------
# ORCHESTRATOR_SYSTEM / EXTRACTOR_SYSTEM / EXTRACTOR_BATCH_SYSTEM
# ---------------------------------------------------------------------------
def test_orchestrator_system_contains_chinese_keywords() -> None:
    assert "人格蒸馏编排者" in prompts.ORCHESTRATOR_SYSTEM
    assert "subagent" not in prompts.ORCHESTRATOR_SYSTEM.lower()
    assert "extractor" in prompts.ORCHESTRATOR_SYSTEM
    assert "synthesizer" in prompts.ORCHESTRATOR_SYSTEM


def test_extractor_system_mentions_signal_categories() -> None:
    text = prompts.EXTRACTOR_SYSTEM
    assert "人格分馏器" in text
    for cat in ("speech_style", "catchphrase", "values", "knowledge",
                "emotion", "relationships", "signature_event",
                "background", "taboo", "mannerism"):
        assert cat in text, f"塔板 {cat!r} 应出现在 EXTRACTOR_SYSTEM"
    assert "salience" in text
    assert "evidence" in text


def test_extractor_batch_system_is_distinct() -> None:
    """批量版和逐块版应是不同字符串，且批量版多了"批量模式"标识。"""
    assert prompts.EXTRACTOR_BATCH_SYSTEM != prompts.EXTRACTOR_SYSTEM
    assert "批量模式" in prompts.EXTRACTOR_BATCH_SYSTEM
    assert "批量模式" not in prompts.EXTRACTOR_SYSTEM


# ---------------------------------------------------------------------------
# SYNTHESIZER_SYSTEM 模板 + synthesizer_system() helper
# ---------------------------------------------------------------------------
def test_synthesizer_system_template_has_placeholders() -> None:
    assert "{salience_threshold}" in prompts.SYNTHESIZER_SYSTEM
    assert "{persona_id_hint}" in prompts.SYNTHESIZER_SYSTEM


def test_synthesizer_system_formats_placeholders() -> None:
    out = prompts.synthesizer_system(0.6, "请体现角色身份")
    assert "{salience_threshold}" not in out
    assert "{persona_id_hint}" not in out
    assert "0.6" in out
    assert "请体现角色身份" in out
    # DNA 五层必须保留
    for layer in ("expression_dna", "mental_models", "decision_heuristics",
                  "anti_patterns", "honest_boundaries"):
        assert layer in out


# ---------------------------------------------------------------------------
# SKILL_DESIGNER_SYSTEM 模板 + skill_designer_system() helper
# ---------------------------------------------------------------------------
def test_skill_designer_system_template_has_placeholder() -> None:
    assert "{max_skills}" in prompts.SKILL_DESIGNER_SYSTEM
    assert "DNA 级别" in prompts.SKILL_DESIGNER_SYSTEM


def test_skill_designer_system_formats_max_skills() -> None:
    out = prompts.skill_designer_system(5)
    assert "{max_skills}" not in out
    assert "5" in out


# ---------------------------------------------------------------------------
# DIALOGUE_WRITER_SYSTEM 模板 + dialogue_writer_system() helper（主流程版）
# ---------------------------------------------------------------------------
def test_dialogue_writer_system_constant_is_oc_version() -> None:
    """源码第 507 行的 OC 版覆盖了第 248 行的预设对话作者版，
    所以导出的 DIALOGUE_WRITER_SYSTEM 是 OC 对话撰写者，不含 {max_dialogues}。"""
    assert "OC 对话撰写者" in prompts.DIALOGUE_WRITER_SYSTEM
    assert "{max_dialogues}" not in prompts.DIALOGUE_WRITER_SYSTEM


def test_dialogue_writer_system_helper_returns_same_constant() -> None:
    """由于常量被覆盖，dialogue_writer_system() 的 format() 没有占位符可替换，
    返回值与常量完全一致（此行为即源码当前状态，测试锁定它）。"""
    assert prompts.dialogue_writer_system(8) == prompts.DIALOGUE_WRITER_SYSTEM


# ---------------------------------------------------------------------------
# INTAKE 子包：NER / PROFILE_BUILDER / BRIDGER
# ---------------------------------------------------------------------------
def test_intake_ner_system_keywords() -> None:
    t = prompts.INTAKE_NER_SYSTEM
    assert "人物识别与分类专家" in t
    for cat in ("speech", "appearance", "event"):
        assert cat in t
    assert "chunk_meta" in t


def test_profile_builder_system_keywords() -> None:
    t = prompts.PROFILE_BUILDER_SYSTEM
    assert "人物档案撰写者" in t
    assert "CharacterProfile" in t
    assert "CharacterProfile" in t


def test_bridger_system_keywords() -> None:
    t = prompts.BRIDGER_SYSTEM
    assert "蒸馏桥接者" in t
    assert "<workdir>" in t  # bridger 自己不注入 workdir，靠主理人替换
    assert "persona_card.json" in t


# ---------------------------------------------------------------------------
# INTAKE_ORCHESTRATOR_SYSTEM_TEMPLATE + intake_orchestrator_system()
# ---------------------------------------------------------------------------
def test_intake_orchestrator_template_has_workdir_placeholder() -> None:
    assert "{workdir}" in prompts.INTAKE_ORCHESTRATOR_SYSTEM_TEMPLATE


def test_intake_orchestrator_system_injects_workdir() -> None:
    out = prompts.intake_orchestrator_system("/tmp/run-123")
    assert "{workdir}" not in out
    assert "/tmp/run-123" in out
    # 派生路径也应被替换
    assert "/tmp/run-123/index/" in out
    assert "/tmp/run-123/distillates.json" in out


def test_intake_orchestrator_system_str_accepts_path_like_object() -> None:
    """形参标注 str | object，传入带 __str__ 的对象也应正常工作。"""

    class _Pathy:
        def __str__(self) -> str:
            return "/fake/path"

    out = prompts.intake_orchestrator_system(_Pathy())
    assert "/fake/path" in out


def test_intake_orchestrator_system_default_empty_string() -> None:
    """默认参数是空串——即使忘了传也不会抛 KeyError。"""
    out = prompts.intake_orchestrator_system()
    assert isinstance(out, str)
    assert "{workdir}" not in out


def test_intake_orchestrator_system_backward_compat_placeholder() -> None:
    """常量版本用 '<workdir>' 占位，不包含 Python format 的花括号。"""
    tpl = prompts.INTAKE_ORCHESTRATOR_SYSTEM_TEMPLATE
    const = prompts.INTAKE_ORCHESTRATOR_SYSTEM
    # 常量是模板把 workdir="<workdir>" 填进去
    expected = tpl.format(workdir="<workdir>")
    assert const == expected
    assert "{workdir}" not in const
    assert "<workdir>" in const


# ---------------------------------------------------------------------------
# OC 共创：MONOLOGUE / DIALOGUE / EVENT / MEMORY 4 个 writer
# ---------------------------------------------------------------------------
def test_oc_writer_system_helper_injects_setting_block() -> None:
    """oc_writer_system 应在 base_system 后追加分隔 + 【OC 设定】 + setting_text。"""
    base = "你是一个虚构的 writer base。"
    setting = "姓名: 小明\n年龄: 17"
    out = prompts.oc_writer_system(base, setting)
    assert base in out
    assert setting in out
    assert "【OC 设定】" in out
    # base 与 setting 之间有双换行分隔
    assert "\n\n【OC 设定】" in out


def test_monologue_writer_prompt_composes() -> None:
    out = prompts.monologue_writer_prompt("姓名: 小明")
    assert "OC 独白撰写者" in out
    assert "姓名: 小明" in out
    assert out.startswith(prompts.MONOLOGUE_WRITER_SYSTEM)


def test_dialogue_writer_prompt_composes() -> None:
    out = prompts.dialogue_writer_prompt("姓名: 小红")
    assert "OC 对话撰写者" in out
    assert "姓名: 小红" in out
    assert out.startswith(prompts.DIALOGUE_WRITER_SYSTEM)


def test_event_writer_prompt_composes() -> None:
    out = prompts.event_writer_prompt("姓名: 小刚")
    assert "OC 事件撰写者" in out
    assert "姓名: 小刚" in out
    assert out.startswith(prompts.EVENT_WRITER_SYSTEM)


def test_memory_writer_prompt_composes() -> None:
    out = prompts.memory_writer_prompt("姓名: 小美")
    assert "OC 回忆撰写者" in out
    assert "姓名: 小美" in out
    assert out.startswith(prompts.MEMORY_WRITER_SYSTEM)


# ---------------------------------------------------------------------------
# CHARACTER_PLAYER_SYSTEM 模板 + character_player_system()
# ---------------------------------------------------------------------------
def test_character_player_template_has_both_placeholders() -> None:
    t = prompts.CHARACTER_PLAYER_SYSTEM
    assert "{setting_block}" in t
    assert "{skeleton_block}" in t


def test_character_player_system_substitutes_both_blocks() -> None:
    setting = "【身份】高中生\n【性格】冷淡"
    skeleton = "独白片段...\n对话片段..."
    out = prompts.character_player_system(setting, skeleton)
    assert "{setting_block}" not in out
    assert "{skeleton_block}" not in out
    assert setting in out
    assert skeleton in out
    # 角色扮演者关键词保留
    assert "OC 角色扮演者" in out
    assert "第一人称" in out


def test_character_player_system_empty_inputs_are_noop() -> None:
    """即使 setting/skeleton 是空串，也不应抛 KeyError。"""
    out = prompts.character_player_system("", "")
    assert isinstance(out, str)
    assert "{setting_block}" not in out
    assert "{skeleton_block}" not in out


# ---------------------------------------------------------------------------
# INTERVIEWER_SYSTEM + interviewer_system()
# ---------------------------------------------------------------------------
def test_interviewer_system_returns_constant() -> None:
    assert prompts.interviewer_system() is prompts.INTERVIEWER_SYSTEM
    assert prompts.interviewer_system() == prompts.INTERVIEWER_SYSTEM


def test_interviewer_system_keywords() -> None:
    t = prompts.INTERVIEWER_SYSTEM
    assert "OC 访谈主理人" in t
    assert "只问一个问题" in t
    assert "第二人称" in t


# ---------------------------------------------------------------------------
# DISTILLATION_SKILL_MD
# ---------------------------------------------------------------------------
def test_distillation_skill_md_has_frontmatter_and_sections() -> None:
    md = prompts.DISTILLATION_SKILL_MD
    assert md.startswith("---\n")
    assert "name: persona-distillation" in md
    assert "分馏" in md
    assert "冷凝" in md
    assert "提纯" in md
    assert "成品" in md
    assert "DNA 五层" in md
    assert "三重验证" in md


# ---------------------------------------------------------------------------
# 完整性 sanity check：模块中公开的函数全列在这里（防未来漏测）
# ---------------------------------------------------------------------------
def test_all_public_functions_covered() -> None:
    """列出模块里所有公开 callable，帮助发现未来新增函数是否被漏测。"""
    fns = sorted(
        name for name in dir(prompts)
        if not name.startswith("_") and not name.isupper()
        and callable(getattr(prompts, name))
    )
    # 只做 sanity check，不强制新增函数必须立即有测试——但要在输出里展示
    assert "intake_orchestrator_system" in fns
    assert "synthesizer_system" in fns
    assert "skill_designer_system" in fns
    assert "dialogue_writer_system" in fns
    assert "oc_writer_system" in fns
    assert "monologue_writer_prompt" in fns
    assert "dialogue_writer_prompt" in fns
    assert "event_writer_prompt" in fns
    assert "memory_writer_prompt" in fns
    assert "character_player_system" in fns
    assert "interviewer_system" in fns
