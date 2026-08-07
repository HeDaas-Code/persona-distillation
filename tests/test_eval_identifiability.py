"""``eval.identifiability`` 单元测试。

覆盖可识别度评估器中的纯逻辑函数（不依赖真实 LLM）：
- ``_extract_content``：兼容 LangChain 新旧 content 格式
- ``_parse_json``：从 markdown/前后缀噪声中解析 JSON
- ``_build_corpus_digest``：把 distillates 拼成语料摘要
- ``_split_probe_qa``：把 LLM 整体回答按 "Q:" 分段
- ``_fuzzy_match``：模糊匹配 judge 猜测名与 PersonaCard
- ``evaluate``：端到端 happy path + 异常降级（mock llm）

跑法：``python -m pytest tests/test_eval_identifiability.py -v``
"""
from __future__ import annotations

from typing import Any

from persona_distillation.eval import identifiability
from persona_distillation.schemas import (
    Distillate,
    IdentifiabilityScore,
    PersonaCard,
    PersonaSignal,
    SignalCategory,
)


# ---------------------------------------------------------------------------
# 辅助：mock LangChain 风格响应对象
# ---------------------------------------------------------------------------
class _FakeResp:
    """用于 _extract_content 的 LangChain 风格假响应。"""

    def __init__(self, content: Any) -> None:
        self.content = content


# ---------------------------------------------------------------------------
# 辅助：最小可复现 fixture
# ---------------------------------------------------------------------------
def _mini_card() -> PersonaCard:
    """构造一张最小 PersonaCard（荒川老师），给模糊匹配测试用。"""
    return PersonaCard(
        persona_id="arakawa_sensei",
        display_name="荒川老师",
        system_prompt="你是一位严厉但关心学生的语文老师。",
    )


def _mini_card_no_display_name() -> PersonaCard:
    """只有 persona_id，没有 display_name 的边界情况。"""
    return PersonaCard(
        persona_id="dr_kudo",
        system_prompt="你是工藤新一。",
    )


def _mini_distillates() -> list[Distillate]:
    """构造 2 条带 signals 的 Distillate。"""
    return [
        Distillate(
            source_file="arakawa.txt",
            chunk_index=0,
            char_start=0,
            char_end=100,
            summary="荒川上课风格严厉，口头禅是'嘛'。",
            signals=[
                PersonaSignal(
                    category=SignalCategory.CATCHPHRASE,
                    content="口头禅：嘛",
                ),
                PersonaSignal(
                    category=SignalCategory.VALUES,
                    content="重基础轻技巧",
                ),
            ],
        ),
        Distillate(
            source_file="arakawa.md",
            chunk_index=3,
            char_start=200,
            char_end=400,
            summary="",
            signals=[
                PersonaSignal(
                    category=SignalCategory.SPEECH_STYLE,
                    content="说话节奏快，常用短句",
                ),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# _extract_content
# ---------------------------------------------------------------------------
def test_extract_content_plain_str() -> None:
    """content 是普通字符串时原样剥离空白。"""
    assert identifiability._extract_content(_FakeResp("  hello  ")) == "hello"


def test_extract_content_empty_str() -> None:
    """content 是空串或 None 时返回空串。"""
    assert identifiability._extract_content(_FakeResp("")) == ""
    assert identifiability._extract_content(_FakeResp(None)) == ""


def test_extract_content_list_of_dict() -> None:
    """content 是 list[dict]（新版 LangChain 常见格式）。"""
    content = [
        {"type": "text", "text": "probe 1"},
        {"type": "text", "text": "probe 2"},
    ]
    assert identifiability._extract_content(_FakeResp(content)) == "probe 1probe 2"


def test_extract_content_list_of_dict_no_text_key() -> None:
    """dict 缺少 text 键 → 退回空串。"""
    content = [{"type": "image_url", "url": "xxx"}]
    assert identifiability._extract_content(_FakeResp(content)) == ""


def test_extract_content_list_mixed_types() -> None:
    """list 中混合 dict 和 str → 用 str() 兜底。"""
    content = [{"type": "text", "text": "a"}, "b", {"type": "text", "text": "c"}]
    assert identifiability._extract_content(_FakeResp(content)) == "abc"


# ---------------------------------------------------------------------------
# _parse_json
# ---------------------------------------------------------------------------
def test_parse_json_plain() -> None:
    """纯净 JSON 字符串直接解析。"""
    assert identifiability._parse_json('{"name": "荒川老师", "confidence": 0.8}') == {
        "name": "荒川老师",
        "confidence": 0.8,
    }


def test_parse_json_markdown_code_block_json_lang() -> None:
    """```json ... ``` 包裹正确剥离。"""
    text = """```json
{"name": "荒川", "confidence": 0.5}
```"""
    assert identifiability._parse_json(text) == {"name": "荒川", "confidence": 0.5}


def test_parse_json_markdown_code_block_no_lang() -> None:
    """``` ... ``` 无语言标签也剥离。"""
    text = "```\n{\"name\": \"x\"}\n```"
    assert identifiability._parse_json(text) == {"name": "x"}


def test_parse_json_prefix_suffix_noise() -> None:
    """LLM 前/后加解释文字 → 截取首 { 到尾 }。"""
    text = "好的，我的判断如下：\n{\"name\": \"荒川\", \"confidence\": 0.9}\n完毕。"
    data = identifiability._parse_json(text)
    assert data["name"] == "荒川"
    assert data["confidence"] == 0.9


def test_parse_json_invalid_raises() -> None:
    """完全无法解析 → 抛 JSONDecodeError（evaluate 里 catch）。"""
    import json

    try:
        identifiability._parse_json("这不是 json")
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("应抛 json.JSONDecodeError")


def test_parse_json_empty_but_valid() -> None:
    """空 JSON 对象 {} → 正常解析。"""
    assert identifiability._parse_json("{}") == {}


# ---------------------------------------------------------------------------
# _build_corpus_digest
# ---------------------------------------------------------------------------
def test_build_corpus_digest_empty() -> None:
    """distillates 为空列表 → 占位字符串。"""
    assert identifiability._build_corpus_digest([]) == "(无原语料摘要)"


def test_build_corpus_digest_contains_summary_and_signals() -> None:
    """输出包含分块摘要和各 signal。"""
    digest = identifiability._build_corpus_digest(_mini_distillates())
    assert "分块1摘要: 荒川上课风格严厉，口头禅是'嘛'。" in digest
    assert "[catchphrase] 口头禅：嘛" in digest
    assert "[values] 重基础轻技巧" in digest


def test_build_corpus_digest_no_summary_only_signals() -> None:
    """分块只有 signals 没有 summary → 正常输出 signals。"""
    ds = [
        Distillate(
            source_file="c.txt",
            chunk_index=0,
            char_start=0,
            char_end=50,
            summary="",
            signals=[
                PersonaSignal(
                    category=SignalCategory.EMOTION,
                    content="容易激动",
                ),
            ],
        )
    ]
    digest = identifiability._build_corpus_digest(ds)
    assert "(信号无摘要)" not in digest or "分块1摘要" not in digest or "[emotion] 容易激动" in digest
    assert "[emotion] 容易激动" in digest


def test_build_corpus_digest_deterministic() -> None:
    """相同输入两次调用输出完全一致。"""
    ds = _mini_distillates()
    assert identifiability._build_corpus_digest(ds) == identifiability._build_corpus_digest(ds)


# ---------------------------------------------------------------------------
# _split_probe_qa — identifiability 独有的核心函数
# ---------------------------------------------------------------------------
def test_split_probe_qa_empty_text() -> None:
    """空文本 → 空列表。"""
    assert identifiability._split_probe_qa("") == []


def test_split_probe_qa_single_probe_fallback() -> None:
    """只有一段回答（无 Q: 标记）→ 回退为整个文本单元素列表。"""
    text = "我是荒川老师。"
    result = identifiability._split_probe_qa(text)
    assert len(result) == 1
    assert result[0] == "我是荒川老师。"


def test_split_probe_qa_two_probes() -> None:
    """两个 Q: 标记 → 正确分成两段。"""
    text = "Q: 你怎么看待朋友？\nA: 朋友是人生的镜子。\n\nQ: 最近在忙什么？\nA: 备课。"
    result = identifiability._split_probe_qa(text)
    assert len(result) == 2
    assert result[0].startswith("Q: 你怎么看待朋友？")
    assert result[1].startswith("Q: 最近在忙什么？")


def test_split_probe_qa_five_probes() -> None:
    """5 个 probe 回答全部正确分割。"""
    text = (
        "Q: 你怎么看待朋友？\nA: 重要。\n\n"
        "Q: 最近在忙什么？\nA: 备课。\n\n"
        "Q: 遇到不公时会怎样？\nA: 会抗争。\n\n"
        "Q: 你最看重什么？\nA: 诚实。\n\n"
        "Q: 用一个比喻描述自己。\nA: 像剑。"
    )
    result = identifiability._split_probe_qa(text)
    assert len(result) == 5
    for i, seg in enumerate(result):
        assert seg.startswith("Q:")
    # 最后一段包含最后一个 Q: 到文本末尾
    assert "像剑" in result[4]


def test_split_probe_qa_case_insensitive() -> None:
    """Q: 的匹配是大小写不敏感的（实现用了 upper()）。"""
    text = "q: 第一个问题\nA: 回答一\n\nQ: 第二个问题\nA: 回答二"
    result = identifiability._split_probe_qa(text)
    assert len(result) == 2


def test_split_probe_qa_only_one_q_marker_fallback() -> None:
    """只有一个 Q: 标记（少于 2 个）→ 回退单元素列表。"""
    text = "Q: 问题一\nA: 回答一"
    result = identifiability._split_probe_qa(text)
    assert len(result) == 1


def test_split_probe_qa_whitespace_trimmed() -> None:
    """每段被 strip() 处理。"""
    text = "  Q: 问题1\nA: 回答1  \n\n  Q: 问题2\nA: 回答2  "
    result = identifiability._split_probe_qa(text)
    assert result[0].endswith("回答1")
    assert not result[0].endswith("  ")


# ---------------------------------------------------------------------------
# _fuzzy_match — identifiability 独有的核心函数
# ---------------------------------------------------------------------------
def test_fuzzy_match_exact_display_name() -> None:
    """精确匹配 display_name → True。"""
    assert identifiability._fuzzy_match("荒川老师", _mini_card()) is True


def test_fuzzy_match_exact_persona_id() -> None:
    """精确匹配 persona_id → True。"""
    assert identifiability._fuzzy_match("arakawa_sensei", _mini_card()) is True


def test_fuzzy_match_substring_guessed_in_card() -> None:
    """guessed 是 display_name 的子串 → True。"""
    assert identifiability._fuzzy_match("荒川", _mini_card()) is True


def test_fuzzy_match_substring_card_in_guessed() -> None:
    """display_name 是 guessed 的子串 → True。"""
    assert identifiability._fuzzy_match("这是荒川老师吧", _mini_card()) is True


def test_fuzzy_match_case_insensitive() -> None:
    """大小写不敏感匹配。"""
    assert identifiability._fuzzy_match("ARAKAWA_SENSEI", _mini_card()) is True
    assert identifiability._fuzzy_match("荒川老师", _mini_card()) is True


def test_fuzzy_match_empty_guessed() -> None:
    """guessed 为空或纯空白 → False。"""
    assert identifiability._fuzzy_match("", _mini_card()) is False
    assert identifiability._fuzzy_match("   ", _mini_card()) is False


def test_fuzzy_match_no_display_name_only_persona_id() -> None:
    """只有 persona_id 可用时，匹配 persona_id。"""
    card = _mini_card_no_display_name()
    assert identifiability._fuzzy_match("dr_kudo", card) is True
    assert identifiability._fuzzy_match("工藤新一", card) is False  # 不是子串


def test_fuzzy_match_wrong_name() -> None:
    """完全不相关的名字 → False。"""
    assert identifiability._fuzzy_match("张三", _mini_card()) is False


def test_fuzzy_match_partial_persona_id() -> None:
    """guessed 是 persona_id 子串 → True。"""
    assert identifiability._fuzzy_match("arakawa", _mini_card()) is True


# ---------------------------------------------------------------------------
# evaluate —— 端到端测试（mock LLM）
# ---------------------------------------------------------------------------
class _SequentialMockLLM:
    """按顺序返回多个预置的 probe 回答和 guess 结果。"""

    def __init__(self, probe_text: str, guess_json_text: str) -> None:
        self.probe_text = probe_text
        self.guess_json_text = guess_json_text
        self.call_count = 0

    def invoke(self, *a: Any, **kw: Any) -> _FakeResp:
        self.call_count += 1
        if self.call_count == 1:
            return _FakeResp(self.probe_text)
        return _FakeResp(self.guess_json_text)


def test_evaluate_happy_path_correct_guess() -> None:
    """judge 正确猜出人物 → correct=True, confidence 在 [0,1]。"""
    probe_text = (
        "Q: 你怎么看待朋友？\nA: 朋友是镜子。\n\n"
        "Q: 最近在忙什么？\nA: 备课。\n\n"
        "Q: 遇到不公时会怎样？\nA: 抗争。\n\n"
        "Q: 你最看重什么？\nA: 诚实。\n\n"
        "Q: 用一个比喻描述自己。\nA: 像剑。"
    )
    llm = _SequentialMockLLM(probe_text, '{"name": "荒川老师", "confidence": 0.85}')
    score = identifiability.evaluate(_mini_card(), _mini_distillates(), llm)
    assert isinstance(score, IdentifiabilityScore)
    assert score.confidence == 0.85
    assert score.correct is True
    assert score.guessed_name == "荒川老师"
    assert len(score.probes) == 5
    assert score.error == ""


def test_evaluate_happy_path_wrong_guess() -> None:
    """judge 猜错 → correct=False。"""
    probe_text = "Q: 你怎么看待朋友？\nA: 不知道。\n\nQ: 最近在忙什么？\nA: 睡觉。"
    llm = _SequentialMockLLM(probe_text, '{"name": "张三", "confidence": 0.3}')
    score = identifiability.evaluate(_mini_card(), _mini_distillates(), llm)
    assert score.confidence == 0.3
    assert score.correct is False
    assert score.guessed_name == "张三"


def test_evaluate_confidence_clamped_to_zero_one() -> None:
    """confidence 超出 [0,1] 范围时被 clamp。"""
    probe_text = "Q: 问题\nA: 回答"
    llm = _SequentialMockLLM(probe_text, '{"name": "荒川", "confidence": 1.5}')
    score = identifiability.evaluate(_mini_card(), _mini_distillates(), llm)
    assert score.confidence == 1.0

    llm2 = _SequentialMockLLM(probe_text, '{"name": "荒川", "confidence": -0.3}')
    score2 = identifiability.evaluate(_mini_card(), _mini_distillates(), llm2)
    assert score2.confidence == 0.0


def test_evaluate_n_probes_parameter() -> None:
    """n_probes 参数限制使用的 probe 数量（≤5）。"""
    probe_text = "Q: 问题一\nA: 答一\n\nQ: 问题二\nA: 答二"
    llm = _SequentialMockLLM(probe_text, '{"name": "荒川", "confidence": 0.7}')
    score = identifiability.evaluate(_mini_card(), _mini_distillates(), llm, n_probes=2)
    # n_probes=2 → 应该只用到前 2 个 probe
    assert len(score.probes) == 2


def test_evaluate_n_probes_capped_at_5() -> None:
    """n_probes 超过 5 时被截断（_PROBES 只有 5 个）。"""
    probe_text = "Q: a\nA: 1"
    llm = _SequentialMockLLM(probe_text, '{"name": "荒川", "confidence": 0.7}')
    score = identifiability.evaluate(_mini_card(), _mini_distillates(), llm, n_probes=99)
    assert len(score.probes) >= 1  # 至少有 fallback 的一段


def test_evaluate_n_probes_at_least_1() -> None:
    """n_probes ≤0 时至少用 1 个 probe。"""
    probe_text = "Q: a\nA: 1"
    llm = _SequentialMockLLM(probe_text, '{"name": "荒川", "confidence": 0.7}')
    score = identifiability.evaluate(_mini_card(), _mini_distillates(), llm, n_probes=-5)
    # 至少不会崩溃
    assert isinstance(score, IdentifiabilityScore)


def test_evaluate_probe_fallback_on_no_q_markers() -> None:
    """probe 生成失败（无 Q: 标记）→ _split_probe_qa 回退单元素。"""
    llm = _SequentialMockLLM("我是荒川，我热爱教学。", '{"name": "荒川", "confidence": 0.7}')
    score = identifiability.evaluate(_mini_card(), _mini_distillates(), llm)
    assert len(score.probes) == 1  # fallback
    assert "荒川" in score.probes[0]


def test_evaluate_llm_exception_returns_minus_one() -> None:
    """LLM 任何一步抛异常 → 返回 confidence=-1.0 + error 信息。"""

    class _RaisingLLM:
        def invoke(self, *a: Any, **kw: Any) -> _FakeResp:
            raise RuntimeError("网络炸了")

    score = identifiability.evaluate(_mini_card(), _mini_distillates(), _RaisingLLM())
    assert score.confidence == -1.0
    assert score.correct is False
    assert score.error != ""
    assert "网络炸了" in score.error


def test_evaluate_guess_bad_json_falls_back_to_minus_one() -> None:
    """judge 返回无法解析的文本 → confidence=-1.0。"""
    probe_text = "Q: 问题\nA: 回答"
    llm = _SequentialMockLLM(probe_text, "我不输出 JSON")
    score = identifiability.evaluate(_mini_card(), _mini_distillates(), llm)
    assert score.confidence == -1.0


def test_evaluate_guess_missing_name_defaults_empty() -> None:
    """judge 返回缺 name 键 → guessed_name 为空串，correct=False。"""
    probe_text = "Q: 问题\nA: 回答"
    llm = _SequentialMockLLM(probe_text, '{"confidence": 0.5}')
    score = identifiability.evaluate(_mini_card(), _mini_distillates(), llm)
    assert score.guessed_name == ""
    assert score.correct is False


def test_evaluate_guess_missing_confidence_defaults_zero() -> None:
    """judge 返回缺 confidence 键 → 默认为 0.0。"""
    probe_text = "Q: 问题\nA: 回答"
    llm = _SequentialMockLLM(probe_text, '{"name": "荒川"}')
    score = identifiability.evaluate(_mini_card(), _mini_distillates(), llm)
    assert score.confidence == 0.0
    assert score.correct is True


def test_evaluate_probe_contains_system_prompt() -> None:
    """probe 生成阶段的 SystemMessage 包含 card.system_prompt。"""

    class _SpyLLM:
        probe_system: str = ""
        guess_system: str = ""
        _phase = 0

        def invoke(self, msgs: Any, **kw: Any) -> _FakeResp:
            self._phase += 1
            sys_content = str(getattr(msgs[0], "content", ""))
            if self._phase == 1:
                self.probe_system = sys_content
                return _FakeResp("Q: 问题\nA: 回答")
            self.guess_system = sys_content
            import json as _json

            return _FakeResp(_json.dumps({"name": "荒川", "confidence": 0.8}))

    spy = _SpyLLM()
    identifiability.evaluate(_mini_card(), _mini_distillates(), spy)
    assert "你是一位严厉但关心学生的语文老师" in spy.probe_system
    # guess 阶段的 system 是 judge prompt，不应包含角色设定
    assert "你是人物识别的独立评委" in spy.guess_system


def test_evaluate_guess_contains_corpus_digest() -> None:
    """盲猜阶段的 HumanMessage 包含语料摘要（判断了"原语料摘要"关键字）。"""

    class _SpyLLM:
        captured_human: str = ""
        _phase = 0

        def invoke(self, msgs: Any, **kw: Any) -> _FakeResp:
            self._phase += 1
            if self._phase == 1:
                return _FakeResp("Q: 问题\nA: 回答")
            # guess 阶段——捕获 HumanMessage
            self.captured_human = str(getattr(msgs[1], "content", ""))
            import json as _json

            return _FakeResp(_json.dumps({"name": "荒川", "confidence": 0.8}))

    spy = _SpyLLM()
    identifiability.evaluate(_mini_card(), _mini_distillates(), spy)
    assert "原语料摘要" in spy.captured_human
    assert "Probe Q&A" in spy.captured_human
    assert "荒川上课风格严厉" in spy.captured_human


def test_evaluate_with_empty_distillates() -> None:
    """distillates 为空列表时也能正常工作（不崩）。"""
    probe_text = "Q: 问题\nA: 回答"
    llm = _SequentialMockLLM(probe_text, '{"name": "荒川", "confidence": 0.6}')
    score = identifiability.evaluate(_mini_card(), [], llm)
    assert isinstance(score, IdentifiabilityScore)
    assert score.confidence == 0.6
    assert score.correct is True
