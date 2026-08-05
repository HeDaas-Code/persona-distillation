"""``eval.fidelity`` 单元测试。

覆盖忠实度评估器中的纯逻辑函数（不依赖真实 LLM）：
- ``_extract_content``：兼容 LangChain 新旧 content 格式
- ``_parse_json``：从 markdown/前后缀噪声中解析 JSON
- ``_build_corpus_digest``：把 distillates 拼成语料摘要
- ``_build_card_digest``：把 PersonaCard 拼成摘要
- ``evaluate``：异常降级路径（mock llm）

跑法：``python -m pytest tests/test_eval_fidelity.py -v``
"""
from __future__ import annotations

from typing import Any

from persona_distillation.eval import fidelity
from persona_distillation.schemas import (
    AntiPattern,
    DecisionHeuristic,
    Distillate,
    ExpressionDNA,
    FidelityScore,
    HonestBoundary,
    MentalModel,
    PersonaCard,
    PersonaSignal,
    SignalCategory,
    VerificationResult,
)


# ---------------------------------------------------------------------------
# 辅助函数：构造最小可复现的 fixture
# ---------------------------------------------------------------------------
def _mini_card() -> PersonaCard:
    """构造一张最小但字段齐全的 PersonaCard，给 digest 测试用。"""
    return PersonaCard(
        persona_id="arakawa_sensei",
        display_name="荒川老师",
        system_prompt="[身份] 语文老师\n[性格] 严厉",
        expression_dna=ExpressionDNA(
            vocabulary=["嘛", "总之"],
            rhythm="短句",
            rhetorical_tics=["反问"],
            signature_metaphors=["把书当剑"],
            opening_samples=["嘛，今天的课——"],
        ),
        mental_models=[
            MentalModel(
                name="聚焦即说不",
                principle="能砍掉的都砍掉",
                verification=VerificationResult(
                    cross_domain=True,
                    generative=True,
                    exclusive=True,
                ),
            )
        ],
        decision_heuristics=[
            DecisionHeuristic(rule="先问物理极限", trigger="优化问题", example="例")
        ],
        anti_patterns=[
            AntiPattern(pattern="把书当摆设", reason="浪费资源", evidence="证据")
        ],
        honest_boundaries=[
            HonestBoundary(limitation="教不了数学", reason="专业边界")
        ],
    )


def _mini_distillates() -> list[Distillate]:
    """构造 2 条带 signals 的 Distillate。"""
    return [
        Distillate(
            source_file="a.txt",
            chunk_index=0,
            char_start=0,
            char_end=100,
            summary="第一块：荒川上课严厉。",
            signals=[
                PersonaSignal(
                    category=SignalCategory.CATCHPHRASE,
                    content="口头禅：嘛",
                    evidence="嘛，坐下。",
                ),
                PersonaSignal(
                    category=SignalCategory.VALUES,
                    content="重基础轻技巧",
                    evidence="先把课本吃透。",
                ),
            ],
        ),
        Distillate(
            source_file="b.md",
            chunk_index=3,
            char_start=200,
            char_end=400,
            summary="",  # 无 summary 的情况
            signals=[
                PersonaSignal(
                    category=SignalCategory.SPEECH_STYLE,
                    content="短句",  # 无 evidence 的情况
                )
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# _extract_content
# ---------------------------------------------------------------------------
class FakeResp:
    """用于 _extract_content 的 LangChain 风格假响应。"""

    def __init__(self, content: Any) -> None:
        self.content = content


def test_extract_content_plain_str() -> None:
    """content 是普通字符串时，应原样剥离空白。"""
    assert fidelity._extract_content(FakeResp("  hello  ")) == "hello"


def test_extract_content_empty_str() -> None:
    """content 是空串时，应返回空串。"""
    assert fidelity._extract_content(FakeResp("")) == ""
    # content 是 None（某些 langchain 失败情况）→ 转空串
    assert fidelity._extract_content(FakeResp(None)) == ""


def test_extract_content_list_of_str() -> None:
    """content 是 list[str]（langchain 少见但兼容）。"""
    assert fidelity._extract_content(FakeResp(["a", "b", "c"])) == "abc"


def test_extract_content_list_of_dict() -> None:
    """content 是 list[dict]，新版 LangChain 常见格式。"""
    content = [
        {"type": "text", "text": "hello "},
        {"type": "text", "text": "world"},
        # 非 dict 元素被 str() 兜底
        "尾",
    ]
    assert fidelity._extract_content(FakeResp(content)) == "hello world尾"


def test_extract_content_list_of_dict_no_text_key() -> None:
    """dict 缺少 text 键 → 退回 str()，不抛异常。"""
    content = [{"type": "image_url", "url": "xxx"}]
    assert fidelity._extract_content(FakeResp(content)) == ""


# ---------------------------------------------------------------------------
# _parse_json
# ---------------------------------------------------------------------------
def test_parse_json_plain() -> None:
    """纯净 JSON 字符串应直接解析。"""
    assert fidelity._parse_json('{"score": 0.8, "reasons": ["a", "b"]}') == {
        "score": 0.8,
        "reasons": ["a", "b"],
    }


def test_parse_json_markdown_code_block_json_lang() -> None:
    """```json ... ``` 包裹应正确剥离。"""
    text = """```json
{"score": 0.5, "reasons": ["r1"]}
```"""
    assert fidelity._parse_json(text) == {"score": 0.5, "reasons": ["r1"]}


def test_parse_json_markdown_code_block_no_lang() -> None:
    """``` ... ``` 没有语言标签也应剥离。"""
    text = "```\n{\"score\": 0.1}\n```"
    assert fidelity._parse_json(text) == {"score": 0.1}


def test_parse_json_prefix_suffix_noise() -> None:
    """LLM 常在前/后加解释文字，应截取第一个 { 到最后一个 }。"""
    text = "好的我来评估：\n{\"score\": 0.9, \"reasons\": [\"好\", \"不错\"]}\n以上为结果。"
    data = fidelity._parse_json(text)
    assert data["score"] == 0.9
    assert data["reasons"] == ["好", "不错"]


def test_parse_json_multi_object_takes_last() -> None:
    """文本中含多个 {} 块时，取最外层（首 { 到尾 }）。

    注意：该子串不是合法 JSON（两对象中间夹文字），应抛 JSONDecodeError。
    """
    import json

    text = '前缀 { "a": 1 } 中间 { "b": 2 } 后缀'
    # 实现会截取 "{ \"a\": 1 } 中间 { \"b\": 2 }"，该串不是合法 JSON
    try:
        fidelity._parse_json(text)
    except json.JSONDecodeError:
        return  # 预期行为：抛错
    raise AssertionError("多对象夹文字的子串不是合法 JSON，应抛 JSONDecodeError")


def test_parse_json_invalid_raises() -> None:
    """完全无法解析的内容 → 抛 JSONDecodeError（调用方 evaluate 里 catch）。"""
    import json

    try:
        fidelity._parse_json("这不是 json")
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("应抛 json.JSONDecodeError")


# ---------------------------------------------------------------------------
# _build_corpus_digest
# ---------------------------------------------------------------------------
def test_build_corpus_digest_empty() -> None:
    """distillates 为空列表时返回占位字符串，不抛异常。"""
    assert fidelity._build_corpus_digest([]) == "(无原语料摘要)"


def test_build_corpus_digest_contains_metadata() -> None:
    """摘要应包含 source_file、chunk_index、summary 与每条 signal 的 category+内容。"""
    digest = fidelity._build_corpus_digest(_mini_distillates())
    # 分块 1：有 summary + 2 signals（1 条带 evidence，1 条不带）
    assert "分块 1" in digest
    assert "来源: a.txt" in digest
    assert "chunk=0" in digest
    assert "摘要: 第一块：荒川上课严厉。" in digest
    assert "[catchphrase] 口头禅：嘛" in digest or "[catchphrase]" in digest
    assert "｜证据: 嘛，坐下。" in digest
    # 分块 2：无 summary，signal 无 evidence
    assert "分块 2" in digest
    assert "来源: b.md" in digest
    assert "chunk=3" in digest


def test_build_corpus_digest_signal_order_preserved() -> None:
    """输出顺序应与 distillates / signals 枚举顺序一致（确定性）。"""
    ds = _mini_distillates()
    d1 = fidelity._build_corpus_digest(ds)
    d2 = fidelity._build_corpus_digest(ds)
    assert d1 == d2, "同一输入应产生完全相同的输出"


# ---------------------------------------------------------------------------
# _build_card_digest
# ---------------------------------------------------------------------------
def test_build_card_digest_mentions_sections() -> None:
    """PersonaCard 摘要应包含所有 DNA 五层的 section 标题。"""
    digest = fidelity._build_card_digest(_mini_card())
    assert "### PersonaCard" in digest
    assert "persona_id=arakawa_sensei" in digest
    assert "## system_prompt" in digest
    assert "[身份] 语文老师" in digest
    assert "## expression_dna" in digest
    assert "## mental_models" in digest
    assert "## decision_heuristics" in digest
    assert "## anti_patterns" in digest
    assert "## honest_boundaries" in digest


def test_build_card_digest_empty_edna_no_crash() -> None:
    """空 expression_dna + 空 DNA 五层应正常输出 (空) 占位或空列表。"""
    card = PersonaCard(
        persona_id="empty",
        system_prompt="",
        # 其余默认空
    )
    digest = fidelity._build_card_digest(card)
    assert "(空)" in digest or "[]" in digest
    assert "## mental_models" in digest  # section 头仍要在


def test_build_card_digest_deterministic() -> None:
    """相同输入两次调用输出完全一致。"""
    card = _mini_card()
    assert fidelity._build_card_digest(card) == fidelity._build_card_digest(card)


# ---------------------------------------------------------------------------
# evaluate —— 仅覆盖 LLM 失败 / 异常降级路径（不依赖真实 LLM）
# ---------------------------------------------------------------------------
class _RaisingLLM:
    """invoke 直接抛异常。"""

    def invoke(self, *a: Any, **kw: Any) -> Any:
        raise RuntimeError("网络炸了")


def test_evaluate_llm_exception_returns_minus_one() -> None:
    """LLM 抛任何异常时，evaluate 不向上抛，返回 score=-1.0 + 原因。"""
    score = fidelity.evaluate(_mini_card(), _mini_distillates(), _RaisingLLM())
    assert isinstance(score, FidelityScore)
    assert score.score == -1.0
    assert len(score.reasons) >= 1
    assert "judge 调用失败" in score.reasons[0]
    assert "网络炸了" in score.reasons[0]


class _BadJsonLLM:
    """invoke 返回非 JSON 文本。"""

    def __init__(self, text: str) -> None:
        self.text = text

    def invoke(self, *a: Any, **kw: Any) -> Any:
        return FakeResp(self.text)


def test_evaluate_bad_json_falls_back_to_minus_one() -> None:
    """LLM 返回的文本无法解析 JSON → score=-1.0。"""
    score = fidelity.evaluate(_mini_card(), _mini_distillates(), _BadJsonLLM("我不输出json"))
    assert score.score == -1.0
    assert any("judge 调用失败" in r for r in score.reasons)


class _GoodJsonLLM:
    def __init__(self, data: dict) -> None:
        import json as _json

        self.text = _json.dumps(data, ensure_ascii=False)

    def invoke(self, *a: Any, **kw: Any) -> Any:
        return FakeResp(self.text)


def test_evaluate_happy_path_clamps_score() -> None:
    """合法 JSON 下：score 被 clamp 到 [0, 1]；reasons 转成 list[str]。"""
    llm = _GoodJsonLLM({"score": 2.0, "reasons": ["r1", 2, "r3"]})
    result = fidelity.evaluate(_mini_card(), _mini_distillates(), llm)
    assert result.score == 1.0, "score=2.0 应被 clamp 到 1.0"
    assert result.reasons == ["r1", "2", "r3"], "非字符串 reasons 应 str() 转换"


def test_evaluate_negative_score_clamped() -> None:
    """负 score 被 clamp 到 0。"""
    llm = _GoodJsonLLM({"score": -0.5, "reasons": []})
    result = fidelity.evaluate(_mini_card(), _mini_distillates(), llm)
    assert result.score == 0.0
    assert result.reasons == []


def test_evaluate_score_missing_defaults_zero() -> None:
    """score 键缺失 → 默认 0.0。"""
    llm = _GoodJsonLLM({"reasons": ["为啥没有分？"]})
    result = fidelity.evaluate(_mini_card(), _mini_distillates(), llm)
    assert result.score == 0.0
    assert result.reasons == ["为啥没有分？"]


def test_evaluate_reasons_is_string_wrapped() -> None:
    """reasons 是单个字符串 → 包成 [str]。"""
    llm = _GoodJsonLLM({"score": 0.7, "reasons": "一条理由"})
    result = fidelity.evaluate(_mini_card(), _mini_distillates(), llm)
    assert result.reasons == ["一条理由"]


def test_evaluate_passes_corpus_and_card() -> None:
    """确认 llm.invoke 确实收到了两个 digest（通过 prompt 中关键字判断）。"""

    class _SpyLLM:
        captured_prompt: str = ""

        def invoke(self, msgs: Any, **kw: Any) -> Any:
            # msgs[0] = SystemMessage; msgs[1] = HumanMessage
            self.captured_prompt = str(getattr(msgs[1], "content", ""))
            import json as _json

            return FakeResp(_json.dumps({"score": 0.8, "reasons": ["ok"]}))

    spy = _SpyLLM()
    fidelity.evaluate(_mini_card(), _mini_distillates(), spy)
    assert "## 原语料摘要" in spy.captured_prompt
    assert "## PersonaCard 摘要" in spy.captured_prompt
    assert "分块 1" in spy.captured_prompt  # corpus digest 内容
    assert "聚焦即说不" in spy.captured_prompt  # card digest 内容（心智模型名）
