"""``oc_writer.generate_oc_corpus`` / ``OCSetting.to_prompt_text`` 单元测试。

Issue #13：已转为 pytest 风格——``test_*`` 函数由 pytest 自动收集，
直接用 ``assert`` 断言，不再需要 ``main()`` / ``try/except`` 包装。
所有 LLM 路径走 ``FakeLLM``，不调真实 API。

跑法：``python -m pytest tests/test_oc_writer.py -v``
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from persona_distillation.intake.oc_writer import OCSetting, generate_oc_corpus


# ---------------------------------------------------------------------------
# FakeLLM：模拟 LangChain BaseChatModel
# ---------------------------------------------------------------------------
class _FakeResp:
    """模拟 LangChain 返回对象：有 ``.content`` 属性。"""

    def __init__(self, content: Any) -> None:
        self.content = content


class FakeLLM:
    """可记录调用次数 / 返回内容 / 触发异常的 Fake LLM。

    - ``content`` 是字符串时直接返回 ``_FakeResp(str)``
    - ``content`` 是 list 时模拟 langchain 新版返回（content 为 list）
    - ``raise_on_invoke`` 非 None 时，``invoke`` 抛指定异常
    - ``call_count`` 记录 invoke 调用次数
    """

    def __init__(
        self,
        content: Any = "这是 fake LLM 生成的骨架正文。",
        raise_on_invoke: BaseException | None = None,
    ) -> None:
        self.content = content
        self.raise_on_invoke = raise_on_invoke
        self.call_count = 0
        # 记录每次 invoke 收到的 messages（便于断言）
        self.calls: list[list] = []

    def invoke(self, messages: list) -> _FakeResp:
        self.call_count += 1
        self.calls.append(messages)
        if self.raise_on_invoke is not None:
            raise self.raise_on_invoke
        return _FakeResp(self.content)


# ---------------------------------------------------------------------------
# 构造一份共用的 OC 设定
# ---------------------------------------------------------------------------
def _make_setting() -> OCSetting:
    return OCSetting(
        name="小满",
        age="17",
        background="海边小镇的高二学生，家里开着老书店",
        traits="外冷内热、好奇心强、嘴硬心软",
        worldview="书是有生命的，被读懂才能完整",
        catchphrase="嘛，再看看吧。",
    )


# ---------------------------------------------------------------------------
# 用例
# ---------------------------------------------------------------------------
def test_oc_setting_to_prompt_text() -> None:
    """OCSetting.to_prompt_text() 输出应含全部 6 个字段（姓名/年龄/背景/性格核心/世界观/口头禅）。"""
    s = _make_setting()
    text = s.to_prompt_text()
    assert s.name in text, f"to_prompt_text 缺姓名: {text!r}"
    assert s.age in text, f"to_prompt_text 缺年龄: {text!r}"
    assert s.background in text, f"to_prompt_text 缺背景: {text!r}"
    assert s.traits in text, f"to_prompt_text 缺性格核心: {text!r}"
    assert s.worldview in text, f"to_prompt_text 缺世界观: {text!r}"
    assert s.catchphrase in text, f"to_prompt_text 缺口头禅: {text!r}"
    # 6 个字段标签都应在
    for label in ("姓名", "年龄", "背景", "性格核心", "世界观", "口头禅"):
        assert label in text, f"to_prompt_text 缺标签 {label}: {text!r}"


def test_generate_oc_corpus_creates_4_files() -> None:
    """generate_oc_corpus 应落盘 4 个 .md 文件，每个开头是 `# {类型名} · {name}`。"""
    setting = _make_setting()
    llm = FakeLLM(content="这是骨架正文，应至少有几十个字。")
    with tempfile.TemporaryDirectory() as td:
        result = generate_oc_corpus(setting, Path(td), "xiaoman", llm)

        # 返回 dict 含 paths / word_counts / corpus_dir
        assert "paths" in result, f"返回 dict 缺 paths: {result}"
        assert "word_counts" in result, f"返回 dict 缺 word_counts: {result}"
        assert "corpus_dir" in result, f"返回 dict 缺 corpus_dir: {result}"

        # 4 个 key 都齐全
        for key in ("monologue", "dialogue", "event", "memory"):
            assert key in result["paths"], f"paths 缺 {key}: {result['paths']}"
            assert key in result["word_counts"], f"word_counts 缺 {key}"

        # 4 个 .md 文件确实落盘
        corpus_dir = Path(result["corpus_dir"])
        assert corpus_dir.exists(), f"corpus_dir 不存在: {corpus_dir}"
        for fname in ("monologue.md", "dialogue.md", "event.md", "memory.md"):
            f = corpus_dir / fname
            assert f.exists(), f"骨架文件未落盘: {f}"

        # 每个文件开头应是 `# {类型名} · {name}`
        expected_headers = {
            "monologue.md": f"# 独白 · {setting.name}",
            "dialogue.md": f"# 对话 · {setting.name}",
            "event.md": f"# 事件 · {setting.name}",
            "memory.md": f"# 回忆 · {setting.name}",
        }
        for fname, header in expected_headers.items():
            body = (corpus_dir / fname).read_text(encoding="utf-8")
            assert body.startswith(header), (
                f"{fname} 开头应是 {header!r}，实际: {body[:50]!r}"
            )

        # word_counts 应等于正文长度（不含标题行）
        for key, fname in (
            ("monologue", "monologue.md"),
            ("dialogue", "dialogue.md"),
            ("event", "event.md"),
            ("memory", "memory.md"),
        ):
            body = (corpus_dir / fname).read_text(encoding="utf-8")
            # 文件结构: `# 标题\n\n正文`，去掉标题行 + 两个换行后是正文
            expected_body = body.split("\n\n", 1)[1] if "\n\n" in body else ""
            assert result["word_counts"][key] == len(expected_body), (
                f"word_counts[{key}]={result['word_counts'][key]} "
                f"与正文长度 {len(expected_body)} 不符"
            )


def test_generate_oc_corpus_llm_called_4_times() -> None:
    """generate_oc_corpus 应对每个 writer 调一次 LLM（4 个 writer → 4 次调用）。"""
    setting = _make_setting()
    llm = FakeLLM(content="正文内容，至少凑几个字。")
    with tempfile.TemporaryDirectory() as td:
        generate_oc_corpus(setting, Path(td), "xiaoman", llm)
        assert llm.call_count == 4, (
            f"应调 LLM 4 次（4 个 writer），实际 {llm.call_count}"
        )


def test_generate_oc_corpus_llm_list_content() -> None:
    """LLM 返回 content 为 list（langchain 新版格式）时也应正确解析。"""
    setting = _make_setting()
    # 模拟 langchain 新版 content 为 list[dict]
    list_content = [
        {"type": "text", "text": "这是 list 格式 content 的正文。"},
    ]
    llm = FakeLLM(content=list_content)
    with tempfile.TemporaryDirectory() as td:
        result = generate_oc_corpus(setting, Path(td), "xiaoman", llm)
        corpus_dir = Path(result["corpus_dir"])
        body = (corpus_dir / "monologue.md").read_text(encoding="utf-8")
        assert "这是 list 格式 content 的正文。" in body, (
            f"list content 未被正确解析，文件内容: {body!r}"
        )


def test_generate_oc_corpus_llm_failure() -> None:
    """LLM 抛异常时 generate_oc_corpus 应向上抛（按 docstring 契约）。"""
    setting = _make_setting()
    llm = FakeLLM(raise_on_invoke=RuntimeError("fake LLM down"))
    with tempfile.TemporaryDirectory() as td:
        try:
            generate_oc_corpus(setting, Path(td), "xiaoman", llm)
        except RuntimeError as e:
            assert "fake LLM down" in str(e), (
                f"应透传 RuntimeError，实际: {e!r}"
            )
            return
        except Exception:  # noqa: BLE001
            # 允许被包装成别的异常类型，但必须向上抛
            return
    raise AssertionError("LLM 抛异常时 generate_oc_corpus 应向上抛，但未抛")
