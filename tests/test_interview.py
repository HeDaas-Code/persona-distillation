"""``interview.run_interview`` 单元测试。

Issue #13：已转为 pytest 风格——``test_*`` 函数由 pytest 自动收集，
直接用 ``assert`` 断言，不再需要 ``main()`` / ``try/except`` 包装。
所有 LLM 路径走 ``FakeInterviewLLM``，不调真实 API。

跑法：``python -m pytest tests/test_interview.py -v``
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from persona_distillation.intake.interview import run_interview
from persona_distillation.intake.oc_writer import OCSetting


# ---------------------------------------------------------------------------
# FakeLLM：根据消息内容区分 interviewer / character_player 返回不同文本
# ---------------------------------------------------------------------------
class _FakeResp:
    """模拟 LangChain 返回对象：有 ``.content`` 属性。"""

    def __init__(self, content: Any) -> None:
        self.content = content


class FakeInterviewLLM:
    """访谈用 Fake LLM：``invoke`` 根据 system prompt 内容区分调用方。

    - 第 1 条消息是 SystemMessage，content 含 "访谈" / "interviewer" 关键字 → 返回问题
    - 否则视为 character_player 调用 → 返回 OC 回答
    - ``call_count`` 记录调用次数；``question_calls`` / ``answer_calls`` 分别计数
    """

    def __init__(
        self,
        question: str = "你最珍视的一段回忆是什么？",
        answer: str = "嘛，再看看吧。大概是小时候在书店里第一次读完一整本书的那个下午。",
    ) -> None:
        self.question = question
        self.answer = answer
        self.call_count = 0
        self.question_calls = 0
        self.answer_calls = 0

    def invoke(self, messages: list) -> _FakeResp:
        self.call_count += 1
        # 第 1 条应是 SystemMessage，取其 content 判断调用方
        sys_content = ""
        if messages:
            first = messages[0]
            content = getattr(first, "content", "") or ""
            sys_content = str(content)
        # interviewer_system 含 "主理人" 关键字（CHARACTER_PLAYER_SYSTEM 仅含 "访谈者"，
        # 不含 "主理人"，故用 "主理人" 作为判别特征最稳）
        if "主理人" in sys_content:
            self.question_calls += 1
            return _FakeResp(self.question)
        # 兜底：默认走 character_player（answer）分支
        self.answer_calls += 1
        return _FakeResp(self.answer)


# ---------------------------------------------------------------------------
# 构造一份共用的 OC 设定 + 骨架目录
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


def _make_skeleton(workdir: Path, persona_id: str) -> Path:
    """在 ``<workdir>/<persona_id>/oc_corpus/`` 下造 4 个骨架文件。

    返回 corpus_dir 路径。
    """
    corpus_dir = workdir / persona_id / "oc_corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    # 4 个文件名与 oc_writer 保持一致
    for fname, label in (
        ("monologue.md", "独白"),
        ("dialogue.md", "对话"),
        ("event.md", "事件"),
        ("memory.md", "回忆"),
    ):
        (corpus_dir / fname).write_text(
            f"# {label} · 小满\n\n这是骨架正文片段。",
            encoding="utf-8",
        )
    return corpus_dir


# ---------------------------------------------------------------------------
# 用例
# ---------------------------------------------------------------------------
def test_run_interview_creates_markdown() -> None:
    """run_interview 应落盘 interview.md，含 # 角色访谈记录 / ## 第 N 轮 / **主理人** / **{name}**。"""
    setting = _make_setting()
    llm = FakeInterviewLLM()
    with tempfile.TemporaryDirectory() as td:
        workdir = Path(td)
        persona_id = "xiaoman"
        # 先造骨架目录
        _make_skeleton(workdir, persona_id)

        result = run_interview(setting, n_rounds=2, workdir=workdir,
                               persona_id=persona_id, llm=llm)

        # 返回 dict 含 path
        assert "path" in result, f"返回 dict 缺 path: {result}"
        out_path = Path(result["path"])
        assert out_path.exists(), f"interview.md 未落盘: {out_path}"
        # 落盘路径应是 <workdir>/<persona_id>/interview.md
        assert out_path == workdir / persona_id / "interview.md", (
            f"interview.md 路径不对: {out_path}，期望 {workdir / persona_id / 'interview.md'}"
        )

        # 内容格式校验
        content = out_path.read_text(encoding="utf-8")
        assert "# 角色访谈记录" in content, (
            f"interview.md 应含 '# 角色访谈记录'，实际: {content[:80]!r}"
        )
        # 每轮都应有 `## 第 N 轮`
        for i in (1, 2):
            assert f"## 第 {i} 轮" in content, (
                f"interview.md 缺 '## 第 {i} 轮'，内容: {content!r}"
            )
        # 主理人 / OC 名都应出现
        assert "**主理人**" in content, (
            f"interview.md 缺 '**主理人**'，内容: {content!r}"
        )
        assert f"**{setting.name}**" in content, (
            f"interview.md 缺 '**{setting.name}**'，内容: {content!r}"
        )


def test_run_interview_skeleton_not_found() -> None:
    """骨架目录不存在时 run_interview 应抛 FileNotFoundError，消息含"请先调 generate_oc_corpus"。"""
    setting = _make_setting()
    llm = FakeInterviewLLM()
    with tempfile.TemporaryDirectory() as td:
        workdir = Path(td)
        persona_id = "nonexistent"
        # 故意不造骨架目录
        try:
            run_interview(setting, n_rounds=2, workdir=workdir,
                          persona_id=persona_id, llm=llm)
        except FileNotFoundError as e:
            msg = str(e)
            assert "请先调 generate_oc_corpus" in msg, (
                f"FileNotFoundError 消息应含 '请先调 generate_oc_corpus'，实际: {msg!r}"
            )
            return
        except Exception as e:  # noqa: BLE001
            raise AssertionError(
                f"应抛 FileNotFoundError，实际抛 {type(e).__name__}: {e}"
            )
    raise AssertionError("应抛 FileNotFoundError，但未抛")


def test_run_interview_returns_correct_rounds() -> None:
    """run_interview 返回 dict 的 rounds 字段应等于 n_rounds。"""
    setting = _make_setting()
    llm = FakeInterviewLLM()
    n_rounds = 3
    with tempfile.TemporaryDirectory() as td:
        workdir = Path(td)
        persona_id = "xiaoman"
        _make_skeleton(workdir, persona_id)

        result = run_interview(setting, n_rounds=n_rounds, workdir=workdir,
                               persona_id=persona_id, llm=llm)

        assert "rounds" in result, f"返回 dict 缺 rounds: {result}"
        assert result["rounds"] == n_rounds, (
            f"rounds 应等于 {n_rounds}，实际 {result['rounds']}"
        )
        # 顺带验证：每轮 2 次 LLM 调用（interviewer + character_player），3 轮 = 6 次
        assert llm.call_count == n_rounds * 2, (
            f"LLM 调用次数应为 {n_rounds * 2}（每轮 interviewer + character_player），"
            f"实际 {llm.call_count}"
        )
        # interviewer / character_player 调用次数应各为 n_rounds
        assert llm.question_calls == n_rounds, (
            f"interviewer 调用次数应为 {n_rounds}，实际 {llm.question_calls}"
        )
        assert llm.answer_calls == n_rounds, (
            f"character_player 调用次数应为 {n_rounds}，实际 {llm.answer_calls}"
        )
