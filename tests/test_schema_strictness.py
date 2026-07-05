"""``PersonaCard.extra='forbid'`` 严格 schema 测试。

背景：``PersonaCard`` 显式 ``ConfigDict(extra='forbid')``，
LLM/前端若塞 schema 外字段（典型如 ``tools`` / ``skills``），
``model_validate`` 必须抛 ``pydantic.ValidationError``，把错误"显式抛回"
而不是被默默吞掉。

本文件用 stdlib ``assert`` + 自定义 run 跑通，可直接：
``python -m tests.test_schema_strictness``。
"""
from __future__ import annotations

import sys
from typing import Any, Callable

from pydantic import ValidationError

from persona_distillation.schemas import PersonaCard


PASS = "[PASS]"
FAIL = "[FAIL]"

results: list[tuple[str, bool, str]] = []


def _run(name: str, fn: Callable[[], None]) -> None:
    """执行单条测试：失败打印 ``[FAIL]``；成功由测试函数自己打印 ``[PASS]``。"""
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        results.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"{FAIL} {name}  →  {type(e).__name__}: {e}")
    else:
        results.append((name, True, ""))


# ---------------------------------------------------------------------------
# 用例
# ---------------------------------------------------------------------------
def test_extra_forbid_tools_rejected() -> None:
    """塞入 schema 外的 ``tools`` 字段必须抛 ValidationError。"""
    bad = {
        "persona_id": "x",
        "system_prompt": "y",
        "error_reply": "z",
        "tools": {"mode": "default_all"},  # schema 外字段
    }
    try:
        PersonaCard.model_validate(bad)
    except ValidationError as e:
        # 错误信息应包含违规字段名 "tools"
        assert "tools" in str(e), f"ValidationError 文案不含 'tools': {e}"
        print(f"{PASS} test_extra_forbid_tools_rejected")
        return
    raise AssertionError("应抛 ValidationError，但 model_validate 静默通过")


def test_extra_forbid_skills_rejected() -> None:
    """塞入 schema 外的 ``skills`` 字段必须抛 ValidationError。"""
    bad = {
        "persona_id": "x",
        "system_prompt": "y",
        "error_reply": "z",
        "skills": [{"name": "fake-skill"}],  # schema 外字段
    }
    try:
        PersonaCard.model_validate(bad)
    except ValidationError as e:
        assert "skills" in str(e), f"ValidationError 文案不含 'skills': {e}"
        print(f"{PASS} test_extra_forbid_skills_rejected")
        return
    raise AssertionError("应抛 ValidationError，但 model_validate 静默通过")


def test_extra_allow_in_default_field() -> None:
    """合法字段（persona_id/system_prompt/error_reply/tags/...）应通过。"""
    good: dict[str, Any] = {
        "persona_id": "arakawa_sensei",
        "display_name": "荒川老师",
        "system_prompt": "嘛，再看看吧。",
        "error_reply": "嘛，网络出问题了。",
        "tags": ["严厉", "关心"],
        "traits_summary": "短句带叹词",
    }
    card = PersonaCard.model_validate(good)
    assert card.persona_id == "arakawa_sensei"
    assert card.display_name == "荒川老师"
    assert card.tags == ["严厉", "关心"]
    assert card.mental_models == []  # 默认空
    assert card.anti_patterns == []  # 默认空
    print(f"{PASS} test_extra_allow_in_default_field")


def test_extra_forbid_unknown_random_field() -> None:
    """任意未知字段（连 yaml 风格的也）必须被拒。"""
    bad = {
        "persona_id": "x",
        "system_prompt": "y",
        "error_reply": "z",
        "foo_bar": "baz",  # 任意未知
    }
    try:
        PersonaCard.model_validate(bad)
    except ValidationError as e:
        assert "foo_bar" in str(e), f"ValidationError 文案不含 'foo_bar': {e}"
        print(f"{PASS} test_extra_forbid_unknown_random_field")
        return
    raise AssertionError("应抛 ValidationError，但 model_validate 静默通过")


def test_extra_forbid_does_not_swallow_missing_required() -> None:
    """缺 system_prompt（真必填）仍必须抛 ValidationError。

    P0-修复：error_reply 已改为可选（pipeline 兜底），
    但 system_prompt 仍为必填——这是 LLM 必须返回的核心契约。
    """
    bad = {
        "persona_id": "x",
        # system_prompt 故意缺失
        "error_reply": "z",
    }
    try:
        PersonaCard.model_validate(bad)
    except ValidationError as e:
        assert "system_prompt" in str(e), f"ValidationError 文案不含 'system_prompt': {e}"
        print(f"{PASS} test_extra_forbid_does_not_swallow_missing_required")
        return
    raise AssertionError("应抛 ValidationError（缺 system_prompt），但静默通过")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 70)
    print("PersonaCard schema 严格性测试 (extra='forbid')")
    print("=" * 70)

    tests: list[tuple[str, Callable[[], None]]] = [
        ("test_extra_forbid_tools_rejected", test_extra_forbid_tools_rejected),
        ("test_extra_forbid_skills_rejected", test_extra_forbid_skills_rejected),
        ("test_extra_allow_in_default_field", test_extra_allow_in_default_field),
        ("test_extra_forbid_unknown_random_field", test_extra_forbid_unknown_random_field),
        ("test_extra_forbid_does_not_swallow_missing_required",
         test_extra_forbid_does_not_swallow_missing_required),
    ]

    for name, fn in tests:
        _run(name, fn)

    print("=" * 70)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"总计：{passed} 通过 / {failed} 失败 / {len(results)} 项")
    print("=" * 70)
    if failed:
        print("\n失败项：")
        for name, ok, detail in results:
            if not ok:
                print(f"  - {name}  {detail}")
        return 1
    print("\n所有 schema 严格性测试通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
