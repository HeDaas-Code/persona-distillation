"""``invoke_structured`` 的 JSON 抢救逻辑单元测试（Issue #2 / Task 6）。

背景：``invoke_structured`` 在 ``structured_response`` 缺失时，需从
AIMessage.content 里抢救 JSON。旧逻辑用 ``find("{")/rfind("}")`` 粗暴
截取，会把含花括号的自然语言或跨多对象文本误判为 JSON。新逻辑改用
栈匹配（``_extract_first_json_object``）提取第一个完整 JSON 对象，
并由 ``json.loads`` 拒绝非 JSON 候选。

跑法：``python -m pytest tests/test_invoke_structured.py -v``
"""
from __future__ import annotations

import json

from persona_distillation.agents import _extract_first_json_object


def test_extract_first_json_object_simple():
    """简单 JSON 对象正确提取。"""
    text = '{"a": 1}'
    assert _extract_first_json_object(text) == '{"a": 1}'


def test_extract_first_json_object_nested():
    """嵌套 JSON：提取最外层完整对象（内层花括号不提前闭合）。"""
    text = '{"a": {"b": 2}}'
    assert _extract_first_json_object(text) == '{"a": {"b": 2}}'


def test_extract_first_json_object_with_braces_in_string():
    """字符串值内的花括号不干扰配对计数。"""
    text = '{"key": "value with {brace}"}'
    assert _extract_first_json_object(text) == '{"key": "value with {brace}"}'


def test_extract_first_json_object_no_json():
    """纯文本（无花括号）返回 None。"""
    text = "关于这个角色，我们发现……"
    assert _extract_first_json_object(text) is None


def test_extract_first_json_object_natural_language_braces():
    """含花括号的自然语言：helper 返回花括号配对子串，json.loads 拒绝之。

    ``_extract_first_json_object`` 只做花括号配对，不验证 JSON 语法。
    ``关于{我们}发现`` 中 ``{我们}`` 花括号配对，故 helper 返回该子串；
    但 ``json.loads('{我们}')`` 抛 ``JSONDecodeError``，``invoke_structured``
    据此 continue 到下一条消息——这是系统级保护，避免误判自然语言为 JSON。
    """
    result = _extract_first_json_object("关于{我们}发现")
    # helper 返回花括号配对子串（非 None），但不是有效 JSON
    assert result == "{我们}"
    # json.loads 会拒绝，invoke_structured 据此继续下一条消息
    try:
        json.loads(result)
        raise AssertionError("json.loads 应当拒绝非 JSON 候选 '{我们}'")
    except json.JSONDecodeError:
        pass  # 预期失败


def test_extract_first_json_object_text_before_after():
    """JSON 前后均有自然语言：正确提取中间 JSON。"""
    text = '这是结果：{"a": 1} 以上。'
    assert _extract_first_json_object(text) == '{"a": 1}'
