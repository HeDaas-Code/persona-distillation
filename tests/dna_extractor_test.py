"""``dna_extractor`` 单元测试。

设计原则：

- 不依赖 ``pytest``；每个 ``test_NN_*`` 函数用 ``assert`` + ``try/except`` 独立跑通，
  通过后打印 ``[PASS]``，失败打印 ``[FAIL] + 异常信息``。
- 也可被 ``pytest`` 收集（如果将来安装）—— 函数体以 ``test_`` 开头。
- 可直接运行：``python -m tests.dna_extractor_test``。

覆盖维度：

1. 节标题解析（中文方括号 / 全角 / 英文冒号 / Markdown / 乱序 / 部分缺失）
2. 心智模型三重验证过滤
3. 表达 DNA 子字段（高频词 / 比喻 / 开场白）解析
4. 回填函数的不可变 / no-op / warning 行为
"""
from __future__ import annotations

import logging
import sys
from typing import Any, Callable

from persona_distillation.intake.dna_extractor import (
    backfill_dna_from_system_prompt,
    extract_dna_from_system_prompt,
)
from persona_distillation.schemas import PersonaCard


PASS = "[PASS]"
FAIL = "[FAIL]"

results: list[tuple[str, bool, str]] = []


def _run(name: str, fn: Callable[[], None]) -> None:
    """执行单条测试：成功追加到 results，失败捕获后追加。

    测试函数内部自己 print ``[PASS]``；失败时本函数打印 ``[FAIL]``。
    """
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        results.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"{FAIL} {name}  →  {type(e).__name__}: {e}")
    else:
        results.append((name, True, ""))


# ---------------------------------------------------------------------------
# 节标题解析变体
# ---------------------------------------------------------------------------
def test_01_chinese_bracket_basic():
    sp = """[身份]
我是店主。
[性格]
- 外冷内热
- 话少
[心智模型]
- 时间定值：价值由时间判断
- 书脊-人脊同构：物的尊严即人的尊严
- 栖居而非消费：与物共处
[雷区]
- 讨价还价 → 拒绝
- 当面折书角 → 请出
"""
    r = extract_dna_from_system_prompt(sp)
    # P0-修复：backfill 是安全网，三验证不齐也保留（仅 synthesizer 主路径严守）
    assert len(r["mental_models"]) == 3  # 名称+原理 形式可识别
    assert len(r["anti_patterns"]) == 2
    print("[PASS] test_01_chinese_bracket_basic")


def test_02_chinese_fullwidth_bracket() -> None:
    """全角方括号【心智模型】应被识别。"""
    sp = """【心智模型】
- 慢判断：先停一拍再说
  - 跨域：写作、决策、聊天
  - 生成力：被催时也不脱口
  - 排他性：多数人靠速度
- 极简：用更少词
  - 跨域：日记、对话
  - 生成力：宁缺毋滥
  - 排他性：比多≠比好
【雷区】
- 抢话 → 沉默
"""
    r = extract_dna_from_system_prompt(sp)
    assert len(r["mental_models"]) == 2
    assert {m.name for m in r["mental_models"]} == {"慢判断", "极简"}
    assert len(r["anti_patterns"]) == 1
    assert r["anti_patterns"][0].pattern == "抢话"
    print(f"{PASS} test_02_chinese_fullwidth_bracket")


def test_03_english_colon() -> None:
    """英文冒号 Mental Models: 应被识别（验证子项仍用中文 key）。"""
    sp = """Mental Models:
- Focus: say no to 100 good ideas
  - 跨域：products, hiring
  - 生成力：when asked to grow, asks what to cut
  - 排他性：most people grow by addition
Anti-Patterns:
- Direct answer: refuse
"""
    r = extract_dna_from_system_prompt(sp)
    assert len(r["mental_models"]) == 1
    assert r["mental_models"][0].name.lower() == "focus"
    assert len(r["anti_patterns"]) == 1
    print(f"{PASS} test_03_english_colon")


def test_04_markdown_cn() -> None:
    """Markdown 中文标题 ## 心智模型 应被识别。"""
    sp = """## 心智模型
- 栖居而非消费：与物共处
  - 跨域：旧书、建筑、人际
  - 生成力：拒绝一次性消费场景
  - 排他性：消费主义主流
## 雷区
- 折书角 → 赶出
"""
    r = extract_dna_from_system_prompt(sp)
    assert len(r["mental_models"]) == 1
    assert r["mental_models"][0].name == "栖居而非消费"
    assert len(r["anti_patterns"]) == 1
    print(f"{PASS} test_04_markdown_cn")


def test_05_out_of_order() -> None:
    """节乱序时也应正确归类。"""
    sp = """[雷区]
- 折书角 → 赶出
[心智模型]
- 时间定值：时间即价值
  - 跨域：旧书、人际
  - 生成力：拒绝一次性买卖
  - 排他性：与效率最大化相反
[身份]
店主
"""
    r = extract_dna_from_system_prompt(sp)
    assert len(r["mental_models"]) == 1
    assert len(r["anti_patterns"]) == 1
    # 节顺序不影响结果
    assert r["anti_patterns"][0].pattern == "折书角"
    assert r["mental_models"][0].name == "时间定值"
    print(f"{PASS} test_05_out_of_order")


def test_06_partial_missing() -> None:
    """只出现部分节时，缺失字段应为空，不抛错。"""
    sp = """[雷区]
- 讨价还价 → 拒绝
"""
    r = extract_dna_from_system_prompt(sp)
    assert len(r["anti_patterns"]) == 1
    assert len(r["mental_models"]) == 0
    assert len(r["decision_heuristics"]) == 0
    assert len(r["honest_boundaries"]) == 0
    # expression_dna 仍应是完整 dict
    ed = r["expression_dna"]
    assert isinstance(ed, dict)
    assert "vocabulary" in ed
    assert "rhythm" in ed
    print(f"{PASS} test_06_partial_missing")


# ---------------------------------------------------------------------------
# 回填行为
# ---------------------------------------------------------------------------
def test_07_no_op_when_already_populated() -> None:
    """5 字段已全填时 backfill 不应修改。"""
    from persona_distillation.schemas import (
        AntiPattern,
        DecisionHeuristic,
        ExpressionDNA,
        HonestBoundary,
        MentalModel,
        VerificationResult,
    )

    card = PersonaCard(
        persona_id="sensei",
        system_prompt="[身份] 老师",
        error_reply="嘛，出错了。",
        expression_dna=ExpressionDNA(vocabulary=["嘛"]),
        mental_models=[
            MentalModel(
                name="聚焦",
                principle="说不",
                verification=VerificationResult(
                    cross_domain=True, generative=True, exclusive=True
                ),
            )
        ],
        decision_heuristics=[DecisionHeuristic(rule="先反问", trigger="学生问")],
        anti_patterns=[AntiPattern(pattern="直接给", reason="要学生想")],
        honest_boundaries=[HonestBoundary(limitation="不教数学", reason="外行")],
    )
    new_card = backfill_dna_from_system_prompt(card)
    # 5 字段都有，no-op：返回原对象
    assert new_card is card
    assert len(new_card.mental_models) == 1
    assert new_card.mental_models[0].name == "聚焦"
    print(f"{PASS} test_07_no_op_when_already_populated")


def test_08_empty_system_prompt() -> None:
    """空字符串输入应返回空字段，不抛异常。"""
    r = extract_dna_from_system_prompt("")
    assert r["mental_models"] == []
    assert r["anti_patterns"] == []
    assert r["decision_heuristics"] == []
    assert r["honest_boundaries"] == []
    assert r["expression_dna"]["vocabulary"] == []
    print(f"{PASS} test_08_empty_system_prompt")


# ---------------------------------------------------------------------------
# bullet / 嵌套 / 表达 DNA 子字段
# ---------------------------------------------------------------------------
def test_09_multi_line_bullets() -> None:
    """多行 bullet（不同前缀 + 嵌套）应正确切分。"""
    sp = """[心智模型]
* 慢判断：先停一拍
  - 跨域：写作、决策
  - 生成力：被催也不脱口
  - 排他性：常人靠速度
1. 极简：更少词
  - 跨域：日记、对话
  - 生成力：宁缺毋滥
  - 排他性：主流求多
[雷区]
1) 抢话 → 沉默
· 打断 → 走开
"""
    r = extract_dna_from_system_prompt(sp)
    # 两个心智模型都通过三验证
    assert len(r["mental_models"]) == 2
    names = {m.name for m in r["mental_models"]}
    assert names == {"慢判断", "极简"}
    # 三种 bullet 前缀都被吃
    assert len(r["anti_patterns"]) == 2
    patterns = {a.pattern for a in r["anti_patterns"]}
    assert patterns == {"抢话", "打断"}
    print(f"{PASS} test_09_multi_line_bullets")


def test_10_mental_model_with_triple_verification() -> None:
    """心智模型：完整三项齐→passed=True；缺项时 backfill 仍保留（passed=False）。

    P0-修复：backfill 是安全网，应保留所有可识别的项，仅在验证字段标 False。
    """
    sp = """[心智模型]
- 完整：三项齐
  - 跨域：a、b
  - 生成力：c
  - 排他性：d
- 缺生成力：少一项
  - 跨域：a、b
  - 排他性：d
- 缺排他性：少一项
  - 跨域：a
  - 生成力：c
"""
    r = extract_dna_from_system_prompt(sp)
    by_name = {m.name: m for m in r["mental_models"]}
    # 三项齐全的必须收
    assert "完整" in by_name
    assert by_name["完整"].verification.passed is True
    # backfill 路径：缺项仍保留，但 passed=False
    assert "缺生成力" in by_name
    assert by_name["缺生成力"].verification.passed is False
    assert "缺排他性" in by_name
    assert by_name["缺排他性"].verification.passed is False
    print(f"{PASS} test_10_mental_model_with_triple_verification")


def test_11_expression_dna_vocabulary() -> None:
    """[说话风格] 下「高频词：… / 词汇：…」应解析为 vocabulary 列表。

    注意：同 key 多条 bullet 会互相覆盖（最后一条胜出），
    故单条 bullet 内放全部词汇。
    """
    sp = """[说话风格]
- 高频词：嘛，唉，再看看，时间说了算
- 句式：短句带叹词
"""
    r = extract_dna_from_system_prompt(sp)
    vocab = r["expression_dna"]["vocabulary"]
    assert "嘛" in vocab
    assert "唉" in vocab
    assert "再看看" in vocab
    assert "时间说了算" in vocab
    assert r["expression_dna"]["rhythm"] == "短句带叹词"
    print(f"{PASS} test_11_expression_dna_vocabulary")


def test_12_signature_metaphors() -> None:
    """「比喻偏好：…」应解析为 signature_metaphors 列表。"""
    sp = """[表达DNA]
- 比喻偏好：书脊即人脊，时间是货币，物的尊严
"""
    r = extract_dna_from_system_prompt(sp)
    mets = r["expression_dna"]["signature_metaphors"]
    assert "书脊即人脊" in mets
    assert "时间是货币" in mets
    assert "物的尊严" in mets
    print(f"{PASS} test_12_signature_metaphors")


def test_13_opening_samples() -> None:
    """[开场白示范] 下「『…』」或裸引文应解析为 opening_samples。"""
    sp = """[开场白示范]
- 「嘛……欢迎来纸鱼堂。」
- "又来啦？"
"""
    r = extract_dna_from_system_prompt(sp)
    samples = r["expression_dna"]["opening_samples"]
    assert "嘛……欢迎来纸鱼堂。" in samples
    assert "又来啦？" in samples
    print(f"{PASS} test_13_opening_samples")


# ---------------------------------------------------------------------------
# 不可变 / 日志
# ---------------------------------------------------------------------------
def test_14_backfill_returns_new_object() -> None:
    """回填时返回新对象（model_copy），不修改入参。"""
    card = PersonaCard(
        persona_id="shop",
        system_prompt="[雷区]\n- 讨价还价 → 拒绝",
        error_reply="x",
    )
    new_card = backfill_dna_from_system_prompt(card)
    assert new_card is not card
    # 原 card 仍为空
    assert len(card.anti_patterns) == 0
    # 新 card 已填
    assert len(new_card.anti_patterns) >= 1
    print(f"{PASS} test_14_backfill_returns_new_object")


def test_15_backfill_logs_warning_when_empty() -> None:
    """DNA 全空 + system_prompt 空时，应记录 warning（来自 PersonaCard.model_post_init）。"""
    # 先装 handler，再创建 card，以便捕获 __init__ 期间的 warning
    caplog_records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            caplog_records.append(record)

    handler = _ListHandler(level=logging.WARNING)
    root = logging.getLogger()
    root.addHandler(handler)
    prev_level = root.level
    root.setLevel(logging.WARNING)
    try:
        card = PersonaCard(persona_id="empty", system_prompt="", error_reply="x")
        new_card = backfill_dna_from_system_prompt(card)
    finally:
        root.removeHandler(handler)
        root.setLevel(prev_level)

    # 不抛错、5 字段仍空
    assert new_card is card or len(new_card.anti_patterns) == 0
    # 应至少记录一条 WARNING（"DNA 字段全空" / "抽取后无任何可入字段" 等）
    warnings = [r for r in caplog_records if r.levelno >= logging.WARNING]
    assert len(warnings) >= 1, f"期望 ≥1 条 warning，实际 {len(warnings)}"
    # 至少有一条消息含 "DNA 字段全空"（model_post_init 的特征关键字）
    assert any("DNA 字段全空" in r.getMessage() for r in warnings), \
        f"warning 内容: {[r.getMessage()[:60] for r in warnings]}"
    print(f"{PASS} test_15_backfill_logs_warning_when_empty")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 70)
    print("dna_extractor 单元测试")
    print("=" * 70)

    tests: list[tuple[str, Callable[[], None]]] = [
        ("test_01_chinese_bracket_basic", test_01_chinese_bracket_basic),
        ("test_02_chinese_fullwidth_bracket", test_02_chinese_fullwidth_bracket),
        ("test_03_english_colon", test_03_english_colon),
        ("test_04_markdown_cn", test_04_markdown_cn),
        ("test_05_out_of_order", test_05_out_of_order),
        ("test_06_partial_missing", test_06_partial_missing),
        ("test_07_no_op_when_already_populated", test_07_no_op_when_already_populated),
        ("test_08_empty_system_prompt", test_08_empty_system_prompt),
        ("test_09_multi_line_bullets", test_09_multi_line_bullets),
        ("test_10_mental_model_with_triple_verification", test_10_mental_model_with_triple_verification),
        ("test_11_expression_dna_vocabulary", test_11_expression_dna_vocabulary),
        ("test_12_signature_metaphors", test_12_signature_metaphors),
        ("test_13_opening_samples", test_13_opening_samples),
        ("test_14_backfill_returns_new_object", test_14_backfill_returns_new_object),
        ("test_15_backfill_logs_warning_when_empty", test_15_backfill_logs_warning_when_empty),
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
    print("\n所有 dna_extractor 测试通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
