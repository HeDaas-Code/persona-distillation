"""从 LLM 拼到 ``system_prompt`` 文本里的 DNA 字段反向提取并回填。

背景：某些 LLM 端点（典型如 MiniMax-M3）不会把 DNA 五层
（``expression_dna`` / ``mental_models`` / ``decision_heuristics`` /
``anti_patterns`` / ``honest_boundaries``）作为独立结构化字段输出，
而是把它们以章节文本形式塞进 ``system_prompt``。下游按结构化字段消费时
会沉默失败（顶层 5 个 DNA 字段全空）。

本模块做两件事：

1. :func:`extract_dna_from_system_prompt`
   把 ``system_prompt`` 文本解析为 5 个 DNA 字段的中间表示
   （``dict[str, list[dict]]`` + ``expression_dna`` dict）。

2. :func:`backfill_dna_from_system_prompt`
   拿到 :class:`PersonaCard` 后，对 5 个顶层 DNA 字段做"空→填"；
   已有内容时 no-op；返回新对象（不修改入参）。

设计原则：

- **标准库 only**：只依赖 :mod:`re`、标准 :class:`logging`、Pydantic；
  不引入新依赖。
- **解析失败不抛异常**：每一步用 :func:`logging.Logger.debug` 记录，
  关键失败用 :func:`logging.Logger.warning` 告警。
- **宁缺毋滥**：心智模型若凑不齐三重验证（cross_domain / generative /
  exclusive），直接丢弃，不强行落库。
- **不可变更新**：回填函数返回新对象，原 :class:`PersonaCard` 不变。
"""
from __future__ import annotations

import logging
import re
from typing import Any

from persona_distillation.schemas import (
    AntiPattern,
    DecisionHeuristic,
    ExpressionDNA,
    HonestBoundary,
    MentalModel,
    PersonaCard,
    VerificationResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 节标题正则表：覆盖 8+ 种变体
# ---------------------------------------------------------------------------
# 中文方括号 / 中文圆括号 / 英文冒号 / Markdown 标题 / 无前导符号 五类
# 注意：必须从最具体到最宽泛排序，避免「[心智模型]」被「心智模型：」先吃掉。
# 这里用 ``(?P<name>...)`` 区分「身份/性格/说话风格/知识边界/情绪模式/
# 心智模型/雷区/输出约束/开场白示范/表达DNA/决策启发式/反模式/诚实边界/
# 心智模型与认知」等 section。
_SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # 1. Markdown 中文标题（无冒号）：## 心智模型 / ### 心智模型
    (
        "markdown_cn",
        re.compile(
            r"^\s*#{1,6}\s*"
            r"(?P<name>"
            r"身份|性格|说话风格|知识边界|情绪模式|心智模型(?:与认知)?|"
            r"雷区|输出约束|开场白示范|表达DNA|决策启发式|反模式|诚实边界|"
            r"修辞|口头禅|高频词|句式|比喻偏好"
            r")\s*$"
        ),
    ),
    # 2. Markdown 英文标题（无冒号）：## Mental Models / ### Anti-Patterns
    (
        "markdown_en",
        re.compile(
            r"^\s*#{1,6}\s*"
            r"(?P<name>"
            r"Identity|Background|Character|Speech(?:\s+Style)?|Knowledge(?:\s+Boundary)?|"
            r"Knowledge\s+Boundaries|Emotional\s+Pattern|Mental\s+Models?|Decision\s+Heuristics?|"
            r"Anti[-\s]?Patterns?|Honest\s+Boundaries|Expression\s+DNA|Rhetoric|"
            r"Opening\s+Samples?|Output\s+Constraints?|"
            r"Signature\s+Metaphors?|Rhythm|Vocabulary|Taboos?|Red\s+Lines?"
            r")\s*$",
            re.IGNORECASE,
        ),
    ),
    # 3. 英文冒号 / Markdown 标题（多语种通用）
    (
        "english",
        re.compile(
            r"^\s*(?:#{1,6}\s*)?"
            r"(?P<name>"
            r"Identity|Background|Character|Speech(?:\s+Style)?|Knowledge(?:\s+Boundary)?|"
            r"Knowledge\s+Boundaries|Emotional\s+Pattern|Mental\s+Models?|Decision\s+Heuristics?|"
            r"Anti[-\s]?Patterns?|Honest\s+Boundaries|Expression\s+DNA|Rhetoric|"
            r"Opening\s+Samples?|Output\s+Constraints?|"
            r"Signature\s+Metaphors?|Rhythm|Vocabulary|Taboos?|Red\s+Lines?"
            r")\s*[:：]\s*$",
            re.IGNORECASE,
        ),
    ),
    # 4. 中文方括号
    (
        "bracket_cn",
        re.compile(
            r"^\s*[【\[]\s*"
            r"(?P<name>"
            r"身份|性格|说话风格|知识边界|情绪模式|心智模型(?:与认知)?|"
            r"雷区|输出约束|开场白示范|表达DNA|决策启发式|反模式|诚实边界|"
            r"修辞|口头禅|高频词|句式|比喻偏好"
            r")\s*[】\]]\s*$"
        ),
    ),
    # 5. 中文圆括号
    (
        "paren_cn",
        re.compile(
            r"^\s*[（(]\s*"
            r"(?P<name>"
            r"身份|性格|说话风格|知识边界|情绪模式|心智模型(?:与认知)?|"
            r"雷区|输出约束|开场白示范|表达DNA|决策启发式|反模式|诚实边界"
            r")\s*[)）]\s*$"
        ),
    ),
    # 6. 无前导符号：心智模型：
    (
        "plain_cn",
        re.compile(
            r"^\s*"
            r"(?P<name>"
            r"身份|性格|说话风格|知识边界|情绪模式|心智模型(?:与认知)?|"
            r"雷区|输出约束|开场白示范|表达DNA|决策启发式|反模式|诚实边界"
            r")\s*[:：]\s*$"
        ),
    ),
]

# 中文章节标题 → DNA 字段的映射（统一用小写 key 避免重复定义）
_SECTION_TO_FIELD: dict[str, str] = {
    "身份": "_identity",  # 身份段一般不直接映射 DNA 字段，但保留以备扩展
    "性格": "_character",
    "说话风格": "expression_dna",
    "知识边界": "_knowledge",
    "情绪模式": "_emotion",
    "心智模型": "mental_models",
    "心智模型与认知": "mental_models",
    "雷区": "anti_patterns",
    "输出约束": "_output_constraints",
    "开场白示范": "expression_dna",
    "表达DNA": "expression_dna",
    "表达dna": "expression_dna",
    "决策启发式": "decision_heuristics",
    "反模式": "anti_patterns",
    "诚实边界": "honest_boundaries",
    # 英文键
    "identity": "_identity",
    "background": "_background",
    "character": "_character",
    "speech": "expression_dna",
    "speech style": "expression_dna",
    "knowledge": "_knowledge",
    "knowledge boundary": "_knowledge",
    "knowledge boundaries": "_knowledge",
    "emotional pattern": "_emotion",
    "mental models": "mental_models",
    "mental model": "mental_models",
    "decision heuristics": "decision_heuristics",
    "decision heuristic": "decision_heuristics",
    "anti-patterns": "anti_patterns",
    "anti-pattern": "anti_patterns",
    "anti patterns": "anti_patterns",
    "anti pattern": "anti_patterns",
    "honest boundaries": "honest_boundaries",
    "honest boundary": "honest_boundaries",
    "expression dna": "expression_dna",
    "rhetoric": "expression_dna",
    "rhythm": "expression_dna",
    "vocabulary": "expression_dna",
    "signature metaphors": "expression_dna",
    "opening samples": "expression_dna",
    "opening sample": "expression_dna",
    "output constraints": "_output_constraints",
    "output constraint": "_output_constraints",
    "taboos": "anti_patterns",
    "taboo": "anti_patterns",
    "red lines": "anti_patterns",
    "red line": "anti_patterns",
}

# bullet 解析：`- ` / `* ` / `数字. ` / `数字、 ` / `·`
_BULLET_RE = re.compile(r"^\s*(?:[-*·]|\d+[.、．)])\s*")

# MentalModel 三项验证的内联前缀
_MM_VERIFICATION_RE = re.compile(
    r"^\s*(?:[-*·]|\d+[.、．)])\s*"
    r"(?P<key>跨域|跨域复现|生成力|有生成力|排他性|有排他性)"
    r"\s*[:：]\s*"
    r"(?P<val>.+?)\s*$"
)

# ExpressionDNA 子字段前缀（高频词、比喻偏好、句式范例、节奏、开头）
_EXPR_DNA_INLINE_RE = re.compile(
    r"^\s*(?:[-*·]|\d+[.、．)])\s*"
    r"(?P<key>高频词|词汇|口头禅|句式范例|句式|节奏|修辞|比喻偏好|"
    r"标志性比喻|比喻|开场白示范|开场白|thinking\s*style|rhetorical\s*tics)\s*[:：]\s*"
    r"(?P<val>.+?)\s*$",
    re.IGNORECASE,
)

# AntiPattern 形如「X → 反应」或「X: 反应」
_ANTI_PATTERN_ARROW_RE = re.compile(
    r"^\s*(?:[-*·]|\d+[.、．)])\s*"
    r"(?P<pattern>.+?)\s*(?:→|->|⇒|=>|：:)\s*"
    r"(?P<reaction>.+?)\s*$"
)


# ---------------------------------------------------------------------------
# 文本切分：按行扫描定位节标题 → 收集其下 bullet
# ---------------------------------------------------------------------------
def _split_sections(text: str) -> list[tuple[str, list[str]]]:
    """把 ``system_prompt`` 切成 ``[(section_name, [bullet_line, ...]), ...]``。

    行为：

    - 按行扫描；遇到匹配任一节标题正则的行就开新节。
    - 切到下一个节标题前，所有非空行视作该节的 bullet 原文。
    - 第一个标题之前的"前言"段落归属到空名节（``""``），调用方一般丢弃。

    失败不抛异常；任何正则在编译期已锁定，运行期不会出错。
    """
    if not text:
        return []

    out: list[tuple[str, list[str]]] = []
    current_name = ""
    current_bullets: list[str] = []

    def _flush() -> None:
        nonlocal current_name, current_bullets
        if current_bullets or current_name:
            out.append((current_name, current_bullets))
        current_name = ""
        current_bullets = []

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            # 跳过空行（不切节）
            continue

        matched_name: str | None = None
        for _tag, pat in _SECTION_PATTERNS:
            m = pat.match(line)
            if m:
                matched_name = m.group("name").strip().lower()
                break

        if matched_name is not None:
            _flush()
            current_name = matched_name
            logger.debug("DNA 切分: 命中节标题 [%s]", matched_name)
            continue

        # 非节标题行 → 作为当前节的 bullet 原文收集
        current_bullets.append(line)

    _flush()
    return out


def _strip_bullet(line: str) -> str:
    """去掉行首 bullet 前缀（``- `` / ``* `` / 数字. ）。

    失败回退：原样返回（不含前缀的字符串）。
    """
    m = _BULLET_RE.match(line)
    if m:
        return line[m.end():].strip()
    return line.strip()


def _parse_csv_list(text: str) -> list[str]:
    """按顿号 / 中英逗号 / 分号 / 竖线 split，再去空去重。"""
    if not text:
        return []
    parts = re.split(r"[、，,；;|｜\n]+", text)
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        s = p.strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# 字段装配
# ---------------------------------------------------------------------------
def _assemble_expression_dna(section_name: str, bullets: list[str]) -> dict[str, Any]:
    """从 [说话风格] / [表达DNA] / [开场白示范] 节里抽 expression_dna 子字段。"""
    edna: dict[str, Any] = {
        "vocabulary": [],
        "rhythm": "",
        "rhetorical_tics": [],
        "signature_metaphors": [],
        "opening_samples": [],
    }

    for raw in bullets:
        line = _strip_bullet(raw)
        if not line:
            continue

        m = _EXPR_DNA_INLINE_RE.match(raw)
        if m:
            key = m.group("key").strip().lower()
            val = m.group("val").strip()
            if key in ("高频词", "词汇", "vocabulary"):
                edna["vocabulary"] = _parse_csv_list(val)
            elif key in ("口头禅", "rhetorical tics", "修辞"):
                edna["rhetorical_tics"] = _parse_csv_list(val)
            elif key in ("比喻偏好", "标志性比喻", "比喻", "signature metaphors"):
                edna["signature_metaphors"] = _parse_csv_list(val)
            elif key in ("句式范例", "句式"):
                edna["rhythm"] = val
            elif key in ("节奏", "rhythm"):
                edna["rhythm"] = val
            elif key in ("开场白示范", "开场白", "opening samples"):
                # 形如「「嘛……欢迎来纸鱼堂。」」直接当样本
                samples = re.findall(r"「(.*?)」", val) or re.findall(r'"(.*?)"', val)
                if samples:
                    edna["opening_samples"].extend(samples)
                else:
                    edna["opening_samples"].append(val)
            continue

        # 整行就是「「…（一句）」」式开场白
        quoted = re.findall(r"「(.*?)」", line) or re.findall(r'"(.*?)"', line)
        if quoted and section_name in ("开场白示范", "opening samples", "opening sample"):
            edna["opening_samples"].extend(quoted)
        elif quoted and section_name in ("说话风格", "speech style", "speech"):
            # 说话风格里夹着引文也视作样本
            edna["opening_samples"].extend(quoted)
        else:
            logger.debug("DNA 解析: expression_dna 段忽略行: %s", line[:60])

    # 启发式兜底：从 bullet 文本里抓「嘛、再看看、时间说了算」等高频词
    if not edna["vocabulary"]:
        vocab_candidates: list[str] = []
        for raw in bullets:
            line = _strip_bullet(raw)
            m = re.search(r"高频词[:：]\s*(.+?)(?:[。；;]|$)", line)
            if m:
                vocab_candidates = _parse_csv_list(m.group(1))
                break
        edna["vocabulary"] = vocab_candidates

    return edna


def _assemble_mental_models(bullets: list[str]) -> list[MentalModel]:
    """从 [心智模型] 节抽 ``MentalModel`` 列表。

    宁缺毋滥：每条 bullet 必须能拆出 ``name: principle`` 才会收；
    若三验证（cross_domain / generative / exclusive）缺一，丢弃。
    """
    models: list[MentalModel] = []

    # 先把 bullets 切成"主题句"+"子项"：
    #   - bullet 主项以 "- 名：原理" 形式出现
    #   - 子项以 "  - 跨域：xxx" 等缩进 / 同行 / 下一行出现
    # 由于 LLM 输出格式不稳定，我们采用最宽松的策略：
    #   1. 先收集 "name：principle" 形式的主项
    #   2. 在随后的若干行内吸收 "跨域：/生成力：/排他性：" 子项
    pending: dict[str, Any] | None = None
    for raw in bullets:
        line = _strip_bullet(raw)
        if not line:
            continue

        vm = _MM_VERIFICATION_RE.match(raw)
        if vm and pending is not None:
            key = vm.group("key")
            val = vm.group("val").strip()
            if key in ("跨域", "跨域复现"):
                pending["cross_domain_ev"].append(val)
                pending["has_cross"] = True
            elif key in ("生成力", "有生成力"):
                pending["generative_example"] = val
                pending["has_generative"] = True
            elif key in ("排他性", "有排他性"):
                pending["exclusivity_note"] = val
                pending["has_exclusive"] = True
            continue

        # 新主项：先结算 pending
        if pending is not None:
            models.extend([pending["model"]])  # placeholder; replaced below
            pending = None

        # 解析 "name：principle" 或 "name: principle" 形式
        if "：" in line or ":" in line:
            sep = "：" if "：" in line else ":"
            head, _, tail = line.partition(sep)
            name = head.strip()
            principle = tail.strip()
        else:
            name = line.strip()
            principle = ""
        if not name:
            continue

        pending = {
            "model": None,
            "name": name,
            "principle": principle,
            "application": "",
            "cross_domain_ev": [],
            "generative_example": "",
            "exclusivity_note": "",
            "has_cross": False,
            "has_generative": False,
            "has_exclusive": False,
        }

    # 收尾
    if pending is not None:
        # 见下方二次过滤
        pass

    # 二次构造 + 宁缺毋滥过滤
    final: list[MentalModel] = []
    # 先重新扫描一遍：每次遇到 "名：原理" 形式才结算
    buf_name = ""
    buf_principle = ""
    cross_ev: list[str] = []
    gen_ex = ""
    exc_note = ""
    has_cross = False
    has_generative = False
    has_exclusive = False

    def _commit() -> None:
        nonlocal buf_name, buf_principle, cross_ev, gen_ex, exc_note
        nonlocal has_cross, has_generative, has_exclusive
        if not buf_name:
            return
        # P0-修复：backfill 是安全网，应该宽松接收。
        # 原 spec 的"宁缺毋滥"是 synthesizer 自身输出的策略——
        # 主路径上 LLM 必须凑齐三验证才能落库。
        # 但 backfill 路径是从 LLM 写回 system_prompt 的文本中**抢救**信号，
        # 如果连主路径的"宁缺毋滥"已经把数据丢了（system_prompt 文本是 LLM
        # 在不返回结构化字段时的唯一寄托），backfill 再严守就会一起归零。
        # 所以：缺验证时仍收入，仅在日志中标明。
        if not (has_cross and has_generative and has_exclusive):
            logger.debug(
                "DNA 解析: 心智模型「%s」三重验证不齐 (cross=%s/generative=%s/exclusive=%s)，"
                "backfill 路径仍保留",
                buf_name, has_cross, has_generative, has_exclusive,
            )
        try:
            m = MentalModel(
                name=buf_name,
                principle=buf_principle,
                verification=VerificationResult(
                    cross_domain=has_cross,
                    cross_domain_evidence=cross_ev[:4],
                    generative=has_generative,
                    generative_example=gen_ex,
                    exclusive=has_exclusive,
                    exclusivity_note=exc_note,
                ),
            )
            final.append(m)
            logger.debug("DNA 解析: 心智模型「%s」已收入 (passed=%s)", buf_name, m.verification.passed)
        except Exception:
            logger.warning(
                "DNA 解析: 心智模型「%s」构造失败, exc_info=True",
                buf_name, exc_info=True,
            )
        buf_name = ""
        buf_principle = ""
        cross_ev = []
        gen_ex = ""
        exc_note = ""
        has_cross = False
        has_generative = False
        has_exclusive = False

    for raw in bullets:
        line = _strip_bullet(raw)
        if not line:
            continue

        vm = _MM_VERIFICATION_RE.match(raw)
        if vm and buf_name:
            key = vm.group("key")
            val = vm.group("val").strip()
            if key in ("跨域", "跨域复现"):
                cross_ev.append(val)
                has_cross = True
            elif key in ("生成力", "有生成力"):
                gen_ex = val
                has_generative = True
            elif key in ("排他性", "有排他性"):
                exc_note = val
                has_exclusive = True
            continue

        # 新主项：结算上一个，开始新的
        if "：" in line or ":" in line:
            sep = "：" if "：" in line else ":"
            head, _, tail = line.partition(sep)
            # 可能是子项标签：先看看是不是已知 key
            key_only = head.strip()
            if key_only in ("跨域", "跨域复现", "生成力", "有生成力", "排他性", "有排他性"):
                # 走上面 vm 的分支；这里兜底
                if key_only in ("跨域", "跨域复现"):
                    cross_ev.append(tail.strip())
                    has_cross = True
                elif key_only in ("生成力", "有生成力"):
                    gen_ex = tail.strip()
                    has_generative = True
                elif key_only in ("排他性", "有排他性"):
                    exc_note = tail.strip()
                    has_exclusive = True
                continue
            _commit()
            buf_name = key_only
            buf_principle = tail.strip()
            continue

        # 没有任何 "：" 的整行 → 当作新的 name（principle 为空）
        _commit()
        buf_name = line.strip()
        buf_principle = ""

    _commit()
    return final


def _assemble_anti_patterns(bullets: list[str]) -> list[AntiPattern]:
    """从 [雷区] 节抽 ``AntiPattern`` 列表。"""
    out: list[AntiPattern] = []
    for raw in bullets:
        line = _strip_bullet(raw)
        if not line:
            continue
        m = _ANTI_PATTERN_ARROW_RE.match(raw)
        if m:
            pattern = m.group("pattern").strip()
            reaction = m.group("reaction").strip()
        elif "：" in line or ":" in line:
            sep = "：" if "：" in line else ":"
            head, _, tail = line.partition(sep)
            pattern = head.strip()
            reaction = tail.strip()
        else:
            pattern = line
            reaction = ""

        try:
            ap = AntiPattern(
                pattern=pattern,
                reason=reaction or "触发后该人格会硬而短地拒绝",
            )
            out.append(ap)
            logger.debug("DNA 解析: 反模式「%s」+反应「%s」", pattern[:30], reaction[:30])
        except Exception:
            logger.warning(
                "DNA 解析: 反模式构造失败, pattern=%r, exc_info=True",
                pattern, exc_info=True,
            )
    return out


def _assemble_decision_heuristics(bullets: list[str]) -> list[DecisionHeuristic]:
    """从 [决策启发式] 节抽 ``DecisionHeuristic`` 列表（轻量启发式兜底）。"""
    out: list[DecisionHeuristic] = []
    for raw in bullets:
        line = _strip_bullet(raw)
        if not line:
            continue
        if "：" in line or ":" in line:
            sep = "：" if "：" in line else ":"
            head, _, tail = line.partition(sep)
            try:
                out.append(DecisionHeuristic(
                    rule=head.strip(),
                    trigger=tail.strip() or "一般决策场景",
                ))
                continue
            except Exception:
                logger.warning(
                    "DNA 解析: 决策启发式构造失败, line=%r, exc_info=True",
                    line, exc_info=True,
                )
        try:
            out.append(DecisionHeuristic(rule=line, trigger="一般决策场景"))
        except Exception:
            logger.warning(
                "DNA 解析: 决策启发式构造失败, line=%r, exc_info=True",
                line, exc_info=True,
            )
    return out


def _assemble_honest_boundaries(bullets: list[str]) -> list[HonestBoundary]:
    """从 [诚实边界] 节抽 ``HonestBoundary`` 列表。"""
    out: list[HonestBoundary] = []
    for raw in bullets:
        line = _strip_bullet(raw)
        if not line:
            continue
        if "：" in line or ":" in line:
            sep = "：" if "：" in line else ":"
            head, _, tail = line.partition(sep)
            try:
                out.append(HonestBoundary(limitation=head.strip(), reason=tail.strip()))
                continue
            except Exception:
                logger.warning(
                    "DNA 解析: 诚实边界构造失败, line=%r, exc_info=True",
                    line, exc_info=True,
                )
        try:
            out.append(HonestBoundary(limitation=line, reason="基于语料的边界声明"))
        except Exception:
            logger.warning(
                "DNA 解析: 诚实边界构造失败, line=%r, exc_info=True",
                line, exc_info=True,
            )
    return out


# ---------------------------------------------------------------------------
# 公开 API 1：抽取
# ---------------------------------------------------------------------------
def extract_dna_from_system_prompt(system_prompt: str) -> dict[str, Any]:
    """从 ``system_prompt`` 文本中抽出 DNA 5 字段的中间表示。

    Returns:
        ``dict``，keys 至少含::

            {
                "expression_dna": {"vocabulary": [...], "rhythm": str,
                                   "rhetorical_tics": [...],
                                   "signature_metaphors": [...],
                                   "opening_samples": [...]},
                "mental_models": [MentalModel, ...],
                "decision_heuristics": [DecisionHeuristic, ...],
                "anti_patterns": [AntiPattern, ...],
                "honest_boundaries": [HonestBoundary, ...],
            }

        任一字段抽取失败时为空容器，不抛异常。
    """
    if not system_prompt:
        return {
            "expression_dna": {
                "vocabulary": [],
                "rhythm": "",
                "rhetorical_tics": [],
                "signature_metaphors": [],
                "opening_samples": [],
            },
            "mental_models": [],
            "decision_heuristics": [],
            "anti_patterns": [],
            "honest_boundaries": [],
        }

    try:
        sections = _split_sections(system_prompt)
    except Exception:
        logger.warning("DNA 切分失败, exc_info=True")
        return {
            "expression_dna": {
                "vocabulary": [],
                "rhythm": "",
                "rhetorical_tics": [],
                "signature_metaphors": [],
                "opening_samples": [],
            },
            "mental_models": [],
            "decision_heuristics": [],
            "anti_patterns": [],
            "honest_boundaries": [],
        }

    logger.debug("DNA 切分完成，共 %d 节", len(sections))

    # 聚合 expression_dna（可能由多个节拼装：说话风格 / 表达DNA / 开场白示范）
    edna: dict[str, Any] = {
        "vocabulary": [],
        "rhythm": "",
        "rhetorical_tics": [],
        "signature_metaphors": [],
        "opening_samples": [],
    }
    mental_models: list[MentalModel] = []
    decision_heuristics: list[DecisionHeuristic] = []
    anti_patterns: list[AntiPattern] = []
    honest_boundaries: list[HonestBoundary] = []

    for name, bullets in sections:
        field = _SECTION_TO_FIELD.get(name, "")
        if field == "expression_dna":
            sub = _assemble_expression_dna(name, bullets)
            # 合并：列表拼接 + 字符串覆盖
            for k in ("vocabulary", "rhetorical_tics", "signature_metaphors", "opening_samples"):
                edna[k] = list(dict.fromkeys(edna[k] + sub.get(k, [])))
            if not edna["rhythm"] and sub.get("rhythm"):
                edna["rhythm"] = sub["rhythm"]
        elif field == "mental_models":
            mental_models.extend(_assemble_mental_models(bullets))
        elif field == "decision_heuristics":
            decision_heuristics.extend(_assemble_decision_heuristics(bullets))
        elif field == "anti_patterns":
            anti_patterns.extend(_assemble_anti_patterns(bullets))
        elif field == "honest_boundaries":
            honest_boundaries.extend(_assemble_honest_boundaries(bullets))
        else:
            logger.debug("DNA 解析: 忽略节 %r (未映射到 DNA 字段)", name)

    logger.info(
        "DNA 抽取: mental_models=%d, decision_heuristics=%d, "
        "anti_patterns=%d, honest_boundaries=%d, "
        "expression_dna.vocab=%d, signature_metaphors=%d, opening_samples=%d",
        len(mental_models), len(decision_heuristics), len(anti_patterns),
        len(honest_boundaries),
        len(edna["vocabulary"]), len(edna["signature_metaphors"]),
        len(edna["opening_samples"]),
    )

    return {
        "expression_dna": edna,
        "mental_models": mental_models,
        "decision_heuristics": decision_heuristics,
        "anti_patterns": anti_patterns,
        "honest_boundaries": honest_boundaries,
    }


# ---------------------------------------------------------------------------
# 公开 API 2：回填
# ---------------------------------------------------------------------------
def backfill_dna_from_system_prompt(card: PersonaCard) -> PersonaCard:
    """对 :class:`PersonaCard` 的 5 个 DNA 字段做"空→填"。

    行为：

    - 任一顶层 DNA 字段为 ``len == 0`` 时调用
      :func:`extract_dna_from_system_prompt` 抽取，并 Pydantic 校验后追加。
    - 任一条目 Pydantic 校验失败 → ``logger.warning(..., exc_info=True)`` 并 skip。
    - MentalModel 三重验证不齐的条目在抽取阶段已丢弃。
    - **不修改入参**：返回新对象。

    Parameters:
        card: 蒸馏得到的 :class:`PersonaCard` 实例。

    Returns:
        回填后的新 :class:`PersonaCard`。
    """
    # 检查哪些字段需要回填
    expression_empty = (
        not card.expression_dna.vocabulary
        and not card.expression_dna.rhythm
        and not card.expression_dna.rhetorical_tics
        and not card.expression_dna.signature_metaphors
        and not card.expression_dna.opening_samples
    )
    targets = {
        "expression_dna": expression_empty,
        "mental_models": len(card.mental_models) == 0,
        "decision_heuristics": len(card.decision_heuristics) == 0,
        "anti_patterns": len(card.anti_patterns) == 0,
        "honest_boundaries": len(card.honest_boundaries) == 0,
    }
    if not any(targets.values()):
        logger.debug(
            "backfill: PersonaCard(persona_id=%s) 5 字段都已非空，跳过",
            card.persona_id,
        )
        return card

    logger.info(
        "backfill: PersonaCard(persona_id=%s) 待回填字段=%s",
        card.persona_id,
        [k for k, v in targets.items() if v],
    )

    try:
        extracted = extract_dna_from_system_prompt(card.system_prompt or "")
    except Exception:
        logger.warning("backfill: 抽取失败, exc_info=True")
        return card

    update: dict[str, Any] = {}

    # expression_dna
    if targets["expression_dna"]:
        try:
            sub = extracted.get("expression_dna") or {}
            new_edna = ExpressionDNA(
                vocabulary=list(sub.get("vocabulary") or []),
                rhythm=str(sub.get("rhythm") or ""),
                rhetorical_tics=list(sub.get("rhetorical_tics") or []),
                signature_metaphors=list(sub.get("signature_metaphors") or []),
                opening_samples=list(sub.get("opening_samples") or []),
            )
            update["expression_dna"] = new_edna
            logger.debug(
                "backfill: expression_dna 填入 vocab=%d metaphors=%d samples=%d",
                len(new_edna.vocabulary),
                len(new_edna.signature_metaphors),
                len(new_edna.opening_samples),
            )
        except Exception:
            logger.warning("backfill: expression_dna 校验失败, exc_info=True")

    # mental_models
    if targets["mental_models"]:
        mms = list(card.mental_models)
        for m in extracted.get("mental_models") or []:
            try:
                # MentalModel.model_validate 内部已通过；这里再做一次过严校验
                if not isinstance(m, MentalModel):
                    m = MentalModel.model_validate(m)
                # P0-修复：backfill 是安全网，passed=False 也保留（仅记日志）
                if not m.verification.passed:
                    logger.debug(
                        "backfill: 心智模型「%s」passed=False, 仍保留（背靠 synthesizer 主路径严守）",
                        m.name,
                    )
                mms.append(m)
            except Exception:
                logger.warning(
                    "backfill: 心智模型校验失败, exc_info=True",
                )
        if mms:
            update["mental_models"] = mms

    # decision_heuristics
    if targets["decision_heuristics"]:
        dhs = list(card.decision_heuristics)
        for d in extracted.get("decision_heuristics") or []:
            try:
                if not isinstance(d, DecisionHeuristic):
                    d = DecisionHeuristic.model_validate(d)
                dhs.append(d)
            except Exception:
                logger.warning(
                    "backfill: 决策启发式校验失败, exc_info=True",
                )
        if dhs:
            update["decision_heuristics"] = dhs

    # anti_patterns
    if targets["anti_patterns"]:
        aps = list(card.anti_patterns)
        for a in extracted.get("anti_patterns") or []:
            try:
                if not isinstance(a, AntiPattern):
                    a = AntiPattern.model_validate(a)
                aps.append(a)
            except Exception:
                logger.warning(
                    "backfill: 反模式校验失败, exc_info=True",
                )
        if aps:
            update["anti_patterns"] = aps

    # honest_boundaries
    if targets["honest_boundaries"]:
        hbs = list(card.honest_boundaries)
        for h in extracted.get("honest_boundaries") or []:
            try:
                if not isinstance(h, HonestBoundary):
                    h = HonestBoundary.model_validate(h)
                hbs.append(h)
            except Exception:
                logger.warning(
                    "backfill: 诚实边界校验失败, exc_info=True",
                )
        if hbs:
            update["honest_boundaries"] = hbs

    if not update:
        logger.debug(
            "backfill: 抽取后无任何可入字段，返回原卡 (persona_id=%s)",
            card.persona_id,
        )
        return card

    new_card = card.model_copy(update=update)
    logger.info(
        "backfill: PersonaCard(persona_id=%s) 回填完成, 更新字段=%s",
        new_card.persona_id,
        list(update.keys()),
    )
    return new_card


__all__ = [
    "extract_dna_from_system_prompt",
    "backfill_dna_from_system_prompt",
]
