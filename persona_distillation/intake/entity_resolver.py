"""跨 chunk 实体归并器。

NER 在单 chunk 内做名字规范化，但不同 chunk 之间没有归并：同一人物在不同
chunk 里被识别成「荒川善次」「老师」「荒川」三个独立条目。本模块在
:class:`IndexStore` 上做后置归并，采用 **三重信号融合** 判断是否同一人：

a. **别名交叉**：A 的 aliases 含 B 的 name（或反之）。
b. **字符串相似**：Levenshtein 距离 ≤ 2 *或* Jaro-Winkler ≥ 0.85。
c. **嵌入相似**：取各人物 top-5 evidence 的 embedding 平均值，cosine ≥ 0.8。

判定规则：
- 命中 ≥ 2 重 → 自动合并（``auto_merge=True`` 时调
  :meth:`IndexStore.merge_characters`）。
- 命中 1 重 → 标记「待确认」（不合并，留给人工 / LLM 后续审）。
- 命中 0 重 → 不动。

降级：无 ``llm`` / 无 embedding（``HashEmbeddings`` 或 ``store._embedding``
为 ``None``）时，c 信号不可用，只做 a+b。命中 1 重即标记「待确认」，
不做自动合并（避免误并）。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from persona_distillation.intake.embedder import HashEmbeddings
from persona_distillation.intake.index_store import IndexStore
from persona_distillation.intake.schemas import IndexCategory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 字符串相似度：纯 Python 实现 Levenshtein + Jaro-Winkler
# ---------------------------------------------------------------------------
def levenshtein(a: str, b: str) -> int:
    """计算两字符串的 Levenshtein 编辑距离（纯 Python，O(len(a)*len(b))）。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # 滚动数组，省内存
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(
                cur[j - 1] + 1,        # 插入
                prev[j] + 1,           # 删除
                prev[j - 1] + cost,    # 替换
            )
        prev = cur
    return prev[-1]


def jaro(a: str, b: str) -> float:
    """Jaro 相似度（0~1，1 表示完全相同）。"""
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    # 匹配窗口：max(len(a), len(b)) // 2 - 1
    match_dist = max(la, lb) // 2 - 1
    if match_dist < 0:
        match_dist = 0
    a_match = [False] * la
    b_match = [False] * lb
    matches = 0
    for i in range(la):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, lb)
        for j in range(start, end):
            if b_match[j] or a[i] != b[j]:
                continue
            a_match[i] = True
            b_match[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    # 计算转置数
    transpositions = 0
    k = 0
    for i in range(la):
        if not a_match[i]:
            continue
        while not b_match[k]:
            k += 1
        if a[i] != b[k]:
            transpositions += 1
        k += 1
    transpositions //= 2
    return (
        matches / la
        + matches / lb
        + (matches - transpositions) / matches
    ) / 3.0


def jaro_winkler(a: str, b: str, prefix_weight: float = 0.1) -> float:
    """Jaro-Winkler 相似度（在 Jaro 基础上对公共前缀加权）。"""
    j = jaro(a, b)
    # 公共前缀最多 4 字符
    prefix = 0
    for i in range(min(4, len(a), len(b))):
        if a[i] == b[i]:
            prefix += 1
        else:
            break
    return j + prefix * prefix_weight * (1.0 - j)


def strings_similar(a: str, b: str, *, lev_max: int = 2, jw_min: float = 0.85) -> bool:
    """字符串相似判定：Levenshtein ≤ ``lev_max`` 或 Jaro-Winkler ≥ ``jw_min``。"""
    if not a or not b:
        return False
    if a == b:
        return True
    if levenshtein(a, b) <= lev_max:
        return True
    return jaro_winkler(a, b) >= jw_min


# ---------------------------------------------------------------------------
# 嵌入相似度
# ---------------------------------------------------------------------------
def _cosine(u: list[float], v: list[float]) -> float:
    """余弦相似度（向量已 L2 归一化时退化为点积，这里仍做归一化兜底）。"""
    if not u or not v or len(u) != len(v):
        return 0.0
    nu = math.sqrt(sum(x * x for x in u)) or 1.0
    nv = math.sqrt(sum(x * x for x in v)) or 1.0
    return sum(x * y for x, y in zip(u, v)) / (nu * nv)


def _character_mean_embedding(
    store: IndexStore,
    name: str,
    embedder: Any,
    *,
    top_k: int = 5,
) -> list[float] | None:
    """取该人物 top-K evidence（按 chunk_index 顺序取前 K 条）的 embedding 平均。

    调用方应保证 ``embedder`` 不是 :class:`HashEmbeddings`（伪嵌入），否则
    相似度无意义。失败时返回 ``None``。
    """
    entries = store.get_character_entries(name)
    if not entries:
        return None
    # 优先用 speech / appearance（人物特征更明显的类别），再补 event
    entries.sort(key=lambda e: (
        0 if e.category == IndexCategory.SPEECH else
        1 if e.category == IndexCategory.APPEARANCE else 2,
        e.chunk_index,
    ))
    sample = entries[: max(top_k, 1)]
    texts = [e.text for e in sample if e.text]
    if not texts:
        return None
    try:
        vecs = embedder.embed_documents(texts)
    except Exception as e:  # noqa: BLE001
        logger.warning("embed_documents 失败 (name=%s): %s", name, e)
        return None
    if not vecs:
        return None
    dim = len(vecs[0])
    summed = [0.0] * dim
    for v in vecs:
        for i, x in enumerate(v):
            summed[i] += x
    n = len(vecs)
    mean = [x / n for x in summed]
    return mean


# ---------------------------------------------------------------------------
# 信号判定
# ---------------------------------------------------------------------------
@dataclass
class CharacterSignals:
    """对一对 (A, B) 人物计算的合并信号。"""

    name_a: str
    name_b: str
    alias_hit: bool = False      # 别名交叉
    string_hit: bool = False     # 字符串相似
    embedding_hit: bool = False  # 嵌入相似
    embedding_score: float = 0.0  # 嵌入 cosine 分数（diagnostic）

    @property
    def hit_count(self) -> int:
        return int(self.alias_hit) + int(self.string_hit) + int(self.embedding_hit)


@dataclass
class ResolveResult:
    """``resolve_entities`` 的返回。"""

    auto_merged: list[tuple[str, str]] = field(default_factory=list)
    """已自动合并的 ``(source, target)`` 列表（source 已并入 target）。"""

    pending: list[CharacterSignals] = field(default_factory=list)
    """仅命中 1 重、未自动合并、留待人工确认的对照。"""

    skipped: list[tuple[str, str, str]] = field(default_factory=list)
    """被跳过的对照 ``(name_a, name_b, reason)``。"""

    @property
    def total_actions(self) -> int:
        return len(self.auto_merged) + len(self.pending)


def _embedding_available(store: IndexStore, llm: Any) -> tuple[bool, Any]:
    """检查 store 是否有可用的「真」嵌入（非 HashEmbeddings / 非 None）。"""
    emb = getattr(store, "_embedding", None)
    if emb is None:
        return False, None
    if isinstance(emb, HashEmbeddings):
        return False, None
    # llm 在这里仅用于决定是否启用更激进的合并策略；嵌入判定本身不依赖 LLM
    return True, emb


def _pick_target(a: str, b: str, store: IndexStore) -> tuple[str, str]:
    """从一对人物里选合并 target：选 mention_count 更多的；并列时选字符更长的。"""
    cur = store._conn.execute(
        "SELECT character_name, COUNT(*) FROM entries "
        "WHERE character_name IN (?, ?) GROUP BY character_name",
        (a, b),
    )
    counts = {name: n for name, n in cur.fetchall()}
    ca = counts.get(a, 0)
    cb = counts.get(b, 0)
    if ca > cb:
        return a, b  # (target, source)
    if cb > ca:
        return b, a
    # 并列：选字符更长的（通常是更规范的称谓，如「荒川善次」>「荒川」）
    if len(a) >= len(b):
        return a, b
    return b, a


def resolve_entities(
    store: IndexStore,
    llm: Any = None,
    auto_merge: bool = True,
    threshold: float = 0.85,
    *,
    embedding_cos_min: float = 0.8,
    top_k_evidence: int = 5,
) -> ResolveResult:
    """跨 chunk 实体归并。

    Args:
        store: 已建立索引的 :class:`IndexStore`。
        llm: 预留（当前归并不依赖 LLM；未来可加 LLM 复核「待确认」对）。
        auto_merge: 是否对命中 ≥2 重的对照自动调用
            :meth:`IndexStore.merge_characters`。``False`` 时所有命中都进
            ``pending``。
        threshold: 字符串相似度阈值（Jaro-Winkler 下限，Levenshtein 仍按 ≤2）。
        embedding_cos_min: 嵌入相似判定下限（cosine）。
        top_k_evidence: 计算人物 mean embedding 时取的 evidence 条数。

    Returns:
        :class:`ResolveResult`，含 ``auto_merged`` / ``pending`` / ``skipped``。

    降级：当 ``store`` 没有真嵌入（``HashEmbeddings`` / ``None``）时，
    只用别名 + 字符串两重信号；命中 1 重即标记「待确认」，不做自动合并。
    """
    result = ResolveResult()
    chars = store.list_characters()
    names = [c["character_name"] for c in chars]
    if len(names) < 2:
        return result

    # 预取 aliases 映射：name -> set(aliases)
    aliases_map: dict[str, set[str]] = {
        c["character_name"]: set(c.get("aliases", [])) for c in chars
    }

    embed_available, embedder = _embedding_available(store, llm)
    # 预计算各人物 mean embedding（懒加载，按需算）
    emb_cache: dict[str, list[float] | None] = {}

    def _emb_of(n: str) -> list[float] | None:
        if n not in emb_cache:
            emb_cache[n] = (
                _character_mean_embedding(store, n, embedder, top_k=top_k_evidence)
                if embed_available
                else None
            )
        return emb_cache[n]

    # 已被合并的人物跳过
    consumed: set[str] = set()
    for i, a in enumerate(names):
        if a in consumed:
            continue
        for b in names[i + 1 :]:
            if b in consumed:
                continue
            sig = CharacterSignals(name_a=a, name_b=b)
            # a. 别名交叉
            if b in aliases_map.get(a, set()) or a in aliases_map.get(b, set()):
                sig.alias_hit = True
            # b. 字符串相似
            if strings_similar(a, b, jw_min=threshold):
                sig.string_hit = True
            # c. 嵌入相似
            if embed_available:
                ea, eb = _emb_of(a), _emb_of(b)
                if ea is not None and eb is not None:
                    score = _cosine(ea, eb)
                    sig.embedding_score = score
                    sig.embedding_hit = score >= embedding_cos_min

            # 决策：命中 ≥2 重 → 自动合并；命中 1 重 → 待确认；0 重 → 跳过。
            # 无嵌入时也要求 2 重（即 alias+string 同时命中）才合并，避免误并。
            if sig.hit_count >= 2:
                if auto_merge:
                    target, source = _pick_target(a, b, store)
                    try:
                        moved = store.merge_characters(source, target)
                        if moved > 0:
                            result.auto_merged.append((source, target))
                            consumed.add(source)
                            # 把 source 的 aliases 也并入 target（内存视图同步）
                            aliases_map[target] |= aliases_map.get(source, set())
                            aliases_map[target].add(source)
                            aliases_map.pop(source, None)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("merge_characters 失败 %s→%s: %s", source, target, e)
                        result.skipped.append((a, b, f"merge_failed: {e}"))
                else:
                    result.pending.append(sig)
            elif sig.hit_count == 1:
                result.pending.append(sig)
            # hit_count == 0：不收集（绝大多数对都是 0）

    logger.info(
        "resolve_entities: 自动合并 %d 对 / 待确认 %d 对 / 跳过 %d 对 (embedding=%s)",
        len(result.auto_merged), len(result.pending), len(result.skipped),
        embed_available,
    )
    return result
