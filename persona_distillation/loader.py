"""多文本语料加载器（多模态聚合的文本侧实现）。

支持目录递归加载多种文本格式（.txt/.md/.markdown/.json/.jsonl/.csv/.log/.rst），
单文件输入也兼容。结构扩展点：在 ``_EXT_LOADERS`` 注册图像/PDF 描述器即可把
视觉/音频模态纳入聚合。
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import uuid as _uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".rst", ".log",
    ".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml",
}

# P2-1: 默认单文件输入大小上限（MB）。可通过 load_corpus(max_mb=...) 覆盖。
DEFAULT_MAX_INPUT_MB = 100


@dataclass
class LoadedDoc:
    """一篇加载后的文档。

    ``content_hash`` 是解码后正文的 SHA-256 hex，``corpus_uuid`` 是基于该哈希
    生成的确定性 UUID v5。两者都算在「解码后」的字符串上，因此编码不同但
    内容相同的文件（如 UTF-8 vs UTF-16）会得到相同的 hash / uuid，
    便于跨运行与跨源的去重和缓存命中。
    """

    path: str
    relpath: str
    ext: str
    text: str
    content_hash: str = ""
    corpus_uuid: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return Path(self.relpath).stem


def _read_text_auto(p: Path) -> str:
    """BOM 感知 + 多编码兜底的文本读取。

    顺序：
    1. UTF-16 LE/BE BOM → 对应 UTF-16 解码
    2. UTF-8 BOM → UTF-8 解码（去 BOM）
    3. 无 BOM：先试 UTF-8（严格），失败试 UTF-16，再失败试 GB18030（中文 Windows 常见）

    这样既能正确读 UTF-16 编码的中文小说（如轻小说文库导出的 .txt），
    也不会误伤普通 UTF-8 文件。
    """
    raw = p.read_bytes()
    if raw.startswith(b"\xff\xfe"):
        return raw[2:].decode("utf-16-le", errors="replace")
    if raw.startswith(b"\xfe\xff"):
        return raw[2:].decode("utf-16-be", errors="replace")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors="replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("utf-16")
    except UnicodeDecodeError:
        pass
    return raw.decode("gb18030", errors="replace")


def _load_text(p: Path) -> str:
    return _read_text_auto(p)


def _load_json(p: Path) -> str:
    data = json.loads(_read_text_auto(p))

    def _flatten(obj) -> str:
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            # 优先取常见的正文字段
            for k in ("text", "content", "body", "dialogue", "message"):
                if k in obj and isinstance(obj[k], str):
                    return obj[k]
            return json.dumps(obj, ensure_ascii=False)
        if isinstance(obj, list):
            return "\n".join(_flatten(x) for x in obj)
        return str(obj)

    return _flatten(data)


def _load_jsonl(p: Path) -> str:
    lines: list[str] = []
    for raw in _read_text_auto(p).splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            lines.append(raw)
            continue
        if isinstance(obj, dict):
            for k in ("text", "content", "body", "message", "utterance"):
                v = obj.get(k)
                if isinstance(v, str):
                    lines.append(v)
                    break
            else:
                lines.append(json.dumps(obj, ensure_ascii=False))
        elif isinstance(obj, str):
            lines.append(obj)
    return "\n".join(lines)


def _load_csv(p: Path, delimiter: str = ",") -> str:
    rows: list[str] = []
    # csv 模块需要文本模式流；先用 _read_text_auto 解码拿到正确字符串，再用 StringIO 喂 csv
    import io

    text = _read_text_auto(p)
    with io.StringIO(text, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            rows.append(delimiter.join(row))
    return "\n".join(rows)


_EXT_LOADERS: dict[str, Callable[[Path], str]] = {
    ".txt": _load_text,
    ".md": _load_text,
    ".markdown": _load_text,
    ".rst": _load_text,
    ".log": _load_text,
    ".yaml": _load_text,
    ".yml": _load_text,
    ".json": _load_json,
    ".jsonl": _load_jsonl,
    ".csv": _load_csv,
    ".tsv": lambda p: _load_csv(p, "\t"),
}


def _check_size(p: Path, max_mb: int) -> None:
    """P2-1: 大小上限校验。max_mb <= 0 表示不限制。"""
    if max_mb <= 0:
        return
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        logger.error("文件 %s 大小 %.1f MB 超过上限 %d MB", p, size_mb, max_mb)
        raise ValueError(
            f"文件 {p} 大小 {size_mb:.1f} MB 超过上限 {max_mb} MB。"
            f" 使用 load_corpus(input_path, max_mb=0) 禁用上限。"
        )


def load_corpus(input_path: str | Path, max_mb: int = DEFAULT_MAX_INPUT_MB) -> list[LoadedDoc]:
    """加载文件或目录，返回文档列表。

    Parameters:
        input_path: 文件或目录路径
        max_mb: 单文件大小上限（MB）。0 表示不限制。默认 100。

    多模态聚合语义：每个文件独立成篇，后续由分块器与蒸馏流水线分别处理，
    最终在合成阶段聚合为统一人格卡。
    """
    root = Path(input_path)
    if not root.exists():
        raise FileNotFoundError(f"输入路径不存在: {root}")

    if root.is_file():
        _check_size(root, max_mb)
        docs = [_load_one(root, root.parent)]
    else:
        docs: list[LoadedDoc] = []
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in _EXT_LOADERS:
                _check_size(p, max_mb)
                docs.append(_load_one(p, root))
    if not docs:
        raise ValueError(f"未在 {root} 找到任何可加载的文本文件，支持的后缀: {sorted(_TEXT_EXTS)}")
    logger.info("load_corpus: 加载了 %d 个文件 (max_mb=%d)", len(docs), max_mb)
    return docs


def _load_one(p: Path, base: Path) -> LoadedDoc:
    ext = p.suffix.lower()
    loader = _EXT_LOADERS.get(ext, _load_text)
    text = loader(p)
    # 哈希算在解码后的字符串上：编码不同但内容相同的文件会得到相同 hash / uuid。
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    corpus_uuid = str(_uuid.uuid5(_uuid.NAMESPACE_URL, content_hash))
    return LoadedDoc(
        path=str(p),
        relpath=str(p.relative_to(base)),
        ext=ext,
        text=text,
        content_hash=content_hash,
        corpus_uuid=corpus_uuid,
        meta={
            "size_chars": len(text),
            "content_hash": content_hash,
            "corpus_uuid": corpus_uuid,
        },
    )
