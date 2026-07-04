"""多文本语料加载器（多模态聚合的文本侧实现）。

支持目录递归加载多种文本格式（.txt/.md/.markdown/.json/.jsonl/.csv/.log/.rst），
单文件输入也兼容。结构扩展点：在 ``_EXT_LOADERS`` 注册图像/PDF 描述器即可把
视觉/音频模态纳入聚合。
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

_TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".rst", ".log",
    ".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml",
}


@dataclass
class LoadedDoc:
    """一篇加载后的文档。"""

    path: str
    relpath: str
    ext: str
    text: str
    meta: dict = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return Path(self.relpath).stem


def _load_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def _load_json(p: Path) -> str:
    data = json.loads(p.read_text(encoding="utf-8", errors="replace"))

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
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
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
    with p.open(encoding="utf-8", errors="replace", newline="") as f:
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


def load_corpus(input_path: str | Path) -> list[LoadedDoc]:
    """加载文件或目录，返回文档列表。

    多模态聚合语义：每个文件独立成篇，后续由分块器与蒸馏流水线分别处理，
    最终在合成阶段聚合为统一人格卡。
    """
    root = Path(input_path)
    if not root.exists():
        raise FileNotFoundError(f"输入路径不存在: {root}")

    if root.is_file():
        docs = [_load_one(root, root.parent)]
    else:
        docs: list[LoadedDoc] = []
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in _EXT_LOADERS:
                docs.append(_load_one(p, root))
    if not docs:
        raise ValueError(f"未在 {root} 找到任何可加载的文本文件，支持的后缀: {sorted(_TEXT_EXTS)}")
    return docs


def _load_one(p: Path, base: Path) -> LoadedDoc:
    ext = p.suffix.lower()
    loader = _EXT_LOADERS.get(ext, _load_text)
    text = loader(p)
    return LoadedDoc(
        path=str(p),
        relpath=str(p.relative_to(base)),
        ext=ext,
        text=text,
        meta={"size_chars": len(text)},
    )
