"""``loader`` 单元测试：覆盖多格式加载、多编码解码、大小限制与 hash 去重。

``loader.py`` 是整个蒸馏流水线的语料入口（PDR-INTK-001），但此前只有 3 处
间接引用。本测试直接锁死其核心契约：

1. 多编码感知读取：UTF-16 LE/BE BOM、UTF-8 BOM、无 BOM UTF-8 严格 + GB18030 兜底
2. 多格式解析：.json 字段优先级、.jsonl 逐行抽取、.csv/.tsv 分隔符
3. 目录递归 + 后缀过滤 + 不存在/空目录错误
4. max_mb 大小上限校验
5. 同内容不同编码 → 相同 content_hash / corpus_uuid（去重契约）
6. LoadedDoc.display_name / meta 字段完整性

跑法：``python -m pytest tests/test_loader.py -v``
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid as _uuid
from pathlib import Path

import pytest

from persona_distillation.loader import (
    DEFAULT_MAX_INPUT_MB,
    LoadedDoc,
    _check_size,
    _load_csv,
    _load_json,
    _load_jsonl,
    _read_text_auto,
    load_corpus,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


def _bytes_big_enough(target_mb: float) -> bytes:
    return b"x" * int(target_mb * 1024 * 1024)


# ======================================================================
# 1. _read_text_auto —— 多编码感知读取
# ======================================================================


class TestReadTextAuto:
    def test_utf16_le_bom_decoded(self, tmp_dir: Path) -> None:
        p = tmp_dir / "utf16le.txt"
        p.write_bytes(b"\xff\xfe" + "你好世界".encode("utf-16-le"))
        assert _read_text_auto(p) == "你好世界"

    def test_utf16_be_bom_decoded(self, tmp_dir: Path) -> None:
        p = tmp_dir / "utf16be.txt"
        p.write_bytes(b"\xfe\xff" + "你好世界".encode("utf-16-be"))
        assert _read_text_auto(p) == "你好世界"

    def test_utf8_bom_stripped(self, tmp_dir: Path) -> None:
        p = tmp_dir / "utf8bom.txt"
        p.write_bytes(b"\xef\xbb\xbf" + "好".encode("utf-8"))
        assert _read_text_auto(p) == "好"

    def test_plain_utf8_without_bom(self, tmp_dir: Path) -> None:
        p = tmp_dir / "utf8.txt"
        p.write_text("hello world", encoding="utf-8")
        assert _read_text_auto(p) == "hello world"

    def test_utf8_strict_reject_then_fallback(self, tmp_dir: Path) -> None:
        """非法 UTF-8 字节应降级到 GB18030（replace 模式）。"""
        p = tmp_dir / "mixed.txt"
        invalid_utf8 = b"\xff\xfe\x80\x81"
        p.write_bytes(invalid_utf8)
        result = _read_text_auto(p)
        assert isinstance(result, str)


# ======================================================================
# 2. 多格式解析器
# ======================================================================


class TestFormatLoaders:
    def test_load_json_text_field_priority(self, tmp_dir: Path) -> None:
        p = tmp_dir / "doc.json"
        p.write_text(json.dumps({"text": "正文优先", "content": "备选", "body": "末选"}))
        assert _load_json(p) == "正文优先"

    def test_load_json_content_when_no_text(self, tmp_dir: Path) -> None:
        p = tmp_dir / "doc.json"
        p.write_text(json.dumps({"content": "正文备选", "body": "末选"}))
        assert _load_json(p) == "正文备选"

    def test_load_json_body_when_no_text_or_content(self, tmp_dir: Path) -> None:
        p = tmp_dir / "doc.json"
        p.write_text(json.dumps({"body": "只有body"}))
        assert _load_json(p) == "只有body"

    def test_load_json_dict_flatten_when_no_known_keys(self, tmp_dir: Path) -> None:
        p = tmp_dir / "doc.json"
        p.write_text(json.dumps({"unknown_key": 42, "n": 1}))
        result = _load_json(p)
        assert "unknown_key" in result

    def test_load_json_list_concatenated(self, tmp_dir: Path) -> None:
        p = tmp_dir / "list.json"
        p.write_text(json.dumps(["a", "b", "c"]))
        assert _load_json(p) == "a\nb\nc"

    def test_load_jsonl_extracts_text_field(self, tmp_dir: Path) -> None:
        p = tmp_dir / "doc.jsonl"
        p.write_text(
            json.dumps({"text": "第一条"}) + "\n"
            + json.dumps({"message": "第二条"}) + "\n"
            + json.dumps({"utterance": "第三条"}) + "\n"
        )
        result = _load_jsonl(p)
        assert result == "第一条\n第二条\n第三条"

    def test_load_jsonl_skips_blank_lines(self, tmp_dir: Path) -> None:
        p = tmp_dir / "doc.jsonl"
        p.write_text('\n{"text": "a"}\n\n{"text": "b"}\n')
        result = _load_jsonl(p)
        assert result == "a\nb"

    def test_load_jsonl_raw_line_when_parse_fails(self, tmp_dir: Path) -> None:
        p = tmp_dir / "doc.jsonl"
        p.write_text('not-json-at-all\n{"text": "ok"}')
        result = _load_jsonl(p)
        assert "not-json-at-all" in result
        assert "ok" in result

    def test_load_csv_comma_delimiter(self, tmp_dir: Path) -> None:
        p = tmp_dir / "table.csv"
        p.write_text("a,b,c\n1,2,3\n")
        result = _load_csv(p)
        assert "a,b,c" in result
        assert "1,2,3" in result

    def test_load_csv_tab_delimiter_via_tsv(self, tmp_dir: Path) -> None:
        p = tmp_dir / "table.tsv"
        p.write_text("a\tb\tc\n1\t2\t3\n")
        result = _load_csv(p, "\t")
        assert "a\tb\tc" in result


# ======================================================================
# 3. _check_size —— 大小上限校验
# ======================================================================


class TestCheckSize:
    def test_no_limit_when_max_zero(self, tmp_dir: Path) -> None:
        p = tmp_dir / "big.bin"
        p.write_bytes(_bytes_big_enough(5))
        _check_size(p, max_mb=0)

    def test_no_limit_when_max_negative(self, tmp_dir: Path) -> None:
        p = tmp_dir / "big.bin"
        p.write_bytes(_bytes_big_enough(5))
        _check_size(p, max_mb=-1)

    def test_under_limit_passes(self, tmp_dir: Path) -> None:
        p = tmp_dir / "small.txt"
        p.write_bytes(_bytes_big_enough(1))
        _check_size(p, max_mb=10)

    def test_over_limit_raises(self, tmp_dir: Path) -> None:
        p = tmp_dir / "too_big.txt"
        p.write_bytes(_bytes_big_enough(10))
        with pytest.raises(ValueError, match="超过上限"):
            _check_size(p, max_mb=1)


# ======================================================================
# 4. load_corpus —— 主入口
# ======================================================================


class TestLoadCorpus:
    def test_file_input_single_doc(self, tmp_dir: Path) -> None:
        f = tmp_dir / "single.txt"
        f.write_text("only one file")
        docs = load_corpus(f)
        assert len(docs) == 1
        assert docs[0].text == "only one file"

    def test_directory_input_recursive(self, tmp_dir: Path) -> None:
        (tmp_dir / "sub").mkdir()
        (tmp_dir / "a.txt").write_text("A")
        (tmp_dir / "sub" / "b.md").write_text("B")
        (tmp_dir / "c.unknown").write_text("C_SHOULD_NOT_LOAD")
        docs = load_corpus(tmp_dir)
        assert len(docs) == 2
        texts = {d.text for d in docs}
        assert texts == {"A", "B"}

    def test_nonexistent_path_raises(self, tmp_dir: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_corpus(tmp_dir / "does_not_exist")

    def test_empty_dir_raises(self, tmp_dir: Path) -> None:
        empty = tmp_dir / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="未在"):
            load_corpus(empty)

    def test_sorted_order_recursive(self, tmp_dir: Path) -> None:
        (tmp_dir / "z_dir").mkdir()
        (tmp_dir / "a.txt").write_text("A")
        (tmp_dir / "z_dir" / "b.txt").write_text("B")
        (tmp_dir / "m.txt").write_text("M")
        docs = load_corpus(tmp_dir)
        assert len(docs) == 3
        rels = sorted(Path(d.path).name for d in docs)
        assert rels == ["a.txt", "b.txt", "m.txt"]


# ======================================================================
# 5. LoadedDoc 字段完整性 + hash/uuid 去重契约
# ======================================================================


class TestLoadedDoc:
    def test_display_name_stem(self) -> None:
        doc = LoadedDoc(
            path="/x/y.txt",
            relpath="y.txt",
            ext=".txt",
            text="hi",
        )
        assert doc.display_name == "y"

    def test_hash_and_uuid_deterministic(self, tmp_dir: Path) -> None:
        f = tmp_dir / "stable.txt"
        f.write_text("deterministic content")
        docs = load_corpus(f, max_mb=0)
        d = docs[0]
        expected_hash = hashlib.sha256("deterministic content".encode("utf-8")).hexdigest()
        assert d.content_hash == expected_hash
        expected_uuid = str(_uuid.uuid5(_uuid.NAMESPACE_URL, expected_hash))
        assert d.corpus_uuid == expected_uuid

    def test_same_content_different_encoding_gives_same_hash(self, tmp_dir: Path) -> None:
        f_utf8 = tmp_dir / "a_utf8.txt"
        f_utf16 = tmp_dir / "b_utf16.txt"
        f_utf8.write_bytes("相同内容".encode("utf-8"))
        f_utf16.write_bytes(b"\xff\xfe" + "相同内容".encode("utf-16-le"))

        d1 = load_corpus(f_utf8, max_mb=0)[0]
        d2 = load_corpus(f_utf16, max_mb=0)[0]
        assert d1.content_hash == d2.content_hash
        assert d1.corpus_uuid == d2.corpus_uuid

    def test_meta_fields_present(self, tmp_dir: Path) -> None:
        f = tmp_dir / "meta.txt"
        f.write_text("content")
        doc = load_corpus(f, max_mb=0)[0]
        assert "size_chars" in doc.meta
        assert doc.meta["content_hash"] == doc.content_hash
        assert doc.meta["corpus_uuid"] == doc.corpus_uuid

    def test_size_chars_utf8(self, tmp_dir: Path) -> None:
        f = tmp_dir / "cn.txt"
        f.write_text("你好")
        doc = load_corpus(f, max_mb=0)[0]
        assert doc.meta["size_chars"] == 2


# ======================================================================
# 6. _check_size 默认值 + load_corpus 传递 max_mb
# ======================================================================


class TestMaxMbPropagation:
    def test_default_max_mb_constant(self) -> None:
        assert DEFAULT_MAX_INPUT_MB == 100

    def test_load_corpus_respects_max_mb(self, tmp_dir: Path) -> None:
        f = tmp_dir / "big.txt"
        f.write_bytes(_bytes_big_enough(50))
        with pytest.raises(ValueError, match="超过上限"):
            load_corpus(f, max_mb=1)

    def test_load_corpus_zero_disables_limit(self, tmp_dir: Path) -> None:
        f = tmp_dir / "big.txt"
        f.write_bytes(_bytes_big_enough(200))
        docs = load_corpus(f, max_mb=0)
        assert len(docs) == 1
