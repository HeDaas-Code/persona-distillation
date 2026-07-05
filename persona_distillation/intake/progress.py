"""分块解析进度指示器。

向 stderr 输出实时进度条，不干扰 LLM 对话流（stdout 留给 LLM）。
"""
from __future__ import annotations

import sys


class ProgressReporter:
    """实时进度指示器，输出到 stderr。

    用 ``\\r`` 原地更新，不换行，避免污染 LLM 对话流。

    Example::

        reporter = ProgressReporter(total=30, label="侦探AI.txt", show=True)
        for i, chunk in enumerate(chunks):
            # ... 处理 chunk ...
            reporter.update(done=i + 1, current_names="相以, 合尾创")
        reporter.finish("30/30 块, 新增 145 条提及, 跳过 0 块")
    """

    def __init__(self, total: int, label: str, *, show: bool = True) -> None:
        self.total = total
        self.label = label
        self.show = show
        self._started = False

    def update(self, done: int, current_names: str = "") -> None:
        """更新进度条。

        Args:
            done: 已完成数（含跳过的缓存命中）
            current_names: 当前 chunk 提取到的人物名，逗号分隔（可空）
        """
        if not self.show:
            return
        if self.total > 0:
            pct = done * 100 // self.total
            pct_str = f"{pct}%"
            progress = f"[{done}/{self.total}]"
        else:
            pct_str = "?"
            progress = f"[{done}/0]"
        names_part = f" | 已提取: {current_names}" if current_names else ""
        line = f"\r{progress} {self.label} {pct_str}{names_part}"
        # 残影修复：行以 \r 开头原地覆盖，若后一次比前一次短，终端会残留前一次行尾字符。
        # 先 pad 到至少 80 字符宽（用空格覆盖旧字符），再截断超过 80 的部分。
        if len(line) < 80:
            line = line.ljust(80)
        if len(line) > 80:
            line = line[:77] + "..."
        sys.stderr.write(line)
        sys.stderr.flush()
        self._started = True

    def finish(self, summary: str) -> None:
        """完成时打印最终统计（换行后打印 summary）。"""
        if not self.show:
            return
        if self._started:
            sys.stderr.write("\n")
            sys.stderr.flush()
        sys.stderr.write(f"✓ {self.label} 完成: {summary}\n")
        sys.stderr.flush()
