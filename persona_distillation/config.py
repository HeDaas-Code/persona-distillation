"""运行配置。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DistillationConfig:
    """人格蒸馏流水线配置。

    Attributes:
        model: 模型字符串，格式 ``provider:model``。默认 ``minimax:MiniMax-M3``，
            通过 OpenAI 兼容协议接入（``base_url=https://api.minimax.io/v1``），
            需设置环境变量 ``MINIMAX_API_KEY``。也支持 ``openai:gpt-4o-mini`` 等
            deepagents 原生 provider。
        minimax_base_url: minimax provider 的 OpenAI 兼容 endpoint。
        minimax_api_key_env: 读取 API key 的环境变量名。
        persona_id: 目标人格 ID（角色卡左侧"人格ID"字段）。留空则由合成器推断。
        chunk_size: 单个分块的目标 token 数。
        chunk_overlap: 相邻分块重叠 token 数，避免切断语境。
        max_chunks_per_file: 单文件最多切多少块，0 表示不限制（用于成本控制）。
        salience_threshold: 显著度低于该值的信号在提纯阶段会被丢弃。
        max_skills: 产出的人格 Skills 数量上限。
        max_preset_dialogues: 预设对话对数量上限。
        default_error_reply: 当合成器未给出报错回复时的兜底文案。
        workdir: 中间产物（蒸馏液 JSONL）落盘目录。留空则使用临时目录。
        extra_skills_dirs: 让主智能体加载的额外 skill 目录（框架自身的"蒸馏 skills"）。
    """

    model: str = "minimax:MiniMax-M3"
    minimax_base_url: str = "https://api.minimax.io/v1"
    minimax_api_key_env: str = "MINIMAX_API_KEY"
    persona_id: str = ""
    chunk_size: int = 1800
    chunk_overlap: int = 200
    max_chunks_per_file: int = 0
    salience_threshold: float = 0.35
    max_skills: int = 6
    max_preset_dialogues: int = 8
    default_error_reply: str = "（人格暂时失语，请稍后再试。）"
    workdir: str = ""
    extra_skills_dirs: list[str] = field(default_factory=list)
    debug: bool = False

    def resolve_workdir(self) -> Path:
        if self.workdir:
            p = Path(self.workdir)
        else:
            import tempfile

            p = Path(tempfile.mkdtemp(prefix="persona_distill_"))
        p.mkdir(parents=True, exist_ok=True)
        (p / "distillates").mkdir(exist_ok=True)
        (p / "skills").mkdir(exist_ok=True)
        return p
