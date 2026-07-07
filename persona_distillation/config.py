"""运行配置。"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# 启动时自动加载项目根目录的 .env（若 python-dotenv 可用）
def _load_dotenv_once() -> None:
    """P0-2 增强：自动加载 .env 文件，简化首次配置。

    - 优先用 python-dotenv（已随 requirements 间接安装）
    - 失败时静默回退到 os.environ（不阻断冷启动）
    - 仅在模块首次导入时执行一次
    """
    try:
        from dotenv import load_dotenv  # type: ignore

        # 找项目根目录的 .env（config.py 在 persona_distillation/ 下，向上 1 层）
        here = Path(__file__).resolve().parent.parent
        env_path = here / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
            logger.debug("已加载 .env: %s", env_path)
    except ImportError:
        # python-dotenv 未安装；不报错，使用系统 env 即可
        pass


_load_dotenv_once()


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
        dry_run: 跳过 API key 校验（仅用于 CI / 单测）。默认 False。
        max_input_mb: 单文件输入大小上限（MB）。0 表示不限制。
    """

    # 默认从 env 读，便于 dotenv 部署；用户显式传参可覆盖
    model: str = field(
        default_factory=lambda: os.environ.get("MINIMAX_MODEL", "minimax:MiniMax-M3")
    )
    minimax_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "MINIMAX_BASE_URL", "https://api.minimax.io/v1"
        )
    )
    minimax_api_key_env: str = "MINIMAX_API_KEY"
    persona_id: str = field(
        default_factory=lambda: os.environ.get("MINIMAX_PERSONA_ID", "")
    )
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
    # P0-2: 跳过 API key 校验（仅 CI / 单测用）
    dry_run: bool = False
    # P2-1: 单文件输入大小上限（MB），0 = 不限制
    max_input_mb: int = 100
    # ---- intake 子包新增字段 ----
    # 嵌入模型（langchain Embeddings），离线模式用 _HashEmbeddings
    embedding_model: str = field(
        default_factory=lambda: os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
    )
    # 重排序模型（cross-encoder）
    rerank_model: str = field(
        default_factory=lambda: os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-base")
    )
    # 索引检索 top-k / 重排序后保留 top-n
    index_top_k: int = 10
    rerank_top_n: int = 6
    # 蒸馏语料重建时每类 excerpts 的上限
    # 0 = 不限，保留全部索引条目（推荐，蒸馏需要全量语料提取 DNA）
    # >0 = 每类最多保留 N 条（极端长文本时控制内存/磁盘）
    # 注意：与 rerank_top_n 区别——rerank_top_n 控制搜索结果展示和 LLM summary block 的 top-k，
    #       profile_max_entries 控制蒸馏语料（speech.md/appearance.md/events.md）的全量保留
    profile_max_entries: int = 0
    # intake 阶段分块目标 token 数（比蒸馏阶段小，保证 NER 粒度）
    intake_chunk_size: int = 1200
    intake_chunk_overlap: int = 120
    # Issue #16: NER 并行度。NER 是 HTTP I/O bound，用 ThreadPoolExecutor 并行
    # 调 LLM 可显著加速。Phase 1 重构后 NER 主路径走 SubAgent 批量 prompt
    # （intake_ner 一次性接收全部 chunk），本字段供 Python 直跑 / 兜底并行路径使用。
    # 写库（IndexStore.add_many）始终串行，避免 SQLite locked / Chroma 损坏。
    ner_parallel: int = 4
    # 离线模式：跳过真实模型下载，用伪 embedding + 关键词检索
    offline: bool = field(
        default_factory=lambda: os.environ.get("OFFLINE", "0") in ("1", "true", "True")
    )
    # P0-4: 启用提示注入防护（推荐开启）
    detect_injection: bool = True
    # intake 阶段是否向 stderr 输出分块解析进度条
    show_progress: bool = True
    """intake 阶段是否向 stderr 输出分块解析进度条。"""
    # 是否启用 chunk 级缓存（断点续传，跳过已处理 chunk）
    enable_chunk_cache: bool = True
    """是否启用 chunk 级缓存（断点续传，跳过已处理 chunk）。"""
    # 跨 chunk 实体归并：是否在 list_characters 之前自动合并同一人物
    # （三重信号：别名交叉 + 字符串相似 + 嵌入相似；命中 ≥2 重自动合并）
    auto_merge: bool = True
    """跨 chunk 实体归并：是否自动合并同一人物（命中 ≥2 重信号时）。"""
    auto_merge_threshold: float = 0.85
    """自动合并的字符串相似度阈值（Jaro-Winkler 下限，Levenshtein 仍按 ≤2）。"""
    # Issue #18.a: chunk 去重阈值（cosine ≥ 该值视为重复，跳过）
    # HashEmbeddings 时退化为 SHA-256 精确匹配，本字段被忽略
    chunk_dedup_threshold: float = 0.95
    """chunk 去重阈值：cosine ≥ 该值视为重复；HashEmbeddings 时退化为 SHA-256 精确匹配。"""

    def __post_init__(self) -> None:
        """P0-2: 启动时校验 API key，避免跑到一半才报缺 key。"""
        if self.offline or self.dry_run:
            logger.debug(
                "offline=%s dry_run=%s，跳过 API key 校验", self.offline, self.dry_run,
            )
            return
        if self.model.startswith("minimax:"):
            key = os.environ.get(self.minimax_api_key_env, "")
            if not key:
                env = self.minimax_api_key_env
                logger.error(
                    "环境变量 %s 未设置。请 export %s=<your-key> "
                    "（或启用 dry_run=True 用于测试）",
                    env,
                    env,
                )
                raise ValueError(
                    f"环境变量 {env} 未设置（model={self.model} 走 minimax provider）。"
                    f"请在 https://api.minimax.io 获取后执行："
                    f"export {env}=<your-key>。"
                    f"测试场景可设置 DistillationConfig(dry_run=True) 跳过校验。"
                )

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
