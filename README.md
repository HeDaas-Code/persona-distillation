<div align="center">

# 人格 · 蒸馏
## Persona Distillation Framework

**基于 LangChain DeepAgents · MiniMax-M3**

[![Smoke Tests](https://github.com/HeDaas-Code/persona-distillation/actions/workflows/smoke.yml/badge.svg)](https://github.com/HeDaas-Code/persona-distillation/actions/workflows/smoke.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

把多文本长语料视作 _原液_，借化学蒸馏的三段隐喻——_分馏_、_冷凝_、_提纯_——
逐层分离出角色的人格信号，最终灌装成一张可注入的人格卡、
一组 **DNA 级别** 人格 Skills 与若干预设对话。

</div>

---

## 三种工作模式

> **CLI 直跑模式**（确定性、可复现）—— `distill` 子命令一键完成整套蒸馏。
>
> **主理人 Agent 模式**（交互式）—— `chat` 子命令启动 intake 子包，先做人物识别 + 索引 + 档案，让用户从候选人物里选一位再蒸馏。
>
> **WebUI 调试模式**（可视化）—— `webui` 子命令启动 Gradio 四 Tab 面板：蒸馏参数 / 产物浏览 / 主理人 Agent 对话 / OC 共创。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 API Key（以 MiniMax 为例）
export MINIMAX_API_KEY=sk-...
source .env  # 或用 .env 文件（python-dotenv 自动加载）

# —— CLI 直跑 ——
python -m persona_distillation.main distill ./examples/sample_corpus ./out \
    --model minimax:MiniMax-M3 --persona-id arakawa_sensei

# 查看产物
python -m persona_distillation.main inspect ./out

# 质量评估（离线：只跑覆盖度；在线：三维度全跑）
python -m persona_distillation.main eval ./out --offline
python -m persona_distillation.main eval ./out --model minimax:MiniMax-M3

# —— 主理人 Agent 模式 ——
python -m persona_distillation.main chat

# —— WebUI ——
python -m persona_distillation.main webui --host 0.0.0.0 --port 7860

# —— OC 共创（从设定捏造语料 → 蒸馏）——
python -m persona_distillation.main cocreate
```

## 蒸馏方法论

```
原液（多文本语料）
  │
  ▼
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  分馏     │───▶│  冷凝     │───▶│  提纯     │───▶│  成品     │
│ Fractional│    │ Condense │    │  Purify  │    │  Final   │
│          │    │          │    │          │    │          │
│ 按 Signal │    │ 跨分块/  │    │ 去冲突 + │    │ Persona  │
│ Category  │    │ 跨文件   │    │ salience │    │ Card +   │
│ 塔板分离  │    │ 聚合     │    │ 取舍     │    │ Skills + │
│           │    │          │    │          │    │ Dialogs  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
      │                                              │
      │  intake 子包（预处理）                        │  schemas 产出
      ▼                                              ▼
  IndexStore                                     落盘 JSON + SKILL.md
  (Chroma + SQLite)                              可注入 Agent 平台
```

### 四阶段详解

| 阶段 | 化学隐喻 | 代码实现 | 产出 |
|------|---------|---------|------|
| **分馏** | 按沸点分离组分 | `Extractor` SubAgent 逐 chunk 按 SignalCategory 提取 | `Distillate` 列表（含 evidence + salience） |
| **冷凝** | 气态 → 液态聚合 | `Synthesizer` SubAgent 跨 chunk 聚合；>30 条走 map-reduce 分批冷凝 | `PersonaCard`（persona_id + system_prompt + error_reply） |
| **提纯** | 去杂质 | salience < 阈值丢弃；冲突信号取舍 | `PersonaSkill` 列表（DNA 五层） |
| **成品** | 灌装 | `SkillDesigner` + `DialogueWriter` | `SKILL.md` 文件 + `PresetDialogue` 列表 |

### DNA 五层

每个 `PersonaSkill` 包含以下五层结构化人格信号：

| 层 | 类 | 含义 |
|----|----|------|
| 表达 DNA | `ExpressionDNA` | 词汇偏好 / 句式节奏 / 修辞习惯 / 标志性比喻 / 开场白示范 |
| 心智模型 | `MentalModel` | 深层信念 + **三重验证**（引文 + 逻辑推演 + 反例检测） |
| 决策启发式 | `DecisionHeuristic` | 面对选择时的行为规则 |
| 反模式 | `AntiPattern` | "不该这样做"的负面信号 |
| 诚实边界 | `HonestBoundary` | 不做的事 + 理由（定格为蒸馏时点的信念） |

## Intake 预处理子包

intake 子包是主理人 Agent 模式的核心，负责蒸馏前的语料预处理：
从原始文本到结构化人物索引，再到蒸馏语料重建。

```
原始语料 ──▶ load_and_chunk ──▶ NER (intake_ner SubAgent)
                                     │
                                     ▼
                              index_characters
                                     │
                                     ▼
                              IndexStore
                              (Chroma + SQLite)
                                     │
                   ┌─────────────────┼─────────────────┐
                   ▼                 ▼                 ▼
            resolve_characters   list_characters   get_character_entries
            (实体归并)            (列出人物)        (取索引条目)
                                     │
                                     ▼
                              profile_builder SubAgent
                                     │
                                     ▼
                              CharacterProfile
                                     │
                                     ▼
                    rebuild_corpus_dir (条件脱名)
                                     │
                                     ▼
                              PersonaDistiller
```

### 预处理四道防线

| 防线 | 机制 | 实现 |
|------|------|------|
| 注入防护 | 检测 chunk 内的 prompt injection 模式 | `name_extractor._detect_injection` |
| Evidence 校验 | NER 返回的 evidence 必须是原文精确子串 | `name_extractor._validate_evidence` |
| JSON 抢救 | LLM 返回非标准 JSON 时用栈匹配提取 | `agents._extract_first_json_object` |
| 失败可见性 | 分馏失败率 >50% 中止；metadata 含失败统计 | `pipeline.py` RuntimeError + stats |

### P3 预处理优化（Issue #16-#19）

| 优化项 | Issue | 状态 | 实现 |
|--------|-------|------|------|
| 多线程并行 NER | #16 | ✅ | `config.ner_parallel` + `progress.py` 线程安全锁 + 批量写库 |
| 嵌入/重排灵活使用 | #18 | ✅ | chunk 去重（`chunker.dedup_chunks`）+ 多 query 重排（`profile_builder._multi_query_rerank`）|
| 跨 chunk 实体归并 | #17 | ✅ | `entity_resolver.py` 三重信号融合（别名交叉 + 字符串相似 + 嵌入相似）|
| 关系提取 + 条件脱名 | #19 | ✅ | NER prompt 加 `co_mentioned` + `relation_to`；`bridge.py` 条件脱名 + `relationships.json` |

## 质量评估（eval 子包）

蒸馏完成后可对产物跑三维度质量评估：

| 维度 | 方法 | 指标 |
|------|------|------|
| **覆盖度** (Coverage) | 纯规则统计 DNA 五层完整度 | `total_score ∈ [0,1]` + 各层计数 + 验证通过率 |
| **忠实度** (Fidelity) | LLM-as-judge 对比 PersonaCard 与原语料 | `score ∈ [0,1]` + 3 条理由 |
| **可识别度** (Identifiability) | Probe + 盲猜：让人格卡回答 5 个通用问题，独立 judge 猜人物 | `correct: bool` + `confidence ∈ [0,1]` |

```bash
# 离线模式（只跑覆盖度，不调 LLM）
python -m persona_distillation.main eval ./out --offline

# 完整模式（三维度全跑）
python -m persona_distillation.main eval ./out --model minimax:MiniMax-M3
```

综合评分加权：coverage 0.3 + fidelity 0.4 + identifiability 0.3

## OC 共创

从设定文本捏造语料，再蒸馏成人格卡——"无中生有"的工作流：

```
OC 设定文本
  │
  ▼
Stage 1: 骨架生成（monologue + dialogue + event + memory 四篇文本）
  │
  ▼
Stage 2: 血肉访谈（interviewer 对角色做访谈，扩充语料）
  │
  ▼
Stage 3: 蒸馏（复用 PersonaDistiller 全流程）
  │
  ▼
PersonaCard + Skills + PresetDialogues
```

CLI: `python -m persona_distillation.main cocreate`
WebUI: 第四个 Tab "OC 共创"

## 模块结构

```
persona_distillation/
├── __init__.py              # 对外导出
├── main.py                  # CLI 入口（distill / inspect / chat / webui / cocreate / eval）
├── config.py                # DistillationConfig 配置
├── schemas.py               # Pydantic 数据契约（PersonaCard / DNA 五层 / EvalReport）
├── agents.py                # DeepAgents 工厂（SubAgent + invoke_structured）
├── pipeline.py              # 确定性蒸馏流水线 PersonaDistiller
├── prompts.py               # 蒸馏方法论系统提示词 + OC 共创 prompts
├── skills_writer.py         # PersonaSkill → SKILL.md 落盘
├── chunker.py               # Token 感知分块器 + chunk 去重
├── intake/                  # 预处理子包
│   ├── __init__.py
│   ├── tools.py             # 主理人 Agent 工具桥接（纯 IO + SubAgent 交接）
│   ├── name_extractor.py    # LLM-NER（含注入防护 + evidence 校验）
│   ├── index_store.py       # Chroma + SQLite 双写索引（含 merge_characters）
│   ├── entity_resolver.py   # 跨 chunk 实体归并（三重信号融合）
│   ├── profile_builder.py   # 人物档案构建（多 query 重排 + LLM summary）
│   ├── bridge.py            # 蒸馏桥接（条件脱名 + relationships.json）
│   ├── embedder.py          # 嵌入 + 重排序器（离线降级 HashEmbeddings）
│   ├── progress.py          # 线程安全进度指示器
│   └── schemas.py           # intake 专用 schema（NameMention / NameIndexEntry）
├── eval/                    # 质量评估子包
│   ├── __init__.py
│   ├── coverage.py          # 覆盖度（纯规则）
│   ├── fidelity.py          # 忠实度（LLM-as-judge）
│   ├── identifiability.py   # 可识别度（probe + 盲猜）
│   └── report.py            # 汇总报告
└── webui/                   # Gradio WebUI
    ├── __init__.py          # build_ui + launch
    ├── state.py             # 共享状态 + 跨 Tab 联动 + 主题样式
    ├── tab_distill.py       # Tab 1: 蒸馏参数
    ├── tab_browse.py        # Tab 2: 产物浏览
    ├── tab_agent.py         # Tab 3: 主理人 Agent 对话
    └── tab_cocreate.py      # Tab 4: OC 共创
```

## CLI 子命令

| 子命令 | 用途 | 需要 API Key |
|--------|------|-------------|
| `distill` | CLI 直跑蒸馏 | ✅ |
| `inspect` | 查看蒸馏产物 | ❌ |
| `chat` | 主理人 Agent 交互 | ✅ |
| `webui` | Gradio 调试面板 | 产物浏览 Tab 不需要 |
| `cocreate` | OC 共创蒸馏 | ✅ |
| `eval` | 质量评估（`--offline` 不需要） | `--offline` 时不需要 |

## 配置

通过 `DistillationConfig` dataclass 或环境变量配置：

```python
from persona_distillation import DistillationConfig

cfg = DistillationConfig(
    model="minimax:MiniMax-M3",      # provider:model 格式
    chunk_size=1800,                 # 蒸馏阶段分块 token 数
    intake_chunk_size=1200,          # intake NER 分块 token 数
    ner_parallel=4,                  # NER 并行度（I/O bound）
    auto_merge=True,                 # 跨 chunk 实体归并
    auto_merge_threshold=0.85,       # 字符串相似度阈值
    chunk_dedup_threshold=0.95,      # chunk 去重 cosine 阈值
    detect_injection=True,           # 提示注入防护
    enable_chunk_cache=True,         # 断点续传
    offline=False,                   # 离线模式（HashEmbeddings）
    dry_run=False,                   # 跳过 API key 校验（CI/测试）
)
```

关键环境变量：

| 变量 | 用途 |
|------|------|
| `MINIMAX_API_KEY` | MiniMax API 密钥 |
| `MINIMAX_MODEL` | 模型名（默认 `minimax:MiniMax-M3`） |
| `MINIMAX_BASE_URL` | OpenAI 兼容 endpoint |
| `EMBEDDING_MODEL` | 嵌入模型（默认 `BAAI/bge-m3`） |
| `RERANK_MODEL` | 重排序模型（默认 `BAAI/bge-reranker-base`） |
| `OFFLINE` | 离线模式（`1`/`true`） |

## 产物结构

```
out/
├── distillation_result.json    # 完整蒸馏结果（PersonaCard + Skills + Dialogues + metadata）
├── persona_card.json           # 人格卡单独导出
├── skills/
│   └── <skill_name>/
│       └── SKILL.md            # Anthropic Agent Skills 规范
└── eval_report.json            # 质量评估报告（eval 子命令产出）
```

## 测试

```bash
# 全量 pytest（50 tests）
python -m pytest tests/ -v

# 离线烟雾测试（无需 API key）
python -m tests.smoke_test
python -m tests.optimization_verify
```

| 测试文件 | 覆盖范围 |
|---------|---------|
| `test_eval_coverage.py` | 覆盖度评估器（纯规则，含空卡 / 验证失败 / 权重验证） |
| `test_eval_report.py` | 汇总报告（FakeLLM 模拟 fidelity + identifiability + JSON round-trip） |
| `test_invoke_structured.py` | JSON 抢救逻辑（栈匹配 / 自然语言花括号 / 嵌套对象） |
| `test_pipeline_failure.py` | 分馏失败可见性（高失败率中止 / 低失败率继续 / metadata 统计） |
| `test_chunk_cache.py` | 分块缓存（确定性 UUID / 断点续传 / corpus_registry） |
| `test_schema_strictness.py` | Schema 严格性（Pydantic 校验 / 必填字段） |
| `test_interview.py` | OC 共创访谈流程 |
| `test_oc_writer.py` | OC 共创文本生成 |
| `dna_extractor_test.py` | DNA 提取器 |
| `smoke_test.py` | 离线烟雾测试（schemas / loader / chunker / skills_writer） |
| `optimization_verify.py` | 优化验证器 |

## 技术栈

`DeepAgents` · `LangChain` · `MiniMax-M3` · `Pydantic` · `tiktoken` · `ChromaDB` · `sentence-transformers` · `Gradio`

## 致谢

- [nuwa-skill](https://github.com/alchaincyf/nuwa-skill) — DNA 级别认知操作系统与三重验证方法论
- [Anthropic Agent Skills](https://docs.anthropic.com/en/docs/agents/skills) — SKILL.md 规范
- [DeepAgents](https://github.com/langchain-ai/deepagents) — 子智能体编排引擎
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) · [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) — intake 子包默认嵌入 / 重排序模型

## License

MIT
