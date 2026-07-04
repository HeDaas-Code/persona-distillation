<div align="center">

# 人格 · 蒸馏
## Persona Distillation Framework

**基于 LangChain DeepAgents · MiniMax-M3**

把多文本长语料视作 _原液_，借化学蒸馏的三段隐喻——_分馏_、_冷凝_、_提纯_——
逐层分离出角色的人格信号，最终灌装成一张可注入的人格卡、
一组 **DNA 级别** 人格 Skills 与若干预设对话。

</div>

---

> **灵感**：Skills 生成逻辑参考 [nuwa-skill](https://github.com/alchaincyf/nuwa-skill) 的认知操作系统方法论——
> 捕捉的是 _HOW they think_，不是 _WHAT they said_。
> 每个人格 Skill 是一套可运行的认知操作系统，而非语录合集。

## § 01 · 方法论

借蒸馏四阶，把原文里散落的"人"析出来。

| 阶段 | 隐喻 | 执行者 | 产出 |
|:---:|:---|:---|:---|
| **STAGE 01** | 🔥 分馏 `Fractional Distillation` | `extractor` 子智能体 | 每块文本按 10 类塔板分离信号，附原文证据与显著度 |
| **STAGE 02** | ❄ 冷凝 `Condensation` | `synthesizer` 子智能体 | 同类合并、跨文件互证（salience 上调）、矛盾择优 |
| **STAGE 03** | ❄ 提纯 `Purification` | `synthesizer` 子智能体 | 丢弃低显著信号，每类保留 3~6 条，**同步提炼 DNA 五层** |
| **STAGE 04** | 📜 成品 `Final Product` | `skill_designer` + `dialogue_writer` | 把 DNA 五层灌装为可运行 PersonaSkill + 预设对话 |

### DNA 五层认知操作系统（参考 nuwa-skill）

提纯阶段不再只产出 `system_prompt`，而是同步提炼人物的**认知操作系统**：

| 层 | 内容 | 体现 |
|:---:|:---|:---|
| 🗣️ **表达 DNA** | `ExpressionDNA` | 语气、节奏、用词偏好、标志性比喻、开场白示范 |
| 🧠 **心智模型** | `MentalModel[]` | 3~7 个看世界的"镜片"，**须经三重验证才收录** |
| ⚖️ **决策启发式** | `DecisionHeuristic[]` | 推理捷径与判断规则 |
| 🚫 **反模式** | `AntiPattern[]` | 绝对不会做什么——比正向规则更能勾勒人格边界 |
| 📏 **诚实边界** | `HonestBoundary[]` | skill 真正做不到什么（一个不说明自身局限的 skill 不值得信任） |

### 三重验证法（Triple Verification）

候选心智模型必须**同时**通过三项才会被收录，宁缺毋滥，避免把通用常识误当人格特质：

| 验证 | 标准 | 示例 |
|:---:|:---|:---|
| 🔄 **跨域复现** | 该模型在此人讨论的 **≥2 个不同领域**出现（一次性表态不算） | 纳瓦尔的"杠杆"——在财富/个人成长/职业选择三域复现 |
| 🧠 **有生成力** | 能推断此人对**新问题**的立场，而非只描述既有观点 | 芒格的"逆向思维"——面对"如何成功"→他先想"如何确保失败" |
| 🎯 **有排他性** | 不是所有聪明人都会这样想，体现独特视角 | "反脆弱"是塔勒布的，不是通用智慧 |

未通过的一律丢弃。框架用 `triple_verification.py` 对 LLM 初判做规则式复核：
跨域用 distillates 证据收集、生成力与排他性要求 LLM 必须给出示例文本。

---

## § 02 · 管线全景

从一摞语料，到一张可注入的人格卡。

```
┌─────────────────────────────────────────────────────────────────────────┐
│  INPUT LAYER · 摄入层                          ↻ MiniMax-M3              │
│                                                                         │
│  语料/Corpus ──→ load_corpus() ──→ chunk_text() ──→ Chunks[]            │
│  txt·md·json·csv   loader.py       chunker.py        tiktoken感知分块    │
│                                            10 塔板:                    │
│                                            speech·catch·values·know    │
│                                            emo·rel·event·bg·taboo·man  │
└─────────────────────────────────────────────────────────────────────────┘
                                  ⇣
┌─────────────────────────────────────────────────────────────────────────┐
│  ① STAGE · 分馏 FRACTIONAL DISTILLATION              🔥 heat applied    │
│                                                                         │
│              ┌──────────────┐    ●●● 10塔板    ┌──────────────┐         │
│              │  extractor   │ ──→ 滴滴 ──→     │  Distillate[]│         │
│              │  分馏器子智能体│                  │  蒸馏液       │         │
│              └──────────────┘                  └──────────────┘         │
│              agents.py                         signals·evidence·salience │
│              response_format=Distillate        落盘 distillates/*.json   │
└─────────────────────────────────────────────────────────────────────────┘
                                  ⇣
┌─────────────────────────────────────────────────────────────────────────┐
│  ②③ STAGE · 冷凝 CONDENSATION ⟶ 提纯 PURIFICATION   ❄ salience ≥ 0.35  │
│                                                                         │
│   ┌──────────────────┐    ⟶    ┌──────────────────────────────────┐    │
│   │   synthesizer    │         │        PersonaCard               │    │
│   │   冷凝·提纯器     │         │   persona_id · system_prompt     │    │
│   │                  │         │   error_reply · tags             │    │
│   │  冷凝:同类合并    │         │   ┌────────────────────────────┐ │    │
│   │      跨文件互证   │         │   │  + DNA 五层认知操作系统     │ │    │
│   │  提纯:丢低显著    │         │   │  ExpressionDNA              │ │    │
│   │      取3~6条      │         │   │  MentalModel[] ←三重验证    │ │    │
│   └──────────────────┘         │   │  DecisionHeuristic[]        │ │    │
│                                │   │  AntiPattern[]              │ │    │
│                                │   │  HonestBoundary[]           │ │    │
│                                │   └────────────────────────────┘ │    │
│                                └──────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                  ⇣  三重验证过滤未通过模型
┌─────────────────────────────────────────────────────────────────────────┐
│  ④ STAGE · 成品 FINAL PRODUCT — 灌装 DNA 级别 Skills 与预设对话          │
│                                                                         │
│   ┌─────────────────────────┐    ┌─────────────────────────┐           │
│   │    skill_designer       │    │    dialogue_writer      │           │
│   │    技能设计师(DNA级)     │    │    预设对话作者          │           │
│   │                         │    │                         │           │
│   │  response_format=       │    │  response_format=       │           │
│   │   PersonaSkillList      │    │   PresetDialogueList    │           │
│   │                         │    │                         │           │
│   │  perspective·refuse·    │    │  寒暄·探问·踩雷·        │           │
│   │  deep-dive·recall·      │    │  求助·倾诉·告别         │           │
│   │  farewell               │    │                         │           │
│   └─────────────────────────┘    └─────────────────────────┘           │
│   每个 skill 含:角色扮演规则·回答工作流·心智模型·决策启发式·反模式·诚实边界│
└─────────────────────────────────────────────────────────────────────────┘
                                  ⇣
┌─────────────────────────────────────────────────────────────────────────┐
│  OUTPUT LAYER · 落盘层 (renderer.py + skills_writer.py)                 │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │persona_card  │  │skills/*/     │  │preset_       │  │distillates │ │
│  │  .json       │  │  SKILL.md    │  │  dialogues   │  │  .jsonl    │ │
│  │              │  │              │  │  .json       │  │            │ │
│  │人格ID·系统   │  │Anthropic     │  │user/assistant│  │中间蒸馏液  │ │
│  │提示词·报错   │  │Skills规范    │  │对话对+intent │  │可审计·复现 │ │
│  │+persona_card │  │SkillsMW可加载│  │界面右侧"预设 │  │+result.json│ │
│  │  .md         │  │              │  │对话"         │  │            │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘

┌─ ENGINE · 底座（贯穿全管线）─────────────────────────────────────────────┐
│                                                                         │
│   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐     │
│   │   DeepAgents     │  │ persona-         │  │   MiniMax-M3     │     │
│   │   子智能体编排    │  │  distillation    │  │   推理模型        │     │
│   │                  │  │   蒸馏方法论Skill │  │                  │     │
│   │ create_deep_agent│  │ SKILL.md注入主AG │  │ OpenAI兼容协议    │     │
│   │ SubAgent·task    │  │ SkillsMiddleware │  │ api.minimax.io   │     │
│   │ FilesystemMW·Todo│  │ prompts.py       │  │ build_model()    │     │
│   └──────────────────┘  └──────────────────┘  └──────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## § 03 · 模块清单

每个文件各司其职，可单独替换或扩展。

| 模块 | 文件 | 职责 |
|:---|:---|:---|
| **config** | `config.py` | 运行配置中枢。模型字符串、分块参数、显著度阈值、Skills 数量上限。默认 `minimax:MiniMax-M3` |
| **schemas** | `schemas.py` | 结构化契约。PersonaCard / PersonaSkill / Distillate + **DNA 五层**（ExpressionDNA/MentalModel/DecisionHeuristic/AntiPattern/HonestBoundary）+ VerificationResult |
| **loader** | `loader.py` | 多文本聚合摄入。txt/md 直读、json/jsonl 抽取正文、csv 按行展开 |
| **chunker** | `chunker.py` | tiktoken 感知分块 + 重叠，避免从句中劈开 |
| **prompts** | `prompts.py` | 蒸馏方法论 SKILL.md + 四个子智能体系统提示词（含 DNA 提炼与三重验证要求） |
| **agents** | `agents.py` | DeepAgents 装配工厂。`build_model` 解析 `minimax:MiniMax-M3`；四个带 `response_format` 的子智能体 |
| **pipeline** | `pipeline.py` | 确定性流水线。加载→分块→分馏→冷凝提纯→**三重验证**→设计 Skills→撰写对话 |
| **triple_verification** | `triple_verification.py` | **DNA 三重验证**。跨域复现（distillates 证据复核）+ 生成力 + 排他性，未通过的心智模型一律丢弃 |
| **renderer** | `renderer.py` | 人格卡渲染。`persona_card.json`（机器可导入）+ `persona_card.md`（人类可读） |
| **skills_writer** | `skills_writer.py` | Skills 目录落盘。每个 skill 写成 nuwa 风格 SKILL.md：角色扮演规则 + 回答工作流 + 心智模型（含三重验证证据）+ 决策启发式 + 表达DNA + 反模式 + 诚实边界 |
| **main** | `main.py` | CLI 入口。`distill` / `inspect` 两个子命令 |

---

## § 04 · DNA 级别 SKILL.md 结构

每个落盘的 `skills/<persona_id>-<scope>/SKILL.md` 遵循以下结构（参考 nuwa-skill）：

```markdown
---
name: <persona_id>-perspective
description: 用 <persona> 的视角分析...
license: MIT
---

# <Persona> · 思维操作系统

## 角色扮演规则（最重要）
**此 Skill 激活后，直接以 <Persona> 的身份回应。**
- 用「我」而非第三人称转述
- 🛑 STOP（仅一次）：首次激活输出免责声明，后续绝不重复
- 🚪 EXIT TRIGGER：用户说「退出」「切回正常」时立即恢复

## When to Use
触发场景...

## 回答工作流 (Agentic Protocol)
1. 用 mental_models 重新框定用户问题
2. 用 decision_heuristics 给出判断
3. 用 expression_dna 表达
4. 触及 anti_patterns 时果断拒绝

## 心智模型 (Mental Models)
> 每个模型均通过三重验证：跨域复现 · 有生成力 · 有排他性

### 聚焦即说不
**原理**：专注是说不对 100 个好主意
**跨域复现证据**：[产品]... [招聘]...
**生成力示例**：问如何扩张 → 他先问能砍掉什么
**排他性**：多数人靠加法扩张，他靠减法

## 决策启发式 (Decision Heuristics)
- **先问物理极限** —— 触发：优化任何系统时

## 表达 DNA (Expression DNA)
- **偏好词汇**：insanely great, shit
- **节奏**：短句、极端确定
- **标志性比喻**：...

## 反模式 (Anti-Patterns) —— 绝对不会做什么
- 🚫 **妥协** —— 绝不接受次优

## 诚实边界 (Honest Boundaries)
- ⚠️ **无法蒸馏直觉** —— 框架能提取，灵感不能
- ⚠️ **仅基于公开语料的快照** —— 不等于本人真实信念
```

---

## § 05 · 字段映射

产出与角色卡暗色界面一一对应。

| 框架产出 | → | 角色卡界面 |
|:---|:---:|:---|
| `persona_card.persona_id` | ⟶ | 左侧 · 人格ID 输入框 |
| `persona_card.system_prompt` | ⟶ | 左侧 · 系统提示词 区域 |
| `persona_card.error_reply` | ⟶ | 左侧 · 自定义报错回复信息 |
| `skills/*/SKILL.md` | ⟶ | 右侧 · Skills 选择（可指定） |
| `preset_dialogues.json` | ⟶ | 右侧 · 预设对话（可添加） |

---

## 快速开始

### 安装

```bash
pip install -r requirements.txt
export MINIMAX_API_KEY=sk-...   # 见 https://platform.minimax.io
```

### 确定性流水线（推荐，可复现）

```python
from persona_distillation import PersonaDistiller, DistillationConfig

distiller = PersonaDistiller(DistillationConfig(
    model="minimax:MiniMax-M3", persona_id="arakawa_sensei"
))
result = distiller.distill("examples/sample_corpus", output_dir="./out")
```

### CLI

```bash
# 蒸馏（默认 --model minimax:MiniMax-M3）
python -m persona_distillation.main distill ./examples/sample_corpus ./out \
    --persona-id arakawa_sensei

# 查看已蒸馏结果
python -m persona_distillation.main inspect ./out
```

### 自主编排模式（交互式）

```python
from persona_distillation.agents import build_orchestrator
from persona_distillation.config import DistillationConfig

agent = build_orchestrator(DistillationConfig(model="minimax:MiniMax-M3"))
agent.invoke({"messages": [{"role": "user", "content": "蒸馏 ./corpus 到 ./out"}]})
```

---

## 产出结构

```
out/
├── persona_card.json          # 人格卡（含 DNA 五层）
├── persona_card.md            # 人类可读版
├── skills/
│   ├── <persona_id>-perspective/SKILL.md   # DNA 级别 skill
│   ├── <persona_id>-refuse/SKILL.md
│   ├── <persona_id>-deep-dive/SKILL.md
│   └── ...
├── preset_dialogues.json      # 预设对话对
├── distillates.jsonl          # 中间蒸馏液（可审计）
└── distillation_result.json   # 完整结果（不含 distillates）
```

---

## 技术栈

`DeepAgents` · `LangChain` · `MiniMax-M3` · `Pydantic` · `tiktoken`

## 致谢

- [nuwa-skill](https://github.com/alchaincyf/nuwa-skill) — DNA 级别认知操作系统与三重验证方法论
- [Anthropic Agent Skills](https://docs.anthropic.com/en/docs/agents/skills) — SKILL.md 规范
- [DeepAgents](https://github.com/langchain-ai/deepagents) — 子智能体编排引擎

## License

MIT
