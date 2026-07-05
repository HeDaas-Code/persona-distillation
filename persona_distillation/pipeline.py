"""确定性人格蒸馏流水线。

:class:`PersonaDistiller` 把多文本长语料按"分馏 → 冷凝 → 提纯 → 成品"四阶段
顺序调度 DeepAgents 子智能体，强制 ``response_format`` 结构化产出，可复现、可审计。
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from persona_distillation.agents import (
    PersonaSkillList,
    PresetDialogueList,
    build_dialogue_writer_agent,
    build_extractor_agent,
    build_skill_designer_agent,
    build_synthesizer_agent,
    invoke_structured,
)
from persona_distillation.chunker import chunk_text
from persona_distillation.config import DistillationConfig
from persona_distillation.loader import load_corpus
from persona_distillation.schemas import (
    Distillate,
    DistillationResult,
    PersonaCard,
    PersonaSignal,
)
from persona_distillation.triple_verification import filter_verified


class PersonaDistiller:
    """人格蒸馏器。

    Parameters:
        cfg: 蒸馏配置。``cfg.model`` 决定使用的 LLM。
    """

    def __init__(self, cfg: DistillationConfig | None = None) -> None:
        self.cfg = cfg or DistillationConfig()
        self.workdir = self.cfg.resolve_workdir()
        # 延迟构建 agent，避免无 API key 时 import 即报错
        self._extractor = None
        self._synthesizer = None
        self._skill_designer = None
        self._dialogue_writer = None

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------
    def distill(
        self,
        input_path: str | Path,
        *,
        persona_id: str | None = None,
        output_dir: str | Path | None = None,
    ) -> DistillationResult:
        """对 ``input_path``（文件或目录）执行完整蒸馏。

        Parameters:
            input_path: 语料文件或目录。
            persona_id: 覆盖配置里的人格 ID。
            output_dir: 若给定，蒸馏完成后落盘到该目录。
        """
        if persona_id:
            self.cfg.persona_id = persona_id

        t0 = time.time()
        docs = load_corpus(input_path)
        logger.info("加载 %d 篇文档", len(docs))

        # ---- Stage 1: 分馏 ----
        distillates = self._fractional_distillation(docs)
        logger.info("分馏完成，共 %d 个分块的蒸馏液", len(distillates))

        # ---- Stage 2 & 3: 冷凝 + 提纯 ----
        persona_card = self._condense_and_purify(distillates)
        # 三重验证：过滤未通过的心智模型（宁缺毋滥）
        persona_card = self._verify_mental_models(persona_card, distillates)
        logger.info("提纯完成，persona_id=%s", persona_card.persona_id)

        # ---- Stage 4a: Skills ----
        skills = self._design_skills(persona_card)
        logger.info("设计 %d 个 Skills", len(skills))

        # ---- Stage 4b: 预设对话 ----
        dialogues = self._author_dialogues(persona_card)
        logger.info("撰写 %d 组预设对话", len(dialogues))

        result = DistillationResult(
            persona_card=persona_card,
            skills=skills,
            preset_dialogues=dialogues,
            distillates=distillates,
            metadata={
                "model": self.cfg.model,
                "n_docs": len(docs),
                "n_chunks": len(distillates),
                "elapsed_sec": round(time.time() - t0, 1),
                "workdir": str(self.workdir),
            },
        )

        if output_dir is not None:
            result.save(output_dir)
            logger.info("已落盘到 %s", Path(output_dir))
        return result

    # ------------------------------------------------------------------
    # Stage 1: 分馏——逐块抽取
    # ------------------------------------------------------------------
    def _fractional_distillation(self, docs) -> list[Distillate]:
        agent = self._get_extractor()
        distillates: list[Distillate] = []
        for doc in docs:
            chunks = chunk_text(
                doc.text,
                target_tokens=self.cfg.chunk_size,
                overlap_tokens=self.cfg.chunk_overlap,
                max_chunks=self.cfg.max_chunks_per_file,
            )
            if not chunks:
                continue
            for ch in chunks:
                prompt = self._extractor_prompt(doc, ch, len(chunks))
                try:
                    dist = invoke_structured(agent, prompt, Distillate)
                except Exception as e:
                    logger.warning(
                        "extractor %s#%d 失败: %s", doc.relpath, ch.index, e
                    )
                    continue
                # 校正定位字段，确保与原文一致
                dist = dist.model_copy(
                    update={
                        "source_file": doc.relpath,
                        "chunk_index": ch.index,
                        "char_start": ch.char_start,
                        "char_end": ch.char_end,
                    }
                )
                distillates.append(dist)
                self._persist_distillate(dist)
        return distillates

    @staticmethod
    def _extractor_prompt(doc, chunk, total: int) -> str:
        return (
            f"【来源】{doc.relpath}（第 {chunk.index + 1}/{total} 块，"
            f"字符 {chunk.char_start}-{chunk.char_end}，约 {chunk.token_count} tokens）\n\n"
            f"【正文】\n{chunk.text}\n\n"
            "请对以上文本进行人格分馏，按 SignalCategory 塔板分离信号，"
            "每条附 evidence 与 salience，并给出 ≤120 字的 summary。"
        )

    def _persist_distillate(self, dist: Distillate) -> None:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in dist.source_file)
        path = self.workdir / "distillates" / f"{safe}_{dist.chunk_index:04d}.json"
        path.write_text(dist.model_dump_json(indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Stage 2 & 3: 冷凝 + 提纯
    # ------------------------------------------------------------------
    def _condense_and_purify(self, distillates: list[Distillate]) -> PersonaCard:
        agent = self._get_synthesizer()
        bundle = self._bundle_distillates(distillates)
        prompt = (
            "以下是来自多文件多分块的全部 Distillate（分馏液）。"
            "请执行冷凝：按 category 分组、合并重复、跨文件互证上调 salience、"
            "矛盾项择优；再执行提纯：丢弃 salience < "
            f"{self.cfg.salience_threshold} 的信号，每个 category 保留 3~6 条，"
            "最后产出 PersonaCard（system_prompt 必须含 [身份]/[性格]/[说话风格]/"
            "[知识边界]/[情绪模式]/[雷区]/[输出约束] 七段，并附 1~2 段开场白示范）。\n\n"
            f"【分馏液汇总】\n{bundle}"
        )
        card = invoke_structured(agent, prompt, PersonaCard)
        if not card.error_reply:
            card = card.model_copy(update={"error_reply": self.cfg.default_error_reply})
        # 若用户强指定 persona_id，覆盖之
        if self.cfg.persona_id:
            card = card.model_copy(update={"persona_id": self.cfg.persona_id})
        # ---- P1: DNA 结构化字段回填 ----
        # 某些 LLM 端点（典型如 MiniMax-M3）会把 DNA 五层
        # 拼到 system_prompt 文本里，导致顶层字段全空。
        # 这里从 system_prompt 反向解析并填充（已有内容时 no-op）。
        try:
            from persona_distillation.intake.dna_extractor import (
                backfill_dna_from_system_prompt,
            )
            card = backfill_dna_from_system_prompt(card)
        except Exception:
            logger.warning(
                "pipeline._condense_and_purify: backfill_dna_from_system_prompt 失败, exc_info=True",
            )
        return card

    @staticmethod
    def _bundle_distillates(distillates: list[Distillate]) -> str:
        """把蒸馏液压成 LLM 可读的文本（按文件分组）。"""
        by_file: dict[str, list[Distillate]] = {}
        for d in distillates:
            by_file.setdefault(d.source_file, []).append(d)
        lines: list[str] = []
        for f, items in by_file.items():
            lines.append(f"### 文件 {f}（{len(items)} 块）")
            for d in sorted(items, key=lambda x: x.chunk_index):
                lines.append(
                    f"- 块#{d.chunk_index} [{d.char_start}-{d.char_end}] 速写: {d.summary}"
                )
                for s in d.signals:
                    lines.append(
                        f"    * [{s.category.value}] salience={s.salience:.2f}  "
                        f"{s.content}  «{s.evidence}»"
                    )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 三重验证（Triple Verification）
    # ------------------------------------------------------------------
    def _verify_mental_models(
        self, card: PersonaCard, distillates: list[Distillate]
    ) -> PersonaCard:
        """对 PersonaCard 的候选心智模型执行三重验证，丢弃未通过的。"""
        if not card.mental_models:
            return card
        passed, reports = filter_verified(card.mental_models, distillates)
        dropped = len(card.mental_models) - len(passed)
        if dropped:
            logger.info(
                "三重验证：%d 个候选 → 通过 %d 个，丢弃 %d 个",
                len(card.mental_models),
                len(passed),
                dropped,
            )
        # 把验证证据写回（filter_verified 会更新 verification 字段）
        return card.model_copy(update={"mental_models": passed})

    # ------------------------------------------------------------------
    # Stage 4a: Skills
    # ------------------------------------------------------------------
    def _design_skills(self, card: PersonaCard) -> list:
        agent = self._get_skill_designer()
        prompt = (
            "基于以下人格卡（含 DNA 五层），设计 DNA 级别 PersonaSkill 列表。"
            "每个 skill 的 name 必须以 persona_id 作前缀；"
            "mental_models 只能复用人格卡里已通过三重验证的模型，不得新造。\n\n"
            f"【人格卡】\n{card.model_dump_json(indent=2)}"
        )
        result = invoke_structured(agent, prompt, PersonaSkillList)
        skills = result.skills
        # 二次过滤：skill 设计师可能误造未验证模型，这里再保险一次
        verified_names = {m.name for m in card.mental_models}
        cleaned: list = []
        for sk in skills:
            if verified_names:
                sk = sk.model_copy(
                    update={
                        "mental_models": [
                            m for m in sk.mental_models if m.name in verified_names
                        ]
                    }
                )
            cleaned.append(sk)
        return cleaned

    # ------------------------------------------------------------------
    # Stage 4b: 预设对话
    # ------------------------------------------------------------------
    def _author_dialogues(self, card: PersonaCard) -> list:
        agent = self._get_dialogue_writer()
        prompt = (
            "基于以下人格卡撰写预设对话对。"
            "user 一侧覆盖寒暄/探问背景/踩雷/求助/倾诉/告别等意图；"
            "assistant 严格体现说话风格、口头禅与雷区反应。\n\n"
            f"【人格卡】\n{card.model_dump_json(indent=2)}"
        )
        result = invoke_structured(agent, prompt, PresetDialogueList)
        return result.dialogues

    # ------------------------------------------------------------------
    # 懒加载 agent
    # ------------------------------------------------------------------
    def _get_extractor(self):
        if self._extractor is None:
            self._extractor = build_extractor_agent(self.cfg)
        return self._extractor

    def _get_synthesizer(self):
        if self._synthesizer is None:
            self._synthesizer = build_synthesizer_agent(self.cfg)
        return self._synthesizer

    def _get_skill_designer(self):
        if self._skill_designer is None:
            self._skill_designer = build_skill_designer_agent(self.cfg)
        return self._skill_designer

    def _get_dialogue_writer(self):
        if self._dialogue_writer is None:
            self._dialogue_writer = build_dialogue_writer_agent(self.cfg)
        return self._dialogue_writer
