"""离线烟雾测试——不调用 LLM，验证框架核心组件可正常工作。

跑通即说明：schemas/loader/chunker/skills_writer/triple_verification/renderer
全部 import 成功、数据结构合法、DNA 级别 SKILL.md 能正确生成。

集成测试套件——有顺序依赖，不要拆成独立 pytest 函数：
- Test 2 构造的 ``m`` / ``bad`` (MentalModel) 被 Test 4 / 5 / 6 / 11 复用
- Test 5 构造的 ``sk`` (PersonaSkill) 被 Test 6 复用
- Test 1 失败时直接 return（后续 import 缺失，跑下去无意义）

跑法：``python -m tests.smoke_test``（不要用 pytest 收集，本文件无 ``test_*``
函数，pytest 只会静默跳过——所有断言都在 ``main()`` 内顺序执行）。
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


def main() -> int:
    failures: list[str] = []

    # 1. import 全部公共 API
    try:
        from persona_distillation import (  # noqa: F401
            AntiPattern,
            Chunk,
            CharacterProfile,
            DecisionHeuristic,
            Distillate,
            DistillationConfig,
            DistillationResult,
            ExpressionDNA,
            HonestBoundary,
            IndexCategory,
            IndexStore,
            LoadedDoc,
            MentalModel,
            NameExtractionResult,
            NameIndexEntry,
            NameMention,
            PersonaCard,
            PersonaDistiller,
            PersonaSignal,
            PersonaSkill,
            PresetDialogue,
            SignalCategory,
            VerificationResult,
            build_intake_orchestrator,
            build_profile,
            chunk_text,
            distill_character,
            extract_names_from_chunk,
            load_corpus,
            rebuild_corpus_dir,
            verify_mental_models,
        )
        print("[1/10] imports OK")
    except Exception as e:
        failures.append(f"imports: {e}")
        print(f"[1/10] FAIL imports: {e}")
        return 1  # 后续测试无意义

    # 2. schemas: DNA 五层 + Pydantic 校验
    try:
        m = MentalModel(
            name="聚焦即说不",
            principle="专注是说不对100个好主意",
            verification=VerificationResult(
                cross_domain=True,
                cross_domain_evidence=["[产品] ", "[招聘]"],
                generative=True,
                generative_example="问如何扩张→先问能砍掉什么",
                exclusive=True,
                exclusivity_note="多数人靠加法扩张",
            ),
        )
        assert m.verification.passed is True
        bad = MentalModel(
            name="通用常识",
            principle="要努力",
            verification=VerificationResult(),
        )
        assert bad.verification.passed is False
        print("[2/10] schemas + DNA 五层 + verification.passed OK")
    except Exception as e:
        failures.append(f"schemas: {e}")
        print(f"[2/10] FAIL schemas: {e}")

    # 3. loader + chunker（用 examples/sample_corpus）
    try:
        corpus = Path(__file__).resolve().parent.parent / "examples" / "sample_corpus"
        docs = load_corpus(corpus)
        assert len(docs) >= 1, "应至少加载一篇文档"
        chunks = chunk_text(docs[0].text, target_tokens=512, overlap_tokens=64)
        assert len(chunks) >= 1, "应至少切出一块"
        assert chunks[0].index == 0
        print(f"[3/10] loader({len(docs)} docs) + chunker({len(chunks)} chunks) OK")
    except Exception as e:
        failures.append(f"loader/chunker: {e}")
        print(f"[3/10] FAIL loader/chunker: {e}")

    # 4. triple_verification: 过滤未通过模型
    try:
        from persona_distillation.triple_verification import filter_verified

        candidates = [m, bad]
        passed, reports = filter_verified(candidates, [])
        assert len(passed) == 1, f"应通过1个，实际{len(passed)}"
        assert passed[0].name == "聚焦即说不"
        assert len(reports) == 2
        print(f"[4/10] triple_verification: 2候选→通过{len(passed)} OK")
    except Exception as e:
        failures.append(f"triple_verification: {e}")
        print(f"[4/10] FAIL triple_verification: {e}")

    # 5. skills_writer: DNA 级别 SKILL.md 落盘
    try:
        from persona_distillation.skills_writer import write_skills

        sk = PersonaSkill(
            name="arakawa-perspective",
            description="用荒川老师视角分析",
            when_to_use="用户要求以荒川老师视角分析时",
            expression_dna=ExpressionDNA(
                vocabulary=["嘛", "唉"],
                rhythm="短句带叹词",
                signature_metaphors=["把书读薄"],
                opening_samples=["嘛，又来问问题了？"],
            ),
            mental_models=[m],
            decision_heuristics=[
                DecisionHeuristic(rule="先反问", trigger="学生提问", example="嘛？")
            ],
            anti_patterns=[AntiPattern(pattern="直接给答案", reason="要让学生自己想")],
            honest_boundaries=[
                HonestBoundary(limitation="只懂语文", reason="跨学科有限")
            ],
            instructions="1. 用心智模型框定问题\n2. 用表达DNA回应",
        )
        with tempfile.TemporaryDirectory() as td:
            paths = write_skills("arakawa_sensei", [sk], td)
            assert len(paths) == 1
            md = paths[0].read_text(encoding="utf-8")
            assert "角色扮演规则" in md
            assert "心智模型" in md
            assert "聚焦即说不" in md
            assert "跨域复现证据" in md
            assert "反模式" in md
            assert "诚实边界" in md
            assert "无法蒸馏直觉" in md  # 兜底边界
        print("[5/10] skills_writer: DNA SKILL.md 落盘 OK")
    except Exception as e:
        failures.append(f"skills_writer: {e}")
        print(f"[5/10] FAIL skills_writer: {e}")

    # 6. renderer + DistillationResult.save
    try:
        from persona_distillation.renderer import render_persona_card

        card = PersonaCard(
            persona_id="arakawa_sensei",
            display_name="荒川老师",
            system_prompt="[身份] 语文老师\n[性格] 严厉但关心学生",
            error_reply="嘛，网络出问题了，待会儿再问。",
            tags=["严厉", "关心"],
            mental_models=[m],
        )
        result = DistillationResult(
            persona_card=card,
            skills=[sk],
            preset_dialogues=[
                PresetDialogue(user="老师好", assistant="嘛，来了。", intent="寒暄")
            ],
            distillates=[
                Distillate(
                    source_file="a.txt",
                    chunk_index=0,
                    char_start=0,
                    char_end=10,
                    signals=[
                        PersonaSignal(
                            category=SignalCategory.CATCHPHRASE,
                            content="嘛",
                            evidence="嘛，又来",
                            salience=0.9,
                        )
                    ],
                    summary="严厉的开场",
                )
            ],
        )
        with tempfile.TemporaryDirectory() as td:
            result.save(td)
            out = Path(td)
            assert (out / "persona_card.json").exists()
            assert (out / "persona_card.md").exists()
            assert (out / "preset_dialogues.json").exists()
            assert (out / "distillates.jsonl").exists()
            assert (out / "distillation_result.json").exists()
            skill_md = list((out / "skills").rglob("SKILL.md"))
            assert len(skill_md) == 1
            # round-trip
            loaded = DistillationResult.load(out / "distillation_result.json")
            assert loaded.persona_card.persona_id == "arakawa_sensei"
            assert len(loaded.persona_card.mental_models) == 1
        print("[6/10] renderer + DistillationResult.save/load OK")
    except Exception as e:
        failures.append(f"renderer/save: {e}")
        print(f"[6/10] FAIL renderer/save: {e}")

    # 7. intake schemas + name_extractor（启发式 fallback）
    try:
        from persona_distillation.chunker import Chunk
        from persona_distillation.intake.schemas import (
            IndexCategory,
            NameExtractionResult,
            NameIndexEntry,
            NameMention,
        )
        # 启发式提取（无 LLM 场景）
        chunk = Chunk(text="荒川老师说：嘛，再看看吧。小明点点头。", index=0,
                      char_start=0, char_end=20, token_count=20)
        mentions = extract_names_from_chunk(chunk, source="test.txt", llm=None)
        assert isinstance(mentions, list)
        assert all(isinstance(m, NameMention) for m in mentions)
        # 至少能抓到 荒川 / 小明 中至少一个
        names = {m.name for m in mentions}
        assert names & {"荒川", "荒川老师", "小明"}, f"应识别到人名，实际: {names}"
        # schema 校验
        ent = NameIndexEntry(
            character_name="荒川",
            category=IndexCategory.SPEECH,
            text="嘛，再看看吧。",
            source="test.txt",
        )
        assert len(ent.uuid) == 36  # uuid4
        ext = NameExtractionResult(mentions=mentions)
        assert len(ext.mentions) == len(mentions)
        print(f"[7/10] name_extractor: 启发式抓到 {len(mentions)} 条 mentions OK")
    except Exception as e:
        failures.append(f"name_extractor: {e}")
        print(f"[7/10] FAIL name_extractor: {e}")

    # 8. IndexStore：SQLite + Chroma 写 / 查
    try:
        from persona_distillation.intake.embedder import HashEmbeddings
        from persona_distillation.intake.index_store import IndexStore
        from persona_distillation.intake.schemas import (
            IndexCategory,
            NameIndexEntry,
        )

        with tempfile.TemporaryDirectory() as td:
            embed = HashEmbeddings(dim=64)
            store = IndexStore(td, embedding=embed)
            e1 = NameIndexEntry(
                character_name="荒川",
                category=IndexCategory.SPEECH,
                text="嘛，再看看吧。",
                source="a.txt",
            )
            e2 = NameIndexEntry(
                character_name="荒川",
                category=IndexCategory.APPEARANCE,
                text="五十二岁，穿藏青色开衫。",
                source="a.txt",
            )
            e3 = NameIndexEntry(
                character_name="小明",
                category=IndexCategory.EVENT,
                text="来旧书店买书。",
                source="a.txt",
            )
            store.add_many([e1, e2, e3])
            assert store.count() == 3
            arakawa = store.get_character_entries("荒川")
            assert len(arakawa) == 2
            speech_only = store.get_character_entries("荒川", IndexCategory.SPEECH)
            assert len(speech_only) == 1 and speech_only[0].text == "嘛，再看看吧。"
            chars = store.list_characters()
            assert {c["character_name"] for c in chars} == {"荒川", "小明"}
            assert chars[0]["mention_count"] == 2  # 荒川 排第一
            # 关键词检索（fallback 或 chroma）
            results = store.search("开衫")
            assert any("开衫" in r.text for r in results)
            store.close()
        print("[8/10] IndexStore: 写/查/聚合 OK")
    except Exception as e:
        failures.append(f"index_store: {e}")
        print(f"[8/10] FAIL index_store: {e}")

    # 9. profile_builder：聚合 + 兜底 summary
    try:
        from persona_distillation.intake.embedder import HashEmbeddings
        from persona_distillation.intake.index_store import IndexStore
        from persona_distillation.intake.profile_builder import build_profile
        from persona_distillation.intake.schemas import (
            IndexCategory,
            NameIndexEntry,
        )

        with tempfile.TemporaryDirectory() as td:
            store = IndexStore(td, embedding=HashEmbeddings(dim=64))
            for i, (cat, txt) in enumerate([
                (IndexCategory.SPEECH, "嘛，再看看吧。这版是岩波文库。"),
                (IndexCategory.SPEECH, "书不还价。"),
                (IndexCategory.APPEARANCE, "五十二岁，穿藏青色开衫。"),
                (IndexCategory.EVENT, "年轻时在东京念比较文学。"),
            ]):
                store.add(NameIndexEntry(
                    character_name="荒川", category=cat, text=txt, source="a.txt",
                ))
            profile = build_profile("荒川", store, llm=None, top_n=2)
            assert profile.character_name == "荒川"
            assert profile.mention_count == 4
            assert profile.speech_count == 2
            assert profile.appearance_count == 1
            assert profile.event_count == 1
            assert len(profile.speech_excerpts) == 2
            assert len(profile.appearance_excerpts) == 1
            assert len(profile.event_excerpts) == 1
            assert profile.summary  # 兜底 summary 非空
            store.close()
        print("[9/10] profile_builder: 聚合 + 兜底 summary OK")
    except Exception as e:
        failures.append(f"profile_builder: {e}")
        print(f"[9/10] FAIL profile_builder: {e}")

    # 10. bridge.rebuild_corpus_dir：3 个 md 落盘
    try:
        from persona_distillation.intake.bridge import (
            rebuild_corpus_dir,
            slugify,
        )
        from persona_distillation.intake.schemas import (
            IndexCategory,
            NameIndexEntry,
        )
        from persona_distillation.intake.profile_builder import build_profile
        from persona_distillation.intake.embedder import HashEmbeddings
        from persona_distillation.intake.index_store import IndexStore

        assert slugify("荒川善次") == "荒川善次"
        assert slugify("Mr. Smith") == "mr-smith"
        assert slugify("!!!") == "persona"  # 全无效字符走 fallback
        with tempfile.TemporaryDirectory() as td:
            store = IndexStore(td, embedding=HashEmbeddings(dim=64))
            store.add(NameIndexEntry(
                character_name="荒川", category=IndexCategory.SPEECH,
                text="嘛，再看看吧。", source="a.txt"))
            store.add(NameIndexEntry(
                character_name="荒川", category=IndexCategory.APPEARANCE,
                text="五十二岁，穿藏青色开衫。", source="a.txt"))
            profile = build_profile("荒川", store, llm=None, top_n=2)
            with tempfile.TemporaryDirectory() as wd:
                out = rebuild_corpus_dir(profile, Path(wd))
                assert out.exists()
                assert (out / "speech.md").exists()
                assert (out / "appearance.md").exists()
                assert (out / "events.md").exists()
                assert (out / "summary.md").exists()
                speech_body = (out / "speech.md").read_text(encoding="utf-8")
                assert "嘛，再看看吧" in speech_body
            store.close()
        print("[10/10] bridge.rebuild_corpus_dir: 3 个 md 落盘 OK")
    except Exception as e:
        failures.append(f"bridge: {e}")
        print(f"[10/10] FAIL bridge: {e}")

    # 11. dna_extractor backfill 集成：构造含 [心智模型] / [雷区] 的 system_prompt，
    #     调用 backfill_dna_from_system_prompt，断言 anti_patterns 必有、mental_models 可空。
    try:
        from persona_distillation.intake.dna_extractor import (
            backfill_dna_from_system_prompt,
        )

        sp = """[身份]
店主
[性格]
- 外冷内热
[心智模型]
- 时间定值：价值由时间判断
- 书脊-人脊同构：物的尊严即人的尊严
- 栖居而非消费：与物共处
[雷区]
- 讨价还价 → 拒绝
- 当面折书角 → 请出
- 拍照发朋友圈 → 不接待
"""
        card = PersonaCard(
            persona_id="arakawa_sensei",
            system_prompt=sp,
            error_reply="嘛，网络出问题了。",
        )
        assert len(card.anti_patterns) == 0
        assert len(card.mental_models) == 0
        new_card = backfill_dna_from_system_prompt(card)
        # 不可变：原 card 不变
        assert len(card.anti_patterns) == 0
        # anti_patterns 必填：3 验证不齐时 mental_models 为 0，但 anti_patterns 一定有
        assert len(new_card.anti_patterns) >= 2
        assert len(new_card.mental_models) + len(new_card.anti_patterns) >= 3
        print("[11/11] dna_extractor backfill integration OK")
    except Exception as e:
        failures.append(f"dna_backfill_integration: {e}")
        print(f"[11/11] FAIL dna_backfill_integration: {e}")

    print()
    if failures:
        print(f"=== {len(failures)} FAILURES ===")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("=== ALL 11 SMOKE TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
