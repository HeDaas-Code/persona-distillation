"""离线烟雾测试——不调用 LLM，验证框架核心组件可正常工作。

跑通即说明：schemas/loader/chunker/skills_writer/triple_verification/renderer
全部 import 成功、数据结构合法、DNA 级别 SKILL.md 能正确生成。

CI 用：python -m tests.smoke_test
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
            DecisionHeuristic,
            Distillate,
            DistillationConfig,
            DistillationResult,
            ExpressionDNA,
            HonestBoundary,
            LoadedDoc,
            MentalModel,
            PersonaCard,
            PersonaDistiller,
            PersonaSignal,
            PersonaSkill,
            PresetDialogue,
            SignalCategory,
            VerificationResult,
            chunk_text,
            load_corpus,
            verify_mental_models,
        )
        print("[1/6] imports OK")
    except Exception as e:
        failures.append(f"imports: {e}")
        print(f"[1/6] FAIL imports: {e}")
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
        print("[2/6] schemas + DNA 五层 + verification.passed OK")
    except Exception as e:
        failures.append(f"schemas: {e}")
        print(f"[2/6] FAIL schemas: {e}")

    # 3. loader + chunker（用 examples/sample_corpus）
    try:
        corpus = Path(__file__).resolve().parent.parent / "examples" / "sample_corpus"
        docs = load_corpus(corpus)
        assert len(docs) >= 1, "应至少加载一篇文档"
        chunks = chunk_text(docs[0].text, target_tokens=512, overlap_tokens=64)
        assert len(chunks) >= 1, "应至少切出一块"
        assert chunks[0].index == 0
        print(f"[3/6] loader({len(docs)} docs) + chunker({len(chunks)} chunks) OK")
    except Exception as e:
        failures.append(f"loader/chunker: {e}")
        print(f"[3/6] FAIL loader/chunker: {e}")

    # 4. triple_verification: 过滤未通过模型
    try:
        from persona_distillation.triple_verification import filter_verified

        candidates = [m, bad]
        passed, reports = filter_verified(candidates, [])
        assert len(passed) == 1, f"应通过1个，实际{len(passed)}"
        assert passed[0].name == "聚焦即说不"
        assert len(reports) == 2
        print(f"[4/6] triple_verification: 2候选→通过{len(passed)} OK")
    except Exception as e:
        failures.append(f"triple_verification: {e}")
        print(f"[4/6] FAIL triple_verification: {e}")

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
        print("[5/6] skills_writer: DNA SKILL.md 落盘 OK")
    except Exception as e:
        failures.append(f"skills_writer: {e}")
        print(f"[5/6] FAIL skills_writer: {e}")

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
        print("[6/6] renderer + DistillationResult.save/load OK")
    except Exception as e:
        failures.append(f"renderer/save: {e}")
        print(f"[6/6] FAIL renderer/save: {e}")

    print()
    if failures:
        print(f"=== {len(failures)} FAILURES ===")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("=== ALL SMOKE TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
