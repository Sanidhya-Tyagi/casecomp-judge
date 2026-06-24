

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import fitz  # PyMuPDF
import pytest

from casecomp_judge.agents.judge import JudgeAgent
from casecomp_judge.agents.vision_reader import build_vision_enriched_text
from casecomp_judge.extraction.pdf_extractor import extract_deck
from casecomp_judge.utils.ollama_client import OllamaClient, OllamaResponse


RUBRIC = {
    "rubric_name": "Test Rubric",
    "scale": {"min": 1, "max": 10},
    "criteria": [
        {"id": "crit_a", "name": "Criterion A", "weight": 50, "description": "..."},
        {"id": "crit_b", "name": "Criterion B", "weight": 50, "description": "..."},
    ],
    "overall_verdict": {"bands": [{"min_score": 0, "label": "Unscored"}]},
}


@pytest.fixture
def mixed_deck(tmp_path: Path):
    """A 3-slide deck: text-only, chart-heavy, text-only."""
    pdf_path = tmp_path / "mixed.pdf"
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_text((72, 72), "Slide 1: plain text")

    p2 = doc.new_page()
    p2.insert_text((72, 40), "Slide 2: chart")
    for x, h in [(100, 5), (180, 8), (260, 3), (340, 9)]:
        rect = fitz.Rect(x, 500 - h * 20, x + 50, 500)
        p2.draw_rect(rect, color=(0, 0, 1), fill=(0.3, 0.5, 0.9))

    p3 = doc.new_page()
    p3.insert_text((72, 72), "Slide 3: plain text again")

    doc.save(str(pdf_path))
    doc.close()

    output_dir = tmp_path / "output"
    return extract_deck(pdf_path, output_dir=output_dir, render_images=True)


# ----------------------------------------------------------------------
# Vision routing
# ----------------------------------------------------------------------


def test_routing_skips_text_only_slides(mixed_deck) -> None:
    assert mixed_deck.slides[0].has_visual_content is False
    assert mixed_deck.slides[1].has_visual_content is True
    assert mixed_deck.slides[2].has_visual_content is False


def test_build_vision_enriched_text_only_calls_flagged_slides(mixed_deck) -> None:
    call_log = []

    def fake_generate_with_image(self, prompt, image_path, system=None):
        call_log.append(image_path)
        return OllamaResponse(text="A bar chart with rising values.", raw={})

    with patch(
        "casecomp_judge.utils.ollama_client.OllamaClient.generate_with_image",
        fake_generate_with_image,
    ):
        client = OllamaClient(model="qwen2.5vl")
        descriptions = build_vision_enriched_text(client, mixed_deck)

    assert len(call_log) == 1
    assert "slide_002" in call_log[0]
    assert descriptions[1] == ""  # skipped, not called
    assert descriptions[3] == ""  # skipped, not called
    assert "bar chart" in descriptions[2].lower()


def test_force_all_bypasses_routing(mixed_deck) -> None:
    call_log = []

    def fake_generate_with_image(self, prompt, image_path, system=None):
        call_log.append(image_path)
        return OllamaResponse(text="Nothing notable.", raw={})

    with patch(
        "casecomp_judge.utils.ollama_client.OllamaClient.generate_with_image",
        fake_generate_with_image,
    ):
        client = OllamaClient(model="qwen2.5vl")
        build_vision_enriched_text(client, mixed_deck, force_all=True)

    assert len(call_log) == 3  # all slides called regardless of heuristic


# ----------------------------------------------------------------------
# Self-critique loop
# ----------------------------------------------------------------------


def test_verification_flags_hallucinated_citation(mixed_deck) -> None:
    client = OllamaClient(model="llama3")
    judge = JudgeAgent(client=client, rubric=RUBRIC, enable_self_critique=True, enable_viability_assessment=False)

    slides_by_index = {s.index: s for s in mixed_deck.slides}

    from casecomp_judge.agents.judge import CriterionScore

    bad_cs = CriterionScore(
        id="crit_a",
        score=5.0,
        reasoning="short",
        confidence="high",
        cited_slides=[999],  # doesn't exist
    )
    notes = judge._verify_criterion(bad_cs, slides_by_index, mixed_deck.slide_count)
    assert any("999" in n for n in notes)
    assert any("too short" in n for n in notes)  # also flagged for short reasoning


def test_verification_passes_valid_criterion(mixed_deck) -> None:
    client = OllamaClient(model="llama3")
    judge = JudgeAgent(client=client, rubric=RUBRIC, enable_self_critique=True, enable_viability_assessment=False)
    slides_by_index = {s.index: s for s in mixed_deck.slides}

    from casecomp_judge.agents.judge import CriterionScore

    good_cs = CriterionScore(
        id="crit_a",
        score=7.0,
        reasoning="This is a substantive piece of reasoning citing real evidence from the deck.",
        fault_found="The team asserts margin recovery without addressing competitor response risk.",
        confidence="high",
        cited_slides=[1],
    )
    notes = judge._verify_criterion(good_cs, slides_by_index, mixed_deck.slide_count)
    assert notes == []


def test_verification_flags_missing_fault(mixed_deck) -> None:
    """A criterion with no genuine fault identified should be flagged —
    this is the guard against the judge being too uniformly positive."""
    client = OllamaClient(model="llama3")
    judge = JudgeAgent(client=client, rubric=RUBRIC, enable_self_critique=True, enable_viability_assessment=False)
    slides_by_index = {s.index: s for s in mixed_deck.slides}

    from casecomp_judge.agents.judge import CriterionScore

    too_positive_cs = CriterionScore(
        id="crit_a",
        score=9.0,
        reasoning="This criterion is handled excellently with strong evidence throughout the deck.",
        fault_found="None",  # the lazy pattern we're guarding against
        confidence="high",
        cited_slides=[1],
    )
    notes = judge._verify_criterion(
        too_positive_cs, slides_by_index, mixed_deck.slide_count
    )
    assert any("no genuine fault" in n.lower() for n in notes)


def test_verification_accepts_real_fault(mixed_deck) -> None:
    client = OllamaClient(model="llama3")
    judge = JudgeAgent(client=client, rubric=RUBRIC, enable_self_critique=True, enable_viability_assessment=False)
    slides_by_index = {s.index: s for s in mixed_deck.slides}

    from casecomp_judge.agents.judge import CriterionScore

    cs_with_fault = CriterionScore(
        id="crit_a",
        score=8.0,
        reasoning="Strong evidence, well-grounded reasoning throughout the deck content.",
        fault_found="The team never addresses how a competitor might respond to this pricing move.",
        confidence="high",
        cited_slides=[1],
    )
    notes = judge._verify_criterion(
        cs_with_fault, slides_by_index, mixed_deck.slide_count
    )
    assert notes == []


def test_self_critique_only_revises_flagged_criteria(mixed_deck) -> None:
    rejudge_calls = []
    initial_calls = []

    def fake_generate_json(self, prompt, system=None, images=None):
        if "Criterion to re-examine" in prompt:
            rejudge_calls.append(prompt)
            return {
                "score": 9.0,
                "reasoning": "After a much closer look, the evidence is actually very strong here.",
                "fault_found": "Minor: doesn't quantify the downside scenario, but the core logic holds.",
                "confidence": "high",
                "cited_slides": [2],
            }
        initial_calls.append(prompt)
        return {
            "criterion_scores": [
                {
                    "id": "crit_a",
                    "score": 5.0,
                    "reasoning": "weak",
                    "fault_found": "",
                    "confidence": "low",
                    "cited_slides": [999],
                },
                {
                    "id": "crit_b",
                    "score": 8.0,
                    "reasoning": "Strong, well-grounded reasoning citing real deck content directly.",
                    "fault_found": "Doesn't address how competitors might respond to the pricing change.",
                    "confidence": "high",
                    "cited_slides": [1],
                },
            ],
            "strengths": [],
            "weaknesses": [],
            "overall_feedback": "ok",
        }

    with patch(
        "casecomp_judge.utils.ollama_client.OllamaClient.generate_json",
        fake_generate_json,
    ):
        client = OllamaClient(model="llama3")
        judge = JudgeAgent(
            client=client, rubric=RUBRIC, enable_self_critique=True, max_revision_rounds=1,
            enable_viability_assessment=False,
        )
        report = judge.judge(mixed_deck)

    assert len(initial_calls) == 1
    assert len(rejudge_calls) == 1  # only crit_a was flagged
    assert report.criteria_revised == ["crit_a"]

    crit_a = next(cs for cs in report.criterion_scores if cs.id == "crit_a")
    crit_b = next(cs for cs in report.criterion_scores if cs.id == "crit_b")
    assert crit_a.was_revised is True
    assert crit_a.score == 9.0
    assert crit_b.was_revised is False
    assert crit_b.score == 8.0  # untouched


def test_self_critique_disabled_skips_loop_entirely(mixed_deck) -> None:
    rejudge_calls = []

    def fake_generate_json(self, prompt, system=None, images=None):
        if "Criterion to re-examine" in prompt:
            rejudge_calls.append(prompt)
        return {
            "criterion_scores": [
                {
                    "id": "crit_a",
                    "score": 5.0,
                    "reasoning": "weak",
                    "confidence": "low",
                    "cited_slides": [999],
                },
                {
                    "id": "crit_b",
                    "score": 8.0,
                    "reasoning": "Strong reasoning grounded directly in the deck content.",
                    "confidence": "high",
                    "cited_slides": [1],
                },
            ],
            "strengths": [],
            "weaknesses": [],
            "overall_feedback": "ok",
        }

    with patch(
        "casecomp_judge.utils.ollama_client.OllamaClient.generate_json",
        fake_generate_json,
    ):
        client = OllamaClient(model="llama3")
        judge = JudgeAgent(client=client, rubric=RUBRIC, enable_self_critique=False, enable_viability_assessment=False)
        report = judge.judge(mixed_deck)

    assert len(rejudge_calls) == 0
    assert report.criteria_revised == []


def test_weighted_score_recomputed_after_revision(mixed_deck) -> None:
    def fake_generate_json(self, prompt, system=None, images=None):
        if "Criterion to re-examine" in prompt:
            return {
                "score": 10.0,  # big jump from original 1.0
                "reasoning": "On reflection, this criterion is fully and excellently addressed.",
                "fault_found": "Negligible: a minor formatting inconsistency, nothing substantive.",
                "confidence": "high",
                "cited_slides": [1],
            }
        return {
            "criterion_scores": [
                {
                    "id": "crit_a",
                    "score": 1.0,
                    "reasoning": "bad",
                    "fault_found": "",
                    "confidence": "low",
                    "cited_slides": [],
                },
                {
                    "id": "crit_b",
                    "score": 1.0,
                    "reasoning": "Also fine reasoning grounded properly in the deck content here.",
                    "fault_found": "Doesn't tie the recommendation back to the stated problem clearly.",
                    "confidence": "high",
                    "cited_slides": [1],
                },
            ],
            "strengths": [],
            "weaknesses": [],
            "overall_feedback": "ok",
        }

    with patch(
        "casecomp_judge.utils.ollama_client.OllamaClient.generate_json",
        fake_generate_json,
    ):
        client = OllamaClient(model="llama3")
        judge = JudgeAgent(client=client, rubric=RUBRIC, enable_self_critique=True, enable_viability_assessment=False)
        report = judge.judge(mixed_deck)

    # crit_a revised 1.0 -> 10.0, crit_b stays 1.0 (passed verification cleanly),
    # equal 50/50 weights -> expected weighted score = (10.0*50 + 1.0*50) / 100 = 5.5
    assert report.weighted_score == 5.5


# ----------------------------------------------------------------------
# Structured strengths/weaknesses
# ----------------------------------------------------------------------


def test_parses_structured_weaknesses(mixed_deck) -> None:
    def fake_generate_json(self, prompt, system=None, images=None):
        return {
            "criterion_scores": [
                {"id": "crit_a", "score": 6.0, "reasoning": "Decent overall.", "fault_found": "Doesn't address competitor response.", "confidence": "high", "cited_slides": [1]},
                {"id": "crit_b", "score": 6.0, "reasoning": "Decent overall.", "fault_found": "Assumes linear growth without justification.", "confidence": "high", "cited_slides": [1]},
            ],
            "strengths": [
                {"description": "Clear problem framing grounded in the case data.", "slide": 1, "category": "structural", "significance": "notable"}
            ],
            "weaknesses": [
                {"description": "Recommendation doesn't address competitor response.", "slide": 3, "category": "logical", "severity": "major", "score_impact": "Lowered recommendation_strength by ~2 points."},
                {"description": "Growth assumption is asserted without justification.", "slide": 2, "category": "logical", "severity": "moderate", "score_impact": "Lowered analysis_rigor."},
            ],
            "overall_feedback": "Solid structure but the recommendation needs a risk section.",
        }

    with patch(
        "casecomp_judge.utils.ollama_client.OllamaClient.generate_json",
        fake_generate_json,
    ):
        client = OllamaClient(model="llama3")
        judge = JudgeAgent(client=client, rubric=RUBRIC, enable_self_critique=False, enable_viability_assessment=False)
        report = judge.judge(mixed_deck)

    assert len(report.weaknesses) == 2
    assert report.weaknesses[0].severity == "major"
    assert report.weaknesses[0].slide == 3
    assert "competitor response" in report.weaknesses[0].description
    assert report.strengths[0].significance == "notable"


def test_falls_back_to_plain_string_weaknesses(mixed_deck) -> None:
    """Smaller/local models sometimes ignore nested-object schema instructions
    and return plain strings instead — this must not crash, just degrade
    gracefully to a description-only entry."""

    def fake_generate_json(self, prompt, system=None, images=None):
        return {
            "criterion_scores": [
                {"id": "crit_a", "score": 6.0, "reasoning": "ok", "fault_found": "minor gap", "confidence": "high", "cited_slides": [1]},
                {"id": "crit_b", "score": 6.0, "reasoning": "ok", "fault_found": "minor gap", "confidence": "high", "cited_slides": [1]},
            ],
            "strengths": ["Good market sizing"],
            "weaknesses": ["No risk section", "Generic recommendation"],
            "overall_feedback": "ok",
        }

    with patch(
        "casecomp_judge.utils.ollama_client.OllamaClient.generate_json",
        fake_generate_json,
    ):
        client = OllamaClient(model="llama3")
        judge = JudgeAgent(client=client, rubric=RUBRIC, enable_self_critique=False, enable_viability_assessment=False)
        report = judge.judge(mixed_deck)

    assert len(report.weaknesses) == 2
    assert report.weaknesses[0].description == "No risk section"
    assert report.weaknesses[0].severity == "moderate"  # default
    assert report.strengths[0].description == "Good market sizing"


def test_thin_weaknesses_logged_but_does_not_crash(mixed_deck, caplog) -> None:
    def fake_generate_json(self, prompt, system=None, images=None):
        return {
            "criterion_scores": [
                {"id": "crit_a", "score": 9.0, "reasoning": "great", "fault_found": "trivial", "confidence": "high", "cited_slides": [1]},
                {"id": "crit_b", "score": 9.0, "reasoning": "great", "fault_found": "trivial", "confidence": "high", "cited_slides": [1]},
            ],
            "strengths": [],
            "weaknesses": [],
            "overall_feedback": "ok",
        }

    with patch(
        "casecomp_judge.utils.ollama_client.OllamaClient.generate_json",
        fake_generate_json,
    ):
        client = OllamaClient(model="llama3")
        judge = JudgeAgent(client=client, rubric=RUBRIC, enable_self_critique=False, enable_viability_assessment=False)
        report = judge.judge(mixed_deck)  # must not raise

    assert report.weaknesses == []


# ----------------------------------------------------------------------
# Fact-check integration
# ----------------------------------------------------------------------


def test_fact_check_findings_surfaced_in_report(mixed_deck) -> None:
    from casecomp_judge.agents.fact_checker import FactCheckResult

    fact_results = [
        FactCheckResult(
            claim_text="The global widget market is worth $50B",
            slide=1,
            verdict="contradicted",
            explanation="Sources show the market is closer to $30B.",
            search_query="global widget market size",
            sources_checked=3,
        )
    ]

    captured_prompts = []

    def fake_generate_json(self, prompt, system=None, images=None):
        captured_prompts.append(prompt)
        return {
            "criterion_scores": [
                {"id": "crit_a", "score": 4.0, "reasoning": "The market size claim is contradicted by fact-check.", "fault_found": "Cited market size doesn't match real-world data.", "confidence": "high", "cited_slides": [1]},
                {"id": "crit_b", "score": 6.0, "reasoning": "ok", "fault_found": "minor", "confidence": "high", "cited_slides": [1]},
            ],
            "strengths": [],
            "weaknesses": [
                {"description": "Market size claim contradicted by fact-check.", "slide": 1, "category": "logical", "severity": "major", "score_impact": "Lowered crit_a."}
            ],
            "overall_feedback": "The factual error undermines the market sizing credibility.",
        }

    with patch(
        "casecomp_judge.utils.ollama_client.OllamaClient.generate_json",
        fake_generate_json,
    ):
        client = OllamaClient(model="llama3")
        judge = JudgeAgent(client=client, rubric=RUBRIC, enable_self_critique=False, enable_viability_assessment=False)
        report = judge.judge(mixed_deck, fact_check_results=fact_results)

    # The fact-check block must have been included in what was sent to the model
    assert any("CONTRADICTED" in p for p in captured_prompts)
    assert any("$50B" in p for p in captured_prompts)

    # And it must be surfaced in the final report for the user to see
    assert len(report.fact_check_findings) == 1
    assert report.fact_check_findings[0].verdict == "contradicted"


def test_no_fact_check_results_produces_empty_block(mixed_deck) -> None:
    captured_prompts = []

    def fake_generate_json(self, prompt, system=None, images=None):
        captured_prompts.append(prompt)
        return {
            "criterion_scores": [
                {"id": "crit_a", "score": 6.0, "reasoning": "ok", "fault_found": "minor", "confidence": "high", "cited_slides": [1]},
                {"id": "crit_b", "score": 6.0, "reasoning": "ok", "fault_found": "minor", "confidence": "high", "cited_slides": [1]},
            ],
            "strengths": [],
            "weaknesses": [],
            "overall_feedback": "ok",
        }

    with patch(
        "casecomp_judge.utils.ollama_client.OllamaClient.generate_json",
        fake_generate_json,
    ):
        client = OllamaClient(model="llama3")
        judge = JudgeAgent(client=client, rubric=RUBRIC, enable_self_critique=False, enable_viability_assessment=False)
        report = judge.judge(mixed_deck, fact_check_results=None)

    assert report.fact_check_findings == []
    assert not any("Fact-check results" in p for p in captured_prompts)


# ----------------------------------------------------------------------
# Viability assessment
# ----------------------------------------------------------------------


def test_viability_assessment_runs_and_parses(mixed_deck) -> None:
    captured = {"judge_calls": 0, "viability_calls": 0}

    def fake_generate_json(self, prompt, system=None, images=None):
        if "skeptical strategy consultant" in (system or "").lower():
            captured["viability_calls"] += 1
            return {
                "verdict": "viable_with_caveats",
                "reasoning": "The strategy is plausible but assumes competitors won't react to the pricing change within the stated window.",
                "key_risks": ["Assumes Competitor X won't match the price cut within 6 months."],
                "what_would_need_to_be_true": ["Customer acquisition cost stays below the stated threshold."],
                "cited_slides": [2],
            }
        captured["judge_calls"] += 1
        return {
            "criterion_scores": [
                {"id": "crit_a", "score": 6.0, "reasoning": "ok", "fault_found": "minor gap", "confidence": "high", "cited_slides": [1]},
                {"id": "crit_b", "score": 6.0, "reasoning": "ok", "fault_found": "minor gap", "confidence": "high", "cited_slides": [1]},
            ],
            "strengths": [{"description": "Clear plan", "slide": 1, "category": "structural", "significance": "notable"}],
            "weaknesses": [
                {"description": "Thin risk section", "slide": 2, "category": "logical", "severity": "moderate", "score_impact": "Lowered crit_a."},
                {"description": "No competitor response analysis", "slide": 2, "category": "logical", "severity": "major", "score_impact": "Lowered crit_b."},
            ],
            "overall_feedback": "Reasonable but needs a competitive response plan.",
        }

    with patch(
        "casecomp_judge.utils.ollama_client.OllamaClient.generate_json",
        fake_generate_json,
    ):
        client = OllamaClient(model="llama3")
        judge = JudgeAgent(
            client=client, rubric=RUBRIC, enable_self_critique=False,
            enable_viability_assessment=True,
        )
        report = judge.judge(mixed_deck)

    assert captured["judge_calls"] == 1
    assert captured["viability_calls"] == 1
    assert report.viability.verdict == "viable_with_caveats"
    assert "Competitor X" in report.viability.key_risks[0]
    assert report.viability.cited_slides == [2]


def test_viability_disabled_skips_call_entirely(mixed_deck) -> None:
    viability_calls = []

    def fake_generate_json(self, prompt, system=None, images=None):
        if "skeptical strategy consultant" in (system or "").lower():
            viability_calls.append(prompt)
            return {"verdict": "viable", "reasoning": "ok"}
        return {
            "criterion_scores": [
                {"id": "crit_a", "score": 6.0, "reasoning": "ok", "fault_found": "minor", "confidence": "high", "cited_slides": [1]},
                {"id": "crit_b", "score": 6.0, "reasoning": "ok", "fault_found": "minor", "confidence": "high", "cited_slides": [1]},
            ],
            "strengths": [], "weaknesses": [], "overall_feedback": "ok",
        }

    with patch(
        "casecomp_judge.utils.ollama_client.OllamaClient.generate_json",
        fake_generate_json,
    ):
        client = OllamaClient(model="llama3")
        judge = JudgeAgent(
            client=client, rubric=RUBRIC, enable_self_critique=False,
            enable_viability_assessment=False,
        )
        report = judge.judge(mixed_deck)

    assert len(viability_calls) == 0
    assert report.viability.verdict == "unassessed"


def test_viability_invalid_verdict_falls_back_safely(mixed_deck) -> None:
    def fake_generate_json(self, prompt, system=None, images=None):
        if "skeptical strategy consultant" in (system or "").lower():
            return {"verdict": "probably fine i guess", "reasoning": "..."}
        return {
            "criterion_scores": [
                {"id": "crit_a", "score": 6.0, "reasoning": "ok", "fault_found": "minor", "confidence": "high", "cited_slides": [1]},
                {"id": "crit_b", "score": 6.0, "reasoning": "ok", "fault_found": "minor", "confidence": "high", "cited_slides": [1]},
            ],
            "strengths": [], "weaknesses": [], "overall_feedback": "ok",
        }

    with patch(
        "casecomp_judge.utils.ollama_client.OllamaClient.generate_json",
        fake_generate_json,
    ):
        client = OllamaClient(model="llama3")
        judge = JudgeAgent(
            client=client, rubric=RUBRIC, enable_self_critique=False,
            enable_viability_assessment=True,
        )
        report = judge.judge(mixed_deck)  # must not raise

    assert report.viability.verdict == "unassessed"


def test_viability_llm_failure_does_not_raise(mixed_deck) -> None:
    from casecomp_judge.utils.ollama_client import OllamaError

    def fake_generate_json(self, prompt, system=None, images=None):
        if "skeptical strategy consultant" in (system or "").lower():
            raise OllamaError("simulated failure")
        return {
            "criterion_scores": [
                {"id": "crit_a", "score": 6.0, "reasoning": "ok", "fault_found": "minor", "confidence": "high", "cited_slides": [1]},
                {"id": "crit_b", "score": 6.0, "reasoning": "ok", "fault_found": "minor", "confidence": "high", "cited_slides": [1]},
            ],
            "strengths": [], "weaknesses": [], "overall_feedback": "ok",
        }

    with patch(
        "casecomp_judge.utils.ollama_client.OllamaClient.generate_json",
        fake_generate_json,
    ):
        client = OllamaClient(model="llama3")
        judge = JudgeAgent(
            client=client, rubric=RUBRIC, enable_self_critique=False,
            enable_viability_assessment=True,
        )
        report = judge.judge(mixed_deck)  # must not raise

    assert report.viability.verdict == "unassessed"
    assert "simulated failure" in report.viability.reasoning


def test_viability_does_not_default_to_caveats_when_model_says_unrealistic(mixed_deck) -> None:
    """Guard against the verdict-handling code silently coercing a real
    'unrealistic' verdict into something softer."""

    def fake_generate_json(self, prompt, system=None, images=None):
        if "skeptical strategy consultant" in (system or "").lower():
            return {
                "verdict": "unrealistic",
                "reasoning": "The plan assumes a distribution channel the team has no access to and never addresses this gap.",
                "key_risks": ["No existing relationship with the proposed retail partner."],
                "what_would_need_to_be_true": ["The team would need to secure a partnership not currently in place."],
                "cited_slides": [2],
            }
        return {
            "criterion_scores": [
                {"id": "crit_a", "score": 4.0, "reasoning": "ok", "fault_found": "gap", "confidence": "high", "cited_slides": [1]},
                {"id": "crit_b", "score": 4.0, "reasoning": "ok", "fault_found": "gap", "confidence": "high", "cited_slides": [1]},
            ],
            "strengths": [], "weaknesses": [
                {"description": "x", "slide": 1, "category": "structural", "severity": "major", "score_impact": "x"},
                {"description": "y", "slide": 2, "category": "structural", "severity": "major", "score_impact": "y"},
            ], "overall_feedback": "ok",
        }

    with patch(
        "casecomp_judge.utils.ollama_client.OllamaClient.generate_json",
        fake_generate_json,
    ):
        client = OllamaClient(model="llama3")
        judge = JudgeAgent(
            client=client, rubric=RUBRIC, enable_self_critique=False,
            enable_viability_assessment=True,
        )
        report = judge.judge(mixed_deck)

    assert report.viability.verdict == "unrealistic"
