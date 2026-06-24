"""Judge agent.

Scores a deck (using its extracted text + the prior summary) against
an editable YAML rubric, producing per-criterion scores with reasoning,
an overall weighted score, a verdict label, and strengths/weaknesses.

Self-critique loop
-------------------
A single forward pass through an LLM is not the same as judging well —
the model can be confidently wrong, or honestly uncertain because the
deck under-specifies something. This agent makes that uncertainty
explicit and acts on it:

  1. INITIAL JUDGE PASS — score every criterion, and for each one report
     a confidence level ("high"/"medium"/"low") and which slide numbers
     the reasoning is actually grounded in.
  2. VERIFY — independently check each criterion's output: does it cite
     slides that actually exist in the deck? Is the reasoning concrete
     (mentions specifics) rather than generic boilerplate? Criteria that
     fail verification are treated as low-confidence even if the model
     claimed otherwise.
  3. TARGETED RE-JUDGE — for criteria that are low-confidence or failed
     verification, re-run a narrower prompt for *just that criterion*,
     re-supplying the specific cited slides (re-running vision on them
     if a vision client is available) with explicit instruction to look
     harder before answering. Bounded to one retry round per criterion
     so this can never loop indefinitely.

This means the judge doesn't just produce a score — it produces a score
*and a record of where it wasn't sure, and what it did about it*, which
is surfaced in the final report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from casecomp_judge.agents.summarizer import DeckSummary
from casecomp_judge.extraction.pdf_extractor import DeckExtraction
from casecomp_judge.utils.ollama_client import OllamaClient, OllamaError

logger = logging.getLogger("casecomp_judge.agents.judge")


JUDGE_SYSTEM_PROMPT = """\
You are a strict, demanding case competition judge at a top-tier \
competition where the bar for a strong score is genuinely high and most \
submissions fall short of it. Your job is not to be encouraging — it is \
to find the truth about how strong this analysis actually is. Teams that \
hear only praise never improve; your value is in surfacing what's \
actually wrong, not in making people feel good. Assume by default that \
there are real flaws to find — the absence of an obvious flaw is not the \
same as the work being excellent, it may just mean you haven't looked \
hard enough yet.

For every criterion, you must actively hunt for faults in three \
categories before you settle on a score:
  - STRUCTURAL faults: missing steps in the analysis, frameworks applied \
    but never connected back to the recommendation, slides that assert a \
    conclusion the preceding content doesn't support, gaps in the \
    storyline (situation -> complication -> resolution).
  - LOGICAL faults: unstated or implausible assumptions, conclusions that \
    don't follow from the data shown, correlation treated as causation, \
    cherry-picked evidence, internal contradictions between slides.
  - INTELLECTUAL faults: generic or templated thinking dressed up as \
    insight, analysis that restates the case prompt rather than adding to \
    it, surface-level treatment of a complex tradeoff, missing the \
    obvious counter-argument to the team's own recommendation.

A polished, confident, well-formatted deck is not the same as a strong \
one — visual quality and confident language are exactly the things that \
cause underqualified work to score too high. Do not let them substitute \
for evidence of real analytical rigor. If a slide's claim isn't actually \
backed by the data on that slide, that is a fault, regardless of how \
clean the slide looks.

If fact-check results are provided below, treat a "contradicted" or \
"outdated" verdict on a claim as a concrete, citable fault for whichever \
criterion that claim's slide is most relevant to — a deck that cites a \
real-world statistic incorrectly has a genuine analytical weakness, not \
just a typo. Do NOT penalize "unverifiable" claims — that means the \
search couldn't confirm or deny it, not that it's wrong. Never \
fact-check or penalize the team's own projections/estimates; only \
external claims they did not derive themselves are fair game here.

Calibrate your scores hard, and be stricter than your instinct: a 7-8 \
should be reserved for analysis that is both correct AND demonstrates \
real insight beyond the obvious; reserve 9-10 for work you'd genuinely \
hold up as a model answer, which should be rare. Most competent, \
correctly-executed-but-unsurprising work should land in the 4-6 range — \
treat that as the realistic center of the scale, not 7. Do not default \
to 7+ out of politeness — an unremarkable submission that makes no \
egregious mistakes is still just average, not strong, and average is not \
a compliment. If a score is high, your reasoning must justify why with \
something more specific than "the team did X well" — explain what made \
it genuinely better than a standard treatment, and what a perfect \
version would have additionally done that this submission didn't.

You score submissions strictly against the given rubric, citing specific \
evidence from the deck for every score. You give partial credit fairly \
but do not inflate scores to be encouraging.

For every criterion you score, you must also report your own confidence \
in that score ("high", "medium", or "low") and the specific slide numbers \
your reasoning is grounded in. Be honest about low confidence — if the \
deck is ambiguous, the evidence is thin, or you are inferring rather than \
reading something explicit, say so. Overclaiming confidence is worse than \
admitting uncertainty.

You always respond with valid JSON matching the exact schema given, and \
nothing else — no preamble, no markdown formatting, no explanation outside \
the JSON.
"""

JUDGE_JSON_SCHEMA_TEMPLATE = """\
{{
  "criterion_scores": [
    {{
      "id": "string, must exactly match one of the rubric criterion ids given",
      "score": "number between {min_score} and {max_score}",
      "reasoning": "string, 2-4 sentences citing specific evidence from the deck",
      "fault_found": "string, REQUIRED — the single most significant structural, logical, or intellectual flaw you found for this criterion, stated specifically with evidence. If a score of 9+ is given, this must explain a flaw so minor it didn't affect the score. Never write 'none' or leave this generic.",
      "confidence": "one of: high, medium, low",
      "cited_slides": [1, 2]
    }}
    // one entry per rubric criterion, same order as given
  ],
  "strengths": [
    {{
      "description": "string, a specific strength with concrete evidence — not generic praise",
      "slide": "integer, the slide number this strength is grounded in",
      "category": "one of: structural, logical, intellectual, communication, financial",
      "significance": "one of: minor, notable, major — how much this actually elevates the submission"
    }}
  ],
  "weaknesses": [
    {{
      "description": "string, a specific structural, logical, or intellectual fault — not a generic statement",
      "slide": "integer, the slide number this weakness appears on (0 if it spans the whole deck)",
      "category": "one of: structural, logical, intellectual, communication, financial",
      "severity": "one of: minor, moderate, major — how much this actually hurts the submission's credibility",
      "score_impact": "string, 1 sentence on which criterion/criteria this weakness pulled the score down on, and roughly how much"
    }}
  ],
  "overall_feedback": "string, 3-5 sentences of holistic feedback for the team. Must include at least one concrete thing they should have done differently, stated plainly, not softened."
}}
"""
REQUIRED_MIN_WEAKNESSES = 2
REQUIRED_MIN_STRENGTHS = 1

RE_JUDGE_SYSTEM_PROMPT = """\
You are re-examining a single scoring criterion that was flagged as \
low-confidence on a first pass — either you weren't sure, or your cited \
evidence didn't hold up. You are being given the specific slides you \
cited (now including a closer visual read of any charts/diagrams on \
them) plus a wider slice of the deck for context. Look carefully before \
answering, and remain just as demanding as the first pass — your job is \
to find real structural, logical, or intellectual faults, not to soften \
your judgment because you're taking a second look. If, after this closer \
look, you are still genuinely uncertain, it is correct and expected to \
keep your confidence as "low" — do not inflate confidence or the score \
just because you were asked to look again. You always respond with valid \
JSON matching the exact schema given, and nothing else.
"""

RE_JUDGE_JSON_SCHEMA_TEMPLATE = """\
{{
  "score": "number between {min_score} and {max_score}",
  "reasoning": "string, 2-4 sentences citing specific evidence from the deck",
  "fault_found": "string, REQUIRED — the most significant structural, logical, or intellectual flaw found, stated specifically",
  "confidence": "one of: high, medium, low",
  "cited_slides": [1, 2]
}}
"""

VALID_CONFIDENCE_LEVELS = {"high", "medium", "low"}
MIN_REASONING_WORDS_FOR_VERIFICATION = 8

VIABILITY_SYSTEM_PROMPT = """\
You are a skeptical strategy consultant assessing whether a proposed \
business strategy could actually work in the real world — not whether \
the slides are well-made, and not criterion-by-criterion, but as a whole: \
if this team tried to execute exactly what they propose, would it \
plausibly succeed?

Be skeptical by default. Most proposed strategies have at least one load- \
bearing assumption that, if wrong, breaks the plan — your job is to find \
it, not to take the team's confidence at face value. Specifically interrogate:
  - Does the strategy depend on a competitor NOT reacting, a market NOT \
    shifting, or a resource/capability the team doesn't clearly have?
  - Is the GTM plan sequenced and resourced, or does it skip straight from \
    "strategy" to "success" with no executable first step?
  - Does the impact/financial case actually follow from the team's own \
    assumptions, or does it require something better than their own data \
    supports?
  - Is there a more obvious, simpler path to the same goal that the team \
    didn't consider or didn't explain why they rejected?

Render one of three verdicts:
  - "viable": the strategy is sound, well-supported, and the team has \
    addressed the major risks credibly.
  - "viable_with_caveats": the core strategy is plausible but rests on \
    one or more specific assumptions that are uncertain or unaddressed —
    name them precisely.
  - "unrealistic": the strategy has a disqualifying flaw — an assumption \
    that is very likely false, a resource/capability gap that isn't \
    addressed, or a dependency on something outside the team's control \
    that they haven't accounted for.

Do not default to "viable_with_caveats" as a safe middle answer — only use \
it when the caveats you name are genuinely the determining factor, not as \
a way to avoid committing to a harder verdict. You always respond with \
valid JSON matching the exact schema given, and nothing else.
"""

VIABILITY_JSON_SCHEMA = """\
{
  "verdict": "one of: viable, viable_with_caveats, unrealistic",
  "reasoning": "string, 3-5 sentences explaining the verdict with specific evidence from the deck",
  "key_risks": ["string, a specific risk to execution — not generic ('market risk') but concrete ('assumes Competitor X will not match the price cut within the stated 6-month window')"],
  "what_would_need_to_be_true": ["string, a specific condition that would need to hold for this strategy to succeed, especially any the team has not validated or addressed"],
  "cited_slides": [1, 2]
}
"""

VALID_VIABILITY_VERDICTS = {"viable", "viable_with_caveats", "unrealistic"}


@dataclass
class CriterionScore:
    id: str
    name: str = ""
    weight: float = 0.0
    score: float = 0.0
    reasoning: str = ""
    fault_found: str = ""
    confidence: str = "medium"
    cited_slides: list[int] = field(default_factory=list)
    was_revised: bool = False  # True if the self-critique loop re-judged this
    verification_notes: list[str] = field(default_factory=list)


@dataclass
class StrengthEntry:
    description: str = ""
    slide: int = 0
    category: str = "structural"  # structural / logical / intellectual / communication / financial
    significance: str = "notable"  # minor / notable / major


@dataclass
class WeaknessEntry:
    description: str = ""
    slide: int = 0
    category: str = "structural"
    severity: str = "moderate"  # minor / moderate / major
    score_impact: str = ""


@dataclass
class FactCheckFinding:
    """A fact-check result surfaced in the judge report (see agents/fact_checker.py)."""

    claim_text: str = ""
    slide: int = 0
    verdict: str = "unverifiable"
    explanation: str = ""


@dataclass
class ViabilityAssessment:
    """Dedicated, holistic judgment of whether the deck's strategy is
    actually viable — distinct from the per-criterion rubric scores,
    since "is this executable in the real world" is a question that
    benefits from seeing research, competition, and GTM together
    rather than graded in isolated slices.
    """

    verdict: str = "unassessed"  # viable / viable_with_caveats / unrealistic / unassessed
    reasoning: str = ""
    key_risks: list[str] = field(default_factory=list)
    what_would_need_to_be_true: list[str] = field(default_factory=list)
    cited_slides: list[int] = field(default_factory=list)


@dataclass
class JudgeReport:
    criterion_scores: list[CriterionScore] = field(default_factory=list)
    strengths: list[StrengthEntry] = field(default_factory=list)
    weaknesses: list[WeaknessEntry] = field(default_factory=list)
    overall_feedback: str = ""
    weighted_score: float = 0.0  # on rubric's scale (e.g. out of 10)
    weighted_score_pct: float = 0.0  # normalized to 0-100
    verdict_label: str = ""
    rubric_name: str = ""
    scale_min: float = 1
    scale_max: float = 10
    criteria_revised: list[str] = field(default_factory=list)  # ids re-judged
    fact_check_findings: list[FactCheckFinding] = field(default_factory=list)
    viability: ViabilityAssessment = field(default_factory=ViabilityAssessment)
    raw_model_output: dict[str, Any] = field(default_factory=dict)


class JudgeAgent:
    def __init__(
        self,
        client: OllamaClient,
        rubric: dict[str, Any],
        max_chars_per_slide: int = 4000,
        max_slides_in_prompt: int = 60,
        vision_client: OllamaClient | None = None,
        enable_self_critique: bool = True,
        max_revision_rounds: int = 1,
        enable_viability_assessment: bool = True,
    ) -> None:
        self.client = client
        self.rubric = rubric
        self.max_chars_per_slide = max_chars_per_slide
        self.max_slides_in_prompt = max_slides_in_prompt
        self.vision_client = vision_client
        self.enable_self_critique = enable_self_critique
        self.enable_viability_assessment = enable_viability_assessment
        # Hard cap on re-judge rounds per criterion — bounds the loop so a
        # stubbornly low-confidence criterion can never cause unbounded
        # retries. One round is enough to give the model a real second
        # look; if it's still unsure after that, "low confidence" is
        # itself useful, honest information to put in the report.
        self.max_revision_rounds = max(0, max_revision_rounds)

        self.criteria: list[dict[str, Any]] = rubric.get("criteria", [])
        if not self.criteria:
            raise ValueError("Rubric has no 'criteria' defined.")

        scale = rubric.get("scale", {})
        self.scale_min = float(scale.get("min", 1))
        self.scale_max = float(scale.get("max", 10))

        total_weight = sum(c.get("weight", 0) for c in self.criteria)
        self._weight_normalizer = 100.0 / total_weight if total_weight else 1.0
        if total_weight != 100:
            logger.warning(
                "Rubric criterion weights sum to %s, not 100. Scores will be "
                "auto-normalized.",
                total_weight,
            )

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def judge(
        self,
        deck: DeckExtraction,
        summary: DeckSummary | None = None,
        fact_check_results: list[Any] | None = None,
    ) -> JudgeReport:
        result = self._initial_judge_pass(deck, summary, fact_check_results)
        report = self._build_report(result)
        report.fact_check_findings = self._fact_check_findings(fact_check_results)
        self._check_minimum_critique_counts(report, deck.deck_name)

        if self.enable_self_critique and self.max_revision_rounds > 0:
            report = self._run_self_critique(deck, summary, report)

        if self.enable_viability_assessment:
            report.viability = self._assess_viability(
                deck, summary, fact_check_results
            )

        return report

    def _assess_viability(
        self,
        deck: DeckExtraction,
        summary: DeckSummary | None,
        fact_check_results: list[Any] | None,
    ) -> ViabilityAssessment:
        """Dedicated holistic call: is this strategy actually executable?

        Run as its own LLM call (not folded into per-criterion scoring)
        because "is this viable in the real world" benefits from seeing
        research, competition, and GTM together, rather than graded in
        isolated rubric slices.
        """
        slides = deck.slides[: self.max_slides_in_prompt]
        deck_text = "\n\n".join(self._format_slide(s) for s in slides)
        summary_block = self._summary_block(summary)
        fact_check_block = self._fact_check_block(fact_check_results)

        prompt = f"""\
Deck name: {deck.deck_name}
Total slides: {deck.slide_count}
{summary_block}{fact_check_block}
Assess whether the strategy proposed in this deck is actually viable, per \
the rules in your system prompt. Respond with ONLY a JSON object matching \
this exact schema:
{VIABILITY_JSON_SCHEMA}

=== DECK CONTENT ===
{deck_text}
=== END DECK CONTENT ===
"""
        try:
            result = self.client.generate_json(
                prompt=prompt, system=VIABILITY_SYSTEM_PROMPT
            )
        except OllamaError as exc:
            logger.warning(
                "Viability assessment failed for '%s'; reporting as "
                "unassessed rather than crashing: %s",
                deck.deck_name,
                exc,
            )
            return ViabilityAssessment(
                verdict="unassessed",
                reasoning=f"Viability assessment call failed: {exc}",
            )

        verdict = str(result.get("verdict", "unassessed")).lower()
        if verdict not in VALID_VIABILITY_VERDICTS:
            logger.warning(
                "Viability model returned unrecognized verdict '%s' for "
                "'%s'; treating as unassessed.",
                verdict,
                deck.deck_name,
            )
            verdict = "unassessed"

        raw_cited = result.get("cited_slides", []) or []
        cited_slides = [
            int(i) for i in raw_cited if isinstance(i, (int, float))
        ]

        return ViabilityAssessment(
            verdict=verdict,
            reasoning=str(result.get("reasoning", "")),
            key_risks=list(result.get("key_risks", []) or []),
            what_would_need_to_be_true=list(
                result.get("what_would_need_to_be_true", []) or []
            ),
            cited_slides=cited_slides,
        )

    @staticmethod
    def _check_minimum_critique_counts(report: JudgeReport, deck_name: str) -> None:
        """Sanity-check that the judge actually surfaced enough weaknesses.

        This doesn't trigger an LLM re-judge (the per-criterion
        verification loop already handles fault-finding at that level);
        it's a deck-level visibility check, logged so a suspiciously
        thin weaknesses list is discoverable rather than silently
        accepted.
        """
        if len(report.weaknesses) < REQUIRED_MIN_WEAKNESSES:
            logger.warning(
                "Judge returned only %d weakness(es) for '%s' (expected >= %d). "
                "The model may be scoring too leniently overall — check the "
                "report's weaknesses section.",
                len(report.weaknesses),
                deck_name,
                REQUIRED_MIN_WEAKNESSES,
            )
        if len(report.strengths) < REQUIRED_MIN_STRENGTHS:
            logger.warning(
                "Judge returned 0 strengths for '%s'.", deck_name
            )

    @staticmethod
    def _fact_check_findings(fact_check_results: list[Any] | None) -> list[FactCheckFinding]:
        if not fact_check_results:
            return []
        findings = []
        for r in fact_check_results:
            findings.append(
                FactCheckFinding(
                    claim_text=getattr(r, "claim_text", ""),
                    slide=getattr(r, "slide", 0),
                    verdict=getattr(r, "verdict", "unverifiable"),
                    explanation=getattr(r, "explanation", ""),
                )
            )
        return findings

    @staticmethod
    def _fact_check_block(fact_check_results: list[Any] | None) -> str:
        if not fact_check_results:
            return ""
        lines = [
            "\nFact-check results for claims found in this deck (use per the "
            "rules in your system prompt — contradicted/outdated claims are "
            "citable faults, unverifiable claims are NOT penalties):"
        ]
        for r in fact_check_results:
            verdict = getattr(r, "verdict", "unverifiable")
            claim = getattr(r, "claim_text", "")
            slide = getattr(r, "slide", 0)
            explanation = getattr(r, "explanation", "")
            lines.append(
                f"- [Slide {slide}] \"{claim}\" -> {verdict.upper()}: {explanation}"
            )
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Stage 1: initial judge pass (all criteria, one call)
    # ------------------------------------------------------------------

    def _initial_judge_pass(
        self,
        deck: DeckExtraction,
        summary: DeckSummary | None,
        fact_check_results: list[Any] | None = None,
    ) -> dict[str, Any]:
        slides = deck.slides[: self.max_slides_in_prompt]
        deck_text = "\n\n".join(self._format_slide(s) for s in slides)
        summary_block = self._summary_block(summary)
        fact_check_block = self._fact_check_block(fact_check_results)
        criteria_block = self._criteria_block(self.criteria)
        json_schema = JUDGE_JSON_SCHEMA_TEMPLATE.format(
            min_score=self.scale_min, max_score=self.scale_max
        )

        prompt = f"""\
Deck name: {deck.deck_name}
Total slides: {deck.slide_count}
{summary_block}{fact_check_block}
You must score this deck against the following rubric criteria, on a \
scale from {self.scale_min} to {self.scale_max} for each:

{criteria_block}

Respond with ONLY a JSON object matching this exact schema (one \
criterion_scores entry per criterion listed above, using the exact id \
given for each):
{json_schema}

=== DECK CONTENT ===
{deck_text}
=== END DECK CONTENT ===
"""
        try:
            return self.client.generate_json(
                prompt=prompt, system=JUDGE_SYSTEM_PROMPT
            )
        except OllamaError as exc:
            logger.error("Judging failed for '%s': %s", deck.deck_name, exc)
            raise

    # ------------------------------------------------------------------
    # Stage 2 + 3: verify, then targeted re-judge for flagged criteria
    # ------------------------------------------------------------------

    def _run_self_critique(
        self,
        deck: DeckExtraction,
        summary: DeckSummary | None,
        report: JudgeReport,
    ) -> JudgeReport:
        slides_by_index = {s.index: s for s in deck.slides}
        flagged: list[CriterionScore] = []

        for cs in report.criterion_scores:
            notes = self._verify_criterion(cs, slides_by_index, deck.slide_count)
            cs.verification_notes = notes
            needs_revision = cs.confidence == "low" or len(notes) > 0
            if needs_revision:
                flagged.append(cs)

        if not flagged:
            logger.info(
                "Self-critique: all %d criteria passed verification on first "
                "pass for '%s'.",
                len(report.criterion_scores),
                deck.deck_name,
            )
            return report

        logger.info(
            "Self-critique: %d/%d criteria flagged for re-judging on '%s': %s",
            len(flagged),
            len(report.criterion_scores),
            deck.deck_name,
            [c.id for c in flagged],
        )

        criteria_by_id = {c["id"]: c for c in self.criteria}
        revised_ids: list[str] = []

        for cs in flagged:
            criterion_def = criteria_by_id.get(cs.id)
            if criterion_def is None:
                continue
            try:
                revised = self._re_judge_criterion(
                    deck=deck,
                    summary=summary,
                    criterion_def=criterion_def,
                    prior=cs,
                    slides_by_index=slides_by_index,
                )
            except OllamaError as exc:
                logger.warning(
                    "Re-judge failed for criterion '%s' on '%s'; keeping "
                    "original score. Error: %s",
                    cs.id,
                    deck.deck_name,
                    exc,
                )
                continue

            if revised is not None:
                self._apply_revision(report, cs.id, revised)
                revised_ids.append(cs.id)

        report.criteria_revised = revised_ids
        if revised_ids:
            report = self._recompute_weighted_score(report)
        return report

    def _verify_criterion(
        self,
        cs: CriterionScore,
        slides_by_index: dict[int, Any],
        slide_count: int,
    ) -> list[str]:
        """Independently sanity-check a criterion's output.

        Returns a list of human-readable problems found (empty = passed).
        This is deliberately cheap/local — no extra LLM call — so the
        verification step itself can't be the thing that's wrong.
        """
        notes: list[str] = []

        if cs.confidence not in VALID_CONFIDENCE_LEVELS:
            notes.append(
                f"Model reported confidence='{cs.confidence}', not a "
                f"recognized level; treating as low confidence."
            )
            cs.confidence = "low"

        if not cs.cited_slides:
            notes.append("No cited_slides given — reasoning isn't grounded.")
        else:
            bad_refs = [
                i for i in cs.cited_slides if i not in slides_by_index
            ]
            if bad_refs:
                notes.append(
                    f"Cited slide(s) {bad_refs} don't exist in this "
                    f"{slide_count}-slide deck — likely hallucinated."
                )

        word_count = len(cs.reasoning.split())
        if word_count < MIN_REASONING_WORDS_FOR_VERIFICATION:
            word_label = "word" if word_count == 1 else "words"
            notes.append(
                f"Reasoning is only {word_count} {word_label} — too short to "
                f"be substantive evidence."
            )

        fault_text = cs.fault_found.strip().lower()
        lazy_fault_markers = ("none", "n/a", "no fault", "no issues", "no flaws", "")
        if any(
            fault_text == marker or (marker and fault_text.startswith(marker))
            for marker in lazy_fault_markers
        ):
            notes.append(
                "No genuine fault was identified for this criterion — the "
                "judge may be scoring too leniently rather than actually "
                "finding a flaw."
            )

        return notes

    def _re_judge_criterion(
        self,
        deck: DeckExtraction,
        summary: DeckSummary | None,
        criterion_def: dict[str, Any],
        prior: CriterionScore,
        slides_by_index: dict[int, Any],
    ) -> dict[str, Any] | None:
        """Re-examine one criterion with closer attention to its cited slides.

        If a vision client is configured, re-runs vision on the cited
        slides (or, if no valid slides were cited, the first few slides)
        before re-judging — giving the model a genuinely better look,
        not just a second guess at the same information.
        """
        target_indices = [
            i for i in prior.cited_slides if i in slides_by_index
        ] or list(slides_by_index.keys())[: min(3, len(slides_by_index))]

        target_slides = [slides_by_index[i] for i in target_indices]

        if self.vision_client is not None:
            from casecomp_judge.agents.vision_reader import describe_slide_visually

            for slide in target_slides:
                if not slide.image_path:
                    continue
                logger.info(
                    "Self-critique: re-reading slide %d visually for "
                    "criterion '%s'...",
                    slide.index,
                    criterion_def["id"],
                )
                fresh_description = describe_slide_visually(
                    self.vision_client, slide, deck.deck_name
                )
                if fresh_description and "no additional visual" not in (
                    fresh_description.lower()
                ):
                    marker = "[Closer visual re-read]:"
                    if marker not in slide.text:
                        slide.text = f"{slide.text.strip()}\n\n{marker} {fresh_description}"

        focus_text = "\n\n".join(
            self._format_slide(s) for s in target_slides
        )
        wider_context = "\n\n".join(
            self._format_slide(s)
            for s in deck.slides[: self.max_slides_in_prompt]
            if s.index not in target_indices
        )

        summary_block = self._summary_block(summary)
        json_schema = RE_JUDGE_JSON_SCHEMA_TEMPLATE.format(
            min_score=self.scale_min, max_score=self.scale_max
        )

        prompt = f"""\
Deck name: {deck.deck_name}
Criterion to re-examine: "{criterion_def['id']}" | {criterion_def['name']} \
(weight {criterion_def.get('weight', 0)}): {criterion_def.get('description', '').strip()}
{summary_block}
Your first-pass answer for this criterion was:
  Score: {prior.score}
  Reasoning: {prior.reasoning}
  Fault found (if any): {prior.fault_found or "(none identified — this is itself part of why it was flagged)"}
  Confidence: {prior.confidence}
  Problems found on review: {"; ".join(prior.verification_notes) or "(flagged as low-confidence)"}

Below are the specific slides you cited (re-examined closely, including \
a fresh visual read of any charts/diagrams), followed by the rest of the \
deck for context. Re-score this ONE criterion only.

Respond with ONLY a JSON object matching this exact schema:
{json_schema}

=== CITED SLIDES (look closely here) ===
{focus_text}
=== END CITED SLIDES ===

=== REST OF DECK (context only) ===
{wider_context}
=== END REST OF DECK ===
"""
        return self.client.generate_json(
            prompt=prompt, system=RE_JUDGE_SYSTEM_PROMPT
        )

    @staticmethod
    def _apply_revision(
        report: JudgeReport, criterion_id: str, revised: dict[str, Any]
    ) -> None:
        for cs in report.criterion_scores:
            if cs.id != criterion_id:
                continue
            try:
                cs.score = float(revised.get("score", cs.score))
            except (TypeError, ValueError):
                pass
            cs.reasoning = str(revised.get("reasoning", cs.reasoning))
            cs.fault_found = str(revised.get("fault_found", cs.fault_found))
            confidence = str(revised.get("confidence", cs.confidence)).lower()
            cs.confidence = (
                confidence if confidence in VALID_CONFIDENCE_LEVELS else cs.confidence
            )
            cited = revised.get("cited_slides", cs.cited_slides)
            if isinstance(cited, list):
                cs.cited_slides = [
                    int(i) for i in cited if isinstance(i, (int, float))
                ]
            cs.was_revised = True
            return

    def _recompute_weighted_score(self, report: JudgeReport) -> JudgeReport:
        weighted_sum = sum(cs.score * cs.weight for cs in report.criterion_scores)
        total_weight = sum(cs.weight for cs in report.criterion_scores)
        weighted_score = weighted_sum / total_weight if total_weight else 0.0

        score_range = self.scale_max - self.scale_min
        weighted_score_pct = (
            ((weighted_score - self.scale_min) / score_range) * 100
            if score_range
            else 0.0
        )

        report.weighted_score = round(weighted_score, 2)
        report.weighted_score_pct = round(weighted_score_pct, 1)
        report.verdict_label = self._verdict_for_score(weighted_score)
        return report

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _summary_block(self, summary: DeckSummary | None) -> str:
        if summary is None:
            return ""
        return f"""
A prior summarization pass produced this structured summary of the deck \
(use it for context, but base your scores on the full deck content below, \
not just this summary):

Executive summary: {summary.executive_summary}
Problem statement: {summary.problem_statement}
Recommendation: {summary.recommendation}
Key insights: {"; ".join(summary.key_insights) or "(none extracted)"}
"""

    @staticmethod
    def _criteria_block(criteria: list[dict[str, Any]]) -> str:
        return "\n".join(
            f"- id: \"{c['id']}\" | {c['name']} (weight {c.get('weight', 0)}): "
            f"{c.get('description', '').strip()}"
            for c in criteria
        )

    def _format_slide(self, slide) -> str:  # noqa: ANN001
        text = slide.text.strip() or "[no extractable text on this slide]"
        if len(text) > self.max_chars_per_slide:
            text = text[: self.max_chars_per_slide] + " […truncated]"
        return f"--- Slide {slide.index} ---\n{text}"

    def _build_report(self, result: dict[str, Any]) -> JudgeReport:
        raw_scores = result.get("criterion_scores", []) or []
        raw_by_id = {s.get("id"): s for s in raw_scores if "id" in s}

        criterion_scores: list[CriterionScore] = []
        weighted_sum = 0.0
        total_weight_used = 0.0

        for c in self.criteria:
            cid = c["id"]
            raw = raw_by_id.get(cid)
            if raw is None:
                logger.warning(
                    "Judge model did not return a score for criterion '%s'; "
                    "defaulting to scale midpoint.",
                    cid,
                )
                score_val = (self.scale_min + self.scale_max) / 2
                reasoning = "(Model did not provide a score for this criterion.)"
                fault_found = ""
                confidence = "low"
                cited_slides: list[int] = []
            else:
                try:
                    score_val = float(raw.get("score", 0))
                except (TypeError, ValueError):
                    score_val = (self.scale_min + self.scale_max) / 2
                score_val = max(self.scale_min, min(self.scale_max, score_val))
                reasoning = str(raw.get("reasoning", ""))
                fault_found = str(raw.get("fault_found", ""))
                confidence = str(raw.get("confidence", "medium")).lower()
                if confidence not in VALID_CONFIDENCE_LEVELS:
                    confidence = "low"
                raw_cited = raw.get("cited_slides", []) or []
                cited_slides = [
                    int(i) for i in raw_cited if isinstance(i, (int, float))
                ]

            weight = float(c.get("weight", 0))
            criterion_scores.append(
                CriterionScore(
                    id=cid,
                    name=c.get("name", cid),
                    weight=weight,
                    score=score_val,
                    reasoning=reasoning,
                    fault_found=fault_found,
                    confidence=confidence,
                    cited_slides=cited_slides,
                )
            )
            weighted_sum += score_val * weight
            total_weight_used += weight

        weighted_score = (
            weighted_sum / total_weight_used if total_weight_used else 0.0
        )
        score_range = self.scale_max - self.scale_min
        weighted_score_pct = (
            ((weighted_score - self.scale_min) / score_range) * 100
            if score_range
            else 0.0
        )

        verdict_label = self._verdict_for_score(weighted_score)

        return JudgeReport(
            criterion_scores=criterion_scores,
            strengths=self._parse_strengths(result.get("strengths", []) or []),
            weaknesses=self._parse_weaknesses(result.get("weaknesses", []) or []),
            overall_feedback=str(result.get("overall_feedback", "")),
            weighted_score=round(weighted_score, 2),
            weighted_score_pct=round(weighted_score_pct, 1),
            verdict_label=verdict_label,
            rubric_name=self.rubric.get("rubric_name", "Rubric"),
            scale_min=self.scale_min,
            scale_max=self.scale_max,
            raw_model_output=result,
        )

    @staticmethod
    def _parse_strengths(raw_strengths: list[Any]) -> list[StrengthEntry]:
        """Parse strengths into structured entries.

        Falls back gracefully if the model returned plain strings
        instead of the requested objects — smaller/local models don't
        always follow nested schema instructions precisely, and a
        plain string is still useful information, just less detailed.
        """
        entries: list[StrengthEntry] = []
        for item in raw_strengths:
            if isinstance(item, dict):
                entries.append(
                    StrengthEntry(
                        description=str(item.get("description", "")),
                        slide=JudgeAgent._safe_int(item.get("slide", 0)),
                        category=str(item.get("category", "structural")),
                        significance=str(item.get("significance", "notable")),
                    )
                )
            elif isinstance(item, str) and item.strip():
                entries.append(StrengthEntry(description=item.strip()))
        return entries

    @staticmethod
    def _parse_weaknesses(raw_weaknesses: list[Any]) -> list[WeaknessEntry]:
        entries: list[WeaknessEntry] = []
        for item in raw_weaknesses:
            if isinstance(item, dict):
                entries.append(
                    WeaknessEntry(
                        description=str(item.get("description", "")),
                        slide=JudgeAgent._safe_int(item.get("slide", 0)),
                        category=str(item.get("category", "structural")),
                        severity=str(item.get("severity", "moderate")),
                        score_impact=str(item.get("score_impact", "")),
                    )
                )
            elif isinstance(item, str) and item.strip():
                entries.append(WeaknessEntry(description=item.strip()))
        return entries

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _verdict_for_score(self, score: float) -> str:
        bands = self.rubric.get("overall_verdict", {}).get("bands", [])
        for band in sorted(
            bands, key=lambda b: b.get("min_score", 0), reverse=True
        ):
            if score >= band.get("min_score", 0):
                return band.get("label", "")
        return "Unscored"
