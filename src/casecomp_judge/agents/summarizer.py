"""Summarizer agent.

Reads extracted deck text and produces a structured summary: problem
statement, approach/framework used, key insights, financial highlights,
final recommendation, and a one-paragraph executive summary.

Uses Ollama's JSON mode for reliable, parseable structured output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from casecomp_judge.extraction.pdf_extractor import DeckExtraction
from casecomp_judge.utils.ollama_client import OllamaClient, OllamaError

logger = logging.getLogger("casecomp_judge.agents.summarizer")


SUMMARIZER_SYSTEM_PROMPT = """\
You are an expert business case competition analyst. You read case \
competition slide decks (extracted as text, slide by slide) and produce \
accurate, neutral, structured summaries. You never invent information \
that is not present in the deck. If something is unclear or missing, say \
so explicitly rather than guessing. You always respond with valid JSON \
matching the exact schema given, and nothing else — no preamble, no \
markdown formatting, no explanation outside the JSON.
"""

SUMMARY_JSON_SCHEMA = """\
{
  "executive_summary": "string, 2-4 sentences, the single most informative summary of the whole deck",
  "problem_statement": "string, the core business problem as the team framed it",
  "approach": "string, frameworks/methodologies the team used (e.g. Porter's Five Forces, market sizing, etc.) and how they applied them",
  "key_insights": ["string", "..."],
  "recommendation": "string, the team's final recommendation, as specific as the deck states it",
  "financial_highlights": "string, key numbers/financials mentioned (revenue, cost, ROI, market size, etc.) or 'Not substantively addressed' if absent",
  "risks_and_caveats_noted": ["string", "..."],
  "missing_or_unclear": ["string - things the deck should have addressed but didn't, or that were ambiguous"]
}
"""


@dataclass
class DeckSummary:
    executive_summary: str = ""
    problem_statement: str = ""
    approach: str = ""
    key_insights: list[str] = field(default_factory=list)
    recommendation: str = ""
    financial_highlights: str = ""
    risks_and_caveats_noted: list[str] = field(default_factory=list)
    missing_or_unclear: list[str] = field(default_factory=list)
    raw_model_output: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeckSummary":
        return cls(
            executive_summary=str(data.get("executive_summary", "")),
            problem_statement=str(data.get("problem_statement", "")),
            approach=str(data.get("approach", "")),
            key_insights=list(data.get("key_insights", []) or []),
            recommendation=str(data.get("recommendation", "")),
            financial_highlights=str(data.get("financial_highlights", "")),
            risks_and_caveats_noted=list(
                data.get("risks_and_caveats_noted", []) or []
            ),
            missing_or_unclear=list(data.get("missing_or_unclear", []) or []),
            raw_model_output=data,
        )


class SummarizerAgent:
    def __init__(
        self,
        client: OllamaClient,
        max_chars_per_slide: int = 4000,
        max_slides_in_prompt: int = 60,
    ) -> None:
        self.client = client
        self.max_chars_per_slide = max_chars_per_slide
        self.max_slides_in_prompt = max_slides_in_prompt

    def summarize(self, deck: DeckExtraction) -> DeckSummary:
        slides = deck.slides[: self.max_slides_in_prompt]
        truncated_note = ""
        if len(deck.slides) > self.max_slides_in_prompt:
            truncated_note = (
                f"\n\n[NOTE: deck has {len(deck.slides)} slides; only the "
                f"first {self.max_slides_in_prompt} are shown below due to "
                f"context limits.]"
            )

        deck_text = "\n\n".join(
            self._format_slide(s) for s in slides
        )

        prompt = f"""\
Deck name: {deck.deck_name}
Total slides: {deck.slide_count}{truncated_note}

Below is the extracted text of a case competition slide deck, slide by \
slide. Read it carefully and produce a structured summary.

Respond with ONLY a JSON object matching this exact schema:
{SUMMARY_JSON_SCHEMA}

=== DECK CONTENT ===
{deck_text}
=== END DECK CONTENT ===
"""

        try:
            result = self.client.generate_json(
                prompt=prompt, system=SUMMARIZER_SYSTEM_PROMPT
            )
        except OllamaError as exc:
            logger.error("Summarization failed for '%s': %s", deck.deck_name, exc)
            raise

        return DeckSummary.from_dict(result)

    def _format_slide(self, slide) -> str:  # noqa: ANN001
        text = slide.text.strip() or "[no extractable text on this slide]"
        if len(text) > self.max_chars_per_slide:
            text = text[: self.max_chars_per_slide] + " […truncated]"
        return f"--- Slide {slide.index} ---\n{text}"
