"""Fact-checker agent.

Extracts clearly-checkable factual claims from a deck — named market
sizes, public company statistics, well-known industry figures — and
verifies each one against live web search results. Deliberately
narrow in scope: a team's own projections, internal estimates, and
assumptions are NOT fact-checked here, since those aren't factual
claims to begin with, they're the team's analysis. Penalizing a team
for a projection not matching a Google search would be incoherent.

Two-call design per deck:
  1. EXTRACT — one LLM call reads the full deck and pulls out only
     claims that are checkable external facts (not the team's own
     numbers/estimates), each tagged with the slide it came from and
     a short, search-friendly claim statement.
  2. VERIFY — for each extracted claim, search the web (no API key
     required — see utils/web_search.py) and ask the model to compare
     the claim against the search snippets, returning a verdict:
     "confirmed", "contradicted", "outdated", or "unverifiable" (no
     useful search results), plus a brief explanation.

This produces a list of FactCheckResult objects that the judge agent
weighs as evidence: a "contradicted" claim is a concrete, citable
fault; "unverifiable" claims are reported but not penalized — failure
to verify is not evidence of being wrong, and contradiction. The
sheer existence of a useless or rate-limited search result must never
silently turn into a deduction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from casecomp_judge.extraction.pdf_extractor import DeckExtraction
from casecomp_judge.utils.ollama_client import OllamaClient, OllamaError
from casecomp_judge.utils.web_search import search as web_search

logger = logging.getLogger("casecomp_judge.agents.fact_checker")


EXTRACT_SYSTEM_PROMPT = """\
You read business case competition decks and extract ONLY factual \
claims that are checkable against public, real-world information — \
named market sizes, growth rates, statistics about real companies, \
well-known industry figures, population/economic data, or similar \
claims that exist independently of this team's analysis.

You must NOT extract:
  - The team's own projections, forecasts, or estimates (e.g. "we \
    project 15% margin improvement") — these are the team's analysis, \
    not factual claims to be checked against the internet.
  - Numbers the team derived through their own calculations shown in \
    the deck (e.g. a TAM the team calculated from stated assumptions).
  - Vague or unquantified claims with nothing concrete to check.

You SHOULD extract things like:
  - "The global widget market is worth $40B" (checkable industry stat)
  - "Company X holds 30% market share" (checkable competitive fact)
  - "The US smartphone penetration rate is 85%" (checkable public stat)

If the deck contains no checkable claims, return an empty list — this \
is a normal and expected outcome for many decks, not a failure.

You always respond with valid JSON matching the exact schema given, and \
nothing else — no preamble, no markdown formatting, no explanation \
outside the JSON.
"""

EXTRACT_JSON_SCHEMA = """\
{
  "claims": [
    {
      "claim_text": "string, the factual claim as stated in the deck, concise",
      "search_query": "string, a short search-engine-friendly query to verify this claim",
      "slide": "integer, the slide number this claim appears on"
    }
  ]
}
"""

VERIFY_SYSTEM_PROMPT = """\
You are verifying a single factual claim from a business presentation \
against real web search results. Compare the claim to what the search \
results actually say. Be precise about what counts as confirmation: \
the search results must support the SAME factual claim, not just a \
related topic. A claim can be directionally right but quantitatively \
wrong (e.g. claim says $40B, sources say $25B) — that counts as \
contradicted, not confirmed, if the discrepancy is material.

If the search results are irrelevant, too sparse, or contradict each \
other without a clear consensus, the correct verdict is "unverifiable" \
— do not guess a verdict you can't actually support from the results \
given. An "unverifiable" verdict is not a failure and is not evidence \
that the claim is wrong.

You always respond with valid JSON matching the exact schema given, and \
nothing else.
"""

VERIFY_JSON_SCHEMA = """\
{
  "verdict": "one of: confirmed, contradicted, outdated, unverifiable",
  "explanation": "string, 1-2 sentences explaining the verdict with reference to what the search results actually showed"
}
"""

VALID_VERDICTS = {"confirmed", "contradicted", "outdated", "unverifiable"}


@dataclass
class ExtractedClaim:
    claim_text: str
    search_query: str
    slide: int


@dataclass
class FactCheckResult:
    claim_text: str
    slide: int
    verdict: str  # confirmed / contradicted / outdated / unverifiable
    explanation: str
    search_query: str
    sources_checked: int = 0


def extract_checkable_claims(
    client: OllamaClient, deck: DeckExtraction, max_slides_in_prompt: int = 60
) -> list[ExtractedClaim]:
    """First call: pull out only externally-checkable factual claims."""
    slides = deck.slides[:max_slides_in_prompt]
    deck_text = "\n\n".join(
        f"--- Slide {s.index} ---\n{s.text.strip() or '[no text]'}" for s in slides
    )

    prompt = f"""\
Deck name: {deck.deck_name}

Read the deck below and extract ONLY checkable external factual claims, \
per the rules in your system prompt. Respond with ONLY a JSON object \
matching this schema:
{EXTRACT_JSON_SCHEMA}

=== DECK CONTENT ===
{deck_text}
=== END DECK CONTENT ===
"""
    try:
        result = client.generate_json(prompt=prompt, system=EXTRACT_SYSTEM_PROMPT)
    except OllamaError as exc:
        logger.warning(
            "Claim extraction failed for '%s'; proceeding with no fact-checks: %s",
            deck.deck_name,
            exc,
        )
        return []

    claims_raw = result.get("claims", []) or []
    claims: list[ExtractedClaim] = []
    for c in claims_raw:
        try:
            claims.append(
                ExtractedClaim(
                    claim_text=str(c["claim_text"]),
                    search_query=str(c.get("search_query", c["claim_text"])),
                    slide=int(c.get("slide", 0)),
                )
            )
        except (KeyError, TypeError, ValueError):
            logger.debug("Skipping malformed extracted claim: %r", c)
            continue
    return claims


def verify_claim(
    client: OllamaClient,
    claim: ExtractedClaim,
    max_search_results: int = 4,
) -> FactCheckResult:
    """Search the web for one claim and ask the model to judge it."""
    results = web_search(claim.search_query, max_results=max_search_results)

    if not results:
        return FactCheckResult(
            claim_text=claim.claim_text,
            slide=claim.slide,
            verdict="unverifiable",
            explanation=(
                "No usable web search results were found for this claim "
                "(search may have failed or returned nothing relevant)."
            ),
            search_query=claim.search_query,
            sources_checked=0,
        )

    sources_block = "\n\n".join(
        f"[Source {i+1}] {r.title}\n{r.snippet}\n({r.url})"
        for i, r in enumerate(results)
    )

    prompt = f"""\
Claim to verify: "{claim.claim_text}"

Search results:
{sources_block}

Respond with ONLY a JSON object matching this schema:
{VERIFY_JSON_SCHEMA}
"""
    try:
        result = client.generate_json(prompt=prompt, system=VERIFY_SYSTEM_PROMPT)
    except OllamaError as exc:
        logger.warning(
            "Claim verification failed for %r: %s", claim.claim_text, exc
        )
        return FactCheckResult(
            claim_text=claim.claim_text,
            slide=claim.slide,
            verdict="unverifiable",
            explanation=f"Verification call failed: {exc}",
            search_query=claim.search_query,
            sources_checked=len(results),
        )

    verdict = str(result.get("verdict", "unverifiable")).lower()
    if verdict not in VALID_VERDICTS:
        verdict = "unverifiable"

    return FactCheckResult(
        claim_text=claim.claim_text,
        slide=claim.slide,
        verdict=verdict,
        explanation=str(result.get("explanation", "")),
        search_query=claim.search_query,
        sources_checked=len(results),
    )


def fact_check_deck(
    client: OllamaClient,
    deck: DeckExtraction,
    max_claims: int = 10,
    max_slides_in_prompt: int = 60,
    max_search_results: int = 4,
) -> list[FactCheckResult]:
    """Full fact-check pass: extract claims, then verify each one.

    Capped at `max_claims` to bound latency/search-call volume on decks
    with many checkable claims — the first `max_claims` extracted are
    used; this is a deliberate cost ceiling, not a quality judgment
    about which claims matter most.
    """
    claims = extract_checkable_claims(client, deck, max_slides_in_prompt)
    if not claims:
        logger.info("No checkable factual claims found in '%s'.", deck.deck_name)
        return []

    if len(claims) > max_claims:
        logger.info(
            "Found %d checkable claims in '%s'; checking first %d (max_claims cap).",
            len(claims),
            deck.deck_name,
            max_claims,
        )
        claims = claims[:max_claims]

    results: list[FactCheckResult] = []
    for claim in claims:
        logger.info(
            "Fact-checking claim from slide %d: %r", claim.slide, claim.claim_text
        )
        results.append(
            verify_claim(client, claim, max_search_results=max_search_results)
        )

    confirmed = sum(1 for r in results if r.verdict == "confirmed")
    contradicted = sum(1 for r in results if r.verdict == "contradicted")
    outdated = sum(1 for r in results if r.verdict == "outdated")
    unverifiable = sum(1 for r in results if r.verdict == "unverifiable")
    logger.info(
        "Fact-check complete for '%s': %d confirmed, %d contradicted, "
        "%d outdated, %d unverifiable.",
        deck.deck_name,
        confirmed,
        contradicted,
        outdated,
        unverifiable,
    )
    return results
