"""Report generation.

Combines extraction metadata, the summary, and the judge report into
a single per-deck report, written as both Markdown (human-readable)
and JSON (machine-readable, designed so a future ranking pass can
simply glob and load all report.json files).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from casecomp_judge.agents.judge import JudgeReport
from casecomp_judge.agents.summarizer import DeckSummary
from casecomp_judge.extraction.pdf_extractor import DeckExtraction


def write_report(
    deck: DeckExtraction,
    summary: DeckSummary,
    judge_report: JudgeReport,
    reports_dir: str | Path,
) -> tuple[Path, Path]:
    """Write report.md and report.json for a single deck.

    Returns (markdown_path, json_path).
    """
    out_dir = Path(reports_dir) / deck.deck_name
    out_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).isoformat()

    json_payload = {
        "deck_name": deck.deck_name,
        "source_path": deck.source_path,
        "slide_count": deck.slide_count,
        "generated_at": generated_at,
        "extraction_warnings": deck.warnings,
        "summary": asdict(summary),
        "judge": {
            "rubric_name": judge_report.rubric_name,
            "scale_min": judge_report.scale_min,
            "scale_max": judge_report.scale_max,
            "weighted_score": judge_report.weighted_score,
            "weighted_score_pct": judge_report.weighted_score_pct,
            "verdict_label": judge_report.verdict_label,
            "criteria_revised": judge_report.criteria_revised,
            "criterion_scores": [
                asdict(cs) for cs in judge_report.criterion_scores
            ],
            "strengths": [asdict(s) for s in judge_report.strengths],
            "weaknesses": [asdict(w) for w in judge_report.weaknesses],
            "overall_feedback": judge_report.overall_feedback,
            "fact_check_findings": [
                asdict(f) for f in judge_report.fact_check_findings
            ],
            "viability": asdict(judge_report.viability),
        },
    }
    # Drop noisy raw model dumps from the on-disk JSON to keep it clean;
    # they're already folded into the structured fields above.
    json_payload["summary"].pop("raw_model_output", None)

    json_path = out_dir / "report.json"
    json_path.write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md_path = out_dir / "report.md"
    md_path.write_text(_render_markdown(json_payload), encoding="utf-8")

    return md_path, json_path


def _render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    judge = payload["judge"]

    lines: list[str] = []
    lines.append(f"# Judge Report: {payload['deck_name']}")
    lines.append("")
    lines.append(
        f"*Generated {payload['generated_at']} · "
        f"{payload['slide_count']} slides · "
        f"Source: `{payload['source_path']}`*"
    )
    lines.append("")

    if payload.get("extraction_warnings"):
        lines.append("> ⚠️ **Extraction warnings:**")
        for w in payload["extraction_warnings"]:
            lines.append(f"> - {w}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Overall Verdict")
    lines.append("")
    lines.append(
        f"**Score: {judge['weighted_score']} / {judge['scale_max']}** "
        f"({judge['weighted_score_pct']}%)"
    )
    lines.append("")
    lines.append(f"**Verdict:** {judge['verdict_label']}")
    lines.append("")
    lines.append(f"*Rubric used: {judge['rubric_name']}*")
    lines.append("")

    viability = judge.get("viability", {}) or {}
    if viability.get("verdict") and viability["verdict"] != "unassessed":
        lines.append("---")
        lines.append("")
        lines.append("## Strategic Viability")
        lines.append("")
        viability_marker = {
            "viable": "✅ VIABLE",
            "viable_with_caveats": "⚠️ VIABLE WITH CAVEATS",
            "unrealistic": "❌ UNREALISTIC",
        }.get(viability["verdict"], viability["verdict"].upper())
        lines.append(f"**{viability_marker}**")
        lines.append("")
        lines.append(viability.get("reasoning", "") or "_No reasoning provided._")
        lines.append("")
        if viability.get("key_risks"):
            lines.append("**Key risks to execution:**")
            for risk in viability["key_risks"]:
                lines.append(f"- {risk}")
            lines.append("")
        if viability.get("what_would_need_to_be_true"):
            lines.append("**What would need to be true for this to succeed:**")
            for cond in viability["what_would_need_to_be_true"]:
                lines.append(f"- {cond}")
            lines.append("")
        if viability.get("cited_slides"):
            cited_str = ", ".join(f"Slide {i}" for i in viability["cited_slides"])
            lines.append(f"*Grounded in: {cited_str}*")
            lines.append("")
    elif viability.get("verdict") == "unassessed":
        lines.append("> ℹ️ *Strategic viability assessment was not completed for "
                      "this deck (call failed or was disabled).*")
        lines.append("")

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(summary.get("executive_summary", "") or "_Not available._")
    lines.append("")

    lines.append("## Deck Summary")
    lines.append("")
    lines.append(f"**Problem Statement:** {summary.get('problem_statement', '')}")
    lines.append("")
    lines.append(f"**Approach:** {summary.get('approach', '')}")
    lines.append("")
    lines.append(f"**Recommendation:** {summary.get('recommendation', '')}")
    lines.append("")
    lines.append(
        f"**Financial Highlights:** {summary.get('financial_highlights', '')}"
    )
    lines.append("")

    if summary.get("key_insights"):
        lines.append("**Key Insights:**")
        for item in summary["key_insights"]:
            lines.append(f"- {item}")
        lines.append("")

    if summary.get("risks_and_caveats_noted"):
        lines.append("**Risks/Caveats Noted by Team:**")
        for item in summary["risks_and_caveats_noted"]:
            lines.append(f"- {item}")
        lines.append("")

    if summary.get("missing_or_unclear"):
        lines.append("**Missing or Unclear (per summarizer):**")
        for item in summary["missing_or_unclear"]:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Rubric Scoring")
    lines.append("")

    revised_ids = set(judge.get("criteria_revised", []) or [])
    if revised_ids:
        lines.append(
            f"*Self-critique re-judged {len(revised_ids)} criterion/criteria "
            f"after the first pass flagged them as low-confidence or "
            f"unverified — see notes below.*"
        )
        lines.append("")

    lines.append("| Criterion | Weight | Score | Confidence | Revised? |")
    lines.append("|---|---|---|---|---|")
    for cs in judge["criterion_scores"]:
        revised_marker = "✏️ Yes" if cs.get("was_revised") else "—"
        lines.append(
            f"| {cs['name']} | {cs['weight']}% | "
            f"{cs['score']} / {judge['scale_max']} | "
            f"{cs.get('confidence', 'medium')} | {revised_marker} |"
        )
    lines.append("")

    lines.append("### Criterion Reasoning")
    lines.append("")
    for cs in judge["criterion_scores"]:
        confidence = cs.get("confidence", "medium")
        cited = cs.get("cited_slides", [])
        cited_str = (
            ", ".join(f"Slide {i}" for i in cited) if cited else "no slides cited"
        )
        lines.append(
            f"**{cs['name']}** ({cs['score']}/{judge['scale_max']}) "
            f"— confidence: *{confidence}*, grounded in: {cited_str}"
        )
        lines.append("")
        lines.append(cs["reasoning"] or "_No reasoning provided._")
        if cs.get("fault_found"):
            lines.append("")
            lines.append(f"**Fault identified:** {cs['fault_found']}")
        if cs.get("was_revised"):
            lines.append("")
            lines.append(
                "> 🔁 *This score was revised after a self-critique pass "
                "flagged the first answer as low-confidence or unverified.*"
            )
        notes = cs.get("verification_notes", [])
        if notes:
            lines.append("")
            lines.append("> Verification notes from first pass:")
            for n in notes:
                lines.append(f"> - {n}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Strengths")
    lines.append("")
    strengths = judge.get("strengths", []) or []
    if not strengths:
        lines.append("_None noted._")
    for s in strengths:
        sig = s.get("significance", "notable")
        sig_marker = {"major": "🟢🟢", "notable": "🟢", "minor": "⚪"}.get(sig, "")
        slide_ref = f" (Slide {s['slide']})" if s.get("slide") else ""
        lines.append(f"- {sig_marker} **[{sig}]**{slide_ref} {s.get('description', '')}")
    lines.append("")

    lines.append("## Weaknesses")
    lines.append("")
    weaknesses = judge.get("weaknesses", []) or []
    if len(weaknesses) < 2:
        lines.append(
            "> ⚠️ *Fewer than 2 weaknesses were identified for this deck — "
            "this may indicate overly lenient judging. Review the criterion "
            "reasoning above for a closer look.*"
        )
        lines.append("")
    if not weaknesses:
        lines.append("_None noted._")
    for w in weaknesses:
        sev = w.get("severity", "moderate")
        sev_marker = {"major": "🔴", "moderate": "🟠", "minor": "🟡"}.get(sev, "")
        slide_ref = f" (Slide {w['slide']})" if w.get("slide") else ""
        lines.append(f"- {sev_marker} **[{sev}]**{slide_ref} {w.get('description', '')}")
        if w.get("score_impact"):
            lines.append(f"  - *Score impact:* {w['score_impact']}")
    lines.append("")

    fact_checks = judge.get("fact_check_findings", []) or []
    if fact_checks:
        lines.append("---")
        lines.append("")
        lines.append("## Fact-Check Results")
        lines.append("")
        lines.append(
            "*Claims below are clearly checkable external facts only — the "
            "team's own projections and estimates are never fact-checked.*"
        )
        lines.append("")
        verdict_marker = {
            "confirmed": "✅",
            "contradicted": "❌",
            "outdated": "🕐",
            "unverifiable": "❓",
        }
        for f in fact_checks:
            marker = verdict_marker.get(f.get("verdict", ""), "")
            slide_ref = f" (Slide {f['slide']})" if f.get("slide") else ""
            lines.append(
                f"- {marker} **{f.get('verdict', 'unverifiable').upper()}**"
                f"{slide_ref}: \"{f.get('claim_text', '')}\""
            )
            if f.get("explanation"):
                lines.append(f"  - {f['explanation']}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Overall Feedback")
    lines.append("")
    lines.append(judge.get("overall_feedback", "") or "_Not available._")
    lines.append("")

    return "\n".join(lines)
