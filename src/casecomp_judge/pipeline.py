"""Pipeline orchestrator.

Wires together: detect (DeckSource) -> process (extraction) ->
summarise (SummarizerAgent) -> judge (JudgeAgent) -> report (reporting).

This is the single place that knows the full sequence; every stage
module above is independently testable and replaceable.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from casecomp_judge.agents.fact_checker import fact_check_deck
from casecomp_judge.agents.judge import JudgeAgent, JudgeReport
from casecomp_judge.agents.summarizer import DeckSummary, SummarizerAgent
from casecomp_judge.agents.vision_reader import build_vision_enriched_text
from casecomp_judge.extraction.pdf_extractor import DeckExtraction, extract_deck
from casecomp_judge.ingestion.base import DeckEvent
from casecomp_judge.ingestion.watcher import FolderWatchSource
from casecomp_judge.utils.config import AppConfig, load_rubric
from casecomp_judge.utils.ollama_client import OllamaClient
from casecomp_judge.utils.reporting import write_report

logger = logging.getLogger("casecomp_judge.pipeline")


@dataclass
class PipelineResult:
    deck_name: str
    success: bool
    error: str | None = None
    markdown_report_path: str | None = None
    json_report_path: str | None = None
    weighted_score: float | None = None
    verdict_label: str | None = None
    criteria_revised: list[str] = field(default_factory=list)
    fact_checks_performed: int = 0


class CaseCompPipeline:
    """End-to-end pipeline for a single deck, reusable across batch/watch/single modes."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

        rubric = load_rubric(config.judge.rubric_path)

        self.summarizer_client = OllamaClient(
            model=config.ollama.model,
            host=config.ollama.host,
            timeout_seconds=config.ollama.timeout_seconds,
            temperature=config.ollama.temperature,
            num_ctx=config.ollama.num_ctx,
            max_retries=config.summarizer.max_retries,
            retry_backoff_seconds=config.summarizer.retry_backoff_seconds,
        )
        self.judge_client = OllamaClient(
            model=config.ollama.model,
            host=config.ollama.host,
            timeout_seconds=config.ollama.timeout_seconds,
            temperature=config.ollama.temperature,
            num_ctx=config.ollama.num_ctx,
            max_retries=config.judge.max_retries,
            retry_backoff_seconds=config.judge.retry_backoff_seconds,
        )

        self.summarizer = SummarizerAgent(
            client=self.summarizer_client,
            max_chars_per_slide=config.extraction.max_chars_per_slide,
            max_slides_in_prompt=config.extraction.max_slides_in_prompt,
        )

        self.fact_check_client: OllamaClient | None = None
        if config.fact_checking.enabled:
            self.fact_check_client = OllamaClient(
                model=config.ollama.model,
                host=config.ollama.host,
                timeout_seconds=config.ollama.timeout_seconds,
                temperature=config.ollama.temperature,
                num_ctx=config.ollama.num_ctx,
                max_retries=config.summarizer.max_retries,
                retry_backoff_seconds=config.summarizer.retry_backoff_seconds,
            )

        self.vision_client: OllamaClient | None = None
        if config.extraction.use_vision:
            vision_model = config.ollama.vision_model or config.ollama.model
            self.vision_client = OllamaClient(
                model=vision_model,
                host=config.ollama.host,
                timeout_seconds=config.ollama.timeout_seconds,
                temperature=config.ollama.temperature,
                num_ctx=config.ollama.num_ctx,
                max_retries=config.summarizer.max_retries,
                retry_backoff_seconds=config.summarizer.retry_backoff_seconds,
            )
            if not config.extraction.render_images:
                logger.warning(
                    "extraction.use_vision is true but extraction.render_images "
                    "is false — no slide images will exist to read. Enable "
                    "render_images, or vision will have no effect."
                )

        self.judge = JudgeAgent(
            client=self.judge_client,
            rubric=rubric,
            max_chars_per_slide=config.extraction.max_chars_per_slide,
            max_slides_in_prompt=config.extraction.max_slides_in_prompt,
            vision_client=self.vision_client,
            enable_self_critique=config.judge_critique.enabled,
            max_revision_rounds=config.judge_critique.max_revision_rounds,
            enable_viability_assessment=config.viability.enabled,
        )

        self.source = FolderWatchSource(
            watch_dir=config.watch.watch_dir,
            processed_dir=config.watch.processed_dir,
            supported_extensions=config.watch.supported_extensions,
        )

    def check_ollama(self) -> bool:
        ok = self.judge_client.health_check()
        if self.vision_client is not None:
            ok = self.vision_client.health_check() and ok
        return ok

    def process_path(self, pdf_path: str | Path) -> PipelineResult:
        """Run the full detect-skipped / process / summarise / judge /
        report sequence for a single PDF path."""
        pdf_path = Path(pdf_path)
        deck_name = pdf_path.stem
        logger.info("=== Processing deck: %s ===", deck_name)

        try:
            deck_output_dir = Path(self.config.watch.processed_dir) / deck_name
            deck: DeckExtraction = extract_deck(
                pdf_path=pdf_path,
                output_dir=deck_output_dir,
                render_images=self.config.extraction.render_images,
                dpi=self.config.extraction.dpi,
            )

            if self.vision_client is not None:
                logger.info(
                    "Reading slide visuals for '%s' with vision model '%s'...",
                    deck_name,
                    self.vision_client.model,
                )
                descriptions = build_vision_enriched_text(
                    self.vision_client,
                    deck,
                    max_slides=self.config.extraction.max_slides_in_prompt,
                )
                deck.merge_vision_descriptions(descriptions)

            logger.info("Summarising '%s'...", deck_name)
            summary: DeckSummary = self.summarizer.summarize(deck)

            fact_check_results = []
            if self.fact_check_client is not None:
                logger.info("Fact-checking claims in '%s'...", deck_name)
                fact_check_results = fact_check_deck(
                    self.fact_check_client,
                    deck,
                    max_claims=self.config.fact_checking.max_claims,
                    max_slides_in_prompt=self.config.extraction.max_slides_in_prompt,
                    max_search_results=self.config.fact_checking.max_search_results,
                )

            logger.info("Judging '%s'...", deck_name)
            judge_report: JudgeReport = self.judge.judge(
                deck, summary=summary, fact_check_results=fact_check_results
            )
            if judge_report.criteria_revised:
                logger.info(
                    "Self-critique revised %d criterion/criteria for '%s': %s",
                    len(judge_report.criteria_revised),
                    deck_name,
                    judge_report.criteria_revised,
                )

            md_path, json_path = write_report(
                deck=deck,
                summary=summary,
                judge_report=judge_report,
                reports_dir=self.config.watch.reports_dir,
            )

            if self.config.watch.move_processed_to_archive:
                self._archive(pdf_path)

            logger.info(
                "Done: %s -> score %.2f/%s (%s)",
                deck_name,
                judge_report.weighted_score,
                judge_report.scale_max,
                judge_report.verdict_label,
            )

            return PipelineResult(
                deck_name=deck_name,
                success=True,
                markdown_report_path=str(md_path),
                json_report_path=str(json_path),
                weighted_score=judge_report.weighted_score,
                verdict_label=judge_report.verdict_label,
                criteria_revised=judge_report.criteria_revised,
                fact_checks_performed=len(fact_check_results),
            )

        except Exception as exc:  # noqa: BLE001 - surface all failures per-deck
            logger.exception("Failed to process deck '%s'", deck_name)
            return PipelineResult(deck_name=deck_name, success=False, error=str(exc))

    def _archive(self, pdf_path: Path) -> None:
        archive_dir = Path(self.config.watch.archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)
        dest = archive_dir / pdf_path.name
        try:
            shutil.move(str(pdf_path), str(dest))
            logger.info("Archived '%s' -> '%s'", pdf_path.name, dest)
        except OSError as exc:
            logger.warning("Could not archive '%s': %s", pdf_path.name, exc)

    def run_batch(self) -> list[PipelineResult]:
        """Process every currently-unseen deck in the watch directory once, then exit."""
        events: list[DeckEvent] = self.source.poll_once()
        if not events:
            logger.info(
                "No new decks found in '%s'.", self.config.watch.watch_dir
            )
            return []

        results = [self.process_path(event.path) for event in events]
        self._log_summary(results)
        return results

    def run_watch(self) -> None:
        """Run forever, processing new decks as they arrive."""
        try:
            for event in self.source.watch(
                poll_interval_seconds=self.config.watch.poll_interval_seconds
            ):
                self.process_path(event.path)
        except KeyboardInterrupt:
            logger.info("Watch mode stopped by user.")

    @staticmethod
    def _log_summary(results: list[PipelineResult]) -> None:
        succeeded = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        logger.info(
            "Batch complete: %d succeeded, %d failed.", len(succeeded), len(failed)
        )
        for r in failed:
            logger.error("  FAILED: %s — %s", r.deck_name, r.error)
