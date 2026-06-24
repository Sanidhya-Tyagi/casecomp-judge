"""Command-line interface for the casecomp judge pipeline.

Usage:
    python run.py batch              # process all decks currently in watch dir, then exit
    python run.py watch              # run forever, processing new decks as they arrive
    python run.py file <path.pdf>    # process a single specific file

Reads a `.env` file in the project root if present (e.g. for
TAVILY_API_KEY) — see `.env.example`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from dotenv import load_dotenv

from casecomp_judge.pipeline import CaseCompPipeline
from casecomp_judge.utils.config import load_config, setup_logging

load_dotenv()  # no-op if .env doesn't exist; never overrides already-set env vars


@click.group()
@click.option(
    "--config",
    "config_path",
    default="config/config.yaml",
    show_default=True,
    help="Path to config.yaml",
)
@click.pass_context
def cli(ctx: click.Context, config_path: str) -> None:
    """CaseComp Judge — agentic pipeline for casecomp deck PDFs."""
    ctx.ensure_object(dict)
    app_config = load_config(config_path)
    setup_logging(app_config)
    ctx.obj["config"] = app_config


@cli.command()
@click.pass_context
def batch(ctx: click.Context) -> None:
    """Process every deck currently sitting in the watch folder, then exit."""
    pipeline = CaseCompPipeline(ctx.obj["config"])
    if not pipeline.check_ollama():
        click.echo(
            "⚠️  Could not verify Ollama is reachable. Continuing anyway — "
            "make sure `ollama serve` is running.",
            err=True,
        )
    results = pipeline.run_batch()
    if not results:
        click.echo("No new decks to process.")
        return
    for r in results:
        if r.success:
            click.echo(
                f"✅ {r.deck_name}: {r.weighted_score} ({r.verdict_label}) "
                f"-> {r.markdown_report_path}"
            )
        else:
            click.echo(f"❌ {r.deck_name}: FAILED — {r.error}", err=True)
    failed_count = sum(1 for r in results if not r.success)
    if failed_count:
        sys.exit(1)


@cli.command()
@click.pass_context
def watch(ctx: click.Context) -> None:
    """Watch the configured folder forever, processing new decks as they land."""
    pipeline = CaseCompPipeline(ctx.obj["config"])
    if not pipeline.check_ollama():
        click.echo(
            "⚠️  Could not verify Ollama is reachable. Continuing anyway — "
            "make sure `ollama serve` is running.",
            err=True,
        )
    pipeline.run_watch()


@cli.command(name="file")
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def process_file(ctx: click.Context, pdf_path: Path) -> None:
    """Process a single specific PDF file (does not need to be in the watch folder)."""
    pipeline = CaseCompPipeline(ctx.obj["config"])
    if not pipeline.check_ollama():
        click.echo(
            "⚠️  Could not verify Ollama is reachable. Continuing anyway — "
            "make sure `ollama serve` is running.",
            err=True,
        )
    result = pipeline.process_path(pdf_path)
    if result.success:
        click.echo(
            f"✅ {result.deck_name}: {result.weighted_score} "
            f"({result.verdict_label})"
        )
        click.echo(f"   Markdown report: {result.markdown_report_path}")
        click.echo(f"   JSON report:     {result.json_report_path}")
    else:
        click.echo(f"❌ {result.deck_name}: FAILED — {result.error}", err=True)
        sys.exit(1)


def main() -> None:
    cli(obj={})


if __name__ == "__main__":
    main()
