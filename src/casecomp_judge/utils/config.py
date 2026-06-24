"""Configuration loading utilities.

Centralizes reading of config/config.yaml and config/rubric.yaml so
every module gets a consistent, validated view of settings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config/config.yaml")
DEFAULT_RUBRIC_PATH = Path("config/rubric.yaml")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path.resolve()}. "
            "Did you run this from the project root?"
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class OllamaConfig:
    model: str = "llama3"
    vision_model: str | None = None  # falls back to `model` if unset
    host: str = "http://localhost:11434"
    timeout_seconds: int = 300
    temperature: float = 0.2
    num_ctx: int = 8192


@dataclass
class ExtractionConfig:
    render_images: bool = True
    dpi: int = 150
    use_vision: bool = False
    max_chars_per_slide: int = 4000
    max_slides_in_prompt: int = 60


@dataclass
class WatchConfig:
    watch_dir: str = "data/watch"
    processed_dir: str = "data/processed"
    reports_dir: str = "data/reports"
    archive_dir: str = "data/archive"
    poll_interval_seconds: int = 5
    move_processed_to_archive: bool = True
    supported_extensions: list[str] = field(default_factory=lambda: [".pdf"])


@dataclass
class AgentRetryConfig:
    max_retries: int = 2
    retry_backoff_seconds: int = 3
    rubric_path: str = "config/rubric.yaml"


@dataclass
class SelfCritiqueConfig:
    enabled: bool = True
    max_revision_rounds: int = 1  # hard cap; 0 disables re-judging entirely


@dataclass
class FactCheckConfig:
    enabled: bool = True
    max_claims: int = 10  # cap on how many extracted claims get verified per deck
    max_search_results: int = 4  # search results fetched per claim


@dataclass
class ViabilityConfig:
    enabled: bool = True  # run the dedicated strategy-viability assessment


@dataclass
class AppConfig:
    ollama: OllamaConfig
    extraction: ExtractionConfig
    watch: WatchConfig
    summarizer: AgentRetryConfig
    judge: AgentRetryConfig
    judge_critique: SelfCritiqueConfig
    fact_checking: FactCheckConfig
    viability: ViabilityConfig
    log_level: str = "INFO"
    log_file: str = "data/pipeline.log"
    raw: dict[str, Any] = field(default_factory=dict)


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load and validate config.yaml into a typed AppConfig object."""
    raw = _load_yaml(Path(path))

    ollama_raw = raw.get("ollama", {})
    extraction_raw = raw.get("extraction", {})
    watch_raw = raw.get("watch", {})
    agents_raw = raw.get("agents", {})
    logging_raw = raw.get("logging", {})

    return AppConfig(
        ollama=OllamaConfig(
            model=ollama_raw.get("model", "llama3"),
            vision_model=ollama_raw.get("vision_model") or None,
            host=ollama_raw.get("host", "http://localhost:11434"),
            timeout_seconds=ollama_raw.get("timeout_seconds", 300),
            temperature=ollama_raw.get("temperature", 0.2),
            num_ctx=ollama_raw.get("num_ctx", 8192),
        ),
        extraction=ExtractionConfig(
            render_images=extraction_raw.get("render_images", True),
            dpi=extraction_raw.get("dpi", 150),
            use_vision=extraction_raw.get("use_vision", False),
            max_chars_per_slide=extraction_raw.get("max_chars_per_slide", 4000),
            max_slides_in_prompt=extraction_raw.get("max_slides_in_prompt", 60),
        ),
        watch=WatchConfig(
            watch_dir=watch_raw.get("watch_dir", "data/watch"),
            processed_dir=watch_raw.get("processed_dir", "data/processed"),
            reports_dir=watch_raw.get("reports_dir", "data/reports"),
            archive_dir=watch_raw.get("archive_dir", "data/archive"),
            poll_interval_seconds=watch_raw.get("poll_interval_seconds", 5),
            move_processed_to_archive=watch_raw.get(
                "move_processed_to_archive", True
            ),
            supported_extensions=watch_raw.get("supported_extensions", [".pdf"]),
        ),
        summarizer=AgentRetryConfig(
            max_retries=agents_raw.get("summarizer", {}).get("max_retries", 2),
            retry_backoff_seconds=agents_raw.get("summarizer", {}).get(
                "retry_backoff_seconds", 3
            ),
        ),
        judge=AgentRetryConfig(
            max_retries=agents_raw.get("judge", {}).get("max_retries", 2),
            retry_backoff_seconds=agents_raw.get("judge", {}).get(
                "retry_backoff_seconds", 3
            ),
            rubric_path=agents_raw.get("judge", {}).get(
                "rubric_path", "config/rubric.yaml"
            ),
        ),
        judge_critique=SelfCritiqueConfig(
            enabled=agents_raw.get("judge_critique", {}).get("enabled", True),
            max_revision_rounds=agents_raw.get("judge_critique", {}).get(
                "max_revision_rounds", 1
            ),
        ),
        fact_checking=FactCheckConfig(
            enabled=agents_raw.get("fact_checking", {}).get("enabled", True),
            max_claims=agents_raw.get("fact_checking", {}).get("max_claims", 10),
            max_search_results=agents_raw.get("fact_checking", {}).get(
                "max_search_results", 4
            ),
        ),
        viability=ViabilityConfig(
            enabled=agents_raw.get("viability", {}).get("enabled", True),
        ),
        log_level=logging_raw.get("level", "INFO"),
        log_file=logging_raw.get("log_file", "data/pipeline.log"),
        raw=raw,
    )


def load_rubric(path: Path | str = DEFAULT_RUBRIC_PATH) -> dict[str, Any]:
    """Load the rubric YAML as a plain dict (judge agent interprets it)."""
    return _load_yaml(Path(path))


def setup_logging(config: AppConfig) -> logging.Logger:
    """Configure root logging once; safe to call multiple times."""
    log_path = Path(config.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("casecomp_judge")
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
