"""Folder-watching implementation of DeckSource.

Watches a directory for new files matching supported extensions
(default: .pdf). Tracks which files have already been handed off
(by name) in an in-memory + on-disk marker so re-running batch mode
doesn't reprocess old decks, while watch mode can run indefinitely.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from casecomp_judge.ingestion.base import DeckEvent, DeckSource

logger = logging.getLogger("casecomp_judge.ingestion")

SEEN_MARKER_FILENAME = ".seen_files"


class FolderWatchSource(DeckSource):
    """Detects new deck files dropped into a local directory."""

    def __init__(
        self,
        watch_dir: str | Path,
        processed_dir: str | Path,
        supported_extensions: list[str] | None = None,
    ) -> None:
        self.watch_dir = Path(watch_dir)
        self.processed_dir = Path(processed_dir)
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.supported_extensions = {
            ext.lower() for ext in (supported_extensions or [".pdf"])
        }
        self._marker_path = self.processed_dir / SEEN_MARKER_FILENAME
        self._seen: set[str] = self._load_seen()

    def _load_seen(self) -> set[str]:
        if self._marker_path.exists():
            return set(
                line.strip()
                for line in self._marker_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        return set()

    def _mark_seen(self, filename: str) -> None:
        self._seen.add(filename)
        with open(self._marker_path, "a", encoding="utf-8") as f:
            f.write(filename + "\n")

    def _candidate_files(self) -> list[Path]:
        files = []
        for entry in sorted(self.watch_dir.iterdir()):
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in self.supported_extensions:
                continue
            if entry.name in self._seen:
                continue
            files.append(entry)
        return files

    def poll_once(self) -> list[DeckEvent]:
        """Return DeckEvents for any new, unseen files in watch_dir.

        A file is only considered 'ready' if its size is stable across
        two checks 0.5s apart — guards against picking up a file that's
        still being copied/uploaded into the folder.
        """
        events: list[DeckEvent] = []
        for path in self._candidate_files():
            if not self._is_stable(path):
                continue
            now = datetime.now(timezone.utc).isoformat()
            events.append(
                DeckEvent(path=str(path), origin="folder_watch", received_at=now)
            )
            self._mark_seen(path.name)
            logger.info("Detected new deck: %s", path.name)
        return events

    @staticmethod
    def _is_stable(path: Path, check_interval: float = 0.5) -> bool:
        try:
            size_before = path.stat().st_size
            time.sleep(check_interval)
            size_after = path.stat().st_size
            return size_before == size_after
        except OSError:
            return False

    def watch(self, poll_interval_seconds: float = 5.0) -> Iterator[DeckEvent]:
        logger.info(
            "Watching '%s' for new decks (poll every %ss). Press Ctrl+C to stop.",
            self.watch_dir,
            poll_interval_seconds,
        )
        while True:
            for event in self.poll_once():
                yield event
            time.sleep(poll_interval_seconds)
