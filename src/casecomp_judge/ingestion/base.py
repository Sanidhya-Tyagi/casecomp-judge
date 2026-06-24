"""Abstract ingestion interface.

Today we only ship a folder-watching source (`FolderWatchSource`), but
the pipeline depends only on this `DeckSource` interface — so adding a
future API or web-upload source later means writing one new class,
not touching the orchestrator, extraction, or agents at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class DeckEvent:
    """Represents a single 'new deck available' event."""

    path: str          # local filesystem path to the PDF (always materialized
                        # to disk, even for future API/upload sources, so the
                        # rest of the pipeline only ever deals with file paths)
    origin: str        # e.g. "folder_watch", "api_upload" — for audit/logging
    received_at: str   # ISO-8601 timestamp


class DeckSource(ABC):
    """Base class for anything that can hand the pipeline new decks.

    Implementations should be resumable/idempotent where possible:
    calling `poll_once()` repeatedly should not re-emit decks already
    handed off (the folder watcher does this by tracking processed
    filenames; a future API source might track delivered message IDs).
    """

    @abstractmethod
    def poll_once(self) -> list[DeckEvent]:
        """Return any new deck events available right now (non-blocking)."""
        raise NotImplementedError

    @abstractmethod
    def watch(self, poll_interval_seconds: float) -> Iterator[DeckEvent]:
        """Block and yield deck events as they arrive, forever (or until
        interrupted). Used for long-running 'watch mode'.
        """
        raise NotImplementedError
