"""Thin client for the local Ollama HTTP API.

Wraps /api/generate and /api/chat with retry logic, timeout handling,
and an option to request structured JSON output (Ollama's `format:
"json"` mode), which the summarizer/judge agents rely on for reliable
parsing.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger("casecomp_judge.ollama")


class OllamaError(RuntimeError):
    """Raised when the Ollama API cannot be reached or errors out."""


def encode_image_file(path: str | Path) -> str:
    """Base64-encode an image file for the Ollama `images` field.

    Ollama expects raw base64 (no `data:image/png;base64,` prefix).
    """
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


@dataclass
class OllamaResponse:
    text: str
    raw: dict[str, Any]


class OllamaClient:
    """Minimal synchronous client for a local Ollama server."""

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        timeout_seconds: int = 300,
        temperature: float = 0.2,
        num_ctx: int = 8192,
        max_retries: int = 2,
        retry_backoff_seconds: int = 3,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def health_check(self) -> bool:
        """Returns True if Ollama is reachable and the model is available."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=10)
            resp.raise_for_status()
            tags = resp.json().get("models", [])
            names = {t.get("name", "").split(":")[0] for t in tags}
            model_base = self.model.split(":")[0]
            if model_base not in names:
                logger.warning(
                    "Model '%s' not found in `ollama list`. Available: %s. "
                    "Run `ollama pull %s` first.",
                    self.model,
                    ", ".join(sorted(names)) or "(none)",
                    self.model,
                )
            return True
        except requests.RequestException as exc:
            logger.error(
                "Could not reach Ollama at %s (%s). Is `ollama serve` running?",
                self.host,
                exc,
            )
            return False

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
        images: list[str] | None = None,
    ) -> OllamaResponse:
        """Call /api/generate once, with retries on transient failure.

        Args:
            prompt: the user/task prompt.
            system: optional system prompt.
            json_mode: if True, asks Ollama to constrain output to valid JSON.
            images: optional list of base64-encoded images (for vision models).
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"
        if images:
            payload["images"] = images

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 2):  # +1 initial, +retries
            try:
                resp = requests.post(
                    f"{self.host}/api/generate",
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data.get("response", "")
                return OllamaResponse(text=text, raw=data)
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning(
                    "Ollama call failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries + 1,
                    exc,
                )
                if attempt <= self.max_retries:
                    time.sleep(self.retry_backoff_seconds)

        raise OllamaError(
            f"Ollama generate() failed after {self.max_retries + 1} attempts "
            f"against {self.host} with model '{self.model}'. "
            f"Last error: {last_exc}"
        )

    def generate_with_image(
        self,
        prompt: str,
        image_path: str | Path,
        system: str | None = None,
    ) -> OllamaResponse:
        """Convenience wrapper: send exactly one image with a text prompt.

        One image per call is the safest default — not all Ollama vision
        models reliably support multiple images in a single request.
        """
        encoded = encode_image_file(image_path)
        return self.generate(prompt=prompt, system=system, images=[encoded])

    def generate_json(
        self,
        prompt: str,
        system: str | None = None,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        """Call generate() in JSON mode and parse the result.

        Falls back to lenient brace-extraction if the model wraps JSON
        in prose despite the format constraint (small/older models
        sometimes do this).
        """
        response = self.generate(
            prompt=prompt, system=system, json_mode=True, images=images
        )
        return self._parse_json_response(response.text)

    @staticmethod
    def _parse_json_response(text: str) -> dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Lenient fallback: extract the first {...} block.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                raise OllamaError(
                    f"Model did not return valid JSON, even after lenient "
                    f"extraction. Raw output (truncated): {text[:500]!r}"
                ) from exc

        raise OllamaError(
            f"Model did not return valid JSON and no JSON object could be "
            f"located. Raw output (truncated): {text[:500]!r}"
        )
