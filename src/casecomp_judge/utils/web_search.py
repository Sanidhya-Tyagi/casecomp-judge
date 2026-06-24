"""Web search, with a real API as the primary backend and a free
no-key scraper as automatic fallback.

Primary: Tavily (https://tavily.com) — a real search API purpose-built
for AI agents, with a free tier (1,000 credits/month, no card
required). Used automatically whenever a `TAVILY_API_KEY` environment
variable is set.

Fallback: DuckDuckGo's HTML endpoint (html.duckduckgo.com) — no key
required, but DuckDuckGo actively rate-limits and CAPTCHAs automated
HTML scraping, so this is unreliable at any real volume. It exists so
the pipeline still works with zero setup before you have a Tavily key,
not as a recommended steady-state backend.

Both backends are isolated behind one function (`search`) so nothing
else in the pipeline needs to know which one actually ran — callers
just get back a list of SearchResult or an empty list.

Failure philosophy: a broken or rate-limited search must NEVER crash
the pipeline. Every failure mode here returns an empty result list
and logs a warning; callers (the fact-checker) treat "no results" as
"unverifiable," not as an error to propagate.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger("casecomp_judge.web_search")

TAVILY_ENDPOINT = "https://api.tavily.com/search"
TAVILY_API_KEY_ENV_VAR = "TAVILY_API_KEY"

DDG_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_warned_no_tavily_key = False  # log the "using fallback" notice once, not per-call


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str


def search(
    query: str,
    max_results: int = 4,
    timeout_seconds: int = 10,
    max_retries: int = 1,
    retry_backoff_seconds: float = 2.0,
) -> list[SearchResult]:
    """Search the web and return parsed results.

    Uses Tavily if `TAVILY_API_KEY` is set in the environment (real API,
    reliable, free tier available). Falls back to scraping DuckDuckGo's
    HTML endpoint if no key is set, or if the Tavily call itself fails —
    so a billing issue or outage on Tavily's end degrades to "less
    reliable" rather than "broken."

    Returns an empty list (never raises) on total failure — network
    error, rate limit, unparseable response, etc. Callers should treat
    an empty list as "couldn't verify," not as a crash signal.
    """
    api_key = os.environ.get(TAVILY_API_KEY_ENV_VAR)
    if api_key:
        results = _search_tavily(
            query, api_key, max_results, timeout_seconds, max_retries, retry_backoff_seconds
        )
        if results:
            return results
        logger.info(
            "Tavily search returned no results for %r; falling back to "
            "DuckDuckGo scraping for this query.",
            query,
        )
    else:
        global _warned_no_tavily_key
        if not _warned_no_tavily_key:
            logger.warning(
                "No %s environment variable set — falling back to scraping "
                "DuckDuckGo's HTML endpoint, which is unreliable at any real "
                "volume (DuckDuckGo actively rate-limits/CAPTCHAs automated "
                "requests). Get a free Tavily API key (1,000 credits/month, "
                "no card) at https://tavily.com and set %s to use it instead.",
                TAVILY_API_KEY_ENV_VAR,
                TAVILY_API_KEY_ENV_VAR,
            )
            _warned_no_tavily_key = True

    return _search_duckduckgo(
        query, max_results, timeout_seconds, max_retries, retry_backoff_seconds
    )


def _search_tavily(
    query: str,
    api_key: str,
    max_results: int,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> list[SearchResult]:
    """Search via the Tavily API. Returns [] on any failure, never raises."""
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 2):
        try:
            response = requests.post(
                TAVILY_ENDPOINT,
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": max_results,
                    "include_answer": False,
                },
                timeout=timeout_seconds,
            )
            if response.status_code == 401:
                logger.error(
                    "Tavily API rejected the key (401 Unauthorized). Check "
                    "that %s is set correctly. Falling back to DuckDuckGo "
                    "for this and future calls until fixed.",
                    TAVILY_API_KEY_ENV_VAR,
                )
                return []
            if response.status_code == 432 or response.status_code == 429:
                logger.warning(
                    "Tavily API rate/credit limit reached (HTTP %d) for "
                    "query %r. Falling back to DuckDuckGo for this call.",
                    response.status_code,
                    query,
                )
                return []
            response.raise_for_status()
            data = response.json()
            return _parse_tavily_response(data, max_results)
        except requests.RequestException as exc:
            last_error = exc
            logger.warning(
                "Tavily search failed for query %r (attempt %d/%d): %s",
                query,
                attempt,
                max_retries + 1,
                exc,
            )
            if attempt <= max_retries:
                time.sleep(retry_backoff_seconds)

    logger.warning(
        "Tavily search permanently failed for query %r after %d attempts: "
        "%s. Falling back to DuckDuckGo for this call.",
        query,
        max_retries + 1,
        last_error,
    )
    return []


def _parse_tavily_response(data: dict, max_results: int) -> list[SearchResult]:
    results: list[SearchResult] = []
    try:
        for item in (data.get("results") or [])[:max_results]:
            title = str(item.get("title", "")).strip()
            content = str(item.get("content", "")).strip()
            url = str(item.get("url", "")).strip()
            if title:
                results.append(SearchResult(title=title, snippet=content, url=url))
    except Exception as exc:  # noqa: BLE001 — parsing must never crash the pipeline
        logger.warning("Failed to parse Tavily response: %s", exc)
        return []
    return results


def _search_duckduckgo(
    query: str,
    max_results: int,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> list[SearchResult]:
    """Fallback: scrape DuckDuckGo's HTML endpoint. Returns [] on any failure."""
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 2):
        try:
            response = requests.post(
                DDG_HTML_ENDPOINT,
                data={"q": query},
                headers={"User-Agent": USER_AGENT},
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            results = _parse_ddg_html(response.text, max_results)
            if results:
                return results
            # Empty parse isn't necessarily an error (genuinely no results),
            # but if the HTML structure changed, or DuckDuckGo served a
            # CAPTCHA page, we'd also get zero results from this — log at
            # debug level so it's discoverable without being noisy.
            logger.debug("DuckDuckGo search for %r returned 0 parsed results.", query)
            return []
        except requests.RequestException as exc:
            last_error = exc
            logger.warning(
                "DuckDuckGo web search failed for query %r (attempt %d/%d): %s",
                query,
                attempt,
                max_retries + 1,
                exc,
            )
            if attempt <= max_retries:
                time.sleep(retry_backoff_seconds)

    logger.warning(
        "DuckDuckGo web search permanently failed for query %r after %d "
        "attempts: %s. Treating as unverifiable rather than crashing.",
        query,
        max_retries + 1,
        last_error,
    )
    return []


def _parse_ddg_html(html: str, max_results: int) -> list[SearchResult]:
    """Minimal, dependency-free HTML parsing for DuckDuckGo's result markup.

    Deliberately uses simple string splitting rather than pulling in a
    full HTML parser dependency — DuckDuckGo's HTML endpoint markup is
    simple and stable enough for this, and a parsing failure here just
    means an empty result (handled safely by the caller), not a crash.
    """
    results: list[SearchResult] = []
    try:
        import re

        # Each result block: <a class="result__a" href="...">TITLE</a> ... <a class="result__snippet" ...>SNIPPET</a>
        link_pattern = re.compile(
            r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            re.DOTALL,
        )
        snippet_pattern = re.compile(
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL
        )

        links = link_pattern.findall(html)
        snippets = snippet_pattern.findall(html)

        for i, (url, raw_title) in enumerate(links[:max_results]):
            title = _strip_tags(raw_title)
            snippet = _strip_tags(snippets[i]) if i < len(snippets) else ""
            if title:
                results.append(SearchResult(title=title, snippet=snippet, url=url))
    except Exception as exc:  # noqa: BLE001 — parsing must never crash the pipeline
        logger.warning("Failed to parse DuckDuckGo HTML response: %s", exc)
        return []

    return results


def _strip_tags(text: str) -> str:
    import html as html_module
    import re

    text = re.sub(r"<[^>]+>", "", text)
    return html_module.unescape(text).strip()
